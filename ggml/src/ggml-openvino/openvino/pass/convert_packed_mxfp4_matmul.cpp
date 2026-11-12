#include "convert_packed_mxfp4_matmul.h"

#include <iostream>

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include "../op/gather_matmul_compressed.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

bool is_expert_gather(const std::shared_ptr<v8::Gather> & gather) {
    if (!gather || gather->get_batch_dims() != 0) {
        return false;
    }
    auto axis = ov::as_type_ptr<v0::Constant>(gather->input_value(2).get_node_shared_ptr());
    return axis && ov::shape_size(axis->get_shape()) == 1 && axis->cast_vector<int64_t>()[0] == 0;
}

}  // namespace

ConvertPackedMxfp4MatMul::ConvertPackedMxfp4MatMul() {
    using namespace ov::pass::pattern;

    auto matmul_pattern =
        wrap_type<v0::MatMul>({any_input(), any_input()}, attrs_match({{"transpose_a", false}, {"transpose_b", true}}));

    const auto callback = [this](Matcher & matcher) {
        const auto matmul = ov::as_type_ptr<v0::MatMul>(matcher.get_match_root());
        if (!matmul || transformation_callback(matmul)) {
            std::cout << "false 1 " << std::endl;
            return false;
        }

        auto expanded_activations = ov::as_type_ptr<v0::Unsqueeze>(matmul->input_value(0).get_node_shared_ptr());
        auto expanded_weights = ov::as_type_ptr<v1::Reshape>(matmul->input_value(1).get_node_shared_ptr());
        auto weight_scale_product =
            expanded_weights ? ov::as_type_ptr<v1::Multiply>(expanded_weights->input_value(0).get_node_shared_ptr())
                             : nullptr;
        if (!expanded_activations || !expanded_weights || !weight_scale_product ||
            matmul->get_input_partial_shape(0).rank() != 4 || matmul->get_input_partial_shape(1).rank() != 4 ||
            expanded_activations->input_value(0).get_partial_shape().rank() != 3) {
            std::cout << "false 2 " << std::endl;
            return false;
        }

        auto convert = ov::as_type_ptr<v0::Convert>(weight_scale_product->input_value(0).get_node_shared_ptr());
        auto quant_gather = convert ? ov::as_type_ptr<v8::Gather>(convert->input_value(0).get_node_shared_ptr()) :
                                     nullptr;
        auto scale_gather = ov::as_type_ptr<v8::Gather>(weight_scale_product->input_value(1).get_node_shared_ptr());
        if (!is_expert_gather(quant_gather) || !is_expert_gather(scale_gather) ||
            quant_gather->input_value(1) != scale_gather->input_value(1) ||
            quant_gather->input_value(1).get_element_type() != ov::element::i32 ||
            quant_gather->input_value(1).get_partial_shape().rank() != 2) {
            std::cout << "false 3 " << std::endl;
            return false;
        }

        auto weight_plane = ov::as_type_ptr<v0::Constant>(quant_gather->input_value(0).get_node_shared_ptr());
        auto scale_plane = ov::as_type_ptr<v0::Constant>(scale_gather->input_value(0).get_node_shared_ptr());
        if (!weight_plane || !scale_plane || weight_plane->get_element_type() != ov::element::f4e2m1 ||
            scale_plane->get_element_type() != ov::element::f32) {
            std::cout << "false 4 " << std::endl;
            return false;
        }
        const auto & weight_shape = weight_plane->get_shape();
        const auto & scale_shape = scale_plane->get_shape();
        if (weight_shape.size() != 4 || weight_shape[3] != 32 ||
            scale_shape != ov::Shape{weight_shape[0], weight_shape[1], weight_shape[2], 1}) {
            std::cout << "false 5 " << std::endl;
            return false;
        }

        // Constant's reshape constructor shares the source buffer, including its lifetime.
        auto weights = std::make_shared<v0::Constant>(
            *weight_plane, ov::Shape{weight_shape[0], weight_shape[1], weight_shape[2] * 32});
        auto scales = std::make_shared<v0::Constant>(
            *scale_plane, ov::Shape{weight_shape[0], weight_shape[1], weight_shape[2]});
        ov::copy_runtime_info(weight_plane, weights);
        ov::copy_runtime_info(scale_plane, scales);

        const auto activations = expanded_activations->input_value(0);
        auto activation_transpose_order =
            v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 0, 2});
        auto activation_transpose = std::make_shared<v1::Transpose>(activations, activation_transpose_order);
        auto empty_bias = std::make_shared<v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto empty_zero_points = std::make_shared<v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto gathered_matmul =
            std::make_shared<ov::op::internal::GatherMatmulCompressed>(activation_transpose,
                                                                       weights,
                                                                       quant_gather->input_value(1),
                                                                       empty_bias,
                                                                       scales,
                                                                       empty_zero_points);
        auto output_transpose_order =
            v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 0, 2});
        auto output_transpose = std::make_shared<v1::Transpose>(gathered_matmul, output_transpose_order);
        auto restored_rank_axis = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
        auto restored_rank = std::make_shared<v0::Unsqueeze>(output_transpose, restored_rank_axis);
        ov::copy_runtime_info(matmul,
                              {activation_transpose_order,
                               activation_transpose,
                               empty_bias,
                               empty_zero_points,
                               gathered_matmul,
                               output_transpose_order,
                               output_transpose,
                               restored_rank_axis,
                               restored_rank});
        const bool replaced = ov::replace_output_update_name(matmul->output(0), restored_rank->output(0));
        std::cout << " Replaced: " << replaced << std::endl;
        return replaced;
    };

    register_matcher(std::make_shared<Matcher>(matmul_pattern, "ov::frontend::ggml::pass::ConvertPackedMxfp4MatMul"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
