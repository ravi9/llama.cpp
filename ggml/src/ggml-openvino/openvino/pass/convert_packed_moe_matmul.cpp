// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_packed_moe_matmul.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <vector>

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/bitwise_and.hpp>
#include <openvino/op/bitwise_right_shift.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <openvino/runtime/tensor.hpp>

#include "../op/gather_matmul_compressed.hpp"

namespace ov::frontend::ggml::pass {
namespace {

namespace v0 = ov::op::v0;
namespace v1 = ov::op::v1;
namespace v8 = ov::op::v8;

constexpr const char * unpacked_planes_key = "__ggml_openvino_mxfp4_unpacked_planes";

struct UnpackedMxfp4Planes {
    std::weak_ptr<v0::Constant> source;
    std::weak_ptr<v0::Constant> weights;
    std::weak_ptr<v0::Constant> scales;
};

std::shared_ptr<v8::Gather> get_packed_mxfp4_gather(const ov::Output<ov::Node>& indices) {
    auto convert = ov::as_type_ptr<v0::Convert>(indices.get_node_shared_ptr());
    auto concat = convert ? ov::as_type_ptr<v0::Concat>(convert->input_value(0).get_node_shared_ptr()) : nullptr;
    if (!concat || concat->get_input_size() != 2 || concat->get_axis() != 4) {
        return nullptr;
    }
    auto low = ov::as_type_ptr<ov::op::v13::BitwiseAnd>(concat->input_value(0).get_node_shared_ptr());
    auto high = ov::as_type_ptr<ov::op::v15::BitwiseRightShift>(concat->input_value(1).get_node_shared_ptr());
    if (!low || !high) {
        return nullptr;
    }
    auto low_slice = ov::as_type_ptr<v8::Slice>(low->input_value(0).get_node_shared_ptr());
    auto high_slice = ov::as_type_ptr<v8::Slice>(high->input_value(0).get_node_shared_ptr());
    if (!low_slice || !high_slice || low_slice->input_value(0) != high_slice->input_value(0)) {
        return nullptr;
    }
    return ov::as_type_ptr<v8::Gather>(low_slice->input_value(0).get_node_shared_ptr());
}

std::shared_ptr<v8::Gather> get_packed_mxfp4_scale_gather(const ov::Output<ov::Node>& indices) {
    auto convert = ov::as_type_ptr<v0::Convert>(indices.get_node_shared_ptr());
    auto slice = convert ? ov::as_type_ptr<v8::Slice>(convert->input_value(0).get_node_shared_ptr()) : nullptr;
    return slice ? ov::as_type_ptr<v8::Gather>(slice->input_value(0).get_node_shared_ptr()) : nullptr;
}

bool is_mxfp4_lut_gather(const ov::Output<ov::Node>& output, size_t lut_size) {
    auto gather = ov::as_type_ptr<v8::Gather>(output.get_node_shared_ptr());
    auto constant = gather ? ov::as_type_ptr<v0::Constant>(gather->input_value(0).get_node_shared_ptr()) : nullptr;
    return constant && constant->get_shape() == ov::Shape{lut_size};
}

bool unpack_packed_mxfp4_weights(const std::shared_ptr<v8::Gather>& weights_gather,
                                 std::shared_ptr<v0::Constant>& weights,
                                 std::shared_ptr<v0::Constant>& scales) {
    auto weights_source = weights_gather->input_value(0).get_node_shared_ptr();
    if (auto reshape = ov::as_type_ptr<v1::Reshape>(weights_source)) {
        weights_source = reshape->input_value(0).get_node_shared_ptr();
    }
    auto packed_constant = ov::as_type_ptr<v0::Constant>(weights_source);
    if (!packed_constant || packed_constant->get_element_type() != ov::element::u8) {
        return false;
    }

    auto packed_shape = packed_constant->get_shape();
    if (packed_shape.size() == 5 && packed_shape[0] == 1) {
        packed_shape.erase(packed_shape.begin());
    }
    if (packed_shape.size() != 4 || packed_shape[3] != 17) {
        return false;
    }

    const size_t experts = packed_shape[0];
    const size_t output_features = packed_shape[1];
    const size_t groups = packed_shape[2];
    constexpr size_t group_size = 32;
    const size_t input_features = groups * group_size;
    const auto* packed_data = packed_constant->get_data_ptr<uint8_t>();

    ov::Tensor weights_tensor(ov::element::f4e2m1, ov::Shape{experts, output_features, input_features});
    ov::Tensor scales_tensor(ov::element::f32, ov::Shape{experts, output_features, groups});
    auto* weights_data = static_cast<uint8_t*>(weights_tensor.data());
    std::memset(weights_data, 0, weights_tensor.get_byte_size());
    auto* scales_data = scales_tensor.data<float>();

    for (size_t expert = 0; expert < experts; ++expert) {
        for (size_t output = 0; output < output_features; ++output) {
            for (size_t group = 0; group < groups; ++group) {
                const size_t src_offset = ((expert * output_features + output) * groups + group) * 17;
                const uint32_t scale_bits = static_cast<uint32_t>(packed_data[src_offset]) << 23;
                float scale = 0.0f;
                if (packed_data[src_offset] == 0) {
                    scale = std::ldexp(1.0f, -127);
                } else if (packed_data[src_offset] == 255) {
                    scale = std::numeric_limits<float>::quiet_NaN();
                } else {
                    std::memcpy(&scale, &scale_bits, sizeof(scale));
                }
                scales_data[(expert * output_features + output) * groups + group] = scale;

                for (size_t k = 0; k < group_size; ++k) {
                    const auto packed_byte = packed_data[src_offset + 1 + (k % 16)];
                    const auto nibble = static_cast<uint8_t>(k < 16 ? packed_byte & 0x0f : packed_byte >> 4);
                    const size_t dst_index =
                        (expert * output_features + output) * input_features + group * group_size + k;
                    weights_data[dst_index / 2] |= static_cast<uint8_t>(nibble << ((dst_index % 2) * 4));
                }
            }
        }
    }

    weights = std::make_shared<v0::Constant>(weights_tensor);
    scales = std::make_shared<v0::Constant>(scales_tensor);
    ov::copy_runtime_info(packed_constant, {weights, scales});
    return true;
}


}  // namespace

ConvertPackedMoeMatMulToGatherMatmul::ConvertPackedMoeMatMulToGatherMatmul() {
    namespace pattern = ov::pass::pattern;

    auto matmul_pattern =
        pattern::wrap_type<v0::MatMul>({pattern::any_input(), pattern::any_input()},
                                       pattern::attrs_match({{"transpose_a", false}, {"transpose_b", true}}));

    const auto callback = [this](pattern::Matcher& m) {
        const auto matmul = ov::as_type_ptr<v0::MatMul>(m.get_match_root());
        if (!matmul || transformation_callback(matmul)) {
            return false;
        }

        auto expanded_activations = ov::as_type_ptr<v0::Unsqueeze>(matmul->input_value(0).get_node_shared_ptr());
        auto expanded_weights = ov::as_type_ptr<v1::Reshape>(matmul->input_value(1).get_node_shared_ptr());
        auto weight_scale_product =
            expanded_weights ? ov::as_type_ptr<v1::Multiply>(expanded_weights->input_value(0).get_node_shared_ptr())
                             : nullptr;
        if (!expanded_activations || !expanded_weights || !weight_scale_product ||
            matmul->get_input_partial_shape(0).rank() != 4 || matmul->get_input_partial_shape(1).rank() != 4 ||
            expanded_activations->input_value(0).get_partial_shape().rank() != 3 ||
            !is_mxfp4_lut_gather(weight_scale_product->input_value(0), 16) ||
            !is_mxfp4_lut_gather(weight_scale_product->input_value(1), 256)) {
            return false;
        }

        auto quant_gather = ov::as_type_ptr<v8::Gather>(weight_scale_product->input_value(0).get_node_shared_ptr());
        auto scale_gather = ov::as_type_ptr<v8::Gather>(weight_scale_product->input_value(1).get_node_shared_ptr());
        auto packed_weights_gather = get_packed_mxfp4_gather(quant_gather->input_value(1));
        auto packed_scale_gather = get_packed_mxfp4_scale_gather(scale_gather->input_value(1));
        if (!packed_weights_gather || packed_weights_gather != packed_scale_gather ||
            packed_weights_gather->get_batch_dims() != 0 ||
            packed_weights_gather->input_value(1).get_element_type() != ov::element::i32 ||
            packed_weights_gather->input_value(1).get_partial_shape().rank() != 2 ||
            packed_weights_gather->input_value(0).get_partial_shape().rank() != 4 ||
            packed_weights_gather->get_output_partial_shape(0).rank() != 5) {
            return false;
        }

        std::shared_ptr<v0::Constant> weights;
        std::shared_ptr<v0::Constant> scales;
        if (!unpack_packed_mxfp4_weights(packed_weights_gather, weights, scales)) {
            return false;
        }

        const auto activations = expanded_activations->input_value(0);
        auto activation_transpose_order =
            v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 0, 2});
        auto activation_transpose = std::make_shared<v1::Transpose>(activations, activation_transpose_order);
        auto empty_bias = std::make_shared<v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto empty_zero_points = std::make_shared<v0::Constant>(ov::element::dynamic, ov::Shape{0});
        auto gathered_matmul =
            std::make_shared<ov::op::internal::GatherMatmulCompressed>(activation_transpose,
                                                                       weights,
                                                                       packed_weights_gather->input_value(1),
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
        return ov::replace_output_update_name(matmul->output(0), restored_rank->output(0));
    };

    auto matcher = std::make_shared<pattern::Matcher>(
        matmul_pattern, "ov::frontend::ggml::pass::ConvertPackedMoeMatMulToGatherMatmul");
    this->register_matcher(matcher, callback);
}

}  // namespace ov::frontend::ggml::pass
