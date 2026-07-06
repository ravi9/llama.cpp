#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

int infer_dynamic_dim_reshape(const NodeContext & context) {
    int dynamic_dim_idx = context.get_input_dynamic_dim(0);
    if (dynamic_dim_idx == -1) {
        return -1;
    }
    auto input_shape = context.get_input_ggml_shape(0);
    auto input_stride = context.get_input_ggml_stride(0);
    auto output_shape = context.get_output_ggml_shape();
    auto output_stride = context.get_output_ggml_stride();
    auto dynamic_dim_stride = input_stride[dynamic_dim_idx];
    for (size_t i = 0; i < output_stride.size(); i++) {
        if (output_stride[i] == dynamic_dim_stride && output_shape[i] == input_shape[dynamic_dim_idx]) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

OutputVector translate_reshape(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    if (context.get_input(0).get_partial_shape().is_static() &&
        context.get_input_shape(0) == context.get_output_shape()) {
        return {context.get_input(0)};
    }

    int op_case = context.get_op_case();
    FRONT_END_CHECK_IMPLEMENTED(
        op_case == 1 || op_case == 2 || op_case == 3 || op_case == 4 || op_case == 5 || op_case == 6,
        "Unsupported RESHAPE case");

    auto output_shape = context.get_output_shape().to_shape();
    std::shared_ptr<ov::Node> new_shape_node;
    if (op_case == 1) {
        if (context.is_stateful()) {
            new_shape_node = ov::op::v0::Constant::create(
                ov::element::i64, {3}, std::vector<int64_t>{-1, (int64_t) output_shape[2], (int64_t) output_shape[3]});
        } else {
            new_shape_node = ov::op::v0::Constant::create(
                ov::element::i64, {4},
                std::vector<int64_t>{(int64_t) output_shape[0], -1, (int64_t) output_shape[2],
                                     (int64_t) output_shape[3]});
        }
    } else if (op_case == 2) {
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64, {4},
            std::vector<int64_t>{(int64_t) output_shape[0], (int64_t) output_shape[1], -1, (int64_t) output_shape[3]});

    } else if (op_case == 3) {
        //  -  14: [     1,  1024,     1,     1] RESHAPE              Vcur-0 (reshaped) (reshaped)
        //         [   512,     2,     1,     1]            0: RESHAPE     Vcur-0 (reshaped)
        //  -  15: [     1, 524288,     1,     1] RESHAPE              cache_v_l0 (reshaped)
        //         [   512,  1024,     1,     1]            0: NONE        cache_v_l0
        //  -  16: [     1, 524288,     1,     1] SET_ROWS             cache_v_l0 (reshaped) (view)
        //         [     1,  1024,     1,     1]            0: RESHAPE     Vcur-0 (reshaped) (reshaped)
        //         [  1024,     1,     1,     1]            1: NONE        leaf_11
        //         [     1, 524288,     1,     1]            2: RESHAPE     cache_v_l0 (reshaped)
        new_shape_node = ov::op::v0::Constant::create(
            ov::element::i64, {4}, std::vector<int64_t>{(int64_t) output_shape[0], (int64_t) output_shape[1], -1, 1});

    } else if (op_case == 4) {
        return {context.get_input(0).get_node_shared_ptr()->input_value(0)};

    } else if (op_case == 5) {
        if (context.is_stateful()) {
            std::vector<int64_t> shape_vec = {1, -1, (int64_t) context.get_output_shape().to_shape()[3]};
            new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {3}, shape_vec);
        } else {
            std::vector<int64_t> shape_vec = {1, 1, -1, (int64_t) context.get_output_shape().to_shape()[3]};
            new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {4}, shape_vec);
        }

        // // Alternative
        // auto token_len = context.get_input("token_len");
        // auto emb_size =
        //     ov::op::v0::Constant::create(ov::element::i64, {1}, {(int64_t) context.get_output_shape().to_shape()[3]});
        // auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        // new_shape_node = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{one, one, token_len, emb_size}, 0);

    } else if (op_case == 6) {
        new_shape_node = ov::op::v0::Constant::create(ov::element::i64, {4}, context.get_output_shape().to_shape());
    }
    auto res = std::make_shared<ov::op::v1::Reshape>(context.get_input(0), new_shape_node, false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
