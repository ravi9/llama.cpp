
#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <climits>
#include <cstdint>
#include <memory>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

int infer_dynamic_dim_cont(const NodeContext & context) {
    int dynamic_dim_idx = context.get_input_dynamic_dim(0);
    if (dynamic_dim_idx == -1) {
        return -1;
    }
    if (context.input_has_same_shape_as_output(0)) {
        return dynamic_dim_idx;
    }

    auto input_shape = context.get_input_ggml_shape(0);
    auto output_shape = context.get_output_ggml_shape();
    auto output_stride = context.get_output_ggml_stride();
    std::vector<size_t> src_logical_nb(input_shape.size());
    src_logical_nb[0] = context.get_input_type_size(0);
    src_logical_nb[1] = src_logical_nb[0] * (input_shape[0] / context.get_input_block_size(0));
    for (size_t i = 2; i < input_shape.size(); i++) {
        src_logical_nb[i] = src_logical_nb[i - 1] * input_shape[i - 1];
    }

    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / context.get_input_type_size(0) *
                              context.get_output_type_size();
    int matched_dim = -1;
    int matched_dim_count = 0;
    for (size_t i = 0; i < output_stride.size(); i++) {
        if (output_stride[i] == dynamic_dim_stride && output_shape[i] == input_shape[dynamic_dim_idx]) {
            matched_dim = static_cast<int>(i);
            matched_dim_count++;
        }
    }
    return matched_dim_count == 1 ? matched_dim : -1;
}

OutputVector translate_cont(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto src_shape = context.get_input_shape(0).to_shape();
    auto dst_shape = context.get_output_shape().to_shape();

    if (context.get_op_dynamic_dim() != -1) {
        dst_shape[3 - context.get_op_dynamic_dim()] = -1;
    }

    auto input = process_view_input_new(context, 0);

    ov::Output<Node> res;
    res = std::make_shared<ov::op::v1::Reshape>(
        input, ov::op::v0::Constant::create(ov::element::i64, {dst_shape.size()}, dst_shape), false);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
