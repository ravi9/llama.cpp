#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

int infer_dynamic_dim_get_rows(const NodeContext & context) {
    int dynamic_dim_idx = context.get_input_dynamic_dim(1);
    if (dynamic_dim_idx == -1) {
        return -1;
    }
    if (dynamic_dim_idx == 0) {
        return 1;
    }
    auto indices_stride = context.get_input_ggml_stride(1);
    auto data_stride = context.get_input_ggml_stride(0);
    auto dynamic_dim_stride = indices_stride[dynamic_dim_idx] / context.get_input_type_size(1) *
                              context.get_input_type_size(0);
    for (size_t i = 0; i < data_stride.size(); i++) {
        if (dynamic_dim_stride == data_stride[i]) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

OutputVector translate_get_rows(const NodeContext & context) {
    num_inputs_check(context, 2, 2);

    Output<Node> res;
    auto data = process_view_input_new(context, 0);
    auto indices = process_view_input_new(context, 1);

    // data[1,b,x,y] ind[1,1,b,x'] test-backend-ops case
    // data[x,y] ind[1,1,1,x'] normal case
    indices =
        std::make_shared<ov::op::v0::Squeeze>(indices, ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
    if (data.get_partial_shape().rank() == 4) {
        if (!(data.get_partial_shape()[1].is_dynamic()) && data.get_partial_shape()[1].get_length() == 1) {
            // Work-around for a bug in ov cpu plugin for test-backend-ops
            data = std::make_shared<ov::op::v0::Squeeze>(data,
                                                         ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 1}));
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
        } else {
            auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
            data =
                std::make_shared<ov::op::v0::Squeeze>(data, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
            res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
        }
    } else if (context.is_stateful() && data.get_partial_shape().rank() == 3) {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {1});
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis, 1);
    } else {
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
        res = std::make_shared<ov::op::v8::Gather>(data, indices, axis);
    }

    if (res.get_element_type() != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(res, context.get_output_type());
    }
    if (!(context.is_stateful())) {
        res = std::make_shared<ov::op::v0::Unsqueeze>(res, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
