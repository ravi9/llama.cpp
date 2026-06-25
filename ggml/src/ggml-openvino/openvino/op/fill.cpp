#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstring>
#include <cstdint>
#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/shape_of.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_fill(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    const int32_t * op_params = context.get_output_op_params();
    FRONT_END_CHECK_IMPLEMENTED(op_params != nullptr, "FILL requires output op params");

    float value;
    std::memcpy(&value, op_params, sizeof(float));

    auto scalar = ov::op::v0::Constant::create(context.get_output_type(), ov::Shape{}, {value});

    ov::Output<ov::Node> target_shape;
    const auto output_shape = context.get_output_shape();
    if (output_shape.rank().is_static() && output_shape.is_static()) {
        const auto static_shape = output_shape.to_shape();
        std::vector<int64_t> shape_values(static_shape.begin(), static_shape.end());
        target_shape = ov::op::v0::Constant::create(ov::element::i64, {shape_values.size()}, shape_values);
    } else {
        auto input = process_view_input_new(context, 0);
        target_shape = std::make_shared<ov::op::v3::ShapeOf>(input, ov::element::i64);
    }

    auto res = std::make_shared<ov::op::v3::Broadcast>(scalar, target_shape);
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov