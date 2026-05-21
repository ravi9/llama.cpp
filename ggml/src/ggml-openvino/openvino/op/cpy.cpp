#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <climits>
#include <memory>
#include <vector>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cpy(const NodeContext & context) {
    auto op_case = context.get_op_case();
    auto input_shape = context.get_input_shape(0);
    auto output_shape = context.get_input_shape(1);

    ov::Output<Node> input;
    if (op_case == 1) {
        int ssm_state_size = context.get_ssm_state_size();
        auto gdn_output_state = std::make_shared<ov::op::v8::Slice>(
            context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {-ssm_state_size}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {2}));
        input = gdn_output_state;
    } else if (op_case == 2) {
        auto cache_r_size = input_shape[3].get_length();
        auto conv_state_last = std::make_shared<ov::op::v8::Slice>(
            context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {-cache_r_size}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
        input = conv_state_last;
    } else {
        input = process_view_input_new(context, 0);
    }

    if (input_shape != output_shape) {
        auto new_shape = ov::op::v0::Constant::create(
            ov::element::i64, {static_cast<size_t>(output_shape.rank().get_length())}, output_shape.to_shape());
        input = std::make_shared<ov::op::v1::Reshape>(input, new_shape, false);
    }

    ov::Output<Node> res;
    if (context.get_input_type(0) != context.get_output_type()) {
        res = std::make_shared<ov::op::v0::Convert>(input, context.get_output_type());
    } else {
        res = input;
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
