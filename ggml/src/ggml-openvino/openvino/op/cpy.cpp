#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <climits>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_cpy(const NodeContext & context) {
    auto op_case = context.get_op_case();
    auto input_shape = context.get_input_shape(0);
    auto output_shape = context.get_input_shape(1);

    // Recurrent state cache writeback with a dynamic active-slot block (inp->s_copy reorder).
    // The active sequences occupy a contiguous slot block [idx, idx+len) of the state cache; write
    // the new rows into that block while preserving the rest, so the result is the full updated
    // cache. op_case 1: gated-delta-net state, op_case 2: conv state, op_case 3: defrag remainder.
    const bool slice_assign = context.has_input("s_copy_active_slot_len") && !context.is_stateful() &&
                              (op_case == 1 || op_case == 2 || op_case == 3);
    if (slice_assign) {
        const int64_t slot_axis = 2;
        auto slot_idx = context.get_input("s_copy_active_slot_idx");
        auto slot_len = context.get_input("s_copy_active_slot_len");
        auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto int_max = ov::op::v0::Constant::create(ov::element::i64, {1}, {INT_MAX});
        auto axis = ov::op::v0::Constant::create(ov::element::i64, {1}, {slot_axis});

        ov::Output<ov::Node> src;
        ov::Output<ov::Node> begin;
        if (op_case == 1) {
            // GDN packs [attn | new_state]; the state is the last ssm_state_size * n_seqs rows.
            int ssm_state_size = context.get_ssm_state_size();
            auto state_rows = std::make_shared<ov::op::v1::Multiply>(
                ov::op::v0::Constant::create(ov::element::i64, {1}, {ssm_state_size}), slot_len);
            auto state_begin = std::make_shared<ov::op::v0::Negative>(state_rows);
            auto state_part =
                std::make_shared<ov::op::v8::Slice>(context.get_input(0), state_begin, int_max, one, axis);
            auto feature = (int64_t) output_shape[3].get_length();
            src = std::make_shared<ov::op::v1::Reshape>(
                state_part,
                ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, feature}), false);
            begin = slot_idx;
        } else if (op_case == 2) {
            auto cache_r_size = (int64_t) input_shape[3].get_length();
            auto conv_state_last = std::make_shared<ov::op::v8::Slice>(
                context.get_input(0), ov::op::v0::Constant::create(ov::element::i64, {1}, {-cache_r_size}), int_max,
                one, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}));
            auto feature = (int64_t) output_shape[3].get_length();
            src = std::make_shared<ov::op::v1::Reshape>(
                conv_state_last,
                ov::op::v0::Constant::create(ov::element::i64, {4}, std::vector<int64_t>{1, 1, -1, feature}), false);
            begin = slot_idx;
        } else {
            // op_case 3: gathered remainder rows already have the cache slot layout [1, 1, extra, feature]
            src = context.get_input(0);
            begin = std::make_shared<ov::op::v1::Add>(slot_idx, slot_len);
        }

        if (src.get_element_type() != context.get_output_type()) {
            src = std::make_shared<ov::op::v0::Convert>(src, context.get_output_type());
        }

        auto base = context.get_input(1);
        auto src_len =
            std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(src, ov::element::i64), axis,
                                                 ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
        auto end = std::make_shared<ov::op::v1::Add>(begin, src_len);
        auto head_part = std::make_shared<ov::op::v8::Slice>(base, zero, begin, one, axis);
        auto tail_part = std::make_shared<ov::op::v8::Slice>(base, end, int_max, one, axis);
        auto res = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{head_part, src, tail_part}, slot_axis);
        return rename_outputs_with_suffix({res}, context.get_name());
    }

    auto input = process_view_input_new(context, 0);

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

    if (res.get_node_shared_ptr() == context.get_input(0).get_node_shared_ptr()) {
        return {res};
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
