#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <cstdint>
#include <memory>
#include <openvino/op/ops.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

namespace {

ov::Output<ov::Node> const_i32(const std::vector<int32_t> & values) {
    return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{values.size()}, values);
}

ov::Output<ov::Node> const_i32_scalar(int32_t value) {
    return ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {value});
}

// Expert weights as [n_expert, rows, cols] in the type of the activations
ov::Output<ov::Node> expert_weights_3d(ov::Output<ov::Node> weights,
                                       int64_t n_expert,
                                       const ov::element::Type & type) {
    const auto rank = weights.get_partial_shape().rank();
    FRONT_END_OP_CONVERSION_CHECK(rank.is_static(), "Expected static rank for MoE expert weights");
    if (rank.get_length() == 4) {
        const auto shape = weights.get_partial_shape();
        FRONT_END_OP_CONVERSION_CHECK(shape.is_static(), "Expected static shape for MoE expert weights");
        weights = std::make_shared<ov::op::v1::Reshape>(
            weights,
            const_i32({static_cast<int32_t>(n_expert), static_cast<int32_t>(shape[2].get_length()),
                       static_cast<int32_t>(shape[3].get_length())}),
            false);
    }
    if (weights.get_element_type() != type) {
        weights = std::make_shared<ov::op::v0::Convert>(weights, type);
    }
    return weights;
}

}  // namespace

// The MoE block in the batched form that the NPUW MoE support (intel_npu/src/plugin/npuw/moe*) looks for; the same graph
// optimum-intel exports for Qwen3-30B-A3B. Every expert is computed for every token and the result is weighted by the
// router scores scattered to all experts (0 for a not selected one); NPUW runs only the routed experts.
//
//   router:  probs -> TopK(K) -> [ReduceSum -> Maximum -> Divide] -> ScatterElementsUpdate(zeros, ids, scores)
//            -> Transpose -> Reshape [n_expert, 1, n_tokens] -> Unsqueeze  = scores [n_expert, 1, n_tokens, 1]
//   experts: x -> Tile -> Reshape [n_expert, n_tokens, n_embd] -> MatMul(gate) -> Swish, MatMul(up) -> Multiply
//            -> MatMul(down) -> Reshape [n_expert, 1, n_tokens, n_embd] -> Multiply(scores) -> ReduceSum(axis 0)
//
// Inputs of the node (see GgmlOvDecoder::set_input_output): the two ADD operands, then router probs, x, the gate, up and
// down expert weights, the down MUL_MAT_ID (for the number of used experts) and, for op case 3, the CLAMP of the weights.
OutputVector translate_moe_tiled(const NodeContext & context) {
    const bool normalize = context.get_op_case() == 3;
    num_inputs_check(context, normalize ? 9 : 8, normalize ? 9 : 8);

    auto probs = context.get_input(2);
    auto x = context.get_input(3);
    const auto act_type = x.get_element_type();

    // ggml [n_embd, n_ff, n_expert] and [n_ff, n_embd, n_expert] -> OV [1, n_expert, rows, cols]
    const auto gate_shape = context.get_input_shape(4);
    FRONT_END_OP_CONVERSION_CHECK(gate_shape.rank().is_static() && gate_shape.rank().get_length() == 4 &&
                                      gate_shape.is_static(),
                                  "MoE expert weights need a static 4D shape");
    const int64_t n_expert = gate_shape[1].get_length();
    const int64_t n_embd = gate_shape[3].get_length();
    // down MUL_MAT_ID: ggml [n_embd, n_expert_used, n_tokens] -> OV [1, n_tokens, n_expert_used, n_embd]
    const auto used_shape = context.get_input_shape(7);
    FRONT_END_OP_CONVERSION_CHECK(used_shape[2].is_static(), "MoE needs a static number of used experts");
    const int32_t n_used = static_cast<int32_t>(used_shape[2].get_length());

    auto gate_w = expert_weights_3d(context.get_input(4), n_expert, act_type);
    auto up_w = expert_weights_3d(context.get_input(5), n_expert, act_type);
    auto down_w = expert_weights_3d(context.get_input(6), n_expert, act_type);

    ov::Output<ov::Node> x_2d =
        std::make_shared<ov::op::v1::Reshape>(x, const_i32({-1, static_cast<int32_t>(n_embd)}), false);
    ov::Output<ov::Node> probs_2d =
        std::make_shared<ov::op::v1::Reshape>(probs, const_i32({-1, static_cast<int32_t>(n_expert)}), false);

    // In a static graph the last layer can run on fewer tokens than the graph: its rows are selected by the output ids
    // (1 row in the prefill graph). Pad the block to the tokens of the graph, so that it is the same as the block of the
    // other layers: NPUW only lowers MoE blocks that it folds into one repeated function. The padded rows are zero and
    // are dropped after the block.
    int64_t n_rows = -1;
    int64_t n_pad = 0;
    if (context.is_static()) {
        const auto rows = x_2d.get_partial_shape()[0];
        FRONT_END_OP_CONVERSION_CHECK(rows.is_static(), "MoE needs a static number of tokens in a static graph");
        n_rows = rows.get_length();
        n_pad = context.get_static_n_tokens() - n_rows;
    }
    if (n_pad > 0) {
        auto pad = [&](const ov::Output<ov::Node> & input) {
            return std::make_shared<ov::op::v12::Pad>(
                input, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 0}),
                ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {n_pad, int64_t{0}}),
                ov::op::v0::Constant::create(input.get_element_type(), ov::Shape{}, {0.0f}), ov::op::PadMode::CONSTANT);
        };
        x_2d = pad(x_2d);
        probs_2d = pad(probs_2d);
    }

    // Router
    auto topk = std::make_shared<ov::op::v11::TopK>(probs_2d, const_i32_scalar(n_used), 1,
                                                    ov::op::v11::TopK::Mode::MAX, ov::op::v11::TopK::SortType::SORT_VALUES,
                                                    ov::element::i32);
    ov::Output<ov::Node> scores = topk->output(0);
    if (normalize) {
        float clamp_min;
        memcpy(&clamp_min, context.get_input_op_params(8), sizeof(clamp_min));
        auto sum = std::make_shared<ov::op::v1::ReduceSum>(scores, const_i32({1}), true);
        auto denominator = std::make_shared<ov::op::v1::Maximum>(
            sum, ov::op::v0::Constant::create(scores.get_element_type(), ov::Shape{}, {clamp_min}));
        scores = std::make_shared<ov::op::v1::Divide>(scores, denominator);
    }
    auto zeros = std::make_shared<ov::op::v3::Broadcast>(
        ov::op::v0::Constant::create(scores.get_element_type(), ov::Shape{}, {0.0f}),
        std::make_shared<ov::op::v3::ShapeOf>(probs_2d, ov::element::i32), ov::op::BroadcastType::NUMPY);
    auto scatter = std::make_shared<ov::op::v12::ScatterElementsUpdate>(zeros, topk->output(1), scores,
                                                                        const_i32_scalar(1));
    auto transposed = std::make_shared<ov::op::v1::Transpose>(scatter, const_i32({1, 0}));
    auto reshaped = std::make_shared<ov::op::v1::Reshape>(transposed, const_i32({static_cast<int32_t>(n_expert), 1, -1}),
                                                          false);
    auto router_scores = std::make_shared<ov::op::v0::Unsqueeze>(reshaped, const_i32_scalar(3));

    // Experts
    auto tiled = std::make_shared<ov::op::v0::Tile>(x_2d, const_i32({static_cast<int32_t>(n_expert), 1}));
    auto expert_in = std::make_shared<ov::op::v1::Reshape>(
        tiled, const_i32({static_cast<int32_t>(n_expert), -1, static_cast<int32_t>(n_embd)}), false);
    auto gate_out = std::make_shared<ov::op::v0::MatMul>(expert_in, gate_w, false, true);
    auto activated = std::make_shared<ov::op::v4::Swish>(gate_out);
    auto up_out = std::make_shared<ov::op::v0::MatMul>(expert_in, up_w, false, true);
    auto merged = std::make_shared<ov::op::v1::Multiply>(activated, up_out);
    auto down_out = std::make_shared<ov::op::v0::MatMul>(merged, down_w, false, true);
    auto expert_out = std::make_shared<ov::op::v1::Reshape>(
        down_out, const_i32({static_cast<int32_t>(n_expert), 1, -1, static_cast<int32_t>(n_embd)}), false);
    auto weighted = std::make_shared<ov::op::v1::Multiply>(expert_out, router_scores);
    ov::Output<ov::Node> reduced = std::make_shared<ov::op::v1::ReduceSum>(weighted, const_i32({0}), false);
    if (n_pad > 0) {
        // [1, tokens, n_embd]: keep the rows of the block
        reduced = std::make_shared<ov::op::v8::Slice>(
            reduced, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {n_rows}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}),
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1}));
    }

    // ggml [n_embd, n_tokens] -> OV [1, 1, n_tokens, n_embd]
    ov::Output<ov::Node> result = std::make_shared<ov::op::v1::Reshape>(
        reduced, const_i32({1, 1, -1, static_cast<int32_t>(n_embd)}), false);
    if (result.get_element_type() != context.get_output_type()) {
        result = std::make_shared<ov::op::v0::Convert>(result, context.get_output_type());
    }
    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
