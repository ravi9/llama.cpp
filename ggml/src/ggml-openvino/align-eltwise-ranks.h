#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ggml_openvino {

// Gives a binary eltwise op's two operands the same rank, by unsqueezing leading axes onto the shorter one.
//
// This is semantically a no-op (NUMPY broadcasting already left-pads the lower-rank operand with 1s, and the result
// rank is max(rank_a, rank_b) either way). It exists purely to work around an OpenVINO GPU-plugin defect: an eltwise
// op whose operands differ in rank is computed incorrectly once the plugin fuses it as a post-op into an `rms`
// primitive (e.g. a rank-3 residual added to a rank-4 RMS-norm output; the model degenerates into repeated tokens on
// GPU while CPU is correct). Equalising the ranks keeps the fusion and makes it compute the right answer.
//
// Only applies when both operands are real tensors: a tensor-vs-Constant mismatch is the RMS norm's own eps/rsqrt
// arithmetic, which folds into the `rms` primitive and is not affected. The norm output itself is never unsqueezed:
// when it is the lower-rank operand, that breaks the fused path instead. Registered for GPU only, disabled by
// GGML_OPENVINO_DISABLE_ELTWISE_RANK_ALIGN. Remove once the plugin guards that fusion.
class AlignEltwiseOperandRanks : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ggml_openvino::AlignEltwiseOperandRanks")
    AlignEltwiseOperandRanks();
};

}  // namespace ggml_openvino
