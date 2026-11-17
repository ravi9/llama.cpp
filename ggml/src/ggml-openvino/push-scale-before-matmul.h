#pragma once

#include "openvino/pass/matcher_pass.hpp"

namespace ggml_openvino {

// Moves a constant scale that follows a MatMul in front of it: MatMul(A, W) * c -> MatMul(A * c, W) (exact, MatMul is
// linear in A). When A is itself a product -- the gated activation in front of an FFN down projection,
// silu(gate) * up -- the scale is applied to one factor, so neither the product nor the MatMul result is ever larger
// than the scaled value. This matters for models that scale a block's output by a constant below one (Granite's
// residual multiplier) while the block itself produces values beyond the f16 range on massive-activation tokens: the
// plugin executes the graph in f16, and the unscaled intermediate overflows to inf even though the scaled result is
// small. Only scales in [1/64, 1) are moved: larger ones would not reduce any magnitude, and much smaller ones would
// push the intermediate toward f16 subnormals.
//
// Registered only for devices that execute in f16; on f32 it is pure overhead (the Multiply after a MatMul can fuse
// into it as a post-op, the one before cannot).
class PushScaleBeforeMatMul : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ggml_openvino::PushScaleBeforeMatMul")
    PushScaleBeforeMatMul();
};

}  // namespace ggml_openvino
