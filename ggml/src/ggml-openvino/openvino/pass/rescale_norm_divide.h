#pragma once

#include <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// GPU-only fp16 overflow guards for L2 norms. Order: RescaleNormDivide, FuseSquareSum, RescaleReduceL2.

// scale x by 1/max|x| in x * s / ||x||; the ratio is unchanged, x*x stays in fp16 range
class RescaleNormDivide : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::RescaleNormDivide")
    RescaleNormDivide();
};

// Sqrt(ReduceSum(x*x)) -> ReduceL2(x)
class FuseSquareSum : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseSquareSum")
    FuseSquareSum();
};

// ReduceL2(x) -> c * ReduceL2(x / c), c = max|x| per row
class RescaleReduceL2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::RescaleReduceL2")
    RescaleReduceL2();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
