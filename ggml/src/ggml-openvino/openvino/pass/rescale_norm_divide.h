#pragma once

#include <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

class RescaleNormDivide : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::RescaleNormDivide")
    RescaleNormDivide();
};

class FuseSquareSum : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::FuseSquareSum")
    FuseSquareSum();
};

class RescaleReduceL2 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::RescaleReduceL2")
    RescaleReduceL2();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
