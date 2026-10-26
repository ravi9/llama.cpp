#pragma once

#include <openvino/pass/matcher_pass.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// GPU runs f32 graphs in f16; ReduceSum(x*x) overflows f16 (max 65504) for large rows, e.g. the
// Gemma 3n gaussian_topk std. Keep x*x -> ReduceSum [-> * k] [-> Sqrt] in f32.
class KeepSumSquareFP32 : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("ov::frontend::ggml::pass::KeepSumSquareFP32")
    KeepSumSquareFP32();
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
