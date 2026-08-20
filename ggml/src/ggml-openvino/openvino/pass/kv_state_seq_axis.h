#include "openvino/pass/pass.hpp"

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

// Moves the sequence axis of the stateful KV cache from dim 1 to dim 2, i.e. from
// [1, seq, n_heads_kv, head_size] to [1, n_heads_kv, seq, head_size], and updates the
// Concat that appends to it. The GPU plugin only appends new tokens in place when the
// growing axis is a spatial axis, so growing dim 1 makes it copy the whole KV state
// every token (cost grows with context length). Only rewrites states that still match
// the frontend layout, so it no-ops if that layout ever changes.
class KVStateSeqAxis : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("ov::frontend::ggml::pass::KVStateSeqAxis")
    bool run_on_model(const std::shared_ptr<ov::Model> & model) override;
};

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
