#pragma once

#include <openvino/op/op.hpp>
#include <string>

// llama.cpp-specific Qwen3.5/Qwen3.6 recurrent cache ops, executed on GPU through the
// OpenVINO custom-kernel mechanism (see openvino/kernels/recurrent_cache.xml). No OpenVINO
// runtime change is required.
//
// The update ops modify the cache input in place and do not return the updated cache. The
// translator must therefore bind the cache input directly to the caller's cache allocation
// (a Parameter, possibly through Reshapes) and keep that allocation exclusive to these ops.
// RecurrentCacheStage copies every source row read by the update before it is overwritten,
// which keeps reset, reorder cycles and duplicate sources correct.
namespace ov::frontend::ggml::op {

// Inputs: cache [1,1,S,F], active rows [1,1,1,B], extra rows [1,1,1,N], reset begin/length
// [1,1,1,1], extra offsets [1,1,1,2] = {destination begin, source begin}.
// Output: staged f32 rows [1,1,B+N,F].
class RecurrentCacheStage : public ov::op::Op {
public:
    OPENVINO_OP("GgmlRecurrentCacheStage", "ggml_opset")
    RecurrentCacheStage() = default;
    explicit RecurrentCacheStage(const OutputVector & args);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor &) override { return true; }
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & args) const override;
};

// Token inputs may be padded along T to a bucket; the token count input holds the real T.
// Inputs: the six stage inputs, QKV [1,B,C,T], snapshot windows [1,1,K,2] = {source start,
// destination row}, depthwise weights [1,1,C,history+1], staged rows, token count [1,1,1,1].
// Output: convolution [1,B,T,C]. Writes the cache in place.
class RecurrentConvUpdate : public ov::op::Op {
public:
    OPENVINO_OP("GgmlRecurrentConvUpdate", "ggml_opset")
    RecurrentConvUpdate() = default;
    RecurrentConvUpdate(const OutputVector & args, int64_t history);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor & visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & args) const override;

private:
    int64_t m_history = 0;
};

// Inputs: Q/K [B,T,HK,K], V [B,T,H,V], cache [1,1,S,H*V*K], gate/beta [B,T,H,1], active rows,
// extra rows, reset begin/length, extra offsets, snapshot write control [1,1,1,4] =
// {begin, sequences, stride, skipped sources}, staged rows, token count [1,1,1,1].
// Output: attention [B,T,H,V]. The newest min(T,snapshot_count) states are written into the
// cache, newest first. GQA heads use GGML's tiled order (h % HK).
class RecurrentGdnUpdate : public ov::op::Op {
public:
    OPENVINO_OP("GgmlRecurrentGdnUpdate", "ggml_opset")
    RecurrentGdnUpdate() = default;
    RecurrentGdnUpdate(const OutputVector & args, bool fuse_qk_l2norm, float q_eps, float k_eps, int64_t snapshot_count);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor & visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & args) const override;

protected:
    bool m_fuse_qk_l2norm = false;
    float m_q_eps = 1e-6f, m_k_eps = 1e-6f;
    int64_t m_snapshot_count = 1;
};

// Standalone snapshot GDN: the same inputs, with the initial state passed as cache and staged
// rows. The newest states go to output 1 [1,1,min(T,snapshot_count)*B,F]; nothing is written in place.
class RecurrentGdnSnapshots : public RecurrentGdnUpdate {
public:
    OPENVINO_OP("GgmlRecurrentGdnSnapshots", "ggml_opset", RecurrentGdnUpdate)
    RecurrentGdnSnapshots() = default;
    RecurrentGdnSnapshots(const OutputVector & args, bool fuse_qk_l2norm, float q_eps, float k_eps, int64_t snapshot_count);
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & args) const override;
};

// True if `cache` is a whole Parameter, possibly through Reshapes: only then do in-place
// writes reach the caller's allocation.
bool is_inplace_cache(const Output<Node> & cache);

// Keep an in-place cache input and its Reshape/Parameter chain out of FP16 compression,
// which would otherwise insert a converted copy for the kernel to write into.
void mark_inplace_cache(const Input<Node> & input);

// RecurrentCacheStage over the first six update inputs.
Output<Node> make_cache_stage(const OutputVector & inputs);

// Builders that pad a dynamic token axis to a power of two before the custom op (kernels are
// compiled per shape) and slice the outputs back. Inputs exclude the token count.
Output<Node> make_conv_update(OutputVector inputs, int64_t history, const std::string & name);
Output<Node> make_gdn_update(OutputVector inputs,
                             bool fuse_qk_l2norm,
                             float q_eps,
                             float k_eps,
                             int64_t snapshot_count,
                             const std::string & name);
// Returns {attention, newest states}.
OutputVector make_gdn_snapshots(OutputVector inputs,
                                bool fuse_qk_l2norm,
                                float q_eps,
                                float k_eps,
                                int64_t snapshot_count);

}  // namespace ov::frontend::ggml::op
