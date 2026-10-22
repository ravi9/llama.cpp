// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "gated_delta_net.hpp"
#include "openvino/op/op.hpp"

// Internal OpenVINO developer API mirror for Qwen3.5/Qwen3.6 in llama.cpp.
// Keep declarations and class layouts in sync with the patched OpenVINO runtime.
// Port/precedence contracts: dev_doc/OPENVINO-RECURRENT-CACHE.md and OV developer header.
namespace ov::op::internal {
// Inputs: original cache, active/extra rows, reset begin/length, extra offsets,
// QKV [1,B,C,T], ordered (window start,destination row) pairs and depthwise
// weights [1,1,C,H+1]. Outputs: convolution [1,B,T,C] and full updated cache_r.
// Original cache reads remain immutable for cyclic row maps.
class OPENVINO_API RecurrentConvCache : public ov::op::Op {
public:
    OPENVINO_OP("RecurrentConvCache");
    RecurrentConvCache() = default;
    RecurrentConvCache(const OutputVector & args, int64_t history);
    void validate_and_infer_types() override;
    bool visit_attributes(AttributeVisitor & visitor) override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & args) const override;

    int64_t get_history() const { return m_history; }

private:
    int64_t m_history = 0;
};

// Attention and full updated cache_s; generic reset/reorder and MTP snapshot writes.
class OPENVINO_API RecurrentGatedDeltaNet : public GatedDeltaNet {
public:
    OPENVINO_OP("RecurrentGatedDeltaNet", "internal", GatedDeltaNet);
    RecurrentGatedDeltaNet() = default;
    RecurrentGatedDeltaNet(const OutputVector & args,
                           bool fuse_qk_l2norm = false,
                           float q_eps = 1e-6f,
                           float k_eps = 1e-6f,
                           int64_t snapshot_count = 1);
    void validate_and_infer_types() override;
    std::shared_ptr<Node> clone_with_new_inputs(const OutputVector & args) const override;
};
}  // namespace ov::op::internal
