// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Local mirror of OpenVINO's internal op; implementation comes from the linked runtime.
// Keep in sync with src/common/transformations/include/ov_ops/gather_matmul_compressed.hpp.

#pragma once

#include "gather_matmul.hpp"

namespace ov::op::internal {

class OPENVINO_API GatherMatmulCompressed : public GatherMatmul {
public:
    OPENVINO_OP("GatherMatmulCompressed", "", GatherMatmul);

    GatherMatmulCompressed() = default;

    GatherMatmulCompressed(const ov::Output<Node>& A,
                           const ov::Output<Node>& B,
                           const ov::Output<Node>& indices,
                           const ov::Output<Node>& bias,
                           const ov::Output<Node>& weight_scales,
                           const ov::Output<Node>& weight_zero_points);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;

    void validate_and_infer_types() override;
};

}  // namespace ov::op::internal
