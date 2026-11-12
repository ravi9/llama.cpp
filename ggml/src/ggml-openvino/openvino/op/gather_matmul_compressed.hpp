// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Local mirror of OpenVINO's internal GatherMatmulCompressed op.
//
// The op implementation is provided by OpenVINO's transformations library. Keep this
// declaration in sync with openvino/src/common/transformations/include/ov_ops/gather_matmul_compressed.hpp.

#pragma once

#include "gather_matmul.hpp"

#ifndef TRANSFORMATIONS_API
#    ifdef OPENVINO_STATIC_LIBRARY
#        define TRANSFORMATIONS_API
#    elif defined(IMPLEMENT_OPENVINO_API)
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    else
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif
#endif

namespace ov::op::internal {

class TRANSFORMATIONS_API GatherMatmulCompressed : public GatherMatmul {
public:
    OPENVINO_OP("GatherMatmulCompressed", "", GatherMatmul)

    GatherMatmulCompressed() = default;

    GatherMatmulCompressed(const ov::Output<Node> & A,
                           const ov::Output<Node> & B,
                           const ov::Output<Node> & indices,
                           const ov::Output<Node> & bias,
                           const ov::Output<Node> & weight_scales,
                           const ov::Output<Node> & weight_zero_points);

    std::shared_ptr<Node> clone_with_new_inputs(const ov::OutputVector & new_args) const override;

    void validate_and_infer_types() override;
};

}  // namespace ov::op::internal
