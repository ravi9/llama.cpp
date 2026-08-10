#include "ggml-openvino-quantization.h"

#include "ggml-openvino-extra.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <optional>

std::optional<ExtraQuantType> ggml_openvino_get_requant_type(const ggml_tensor * tensor, bool no_requant) {
    if (no_requant) {
        return std::nullopt;
    }
    if (strncmp(tensor->name, "token_embd.weight", 17) == 0) {
        return ((ggml_openvino_is_npu() && tensor->type == GGML_TYPE_Q6_K) ? ExtraQuantType::F16 :
                                                                             ExtraQuantType::Q8_0_C);
    }
    if (strncmp(tensor->name, "output.weight", 13) == 0) {
        return ExtraQuantType::Q8_0_C;
    }
    if (ggml_openvino_is_npu()) {
        return ExtraQuantType::Q4_0_128;
    }
    switch (tensor->type) {
    case GGML_TYPE_Q6_K:
    case GGML_TYPE_Q5_K:
        return ExtraQuantType::Q8_0_C;
    default:
        return std::nullopt;
    }
}

// =====================================================
// Extracted Layout Calculation
// =====================================================

ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_tensor * tensor, bool use_bias) {
    ggml_openvino_extracted_layout layout = {};
    layout.is_symmetric = false;

    if (!ggml_is_quantized(tensor->type)) {
        return layout;
    }

    // Most quantized weights use the existing 2D extraction path. 3D expert weights for
    // MUL_MAT_ID (MoE) are also supported, either as MXFP4 (packed, dedicated branch below) or via the
    // generic sizing math below, which is shape-agnostic (based on total element count). Only reject 4D.
    if (tensor->ne[3] != 1) {
        return layout;
    }

    // 3D MoE expert weights that are not requantized (see below) always use the exact f16
    // zero-point extraction (see extract_quantized_weights), which needs a wider zp slot than
    // the packed integer zero point -- must be kept in sync with that function so the buffer
    // sizing here matches what process_weight_tensor actually writes.
    const bool for_gather_matmul = tensor->ne[2] > 1;

    int64_t n_elements = ggml_nelements(tensor);
    const size_t alignment = 64;  // Good for SIMD

    if (tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        layout.weights_per_block = 32;
        layout.is_symmetric = true;
        layout.weights_size = ggml_nbytes(tensor);
        layout.weights_offset = 0;
        layout.total_size = layout.weights_size;
        return layout;
    }

    // Check if requantization is needed (NPU-specific)
    auto requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    if (requant_type.has_value()) {
        layout.is_requant = true;
        layout.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        if (requant_type.value() == ExtraQuantType::F16) {
            layout.weights_size = n_elements * sizeof(uint16_t);  // F16 = 2 bytes
            layout.total_size = layout.weights_size;
            layout.weights_offset = 0;
            // No scales/zp for F16
            return layout;
        }

        // Requant to different quantized format (e.g., Q4_0_128)
        switch (requant_type.value()) {
        case ExtraQuantType::Q4_0_128:
            layout.is_u4 = true;
            layout.weights_per_block = 128;
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q4_0_C:
            layout.is_u4 = true;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_0_32:
            layout.is_u4 = false;
            layout.weights_per_block = 32;
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_0_C:
            layout.is_u4 = false;
            layout.weights_per_block = tensor->ne[0];
            layout.is_symmetric = true;
            break;
        case ExtraQuantType::Q8_1_C:
            layout.is_u4 = false;
            layout.weights_per_block = tensor->ne[0];
            break;
        default:
            layout.weights_per_block = -1;
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
            break;
        }

        if (layout.is_requant) {
            // Calculate sizes for requantized format
            layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_blocks = n_elements / layout.weights_per_block;
            layout.scales_size = n_blocks * sizeof(uint16_t);
            // For symmetric quantization, no zp needed (weights stored as signed)
            if (layout.is_symmetric) {
                layout.zp_size = 0;
            } else {
                layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
            }

            layout.weights_offset = 0;
            layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
            layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
            layout.total_size = layout.zp_offset + layout.zp_size;
            layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));
            return layout;
        }
    }

    // Normal extraction (no requant) - determine format based on tensor type
    layout.is_u4 = false;
    layout.weights_per_block = 32;
    layout.is_symmetric = false;

    switch (tensor->type) {
    case GGML_TYPE_MXFP4:
        layout.is_u4 = true;
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q4_0:
        layout.is_u4 = true;
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q4_K:
        layout.is_u4 = true;
        break;

    case GGML_TYPE_Q8_0:
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q5_1:
        // u8 weights (5-bit values), asymmetric (scale + zero point)
        break;

    case GGML_TYPE_Q6_K:
        layout.weights_per_block = 16;
        layout.is_symmetric = true;
        break;

    case GGML_TYPE_Q5_K:
        break;

    default:
        // Unsupported quantization type
        return layout;
    }

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    layout.weights_size = layout.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block, except MXFP4 which stores one E8M0 byte per block.
    int64_t n_blocks = n_elements / layout.weights_per_block;
    layout.scales_size = n_blocks * (tensor->type == GGML_TYPE_MXFP4 ? sizeof(uint8_t) : sizeof(uint16_t));
    // For symmetric quantization, no zp needed (weights stored as signed). Asymmetric
    // for_gather_matmul (3D MoE expert) weights use an exact f16 zero point (see
    // extract_quantized_weights/make_int8_weights/make_int4_weights), which needs one f16 per
    // block instead of a packed u4/u8 integer zero point.
    if (layout.is_symmetric) {
        layout.zp_size = 0;
    } else if (use_bias || for_gather_matmul) {
        layout.zp_size = n_blocks * sizeof(uint16_t);
    } else {
        layout.zp_size = layout.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
    }

    // Layout in buffer: [weights | scales | zp] with alignment
    layout.weights_offset = 0;
    layout.scales_offset = ((layout.weights_size + alignment - 1) / alignment) * alignment;
    layout.zp_offset = layout.scales_offset + ((layout.scales_size + alignment - 1) / alignment) * alignment;
    layout.total_size = layout.zp_offset + layout.zp_size;
    layout.total_size = std::max(layout.total_size, ggml_nbytes(tensor));

    return layout;
}
