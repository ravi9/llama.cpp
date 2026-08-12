#include "ggml-openvino-quantization.h"

#include "ggml-openvino-extra.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <algorithm>
#include <cstring>
#include <optional>

namespace {

static constexpr size_t QUANT_LAYOUT_ALIGNMENT = 64;

struct QuantStoragePlan {
    size_t total_size = 0;
    size_t weights_offset = 0;
    size_t weights_size = 0;
    size_t scales_offset = 0;
    size_t scales_size = 0;
    size_t zp_offset = 0;
    size_t zp_size = 0;
    bool is_u4 = false;
    int64_t weights_per_block = 0;
    bool is_symmetric = false;
    bool is_requant = false;
    std::optional<ExtraQuantType> requant_type;
};

struct QuantFormatSpec {
    bool is_u4 = false;
    int64_t weights_per_block = 32;
    bool is_symmetric = false;
};

bool get_requant_format_spec(ExtraQuantType type, int64_t ne0, QuantFormatSpec & spec) {
    switch (type) {
    case ExtraQuantType::Q4_0_128:
        spec.is_u4 = true;
        spec.weights_per_block = 128;
        spec.is_symmetric = true;
        return true;
    case ExtraQuantType::Q4_0_C:
        spec.is_u4 = true;
        spec.weights_per_block = ne0;
        spec.is_symmetric = true;
        return true;
    case ExtraQuantType::Q8_0_32:
        spec.is_u4 = false;
        spec.weights_per_block = 32;
        spec.is_symmetric = true;
        return true;
    case ExtraQuantType::Q8_0_C:
        spec.is_u4 = false;
        spec.weights_per_block = ne0;
        spec.is_symmetric = true;
        return true;
    case ExtraQuantType::Q8_1_C:
        spec.is_u4 = false;
        spec.weights_per_block = ne0;
        spec.is_symmetric = false;
        return true;
    default:
        return false;
    }
}

bool get_source_quant_format_spec(ggml_type type, QuantFormatSpec & spec) {
    spec = {};
    switch (type) {
    case GGML_TYPE_MXFP4:
        spec.is_u4 = true;
        spec.is_symmetric = true;
        return true;
    case GGML_TYPE_Q4_0:
        spec.is_u4 = true;
        spec.is_symmetric = true;
        return true;
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q4_K:
        spec.is_u4 = true;
        return true;
    case GGML_TYPE_Q8_0:
        spec.is_symmetric = true;
        return true;
    case GGML_TYPE_Q5_1:
        // u8 weights (5-bit values), asymmetric (scale + zero point)
        return true;
    case GGML_TYPE_Q6_K:
        spec.weights_per_block = 16;
        spec.is_symmetric = true;
        return true;
    case GGML_TYPE_Q5_K:
        return true;
    default:
        return false;
    }
}

void apply_quant_format_spec(QuantStoragePlan & plan, const QuantFormatSpec & spec) {
    plan.is_u4 = spec.is_u4;
    plan.weights_per_block = spec.weights_per_block;
    plan.is_symmetric = spec.is_symmetric;
}

size_t align_quant_layout_offset(size_t size) {
    return ((size + QUANT_LAYOUT_ALIGNMENT - 1) / QUANT_LAYOUT_ALIGNMENT) * QUANT_LAYOUT_ALIGNMENT;
}

void finalize_quant_storage_plan(QuantStoragePlan & plan, size_t min_total_size) {
    plan.weights_offset = 0;
    plan.scales_offset = align_quant_layout_offset(plan.weights_size);
    plan.zp_offset = plan.scales_offset + align_quant_layout_offset(plan.scales_size);
    plan.total_size = plan.zp_offset + plan.zp_size;
    plan.total_size = std::max(plan.total_size, min_total_size);
}

ggml_openvino_extracted_layout to_extracted_layout(const QuantStoragePlan & plan) {
    ggml_openvino_extracted_layout layout = {};
    layout.total_size = plan.total_size;
    layout.weights_offset = plan.weights_offset;
    layout.weights_size = plan.weights_size;
    layout.scales_offset = plan.scales_offset;
    layout.scales_size = plan.scales_size;
    layout.zp_offset = plan.zp_offset;
    layout.zp_size = plan.zp_size;
    layout.is_u4 = plan.is_u4;
    layout.weights_per_block = plan.weights_per_block;
    layout.is_symmetric = plan.is_symmetric;
    layout.is_requant = plan.is_requant;
    layout.requant_type = plan.requant_type;
    return layout;
}

}  // namespace

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
    QuantStoragePlan plan;

    if (!ggml_is_quantized(tensor->type)) {
        return to_extracted_layout(plan);
    }

    // Most quantized weights use the existing 2D extraction path. 3D expert weights for
    // MUL_MAT_ID (MoE) are also supported, either as MXFP4 (packed, dedicated branch below) or via the
    // generic sizing math below, which is shape-agnostic (based on total element count). Only reject 4D.
    if (tensor->ne[3] != 1) {
        return to_extracted_layout(plan);
    }

    // 3D MoE expert weights that are not requantized (see below) always use the exact f16
    // zero-point extraction (see extract_quantized_weights), which needs a wider zp slot than
    // the packed integer zero point -- must be kept in sync with that function so the buffer
    // sizing here matches what process_weight_tensor actually writes.
    const bool for_gather_matmul = tensor->ne[2] > 1;

    int64_t n_elements = ggml_nelements(tensor);
    const size_t raw_size = ggml_nbytes(tensor);

    if (tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        plan.weights_per_block = 32;
        plan.is_symmetric = true;
        plan.weights_size = raw_size;
        plan.weights_offset = 0;
        plan.total_size = plan.weights_size;
        return to_extracted_layout(plan);
    }

    // Check if requantization is needed (NPU-specific)
    auto requant_type = ggml_openvino_get_requant_type(tensor, use_bias);
    if (requant_type.has_value()) {
        plan.is_requant = true;
        plan.requant_type = requant_type;

        // Special case: requant to F16 - just store F16 weights, no scales/zp
        if (requant_type.value() == ExtraQuantType::F16) {
            plan.weights_size = n_elements * sizeof(uint16_t);  // F16 = 2 bytes
            plan.total_size = plan.weights_size;
            plan.weights_offset = 0;
            // No scales/zp for F16
            return to_extracted_layout(plan);
        }

        QuantFormatSpec spec;
        if (!get_requant_format_spec(requant_type.value(), tensor->ne[0], spec)) {
            plan.weights_per_block = -1;
            GGML_ABORT("Code of re-quantizing to channel-wise is not updated");
        }
        apply_quant_format_spec(plan, spec);

        if (plan.is_requant) {
            // Calculate sizes for requantized format
            plan.weights_size = plan.is_u4 ? (n_elements / 2) : n_elements;
            int64_t n_blocks = n_elements / plan.weights_per_block;
            plan.scales_size = n_blocks * sizeof(uint16_t);
            // For symmetric quantization, no zp needed (weights stored as signed)
            if (plan.is_symmetric) {
                plan.zp_size = 0;
            } else {
                plan.zp_size = plan.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
            }

            finalize_quant_storage_plan(plan, raw_size);
            return to_extracted_layout(plan);
        }
    }

    QuantFormatSpec source_spec;
    if (!get_source_quant_format_spec(tensor->type, source_spec)) {
        // Unsupported quantization type
        return to_extracted_layout(plan);
    }
    apply_quant_format_spec(plan, source_spec);

    // Calculate sizes
    // Weights: U4 = n_elements/2 bytes, U8 = n_elements bytes
    plan.weights_size = plan.is_u4 ? (n_elements / 2) : n_elements;

    // Scales: F16 per block, except MXFP4 which stores one E8M0 byte per block.
    int64_t n_blocks = n_elements / plan.weights_per_block;
    plan.scales_size = n_blocks * (tensor->type == GGML_TYPE_MXFP4 ? sizeof(uint8_t) : sizeof(uint16_t));
    // For symmetric quantization, no zp needed (weights stored as signed). Asymmetric
    // for_gather_matmul (3D MoE expert) weights use an exact f16 zero point (see
    // extract_quantized_weights/make_int8_weights/make_int4_weights), which needs one f16 per
    // block instead of a packed u4/u8 integer zero point.
    if (plan.is_symmetric) {
        plan.zp_size = 0;
    } else if (use_bias || for_gather_matmul) {
        plan.zp_size = n_blocks * sizeof(uint16_t);
    } else {
        plan.zp_size = plan.is_u4 ? ((n_blocks + 1) / 2) : n_blocks;
    }

    // Layout in buffer: [weights | scales | zp] with alignment
    finalize_quant_storage_plan(plan, raw_size);

    return to_extracted_layout(plan);
}
