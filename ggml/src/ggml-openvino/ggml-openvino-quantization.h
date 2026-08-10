#pragma once

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <optional>

// Requantization target formats for OpenVINO weight extraction.
enum class ExtraQuantType { F16, Q4_0_C, Q8_1_C, Q4_0_128, Q8_0_C, Q8_0_32 };

// Get requantization type for a tensor type (returns nullopt if no requant needed)
std::optional<ExtraQuantType> ggml_openvino_get_requant_type(const ggml_tensor * tensor, bool no_requant = false);

// For quantized tensors, we need extra space to store extracted weights, scales, and zero points.
struct ggml_openvino_extracted_layout {
    size_t total_size = 0;      // Total bytes needed
    size_t weights_offset = 0;  // Offset to weights in buffer
    size_t weights_size = 0;    // Size of weights in bytes
    size_t scales_offset = 0;   // Offset to scales in buffer
    size_t scales_size = 0;     // Size of scales in bytes
    size_t zp_offset = 0;       // Offset to zero points in buffer
    size_t zp_size = 0;         // Size of zero points in bytes (U4 or U8)
    bool is_u4;                 // true for U4 weights, false for U8
    int64_t weights_per_block;  // weights per scale/zp block
    bool is_symmetric;          // true for symmetric quantization

    // Requantization info
    bool is_requant = false;                     // true if this tensor needs requantization
    std::optional<ExtraQuantType> requant_type;  // target requant type if is_requant
};

// Calculate the buffer layout for extracted quantized data
ggml_openvino_extracted_layout ggml_openvino_get_extracted_layout(const ggml_tensor * tensor, bool use_bias = false);

inline const char * extra_quant_type_name(ExtraQuantType t) {
    switch (t) {
    case ExtraQuantType::F16:
        return "F16";
    case ExtraQuantType::Q4_0_C:
        return "Q4_0_C";
    case ExtraQuantType::Q4_0_128:
        return "Q4_0_128";
    case ExtraQuantType::Q8_0_C:
        return "Q8_0_C";
    case ExtraQuantType::Q8_0_32:
        return "Q8_0_32";
    case ExtraQuantType::Q8_1_C:
        return "Q8_1_C";
    default:
        return "unknown";
    }
}
