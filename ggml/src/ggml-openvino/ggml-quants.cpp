#include "ggml-quants.h"

#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/parallel.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/element_type_traits.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/core/type/float4_e2m1.hpp>
#include <openvino/core/type/float8_e8m0.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/util/attr_types.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>
#include <vector>

static ov::Shape make_weight_shape(const ggml_tensor * tensor) {
    return (tensor->ne[2] > 1) ? ov::Shape{static_cast<size_t>(tensor->ne[2]), static_cast<size_t>(tensor->ne[1]),
                                           static_cast<size_t>(tensor->ne[0])} :
                                  ov::Shape{static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
}

static ov::Shape make_mxfp4_weight_shape(const ggml_tensor * tensor) {
    if (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
        return {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
    }

    ov::Shape shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    return shape;
}

static ov::Shape make_packed_mxfp4_moe_shape(const ggml_tensor * tensor) {
    return {static_cast<size_t>(tensor->ne[3]),
            static_cast<size_t>(tensor->ne[2]),
            static_cast<size_t>(tensor->ne[1]),
            static_cast<size_t>(tensor->ne[0] / MXFP4_BLOCK_SIZE),
            MXFP4_BLOCK_BYTES};
}

// Extract quantized weights from tensor and create weight subgraph
std::shared_ptr<ov::Node> extract_quantized_weights(const ggml_tensor * tensor,
                                                    const void * data,
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    bool use_bias) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor temp_tensor = *tensor;
    temp_tensor.data = const_cast<void *>(data);

    if (tensor->type == GGML_TYPE_MXFP4) {
        extract_mxfp4_data(&temp_tensor, weights, scales);
        auto result = make_mxfp4_weights(weights, scales).get_node_shared_ptr();
        result->set_friendly_name(tensor->name);
        return result;
    }

    // Determine block size based on tensor type
    int64_t weights_per_block;
    bool is_u4;
    switch (tensor->type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q4_K:
        is_u4 = true;
        weights_per_block = 32;
        break;
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q5_K:
        is_u4 = false;
        weights_per_block = 32;
        break;
    case GGML_TYPE_Q6_K:
        is_u4 = false;
        weights_per_block = 16;
        break;
    default:
        throw std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    const bool for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    switch (tensor->type) {
    case GGML_TYPE_Q4_0:
        extract_q4_0_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q4_1:
        extract_q4_1_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    case GGML_TYPE_Q4_K:
        extract_q4_k_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    case GGML_TYPE_Q5_1:
        extract_q5_1_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    case GGML_TYPE_Q8_0:
        extract_q8_0_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q6_K:
        extract_q6_k_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q5_K:
        extract_q5_k_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    default:
        throw std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> weight_node;
    if (is_u4) {
        weight_node = make_int4_weights(weights, scales, zp, weights_per_block, use_bias, for_gather_matmul);
    } else {
        weight_node = make_int8_weights(weights, scales, zp, weights_per_block, use_bias, for_gather_matmul);
    }

    auto result = weight_node.get_node_shared_ptr();
    result->set_friendly_name(tensor->name);
    return result;
}

// Requantize weights to target format, writing to provided buffers
std::shared_ptr<ov::Node> requantize_to_buffers(const ggml_tensor * tensor,
                                                const void * data,
                                                ExtraQuantType requant_type,
                                                int64_t block_size,
                                                ov::Tensor & weights,
                                                ov::Tensor & scales,
                                                ov::Tensor & zp) {
    int64_t n_elements = ggml_nelements(tensor);
    const int64_t ne0 = tensor->ne[0];                 // elements per row
    const int64_t n_rows = n_elements / ne0;
    const auto * type_traits = ggml_get_type_traits(tensor->type);
    const size_t src_row_bytes = ggml_row_size(tensor->type, ne0);

    bool is_u4 = (requant_type == ExtraQuantType::Q4_0_C || requant_type == ExtraQuantType::Q4_0_128);

    // Streaming dequant (opt-in via GGML_OPENVINO_REDUCE_COMPILE_MEM or
    // GGML_OPENVINO_MEMORY_OPTIMIZE): instead of
    // materializing the full n_elements F32 array (e.g. ~1 GB for token_embd), dequantize
    // a chunk of complete rows into a small scratch and quantize/convert it straight into
    // the output buffers, capping the transient F32 footprint at CHUNK_ROWS*ne0 floats.
    //
    // Only valid (and only used) for the Q8_0_C / Q8_1_C / F16 targets whose block size
    // divides a row (channel-wise _C uses block_size == ne0) so no target block straddles
    // a row boundary, and Q8/F16 have no cross-block packing. The u4 (Q4_0) path packs two
    // weights per byte with running zp ORs that assume a single whole-array call, so it is
    // never streamed. When the flag is off, behavior is identical to the original
    // full-materialization path.
    const bool stream_requant = ggml_openvino_reduce_compile_mem_enabled() && !is_u4 &&
                                !(block_size > 0 && ne0 % block_size != 0);

    if (!stream_requant) {
        // Full materialization (original behavior): dequantize the whole tensor to F32,
        // then convert/quantize in one call.
        std::vector<float> weights_f32(n_elements);
        type_traits->to_float(data, weights_f32.data(), n_elements);
        if (requant_type == ExtraQuantType::F16) {
            ggml_get_type_traits(GGML_TYPE_F16)->from_float_ref(weights_f32.data(), weights.data(), n_elements);
            auto result = std::make_shared<ov::op::v0::Constant>(weights);
            result->set_friendly_name(tensor->name);
            return result;
        }
        if (is_u4) {
            quantize_q4_0(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        } else if (requant_type == ExtraQuantType::Q8_1_C) {
            quantize_q8_1(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        } else {
            quantize_q8_0(weights_f32.data(), weights, scales, zp, n_elements, block_size);
        }
    } else {
        // Streaming path for Q8_0_C / Q8_1_C / F16 (covers token_embd, output.weight,
        // and per-layer Q6_K/Q5_K requant — the large transient cases).
        const int64_t CHUNK_ROWS = std::min<int64_t>(n_rows, 256);
        std::vector<float> scratch(CHUNK_ROWS * ne0);
        // F16 destination: 2 bytes/element, advanced per chunk by r0*ne0 elements.
        auto * f16_base = static_cast<uint8_t *>(weights.data());
        for (int64_t r0 = 0; r0 < n_rows; r0 += CHUNK_ROWS) {
            const int64_t rows = std::min(CHUNK_ROWS, n_rows - r0);
            const int64_t elems = rows * ne0;
            const auto * src = static_cast<const uint8_t *>(data) + r0 * src_row_bytes;
            type_traits->to_float(src, scratch.data(), elems);

            if (requant_type == ExtraQuantType::F16) {
                ggml_get_type_traits(GGML_TYPE_F16)
                    ->from_float_ref(scratch.data(), f16_base + (r0 * ne0) * sizeof(uint16_t), elems);
            } else {
                const int64_t block_offset = (r0 * ne0) / block_size;
                if (requant_type == ExtraQuantType::Q8_1_C) {
                    quantize_q8_1(scratch.data(), weights, scales, zp, elems, block_size, block_offset);
                } else {
                    quantize_q8_0(scratch.data(), weights, scales, zp, elems, block_size, block_offset);
                }
            }
        }
        if (requant_type == ExtraQuantType::F16) {
            auto result = std::make_shared<ov::op::v0::Constant>(weights);
            result->set_friendly_name(tensor->name);
            return result;
        }
    }

    // Create the OpenVINO weight subgraph
    ov::Output<ov::Node> weight_node;
    if (is_u4) {
        weight_node = make_int4_weights(weights, scales, zp, block_size);
    } else {
        weight_node = make_int8_weights(weights, scales, zp, block_size);
    }

    auto result = weight_node.get_node_shared_ptr();
    result->set_friendly_name(tensor->name);
    return result;
}

OvWeight process_weight_tensor(const ggml_tensor * tensor, const void * data, void * output_base_ptr, bool use_bias) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(data != nullptr);

    OvWeight result;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape node_shape = make_weight_shape(tensor);

    // Handle F16/F32/BF16 weights
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        switch (tensor->type) {
        case GGML_TYPE_F32:
            element_type = ov::element::f32;
            break;
        case GGML_TYPE_F16:
            element_type = ov::element::f16;
            break;
        case GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            break;
        default:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        if (output_base_ptr && output_base_ptr != data) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_bytes = ggml_nbytes(tensor);
            memcpy(output_base_ptr, data, tensor_bytes);
            result.weights = ov::Tensor(element_type, node_shape, output_base_ptr);
        } else {
            result.weights = ov::Tensor(element_type, node_shape, data);
        }
        result.weight_node = std::make_shared<ov::op::v0::Constant>(result.weights);
        return result;
    }

    // Handle quantized weights
    if (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    result.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    const auto & layout = result.layout;
    if (layout.total_size == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    const bool for_gather_matmul = tensor->ne[2] > 1;
    const bool zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    const bool is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    if (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = make_packed_mxfp4_moe_shape(tensor);
        const size_t tensor_bytes = ggml_nbytes(tensor);
        if (output_base_ptr) {
            auto * buf_base = static_cast<uint8_t *>(output_base_ptr);
            memcpy(buf_base + layout.weights_offset, data, tensor_bytes);
            result.weights = ov::Tensor(ov::element::u8, packed_shape, buf_base + layout.weights_offset);
        } else {
            result.weights = ov::Tensor(ov::element::u8, packed_shape);
            memcpy(result.weights.data(), data, tensor_bytes);
        }
        result.weight_node = make_mxfp4_moe_packed_weights(result.weights).get_node_shared_ptr();
        result.weight_node->set_friendly_name(tensor->name);
        return result;
    }

    if (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_base_ptr = nullptr;
    }

    // F16 requant path - no separate scales/zp needed in result
    if (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        if (output_base_ptr) {
            result.weights = ov::Tensor(ov::element::f16, node_shape,
                                        static_cast<uint8_t *>(output_base_ptr) + layout.weights_offset);
        } else {
            result.weights = ov::Tensor(ov::element::f16, node_shape);
        }
        ov::Tensor dummy_scales, dummy_zp;  // Not used for F16
        result.weight_node =
            requantize_to_buffers(tensor, data, ExtraQuantType::F16, 0, result.weights, dummy_scales, dummy_zp);
        return result;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type weight_type = tensor->type == GGML_TYPE_MXFP4 ?
                                        ov::element::f4e2m1 :
                                        (layout.is_symmetric ? (layout.is_u4 ? ov::element::i4 : ov::element::i8) :
                                                               (layout.is_u4 ? ov::element::u4 : ov::element::u8));
    ov::Shape scale_shape = node_shape;
    scale_shape.back() /= layout.weights_per_block;

    if (tensor->type == GGML_TYPE_MXFP4) {
        node_shape = make_mxfp4_weight_shape(tensor);
        scale_shape = node_shape;
        scale_shape.back() /= layout.weights_per_block;
    }

    if (output_base_ptr) {
        uint8_t * buf_base = static_cast<uint8_t *>(output_base_ptr);
        result.weights = ov::Tensor(weight_type, node_shape, buf_base + layout.weights_offset);
        const ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
        result.scales = ov::Tensor(scale_type, scale_shape, buf_base + layout.scales_offset);
        if (!layout.is_symmetric) {
            ov::element::Type zp_type =
                zp_is_f16 ? ov::element::f16 : (layout.is_u4 ? ov::element::u4 : ov::element::u8);
            result.zp = ov::Tensor(zp_type, scale_shape, buf_base + layout.zp_offset);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } else {
        result.weights = ov::Tensor(weight_type, node_shape);
        const ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
        result.scales = ov::Tensor(scale_type, scale_shape);
        if (!layout.is_symmetric) {
            if (zp_is_f16) {
                result.zp = ov::Tensor(ov::element::f16, scale_shape);
            } else {
                ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
                result.zp = ov::Tensor(zp_type, scale_shape);
            }
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    if (layout.is_requant && layout.requant_type.has_value()) {
        result.weight_node = requantize_to_buffers(tensor, data, layout.requant_type.value(), layout.weights_per_block,
                                                   result.weights, result.scales, result.zp);
    } else {
        result.weight_node =
            extract_quantized_weights(tensor, data, result.weights, result.scales, result.zp, use_bias);
    }

    return result;
}

void quantize_q4_0(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk) {
    assert(k % qk == 0);
    const int nb = k / qk;

    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    if (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        for (int i = 0; i < nb; i++) {
            float amax = 0.0f;
            float max = 0.0f;
            for (int j = 0; j < qk; j++) {
                const float v = x[i * qk + j];
                if (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            const float d = max / -8;
            if (d == 0) {
                scales[i] = ov::float16(1.0f);
                if (i % 2 == 0) {
                    zp[i / 2] = 8;
                } else {
                    zp[i / 2] |= (8 << 4);
                }
                memset(weights + i * qk / 2, 8 | (8 << 4), qk / 2);
                continue;
            }
            const float id = 1.0f / d;
            scales[i] = ov::float16(d);
            if (i % 2 == 0) {
                zp[i / 2] = 8;
            } else {
                zp[i / 2] |= (8 << 4);
            }
            for (int j = 0; j < qk / 2; ++j) {
                const float x0 = x[i * qk + 2 * j] * id;
                const float x1 = x[i * qk + 2 * j + 1] * id;
                const uint8_t xi0 = MIN(15, (int8_t) (x0 + 8.5f));
                const uint8_t xi1 = MIN(15, (int8_t) (x1 + 8.5f));
                weights[i * qk / 2 + j] = xi0 | (xi1 << 4);
            }
        }
    } else {
        // Symmetric: produce signed i4 values in [-8, 7]
        for (int i = 0; i < nb; i++) {
            float amax = 0.0f;
            float max = 0.0f;
            for (int j = 0; j < qk; j++) {
                const float v = x[i * qk + j];
                if (amax < fabsf(v)) {
                    amax = fabsf(v);
                    max = v;
                }
            }
            const float d = max / -8;
            if (d == 0) {
                scales[i] = ov::float16(1.0f);
                // i4 value 0 packed: 0x00
                memset(weights + i * qk / 2, 0, qk / 2);
                continue;
            }
            const float id = 1.0f / d;
            scales[i] = ov::float16(d);
            for (int j = 0; j < qk / 2; ++j) {
                const float x0 = x[i * qk + 2 * j] * id;
                const float x1 = x[i * qk + 2 * j + 1] * id;
                // Signed i4: range [-8, 7]. Quantize as round(x*id), then pack as 4-bit two's complement.
                int8_t si0 = (int8_t) std::max(-8, std::min(7, (int) roundf(x0)));
                int8_t si1 = (int8_t) std::max(-8, std::min(7, (int) roundf(x1)));
                weights[i * qk / 2 + j] = (si0 & 0x0F) | ((si1 & 0x0F) << 4);
            }
        }
    }
}

void quantize_q8_0(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t block_offset) {
    assert(k % qk == 0);
    const int nb = k / qk;

    // block_offset lets a caller quantize a chunk of blocks into the right place in the
    // output buffers (used for streaming requant). x points at this chunk's first block;
    // outputs are advanced by block_offset blocks. Q8 has one scale/zp per block (no
    // nibble packing), so any block boundary is safe.
    auto * weights = static_cast<uint8_t *>(weights_arr.data()) + block_offset * qk;
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() + block_offset;
    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i8);  // Signed i8 path

    if (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data()) + block_offset;
        for (int i = 0; i < nb; i++) {
            float amax = 0.0f;
            for (int j = 0; j < qk; j++) {
                const float v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            const float d = amax / 127.0f;
            const float id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            zp[i] = 128;
            for (int j = 0; j < qk; ++j) {
                const float x0 = x[i * qk + j] * id;
                const int8_t xi0 = roundf(x0);
                weights[i * qk + j] = (uint8_t) (xi0 + 128);
            }
        }
    } else {
        // Symmetric: store signed int8 values directly
        auto * signed_weights = reinterpret_cast<int8_t *>(weights);
        for (int i = 0; i < nb; i++) {
            float amax = 0.0f;
            for (int j = 0; j < qk; j++) {
                const float v = x[i * qk + j];
                amax = std::max(amax, fabsf(v));
            }
            const float d = amax / 127.0f;
            const float id = d ? 1.0f / d : 0.0f;
            scales[i] = ov::float16(d);
            for (int j = 0; j < qk; ++j) {
                const float x0 = x[i * qk + j] * id;
                signed_weights[i * qk + j] = (int8_t) roundf(x0);
            }
        }
    }
}

void quantize_q8_1(const float * x,
                   ov::Tensor & weights_arr,
                   ov::Tensor & scales_arr,
                   ov::Tensor & zp_arr,
                   int64_t k,
                   int64_t qk,
                   int64_t block_offset) {
    assert(k % qk == 0);
    const int nb = k / qk;

    // See quantize_q8_0: block_offset places this chunk's output at the right block.
    auto * weights = static_cast<uint8_t *>(weights_arr.data()) + block_offset * qk;
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() + block_offset;
    auto * zp = static_cast<uint8_t *>(zp_arr.data()) + block_offset;
    for (int i = 0; i < nb; i++) {
        float min = std::numeric_limits<float>::max();
        float max = std::numeric_limits<float>::lowest();

        for (int j = 0; j < qk; j++) {
            const float v = x[i * qk + j];
            min = std::min(v, min);
            max = std::max(v, max);
        }

        const float d = (max - min) / ((1 << 8) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        scales[i] = ov::float16(d);
        // zp = -min / scale (Q8_1 is asymmetric)
        zp[i] = (d != 0.0f) ? (uint8_t) std::round(-min / d) : 0;

        for (int j = 0; j < qk; ++j) {
            const float x0 = (x[i * qk + j] - min) * id;
            const uint8_t xi0 = roundf(x0);
            weights[i * qk + j] = xi0;
        }
    }
}
