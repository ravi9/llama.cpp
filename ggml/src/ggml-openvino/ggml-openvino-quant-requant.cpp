#include "ggml-openvino-quant-weights.h"

#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/type/element_type_traits.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <string>
#include <vector>

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
        // and per-layer Q6_K/Q5_K requant -- the large transient cases).
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
