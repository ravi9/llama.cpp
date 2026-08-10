#include "ggml-quants.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <openvino/core/parallel.hpp>
#include <openvino/core/type/element_type_traits.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/core/type/float8_e8m0.hpp>

void unpack_32_4(const uint8_t * data, uint8_t * dst) {
    std::fill_n(dst, 16, 0);
    for (int j = 0; j < 16; ++j) {
        uint8_t x = (data[j] & 0x0F);
        uint8_t y = (data[j] >> 4);
        if (j % 2 != 0) {
            x <<= 4;
            y <<= 4;
        }
        dst[j / 2] |= x;
        dst[8 + j / 2] |= y;  // Last 16 weights are in the higher bits
    }
}

static void pack_32_mxfp4_for_openvino(const uint8_t * data, uint8_t * dst) {
    for (int j = 0; j < static_cast<int>(MXFP4_BLOCK_QS_SIZE); j += 2) {
        const uint8_t v0 = data[j] & 0x0F;
        const uint8_t v1 = (data[j + 1] & 0x0F) << 4;
        const uint8_t v16 = data[j] >> 4;
        const uint8_t v17 = data[j + 1] & 0xF0;
        dst[j / 2] = v0 | v1;
        dst[MXFP4_BLOCK_SIZE / 4 + j / 2] = v16 | v17;
    }
}

void extract_mxfp4_data(const ggml_tensor * tensor, ov::Tensor & weights_arr, ov::Tensor & scales_arr) {
    GGML_ASSERT(tensor->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(weights_arr.get_element_type() == ov::element::f4e2m1);
    GGML_ASSERT(scales_arr.get_element_type() == ov::element::f8e8m0);

    const auto * data = static_cast<const uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f8e8m0>::value_type>();
    const size_t n_blocks = scales_arr.get_size();

    ov::parallel_for(n_blocks, [&](size_t i) {
        const uint8_t * block = data + i * MXFP4_BLOCK_BYTES;
        pack_32_mxfp4_for_openvino(block + sizeof(uint8_t), weights + i * MXFP4_BLOCK_QS_SIZE);
        scales[i] = ov::float8_e8m0::from_bits(block[0]);
    });
}

void extract_q4_0_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    const uint64_t bytes_per_block = 18;  // 2 bytes scale, 32x0.5 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i4);  // Signed i4 path

    if (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
            if (i % 2 == 0) {
                zp[i / 2] = 8;
            } else {
                zp[i / 2] |= (8 << 4);
            }
            unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
        });
    } else {
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            scales[i] = ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block)));
            unpack_32_4(data + i * bytes_per_block + 2, weights + i * 16);
            for (int j = 0; j < 16; ++j) {
                weights[i * 16 + j] ^= 0x88;
            }
        });
    }
}

void extract_q4_1_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias) {
    const uint64_t bytes_per_block = 20;  // 2 bytes scale, 2 bytes min, 32x0.5 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    if (use_bias) {
        auto * bias = zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            float scale = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block))));
            float min = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
        });
    } else {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            float scale = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block))));
            float min = static_cast<float>(ov::float16::from_bits(*((uint16_t *) (data + i * bytes_per_block + 2))));
            scales[i] = ov::float16(scale);
            uint8_t zp_val = (scale != 0.0f) ? (uint8_t) std::round(-min / scale) : 0;
            if (i % 2 == 0) {
                zp[i / 2] = zp_val & 0x0F;
            } else {
                zp[i / 2] |= (zp_val << 4);
            }
            unpack_32_4(data + i * bytes_per_block + 4, weights + i * 16);
        });
    }
}

void extract_q5_1_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias) {
    const uint64_t bytes_per_block = 24;  // 2 scale + 2 min + 4 qh + 16 (32x0.5) weights
    const int qk = 32;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    auto read_u16 = [](const uint8_t * p) {
        uint16_t v;
        memcpy(&v, p, sizeof(v));
        return v;
    };

    auto unpack_block = [&](const uint8_t * block, uint8_t * dst) {
        uint32_t qh;
        memcpy(&qh, block + 4, sizeof(uint32_t));
        const uint8_t * qs = block + 8;
        for (int j = 0; j < qk / 2; ++j) {
            const uint8_t lo = qs[j] & 0x0F;
            const uint8_t hi = qs[j] >> 4;
            const uint8_t bit_lo = (qh >> j) & 1;
            const uint8_t bit_hi = (qh >> (j + qk / 2)) & 1;
            dst[j] = lo | (bit_lo << 4);
            dst[j + qk / 2] = hi | (bit_hi << 4);
        }
    };

    if (use_bias) {
        auto * bias = zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            const uint8_t * block = data + i * bytes_per_block;
            float scale = static_cast<float>(ov::float16::from_bits(read_u16(block)));
            float min = static_cast<float>(ov::float16::from_bits(read_u16(block + 2)));
            scales[i] = ov::float16(scale);
            bias[i] = ov::float16(min);
            unpack_block(block, weights + i * qk);
        });
    } else {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            const uint8_t * block = data + i * bytes_per_block;
            float scale = static_cast<float>(ov::float16::from_bits(read_u16(block)));
            float min = static_cast<float>(ov::float16::from_bits(read_u16(block + 2)));
            scales[i] = ov::float16(scale);
            zp[i] = (scale != 0.0f) ? (uint8_t) std::lround(-min / scale) : 0;
            unpack_block(block, weights + i * qk);
        });
    }
}

void extract_q8_0_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    const uint64_t weights_per_block = 32;
    const uint64_t bytes_per_block = 34;  // 2 bytes scale, 32x1 byte weights

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i8);

    if (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
            zp[i] = 128;
            for (size_t j = 0; j < weights_per_block; ++j) {
                uint8_t x = block_data[j + 2];
                x ^= 1 << 7;
                weights[i * weights_per_block + j] = x;
            }
        });
    } else {
        ov::parallel_for(scales_arr.get_size(), [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            scales[i] = ov::float16::from_bits(*(uint16_t *) block_data);
            memcpy(weights + i * weights_per_block, block_data + 2, weights_per_block);
        });
    }
}

void unpack_256_4(const uint8_t * data, uint8_t * dst) {
    std::fill_n(dst, 128, 0);

    for (size_t i = 0; i < 4; ++i) {
        for (int j = 0; j < 32; ++j) {
            uint8_t x = (data[i * 32 + j] & 0x0F);
            uint8_t y = (data[i * 32 + j] >> 4);
            if (j % 2 != 0) {
                x <<= 4;
                y <<= 4;
            }
            dst[i * 32 + j / 2] |= x;
            dst[i * 32 + 16 + j / 2] |= y;
        }
    }
}

void extract_q4_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias) {
    const uint64_t bytes_per_block = 2 + 2 + 12 + 128;
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    auto * zp_u4 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.data());
    auto * bias_f16 = use_bias ? zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_data = data + i * bytes_per_block;

        float scale_scales = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data)));
        float scale_mins = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 1)));

        uint8_t * qs1 = block_data + 4;

        float scale_vals[8];
        scale_vals[0] = scale_scales * static_cast<float>((*(qs1) & 0b111111));
        scale_vals[1] = scale_scales * static_cast<float>((*(qs1 + 1) & 0b111111));
        scale_vals[2] = scale_scales * static_cast<float>((*(qs1 + 2) & 0b111111));
        scale_vals[3] = scale_scales * static_cast<float>((*(qs1 + 3) & 0b111111));
        scale_vals[4] = scale_scales * static_cast<float>((*(qs1 + 8) & 0b00001111) | ((*(qs1) >> 6) << 4));
        scale_vals[5] = scale_scales * static_cast<float>((*(qs1 + 9) & 0b00001111) | ((*(qs1 + 1) >> 6) << 4));
        scale_vals[6] = scale_scales * static_cast<float>((*(qs1 + 10) & 0b00001111) | ((*(qs1 + 2) >> 6) << 4));
        scale_vals[7] = scale_scales * static_cast<float>((*(qs1 + 11) & 0b00001111) | ((*(qs1 + 3) >> 6) << 4));

        float min_vals[8];
        min_vals[0] = scale_mins * static_cast<float>((*(qs1 + 4) & 0b111111));
        min_vals[1] = scale_mins * static_cast<float>((*(qs1 + 5) & 0b111111));
        min_vals[2] = scale_mins * static_cast<float>((*(qs1 + 6) & 0b111111));
        min_vals[3] = scale_mins * static_cast<float>((*(qs1 + 7) & 0b111111));
        min_vals[4] = scale_mins * static_cast<float>((*(qs1 + 8) >> 4) | ((*(qs1 + 4) >> 6) << 4));
        min_vals[5] = scale_mins * static_cast<float>((*(qs1 + 9) >> 4) | ((*(qs1 + 5) >> 6) << 4));
        min_vals[6] = scale_mins * static_cast<float>((*(qs1 + 10) >> 4) | ((*(qs1 + 6) >> 6) << 4));
        min_vals[7] = scale_mins * static_cast<float>((*(qs1 + 11) >> 4) | ((*(qs1 + 7) >> 6) << 4));

        for (int j = 0; j < 8; j++) {
            scales[i * 8 + j] = ov::float16(scale_vals[j]);
            if (use_bias) {
                bias_f16[i * 8 + j] = ov::float16(-min_vals[j]);
            } else {
                uint8_t zp_val = (scale_vals[j] != 0.0f) ? (uint8_t) std::round(min_vals[j] / scale_vals[j]) : 0;
                size_t idx = i * 8 + j;
                if (idx % 2 == 0) {
                    zp_u4[idx / 2] = zp_val & 0x0F;
                } else {
                    zp_u4[idx / 2] |= (zp_val << 4);
                }
            }
        }
        unpack_256_4(block_data + 16, weights + i * 128);
    });
}

void extract_q6_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr) {
    const uint64_t bytes_per_block = 128 + 64 + 16 + 2;
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    bool is_symmetric = (weights_arr.get_element_type() == ov::element::i8);

    if (!is_symmetric) {
        auto * zp = static_cast<uint8_t *>(zp_arr.data());
        ov::parallel_for(n_super_block, [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            float scale_factor = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));
            for (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_factor * static_cast<float>(*((int8_t *) (block_data + 128 + 64 + j))));
                zp[j + i * 16] = 32;
            }
            uint8_t * ql = block_data;
            uint8_t * qh = block_data + 128;
            for (int64_t j = 0; j < 32; ++j) {
                weights[i * 256 + j] = (ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4);
                weights[i * 256 + j + 32] = (ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4);
                weights[i * 256 + j + 64] = (ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4);
                weights[i * 256 + j + 96] = (ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4);
                weights[i * 256 + j + 128] = (ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4);
                weights[i * 256 + j + 160] = (ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4);
                weights[i * 256 + j + 192] = (ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4);
                weights[i * 256 + j + 224] = (ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4);
            }
        });
    } else {
        ov::parallel_for(n_super_block, [&](size_t i) {
            uint8_t * block_data = data + i * bytes_per_block;
            float scale_factor = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 104)));
            for (size_t j = 0; j < 16; j++) {
                scales[j + i * 16] =
                    ov::float16(scale_factor * static_cast<float>(*((int8_t *) (block_data + 128 + 64 + j))));
            }
            uint8_t * ql = block_data;
            uint8_t * qh = block_data + 128;
            auto * signed_weights = reinterpret_cast<int8_t *>(weights);
            for (int64_t j = 0; j < 32; ++j) {
                signed_weights[i * 256 + j] = static_cast<int8_t>((ql[j] & 0xF) | (((qh[j] >> 0) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 32] =
                    static_cast<int8_t>((ql[32 + j] & 0xF) | (((qh[j] >> 2) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 64] = static_cast<int8_t>((ql[j] >> 4) | (((qh[j] >> 4) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 96] =
                    static_cast<int8_t>((ql[32 + j] >> 4) | (((qh[j] >> 6) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 128] =
                    static_cast<int8_t>((ql[64 + j] & 0xF) | (((qh[32 + j] >> 0) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 160] =
                    static_cast<int8_t>((ql[96 + j] & 0xF) | (((qh[32 + j] >> 2) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 192] =
                    static_cast<int8_t>((ql[64 + j] >> 4) | (((qh[32 + j] >> 4) & 3) << 4)) - 32;
                signed_weights[i * 256 + j + 224] =
                    static_cast<int8_t>((ql[96 + j] >> 4) | (((qh[32 + j] >> 6) & 3) << 4)) - 32;
            }
        });
    }
}

static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

void extract_q5_k_data(const ggml_tensor * tensor,
                       ov::Tensor & weights_arr,
                       ov::Tensor & scales_arr,
                       ov::Tensor & zp_arr,
                       bool use_bias) {
    const uint64_t bytes_per_block = 4 + 12 + 32 + 128;
    const uint64_t n_super_block = tensor->nb[3] / bytes_per_block;

    auto * data = static_cast<uint8_t *>(tensor->data);
    auto * weights = static_cast<uint8_t *>(weights_arr.data());
    auto * scales = scales_arr.data<ov::element_type_traits<ov::element::f16>::value_type>();

    auto * zp_u8 = use_bias ? nullptr : static_cast<uint8_t *>(zp_arr.data());
    auto * bias_f16 = use_bias ? zp_arr.data<ov::element_type_traits<ov::element::f16>::value_type>() : nullptr;

    ov::parallel_for(n_super_block, [&](size_t i) {
        uint8_t * block_data = data + i * bytes_per_block;

        const float d = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data)));
        const float min_factor = static_cast<float>(ov::float16::from_bits(*((uint16_t *) block_data + 1)));

        const uint8_t * scales_data = block_data + 4;
        const uint8_t * qh = block_data + 4 + 12;
        const uint8_t * ql = block_data + 4 + 12 + 32;

        int is = 0;
        uint8_t u1 = 1;
        uint8_t u2 = 2;

        for (int j = 0; j < 256; j += 64) {
            uint8_t sc;
            uint8_t m;

            get_scale_min_k4(is + 0, scales_data, &sc, &m);
            const float d1 = d * sc;
            const float m1 = min_factor * m;

            get_scale_min_k4(is + 1, scales_data, &sc, &m);
            const float d2 = d * sc;
            const float m2 = min_factor * m;

            scales[i * 8 + is] = ov::float16(d1);
            scales[i * 8 + is + 1] = ov::float16(d2);
            if (use_bias) {
                bias_f16[i * 8 + is] = ov::float16(-m1);
                bias_f16[i * 8 + is + 1] = ov::float16(-m2);
            } else {
                zp_u8[i * 8 + is] = (d1 != 0.0f) ? (uint8_t) std::round(m1 / d1) : 0;
                zp_u8[i * 8 + is + 1] = (d2 != 0.0f) ? (uint8_t) std::round(m2 / d2) : 0;
            }

            for (int l = 0; l < 32; ++l) {
                weights[i * 256 + j + l] = (ql[l] & 0xF) + ((qh[l] & u1) ? 16 : 0);
            }

            for (int l = 0; l < 32; ++l) {
                weights[i * 256 + j + l + 32] = (ql[l] >> 4) + ((qh[l] & u2) ? 16 : 0);
            }

            ql += 32;
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    });
}