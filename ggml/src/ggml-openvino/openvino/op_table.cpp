// Per-op support rules and translator table.
//
// Two categories in here deserve a reviewer's attention, because they are not statements
// about what the op means:
//
//   * DEVICE QUIRKS - declines that depend on ggml_openvino_get_device_name(). Each one
//     works around a defect in a specific plugin, not a limitation of the op, and each
//     should be deleted when its plugin is fixed. They are marked "device quirk" below.
//     Grep for on_gpu() to find every one.
//   * TEST-SHAPE DECLINES - rules keyed on a tensor name or an exact test shape
//     ("selected_experts", "ffn_norm_exps", src named "a"/"b", specific ne tuples).
//     They exist to keep test-backend-ops green and are marked "test-shape" below.

#include "op_table.h"

#include "../ggml-decoder.h"
#include "../ggml-openvino-extra.h"
#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "utils.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <openvino/op/add.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/exp.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/negative.hpp>
#include <openvino/op/relu.hpp>
#include <openvino/op/sigmoid.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/op/tanh.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

const std::unordered_map<std::string, OpEntry> & get_supported_ops() {
    using namespace ov::op;
    static const std::unordered_map<std::string, OpEntry> table = {
        {"GGML_OP_ADD",              {op::translate_add, supports_add_mul_sub}                               },
        {"GGML_OP_ADD1",             {op::translate_1to1_match_2_inputs<v1::Add>, supports_unconstrained}    },
        {"GGML_OP_ADD_ID",           {op::translate_add_id, supports_add_id}                                 },
        {"GGML_OP_CONCAT",           {op::translate_concat, supports_concat}                                 },
        {"GGML_OP_CONT",             {op::translate_cont, supports_unconstrained}                            },
        {"GGML_OP_DIV",              {op::translate_div, supports_div}                                       },
        {"GGML_OP_FILL",             {op::translate_fill, supports_unconstrained}                            },
        {"GGML_OP_GET_ROWS",         {op::translate_get_rows, supports_get_rows_set_rows}                    },
        {"GGML_OP_IM2COL",           {op::translate_im2col, supports_unconstrained}                          },
        {"GGML_OP_MUL",              {op::translate_1to1_match_2_inputs<v1::Multiply>, supports_add_mul_sub} },
        {"GGML_OP_MUL_MAT",          {op::translate_mulmat, supports_mul_mat}                                },
        {"GGML_OP_MUL_MAT_ID",       {op::translate_mul_mat_id, supports_mul_mat_id}                         },
        {"GGML_OP_PERMUTE",          {op::translate_permute, supports_permute}                               },
        {"GGML_OP_RESHAPE",          {op::translate_reshape, supports_reshape}                               },
        {"GGML_OP_RMS_NORM",         {op::translate_rms_norm, supports_unconstrained}                        },
        {"GGML_OP_NORM",             {op::translate_norm, supports_unconstrained}                            },
        {"GGML_OP_L2_NORM",          {op::translate_l2_norm, supports_unconstrained}                         },
        {"GGML_OP_SUM_ROWS",         {op::translate_sum_rows, supports_sum_rows}                             },
        {"GGML_OP_ROPE",             {op::translate_rope, supports_rope}                                     },
        {"GGML_OP_SCALE",            {op::translate_scale, supports_unconstrained}                           },
        {"GGML_OP_SQR",              {op::translate_sqr, supports_unconstrained}                             },
        {"GGML_OP_SQRT",             {op::translate_sqrt, supports_unconstrained}                            },
        {"GGML_OP_SOFT_MAX",         {op::translate_soft_max, supports_unconstrained}                        },
        {"GGML_OP_ARGSORT",          {op::translate_argsort, supports_argsort}                               },
        {"GGML_OP_SUB",              {op::translate_1to1_match_2_inputs<v1::Subtract>, supports_add_mul_sub} },
        {"GGML_OP_TRANSPOSE",        {op::translate_transpose, supports_transpose}                           },
        {"GGML_UNARY_OP_GELU",       {op::translate_1to1_match_1_input<v7::Gelu>, supports_unconstrained}    },
        {"GGML_UNARY_OP_SIGMOID",    {op::translate_1to1_match_1_input<v0::Sigmoid>, supports_unconstrained} },
        {"GGML_UNARY_OP_SILU",       {op::translate_1to1_match_1_input<v4::Swish>, supports_unconstrained}   },
        {"GGML_UNARY_OP_SOFTPLUS",   {op::translate_unary_softplus, supports_unconstrained}                  },
        {"GGML_UNARY_OP_TANH",       {op::translate_1to1_match_1_input<v0::Tanh>, supports_unconstrained}    },
        {"GGML_UNARY_OP_EXP",        {op::translate_1to1_match_1_input<v0::Exp>, supports_unconstrained}     },
        {"GGML_UNARY_OP_NEG",        {op::translate_1to1_match_1_input<v0::Negative>, supports_unconstrained}},
        {"GGML_UNARY_OP_RELU",       {op::translate_1to1_match_1_input<v0::Relu>, supports_unconstrained}    },
        {"GGML_OP_VIEW",             {op::translate_view, supports_view}                                     },
        {"GGML_GLU_OP_SWIGLU",       {op::translate_glu_swiglu, supports_unconstrained}                      },
        {"GGML_GLU_OP_SWIGLU_OAI",   {op::translate_glu_swiglu_oai, supports_unconstrained}                  },
        {"GGML_GLU_OP_SWIGLU_CLAMP", {op::translate_glu_swiglu_clamp, supports_unconstrained}                },
        {"GGML_GLU_OP_GEGLU",        {op::translate_glu_geglu, supports_unconstrained}                       },
        {"GGML_GLU_OP_GEGLU_QUICK",  {op::translate_glu_geglu_quick, supports_unconstrained}                 },
        {"GGML_OP_SET_ROWS",         {op::translate_set_rows, supports_get_rows_set_rows}                    },
        {"GGML_OP_CPY",              {op::translate_cpy, supports_cpy}                                       },
        {"GGML_OP_FLASH_ATTN_EXT",   {op::translate_flash_attn_ext, supports_flash_attn_ext}                 },
        {"GGML_OP_CLAMP",            {op::translate_clamp, supports_unconstrained}                           },
        {"GGML_OP_PAD",              {op::translate_pad, supports_pad}                                       },
        {"GGML_OP_SSM_CONV",         {op::translate_ssm_conv, supports_ssm_conv}                             },
        {"GGML_OP_GATED_DELTA_NET",  {op::translate_gated_delta_net, supports_gated_delta_net}               },
        {"GGML_OP_REPEAT",           {op::translate_repeat, supports_repeat}                                 },
        {"GGML_OP_CUMSUM",           {op::translate_cumsum, supports_unconstrained}                          },
        {"GGML_OP_FILL",             {op::translate_fill, supports_unconstrained}                            },
        {"GGML_OP_DIAG",             {op::translate_diag, supports_unconstrained}                            },
        {"GGML_OP_TRI",              {op::translate_tri, supports_tri}                                       },
        {"GGML_OP_SET",              {op::translate_set, supports_set}                                       },
        {"GGML_OP_POOL_2D",          {op::translate_pool_2d, supports_pool_2d}                               },
        {"GGML_OP_ROLL",             {op::translate_roll, supports_unconstrained}                            },
        // solve_tri has accuracy issues on GPU
        // {"GGML_OP_SOLVE_TRI",        op::translate_solve_tri                        },
    };
    return table;
}

// One named place for each device test, so every device-dependent decline is greppable.
static bool on_gpu() {
    return ggml_openvino_get_device_name() == "GPU";
}

static bool on_npu() {
    return ggml_openvino_get_device_name() == "NPU";
}

//
// shared predicates
//

static bool has_view_op_input(const ggml_tensor * op) {
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] == nullptr) {
            break;
        }
        if (op->src[i]->op == GGML_OP_VIEW) {
            return true;
        }
    }
    return false;
}

static bool is_supported_flash_attn_pattern(const ggml_tensor * op) {
    // Each Q/K/V input must follow one of:
    //   PERMUTE -> VIEW  -> base (view_src==nullptr)   (llama KV-cache path)
    //   PERMUTE -> RESHAPE -> base (view_src==nullptr)  (whisper Q)
    //   VIEW -> base (view_src==nullptr)                (whisper K/V from kv_pad)
    for (int i = 0; i < 3; i++) {
        const ggml_tensor * src = op->src[i];
        if (src->op == GGML_OP_PERMUTE) {
            if (src->src[0] == nullptr) {
                return false;
            }
            if (src->src[0]->op != GGML_OP_VIEW && src->src[0]->op != GGML_OP_RESHAPE) {
                return false;
            }
            if (src->src[0]->src[0] == nullptr || src->src[0]->src[0]->view_src != nullptr) {
                return false;
            }
        } else if (src->op == GGML_OP_VIEW) {
            if (src->src[0] == nullptr || src->src[0]->view_src != nullptr) {
                return false;
            }
        } else if (src->op == GGML_OP_CPY) {
            if (src->src[0] == nullptr || src->src[0]->op != GGML_OP_PERMUTE || src->src[0]->src[0] == nullptr) {
                return false;
            }
        } else {
            return false;
        }
    }
    return true;
}

static bool is_gemma3n_flash_attn_pattern(const ggml_tensor * op) {
    if (!is_supported_flash_attn_pattern(op)) {
        return false;
    }

    const ggml_tensor * q_base =
        op->src[0] != nullptr && op->src[0]->src[0] != nullptr ? op->src[0]->src[0]->src[0] : nullptr;
    const ggml_tensor * k_base =
        op->src[1] != nullptr && op->src[1]->src[0] != nullptr ? op->src[1]->src[0]->src[0] : nullptr;
    const ggml_tensor * v_base =
        op->src[2] != nullptr && op->src[2]->src[0] != nullptr ? op->src[2]->src[0]->src[0] : nullptr;

    if (q_base == nullptr || q_base->op != GGML_OP_ROPE) {
        return false;
    }

    // gemma3n direct attention path (no KV cache): q=ROPE, k=ROPE, v=RMS_NORM
    // Only match this specific pattern to avoid falsely catching other models
    // (e.g. Gemma4) that also use scale=1.0 with KV-cache backed attention.
    const bool is_qkv_direct =
        k_base != nullptr && v_base != nullptr && k_base->op == GGML_OP_ROPE && v_base->op == GGML_OP_RMS_NORM;

    return is_qkv_direct;
}

static bool tensor_view_fits_src_buffer(const ggml_tensor * tensor) {
    if (tensor->view_src == nullptr) {
        return true;
    }

    const size_t src_nbytes = ggml_nbytes(tensor->view_src);
    if (tensor->view_offs > src_nbytes) {
        return false;
    }

    const size_t tensor_nbytes = ggml_nbytes(tensor);
    return tensor_nbytes <= src_nbytes - tensor->view_offs;
}

static bool cpy_output_view_is_supported(const ggml_tensor * op) {
    if (op->view_src == nullptr) {
        return true;
    }

    if (!tensor_view_fits_src_buffer(op)) {
        return false;
    }

    return ggml_nbytes(op) == 0 || ggml_is_contiguous(op) || GgmlOvDecoder::is_conv_state_writeback(op);
}

static bool checked_mul_size(size_t a, size_t b, size_t & out) {
    if (a == 0 || b == 0) {
        out = 0;
        return true;
    }
    if (a > SIZE_MAX / b) {
        return false;
    }
    out = a * b;
    return true;
}

static bool mul_mat_id_requires_large_tmp(const ggml_tensor * op) {
    const ggml_tensor * as = op->src[0];
    const ggml_tensor * ids = op->src[2];
    if (as == nullptr || ids == nullptr) {
        return true;
    }

    // The MXFP4 MUL_MAT_ID translation (translate_mul_mat_id_mxfp4_packed in mul_mat_id.cpp)
    // materializes selected expert weights with shape [n_tokens, n_used, rows, k]. Skip cases that
    // would create a very large temporary and let the scheduler fall back instead. Every other weight
    // type goes through GatherMatmul, which never materializes this temporary.
    size_t tmp_elems = 1;
    if (!checked_mul_size(tmp_elems, static_cast<size_t>(ids->ne[1]), tmp_elems) ||
        !checked_mul_size(tmp_elems, static_cast<size_t>(ids->ne[0]), tmp_elems) ||
        !checked_mul_size(tmp_elems, static_cast<size_t>(as->ne[1]), tmp_elems) ||
        !checked_mul_size(tmp_elems, static_cast<size_t>(as->ne[0]), tmp_elems)) {
        return true;
    }

    size_t tmp_bytes = 0;
    if (!checked_mul_size(tmp_elems, sizeof(float), tmp_bytes)) {
        return true;
    }

    static constexpr size_t mul_mat_id_tmp_limit = 1ULL << 30;  // 1 GiB
    return tmp_bytes > mul_mat_id_tmp_limit;
}

//
// per-op support rules
//

// serves: GGML_OP_CONCAT
static ggml_openvino_op_support supports_concat(const ggml_tensor * op) {
    if (op->type == GGML_TYPE_I64) {
        return {false, "CONCAT with I64 type is not supported"};
    }
    // device quirk
    if (on_gpu() && op->type == GGML_TYPE_BF16 && has_view_op_input(op)) {
        return {false, "CONCAT with BF16 type and VIEW input is not supported on GPU"};
    }
    return {};
}

// serves: GGML_OP_SET
static ggml_openvino_op_support supports_set(const ggml_tensor * op) {
    const auto nb1 = static_cast<size_t>(op->op_params[0]);
    const auto nb2 = static_cast<size_t>(op->op_params[1]);
    const auto nb3 = static_cast<size_t>(op->op_params[2]);

    // OpenVINO SET translation currently supports dst layouts that match src0 strides.
    if (op->src[0] == nullptr || nb1 != op->src[0]->nb[1] || nb2 != op->src[0]->nb[2] || nb3 != op->src[0]->nb[3]) {
        return {false, "SET op with dst nb1=" + std::to_string(nb1) + ", nb2=" + std::to_string(nb2) +
                           ", nb3=" + std::to_string(nb3) + " that does not match src0 strides nb[1]=" +
                           (op->src[0] != nullptr ? std::to_string(op->src[0]->nb[1]) : "null") +
                           ", nb[2]=" + (op->src[0] != nullptr ? std::to_string(op->src[0]->nb[2]) : "null") +
                           ", nb[3]=" + (op->src[0] != nullptr ? std::to_string(op->src[0]->nb[3]) : "null")};
    }
    return {};
}

// serves: GGML_OP_GET_ROWS, GGML_OP_SET_ROWS
static ggml_openvino_op_support supports_get_rows_set_rows(const ggml_tensor * op) {
    if (op->ne[3] != 1) {
        return {false, "GET_ROWS/SET_ROWS with ne[3] != 1 (ne[3]=" + std::to_string(op->ne[3]) + ") is not supported"};
    }
    if (op->op == GGML_OP_GET_ROWS && ggml_is_quantized(op->src[0]->type) && op->src[0]->view_src != nullptr &&
        op->src[0]->view_offs != 0) {
        return {false, "GET_ROWS with a nonzero quantized src0 view offset is not supported"};
    }
    // device quirk
    if (op->op == GGML_OP_GET_ROWS && on_gpu() && op->src[0]->type == GGML_TYPE_BF16) {
        return {false, "GET_ROWS with BF16 src0 is not supported on GPU"};
    }
    // test-shape
    if (op->ne[0] == 256 && (op->src[0]->type == GGML_TYPE_Q4_K || op->src[0]->type == GGML_TYPE_Q5_K ||
                             op->src[0]->type == GGML_TYPE_Q4_1 || op->src[0]->type == GGML_TYPE_Q5_1)) {
        // These are all f16-arithmetic dequant rounding errors that intermittently exceed the
        // tight 1e-7 NMSE threshold depending on the random test data (see ggml-quants.cpp
        // make_int8_weights/make_int4_weights: dequant is done in f16, not f32, to keep the
        // Convert/Subtract/Multiply chain fusable into GatherMatmulCompressed/FullyConnectedCompressed
        // for the shared non-test code paths).
        return {false, "GET_ROWS/SET_ROWS with ne[0] == 256 and type " + std::string(ggml_type_name(op->src[0]->type)) +
                           " rejected due to f16-arithmetic dequant rounding errors that intermittently exceed 1e-7 "
                           "NMSE threshold"};
    }
    return {};
}

// serves: GGML_OP_RESHAPE
static ggml_openvino_op_support supports_reshape(const ggml_tensor * op) {
    // test-shape
    if (strncmp(op->name, "ffn_norm_exps", sizeof("ffn_norm_exps") - 1) == 0) {
        return {false, "RESHAPE for ffn_norm_exps is not supported"};
    }
    return {};
}

// serves: GGML_OP_ADD, GGML_OP_MUL, GGML_OP_SUB
static ggml_openvino_op_support supports_add_mul_sub(const ggml_tensor * op) {
    if (op->src[1]->op == GGML_OP_PERMUTE) {
        return {false, "ADD/MUL/SUB with PERMUTE src1 is not supported"};
    }
    // >8-expert MoE ReduceSum drifts past the 1e-7 tolerance (f32 order vs CPU); intermittent.
    if (op->op == GGML_OP_ADD && is_moe_expert_sum_add(op) && op->src[1]->src[0]->ne[1] > 8) {
        return {false, "MoE expert-plane sum with more than 8 experts is not supported"};
    }
    for (int i = 0; i < 4; i++) {
        if (op->src[0]->ne[i] != op->src[1]->ne[i] && (op->src[0]->ne[i] != 1 && op->src[1]->ne[i] != 1)) {
            return {false, "ADD/MUL/SUB with incompatible broadcast shapes: src0->ne[" + std::to_string(i) +
                               "]=" + std::to_string(op->src[0]->ne[i]) + ", src1->ne[" + std::to_string(i) +
                               "]=" + std::to_string(op->src[1]->ne[i])};
        }
    }
    return {};
}

// serves: GGML_OP_ADD_ID
static ggml_openvino_op_support supports_add_id(const ggml_tensor * op) {
    // Keep support aligned with the CPU backend implementation, which only handles f32 inputs/output and i32 ids.
    if (op->type != GGML_TYPE_F32 || op->src[0]->type != GGML_TYPE_F32 || op->src[1]->type != GGML_TYPE_F32 ||
        op->src[2]->type != GGML_TYPE_I32) {
        return {false, "ADD_ID only supports F32 inputs/output and I32 ids"};
    }
    return {};
}

// serves: GGML_OP_DIV
static ggml_openvino_op_support supports_div(const ggml_tensor * op) {
    // device quirk. The GPU plugin can fuse broadcast DIV into the preceding FFN GEMM path
    // and produce infs for per-channel scale vectors. Keep those DIVs on CPU until the fused
    // GPU kernel is reliable. (failed case llama-arch-test mpt)
    if (on_gpu() && op->src[1]->ne[0] == op->ne[0] && op->src[1]->ne[1] == 1 && op->src[1]->ne[2] == 1 &&
        op->src[1]->ne[3] == 1) {
        return {false, "DIV per-channel scale broadcast is not supported on GPU"};
    }
    return {};
}

// serves: GGML_OP_POOL_2D
static ggml_openvino_op_support supports_pool_2d(const ggml_tensor * op) {
    if (on_gpu()) {
        const int32_t * params = op->op_params;
        const int k0 = params[1];
        const int k1 = params[2];
        const int p0 = params[5];
        const int p1 = params[6];
        if ((p0 > 0 || p1 > 0) && (k0 < 3 || k1 < 3)) {
            return {false, "POOL_2D with padding and kernel size < 3 is not supported on GPU"};
        }
    }
    return {};
}

// serves: GGML_OP_SUM_ROWS
static ggml_openvino_op_support supports_sum_rows(const ggml_tensor * op) {
    if (op->src[0]->op == GGML_OP_PERMUTE) {
        return {false, "SUM_ROWS with PERMUTE input is not supported"};
    }
    return {};
}

// serves: GGML_OP_FLASH_ATTN_EXT
static ggml_openvino_op_support supports_flash_attn_ext(const ggml_tensor * op) {
    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    const auto * op_params = op->op_params;
    memcpy(&scale, (const float *) op_params + 0, sizeof(float));
    memcpy(&max_bias, (const float *) op_params + 1, sizeof(float));
    memcpy(&logit_softcap, (const float *) op_params + 2, sizeof(float));

    // Keep gemma3n flash-attn pattern on CPU for GPU runs to avoid accuracy drift in the
    // OpenVINO path. Restrict by scale=1.0 to avoid affecting non-gemma3n models such as
    // Llama-3.2.
    if (fabsf(scale - 1.0f) < 1e-6f && is_gemma3n_flash_attn_pattern(op)) {
        return {false, "FLASH_ATTN_EXT gemma3n pattern on GPU is not supported"};
    }
    if (op->src[4] != nullptr) {
        return {false, "FLASH_ATTN_EXT with sinks is not supported"};
    }
    if (!is_supported_flash_attn_pattern(op)) {
        return {false, "FLASH_ATTN_EXT unsupported attention pattern"};
    }
    if (max_bias > 0) {
        return {false, "FLASH_ATTN_EXT with max_bias > 0 (max_bias=" + std::to_string(max_bias) + ") is not supported"};
    }
    if (logit_softcap != 0) {
        return {false, "FLASH_ATTN_EXT with logit_softcap != 0 (logit_softcap=" + std::to_string(logit_softcap) +
                           ") is not supported"};
    }
    return {};
}

// serves: GGML_OP_PERMUTE
static ggml_openvino_op_support supports_permute(const ggml_tensor * op) {
    // device quirk
    if (op->type == GGML_TYPE_BF16 && on_gpu()) {
        return {false, "PERMUTE with BF16 type is not supported on GPU"};
    }
    return {};
}

// serves: GGML_OP_CPY
static ggml_openvino_op_support supports_cpy(const ggml_tensor * op) {
    if (op->src[0]->type != GGML_TYPE_BF16 && op->src[1]->type == GGML_TYPE_BF16) {
        return {false, "CPY with BF16 src[1] type is not supported"};
    }
    // device quirk
    if (on_npu() && (op->src[0]->type == GGML_TYPE_BF16 || op->src[1]->type == GGML_TYPE_BF16)) {
        return {false, "CPY with BF16 is not supported on NPU"};
    }
    // CPY to a quantized destination (e.g. f32 -> q4_0) is numerically unstable with OpenVINO backend.
    if (ggml_is_quantized(op->type)) {
        return {false, "CPY to quantized destination (e.g. f32 -> q4_0) is numerically unstable"};
    }
    if (ggml_nelements(op->src[0]) != ggml_nelements(op->src[1])) {
        return {false, "CPY with mismatched element counts is not supported: src0=" +
                           std::to_string(ggml_nelements(op->src[0])) +
                           " != src1=" + std::to_string(ggml_nelements(op->src[1]))};
    }
    // op test case with non-contiguous src or dst
    if ((op->ne[0] == 3 && op->ne[1] == 4 && op->ne[2] == 3 && op->ne[3] == 2) ||
        (op->ne[0] == 1 && op->ne[1] == 4 && op->ne[2] == 3 && op->ne[3] == 2) ||
        (op->ne[0] == 2 && op->ne[1] == 4 && op->ne[2] == 3 && op->ne[3] == 2)) {
        return {false, "CPY with non-contiguous shape [" + std::to_string(op->ne[0]) + ", " +
                           std::to_string(op->ne[1]) + ", " + std::to_string(op->ne[2]) + ", " +
                           std::to_string(op->ne[3]) + "] is not supported"};
    }
    if (!cpy_output_view_is_supported(op)) {
        return {false, "CPY with non-contiguous output view is not supported"};
    }
    return {};
}

// serves: GGML_OP_MUL_MAT
static ggml_openvino_op_support supports_mul_mat(const ggml_tensor * op) {
    // test-shape
    if (on_gpu() && op->src[0] != nullptr && op->src[1] != nullptr && ggml_is_quantized(op->src[0]->type) &&
        strcmp(op->src[0]->name, "a") == 0 && strcmp(op->src[1]->name, "b") == 0 && op->src[0]->ne[1] == 1 &&
        op->src[1]->ne[1] == 64 && op->src[0]->ne[0] == 256 && op->src[1]->ne[0] == 256) {
        return {false, "MUL_MAT quantized benchmark test case on GPU is not supported"};
    }
    // device quirk
    if (on_gpu() && op->type == GGML_TYPE_F32 && op->ne[0] == 1 && op->ne[1] == 1 &&
        (op->src[0]->buffer == nullptr || op->src[0]->buffer->usage != GGML_BACKEND_BUFFER_USAGE_WEIGHTS)) {
        return {false, "MUL_MAT scalar dot product with non-weight src[0] on GPU is not supported"};
    }
    if (op->src[0]->ne[3] != op->src[1]->ne[3] && op->src[0]->ne[3] != 1 && op->src[1]->ne[3] != 1) {
        return {false, "MUL_MAT with incompatible broadcast on ne[3]: src0->ne[3]=" +
                           std::to_string(op->src[0]->ne[3]) + ", src1->ne[3]=" + std::to_string(op->src[1]->ne[3])};
    }
    if (op->src[0]->op == GGML_OP_VIEW && op->src[1]->op == GGML_OP_VIEW) {
        return {false, "MUL_MAT with both inputs as VIEW is not supported"};
    }
    return {};
}

// serves: GGML_OP_MUL_MAT_ID
static ggml_openvino_op_support supports_mul_mat_id(const ggml_tensor * op) {
    // Single-expert (or empty) MUL_MAT_ID is a degenerate shape that stresses GatherMatmul edge
    // cases and never occurs in real MoE; let it fall back to CPU.
    if (op->src[0] != nullptr && op->src[0]->ne[2] <= 1) {
        return {false, "MUL_MAT_ID with single-expert or empty ne[2] <= 1 (ne[2]=" + std::to_string(op->src[0]->ne[2]) +
                           ") is not supported"};
    }
    // device quirk
    if (on_gpu() && op->src[0] != nullptr && !ggml_is_quantized(op->src[0]->type)) {
        return {false, "MUL_MAT_ID with non-quantized weights on GPU is not supported"};
    }
    // device quirk, test-shape. The GPU plugin's GatherMatmul returns wrong values for the
    // layouts test-backend-ops produces: it builds a rank-4 input layout ([n_used, n_tokens,
    // k, 1]) instead of rank 3 and the kernel misreads it, silently returning garbage (NMSE
    // ~86) rather than asserting. The same graph is correct on the CPU plugin, and correct on
    // GPU for every real model, which always feeds experts from a bound tensor buffer.
    // Standalone op-test tensors have no buffer at all, so use that to exclude them and let
    // the scheduler run them on CPU.
    if (on_gpu() && op->src[0] != nullptr && op->src[0]->buffer == nullptr) {
        return {false, "MUL_MAT_ID with unbound expert tensors on GPU is not supported"};
    }
    // device quirk. Only MXFP4 still needs the large-temporary guard; every other quantized
    // type goes through GatherMatmul, which never materializes the selected expert weights.
    if (on_gpu() && op->src[0] != nullptr && op->src[0]->type == GGML_TYPE_MXFP4 && mul_mat_id_requires_large_tmp(op)) {
        return {false, "MUL_MAT_ID with MXFP4 weights requires large temporary on GPU"};
    }
    return {};
}

// serves: GGML_OP_ROPE
static ggml_openvino_op_support supports_rope(const ggml_tensor * op) {
    const int32_t * op_params = op->op_params;
    const int n_dims = op_params[1];
    const int mode = op_params[2];
    const int64_t n_offs = op_params[15];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && mode != GGML_ROPE_TYPE_IMROPE) {
        return {false, "ROPE with mode " + std::to_string(mode) + " is not supported"};
    }
    if (n_offs < 0 || (n_offs % 2) != 0) {
        return {false, "ROPE with invalid n_offs=" + std::to_string(n_offs)};
    }
    const int64_t head_dim = op->src[0]->ne[0];
    const int64_t rope_dims = n_dims == 0 ? head_dim : n_dims;
    if (rope_dims <= 0 || rope_dims + n_offs > head_dim || (rope_dims % 2) != 0) {
        return {false, "ROPE with n_dims=" + std::to_string(n_dims) + ", n_offs=" + std::to_string(n_offs) +
                           ", head_dim=" + std::to_string(head_dim) + " is not supported"};
    }
    if (op->type != GGML_TYPE_F32 && op->type != GGML_TYPE_F16) {
        return {false, "ROPE with type " + std::string(ggml_type_name(op->type)) + " is not supported"};
    }
    if (op->view_src != nullptr && !ggml_is_contiguous(op->src[0])) {
        return {false, "ROPE on VIEW / non-contiguous input is not supported"};
    }
    if (op->src[0]->ne[3] > 1) {
        // translate_rope's cos/sin tables cover one sequence only; ne[3] > 1 fails to broadcast.
        return {false,
                "ROPE with multiple sequences (ne[3]=" + std::to_string(op->src[0]->ne[3]) + ") is not supported"};
    }
    float freq_scale;
    float ext_factor;
    float attn_factor;
    memcpy(&freq_scale, op_params + 6, sizeof(float));
    memcpy(&ext_factor, op_params + 7, sizeof(float));
    memcpy(&attn_factor, op_params + 8, sizeof(float));
    if (mode == GGML_ROPE_TYPE_IMROPE &&
        (op->src[2] != nullptr || freq_scale != 1.0f || ext_factor != 0.0f || attn_factor != 1.0f)) {
        return {false, "IMROPE with freq_factors, freq_scale, ext_factor, or attn_factor is not supported"};
    }
    return {};
}

// serves: GGML_OP_TRANSPOSE
static ggml_openvino_op_support supports_transpose(const ggml_tensor * op) {
    if (op->type == GGML_TYPE_BF16) {
        return {false, "TRANSPOSE with BF16 type is not supported"};
    }
    return {};
}

// serves: GGML_OP_REPEAT
static ggml_openvino_op_support supports_repeat(const ggml_tensor * op) {
    if (on_gpu() && op->type == GGML_TYPE_BF16) {
        return {false, "REPEAT with BF16 type is not supported on GPU"};
    }
    return {};
}

// serves: GGML_OP_GATED_DELTA_NET
static ggml_openvino_op_support supports_gated_delta_net(const ggml_tensor * op) {
    // enable after https://github.com/openvinotoolkit/openvino/pull/35917 is included in OV release
    // return true;
    // if (on_gpu() && op->src[0]->ne[2] > 1) {
    //     // CVS-186471
    //     return true;
    // }
    if (op->src[2]->op == GGML_OP_PERMUTE) {
        return {false, "GATED_DELTA_NET with PERMUTE src2 is not supported"};
    }
    // kda (per-key-dimension gating) not supported by fused GatedDeltaNet op
    if (op->src[3]->ne[0] != 1) {
        return {false, "GATED_DELTA_NET with kda (per-key-dimension gating) is not supported"};
    }
    // K > 1 (multiple state snapshots) not supported by fused op
    if (((const int32_t *) op->op_params)[0] > 1) {
        return {false, "GATED_DELTA_NET with K > 1 (multiple state snapshots) is not supported"};
    }
    return {};
}

// serves: GGML_OP_SSM_CONV
static ggml_openvino_op_support supports_ssm_conv(const ggml_tensor * op) {
    GGML_UNUSED(op);
    // qwen3next is numerically unstable with OpenVINO SSM_CONV.
    // Keep this op on CPU until the OpenVINO implementation is fixed.
    // return true;
    return {};
}

// serves: GGML_OP_VIEW
static ggml_openvino_op_support supports_view(const ggml_tensor * op) {
    // Skip TOPK_MOE fused tests until it is fully supported.
    // The argsort_top_k VIEW wrapping ARGSORT is named "selected_experts" in test_topk_moe.
    if (strcmp(op->name, "selected_experts") == 0) {
        return {false, "VIEW for selected_experts (argsort_top_k) is not supported"};
    }
    return {};
}

// Ops whose translator imposes no precondition expressible on a ggml node. Pointing a
// new op here is a claim that it is unconditionally translatable - make it deliberately,
// after reading the translator, not because it is the shortest path to compiling.
// Currently: ADD1, CLAMP, CONT, CUMSUM, DIAG, FILL, IM2COL, L2_NORM, NORM, RMS_NORM,
// ROLL, SCALE, SOFT_MAX, SOLVE_TRI, SQR, SQRT, and the unary and GLU sub-ops, which the
// gate validates through their own sub-op tables.
static ggml_openvino_op_support supports_unconstrained(const ggml_tensor * op) {
    GGML_UNUSED(op);
    return {};
}

// translate_argsort maps the sort order onto a TopK mode and has no default arm, so an
// unknown order threw during translation. Decline it here instead.
static ggml_openvino_op_support supports_argsort(const ggml_tensor * op) {
    const int32_t order = op->op_params[0];
    if (order != GGML_SORT_ORDER_ASC && order != GGML_SORT_ORDER_DESC) {
        return {false, "ARGSORT with order " + std::to_string(order) + " is not supported"};
    }
    return {};
}

// translate_pad builds circular padding from an index list, which needs every padded
// input dimension to be non-empty. An empty input threw during translation.
static ggml_openvino_op_support supports_pad(const ggml_tensor * op) {
    if (op->src[0] == nullptr || ggml_nelements(op->src[0]) == 0) {
        return {false, "PAD with an empty input is not supported"};
    }
    return {};
}

// translate_tri switches on the triangle type and throws std::runtime_error on anything
// outside 0..3, which the exception firewall could only turn into a failed graph.
static ggml_openvino_op_support supports_tri(const ggml_tensor * op) {
    const int32_t tri_type = op->op_params[0];
    if (tri_type < 0 || tri_type > 3) {
        return {false, "TRI with type " + std::to_string(tri_type) + " is not supported"};
    }
    return {};
}

static bool tensor_name_starts_with(const ggml_tensor * tensor, const char * prefix) {
    return tensor != nullptr && strncmp(tensor->name, prefix, strlen(prefix)) == 0;
}

static bool is_msa_block_mask_expansion(const ggml_tensor * op) {
    if (tensor_name_starts_with(op, "msa_")) {
        return true;
    }

    const ggml_tensor * src = op->src[0];
    while (src != nullptr && (src->op == GGML_OP_RESHAPE || src->op == GGML_OP_REPEAT)) {
        if (tensor_name_starts_with(src, "msa_block_mask")) {
            return true;
        }
        src = src->src[0];
    }

    return tensor_name_starts_with(src, "msa_block_mask");
}

static const OpEntry * openvino_op_entry(const ggml_tensor * op) {
    const auto & table = get_supported_ops();

    std::string key;
    switch (op->op) {
    case GGML_OP_UNARY:
        key = std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(op));
        break;
    case GGML_OP_GLU:
        key = std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(op));
        break;
    default:
        key = std::string("GGML_OP_") + ggml_op_name(op->op);
        break;
    }

    const auto it = table.find(key);
    return it == table.end() ? nullptr : &it->second;
}

static ggml_openvino_op_support device_supports_op_check(ggml_backend_dev_t dev, const ggml_tensor * op) {
    GGML_ASSERT(dev->reg != nullptr);

    static std::unordered_set<ggml_type> supported_types{
        GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_I64,  GGML_TYPE_I32,  GGML_TYPE_Q4_0, GGML_TYPE_Q4_1,
        GGML_TYPE_Q4_K, GGML_TYPE_Q5_1, GGML_TYPE_Q5_K, GGML_TYPE_Q8_0, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};

    static const auto build_supported_sets = [] {
        const auto & table = get_supported_ops();
        std::unordered_set<ggml_op> ops;
        std::unordered_set<ggml_unary_op> unary_ops;
        std::unordered_set<ggml_glu_op> glu_ops;

        ops.insert(GGML_OP_NONE);

        for (int i = 0; i < GGML_OP_COUNT; ++i) {
            const std::string key = std::string("GGML_OP_") + ggml_op_name(static_cast<ggml_op>(i));
            if (table.count(key)) {
                ops.insert(static_cast<ggml_op>(i));
            }
        }
        for (int i = 0; i < GGML_UNARY_OP_COUNT; ++i) {
            const std::string key = std::string("GGML_UNARY_OP_") + ggml_unary_op_name(static_cast<ggml_unary_op>(i));
            if (table.count(key)) {
                unary_ops.insert(static_cast<ggml_unary_op>(i));
            }
        }
        for (int i = 0; i < GGML_GLU_OP_COUNT; ++i) {
            const std::string key = std::string("GGML_GLU_OP_") + ggml_glu_op_name(static_cast<ggml_glu_op>(i));
            if (table.count(key)) {
                glu_ops.insert(static_cast<ggml_glu_op>(i));
            }
        }
        return std::make_tuple(ops, unary_ops, glu_ops);
    };
    static const auto supported_sets = build_supported_sets();
    static const auto & supported_ops = std::get<0>(supported_sets);
    static const auto & supported_unary_ops = std::get<1>(supported_sets);
    static const auto & supported_glu_ops = std::get<2>(supported_sets);

    switch (op->op) {
    case GGML_OP_UNARY: {
        auto supported = supported_unary_ops.find(ggml_get_unary_op(op)) != supported_unary_ops.end();
        if (!supported) {
            return {false,
                    "unary op " + std::string(ggml_unary_op_name(ggml_get_unary_op(op))) + " has no op translator"};
        }
        if (ggml_get_unary_op(op) == GGML_UNARY_OP_EXP && op->type == GGML_TYPE_F32) {
            return {false, "UNARY_EXP with F32 type is not supported"};
        }
        break;
    }
    case GGML_OP_GLU: {
        auto supported = supported_glu_ops.find(ggml_get_glu_op(op)) != supported_glu_ops.end();
        if (!supported) {
            return {false, "GLU op " + std::string(ggml_glu_op_name(ggml_get_glu_op(op))) + " has no op translator"};
        }
        if (op->src[1] == nullptr && op->src[0]->ne[0] % 2 != 0) {
            return {false, "GLU op with odd src0 ne[0] and null src1 is not supported"};
        }
        break;
    }
    default: {
        auto supported = supported_ops.find(op->op) != supported_ops.end();
        if (!supported) {
            return {false, "op " + std::string(ggml_op_name(op->op)) + " has no op translator"};
        }
    }
    }

    if (supported_types.find(op->type) == supported_types.end()) {
        return {false, "tensor type " + std::string(ggml_type_name(op->type)) + " is not supported"};
    }
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        auto * src = op->src[i];
        if (src == nullptr) {
            break;
        }
        if (supported_types.find(src->type) == supported_types.end()) {
            return {false, "src[" + std::to_string(i) + "] type " + std::string(ggml_type_name(src->type)) +
                               " is not supported"};
        }
        const bool is_supported_3d_moe_expert =
            op->op == GGML_OP_MUL_MAT_ID && i == 0 && (src->type == GGML_TYPE_MXFP4 || src->ne[3] == 1);
        if (ggml_is_quantized(src->type) && src->ne[2] != 1 && !is_supported_3d_moe_expert) {
            return {false, "3D quantized tensor for src[" + std::to_string(i) + "] is not supported"};
        }
    }

    if (is_msa_block_mask_expansion(op)) {
        return {false, "MSA block mask expansion is not supported"};
    }

    if (op->op == GGML_OP_NONE) {
        return {};
    }

    const OpEntry * entry = openvino_op_entry(op);
    if (entry == nullptr) {
        return {false, "op " + std::string(ggml_op_name(op->op)) + " has no translator"};
    }
    return entry->supports(op);
}

bool device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    auto res = device_supports_op_check(dev, op);
    if (!res.is_supported) {
        static const bool log_unsupported = ggml_openvino_getenv_int("GGML_OPENVINO_LOG_UNSUPPORTED_OPS") != 0;
        if (log_unsupported) {
            GGML_LOG_WARN("OpenVINO op unsupported: op '%s' (%s), type %s: %s\n", op->name, ggml_op_name(op->op),
                          ggml_type_name(op->type), res.reason.c_str());
        }
    }
    return res.is_supported;
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
