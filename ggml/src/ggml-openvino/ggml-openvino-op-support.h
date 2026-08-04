#pragma once

#include "ggml-backend.h"
#include "ggml.h"

bool ggml_openvino_device_supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op);

// Developer note: OpenVINO op support policy (human-readable summary)

// Common global gates:
// - Type gate: op/src types must be one of F32/F16/BF16/I64/I32/Q4_0/Q4_1/Q4_K/Q5_1/Q5_K/Q8_0/Q6_K/MXFP4.
// - Quantized rank gate: 3D quantized src is generally rejected except MUL_MAT_ID expert weights.
// - MSA mask expansion gate: tensors in the msa_block_mask expansion chain are forced unsupported.
//
// Important per-op policy limits:
/*
|OP Name                | Limitation|
|GGML_OP_ADD            | PERMUTE rhs is unsupported; only broadcast-compatible shapes are accepted|
|GGML_OP_ADD1           | none beyond global gates|
|GGML_OP_ADD_ID         | requires output/src0/src1 F32 and src2 I32|
|GGML_OP_CONCAT         | I64 is unsupported; GPU BF16 + VIEW input is unsupported|
|GGML_OP_CONT           | none beyond global gates|
|GGML_OP_DIV            | GPU per-channel divisor shape [ne0,1,1,1] is forced to CPU|
|GGML_OP_FILL           | none beyond global gates (registered twice in op_table; same translator)|
|GGML_OP_GET_ROWS       | ne[3] must be 1; GPU BF16 source is blocked; selected quantized 256-lane test shapes are blocked|
|GGML_OP_IM2COL         | none beyond global gates|
|GGML_OP_MUL            | PERMUTE rhs is unsupported; only broadcast-compatible shapes are accepted|
|GGML_OP_MUL_MAT        | selected GPU quantized test shape blocked; ne[3] compatibility required; VIEW+VIEW blocked|
|GGML_OP_MUL_MAT_ID     | single-expert cases blocked; GPU BF16 blocked; large temporary GPU cases blocked|
|GGML_OP_PERMUTE        | selected BF16 path is unsupported|
|GGML_OP_RESHAPE        | names prefixed with ffn_norm_exps are blocked|
|GGML_OP_RMS_NORM       | none beyond global gates|
|GGML_OP_NORM           | none beyond global gates|
|GGML_OP_L2_NORM        | none beyond global gates|
|GGML_OP_SUM_ROWS       | PERMUTE input is unsupported|
|GGML_OP_ROPE           | mode limited to NORMAL/NEOX/IMROPE; rope dims must be valid/even; output type F32/F16 only; IMROPE extra factors constrained|
|GGML_OP_SCALE          | none beyond global gates|
|GGML_OP_SQR            | none beyond global gates|
|GGML_OP_SQRT           | none beyond global gates|
|GGML_OP_SOFT_MAX       | none beyond global gates|
|GGML_OP_ARGSORT        | none beyond global gates|
|GGML_OP_SUB            | PERMUTE rhs is unsupported; only broadcast-compatible shapes are accepted|
|GGML_OP_TRANSPOSE      | selected BF16 path is unsupported|
|GGML_UNARY_OP_GELU     | none beyond global gates|
|GGML_UNARY_OP_SIGMOID  | none beyond global gates (registered twice in op_table; same translator)|
|GGML_UNARY_OP_SILU     | none beyond global gates|
|GGML_UNARY_OP_SOFTPLUS | none beyond global gates|
|GGML_UNARY_OP_TANH     | none beyond global gates|
|GGML_OP_UNARY          | F32 EXP is disabled|
|GGML_UNARY_OP_EXP      | translator exists, but F32 EXP is disabled by policy via GGML_OP_UNARY|
|GGML_UNARY_OP_NEG      | none beyond global gates|
|GGML_OP_VIEW           | selected_experts test-gating name is force-disabled|
|GGML_GLU_OP_SWIGLU     | none beyond global gates|
|GGML_GLU_OP_SWIGLU_OAI | none beyond global gates|
|GGML_GLU_OP_GEGLU      | none beyond global gates|
|GGML_OP_SET_ROWS       | ne[3] must be 1|
|GGML_OP_CPY            | BF16 src/dst unsupported; src/dst element counts must match; selected non-contiguous test shapes blocked; VIEW output must fit source and be contiguous|
|GGML_OP_FLASH_ATTN_EXT | sink path unsupported (src[4]==null required); strict q/k/v pattern; max_bias must be 0; logit_softcap must be 0; gemma3n direct pattern forced to CPU|
|GGML_OP_CLAMP          | none beyond global gates|
|GGML_OP_PAD            | none beyond global gates|
|GGML_OP_SSM_CONV       | currently no hard block in policy (comment notes potential numerical instability)|
|GGML_OP_GATED_DELTA_NET| src2 PERMUTE unsupported; src3->ne[0] must be 1; K(op_params[0]) must be <= 1|
|GGML_OP_REPEAT         | selected GPU BF16 path is unsupported|
|GGML_OP_CUMSUM         | none beyond global gates|
|GGML_OP_DIAG           | none beyond global gates|
|GGML_OP_TRI            | none beyond global gates|
|GGML_OP_SET            | dst stride params (nb1/nb2/nb3) must match src0 strides|
*/
