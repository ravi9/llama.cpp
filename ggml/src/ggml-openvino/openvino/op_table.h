#pragma once

#include "ggml-backend.h"
#include "ggml.h"
#include "node_context.h"

#include <string>
#include <unordered_map>
#include <utility>

// Why the gate turned a node away. Default-constructed means supported.
struct ggml_openvino_op_support {
    bool is_supported = true;
    std::string reason;

    operator bool() const { return is_supported; }
};

namespace ov {
namespace frontend {
namespace ggml {

// A rule is a pure function of one node. It must give the same answer every time it is
// asked: the scheduler consults the gate again on every graph rebuild, and a rule that
// changed its mind would move a node between backends mid-run.
using SupportsFunction = ggml_openvino_op_support (*)(const ggml_tensor * op);

// Forward declarations of support rules referenced in get_supported_ops()
static ggml_openvino_op_support supports_add_id(const ggml_tensor * op);
static ggml_openvino_op_support supports_add_mul_sub(const ggml_tensor * op);
static ggml_openvino_op_support supports_argsort(const ggml_tensor * op);
static ggml_openvino_op_support supports_concat(const ggml_tensor * op);
static ggml_openvino_op_support supports_cpy(const ggml_tensor * op);
static ggml_openvino_op_support supports_div(const ggml_tensor * op);
static ggml_openvino_op_support supports_flash_attn_ext(const ggml_tensor * op);
static ggml_openvino_op_support supports_gated_delta_net(const ggml_tensor * op);
static ggml_openvino_op_support supports_get_rows_set_rows(const ggml_tensor * op);
static ggml_openvino_op_support supports_mul_mat(const ggml_tensor * op);
static ggml_openvino_op_support supports_mul_mat_id(const ggml_tensor * op);
static ggml_openvino_op_support supports_pad(const ggml_tensor * op);
static ggml_openvino_op_support supports_permute(const ggml_tensor * op);
static ggml_openvino_op_support supports_pool_2d(const ggml_tensor * op);
static ggml_openvino_op_support supports_repeat(const ggml_tensor * op);
static ggml_openvino_op_support supports_reshape(const ggml_tensor * op);
static ggml_openvino_op_support supports_rope(const ggml_tensor * op);
static ggml_openvino_op_support supports_set(const ggml_tensor * op);
static ggml_openvino_op_support supports_ssm_conv(const ggml_tensor * op);
static ggml_openvino_op_support supports_sum_rows(const ggml_tensor * op);
static ggml_openvino_op_support supports_transpose(const ggml_tensor * op);
static ggml_openvino_op_support supports_tri(const ggml_tensor * op);
static ggml_openvino_op_support supports_unconstrained(const ggml_tensor * op);
static ggml_openvino_op_support supports_view(const ggml_tensor * op);

namespace op {

#define GGML_OP_CONVERTER(op) OutputVector op(const NodeContext & context)

GGML_OP_CONVERTER(translate_add);
GGML_OP_CONVERTER(translate_cont);
GGML_OP_CONVERTER(translate_concat);
GGML_OP_CONVERTER(translate_add_id);
GGML_OP_CONVERTER(translate_div);
GGML_OP_CONVERTER(translate_fill);
GGML_OP_CONVERTER(translate_get_rows);
GGML_OP_CONVERTER(translate_im2col);
GGML_OP_CONVERTER(translate_mulmat);
GGML_OP_CONVERTER(translate_mul_mat_id);
GGML_OP_CONVERTER(translate_permute);
GGML_OP_CONVERTER(translate_reshape);
GGML_OP_CONVERTER(translate_rms_norm);
GGML_OP_CONVERTER(translate_norm);
GGML_OP_CONVERTER(translate_l2_norm);
GGML_OP_CONVERTER(translate_sum_rows);
GGML_OP_CONVERTER(translate_sqr);
GGML_OP_CONVERTER(translate_rope);
GGML_OP_CONVERTER(translate_scale);
GGML_OP_CONVERTER(translate_sqrt);
GGML_OP_CONVERTER(translate_unary_softplus);
GGML_OP_CONVERTER(translate_soft_max);
GGML_OP_CONVERTER(translate_transpose);
GGML_OP_CONVERTER(translate_view);
GGML_OP_CONVERTER(translate_glu_swiglu);
GGML_OP_CONVERTER(translate_glu_swiglu_oai);
GGML_OP_CONVERTER(translate_glu_swiglu_clamp);
GGML_OP_CONVERTER(translate_glu_geglu);
GGML_OP_CONVERTER(translate_glu_geglu_quick);
GGML_OP_CONVERTER(translate_set_rows);
GGML_OP_CONVERTER(translate_cpy);
GGML_OP_CONVERTER(translate_argsort);
GGML_OP_CONVERTER(translate_flash_attn_ext);
GGML_OP_CONVERTER(translate_clamp);
GGML_OP_CONVERTER(translate_pad);
GGML_OP_CONVERTER(translate_ssm_conv);
GGML_OP_CONVERTER(translate_gated_delta_net);
GGML_OP_CONVERTER(translate_repeat);
GGML_OP_CONVERTER(translate_cumsum);
GGML_OP_CONVERTER(translate_fill);
GGML_OP_CONVERTER(translate_set);
GGML_OP_CONVERTER(translate_diag);
GGML_OP_CONVERTER(translate_tri);
GGML_OP_CONVERTER(translate_solve_tri);
GGML_OP_CONVERTER(translate_pool_2d);
GGML_OP_CONVERTER(translate_roll);

}  // namespace op

// One entry per op: how to translate it, and when it may be used. Both members are
// required, so a translator cannot be registered without a support rule - that is what
// keeps the gate from drifting away from what the translators actually accept.
struct OpEntry {
    CreatorFunction translate;
    SupportsFunction supports;

    // Both arguments are required on purpose. Without this constructor OpEntry would be
    // an aggregate, and {translate_foo} would compile with supports silently null - so
    // the one guarantee this type exists to provide would not hold.
    OpEntry(CreatorFunction translate, SupportsFunction supports) :
        translate(std::move(translate)),
        supports(supports) {}
};

const std::unordered_map<std::string, OpEntry> & get_supported_ops();
bool device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op);

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
