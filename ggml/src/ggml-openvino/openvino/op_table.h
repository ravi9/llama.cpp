#pragma once

#include "node_context.h"
#include "op_support.h"

#include <utility>

namespace ov {
namespace frontend {
namespace ggml {

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
GGML_OP_CONVERTER(translate_unary_silu);
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
    CreatorFunction  translate;
    SupportsFunction supports;

    // Both arguments are required on purpose. Without this constructor OpEntry would be
    // an aggregate, and {translate_foo} would compile with supports silently null - so
    // the one guarantee this type exists to provide would not hold.
    OpEntry(CreatorFunction translate, SupportsFunction supports) :
        translate(std::move(translate)),
        supports(supports) {}
};

std::unordered_map<std::string, OpEntry> get_supported_ops();

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
