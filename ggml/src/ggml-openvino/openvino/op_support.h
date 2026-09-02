#pragma once

// Per-op support rules for the OpenVINO backend.
//
// Every entry in the translator table names one of these. The registry requires it, so
// a translator cannot be added without also stating when it may be used - which is what
// keeps the gate and the translators from drifting apart. Previously the conditions
// lived in a switch in ggml-openvino.cpp that 19 of the 54 registered ops never
// reached, so those ops were accepted unchecked and failed later during translation.

#include "ggml.h"

#include <string>

// Why the gate turned a node away. Default-constructed means supported.
struct ggml_openvino_op_support {
    bool        is_supported = true;
    std::string reason;

    operator bool() const { return is_supported; }
};

namespace ov {
namespace frontend {
namespace ggml {

// A rule is a pure function of one node. It must give the same answer every time it is
// asked: the scheduler consults the gate again on every graph rebuild, and a rule that
// changed its mind would move a node between backends mid-run.
//
// NOT EVERY NODE CAN BE DECLINED. If the destination tensor already has a buffer when the
// gate runs, no other backend can take the node and the scheduler aborts instead of falling
// back (ggml-backend.cpp, "pre-allocated tensor ... that cannot run the operation"). The CPU
// backend can only accept an OpenVINO buffer when the buffer type reports is_host, and it
// does not. In practice this means the nodes that write the KV cache: SET_ROWS into
// cache_k_l* / cache_v_l*, and views over them. Ordinary compute nodes have no buffer yet,
// so declining them is always safe, and so is declining a KV read.
//
// Before adding a rule that can fire on a cache write, check that it cannot fire in a real
// model. supports_get_rows_set_rows() is the one to watch: it serves GET_ROWS, which is never
// pre-allocated, and SET_ROWS, which is the cache write.
using SupportsFunction = ggml_openvino_op_support (*)(const ggml_tensor * op);

ggml_openvino_op_support supports_add_id(const ggml_tensor * op);
ggml_openvino_op_support supports_add_mul_sub(const ggml_tensor * op);
ggml_openvino_op_support supports_argsort(const ggml_tensor * op);
ggml_openvino_op_support supports_concat(const ggml_tensor * op);
ggml_openvino_op_support supports_cpy(const ggml_tensor * op);
ggml_openvino_op_support supports_div(const ggml_tensor * op);
ggml_openvino_op_support supports_flash_attn_ext(const ggml_tensor * op);
ggml_openvino_op_support supports_gated_delta_net(const ggml_tensor * op);
ggml_openvino_op_support supports_get_rows_set_rows(const ggml_tensor * op);
ggml_openvino_op_support supports_mul_mat(const ggml_tensor * op);
ggml_openvino_op_support supports_mul_mat_id(const ggml_tensor * op);
ggml_openvino_op_support supports_pad(const ggml_tensor * op);
ggml_openvino_op_support supports_permute(const ggml_tensor * op);
ggml_openvino_op_support supports_pool_2d(const ggml_tensor * op);
ggml_openvino_op_support supports_repeat(const ggml_tensor * op);
ggml_openvino_op_support supports_reshape(const ggml_tensor * op);
ggml_openvino_op_support supports_rope(const ggml_tensor * op);
ggml_openvino_op_support supports_set(const ggml_tensor * op);
ggml_openvino_op_support supports_ssm_conv(const ggml_tensor * op);
ggml_openvino_op_support supports_sum_rows(const ggml_tensor * op);
ggml_openvino_op_support supports_transpose(const ggml_tensor * op);
ggml_openvino_op_support supports_tri(const ggml_tensor * op);
ggml_openvino_op_support supports_unconstrained(const ggml_tensor * op);
ggml_openvino_op_support supports_view(const ggml_tensor * op);

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
