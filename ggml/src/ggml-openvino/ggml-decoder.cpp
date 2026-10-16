#include "ggml-decoder.h"

#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "ggml-openvino.h"
#include "ggml-quants.h"
#include "ggml.h"
#include "utils.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <openvino/core/dimension.hpp>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/partial_shape.hpp>
#include <openvino/core/type/bfloat16.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/runtime/tensor.hpp>
#include <ostream>
#include <set>
#include <stdexcept>
#include <string>
#include <cstring>
#include <unordered_map>
#include <vector>

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph,
                             ModelParams & model_params,
                             ComputeParams & compute_params,
                             std::map<std::string, std::shared_ptr<ov::Node>> & model_weights,
                             bool is_static,
                             bool is_stateful,
                             bool model_is_splitted,
                             bool is_prefill,
                             int prefill_chunk_size) :
    m_is_static(is_static),
    m_is_stateful(is_stateful),
    m_is_prefill(is_prefill),
    m_naive(false),
    m_prefill_chunk_size(prefill_chunk_size),
    m_model_is_splitted(model_is_splitted),
    m_cgraph(cgraph),
    m_model_weights(model_weights),
    m_model_params(model_params),
    m_compute_params(compute_params) {
    static bool printed_address_map = false;
    if (!printed_address_map) {
        if (ggml_openvino_getenv_int("GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS")) {
            printed_address_map = true;
            print_tensor_address_map(cgraph);
        }
    }

    validate_cgraph();

    set_input_output();
    compute_node_dynamic_dims();
    compute_model_inputs();
    compute_model_outputs();

    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        m_node_info_list[node_n].node_op_case = compute_op_case(m_node_info_list[node_n].node);
        m_node_info_list[node_n].node_op_type = compute_op_type(m_node_info_list[node_n].node);
    }
}

void GgmlOvDecoder::update_io(ggml_cgraph * cgraph) {
    m_cgraph = cgraph;
    m_model_inputs.clear();
    m_model_input_nodes.clear();
    m_model_outputs.clear();
    m_node_info_list.clear();
    set_input_output();
    compute_model_inputs();
    compute_model_outputs();
}

// llama keeps separate graphs for batches with and without outputs, so a cache hit can come from a
// graph built in other memory. The decoder then still points at the old graph's tensors.
bool GgmlOvDecoder::is_bound_to(const ggml_cgraph * cgraph) const {
    return m_cgraph == cgraph && cgraph->n_nodes > 0 && m_node_info_list.size() == (size_t) cgraph->n_nodes &&
           m_node_info_list.front().node == cgraph->nodes[0] &&
           m_node_info_list.back().node == cgraph->nodes[cgraph->n_nodes - 1];
}

GgmlOvDecoder::GgmlOvDecoder(ggml_cgraph * cgraph, std::map<std::string, std::shared_ptr<ov::Node>> & model_weights) {
    m_cgraph = cgraph;
    m_model_weights = model_weights;
    m_naive = true;
    set_input_output();
    compute_model_inputs();
    compute_model_outputs();
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        m_node_info_list[node_n].node_op_case = compute_op_case(m_node_info_list[node_n].node);
        m_node_info_list[node_n].node_op_type = compute_op_type(m_node_info_list[node_n].node);
    }
}

namespace {
std::shared_ptr<ov::op::v0::Parameter> create_parameter(const std::string & name, const BackendInputInfo & info) {
    auto param = std::make_shared<ov::op::v0::Parameter>(info.type, info.shape);
    param->set_friendly_name(name);
    param->output(0).get_tensor().set_names({name});
    return param;
}

float op_param_float(const ggml_tensor * node, size_t index) {
    float value = 0.0f;
    memcpy(&value, reinterpret_cast<const float *>(node->op_params) + index, sizeof(float));
    return value;
}

ov::frontend::gguf::RopeConfig rope_config_from_params(const int32_t * params, bool per_op) {
    ov::frontend::gguf::RopeConfig config;
    if (params == nullptr) {
        return config;
    }
    config.n_dims = params[1];
    config.n_ctx_orig = params[4];
    memcpy(&config.freq_base, params + 5, sizeof(float));
    memcpy(&config.freq_scale, params + 6, sizeof(float));
    memcpy(&config.ext_factor, params + 7, sizeof(float));
    memcpy(&config.attn_factor, params + 8, sizeof(float));
    memcpy(&config.beta_fast, params + 9, sizeof(float));
    memcpy(&config.beta_slow, params + 10, sizeof(float));
    config.per_op = per_op;
    config.is_imrope = params[2] == GGML_ROPE_TYPE_IMROPE;
    return config;
}

std::vector<int64_t> permute_order_from_params(const int32_t * params) {
    std::vector<int64_t> perm_values{0, 2, 1, 3};
    if (params != nullptr) {
        for (size_t input_axis = 0; input_axis < perm_values.size(); ++input_axis) {
            const size_t output_axis = static_cast<size_t>(params[input_axis]);
            perm_values[perm_values.size() - 1 - output_axis] = static_cast<int64_t>(perm_values.size() - 1 - input_axis);
        }
    }
    return perm_values;
}

bool is_inplace_op(const ggml_tensor * node) {
    return node->op == GGML_OP_SET_ROWS || node->op == GGML_OP_CPY || (node->op == GGML_OP_SCALE && node->view_src);
}

bool is_same_shape(const ggml_tensor * a, const ggml_tensor * b) {
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (a->ne[i] != b->ne[i]) {
            return false;
        }
    }
    return true;
}

bool is_conv_states_all_tensor(const ggml_tensor * tensor) {
    return tensor != nullptr && strncmp(tensor->name, "conv_states_all", strlen("conv_states_all")) == 0;
}

bool is_full_single_slot_writeback(const ggml_tensor * node) {
    return node->view_src != nullptr && node->view_src->ne[1] == 1 && node->src[1] != nullptr &&
           node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src == node->view_src &&
           node->src[1]->view_offs == 0 && ggml_nbytes(node->src[1]) == ggml_nbytes(node->view_src);
}
}  // namespace

// MoE expert aggregation (build_moe_ffn in llama-graph.cpp): each expert plane is
// `ggml_view_2d(experts, n_embd, n_tokens, experts->nb[2], i*experts->nb[1])` and the planes
// are summed with a chain of ADDs: moe_out = ((view_0 + view_1) + view_2) + ... + view_{n-1}.
// Detected structurally by walking the ADD chain and checking every leaf is a same-shape,
// same-stride VIEW of one common base tensor, indexed by a distinct expert-plane offset, and
// that the chain covers every plane of that base (leaf count == base->ne[1]). Only the
// outermost ADD of the chain satisfies this (inner ADDs see fewer leaves than base->ne[1]).
bool is_moe_expert_sum_add(const ggml_tensor * node) {
    std::vector<const ggml_tensor *> leaves;
    const ggml_tensor * cur = node;
    while (cur->op == GGML_OP_ADD) {
        if (cur->src[0] == nullptr || cur->src[1] == nullptr) {
            return false;
        }
        leaves.push_back(cur->src[1]);
        cur = cur->src[0];
    }
    leaves.push_back(cur);

    const ggml_tensor * base = nullptr;
    std::set<int64_t> plane_indices;
    for (const ggml_tensor * leaf : leaves) {
        if (leaf->op != GGML_OP_VIEW || leaf->src[0] == nullptr) {
            return false;
        }
        const ggml_tensor * leaf_base = leaf->src[0];
        if (base == nullptr) {
            base = leaf_base;
        } else if (leaf_base != base) {
            return false;
        }
        if (leaf->ne[0] != base->ne[0] || leaf->ne[1] != base->ne[2] || leaf->ne[2] != 1 || leaf->ne[3] != 1 ||
            leaf->nb[1] != base->nb[2]) {
            return false;
        }
        if (base->nb[1] == 0 || leaf->view_offs % base->nb[1] != 0) {
            return false;
        }
        int64_t plane = static_cast<int64_t>(leaf->view_offs / base->nb[1]);
        if (plane < 0 || plane >= base->ne[1] || !plane_indices.insert(plane).second) {
            return false;
        }
    }

    return base != nullptr && base->ne[1] > 1 && plane_indices.size() == static_cast<size_t>(base->ne[1]);
}

std::string GgmlOvDecoder::get_tensor_name(const ggml_cgraph * cgraph, const ggml_tensor * tensor) {
    if (tensor == nullptr) {
        return "";
    }
    if ((tensor->flags & GGML_TENSOR_FLAG_COMPUTE) || is_kvcache(tensor, nullptr)) {
        // Hash-table slots depend on tensor addresses and differ between contexts.
        // Graph ordinals disambiguate duplicate names while keeping compiled-model
        // ports identical for equivalent graphs in different contexts.
        const auto * node = std::find(cgraph->nodes, cgraph->nodes + cgraph->n_nodes, tensor);
        if (node != cgraph->nodes + cgraph->n_nodes) {
            return std::string(tensor->name) + "#n" + std::to_string(node - cgraph->nodes);
        }
        const auto * leaf = std::find(cgraph->leafs, cgraph->leafs + cgraph->n_leafs, tensor);
        if (leaf != cgraph->leafs + cgraph->n_leafs) {
            return std::string(tensor->name) + "#l" + std::to_string(leaf - cgraph->leafs);
        }
    }
    if (is_external_view(cgraph, tensor)) {
        // A VIEW produced by an earlier split enters this graph as an input. Sibling views often share a
        // name (e.g. the s_copy main/extra slices are both " (view)"), so key it by what it selects.
        return std::string(tensor->name) + "#v" + tensor->view_src->name + "+" + std::to_string(tensor->view_offs) +
               "x" + std::to_string(ggml_nelements(tensor));
    }
    return tensor->name;
}

static std::string get_tensor_ov_name(const ggml_cgraph * cgraph, const ggml_tensor * tensor) {
    return GgmlOvDecoder::get_tensor_name(cgraph, tensor);
}

static std::string get_tensor_graph_input_ov_name(const GgmlOvDecoder * decoder,
                                                  const ggml_cgraph * cgraph,
                                                  const ggml_tensor * tensor,
                                                  const ggml_tensor * op) {
    if (GgmlOvDecoder::is_inp_pos(tensor, op)) {
        return "inp_pos";
    }
    if (GgmlOvDecoder::is_inp_emb(tensor, op)) {
        return "embd";
    }
    if (GgmlOvDecoder::is_inp_mask(tensor, op)) {
        // Give the two attention masks distinct OV parameter names. build_attn_inp_kq_mask()
        // names the full-attention mask and the sliding-window mask identically, so keying a
        // parameter off the name alone makes the second mask overwrite the first and both
        // attention types read one parameter. Tell them apart by tensor identity, using the
        // SWA classification computed in compute_llm_params(). An empty swa_layers set means
        // there is only one mask in play and the plain name is correct.
        const bool is_swa = decoder->is_swa_mask(tensor);
        if (decoder->is_stateful()) {
            return is_swa ? "self_kq_mask_swa" : "self_kq_mask";
        }
        if (is_swa) {
            return get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    if (!decoder->get_model_params().swa_layers.empty() && GgmlOvDecoder::is_kv_idx(tensor, op) &&
        op->src[2] != nullptr) {
        // same as the masks: the full and the sliding-window cache get KV write indices of the same name
        const ggml_tensor * cache = op->src[2]->view_src != nullptr ? op->src[2]->view_src : op->src[2];
        auto layer = extract_layer_from_name(cache->name);
        if (layer.has_value() && decoder->is_swa_layer(layer.value())) {
            return get_tensor_ov_name(cgraph, tensor) + "_swa";
        }
    }
    return get_tensor_ov_name(cgraph, tensor);
}

void GgmlOvDecoder::set_input_output() {
    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        auto * node = m_cgraph->nodes[node_n];

        NodeInfo current_node_info;
        auto node_name = get_tensor_ov_name(m_cgraph, node);

        current_node_info.node = node;
        current_node_info.node_name = node_name;
        current_node_info.node_op_case = 0;
        current_node_info.data_addr = node->data;

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            // ggml_cast() records the result as its own CPY destination (src[1] == node); the cast is
            // translated from src[0] and dst_type alone.
            if (src == nullptr || src == node) {
                continue;
            }
            // A full single-slot writeback overwrites the whole cache; name the cache itself as the destination
            // (not a whole-storage VIEW of it) so the frontend rebinds the cache storage to the written value.
            if (i == 1 && node->op == GGML_OP_CPY && is_full_single_slot_writeback(node)) {
                src = node->view_src;
            }
            auto src_name = get_tensor_ov_name(m_cgraph, src);
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                src_name = get_tensor_graph_input_ov_name(this, m_cgraph, src, node);
            }
            current_node_info.node_inputs[src_name] = src;
            current_node_info.node_inputs_names.push_back(src_name);
        }

        m_node_info_list.push_back(current_node_info);
    }
}

int GgmlOvDecoder::compute_op_case(const ggml_tensor * node) const {
    int op_case = 0;
    switch (node->op) {
    case GGML_OP_RESHAPE: {
        if (m_naive) {
            break;
        }
        auto * src = node->src[0];
        // Identify recurrent sequence reshapes before size checks, which are ambiguous for one token.
        bool recurrent_sequence = false;
        for (int i = 0; i < m_cgraph->n_nodes && !recurrent_sequence; ++i) {
            const auto * consumer = m_cgraph->nodes[i];
            if (consumer->op == GGML_OP_MUL_MAT_ID && consumer->src[1] == node) {
                return 1;
            } else if (consumer->op == GGML_OP_SSM_CONV) {
                const auto * concat = consumer->src[0];
                if (concat->op == GGML_OP_CONCAT) {
                    const auto * transposed = concat->src[1];
                    recurrent_sequence = transposed->op == GGML_OP_TRANSPOSE && transposed->src[0] == node;
                }
            } else if (consumer->op == GGML_OP_UNARY && ggml_get_unary_op(consumer) == GGML_UNARY_OP_SOFTPLUS) {
                const auto * biased = consumer->src[0];
                recurrent_sequence = biased->op == GGML_OP_ADD && biased->src[0] == node;
            }
        }
        if (recurrent_sequence && node->ne[0] == src->ne[0] && node->ne[3] == 1) {
            return 6;
        }
        if (node->ne[0] == src->ne[0] && node->ne[2] == 1 && node->ne[3] == 1) {
            return 5;
        }
        if (src->op == GGML_OP_RESHAPE && src->src[0]->ne[0] == node->ne[0] && src->src[0]->ne[1] == node->ne[1]) {
            op_case = 4;
        } else if (node->ne[0] * node->ne[1] == src->ne[0]) {
            op_case = 1;
        } else if (src->ne[0] * src->ne[1] == node->ne[0]) {
            op_case = 2;
            if (src->ne[2] * src->ne[3] == node->ne[1]) {
                op_case = 5;
            }
        } else if (node->ne[0] == 1 && src->ne[0] * src->ne[1] * src->ne[2] == node->ne[1]) {
            op_case = 3;
        }
        break;
    }
    case GGML_OP_RMS_NORM: {
        if (node->src[0]->op == GGML_OP_VIEW) {
            if (is_same_shape(node->src[0]->src[0], node->src[0])) {
                op_case = 1;
            } else if (!m_model_params.has_rs_rollback &&
                       node->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // GDN attention is routed directly to this VIEW by get_output_names().
                op_case = 3;
            } else if (node->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                op_case = 2;
            }
        }
        break;
    }
    case GGML_OP_POOL_2D: {
        const ggml_op_pool pool_mode = static_cast<ggml_op_pool>(node->op_params[0]);
        switch (pool_mode) {
        case GGML_OP_POOL_MAX: {
            op_case = 1;
            break;
        }
        case GGML_OP_POOL_AVG: {
            op_case = 2;
            break;
        }
        default:
            op_case = 0;
            break;
        }
        break;
    }
    case GGML_OP_UPSCALE: {
        const int32_t mode_flags = node->op_params[0];
        const ggml_scale_mode scale_mode = static_cast<ggml_scale_mode>(mode_flags & 0xFF);
        switch (scale_mode) {
        case GGML_SCALE_MODE_NEAREST: {
            op_case = 1;
            break;
        }
        case GGML_SCALE_MODE_BILINEAR: {
            op_case = 2;
            break;
        }
        case GGML_SCALE_MODE_BICUBIC: {
            op_case = 3;
            break;
        }
        default:
            op_case = 0;
            break;
        }
        break;
    }
    case GGML_OP_CPY: {
        if (node->src[0]->op == GGML_OP_VIEW) {
            if (node->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                if (!m_model_params.has_rs_rollback) {
                    // op_case 7 replaces a single-slot cache; op_case 10 writes native GDN state
                    // into an active range of a larger non-rollback cache.
                    op_case = is_full_single_slot_writeback(node) ? 7 : 10;
                } else {
                    op_case = 1;
                }
            } else if (GgmlOvDecoder::is_conv_state_writeback(node)) {
                op_case = is_full_single_slot_writeback(node) ? 8 : 2;
                break;
            } else if (is_conv_states_all_tensor(node->view_src) && node->src[1] != nullptr &&
                       node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src == node->view_src) {
                op_case = 4;
                break;
            } else if (node->src[0]->src[0]->op == GGML_OP_GET_ROWS && node->src[1] != nullptr &&
                       node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src != nullptr &&
                       is_recurrent_cache(node->src[1]->view_src) && node->src[1]->view_src->ne[1] == 1) {
                // defrag remainder writeback of a single-slot cache, taken from a view of the one gather of
                // all states (see build_rs)
                op_case = 9;
            }
        } else if (node->src[0]->op == GGML_OP_GET_ROWS && node->src[1] != nullptr &&
                   node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src != nullptr &&
                   is_recurrent_cache(node->src[1]->view_src)) {
            // s_copy defrag remainder writeback: gathered extra state rows copied back into the cache
            op_case = node->src[1]->view_src->ne[1] == 1 ? 9 : 3;
        } else if (node->src[1] != nullptr && node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src != nullptr) {
            // op_case 5: KV write for decoder self-attention (dynamic write offset)
            // op_case 6: KV write for encoder self-attn or cross-attn (static offset)
            const ggml_tensor * kv_buf = node->src[1]->view_src;
            if (kv_buf->ne[1] == 1 && kv_buf->ne[2] == 1 && kv_buf->ne[3] == 1) {
                op_case = 6;
                // Forward-scan the graph for a FLASH_ATTN_EXT that reads from
                // the same buffer. Having a mask (src[3] != nullptr) implies
                // decoder self-attention and the write offset is dynamic.
                for (int i = 0; i < m_cgraph->n_nodes; i++) {
                    const ggml_tensor * n = m_cgraph->nodes[i];
                    if (n->op != GGML_OP_FLASH_ATTN_EXT) {
                        continue;
                    }
                    // K (src[1]) and V (src[2]) are 3-D views whose view_src is
                    // the flat KV buffer we are writing to.
                    if ((n->src[1] != nullptr && n->src[1]->view_src == kv_buf) ||
                        (n->src[2] != nullptr && n->src[2]->view_src == kv_buf)) {
                        if (n->src[3] != nullptr) {
                            op_case = 5;  // decoder self-attention: mask present
                        }
                        break;
                    }
                }
            }
        }
        break;
    }
    case GGML_OP_SCALE: {
        if (node->view_src && node->buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
            op_case = node->view_src->ne[1] == 1 ? 2 : 1;
        }
        break;
    }
    default:
        break;
    }
    return op_case;
}

std::optional<int> extract_layer_from_name(const std::string & name) {
    size_t pos1 = name.find("_l");
    if (pos1 == std::string::npos) {
        return std::nullopt;
    }
    pos1 += 2;
    size_t pos2 = name.find(' ', pos1);
    if (pos2 == std::string::npos) {
        pos2 = name.length();
    }
    std::string layer_str = name.substr(pos1, pos2 - pos1);
    int layer = std::stoi(layer_str);
    return layer;
}

// Recover the sliding window width from ggml's own SWA mask. llama.cpp never passes n_swa to a
// backend, but fill_mask() writes it into the mask: a query row keeps exactly the cells inside
// its window, so the widest row counts min(pos + 1, n_swa) unmasked cells. Counting rather than
// looking for a contiguous band is what makes this work on the KV-cache mask, where columns are
// physical cache cells in arbitrary order, not positions.
// Assumes LLAMA_SWA_TYPE_STANDARD, the only type the caller reconstructs.
static int get_swa_window_from_mask(const ggml_tensor * mask) {
    if (mask->data == nullptr || !ggml_backend_buffer_is_host(mask->buffer)) {
        return -1;
    }
    if (mask->type != GGML_TYPE_F16 && mask->type != GGML_TYPE_F32) {
        return -1;
    }

    const int64_t n_kv = mask->ne[0];
    const int64_t n_tokens = mask->ne[1];
    int64_t window = 0;

    for (int64_t r = 0; r < n_tokens; r++) {
        int64_t kept = 0;
        for (int64_t c = 0; c < n_kv; c++) {
            const size_t i = (size_t) r * n_kv + c;
            const float v = mask->type == GGML_TYPE_F16 ? ggml_fp16_to_fp32(((const ggml_fp16_t *) mask->data)[i]) :
                                                          ((const float *) mask->data)[i];
            if (v > -INFINITY) {
                kept++;
            }
        }
        window = std::max(window, kept);
    }

    return window > 0 ? (int) window : -1;
}

std::pair<ModelParams, ComputeParams> GgmlOvDecoder::compute_llm_params(ggml_cgraph * cgraph, bool is_static) {
    ModelParams model_params;
    ComputeParams compute_params;
    auto get_attention_pattern_case = [](const ggml_tensor * node) -> int {
        if (node == nullptr) {
            return -1;
        }

        switch (node->op) {
        case GGML_OP_FLASH_ATTN_EXT:
            if (node->src[0] == nullptr || node->src[1] == nullptr) {
                return -1;
            }
            switch (node->src[1]->op) {
            case GGML_OP_PERMUTE:
                // case 0: src[1] is PERMUTE of a cache VIEW, mask required
                if (node->src[3] != nullptr && node->src[1]->src[0] != nullptr &&
                    node->src[1]->src[0]->op == GGML_OP_VIEW) {
                    return 0;
                }
                break;
            case GGML_OP_CPY:
                // case 1: src[1] is CPY of a PERMUTE(VIEW), mask required
                if (node->src[3] != nullptr && node->src[1]->src[0] != nullptr &&
                    node->src[1]->src[0]->op == GGML_OP_PERMUTE && node->src[1]->src[0]->src[0] != nullptr &&
                    node->src[1]->src[0]->src[0]->op == GGML_OP_VIEW) {
                    return 1;
                }
                break;
            case GGML_OP_VIEW:
                // cases 4/5/6: whisper - K is a direct non-contiguous VIEW_3D of a KV cache
                if (node->src[1]->view_src != nullptr) {
                    if (node->src[3] != nullptr) {
                        return 4;  // decoder self-attention
                    }
                    return 5;      // cross-attention or encoder self-attention
                }
                break;
            default:
                break;
            }
            break;
        case GGML_OP_SOFT_MAX:
            // case 2: node op is SOFT_MAX, src 0 not null & op is MUL_MAT & the src 0 of MUL_MAT is PERMUTE & the permuted tensor src is the view of cache k
            if (node->src[0] != nullptr && node->src[1] != nullptr && node->src[0]->op == GGML_OP_MUL_MAT &&
                node->src[0]->src[0] != nullptr && node->src[0]->src[1] != nullptr &&
                node->src[0]->src[0]->op == GGML_OP_PERMUTE && node->src[0]->src[0]->src[0] != nullptr &&
                node->src[0]->src[0]->src[0]->op == GGML_OP_VIEW) {
                return 2;
            }
            // case 3: node op is SOFT_MAX, src 0 not null & op is ADD & the src 0 of ADD is MUL_MAT & the src 0 of MUL_MAT is PERMUTE
            if (node->src[0]->op == GGML_OP_ADD && node->src[0]->src[0] != nullptr &&
                node->src[0]->src[0]->op == GGML_OP_MUL_MAT && node->src[0]->src[0]->src[0] != nullptr &&
                node->src[0]->src[0]->src[0]->op == GGML_OP_PERMUTE) {
                return 3;
            }
            break;
        default:
            break;
        }

        return -1;
    };

    // Resolve the attention mask an attention node consumes, mirroring the src layout that
    // get_attention_pattern_case() classifies. Used by the SWA pre-pass below.
    auto get_attention_op_mask = [&get_attention_pattern_case](const ggml_tensor * node) -> const ggml_tensor * {
        switch (get_attention_pattern_case(node)) {
        case 0:
        case 1:
            return node->src[3];
        case 2:
        case 3:
            return node->src[1];
        default:
            return nullptr;
        }
    };

    // Pre-pass: classify sliding-window vs full-attention layers.
    //
    // An interleaved-SWA model keeps two KV caches and two attention masks, and hands each layer
    // whichever pair matches its attention type. The mask tensor does not say which is which: both
    // are named "attn_inp_kq_mask" by build_attn_inp_kq_mask(), and both carry the same n_kv because
    // llama_kv_cache::get_n_kv() pads occupancy up to a common multiple.
    //
    // The KV cache does say. Each cache allocates cache_k_l<N> once at load time with its own cell
    // count: the windowed cache is sized from the window
    // (PAD(min(size_base, n_swa*(unified ? n_seq_max : 1) + n_ubatch), 256), see
    // llama_kv_cache_iswa), the full-attention one spans the whole context. Read the LEAF buffer
    // behind the VIEW rather than the VIEW itself: the leaf extent is a constant per layer, known
    // from the first graph onwards, while the view grows with context depth and would invert the
    // comparison at shallow depth.
    //
    // Layers whose leaf is smaller than the largest leaf are the windowed ones. Equal extents do not
    // rule them out: the windowed cache is padded to n_swa + n_ubatch, which can equal the full
    // context (gemma-4 at -c 1024 -ub 512). See the mask fallback below.
    //
    // Getting this wrong is silent and severe: with the windowed layers classified as
    // full-attention, permute's KV slicing uses attention_size instead of attention_size_swa. The
    // two agree while the context is shorter than the window, then diverge, and the mask add fails
    // shape inference ("Failed to broadcast-merge input shapes") partway into a long prompt.
    {
        std::map<int, int64_t> layer_extent;                      // layer -> leaf cache_k cell count
        std::map<int, const ggml_tensor *> layer_mask;            // layer -> mask it consumes
        int64_t max_extent = 0;

        for (int i = 0; i < cgraph->n_nodes; i++) {
            const ggml_tensor * mask = get_attention_op_mask(cgraph->nodes[i]);
            if (mask == nullptr) {
                continue;
            }
            const ggml_tensor * cache_k_permute = nullptr;
            switch (get_attention_pattern_case(cgraph->nodes[i])) {
            case 0:  cache_k_permute = cgraph->nodes[i]->src[1];                     break;
            case 1:  cache_k_permute = cgraph->nodes[i]->src[1]->src[0];             break;
            case 2:  cache_k_permute = cgraph->nodes[i]->src[0]->src[0];             break;
            default: cache_k_permute = cgraph->nodes[i]->src[0]->src[0]->src[0];     break;
            }
            const ggml_tensor * cache_k_view = cache_k_permute->src[0];
            if (cache_k_view->op != GGML_OP_VIEW) {
                continue;
            }
            const ggml_tensor * leaf = cache_k_view->src[0];
            auto layer = extract_layer_from_name(leaf->name);
            if (!layer.has_value()) {
                continue;
            }
            layer_extent[layer.value()] = leaf->ne[1];
            layer_mask[layer.value()] = mask;
            max_extent = std::max(max_extent, leaf->ne[1]);
        }

        for (const auto & [layer, extent] : layer_extent) {
            if (extent < max_extent) {
                model_params.swa_layers.push_back(layer);
                if (model_params.swa_mask == nullptr) {
                    model_params.swa_mask = layer_mask[layer];
                }
            }
        }

        // equal extents: tell the two masks apart by the layers that read them, the larger group gets the swa role
        // (gemma-4: 4 of 5 layers). Groups of equal size can not be told apart, so leave the layers unclassified.
        // Two masks do not imply swa (deepseek sparse attention builds two full masks), but the role does not need
        // them to: it gives each group its own mask, KV write indices and attention size, which is right either way.
        // The only use of the window itself, the stateful mask rebuild, reads it back from the mask, and on a full
        // mask that gives back the causal mask. Decide from the first graph, before the window shows in the mask, so
        // the graph does not change, and is not recompiled, when it does.
        // TODO: "larger group is swa" is a model-level guess. Revisit when llama.cpp tells the backend which layers
        // use a sliding window. Stateless graphs only need the two groups kept apart (separate mask, KV write index
        // and attention size inputs), not the swa role itself, so the guess could be limited to stateful execution.
        if (model_params.swa_layers.empty()) {
            std::map<const ggml_tensor *, std::vector<int>> mask_layers;
            for (const auto & [layer, mask] : layer_mask) {
                mask_layers[mask].push_back(layer);
            }
            if (mask_layers.size() == 2 &&
                mask_layers.begin()->second.size() != std::next(mask_layers.begin())->second.size()) {
                auto swa = mask_layers.begin();
                if (std::next(swa)->second.size() > swa->second.size()) {
                    swa = std::next(swa);
                }
                model_params.swa_layers = swa->second;
                model_params.swa_mask = swa->first;
            }
        }
        std::sort(model_params.swa_layers.begin(), model_params.swa_layers.end());

        if (ggml_openvino_getenv_int("GGML_OPENVINO_LOG_SWA_LAYERS")) {
            std::string per_layer;
            for (const auto & [layer, extent] : layer_extent) {
                per_layer += " " + std::to_string(layer) + ":" + std::to_string(extent) +
                             (extent < max_extent ? "(swa)" : "");
            }
            GGML_LOG_WARN("ov-swa: attn_layers=%zu max_extent=%ld swa_layers=%zu |%s\n", layer_extent.size(),
                          (long) max_extent, model_params.swa_layers.size(), per_layer.c_str());
        }
    }

    bool rope_seen = false;
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];
        const int attention_pattern_case = get_attention_pattern_case(node);
        if (attention_pattern_case != -1) {
            ggml_tensor * cache_k_permute = nullptr;
            ggml_tensor * mask = nullptr;

            switch (attention_pattern_case) {
            case 0:
                cache_k_permute = node->src[1];
                mask = node->src[3];
                break;
            case 1:
                cache_k_permute = node->src[1]->src[0];
                mask = node->src[3];
                break;
            case 2:
                cache_k_permute = node->src[0]->src[0];
                mask = node->src[1];
                break;
            case 3:
                cache_k_permute = node->src[0]->src[0]->src[0];
                mask = node->src[1];
                break;
            case 4:
            case 5: {
                // whisper: K is a direct VIEW_3D of the KV buffer, no PERMUTE node
                auto * cache_k_view = node->src[1];  // VIEW_3D of kv_self.k or kv_cross.k`
                compute_params.token_len_per_seq = node->src[0]->ne[1];
                if (attention_pattern_case == 4) {
                    compute_params.attention_size = cache_k_view->ne[1];
                } else {
                    compute_params.attention_size_static = cache_k_view->ne[1];
                }
                continue;
            }
            default:
                break;
            }

            assert(cache_k_permute != nullptr);

            model_params.head_size = cache_k_permute->ne[0];
            model_params.n_heads_kv = cache_k_permute->ne[2];
            compute_params.input_len = node->src[0]->ne[1];
            compute_params.token_len_per_seq = node->src[0]->ne[1];

            auto * cache_k_view = cache_k_permute->src[0];
            if (cache_k_view->op != GGML_OP_VIEW || mask == nullptr) {
                continue;
            }

            ggml_tensor * cache_k = cache_k_view->src[0];
            int layer = extract_layer_from_name(cache_k->name).value();

            // Classified by the pre-pass above, which groups layers by mask tensor identity. The
            // mask NAME cannot be used: build_attn_inp_kq_mask() gives both masks the same name.
            const bool layer_is_swa = std::find(model_params.swa_layers.begin(), model_params.swa_layers.end(),
                                                layer) != model_params.swa_layers.end();

            model_params.kv_buffer_ctx_id = ggml_backend_openvino_buffer_get_ctx_id(cache_k->buffer);
            model_params.n_heads_kv_per_layer[layer] = cache_k_permute->ne[2];
            if (layer_is_swa) {
                model_params.ctx_per_seq_swa = cache_k->ne[1];
            } else {
                model_params.ctx_per_seq = cache_k->ne[1];
                model_params.n_seq = cache_k->ne[2];
            }

            compute_params.n_seq_active = mask->ne[3];
            auto seq_size = cache_k->ne[0] * cache_k->ne[1] * ggml_type_size(cache_k->type);
            size_t offset;
            memcpy(&offset, cache_k_view->op_params, sizeof(size_t));
            compute_params.seq_active_start = offset / seq_size;

            // a mask that is itself a graph input has the shape of its OV parameter
            const std::string mask_name = mask->op != GGML_OP_NONE      ? std::string() :
                                          mask == model_params.swa_mask ? get_tensor_ov_name(cgraph, mask) + "_swa" :
                                                                          get_tensor_ov_name(cgraph, mask);
            if (layer_is_swa) {
                compute_params.attention_size_swa = mask->ne[0];
                compute_params.swa_window = get_swa_window_from_mask(mask);
                compute_params.mask_swa_name = mask_name;
            } else {
                compute_params.attention_size = mask->ne[0];
                compute_params.mask_name = mask_name;
            }
            if (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
                compute_params.attention_size_swa = model_params.ctx_per_seq_swa;
                compute_params.token_len_per_seq = 1;
            }
        }

        if (node->op == GGML_OP_MUL_MAT && node->src[0]->op == GGML_OP_PERMUTE &&
            node->src[0]->src[0]->op == GGML_OP_VIEW && is_kvcache(node->src[0]->view_src, node->view_src)) {
            if (node->src[1]->op == GGML_OP_PERMUTE && node->src[1]->src[0]->op == GGML_OP_VIEW &&
                node->src[1]->src[0]->src[0]->op == GGML_OP_ROPE) {
                compute_params.attention_size = node->ne[0];
                compute_params.mask_name.clear();
            }
        }

        // if the node op is TRANSPOSE and its input is PERMUTE and the source of the PERMUTE is VIEW, then get the attention size with the TRANSPOSE node ne[0] (in case no GGML_OP_FLASH_ATTN_EXT)
        if (node->op == GGML_OP_TRANSPOSE && node->src[0]->op == GGML_OP_PERMUTE &&
            node->src[0]->src[0]->op == GGML_OP_VIEW) {
            compute_params.attention_size = node->ne[0];
            compute_params.mask_name.clear();
            if (is_static) {
                compute_params.attention_size = model_params.ctx_per_seq;
            }
        }
        if (node->op == GGML_OP_ROPE) {
            if (compute_params.token_len_per_seq == -1 && node->src[1] != nullptr) {
                compute_params.token_len_per_seq = ggml_nelements(node->src[1]);
            }

            // When multiple ROPE ops in the graph disagree on op_params (e.g. gemma4's
            // mixed SWA/non-SWA layers with different n_dims or freq_base), we cannot
            // share a single precomputed rope_sin/rope_cos. Track divergence so the
            // translator falls back to per-op make_sin_cos in that case.
            static_assert(sizeof(model_params.rope_params) == sizeof(int32_t) * 16, "rope_params size");
            if (!rope_seen) {
                memcpy(model_params.rope_params, node->op_params, sizeof(int32_t) * 16);
                rope_seen = true;
            } else if (memcmp(model_params.rope_params, node->op_params, sizeof(int32_t) * 16) != 0) {
                model_params.mixed_rope_params = true;
            }
        }
        if (node->op == GGML_OP_GATED_DELTA_NET) {
            model_params.state_size = node->src[0]->ne[0];
        }
        if (node->op == GGML_OP_SCALE && node->view_src != nullptr && is_recurrent_cache(node->view_src)) {
            if (model_params.n_rs_slots == -1) {
                model_params.n_rs_slots = node->view_src->ne[1];
            } else {
                GGML_ASSERT(model_params.n_rs_slots == node->view_src->ne[1]);
            }
            compute_params.cache_rs_reset_len = ggml_nelements(node) / node->view_src->ne[0];
            // view_offs is in bytes; the reset range is counted in cache rows.
            compute_params.cache_rs_reset_idx = node->src[0]->view_offs / node->view_src->nb[1];
        }
        // Capture the destination slot block of every recurrent state cache writeback, plus the
        // source window needed by conv state and packed GDN rollback writes. The active sequences
        // occupy a contiguous slot block [begin, begin + n_seqs) of the cache; these offsets move
        // with the batch, so they are fed to the cached model as runtime inputs.
        if (node->op == GGML_OP_CPY && node->view_src != nullptr && is_recurrent_cache(node->view_src) &&
            node->src[1] != nullptr && node->src[1]->op == GGML_OP_VIEW && node->src[1]->view_src == node->view_src) {
            const bool is_conv = is_conv_state_writeback(node);
            const bool is_gdn = node->src[0]->op == GGML_OP_VIEW && node->src[0]->src[0] != nullptr &&
                                node->src[0]->src[0]->op == GGML_OP_GATED_DELTA_NET;
            const bool is_extra = node->src[0]->op == GGML_OP_GET_ROWS;
            const bool is_gdn_rollback = is_gdn && is_same_shape(node->src[0], node->src[1]);

            const ggml_tensor * dest_view = node->src[1];
            const ggml_tensor * cache = node->view_src;
            // Writes into a flat buffer (bailingmoe3 conv_states_all, whisper's KV buffers) land at an element
            // offset that moves with the cache head; the translated CPY updates the buffer from that offset.
            const bool is_flat = !is_conv && !is_gdn && !is_extra &&
                                 (is_conv_states_all_tensor(cache) ||
                                  (cache->ne[1] == 1 && cache->ne[2] == 1 && cache->ne[3] == 1));
            const size_t row_bytes = cache->ne[0] * ggml_type_size(cache->type);
            if (is_gdn_rollback) {
                // Rollback GDN exposes an already-flattened [state, seq, snapshot] VIEW and copies
                // it to an identically-shaped cache VIEW. Non-rollback copies native 4-D state
                // [value, key, head, seq] into flattened cache rows, so the shapes differ. This
                // signature is local to the CPY and still works when fallback splits the graph.
                model_params.has_rs_rollback = true;
            }
            if (row_bytes > 0 && (is_conv || is_gdn || is_extra || is_flat) && !is_full_single_slot_writeback(node)) {
                ComputeParams::RsWriteback writeback;
                writeback.slot_begin = is_flat ? (int) (dest_view->view_offs / ggml_type_size(cache->type)) :
                                                 (int) (dest_view->view_offs / row_bytes);
                if (is_conv) {
                    writeback.src_begin = (int) (node->src[0]->view_offs / node->src[0]->view_src->nb[0]);
                } else if (is_gdn_rollback) {
                    writeback.src_begin = (int) (node->src[0]->view_offs / node->src[0]->view_src->nb[1]);
                }
                compute_params.rs_writebacks[get_tensor_ov_name(cgraph, node)] = writeback;
            }
            if ((is_conv || is_gdn) && !is_full_single_slot_writeback(node)) {
                compute_params.s_copy_active_slot_len = (int) dest_view->ne[1];
            }
        }
    }
    if (model_params.n_heads_kv == -1) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const auto * node = cgraph->nodes[i];
            const ggml_tensor * mask = nullptr;
            if (node->op == GGML_OP_SOFT_MAX) {
                mask = node->src[1];
            } else if (node->op == GGML_OP_FLASH_ATTN_EXT) {
                mask = node->src[3];
            } else {
                continue;
            }
            if (mask == nullptr || mask->op != GGML_OP_NONE || !(mask->flags & GGML_TENSOR_FLAG_INPUT) ||
                node->src[0] == nullptr) {
                continue;
            }
            model_params.is_cacheless_attn = true;
            model_params.n_seq = 1;
            model_params.ctx_per_seq = mask->ne[0];
            compute_params.input_len = node->src[0]->ne[1];
            compute_params.token_len_per_seq = compute_params.input_len;
            break;
        }
    }

    auto * output_tensor = cgraph->nodes[cgraph->n_nodes - 1];
    compute_params.output_len = output_tensor->ne[1];
    if (model_params.is_cacheless_attn) {
        for (int i = 0; i < cgraph->n_nodes; i++) {
            const auto * node = cgraph->nodes[i];
            if (node->op == GGML_OP_GET_ROWS && is_output_idx(node->src[1], node)) {
                compute_params.output_len = node->src[1]->ne[0];
                break;
            }
        }
    }
    // for NPU, output_len is always 1 except for llama-perplexity
    if (is_static && compute_params.output_len == 0) {
        compute_params.output_len = 1;
    }
    model_params.ctx = model_params.ctx_per_seq * model_params.n_seq;
    return {model_params, compute_params};
}

void GgmlOvDecoder::validate_cgraph() const {
    if (m_model_params.n_seq > 1 && m_is_static == true) {
        throw std::runtime_error("n_seq > 1 is not supported on NPU. Try setting -np 1.");
    }
}

ov::PartialShape GgmlOvDecoder::get_graph_input_shape(const ggml_tensor * op,
                                                      const ggml_tensor * input,
                                                      int dynamic_dim_index) const {
    if (m_naive) {
        return input != nullptr ? ov::PartialShape{get_shape(input)} : ov::PartialShape{get_shape(op)};
    }
    ov::PartialShape input_shape;

    if (is_inp_tok(input, op) || is_inp_pos(input, op)) {
        // tokens or positions
        const ov::Dimension len = m_is_static ? ov::Dimension(m_is_prefill ? m_prefill_chunk_size : 1) :
                                                ov::Dimension::dynamic();
        if (m_is_static && is_inp_pos(input, op)) {
            // IMROPE stacks n_planes (t/h/w/e) position planes back to back
            input_shape = ov::PartialShape{1, 1, 1, len.get_length() * get_inp_pos_n_planes(op)};
        } else {
            input_shape = ov::PartialShape{1, 1, 1, len};
        }

    } else if (is_output_idx(input, op)) {
        // output index
        input_shape = ov::PartialShape{1, 1, 1,
                                       m_is_static ? ov::Dimension(m_compute_params.output_len) : ov::Dimension::dynamic()};

    } else if (is_inp_mean(input, op)) {
        input_shape = m_is_static ? ov::PartialShape{1, 1, input->ne[1], m_prefill_chunk_size} :
                                    ov::PartialShape{1, 1, ov::Dimension::dynamic(), ov::Dimension::dynamic()};

    } else if (is_inp_scale_rows(input, op)) {
        input_shape = ov::PartialShape{1, 1, m_is_static ? (m_is_prefill ? m_prefill_chunk_size : 1) : -1, 1};

    } else if (is_inp_mask(input, op)) {
        // mask
        if (m_is_static) {
            input_shape = ov::PartialShape{1, 1, m_is_prefill ? m_prefill_chunk_size : 1, m_model_params.ctx};
        } else if (m_is_stateful) {
            input_shape = ov::PartialShape{1, 1, ov::Dimension::dynamic(), ov::Dimension::dynamic()};
        } else {
            input_shape = ov::PartialShape{ov::Dimension::dynamic(), 1, ov::Dimension::dynamic(), ov::Dimension::dynamic()};
        }

    } else if (is_recurrent_cache(input)) {
        input_shape = ov::PartialShape{get_shape(input)};
        if (!m_is_static && !m_is_stateful && input->ne[1] > 1) {
            input_shape[2] = -1;
        }

    } else if (is_kvcache(input, op)) {
        // kvcache
        input_shape = ov::PartialShape{get_shape(input)};
        // Whisper.cpp uses a fixed size 1D KV buffer [N, 1, 1, 1] (GGML) or [1, 1, 1, N] (OV).
        // the token fill level is handled by token_len_per_seq + dynamic mask input.
        // skip dynamic dim and stateful reshape for this layout.
        const bool is_flat_kv = (input->ne[1] == 1 && input->ne[2] == 1 && input->ne[3] == 1);
        if (!m_is_static && !is_flat_kv) {
            // do not fix ctx size to make llama-bench work across test params
            input_shape[2] = ov::Dimension::dynamic();
        }
        if ((is_stateful() && !is_flat_kv) || is_kvcache_4d(input)) {
            // Convert stateless KV cache layout [1, 1, seq, n_heads_kv * head_size]
            // to stateful layout [1, seq, n_heads_kv, head_size]. Stateless GPU graphs use the same
            // layout for the caches that qualify (see is_kvcache_4d), bound over the same memory.
            // NOTE: Gemma4 uses per-layer-type KV shapes, so no single scalar describes every
            // layer. E2B varies only the head size (sliding 256, full 512); 12B also varies the
            // head COUNT (sliding 8 x 256, full 1 x 512). Take the head count for this tensor's
            // own layer type and derive the head size from its own combined dim, so both layer
            // types get the correct split. Using the model-level count split 12B's sliding
            // states as 1 x 2048 and decoded garbage.
            assert(input_shape.size() == 4 && input_shape[0] == 1 && input_shape[1] == 1 &&
                   input_shape[2].is_dynamic() && input_shape[3].is_static());
            const int n_heads_kv = get_n_heads_kv_for_tensor(input);
            assert(n_heads_kv > 0 && input_shape[3].get_length() % n_heads_kv == 0);
            const int64_t combined_dim = input_shape[3].get_length();  // n_heads_kv * head_size
            const int64_t head_size = combined_dim / n_heads_kv;
            input_shape = {input_shape[0], ov::Dimension::dynamic(), n_heads_kv, head_size};
        }

    } else if (is_kv_idx(input, op)) {
        // kv update index
        const ov::Dimension len = m_is_static ? ov::Dimension(m_is_prefill ? m_prefill_chunk_size : 1) :
                                                ov::Dimension::dynamic();
        input_shape = ov::PartialShape{1, 1, 1, len};

    } else if (is_inp_s_copy(input, op) || is_s_copy_leaf(input)) {
        // On NPU the total slot count (n_seq_max) is fixed at translation time, so the s_copy
        // index list has a static length; on CPU/GPU it may change across compiles (defrag).
        input_shape = m_is_static ? ov::PartialShape{get_shape(input)} :
                        ov::PartialShape{1, 1, 1, ov::Dimension::dynamic()};

    } else if (input != nullptr && op != nullptr && is_inp_emb(input, op)) {
        input_shape = ov::PartialShape{get_shape(input)};
        if (!m_is_static) {
            input_shape[2] = -1;
        }
    }
    else {
        input_shape = ov::PartialShape{get_shape(input)};
    }
    if (dynamic_dim_index != -1 && m_model_is_splitted) {
        input_shape[3 - dynamic_dim_index] = ov::Dimension::dynamic();
    }
    if (op->op == GGML_OP_SOFT_MAX && op->src[1] != nullptr && op->src[1]->op == GGML_OP_NONE &&
        op->src[1]->flags & GGML_TENSOR_FLAG_INPUT && op->src[1] == input) {
        // for softmax input mask, the shape is [1, 1, seq_active, seq_active], where seq_active is determined by the input active sequence length instead of the kv cache sequence length
        if (m_is_static) {
            const int64_t seq_active = m_is_prefill ? m_prefill_chunk_size : 1;
            input_shape[2] = seq_active;
            input_shape[3] = seq_active;
        } else {
            input_shape[2] = ov::Dimension::dynamic();
            input_shape[3] = ov::Dimension::dynamic();
        }
    }
    return input_shape;
}

bool GgmlOvDecoder::is_s_copy_leaf(const ggml_tensor * tensor) const {
    if (tensor == nullptr || tensor->op != GGML_OP_NONE || m_cgraph == nullptr) {
        return false;
    }
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const ggml_tensor * node = m_cgraph->nodes[i];
        if (node->op != GGML_OP_GET_ROWS || node->src[0] == nullptr || node->src[1] == nullptr) {
            continue;
        }
        // The index list may reach the s_copy leaf through one or more VIEWs.
        const ggml_tensor * idx = node->src[1];
        while (idx != nullptr && idx->op == GGML_OP_VIEW) {
            idx = idx->src[0];
        }
        if (idx != tensor) {
            continue;
        }
        // The gathered data must be a recurrent state cache (cache_r/cache_s).
        const ggml_tensor * data = node->src[0];
        while (data != nullptr && (data->op == GGML_OP_VIEW || data->op == GGML_OP_RESHAPE)) {
            data = data->src[0];
        }
        if (data != nullptr && is_recurrent_cache(data)) {
            return true;
        }
    }
    return false;
}

bool GgmlOvDecoder::is_external_view(const ggml_cgraph * cgraph, const ggml_tensor * tensor) {
    if (cgraph == nullptr || tensor == nullptr || tensor->op != GGML_OP_VIEW || tensor->view_src == nullptr) {
        return false;
    }
    const auto nodes_end = cgraph->nodes + cgraph->n_nodes;
    if (std::find(cgraph->nodes, nodes_end, tensor) != nodes_end) {
        return false;
    }
    const ggml_tensor * root = tensor;
    while (root->op == GGML_OP_VIEW && root->src[0] != nullptr) {
        root = root->src[0];
    }
    if (is_kvcache(root, nullptr)) {
        return false;
    }
    const auto leafs_end = cgraph->leafs + cgraph->n_leafs;
    return std::find(cgraph->nodes, nodes_end, root) == nodes_end &&
           std::find(cgraph->leafs, leafs_end, root) == leafs_end;
}

bool GgmlOvDecoder::is_external_view_input(const ggml_tensor * tensor) const {
    return is_external_view(m_cgraph, tensor);
}

std::optional<int64_t> GgmlOvDecoder::get_runtime_input_value(const std::string & name) const {
    if (name == "cache_rs_reset_idx") {
        return std::max(0, m_compute_params.cache_rs_reset_idx);
    }
    if (name == "cache_rs_reset_len") {
        // A step without a reset must not reuse the previous step's reset range.
        return m_compute_params.cache_rs_reset_idx < 0 ? 0 : m_compute_params.cache_rs_reset_len;
    }
    // KV-cache reads (see describe_kv_view).
    if (name == "attention_size") {
        return m_compute_params.attention_size;
    }
    if (name == "attention_size_swa") {
        return m_compute_params.attention_size_swa;
    }
    if (name == "seq_active_start") {
        return m_compute_params.seq_active_start;
    }
    if (name == "swa_window") {
        // Stateful sliding-window mask (see rewrite_stateful_masks). An unknown window keeps every cached
        // position, which matches ggml as long as the context still fits in the window.
        return m_compute_params.swa_window > 0 ? m_compute_params.swa_window : std::numeric_limits<int32_t>::max();
    }
    const std::string slot_prefix = "rs_slot_begin_";
    const std::string src_prefix = "rs_src_begin_";
    const auto writeback = [&](const std::string & key) -> const ComputeParams::RsWriteback & {
        const auto it = m_compute_params.rs_writebacks.find(key);
        if (it == m_compute_params.rs_writebacks.end()) {
            std::string known;
            for (const auto & entry : m_compute_params.rs_writebacks) {
                known += " '" + entry.first + "'";
            }
            throw std::runtime_error("ggml-openvino: no recurrent-state writeback '" + key + "' for input '" + name +
                                     "'; graph has:" + known);
        }
        return it->second;
    };
    if (name.compare(0, slot_prefix.size(), slot_prefix) == 0) {
        return writeback(name.substr(slot_prefix.size())).slot_begin;
    }
    if (name.compare(0, src_prefix.size(), src_prefix) == 0) {
        return writeback(name.substr(src_prefix.size())).src_begin;
    }
    return std::nullopt;
}

bool GgmlOvDecoder::node_is_used_as_src(const int node_idx) {
    ggml_tensor * node = m_cgraph->nodes[node_idx];
    for (int i = node_idx; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * other_node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (other_node->src[j] == node) {
                return true;
            }
        }
    }
    return false;
}

void GgmlOvDecoder::compute_model_inputs() {
    m_model_inputs.clear();
    m_model_input_nodes.clear();
    m_inputs.clear();
    m_kvcache_4d.clear();
    // remember the KV caches that get_graph_input_shape gave the 4-D layout, so that they are bound with that shape
    auto note_kvcache_4d = [this](const ggml_tensor * input, const ggml_tensor * op, const ov::PartialShape & shape) {
        if (!m_is_stateful && is_kvcache(input, op) && shape.rank().is_static() && shape.rank().get_length() == 4 &&
            shape[1].is_dynamic()) {
            m_kvcache_4d.insert(input);
        }
    };
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * node = m_cgraph->nodes[i];
        // the node op is NONE means this node maybe as input of later nodes, we should add it to model inputs for this node.
        if (node->op == GGML_OP_NONE && node_is_used_as_src(i)) {
            std::string node_name = get_tensor_ov_name(m_cgraph, node);
            if (m_model_weights.find(node_name) == m_model_weights.end()) {
                m_inputs[node_name] = node;
                m_model_inputs[node_name] = {get_ov_type(node),
                                             get_graph_input_shape(node, nullptr, m_node_dynamic_dims[node])};
                note_kvcache_4d(node, nullptr, m_model_inputs[node_name].shape);
            }
            continue;
        }
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            std::string src_name = get_tensor_ov_name(m_cgraph, src);
            if (src->flags & GGML_TENSOR_FLAG_INPUT) {
                src_name = get_tensor_graph_input_ov_name(this, m_cgraph, src, node);
            }
            if (m_model_weights.find(src_name) != m_model_weights.end()) {
                continue;
            }
            // A view over a weight is served by the base tensor's Constant, never by a Parameter.
            if (src->view_src != nullptr &&
                m_model_weights.find(get_tensor_ov_name(m_cgraph, src->view_src)) != m_model_weights.end()) {
                continue;
            }

            bool is_intermediate_node = false;
            for (const auto & node_info : m_node_info_list) {
                if (node_info.node == src) {
                    is_intermediate_node = true;
                    break;
                }
            }
            if (is_intermediate_node) {
                continue;
            }
            if (m_model_inputs.find(src_name) != m_model_inputs.end()) {
                continue;
            }

            m_inputs[src_name] = src;

            ggml_backend_buffer * buffer = src->buffer;
            // GGML_BACKEND_BUFFER_USAGE_ANY are kv caches
            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_ANY) {
                if (auto it = std::find(m_model_params.kv_names.begin(), m_model_params.kv_names.end(), src_name);
                    it == m_model_params.kv_names.end()) {
                    m_model_params.kv_names.push_back(src_name);
                }
            }
            // Resolve nested VIEW nodes by following src[0] until the first non-VIEW tensor. A VIEW whose
            // storage root lives outside this graph (e.g. the s_copy index slices computed by an earlier
            // scheduler split) is bound with its own data and extent, under the name its consumer uses.
            const ggml_tensor * view_root = src;
            while (view_root->op == GGML_OP_VIEW && view_root->src[0] != nullptr) {
                view_root = view_root->src[0];
            }
            if (view_root != src && !is_external_view_input(src)) {
                src = const_cast<ggml_tensor *>(view_root);
                src_name = get_tensor_ov_name(m_cgraph, src);
            }
            m_inputs[src_name] = src;
            m_model_inputs[src_name] = {get_ov_type(src),
                                        get_graph_input_shape(node, src, m_node_dynamic_dims[src])};
            note_kvcache_4d(src, node, m_model_inputs[src_name].shape);
        }
    }

    compute_kv_views();

    // Slot offsets are independent runtime information, even for a static-shape model.
    if (!m_naive) {
        if (m_compute_params.cache_rs_reset_idx != -1 && m_model_params.n_rs_slots != 1) {
            m_model_inputs["cache_rs_reset_idx"] = {ov::element::i64, ov::PartialShape{1}};
            m_model_inputs["cache_rs_reset_len"] = {ov::element::i64, ov::PartialShape{1}};
        }
        for (const auto & entry : m_compute_params.rs_writebacks) {
            m_model_inputs["rs_slot_begin_" + entry.first] = {ov::element::i64, ov::PartialShape{1}};
            // Only a writeback that copies a moving source window has a source offset.
            if (entry.second.src_begin >= 0) {
                m_model_inputs["rs_src_begin_" + entry.first] = {ov::element::i64, ov::PartialShape{1}};
            }
        }
    }
    for (const auto & input : m_model_inputs) {
        m_model_input_nodes[input.first] = create_parameter(input.first, input.second);
    }

    // attention_size(_swa) is the key dimension of the KQ mask. Read it from the mask shape inside the model instead
    // of binding it as a runtime input: the GPU plugin then gets the KV slice shapes from shape inference, rather than
    // waiting for the input value on the device, and can keep the fused attention and StatelessKV kernels.
    if (!m_is_static && !m_is_stateful && ggml_openvino_shape_from_mask_enabled()) {
        const auto from_mask = [this](const std::string & name, const std::string & mask) {
            const auto input_it = m_model_input_nodes.find(name);
            const auto mask_it = m_model_input_nodes.find(mask);
            if (input_it == m_model_input_nodes.end() || mask.empty() || mask_it == m_model_input_nodes.end()) {
                return;
            }
            auto mask_shape = std::make_shared<ov::op::v3::ShapeOf>(mask_it->second, ov::element::i64);
            auto key_dim = std::make_shared<ov::op::v8::Gather>(
                mask_shape, ov::op::v0::Constant::create(ov::element::i64, {1}, {3}),
                ov::op::v0::Constant::create(ov::element::i64, {}, {0}));
            key_dim->set_friendly_name(name);
            input_it->second = key_dim;
        };
        from_mask("attention_size", m_compute_params.mask_name);
        from_mask("attention_size_swa", m_compute_params.mask_swa_name);
    }

    // The gguf frontend seeds get_model_inputs() into its tensor map but seeds no weights; it
    // expects the decoder to supply them. This backend already decoded every weight into a ready
    // f32 decompression subgraph at load time (ggml_openvino_*_weight_extra::weight_node, surfaced
    // via m_model_weights), so hand those pre-built nodes over here rather than have translate_weight
    // re-decode raw bytes -- which the backend has overwritten in place with the extracted layout.
    for (const auto & [weight_name, weight_node] : m_model_weights) {
        if (weight_node != nullptr) {
            m_model_input_nodes[weight_name] = weight_node;
        }
    }
}

void GgmlOvDecoder::compute_model_outputs() {
    m_model_outputs.clear();
    m_model_output_names.clear();
    for (int node_n = 0; node_n < m_cgraph->n_nodes; node_n++) {
        auto * cur_node = m_cgraph->nodes[node_n];
        // if the node op is NONE means this node is not used at all, we can skip it directly without adding to model outputs.
        if (cur_node->op == GGML_OP_NONE || cur_node->op == GGML_OP_VIEW || cur_node->op == GGML_OP_RESHAPE) {
            continue;
        }
        if (::is_inplace_op(cur_node) && ggml_nbytes(cur_node) == 0) {
            continue;
        }
        auto cur_node_use_count = m_cgraph->use_counts[ggml_hash_find(&m_cgraph->visited_hash_set, cur_node)];
        if (cur_node_use_count == 0) {
            // The output of in-place ops is the view_src tensor, which is updated in place. We should use the view_src name as the output name to make sure it can be correctly matched with the later ops that use the view_src.
            if (cur_node != nullptr && ::is_inplace_op(cur_node) && ggml_nbytes(cur_node) > 0) {
                cur_node = cur_node->view_src;
            }
        } else {
            int input_use_count = 0;
            for (int i = 0; i < m_cgraph->n_nodes; i++) {
                ggml_tensor * node = m_cgraph->nodes[i];
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != NULL && node->src[j] == cur_node) {
                        input_use_count++;
                    }
                }
            }
            if (input_use_count == cur_node_use_count) {
                cur_node = nullptr;
            }
        }
        if (cur_node != nullptr) {
            std::string cur_node_name = get_tensor_ov_name(m_cgraph, cur_node);
            m_model_outputs[cur_node_name] = cur_node;
            m_model_output_names.insert(cur_node_name);
        }
    }
}

const ggml_tensor * GgmlOvDecoder::get_tensor_used_op(const ggml_tensor * tensor) const {
    if (tensor == nullptr) {
        return nullptr;
    }
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto * node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (node->src[j] == tensor) {
                return node;
            }
        }
    }
    return nullptr;
}

bool GgmlOvDecoder::is_kvcache_4d(const ggml_tensor * cache) const {
    static const bool enabled = ggml_openvino_getenv_int("GGML_OPENVINO_ENABLE_KV_4D") > 0;
    if (!enabled || m_is_static || m_is_stateful || m_model_is_splitted || !ggml_openvino_is_gpu()) {
        return false;
    }
    if (cache->op != GGML_OP_NONE || cache->view_src != nullptr || !is_kvcache(cache, nullptr)) {
        return false;
    }
    // one sequence, [n_heads_kv * head_size, ctx, 1, 1]; whisper's flat KV buffer has ctx == 1
    if (cache->ne[1] <= 1 || cache->ne[2] != 1 || cache->ne[3] != 1) {
        return false;
    }
    const int n_heads_kv = get_n_heads_kv_for_tensor(cache);
    if (n_heads_kv <= 0 || cache->ne[0] % n_heads_kv != 0) {
        return false;
    }
    // check the nodes that take the cache itself as a source (views of those views, such as the PERMUTE of the read
    // VIEW, are checked through them)
    auto feeds_only_permutes = [this](const ggml_tensor * view) {
        for (int i = 0; i < m_cgraph->n_nodes; i++) {
            const auto * node = m_cgraph->nodes[i];
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                if (node->src[j] == view && node->op != GGML_OP_PERMUTE) {
                    return false;
                }
            }
        }
        return true;
    };
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto * node = m_cgraph->nodes[i];
        bool uses_cache = false;
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            uses_cache = uses_cache || node->src[j] == cache;
        }
        if (!uses_cache) {
            continue;
        }
        // write: whole rows into the cache itself
        if (node->op == GGML_OP_SET_ROWS && node->src[2] == cache && node->src[0]->ne[0] == cache->ne[0]) {
            continue;
        }
        // read: the first rows of the cache, each row split into heads, permuted for attention
        if (node->op == GGML_OP_VIEW && node->src[0] == cache && node->view_offs == 0 &&
            node->ne[0] * node->ne[1] == cache->ne[0] && node->nb[2] == cache->nb[1] && node->ne[3] == 1 &&
            feeds_only_permutes(node)) {
            continue;
        }
        return false;
    }
    return true;
}

const ggml_tensor * GgmlOvDecoder::get_tensor_from_name(const std::string & name) const {
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const auto * node = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            const auto * src = node->src[j];
            if (src == nullptr) {
                break;
            }
            if (get_tensor_ov_name(m_cgraph, src) == name) {
                return src;
            }
        }
    }
    return nullptr;
}

std::map<std::string, std::string> GgmlOvDecoder::get_kv_param_res_names() const {
    std::map<std::string, std::string> kv_param_res_names;
    for (const auto & name : m_model_params.kv_names) {
        kv_param_res_names[name] = name;
    }
    return kv_param_res_names;
}

// MUL_MAT_ID's src[0] is the [k, m, n_expert] expert-weight tensor. It is always a constant per-expert
// weight table -- never a computed activation -- regardless of whether the backend happened to mark its
// buffer as GGML_BACKEND_BUFFER_USAGE_WEIGHTS (test-backend-ops, for example, never sets that usage
// flag, unlike real inference). Without this, non-quantized (F16/F32/BF16) expert weights would fall
// through the check below as "not a weight", get decoded as a Parameter/activation instead of a
// Constant, and crash GatherMatmul's "only constant weights are supported" check.
static bool is_mul_mat_id_expert_weight(const ggml_tensor * node, int src_index) {
    return node->op == GGML_OP_MUL_MAT_ID && src_index == 0;
}

std::map<std::string, std::shared_ptr<ov::Node>> GgmlOvDecoder::create_weight_nodes(ggml_cgraph * cgraph, bool naive) {
    std::map<std::string, std::shared_ptr<ov::Node>> model_weights;
    auto * nodes = cgraph->nodes;
    auto n_nodes = cgraph->n_nodes;
    for (int node_i = 0; node_i < n_nodes; node_i++) {
        auto * node = nodes[node_i];
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }

            // A view over a weight is served by the base tensor's Constant.
            ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name = get_tensor_ov_name(cgraph, base);
            if (is_rope_freqs_weight(base, node)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type) ||
                is_mul_mat_id_expert_weight(node, i)) {
                if (model_weights.find(src_name) == model_weights.end()) {
                    auto weight_node = create_weight_node(base, naive);
                    weight_node->set_friendly_name(src_name);
                    model_weights[src_name] = weight_node;
                }
            }
        }
    }
    return model_weights;
}

// Process-lifetime cache for weight nodes built from NON-OpenVINO buffers (e.g. the
// token_embd.weight copy that lives in a CPU/mmap buffer and feeds GET_ROWS). Such
// tensors have no OV buffer context to own a cached extra, so without this they are
// re-extracted/re-requantized on every (re)compile — for token_embd that is a ~1-2 GB
// F32 dequant each time. Keyed by tensor->data, which is stable for the process and
// uniquely identifies the immutable weight bytes. OV-buffer weights keep using the
// per-tensor extra cache and never reach here.
static std::mutex g_nonov_weight_cache_mutex;
static std::unordered_map<const void *, std::shared_ptr<ov::Node>> g_nonov_weight_cache;

std::set<std::string> GgmlOvDecoder::collect_weight_names(ggml_cgraph * cgraph) {
    // Mirrors the name-selection logic of create_weight_nodes() but builds no nodes,
    // so topology checks don't trigger weight extraction/requantization.
    std::set<std::string> names;
    for (int node_i = 0; node_i < cgraph->n_nodes; node_i++) {
        auto * node = cgraph->nodes[node_i];
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            const ggml_tensor * base = src->view_src ? src->view_src : src;
            std::string src_name(base->name);
            if (is_rope_freqs_weight(base, node)) {
                src_name = "rope_freqs.weight";
            }
            ggml_backend_buffer * buffer = base->buffer;
            if (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS || ggml_is_quantized(base->type)) {
                names.insert(src_name);
            }
        }
    }
    return names;
}

std::shared_ptr<ov::Node> GgmlOvDecoder::create_weight_node(ggml_tensor * tensor, bool naive) {
    const bool is_ov_buffer = ggml_backend_buffer_is_openvino(tensor->buffer);

    // Check if we have a pre-built constant from the OpenVINO backend buffer
    // This is set during ggml_backend_openvino_buffer_set_tensor
    if (tensor->extra) {
        OPENVINO_ASSERT(is_ov_buffer, "Unsupported weight tensor: " + std::string(tensor->name) +
                                          " Possibly this is a cpu backend repacked quantized weights");
        // Cast to our extra base type and check the type
        auto * extra_base = static_cast<ggml_openvino_extra_base *>(tensor->extra);

        if (extra_base->type == ggml_openvino_extra_base::Type::WEIGHT) {
            // F16/F32/BF16 weight with shared-memory constant
            auto * weight_extra = static_cast<ggml_openvino_weight_extra *>(tensor->extra);
            if (weight_extra->weight_node) {
                // GGML_LOG_DEBUG("%s: using pre-built weight node for %s\n", __func__, tensor->name);
                return weight_extra->weight_node;
            }
        } else if (extra_base->type == ggml_openvino_extra_base::Type::QUANTIZED_WEIGHT) {
            // Quantized weight with pre-extracted data
            auto * quant_extra = static_cast<ggml_openvino_quantized_weight_extra *>(tensor->extra);
            if (quant_extra->weight_node) {
                // GGML_LOG_DEBUG("%s: using pre-extracted quantized weight node for %s\n", __func__, tensor->name);
                return quant_extra->weight_node;
            }
        }
    }

    // MUL_MAT_ID expert weights are 3D GGML tensors [k, m, n_expert].
    // Keep the full reversed 4D shape when materializing non-quantized constants,
    // otherwise the expert dimension is collapsed and later Gather/MatMul logic
    // only sees a single expert slice.
    if (!ggml_is_quantized(tensor->type) && (tensor->ne[2] > 1 || tensor->ne[3] > 1)) {
        auto weight_tensor = ov::Tensor(get_ov_type(tensor), get_shape(tensor), tensor->data);
        auto weight_node = std::make_shared<ov::op::v0::Constant>(weight_tensor);
        weight_node->set_friendly_name(tensor->name);
        return weight_node;
    }

    // Non-OV-buffer weights (CPU/mmap, e.g. the GET_ROWS token_embd copy) have no buffer
    // context to cache an extra in, so memoize them here keyed by their (stable) data
    // pointer to avoid re-extracting on every recompile. Opt-in via
    // GGML_OPENVINO_REDUCE_COMPILE_MEM or GGML_OPENVINO_MEMORY_OPTIMIZE. Skip
    // for `naive` (test/naive path) since use_bias changes the produced node.
    const bool cacheable_nonov = ggml_openvino_reduce_compile_mem_enabled() && !is_ov_buffer &&
                                 !naive && tensor->data != nullptr;
    if (cacheable_nonov) {
        std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
        auto it = g_nonov_weight_cache.find(tensor->data);
        if (it != g_nonov_weight_cache.end()) {
            return it->second;
        }
    }

    // There are three cases where we need to create a new weight node:
    // 1. weights are in openvino_host_buffer. Weight loading to host buffer will not trigger backend_buffer_set_tensor
    // 2. weights are in cpu/cpu_mapped buffer. On token_embd.weight goes to case 1 or 2, depending on whether mmap or direct_io is used
    // 3. test-backend-ops. buffers in test-backend-ops does not set USAGE_WEIGHT so backend_buffer_set_tensor will not create weight node

    // GGML_LOG_DEBUG("%s: creating new weight node for %s\n", __func__, tensor->name);
    static const std::set<ggml_type> weight_types = {GGML_TYPE_F32,  GGML_TYPE_F16,  GGML_TYPE_BF16, GGML_TYPE_Q8_0,
                                                     GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q5_1, GGML_TYPE_Q4_K,
                                                     GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_MXFP4};
    if (weight_types.find(tensor->type) == weight_types.end()) {
        throw std::runtime_error("Unexpected weight tensor type: " + std::string(tensor->name) + " with type " +
                                 ggml_type_name(tensor->type));
    }

    OvWeight ov_weight;
    if (ggml_is_quantized(tensor->type)) {
        auto use_bias = naive;
        if (is_ov_buffer) {
            // For quantized weights, copy raw data to a temp buffer first because
            // process_weight_tensor reads from data and writes extracted results
            // (weights/scales/zp) to output_base_ptr — they would overlap if both
            // point to tensor->data.
            size_t raw_size = ggml_nbytes(tensor);
            std::vector<uint8_t> tmp(raw_size);
            memcpy(tmp.data(), tensor->data, raw_size);
            ov_weight = process_weight_tensor(tensor, tmp.data(), tensor->data, use_bias);
        } else {
            ov_weight = process_weight_tensor(tensor, tensor->data, nullptr, use_bias);
        }
    } else {
        // For non-quantized weights (F16/F32/BF16), data is already in tensor->data.
        // process_weight_tensor will create an ov::Tensor wrapping tensor->data directly.
        ov_weight = process_weight_tensor(tensor, tensor->data, tensor->data);
    }

    ov_weight.weight_node->set_friendly_name(tensor->name);
    if (!is_ov_buffer) {
        if (cacheable_nonov) {
            std::lock_guard<std::mutex> lock(g_nonov_weight_cache_mutex);
            // Another thread may have inserted concurrently; keep the first.
            auto [it, inserted] = g_nonov_weight_cache.emplace(tensor->data, ov_weight.weight_node);
            return it->second;
        }
        return ov_weight.weight_node;
    }

    ggml_openvino_extra_base * extra;
    if (ov_weight.is_quantized()) {
        extra = new ggml_openvino_quantized_weight_extra(std::move(ov_weight.weights), std::move(ov_weight.scales),
                                                         std::move(ov_weight.zp), ov_weight.weight_node);
    } else {
        extra = new ggml_openvino_weight_extra(std::move(ov_weight.weights), ov_weight.weight_node);
    }
    ggml_openvino_buffer_register_extra(tensor, extra);

    return ov_weight.weight_node;
}

void GgmlOvDecoder::dump_cgraph(const ggml_cgraph * cgraph, std::string & filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Failed to open file" << '\n';
        return;
    }

    file << "=== GRAPH ===\n";

    // clang-format off
    file << "n_nodes = " << cgraph->n_nodes << "\n";
    file << " " << std::setw(3) << "nodes"
                <<  std::setw(15) << "shape"
                << std::setw(20) << "op"
                << std::setw(20) << "name"
                << std::setw(3) << "    "
                << std::setw(62) << "stride"
                << std::setw(20) << "buffer_type"
                << "\n";
    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        // Get buffer type name
        const char * buf_name = "none";
        ggml_backend_buffer_t buf = node->view_src ? node->view_src->buffer : node->buffer;
        if (buf) {
            buf_name = ggml_backend_buffer_name(buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << ", "
             << std::setw(5) << node->ne[2] << ", "
             << std::setw(5) << node->ne[3] << "] "
             << std::left << std::setw(20) << ggml_op_name(node->op) << std::right << " "
             << std::left << std::setw(45) << node->name << std::right
             << std::setw(2) << "[ "
             << std::setw(0) << node->nb[0] << ", "
             << std::setw(5) << node->nb[1] << ", "
             << std::setw(5) << node->nb[2] << ", "
             << std::setw(5) << node->nb[3] << "] "
             << std::right << std::setw(15) << buf_name << std::right
             << "\n";

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (auto* src = node->src[i]) {
                // Get buffer type name for source
                const char * src_buf_name = "none";
                ggml_backend_buffer_t src_buf = src->view_src ? src->view_src->buffer : src->buffer;
                if (src_buf) {
                    src_buf_name = ggml_backend_buffer_name(src_buf);
                }

                file << std::setw(10) << " [ "
                << std::setw(5) << src->ne[0] << ", "
                << std::setw(5) << src->ne[1] << ", "
                << std::setw(5) << src->ne[2] << ", "
                << std::setw(5) << src->ne[3] << "] "
                << std::setw(12)
                << i << ": " << std::left << std::setw(12) << ggml_op_name(src->op) << std::right;
                file << std::left << std::setw(30) << src->name << std::right
                << std::setw(16) << "[ "
                << std::setw(0) << src->nb[0] << ", "
                << std::setw(5) << src->nb[1] << ", "
                << std::setw(5) << src->nb[2] << ", "
                << std::setw(5) << src->nb[3] << "] "
                << std::right << std::setw(15) << src_buf_name << std::right
                << "\n";
            }
        }
    }

    file << "n_leafs = " << cgraph->n_leafs << "\n";
    for (int i = 0; i < cgraph->n_leafs; i++) {
        ggml_tensor * node = cgraph->leafs[i];

        // Get buffer type name for leaf
        const char * leaf_buf_name = "none";
        ggml_backend_buffer_t leaf_buf = node->view_src ? node->view_src->buffer : node->buffer;
        if (leaf_buf) {
            leaf_buf_name = ggml_backend_buffer_name(leaf_buf);
        }

        file << " - " << std::setw(3) << i << ": [ "
             << std::setw(5) << node->ne[0] << ", "
             << std::setw(5) << node->ne[1] << "] "
             << std::setw(8) << ggml_op_name(node->op) << " "
             << std::setw(16) << ggml_get_name(node)
             << std::setw(20) << leaf_buf_name << "\n";
    }
    // clang-format on
    file << "========================================\n";

    file.close();
}

void print_tensor_address_map(const ggml_cgraph * cgraph) {
    std::map<void *, std::vector<std::string>> address_map;
    for (int node_n = 0; node_n < cgraph->n_nodes; node_n++) {
        auto * node = cgraph->nodes[node_n];
        if (node->data) {
            auto it = address_map.find(node->data);
            if (it == address_map.end()) {
                address_map[node->data] = std::vector<std::string>();
            }
            address_map[node->data].push_back(node->name);
        }
    }
    for (const auto & pair : address_map) {
        std::cout << "Address: " << pair.first << '\n';
        for (const auto & name : pair.second) {
            std::cout << name << " ; ";
        }
        std::cout << "\n\n";
    }
}

ov::Shape GgmlOvDecoder::get_shape(const ggml_tensor * tensor) {
    std::vector<size_t> shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    return shape;
}

std::vector<size_t> GgmlOvDecoder::get_stride(const ggml_tensor * tensor) {
    std::vector<size_t> stride;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        stride.push_back(static_cast<size_t>(tensor->nb[i]));
    }
    return stride;
}

ov::element::Type GgmlOvDecoder::get_ov_type(const ggml_tensor * tensor) {
    switch (tensor->type) {
    case GGML_TYPE_F64:
        return ov::element::f64;
    case GGML_TYPE_F32:
        return ov::element::f32;
    case GGML_TYPE_F16:
        return ov::element::f16;
    case GGML_TYPE_BF16:
        return ov::element::bf16;
    case GGML_TYPE_I8:
        return ov::element::i8;
    case GGML_TYPE_I16:
        return ov::element::i16;
    case GGML_TYPE_I32:
        return ov::element::i32;
    case GGML_TYPE_I64:
        return ov::element::i64;
    default:
        return ov::element::dynamic;
    }
}

ov::PartialShape GgmlOvDecoder::get_input_shape(int node_idx, const std::string & name) const {
    // Translators use this as the static GGML shape metadata. The actual OpenVINO
    // tensors carry dynamic extents from Parameters created with get_graph_input_shape().
    return ov::PartialShape(get_shape(m_node_info_list[node_idx].node_inputs.at(name)));
}

size_t GgmlOvDecoder::get_input_size() const {
    if (m_node_idx >= 0) {
        return get_input_size(m_node_idx);
    }
    return m_model_inputs.size();
}

size_t GgmlOvDecoder::get_input_size(int node_idx) const {
    return get_input_names(node_idx).size();
}

std::vector<std::string> GgmlOvDecoder::get_input_names(int node_idx) const {
    const auto & info = m_node_info_list[node_idx];
    const auto * node = info.node;
    auto inputs = info.node_inputs_names;
    if (node->op == GGML_OP_SCALE && info.node_op_case == 1 && m_model_inputs.count("cache_rs_reset_idx")) {
        inputs[0] = get_tensor_ov_name(m_cgraph, node->view_src);
        inputs.push_back("cache_rs_reset_idx");
        inputs.push_back("cache_rs_reset_len");
    } else if (node->op == GGML_OP_CPY && info.node_op_case >= 1 && info.node_op_case <= 3 &&
               m_model_inputs.count("rs_slot_begin_" + info.node_name)) {
        if (info.node_op_case == 1 || info.node_op_case == 2) {
            inputs[0] = get_tensor_ov_name(m_cgraph, node->src[0]->src[0]);
        }
        inputs[1] = get_tensor_ov_name(m_cgraph, node->view_src);
        inputs.push_back("rs_slot_begin_" + info.node_name);
        if (info.node_op_case == 2) {
            inputs.push_back("rs_src_begin_" + info.node_name);
        }
    } else if (node->op == GGML_OP_CPY && info.node_op_case >= 4 && info.node_op_case <= 6) {
        // Flat-buffer writes (conv_states_all, whisper KV): update the whole buffer from a runtime element offset.
        inputs[1] = get_tensor_ov_name(m_cgraph, node->view_src);
        if (m_model_inputs.count("rs_slot_begin_" + info.node_name)) {
            inputs.push_back("rs_slot_begin_" + info.node_name);
        }
    }
    inputs.insert(inputs.end(), info.kv_view_inputs.begin(), info.kv_view_inputs.end());
    return inputs;
}

ov::PartialShape GgmlOvDecoder::get_output_shape(int node_idx) const {
    auto * ggml_tensor = m_node_info_list[node_idx].node;
    // Keep translator-facing output metadata static. Shape polymorphism is already
    // represented by the graph's dynamic input Parameters and propagated by OV ops.
    return ov::PartialShape(get_shape(ggml_tensor));
}

std::vector<std::string> GgmlOvDecoder::get_output_names(int node_idx) const {
    return {m_node_info_list[node_idx].node_name};
}

std::string GgmlOvDecoder::get_inplace_op_src(int node_idx) const {
    auto * node = m_node_info_list[node_idx].node;
    if (!::is_inplace_op(node) || node->view_src == nullptr || ggml_nbytes(node) == 0) {
        return "";
    }
    const int op_case = m_node_info_list[node_idx].node_op_case;
    if (node->op == GGML_OP_CPY && (op_case == 1 || op_case == 2 || op_case == 3) &&
        m_compute_params.s_copy_active_slot_len == -1) {
        return "";
    }
    return get_tensor_ov_name(m_cgraph, node->view_src);
}

bool GgmlOvDecoder::is_view_like_alias_of(int node_idx, const std::string & view_src_name) const {
    auto * node = m_node_info_list[node_idx].node;
    if (node->view_src == nullptr || get_tensor_ov_name(m_cgraph, node->view_src) != view_src_name) {
        return false;
    }
    return node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW;
}

const std::string & GgmlOvDecoder::get_op_name() const {
    return m_node_idx >= 0 ? get_op_name(m_node_idx) : m_node_info_list.front().node_name;
}

int32_t GgmlOvDecoder::get_op_dynamic_dim(int node_idx) const {
    auto it = m_node_dynamic_dims.find(m_node_info_list[node_idx].node);
    if (it == m_node_dynamic_dims.end()) {
        return -1;
    }
    return it->second;
}

const std::string & GgmlOvDecoder::get_op_name(int node_idx) const {
    return m_node_info_list[node_idx].node_name;
}

void GgmlOvDecoder::visit_subgraph(std::function<void(std::shared_ptr<ov::frontend::gguf::GgufDecoder>)> node_visitor) const {
    // Copy the decoder's (heavy) model-scope state once, then rebind it to each node by
    // advancing m_node_idx. translate_node consumes the bound decoder synchronously and never
    // retains it past the call, so a single reused instance is safe -- and it keeps the whole
    // model walk O(n_nodes) instead of deep-copying every map per node.
    auto decoder = std::make_shared<GgmlOvDecoder>(*this);
    for (int node_idx = 0; node_idx < m_cgraph->n_nodes; node_idx++) {
        decoder->m_node_idx = node_idx;
        node_visitor(decoder);
    }
}

ov::Any GgmlOvDecoder::get_attribute(const std::string & name) const {
    const ggml_tensor * node = m_node_idx >= 0 ? m_node_info_list[m_node_idx].node : nullptr;
    if (name == "rope_config") {
        return rope_config_from_params(node != nullptr && node->op == GGML_OP_ROPE ? node->op_params : m_model_params.rope_params,
                                       m_model_params.mixed_rope_params);
    }
    if (name == "swa_window_size") {
        return m_compute_params.swa_window;
    }
    if (node == nullptr) {
        return ov::Any{};
    }
    const int op_case = m_node_info_list[m_node_idx].node_op_case;

    const bool recurrent_copy = node->op == GGML_OP_CPY && op_case >= 1 && op_case <= 3 &&
                                m_model_inputs.count("rs_slot_begin_" + get_op_name(m_node_idx));
    switch (node->op) {
        case GGML_OP_NONE: {
            if (name == "data" || name == "quant_type") {
                const auto node_name = get_tensor_ov_name(m_cgraph, node);
                if (m_model_weights.find(node_name) != m_model_weights.end()) {
                    if (name == "data") {
                        const size_t nbytes = ggml_nbytes(node);
                        return ov::Tensor(ov::element::u8, ov::Shape{nbytes}, node->data);
                    }
                    return std::string(ggml_type_name(node->type));
                }
            }
            break;
        }

        case GGML_OP_SCALE:
            if (name == "axis" && op_case == 1 && m_model_inputs.count("cache_rs_reset_idx")) return int64_t{2};
            if (name == "scale") return op_param_float(node, 0);
            if (name == "bias") return op_param_float(node, 1);
            break;

        case GGML_OP_CPY:
            // A full single-slot recurrent-state writeback (op_case 7: GDN state, 8: conv state) replaces the
            // whole flat cache; it is expressed as an update at offset 0 so the cache storage is rebound.
            if (name == "axis") {
                if (recurrent_copy) return int64_t{2};
                if (op_case >= 4 && op_case <= 8) return int64_t{3};
            }
            if (name == "destination_offset" && op_case >= 4 && op_case <= 6) {
                return static_cast<int64_t>(node->view_offs / ggml_type_size(node->type));
            }
            if (name == "destination_offset" && (op_case == 7 || op_case == 8)) {
                return int64_t{0};
            }
            if (name == "source_slice" && recurrent_copy) {
                if (op_case == 1) {
                    // The state tail of the GATED_DELTA_NET result holds exactly the copied elements, in rows
                    // of the result's packed width (S_v * H).
                    const ggml_tensor * gdn = node->src[0]->src[0];
                    OPENVINO_ASSERT(gdn->ne[0] > 0 && ggml_nelements(node) % gdn->ne[0] == 0,
                                    "GDN copy must cover whole packed rows");
                    return std::vector<int64_t>{2, -ggml_nelements(node) / gdn->ne[0], -1};
                }
                if (op_case == 2) return std::vector<int64_t>{3, 0, node->src[0]->ne[0]};
            }
            if (name == "reshape_target") {
                if (recurrent_copy && op_case <= 2) return std::vector<int64_t>{1, 1, -1, node->ne[0]};
                if (op_case >= 4 && op_case <= 8) return std::vector<int64_t>{1, 1, 1, -1};
            }
            break;

        case GGML_OP_RESHAPE:
            if (name == "reshape_target") {
                const auto shape = get_shape(node);
                std::vector<int64_t> target(shape.begin(), shape.end());
                const int dynamic_dim = m_naive ? -1 : get_op_dynamic_dim(m_node_idx);
                if (dynamic_dim >= 0) target[GGML_MAX_DIMS - 1 - dynamic_dim] = -1;
                // The op_case axis is inferred unless the dynamic axis already is (a Reshape takes one -1).
                const bool has_inferred = dynamic_dim >= 0;
                if (op_case == 1) {
                    target[0] = 0;
                    if (!has_inferred) target[1] = -1;
                } else if (op_case == 2) {
                    target[0] = 0;
                    if (!has_inferred) target[2] = -1;
                }
                return target;
            }
            if (name == "special_zero") return op_case == 1 || op_case == 2;
            break;

        case GGML_OP_CONT:
            if (name == "cont_reshape" && !ggml_are_same_shape(node, node->src[0])) {
                const auto shape = get_shape(node);
                std::vector<int64_t> target(shape.begin(), shape.end());
                const int dynamic_dim = m_naive ? -1 : get_op_dynamic_dim(m_node_idx);
                if (dynamic_dim >= 0) target[GGML_MAX_DIMS - 1 - dynamic_dim] = -1;
                return target;
            }
            [[fallthrough]];
        case GGML_OP_VIEW:
            if (name == "kv_view" && !m_node_info_list[m_node_idx].kv_view.empty()) {
                return m_node_info_list[m_node_idx].kv_view;
            }
            if (name == "view_slice" || name == "view_reshape") {
                auto description = describe_view(node, name == "view_reshape");
                if (!description.empty()) return description;
            }
            if (name == "output_dynamic_axis") {
                const int axis = m_naive ? -1 : get_op_dynamic_dim(m_node_idx);
                return axis < 0 ? -1 : GGML_MAX_DIMS - 1 - axis;
            }
            break;

        case GGML_OP_ROPE:
            if (name == "rope_mode") {
                switch (node->op_params[2]) {
                    case GGML_ROPE_TYPE_NORMAL: return std::string("normal");
                    case GGML_ROPE_TYPE_NEOX: return std::string("neox");
                    case GGML_ROPE_TYPE_IMROPE: return std::string("imrope");
                    case GGML_ROPE_TYPE_MROPE: return std::string("mrope");
                    case GGML_ROPE_TYPE_VISION: return std::string("vision");
                    default: throw std::runtime_error("Unsupported GGML RoPE mode: " + std::to_string(node->op_params[2]));
                }
            }
            break;

        case GGML_OP_RMS_NORM:
        case GGML_OP_NORM:
        case GGML_OP_L2_NORM:
            if (name == "eps") return op_param_float(node, 0);
            break;

        case GGML_OP_SOFT_MAX:
            if (name == "scale") return op_param_float(node, 0);
            if (name == "max_bias") return op_param_float(node, 1);
            break;

        case GGML_OP_FLASH_ATTN_EXT:
            if (name == "scale") return op_param_float(node, 0);
            if (name == "max_bias") return op_param_float(node, 1);
            if (name == "kq_soft_cap") return op_param_float(node, 2);
            if (name == "has_mask") return node->src[3] != nullptr;
            // Stateful graphs keep the fused SDPA, which the stateful-SDPA fusion expects.
            if (name == "manual_gqa_attention") return !m_is_stateful && ggml_openvino_manual_gqa_attn_enabled();
            break;

        case GGML_OP_GLU:
            // GLU family (SWIGLU / GEGLU / SWIGLU_OAI): op_params[1]=swapped, [2]=alpha, [3]=limit.
            if (name == "swapped") return node->op_params[1] != 0;
            if (name == "glu_alpha") return op_param_float(node, 2);
            if (name == "glu_limit") return op_param_float(node, 3);
            break;

        case GGML_OP_ARGSORT:
            if (name == "sort_order") return node->op_params[0];
            break;

        case GGML_OP_CLAMP:
            if (name == "clamp_min") return op_param_float(node, 0);
            if (name == "clamp_max") return op_param_float(node, 1);
            break;

        case GGML_OP_CONCAT:
            if (name == "concat_axis") return node->op_params[0];
            break;

        case GGML_OP_FILL:
            if (name == "fill_value") return op_param_float(node, 0);
            break;

        case GGML_OP_TOP_K:
            if (name == "k") return static_cast<int64_t>(node->ne[0]);
            break;

        case GGML_OP_REPEAT:
            if (name == "repeats") {
                OPENVINO_ASSERT(node->src[0] != nullptr, "REPEAT requires a source operand");
                const auto input_shape = get_shape(node->src[0]);
                const auto output_shape = get_shape(node);
                std::vector<int64_t> repeats(input_shape.size(), 1);
                for (size_t axis = 0; axis < repeats.size(); ++axis) {
                    OPENVINO_ASSERT(input_shape[axis] > 0 && output_shape[axis] % input_shape[axis] == 0,
                                    "REPEAT output dimensions must be integer multiples of the input dimensions");
                    repeats[axis] = static_cast<int64_t>(output_shape[axis] / input_shape[axis]);
                }
                const auto dynamic_dim = get_op_dynamic_dim(m_node_idx);
                if (dynamic_dim >= 0) {
                    const size_t dynamic_axis = repeats.size() - 1 - static_cast<size_t>(dynamic_dim);
                    if (repeats[dynamic_axis] != 1) return ov::Any{};
                }
                return repeats;
            }
            break;

        case GGML_OP_PERMUTE:
            if (name == "perm") return permute_order_from_params(node->op_params);
            break;

        case GGML_OP_IM2COL:
            if (name == "im2col_params") return std::vector<int32_t>(node->op_params, node->op_params + 7);
            break;

        case GGML_OP_IM2COL_3D:
            if (name == "im2col_params") return std::vector<int32_t>(node->op_params, node->op_params + 10);
            break;

        case GGML_OP_CONV_2D_DW:
            if (name == "conv_params") return std::vector<int64_t>(node->op_params, node->op_params + 6);
            break;

        case GGML_OP_CONV_TRANSPOSE_1D:
            if (name == "conv_params") return std::vector<int64_t>(node->op_params, node->op_params + 3);
            break;

        case GGML_OP_CONV_TRANSPOSE_2D:
            if (name == "conv_params") return std::vector<int64_t>(node->op_params, node->op_params + 1);
            break;

        case GGML_OP_CONV_3D:
            if (name == "conv_params") return std::vector<int64_t>(node->op_params, node->op_params + 12);
            break;

        case GGML_OP_UPSCALE:
            if (name == "interpolation_mode") return node->op_params[0];
            if (name == "upscale_sizes") return std::vector<int64_t>{node->ne[1], node->ne[0]};
            break;

        case GGML_OP_PAD:
            if (name == "pad_params") return std::vector<int32_t>(node->op_params, node->op_params + 8);
            if (name == "pad_circular") return node->op_params[8] != 0;
            break;

        case GGML_OP_TRI:
            if (name == "tri_type") return node->op_params[0];
            break;

        case GGML_OP_POOL_2D:
            // {op, k0, k1, s0, s1, p0, p1}
            if (name == "pool_params") return std::vector<int32_t>(node->op_params, node->op_params + 7);
            break;

        case GGML_OP_ROLL:
            if (name == "roll_shifts") return std::vector<int32_t>(node->op_params, node->op_params + 4);
            break;

        case GGML_OP_SET:
            if (name == "set_offset_elems") {
                return static_cast<int64_t>(static_cast<size_t>(reinterpret_cast<const uint32_t *>(node->op_params)[3]) /
                                            ggml_type_size(node->type));
            }
            break;

        default:
            break;
    }

    if (name == "output_type" || name == "dst_type") return get_ov_type(node);
    if (name == "is_swa") {
        if (auto layer = extract_layer_from_name(node->name); layer.has_value()) return is_swa_layer(layer.value()) != 0;
        return false;
    }
    if (name == "view_offset_bytes") {
        size_t offset = node->view_offs;
        if (node->op == GGML_OP_VIEW) {
            // view_offs is relative to the storage root; op_params retains the src[0]-relative offset.
            memcpy(&offset, node->op_params, sizeof(offset));
        }
        return static_cast<int64_t>(offset);
    }
    if (name == "view_input_strides" && node->src[0] != nullptr) return std::vector<int64_t>{(int64_t) node->src[0]->nb[0], (int64_t) node->src[0]->nb[1], (int64_t) node->src[0]->nb[2], (int64_t) node->src[0]->nb[3]};
    if (name == "view_output_strides") return std::vector<int64_t>{(int64_t) node->nb[0], (int64_t) node->nb[1], (int64_t) node->nb[2], (int64_t) node->nb[3]};
    if (name == "input_ggml_shape" && node->src[0] != nullptr) return get_shape(node->src[0]);
    if (name == "gdn_state_slots" && node->op == GGML_OP_GATED_DELTA_NET) return static_cast<int64_t>(node->op_params[0]);
    if (name == "force_ref") return false;
    return ov::Any{};
}

ov::PartialShape GgmlOvDecoder::get_input_shape(const std::string & name) const {
    return get_input_shape(m_node_idx, name);
}

int64_t GgmlOvDecoder::get_input_view_element_offset(const std::string & name) const {
    if (m_node_idx < 0) return 0;
    const auto * view = m_node_info_list[m_node_idx].node_inputs.at(name);
    if (view->op != GGML_OP_VIEW) return 0;
    return static_cast<int64_t>(view->view_offs / ggml_type_size(view->type));
}

std::vector<std::string> GgmlOvDecoder::get_input_names() const {
    return get_input_names(m_node_idx);
}

ov::PartialShape GgmlOvDecoder::get_output_shape() const {
    return get_output_shape(m_node_idx);
}

std::vector<std::string> GgmlOvDecoder::get_output_names() const {
    return get_output_names(m_node_idx);
}

std::string GgmlOvDecoder::compute_op_type(const ggml_tensor * node) {
    switch (node->op) {
    case GGML_OP_UNARY:
        return std::string("GGML_UNARY_OP_") + ggml_unary_op_name(ggml_get_unary_op(node));
    case GGML_OP_GLU:
        return std::string("GGML_GLU_OP_") + ggml_glu_op_name(ggml_get_glu_op(node));
    default:
        return std::string("GGML_OP_") + ggml_op_name(node->op);
    }
}

const std::string & GgmlOvDecoder::get_op_type(int node_idx) const {
    return m_node_info_list[node_idx].node_op_type;
}

const std::string & GgmlOvDecoder::get_op_type() const {
    if (m_node_idx >= 0) {
        return get_op_type(m_node_idx);
    }
    static const std::string unknown_op = "UNKNOWN_GGML_OP";
    return unknown_op;
}

std::vector<int64_t> GgmlOvDecoder::describe_view(const ggml_tensor * node, bool reshape) const {
    const ggml_tensor * src = node->src[0];
    if (node->op != GGML_OP_VIEW || src == nullptr) {
        return {};
    }
    size_t offset = 0;
    memcpy(&offset, node->op_params, sizeof(offset));  // relative to src[0]
    const int dynamic_src = [&] {
        auto it = m_node_dynamic_dims.find(const_cast<ggml_tensor *>(src));
        return m_naive || it == m_node_dynamic_dims.end() ? -1 : it->second;
    }();

    // GATED_DELTA_NET packs [attention rows | recurrent-state rows] along ggml axis 1. The attention
    // view keeps everything ahead of the trailing state rows, whose count is fixed while the token
    // count is not; the state view is that fixed-size tail.
    if (src->op == GGML_OP_GATED_DELTA_NET) {
        // ggml_gated_delta_net: K snapshot slots (op_params[0]) of S_v rows per sequence.
        const int64_t state_rows = static_cast<int64_t>(src->op_params[0]) * src->src[2]->ne[0] * src->src[2]->ne[3];
        if (offset == 0) {
            // [n_seqs, tokens, heads, head_size]; n_seqs is fixed for the converted model (same_graph_extents).
            if (reshape) return {node->ne[3], -1, node->ne[1], node->ne[0]};
            return {2, 0, -state_rows};
        }
        if (reshape) return {node->ne[3], node->ne[2], node->ne[1], node->ne[0]};
        return {2, -state_rows, state_rows};
    }

    // A pure slice: same strides as the source and exactly one axis shortened. Expressing it as an
    // explicit slice keeps a dynamic source axis dynamic, which the strided fallback cannot do.
    if (reshape) {
        return {};
    }
    int axis = -1;
    for (int i = 0; i < GGML_MAX_DIMS; i++) {
        if (node->nb[i] != src->nb[i]) {
            return {};
        }
        if (node->ne[i] != src->ne[i]) {
            if (axis != -1) {
                return {};
            }
            axis = i;
        }
    }
    if (axis == -1 || ggml_is_quantized(src->type) || node->nb[axis] == 0 || offset % node->nb[axis] != 0) {
        return {};
    }
    int64_t start = static_cast<int64_t>(offset / node->nb[axis]);
    const int64_t length = node->ne[axis];
    // Anything inside the offset must be a whole number of steps along the sliced axis only.
    if (start + length > src->ne[axis] || offset >= node->nb[axis] * static_cast<size_t>(src->ne[axis])) {
        return {};
    }
    if (dynamic_src == axis && start > 0) {
        // A window of the dynamic axis is only expressible when it ends at the axis end.
        if (start + length != src->ne[axis]) {
            return {};
        }
        start = -length;
    }
    return {GGML_MAX_DIMS - 1 - axis, start, length};
}

// Attention reads a KV cache through a VIEW whose live length (n_kv) and first stream (sinfo.s0) move with
// the batch while the converted model is reused. Describe the read for the frontend's "kv_view" and name the
// runtime operands that carry those values. Layouts (ggml order):
//   llama.cpp K/V:            cache [F, cells, streams], view [D, H, n_kv, ns]
//   llama.cpp V, `-fa off`:   cache [F, cells, streams], view [n_kv, H, D, ns], cells innermost
//   whisper flat buffer:      cache [N],                 view [D, n_kv, H] at a whole-row offset
std::pair<std::vector<int64_t>, std::vector<std::string>> GgmlOvDecoder::describe_kv_view(
    const ggml_tensor * node, bool & transposed) const {
    transposed = false;
    if (m_naive || m_is_static || m_is_stateful || node->op != GGML_OP_VIEW) {
        return {};
    }
    const ggml_tensor * cache = node->src[0];
    if (cache == nullptr || cache->op != GGML_OP_NONE || cache->view_src != nullptr || !is_kvcache(cache, nullptr) ||
        ggml_is_quantized(cache->type)) {
        return {};
    }
    // Only attention reads qualify; recurrent state caches share the buffer type but are read by
    // SCALE / CPY / GET_ROWS.
    bool consumed = false;
    bool masked_attention = false;
    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        const ggml_tensor * user = m_cgraph->nodes[i];
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            if (user->src[j] != node) {
                continue;
            }
            if (user->op != GGML_OP_PERMUTE && user->op != GGML_OP_FLASH_ATTN_EXT && user->op != GGML_OP_MUL_MAT) {
                return {};
            }
            consumed = true;
            masked_attention |= user->op == GGML_OP_FLASH_ATTN_EXT && user->src[3] != nullptr;
        }
    }
    if (!consumed) {
        return {};
    }

    size_t offset = 0;
    memcpy(&offset, node->op_params, sizeof(offset));  // relative to src[0]
    const int64_t es = static_cast<int64_t>(ggml_type_size(cache->type));
    const int64_t * ne = node->ne;
    const size_t * nb = node->nb;
    const auto bytes = [&](int64_t elements) { return static_cast<size_t>(elements * es); };

    if (cache->ne[1] == 1 && cache->ne[2] == 1 && cache->ne[3] == 1) {
        if (ne[3] != 1 || nb[0] != bytes(1) || nb[2] != bytes(ne[0]) || nb[1] % es != 0) {
            return {};
        }
        const int64_t width = static_cast<int64_t>(nb[1]) / es;
        if (width != ne[0] * ne[2] || offset % bytes(width) != 0 || cache->ne[0] % width != 0) {
            return {};
        }
        std::vector<int64_t> d{1, width, ne[2], ne[0], 1, static_cast<int64_t>(offset / bytes(width)), 0, 0, 2, 1, 3,
                               ne[1]};
        // Self-attention grows with the decoded tokens; cross-attention reads the fixed encoder length.
        if (masked_attention && m_compute_params.attention_size == ne[1]) {
            return {d, {"attention_size"}};
        }
        return {d, {}};
    }

    const int64_t width = cache->ne[0];
    const int64_t cells = cache->ne[1];
    const int64_t streams = cache->ne[2];
    const size_t stream_bytes = bytes(width * cells);
    if (cache->ne[3] != 1 || offset % stream_bytes != 0 || (ne[3] != 1 && nb[3] != stream_bytes)) {
        return {};
    }
    std::vector<int64_t> d;
    int64_t length = 0;
    if (nb[0] == bytes(1) && nb[1] == bytes(ne[0]) && nb[2] == bytes(width) && ne[0] * ne[1] <= width &&
        ne[2] <= cells) {
        d = {streams, width, ne[1], ne[0], ne[3], 0, 0, 0, 1, 2, 3, ne[2]};
        length = ne[2];
    } else if (nb[0] == bytes(1) && nb[2] == bytes(cells) && nb[1] == bytes(ne[2] * cells) && ne[1] * ne[2] <= width &&
               ne[0] <= cells) {
        d = {streams, width, ne[1], ne[2], ne[3], 0, 1, 0, 3, 2, 1, ne[0]};
        length = ne[0];
        transposed = true;
    } else {
        return {};
    }
    const int64_t first_stream = static_cast<int64_t>(offset / stream_bytes);
    if (streams > 1 && first_stream != m_compute_params.seq_active_start) {
        return {};
    }
    const auto layer = extract_layer_from_name(cache->name);
    const bool swa = layer.has_value() && is_swa_layer(layer.value());
    const int live_length = swa ? m_compute_params.attention_size_swa : m_compute_params.attention_size;
    // The live length must be the runtime value the mask describes; a fixed length would be frozen into a
    // model that is reused across decode steps, so leave such a view to the generic path.
    if (live_length != length) {
        return {};
    }
    std::vector<std::string> inputs{swa ? "attention_size_swa" : "attention_size"};
    if (streams > 1) {
        inputs.push_back("seq_active_start");
    }
    return {d, inputs};
}

void GgmlOvDecoder::compute_kv_views() {
    m_transposed_kv_caches.clear();
    for (auto & info : m_node_info_list) {
        bool transposed = false;
        auto description = describe_kv_view(info.node, transposed);
        if (transposed) {
            m_transposed_kv_caches.insert(info.node->src[0]);
        }
        info.kv_view = std::move(description.first);
        info.kv_view_inputs = std::move(description.second);
        if (info.kv_view.empty()) {
            continue;
        }
        for (const auto & name : info.kv_view_inputs) {
            m_model_inputs[name] = {ov::element::i64, ov::PartialShape{1}};
        }
    }
}

void GgmlOvDecoder::compute_node_dynamic_dims() {
    auto visit_node = [&](auto && self, ggml_tensor * node) -> void {
        if (!node) {
            return;
        }

        if (node->op == GGML_OP_CPY) {
            m_node_dynamic_dims[node] = -1;
        }

        if (m_node_dynamic_dims.count(node)) {
            return;
        }
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            ggml_tensor * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            struct ggml_tensor * root_src = nullptr;
            // if (src->org_src) {
            //     root_src = src->org_src;
            // }
            if (root_src) {
                if (is_inp_tok(root_src, node) || is_inp_pos(root_src, node) || is_output_idx(root_src, node)) {
                    m_node_dynamic_dims[root_src] = 0;
                    m_node_dynamic_dims[src] = m_node_dynamic_dims[root_src];
                    continue;
                }
                self(self, root_src);
                m_node_dynamic_dims[src] = m_node_dynamic_dims[root_src];
            } else {
                if (is_inp_tok(src, node) || is_inp_pos(src, node) || is_output_idx(src, node)) {
                    m_node_dynamic_dims[src] = 0;
                    continue;
                }
                if (is_inp_scale_rows(src, node)) {
                    m_node_dynamic_dims[src] = 1;
                    continue;
                }
                if (node->op == GGML_OP_VIEW && src->op == GGML_OP_NONE && is_kvcache(src, node) &&
                    src->ne[1] > 1 && !m_model_is_splitted) {
                    m_node_dynamic_dims[src] = 1;
                    continue;
                }
                self(self, src);
            }
        }
        switch (node->op) {
        case GGML_OP_NONE:
            m_node_dynamic_dims[node] = -1;
            break;
        case GGML_OP_GET_ROWS:
            // out.ne = {src0.ne[0], idx.ne[0], idx.ne[1], idx.ne[2]}: index axis d lands on output axis d + 1.
            // A dynamic src0 only changes which rows exist, never the output extent.
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[1]] != -1) {
                const int idx_dyn = m_node_dynamic_dims[node->src[1]];
                if (idx_dyn < GGML_MAX_DIMS - 1) {
                    m_node_dynamic_dims[node] = idx_dyn + 1;
                }
            }
            break;
        case GGML_OP_MUL:
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[0]];
            }
            if (m_node_dynamic_dims[node->src[1]] != -1) {
                m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[1]];
            }
            break;
        case GGML_OP_MUL_MAT: {
            // out.ne = {src0.ne[1], src1.ne[1], src1.ne[2], src1.ne[3]}. Axis 0 of both operands is the
            // contracted axis and vanishes. src1 (activations / tokens) takes priority over src0 when both
            // carry a dynamic axis, matching the token-count axis the rest of the graph tracks.
            m_node_dynamic_dims[node] = -1;
            const int d0 = m_node_dynamic_dims[node->src[0]];
            const int d1 = m_node_dynamic_dims[node->src[1]];
            if (d1 >= 1) {
                m_node_dynamic_dims[node] = d1;
            } else if (d0 == 1) {
                m_node_dynamic_dims[node] = 0;
            } else if (d0 >= 2) {
                m_node_dynamic_dims[node] = d0;
            }
            break;
        }
        case GGML_OP_PERMUTE:
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                // ggml_permute stores op_params[i] = the result axis that source axis i moves to.
                const int dynamic_dim_idx = m_node_dynamic_dims[node->src[0]];
                m_node_dynamic_dims[node] = node->op_params[dynamic_dim_idx];
                // OPENVINO_ASSERT(dynamic_dim_value == node->ne[m_node_dynamic_dims[node]],
                //                 "Dynamic dim value mismatch for node: " + std::string(node->name) +
                //                     " and its src[0]: " + std::string(node->src[0]->name));
            }
            break;
        case GGML_OP_VIEW: {
            // Use stride-based matching: the stride of a VIEW dimension directly
            // encodes which source dimension it indexes into, so it uniquely
            // identifies the dynamic dim even when two dims share the same size.
            m_node_dynamic_dims[node] = -1;
            if (node->src[0]->op == GGML_OP_GATED_DELTA_NET) {
                // The attention rows are reshaped to [n_seq, tokens, heads, head_size]; the state tail is static.
                size_t view_offset = 0;
                memcpy(&view_offset, node->op_params, sizeof(view_offset));
                if (view_offset == 0 && m_node_dynamic_dims[node->src[0]] != -1) {
                    m_node_dynamic_dims[node] = 2;
                }
                break;
            }
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                auto dynamic_dim_idx = m_node_dynamic_dims[node->src[0]];
                auto dynamic_dim_value = node->src[0]->ne[dynamic_dim_idx];
                auto dynamic_dim_stride =
                    node->src[0]->nb[dynamic_dim_idx] / ggml_type_size(node->src[0]->type) * ggml_type_size(node->type);
                // Size-1 axes can share a stride (a graph captured with one token, or a single KV head);
                // prefer the matching axis that keeps the source extent, then one that is not size 1.
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (node->nb[i] != dynamic_dim_stride) {
                        continue;
                    }
                    if (m_node_dynamic_dims[node] == -1 ||
                        (node->ne[m_node_dynamic_dims[node]] == 1 && node->ne[i] > 1)) {
                        m_node_dynamic_dims[node] = i;
                    }
                    if (node->ne[i] == dynamic_dim_value) {
                        m_node_dynamic_dims[node] = i;
                        break;
                    }
                }
                if (m_node_dynamic_dims[node] != -1 && dynamic_dim_value != node->ne[m_node_dynamic_dims[node]] &&
                    !is_kvcache(node->src[0], node)) {
                    m_node_dynamic_dims[node] = -1;
                    // an empty view, e.g. the extra states of a single-slot recurrent cache, always mismatches
                    if (ggml_nelements(node) > 0) {
                        GGML_LOG_WARN("ggml-openvino: dynamic dim value mismatch for VIEW node '%s', src[0]: '%s'\n",
                                      node->name, node->src[0]->name);
                    }
                }
            }
            break;
        }
        case GGML_OP_TRANSPOSE:
        case GGML_OP_RESHAPE: {
            if (is_same_shape(node->src[0], node)) {
                m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[0]];
                break;
            }
            // RESHAPE requires src[0] to be contiguous, so both src and result
            // have standard compact strides: nb[i] = type_size * prod(ne[0..i-1]).
            // Match src->nb[dynamic_dim] against result->nb[i] to find the output
            // dimension whose flat-memory boundary aligns with the source dynamic
            // boundary. This is unambiguous (result strides are strictly monotone)
            // and handles merged-lower-dim cases that ne-value matching misses.
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                auto dynamic_dim_idx = m_node_dynamic_dims[node->src[0]];
                auto dynamic_dim_stride = node->src[0]->nb[dynamic_dim_idx];
                for (int i = 0; i < GGML_MAX_DIMS; i++) {
                    if (node->nb[i] == dynamic_dim_stride && node->ne[i] == node->src[0]->ne[dynamic_dim_idx]) {
                        m_node_dynamic_dims[node] = i;
                        break;
                    }
                }
                if (m_node_dynamic_dims[node] == -1) {
                    GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for RESHAPE node '%s'\n", node->name);
                }
            }
            break;
        }
        case GGML_OP_FLASH_ATTN_EXT: {
            // Output shape is hard-coded in ggml_flash_attn_ext as:
            //   ne = { v->ne[0], q->ne[2], q->ne[1], q->ne[3] }
            // i.e. output dim 0 <- v dim 0 (head_size, static)
            //      output dim 1 <- q dim 2 (n_heads,   static)
            //      output dim 2 <- q dim 1 (n_tokens,  potentially dynamic)
            //      output dim 3 <- q dim 3 (batch,     static)
            // Using the fixed q-dim -> output-dim mapping table.
            // q is src[0]; the mapping from q's dynamic dim to the output dim is:
            //   q dim 1 -> output dim 2
            //   q dim 2 -> output dim 1
            //   q dim 3 -> output dim 3
            //   q dim 0 -> output dim 0  (head_size axis, unlikely to be dynamic)
            constexpr int q_to_out[GGML_MAX_DIMS] = {0, 2, 1, 3};
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                auto q_dynamic_dim = m_node_dynamic_dims[node->src[0]];
                m_node_dynamic_dims[node] = q_to_out[q_dynamic_dim];
            }
            break;
        }
        case GGML_OP_CONT:
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[0]] != -1) {
                auto dynamic_dim_idx = m_node_dynamic_dims[node->src[0]];
                if (ggml_are_same_shape(node, node->src[0])) {
                    m_node_dynamic_dims[node] = dynamic_dim_idx;
                } else {
                    size_t src_logical_nb[GGML_MAX_DIMS];
                    src_logical_nb[0] = ggml_type_size(node->src[0]->type);
                    src_logical_nb[1] = src_logical_nb[0] * (node->src[0]->ne[0] / ggml_blck_size(node->src[0]->type));
                    for (int i = 2; i < GGML_MAX_DIMS; i++) {
                        src_logical_nb[i] = src_logical_nb[i - 1] * node->src[0]->ne[i - 1];
                    }

                    auto dynamic_dim_stride = src_logical_nb[dynamic_dim_idx] / ggml_type_size(node->src[0]->type) *
                                              ggml_type_size(node->type);
                    int matched_dim_count = 0;
                    int first_matched_dim = -1;
                    for (int i = 0; i < GGML_MAX_DIMS; i++) {
                        if (node->nb[i] == dynamic_dim_stride && node->ne[i] == node->src[0]->ne[dynamic_dim_idx]) {
                            if (first_matched_dim == -1) {
                                first_matched_dim = i;
                            }
                            m_node_dynamic_dims[node] = i;
                            matched_dim_count++;
                        }
                    }
                    if (matched_dim_count > 1 && node->src[0]->ne[dynamic_dim_idx] == 1) {
                        // Single-token capture: every trailing dim is size 1 with the same stride, so
                        // the match is ambiguous. The lowest index is the real axis; the rest are
                        // ggml's size-1 padding. Bailing out here would bake the captured token count
                        // into the static prefill model, which then runs with a different one.
                        m_node_dynamic_dims[node] = first_matched_dim;
                    } else if (matched_dim_count != 1) {
                        m_node_dynamic_dims[node] = -1;
                        GGML_LOG_WARN("ggml-openvino: cannot determine dynamic dim for CONT node '%s', src[0]: '%s'\n",
                                      node->name, node->src[0]->name);
                    }
                }
            }
            break;
        case GGML_OP_CONCAT:
            m_node_dynamic_dims[node] = -1;
            for (int i = 0; i < GGML_MAX_DIMS; i++) {
                if (node->src[0]->ne[i] != node->ne[i]) {
                    m_node_dynamic_dims[node] = i;
                    break;
                }
            }
            break;
        case GGML_OP_SSM_CONV:
        case GGML_OP_GATED_DELTA_NET:
            m_node_dynamic_dims[node] = 1;
            break;
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
        case GGML_OP_NORM:
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_GLU:
        case GGML_OP_ROPE:
        case GGML_OP_SCALE:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_ARGSORT:
        case GGML_OP_ADD_ID:
        case GGML_OP_UNARY:
        case GGML_OP_CUMSUM:
        case GGML_OP_FILL:
        case GGML_OP_SET:
        case GGML_OP_DIAG:
        case GGML_OP_TRI:
        case GGML_OP_DUP:
        // Shape-preserving elementwise ops: the dynamic dim is unchanged from src[0].
        // DIV/CLAMP are used in the MoE routing-weight normalization
        // (sum_rows -> clamp -> div). If they are left untracked here the dynamic
        // (token) dim is lost there, the captured prefill token count gets baked into
        // the downstream reshapes, and every decoder layer after layer 0 turns static
        // (which then triggers the GPU in-place-concat KV-cache corruption).
        case GGML_OP_DIV:
        case GGML_OP_CLAMP:
        case GGML_OP_PAD:
        // Element-wise math (e.g. the squared-ReLU FFN uses SQR), ROLL and UPSCALE keep the dynamic extent.
        case GGML_OP_ADD1:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_LOG:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_ROLL:
        case GGML_OP_UPSCALE:
            m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[0]];
            break;
        case GGML_OP_TOP_K:
            // ne[0] becomes k; the other axes are kept.
            m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[0]] > 0 ? m_node_dynamic_dims[node->src[0]] : -1;
            break;
        case GGML_OP_POOL_2D:
            // Pooling resizes ggml axes 0 and 1 only.
            m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[0]] >= 2 ? m_node_dynamic_dims[node->src[0]] : -1;
            break;
        case GGML_OP_REPEAT:
            // REPEAT's output shape is supplied by src[1]. Keep a varying repeat factor as a
            // runtime shape operand instead of freezing the captured ratio into an attribute.
            if (node->src[1] != nullptr) {
                m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[1]];
            } else {
                // ggml_repeat_4d has no shape operand: the source's dynamic axis survives only if it is
                // not tiled.
                const int src_dim = m_node_dynamic_dims[node->src[0]];
                m_node_dynamic_dims[node] =
                    src_dim >= 0 && node->src[0]->ne[src_dim] == node->ne[src_dim] ? src_dim : -1;
            }
            break;
        case GGML_OP_SUM_ROWS:
            // SUM_ROWS reduces ggml axis 0 to size 1 and preserves all other axes, so the
            // dynamic dim is preserved unless it was axis 0 (then it is summed away).
            m_node_dynamic_dims[node] =
                (m_node_dynamic_dims[node->src[0]] == 0) ? -1 : m_node_dynamic_dims[node->src[0]];
            break;
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_SOLVE_TRI:
            m_node_dynamic_dims[node] = m_node_dynamic_dims[node->src[1]];
            break;
        case GGML_OP_CPY:
        case GGML_OP_SET_ROWS:
        case GGML_OP_SUM:
        case GGML_OP_MEAN:
            m_node_dynamic_dims[node] = -1;
            break;
        case GGML_OP_IM2COL: {
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[1]] != -1) {
                const bool is_2D = node->op_params[6] == 1;
                const int src_dyn = m_node_dynamic_dims[node->src[1]];
                if (is_2D) {
                    if (src_dyn == 0) {
                        m_node_dynamic_dims[node] = 1;  // IW -> OW
                    } else if (src_dyn == 1) {
                        m_node_dynamic_dims[node] = 2;  // IH -> OH
                    } else if (src_dyn == 3) {
                        m_node_dynamic_dims[node] = 3;  // N  -> N
                    }
                } else {
                    if (src_dyn == 0) {
                        m_node_dynamic_dims[node] = 1;  // IW -> OW
                    } else if (src_dyn == 2) {
                        m_node_dynamic_dims[node] = 2;  // N  -> N  (1D: b->ne[2] is the batch/channel dim)
                    }
                }
                if (m_node_dynamic_dims[node] != -1) {
                    OPENVINO_ASSERT(node->src[1]->ne[src_dyn] == node->ne[m_node_dynamic_dims[node]],
                                    "Dynamic dim value mismatch for IM2COL node: " + std::string(node->name) +
                                        " and its src[1]: " + std::string(node->src[1]->name));
                }
            }
            break;
        }
        case GGML_OP_IM2COL_3D: {
            m_node_dynamic_dims[node] = -1;
            if (m_node_dynamic_dims[node->src[1]] != -1) {
                const int src_dyn = m_node_dynamic_dims[node->src[1]];
                if (src_dyn == 0) {
                    m_node_dynamic_dims[node] = 1;  // IW -> OW
                } else if (src_dyn == 1) {
                    m_node_dynamic_dims[node] = 2;  // IH -> OH
                } else if (src_dyn == 3) {
                    m_node_dynamic_dims[node] = 3;  // N  -> N
                }
                if (m_node_dynamic_dims[node] != -1) {
                    OPENVINO_ASSERT(node->src[1]->ne[src_dyn] == node->ne[m_node_dynamic_dims[node]],
                                    "Dynamic dim value mismatch for IM2COL_3D node: " + std::string(node->name) +
                                        " and its src[1]: " + std::string(node->src[1]->name));
                }
            }
            break;
        }
        default:
            // Record "no tracked dynamic axis" explicitly. Consumers read this map with operator[], which
            // would otherwise insert 0 (= dynamic ggml axis 0) for an op that never wrote an entry.
            m_node_dynamic_dims[node] = -1;
            GGML_LOG_DEBUG("ggml-openvino: compute_node_dynamic_dims: unhandled op %s for node '%s'\n",
                           ggml_op_name(node->op), node->name);
            break;
        }
    };

    for (int i = 0; i < m_cgraph->n_nodes; i++) {
        ggml_tensor * node = m_cgraph->nodes[i];
        visit_node(visit_node, node);
    }

    // print the nodes in m_cgraph name & shape with the dynamic dim (the dynamic dim is the dimension with -1 in m_node_dynamic_dims) for debugging
    if (0) {
        for (int i = 0; i < m_cgraph->n_nodes; i++) {
            ggml_tensor * node = m_cgraph->nodes[i];
            int dynamic_dim = m_node_dynamic_dims[node];
            std::cout << "[" << i << "] " << "node_name: " << node->name << " op: " << ggml_op_name(node->op)
                      << " shape: [";
            for (int j = 0; j < 4; j++) {
                if (j == dynamic_dim) {
                    std::cout << "*";
                } else {
                    std::cout << node->ne[j];
                }
                if (j < 3) {
                    std::cout << ", ";
                }
            }
            std::cout << "]" << '\n';
            // print the src name & shape with the dynamic dim for debugging
            for (int j = 0; j < GGML_MAX_SRC; j++) {
                ggml_tensor * src = node->src[j];
                if (src == nullptr) {
                    continue;
                }
                int src_dynamic_dim = m_node_dynamic_dims[src];
                std::cout << "    [" << j << "] src_name: " << src->name << " [";
                for (int k = 0; k < 4; k++) {
                    if (k == src_dynamic_dim) {
                        std::cout << "*";
                    } else {
                        std::cout << src->ne[k];
                    }
                    if (k < 3) {
                        std::cout << ", ";
                    }
                }
                std::cout << "]" << '\n';
            }
            std::cout << '\n';
        }
    }
}
