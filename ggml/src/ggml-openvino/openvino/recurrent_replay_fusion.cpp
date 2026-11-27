#include "recurrent_replay_fusion.h"

#include "../ggml-openvino-extra.h"
#include "ggml-impl.h"
#include "op/recurrent_replay_graph.hpp"
#include "op_table.h"
#include "utils.h"

#include <openvino/op/constant.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reshape.hpp>

namespace ov::frontend::ggml {
namespace {
const ggml_tensor * cache_root(const ggml_tensor * node) {
    while (node && (node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE || node->op == GGML_OP_GET_ROWS)) {
        node = node->src[0];
    }
    return node && node->op == GGML_OP_NONE && GgmlOvDecoder::is_recurrent_cache(node) ? node : nullptr;
}
}  // namespace

RecurrentReplayFusion::RecurrentReplayFusion(const std::shared_ptr<GgmlDecoder> & decoder,
                                             const std::shared_ptr<TensorMap> & tensors) :
    m_decoder(decoder),
    m_tensors(tensors) {
    auto actual = std::dynamic_pointer_cast<GgmlOvDecoder>(decoder);
    if (!actual || actual->m_naive || decoder->is_static() || decoder->is_stateful() || decoder->is_splited_model() ||
        !ggml_openvino_is_gpu() || ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_FUSED_RECURRENCE") ||
        !tensors->count("rs_replay_rows")) {
        return;
    }
    m_graph = actual->get_cgraph();
    for (int i = 0; i < m_graph->n_nodes; ++i) {
        const auto node = m_graph->nodes[i];
        Group group;
        group.compute = i;
        if (node->op == GGML_OP_GATED_DELTA_NET && node->op_params[1] > 0) {
            for (int port : {5, 6, 7, 8}) {
                group.banks.push_back(cache_root(node->src[port]));
            }
        } else if (node->op == GGML_OP_SSM_CONV && node->src[0]->op == GGML_OP_CONT &&
                   node->src[0]->src[0]->op == GGML_OP_TRANSPOSE) {
            const auto cols = node->src[0]->src[0]->src[0];
            if (cols->op != GGML_OP_CONCAT || cols->op_params[0] != 1 || cols->src[0]->op != GGML_OP_GET_ROWS) {
                continue;
            }
            group.conv = true;
            group.current = cols->src[1];
            group.banks.push_back(cache_root(cols->src[0]->src[0]));
        } else {
            continue;
        }
        if (std::any_of(group.banks.begin(), group.banks.end(), [](const ggml_tensor * bank) {
                return !bank || bank->type != GGML_TYPE_F32 || bank->ne[2] != 1 || bank->ne[3] != 1;
            })) {
            continue;
        }
        auto is_bank = [&](const ggml_tensor * t) {
            return std::find(group.banks.begin(), group.banks.end(), t) != group.banks.end();
        };
        // Collect only cache-owned paths. Current-token producers remain outside this group.
        std::function<void(const ggml_tensor *)> collect = [&](const ggml_tensor * t) {
            if (!t || t->op == GGML_OP_NONE || t == group.current || is_bank(t) || !group.consumed.insert(t).second) {
                return;
            }
            for (int port = 0; port < (t->op == GGML_OP_GET_ROWS ? 1 : GGML_MAX_SRC); ++port) {
                if (t->src[port]) {
                    collect(t->src[port]);
                }
            }
        };
        if (group.conv) {
            collect(node->src[0]);
        } else {
            for (int port : {5, 6, 7, 8}) {
                collect(node->src[port]);
            }
        }
        std::map<const ggml_tensor *, int> resets, writes;
        for (int j = 0; j < m_graph->n_nodes; ++j) {
            const auto t = m_graph->nodes[j];
            if (!is_bank(t->view_src)) {
                continue;
            }
            if (GgmlOvDecoder::is_recurrent_cache_reset(t)) {
                float value;
                memcpy(&value, t->op_params, sizeof(value));
                if (value != 0) {
                    continue;
                }
                ++resets[t->view_src];
                collect(t);
            } else if (t->op == GGML_OP_CPY) {
                const auto src = t->src[0];
                const bool extra =
                    src->op == GGML_OP_VIEW && src->src[0]->op == GGML_OP_GET_ROWS && cache_root(src) == t->view_src;
                bool update = false;
                if (group.conv) {
                    update = src->op == GGML_OP_VIEW && src->src[0] == node->src[0]->src[0]->src[0];
                } else {
                    update = src->op == GGML_OP_VIEW && src->src[0] == node;
                    for (int port : {1, 2, 3, 4}) {
                        update |= src->op == GGML_OP_VIEW && src->src[0] == node->src[port];
                    }
                }
                if (!extra && !update) {
                    continue;
                }
                if (update) {
                    ++writes[t->view_src];
                }
                group.consumed.insert(t);
                collect(t->src[1]);
                group.consumed.insert(src);
                if (extra) {
                    collect(src);
                }
            }
        }
        bool complete = true;
        for (const auto bank : group.banks) {
            complete &= resets[bank] == 1 && writes[bank] == (bank == group.banks.back() && !group.conv ? 2 : 1);
        }
        group.consumed.insert(node);
        // A bank reader or writer outside this group would race with the implicit stores.
        for (int j = 0; j < m_graph->n_nodes && complete; ++j) {
            const auto t = m_graph->nodes[j];
            if (group.consumed.count(t)) {
                continue;
            }
            if (is_bank(t->view_src) && t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE) {
                complete = false;
            }
            for (int port = 0; port < GGML_MAX_SRC; ++port) {
                const auto src = t->src[port];
                if (!src || is_bank(src) == false) {
                    continue;
                }
                if (t->op != GGML_OP_VIEW && t->op != GGML_OP_RESHAPE) {
                    complete = false;
                }
            }
        }
        if (!complete) {
            continue;
        }
        const auto id = m_groups.size();
        for (int j = 0; j < m_graph->n_nodes; ++j) {
            if (group.consumed.count(m_graph->nodes[j])) {
                m_nodes[j] = id;
            }
        }
        m_groups.push_back(std::move(group));
    }
    if (ggml_openvino_getenv_int("GGML_OPENVINO_DUMP_CGRAPH")) {
        const auto conv =
            std::count_if(m_groups.begin(), m_groups.end(), [](const Group & group) { return group.conv; });
        GGML_LOG_INFO("ggml-openvino: replay fusion: %zu convolution, %zu GDN groups\n", size_t(conv),
                      m_groups.size() - conv);
    }
}

OutputVector RecurrentReplayFusion::translate(int index, const NodeContext & context) {
    const auto found = m_nodes.find(index);
    if (found == m_nodes.end()) {
        return {};
    }
    const auto & group = m_groups[found->second];
    using namespace ov::op;
    auto bank_input = [&](const ggml_tensor * t) -> Output<Node> {
        const auto name = GgmlOvDecoder::get_tensor_name(m_graph, t);
        const auto found = m_tensors->find(name);
        FRONT_END_OP_CONVERSION_CHECK(found != m_tensors->end(), "Replay input missing: ", name, " while translating ",
                                      context.get_name());
        return found->second;
    };
    if (index != group.compute) {
        const auto node = m_graph->nodes[index];
        auto root = cache_root(node);
        if (!root && node->view_src &&
            std::find(group.banks.begin(), group.banks.end(), node->view_src) != group.banks.end()) {
            root = node->view_src;
        }
        return OutputVector(context.get_output_names().size(), bank_input(root ? root : group.banks[0]));
    }
    OutputVector banks;
    for (const auto bank : group.banks) {
        auto input = bank_input(bank);
        auto source = input.get_node_shared_ptr();
        while (ov::is_type<v1::Reshape>(source)) {
            source = source->input_value(0).get_node_shared_ptr();
        }
        FRONT_END_OP_CONVERSION_CHECK(ov::is_type<v0::Parameter>(source),
                                      "Replay banks require writable Parameter storage");
        banks.push_back(input);
    }
    auto rows = context.get_input("rs_replay_rows"), control = context.get_input("rs_replay_control"),
         reset = context.get_input("rs_replay_reset");
    if (group.conv) {
        auto x = bank_input(group.current);
        const auto history = m_graph->nodes[index]->src[1]->ne[0] - 1;
        auto pattern =
            v0::Constant::create(element::i64, Shape{4}, std::vector<int64_t>{1, 1, group.current->ne[0], history + 1});
        auto weights = std::make_shared<v1::Reshape>(context.get_input(1), pattern, false);
        auto replay = ov::recurrent_replay::build({banks[0], rows, x, weights, control, reset}, true);
        m_updated_banks[GgmlOvDecoder::get_tensor_name(m_graph, group.banks[0])] = replay[1];
        return {replay[0]};
    }
    auto outputs = op::translate_recurrent_gated_delta_net_replay(
        context, {banks[0], banks[1], banks[2], banks[3], rows, control, reset});
    for (size_t i = 0; i < group.banks.size(); ++i) {
        m_updated_banks[GgmlOvDecoder::get_tensor_name(m_graph, group.banks[i])] = outputs[i + 1];
    }
    return {outputs[0], outputs[1]};
}
}  // namespace ov::frontend::ggml
