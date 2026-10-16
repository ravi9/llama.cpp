#ifdef GGML_OPENVINO_HAS_RECURRENT_CACHE
#    include "recurrent_cache_fusion.h"

#    include "../ggml-openvino-extra.h"
#    include "ggml-impl.h"
#    include "op/recurrent_cache.hpp"
#    include "utils.h"

#    include <openvino/op/concat.hpp>
#    include <openvino/op/constant.hpp>
#    include <openvino/op/convert.hpp>
#    include <openvino/op/reshape.hpp>
#    include <openvino/op/slice.hpp>

namespace ov::frontend::ggml {
RecurrentCacheFusion::RecurrentCacheFusion(const std::shared_ptr<GgmlDecoder> & decoder,
                                           const std::shared_ptr<TensorMap> & tensors) :
    m_decoder(decoder),
    m_tensors(tensors) {
    if (!ggml_openvino_is_gpu() || decoder->is_static() || decoder->is_stateful() || decoder->is_splited_model() ||
        ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_RECURRENT_CACHE")) {
        return;
    }
    auto actual = std::dynamic_pointer_cast<GgmlOvDecoder>(decoder);
    if (!actual || !tensors->count("cache_rs_reset_idx") || !tensors->count("cache_rs_reset_len") ||
        !tensors->count("s_copy_active_slot_len")) {
        return;
    }
    const auto graph = actual->get_cgraph();
    std::map<const ggml_tensor *, int> indices;
    for (int i = 0; i < graph->n_nodes; ++i) {
        indices[graph->nodes[i]] = i;
    }
    for (int i = 0; i < graph->n_nodes; ++i) {
        const auto cp = graph->nodes[i];
        if (cp->op != GGML_OP_CPY || decoder->get_op_case(i) != 3 || !cp->src[0] || cp->src[0]->op != GGML_OP_VIEW ||
            !cp->src[0]->src[0] || cp->src[0]->src[0]->op != GGML_OP_GET_ROWS) {
            continue;
        }
        const auto gather = cp->src[0]->src[0];
        const auto root = cp->view_src;
        if (!root || (root->type != GGML_TYPE_F32 && root->type != GGML_TYPE_F16) || root->ne[2] != 1 ||
            root->ne[3] != 1 || gather->src[0]->op != GGML_OP_RESHAPE || gather->src[0]->src[0] != root ||
            gather->src[1]->op != GGML_OP_NONE || gather->src[1]->type != GGML_TYPE_I32) {
            continue;
        }
        Group group;
        group.gather = indices.at(gather);
        group.remainder = i;
        for (int j = 0; j < group.gather; ++j) {
            const auto node = graph->nodes[j];
            if (node->op == GGML_OP_SCALE && node->view_src == root && decoder->get_op_case(j) == 1) {
                float scale, bias;
                memcpy(&scale, node->op_params, sizeof(float));
                memcpy(&bias, node->op_params + 1, sizeof(float));
                if (scale == 0 && bias == 0) {
                    group.reset = j;
                }
            }
        }
        if (group.reset < 0) {
            continue;
        }
        // Find the convolution CONCAT from the active prefix of this gather, then all
        // its cache snapshot writes. No tensor names or captured token counts are used.
        for (int j = i + 1; j < graph->n_nodes; ++j) {
            const auto node = graph->nodes[j];
            if (node->op != GGML_OP_CONCAT || node->op_params[0] != 0 || !node->src[0] ||
                node->src[0]->op != GGML_OP_RESHAPE) {
                continue;
            }
            const auto active = node->src[0]->src[0];
            if (active && active->op == GGML_OP_VIEW && active->src[0] == gather && active->view_offs == 0 &&
                node->src[0]->ne[3] == 1 && node->src[0]->ne[0] * node->src[0]->ne[1] == root->ne[0]) {
                group.concat = j;
                for (int k = j + 1; k < graph->n_nodes; ++k) {
                    const auto write = graph->nodes[k];
                    if (GgmlOvDecoder::is_conv_state_writeback(write) && write->view_src == root &&
                        write->src[0]->src[0] == node && write->src[0]->ne[0] == node->src[0]->ne[0] &&
                        write->src[0]->ne[1] == node->src[0]->ne[1]) {
                        group.snapshots.push_back(k);
                    }
                }
                break;
            }
        }
        if (group.concat >= 0 && group.snapshots.empty()) {
            continue;
        }
        bool complete = tensors->count("rs_slot_begin_" + decoder->get_op_name(i)) &&
                        tensors->count("rs_src_begin_" + decoder->get_op_name(i));
        for (int snapshot : group.snapshots) {
            const auto & name = decoder->get_op_name(snapshot);
            complete &= tensors->count("rs_slot_begin_" + name) && tensors->count("rs_src_begin_" + name);
        }
        const int last = group.snapshots.empty() ? i : group.snapshots.back();
        for (int j = group.reset + 1; j <= last && complete; ++j) {
            const auto node = graph->nodes[j];
            // An unrelated cache write cannot be crossed by this fusion. Cache aliases and
            // reads of the gather stay valid through the normal in-place refresh machinery.
            for (int source = 0; source < GGML_MAX_SRC; ++source) {
                const auto input = node->src[source];
                if (!input || (input != root && input->view_src != root)) {
                    continue;
                }
                const bool alias = node->op == GGML_OP_VIEW || node->op == GGML_OP_RESHAPE;
                const bool matched_read = j == group.gather;
                const bool destination = node->op == GGML_OP_CPY && source == 1;
                if (!alias && !matched_read && !destination) {
                    complete = false;
                }
            }
            if (node->view_src == root && (node->op == GGML_OP_SCALE || node->op == GGML_OP_CPY) &&
                j != group.remainder &&
                std::find(group.snapshots.begin(), group.snapshots.end(), j) == group.snapshots.end()) {
                complete = false;
            }
        }
        if (!complete) {
            continue;
        }
        const auto id = m_groups.size();
        for (int n : {group.reset, group.gather, group.remainder}) {
            m_nodes[n] = id;
        }
        if (group.concat >= 0) {
            m_nodes[group.concat] = id;
        }
        for (int n : group.snapshots) {
            m_nodes[n] = id;
        }
        m_groups.push_back(std::move(group));
    }
}

OutputVector RecurrentCacheFusion::translate(int index, const NodeContext & context) {
    const auto it = m_nodes.find(index);
    if (it == m_nodes.end()) {
        return {};
    }
    auto & group = m_groups[it->second];
    using namespace ov::op;
    auto constant = [](std::initializer_list<size_t> shape, std::initializer_list<int64_t> values) {
        return v0::Constant::create(element::i64, Shape(shape), std::vector<int64_t>(values));
    };
    auto metadata = [&](const std::string & name) -> Output<Node> {
        return std::make_shared<v1::Reshape>(context.get_input(name), constant({4}, {1, 1, 1, 1}), false);
    };
    if (index == group.reset) {
        group.old_cache = context.get_input(0);
        return {group.old_cache};
    }
    NodeContext gather_context(m_decoder, m_tensors, group.gather);
    auto rows = process_view_input_new(gather_context, 1);
    const auto remainder_name = m_decoder->get_op_name(group.remainder);
    auto offsets = std::make_shared<v0::Concat>(OutputVector{context.get_input("rs_slot_begin_" + remainder_name),
                                                             context.get_input("rs_src_begin_" + remainder_name)},
                                                0);
    OutputVector inputs{group.old_cache,
                        rows,
                        rows,
                        metadata("cache_rs_reset_idx"),
                        metadata("cache_rs_reset_len"),
                        context.has_input("rs_offsets_" + remainder_name) ?
                            context.get_input("rs_offsets_" + remainder_name) :
                            std::make_shared<v1::Reshape>(offsets, constant({4}, {1, 1, 1, 2}), false)};
    inputs[1] = context.has_input("rs_active_rows") ?
                    context.get_input("rs_active_rows") :
                    std::make_shared<v8::Slice>(rows, constant({1}, {0}), context.get_input("s_copy_active_slot_len"),
                                                constant({1}, {1}), constant({1}, {3}));
    if (index == group.gather) {
        auto op = std::make_shared<internal::RecurrentStateGather>(inputs);
        op->set_friendly_name(context.get_name() + "/RecurrentStateGather");
        group.gathered = op->output(0);
        group.prepared = op->output(1);
        return {group.gathered};
    }
    if (index == group.remainder) {
        return {group.prepared};
    }
    if (index == group.concat) {
        inputs.push_back(process_view_input_new(context, 1));
        OutputVector windows;
        for (int snapshot : group.snapshots) {
            const auto name = m_decoder->get_op_name(snapshot);
            auto window = std::make_shared<v0::Concat>(
                OutputVector{context.get_input("rs_src_begin_" + name), context.get_input("rs_slot_begin_" + name)}, 0);
            windows.push_back(std::make_shared<v1::Reshape>(window, constant({4}, {1, 1, 1, 2}), false));
        }
        inputs.push_back(context.has_input("rs_windows_" + context.get_name()) ?
                             context.get_input("rs_windows_" + context.get_name()) :
                             std::make_shared<v0::Concat>(windows, 2));
        const auto history = context.get_input_shape(0)[3].get_length();
        auto op = std::make_shared<internal::RecurrentConvCache>(inputs, history);
        op->set_friendly_name(context.get_name() + "/RecurrentConvCache");
        group.conv = op->output(0);
        group.updated = op->output(1);
        return {group.conv};
    }
    return {group.updated};
}
}  // namespace ov::frontend::ggml
#endif
