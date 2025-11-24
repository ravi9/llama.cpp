#pragma once

#include "ggml.h"

#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(ggml_tensor * node,
                std::shared_ptr<TensorMap> & tensor_map,
                bool is_static = false,
                std::string op_type = "",
                TranslateSession * translate_session = nullptr) :
        ov::frontend::NodeContext(op_type),
        m_node(node),
        m_tensor_map(tensor_map),
        m_is_static(is_static),
        m_op_type(op_type),
        m_translate_session(translate_session),
        m_node_name(std::string(node->name)),
        m_op_case(0) {
        std::string node_name;
        if (node->op == GGML_OP_SET_ROWS) {
            node_name = std::string(node->view_src->name);
        } else {
            node_name = std::string(node->name);
        }

        m_output_names.push_back(node_name);
        m_outputs[node_name] = node;

        for (int i = 0; i < GGML_MAX_SRC; i++) {
            auto * src = node->src[i];
            if (src == nullptr) {
                continue;
            }
            std::string src_name = std::string(src->name);
            m_input_names.push_back(src_name);
            m_inputs[src_name] = src;
        }

        m_op_case = compute_op_case(node);
    }

    TranslateSession * get_translate_session() const { return m_translate_session; }

    const std::vector<std::string> & get_input_names() const { return m_input_names; }

    size_t get_input_size() const override { return m_input_names.size(); }

    ov::element::Type get_input_type(size_t index) const {
        switch (m_inputs.at(m_input_names[index])->type) {
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

    PartialShape get_input_shape(size_t index) const {
        std::vector<size_t> shape;
        for (int i = GGML_MAX_DIMS - 2; i >= 0; --i) {
            shape.push_back(static_cast<size_t>(m_inputs.at(m_input_names[index])->ne[i]));
        }
        return ov::PartialShape(shape);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        std::vector<size_t> stride;
        for (int i = GGML_MAX_DIMS - 2; i >= 0; --i) {
            stride.push_back(static_cast<size_t>(m_inputs.at(m_input_names[index])->nb[i]));
        }
        return stride;
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape(size_t index) const {
        std::vector<size_t> shape;
        for (int i = GGML_MAX_DIMS - 2; i >= 0; --i) {
            shape.push_back(static_cast<size_t>(m_outputs.at(m_output_names[index])->ne[i]));
        }
        return ov::PartialShape(shape);
    }

    int32_t * get_input_op_params(size_t index) const { return m_inputs.at(m_input_names[index])->op_params; }

    int32_t * get_output_op_params(size_t index) const { return m_outputs.at(m_output_names[index])->op_params; }

    ov::element::Type get_output_type(size_t index) const {
        switch (m_outputs.at(m_output_names[index])->type) {
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

    Output<Node> get_input(int idx) const override { return m_tensor_map->at(m_input_names[idx]); }

    Output<Node> get_input(const std::string & name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string & name) const { return m_tensor_map->find(name) != m_tensor_map->end(); }

    const std::string & get_name() const override { return m_node_name; }

    ov::Any get_attribute_as_any(const std::string & name) const override {
        return nullptr;
        GGML_UNUSED(name);
    }

    int get_op_case() const { return m_op_case; }

    bool is_static() const { return m_is_static; }

private:
    ggml_tensor * m_node;
    std::shared_ptr<TensorMap> & m_tensor_map;
    bool m_is_static = false;
    std::string m_op_type;
    TranslateSession * m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    std::string m_node_name;
    std::map<std::string, ggml_tensor *> m_inputs;
    std::map<std::string, ggml_tensor *> m_outputs;
    int m_op_case;

    int extract_layer_from_name(const std::string & name) {
        size_t pos1 = name.find("_l");
        pos1 += 2;
        size_t pos2 = name.find(' ', pos1);
        if (pos2 == std::string::npos) {
            pos2 = name.length();
        }
        std::string layer_str = name.substr(pos1, pos2 - pos1);
        int layer = std::stoi(layer_str);
        return layer;
    }

    int compute_op_case(ggml_tensor * node) {
        int op_case = 0;
        switch (node->op) {
        case GGML_OP_RESHAPE: {
            if (node->src[0]->op == GGML_OP_RESHAPE && node->src[0]->src[0]->ne[0] == node->ne[0] &&
                node->src[0]->src[0]->ne[1] == node->ne[1]) {
                op_case = 4;
            } else if (node->ne[0] * node->ne[1] == node->src[0]->ne[0]) {
                op_case = 1;
            } else if (node->src[0]->ne[0] * node->src[0]->ne[1] == node->ne[0]) {
                op_case = 2;
            } else if (node->src[0]->ne[0] * node->src[0]->ne[1] == node->ne[1]) {
                op_case = 3;
            }
            break;
        }
        case GGML_OP_CONT: {
            if (node->src[0]->op == GGML_OP_PERMUTE) {
                op_case = 1;
            } else if (node->src[0]->op == GGML_OP_TRANSPOSE) {
                op_case = 2;
            } else if (node->src[0]->op == GGML_OP_VIEW) {
                // The input comes from a VIEW which is subtensor
                op_case = 3;
            }
            break;
        }
        case GGML_OP_PERMUTE: {
            if (node->src[0]->op != GGML_OP_VIEW) {
                op_case = 1;
            } else if (ggml_is_contiguous(node->src[0])) {
                std::string src_name(node->view_src->name);
                if (src_name.find("cache") == std::string::npos) {
                    op_case = 1;
                } else {
                    // Permute kv cache (view)
                    if (!(std::string(node->src[3]->name).find("swa") != std::string::npos)) {
                        op_case = 2;
                    } else {
                        op_case = 3;
                    }
                }
            }
            break;
        }
        case GGML_OP_MUL_MAT: {
            if (node->src[0]->op == GGML_OP_CONT && node->src[0]->src[0]->op == GGML_OP_TRANSPOSE) {
                op_case = 2;
            } else if (node->src[0]->op == GGML_OP_VIEW && node->src[1]->op == GGML_OP_VIEW) {
                // test-backend-ops case
                op_case = 3;
            }
            break;
        }
        case GGML_OP_GET_ROWS: {
            if (node->src[1]->op == GGML_OP_VIEW) {
                op_case = 2;
            }
            break;
        }
        case GGML_OP_ROPE: {
            if (node->src[0]->op == GGML_OP_VIEW) {
                op_case = 2;
            }
            break;
        }
        case GGML_OP_VIEW: {
            if (node->src[0]->op == GGML_OP_VIEW) {
                auto * src = node->src[0];
                auto * view_src = src->view_src;
                if (view_src->ne[1] != src->ne[2]) {
                    throw std::runtime_error("Unsupported VIEW case");
                }
                op_case = 2;
            }
        }
        default:
            break;
        }
        return op_case;
    }
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext &)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
