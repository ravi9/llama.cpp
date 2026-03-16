#pragma once

#include <cstdint>
#include <openvino/frontend/node_context.hpp>
#include <string>
#include <iostream>
#include "decoder.h"
#include "ggml.h"

struct ggml_tensor;

namespace ov {
namespace frontend {
namespace ggml {

class TranslateSession;

typedef std::map<std::string, Output<Node>> TensorMap;
typedef std::map<const struct ggml_tensor*, Output<Node>> TensorPtrMap;

class NodeContext : public frontend::NodeContext {
public:
    NodeContext(const std::shared_ptr<GgmlDecoder>& decoder,
                std::shared_ptr<TensorMap>& tensor_map,
                std::shared_ptr<TensorPtrMap>& tensor_ptr_map,
                int node_idx,
                TranslateSession* translate_session = nullptr)
        : ov::frontend::NodeContext(decoder->get_op_type(node_idx)),
          m_decoder(decoder),
          m_tensor_map(tensor_map),
          m_tensor_ptr_map(tensor_ptr_map),
          m_node_idx(node_idx),
          m_translate_session(translate_session) {
        m_input_names = decoder->get_input_names(m_node_idx);
        m_output_names = decoder->get_output_names(m_node_idx);

        m_input_tensors = decoder->get_input_tensors(m_node_idx);
        m_output_tensors = decoder->get_output_tensors(m_node_idx);
    }

    TranslateSession* get_translate_session() const {
        return m_translate_session;
    }

    const std::vector<std::string>& get_input_names() const { return m_input_names; }

    size_t get_input_size() const override {
        return m_decoder->get_input_size(m_node_idx);
    }

    ov::element::Type get_input_type(size_t index) const {
        return m_decoder->get_input_type(m_node_idx, m_input_names[index]);
    }

    PartialShape get_input_shape(size_t input_index) const {
        return m_decoder->get_input_shape(m_node_idx, m_input_names[input_index]);
    }

    std::vector<size_t> get_input_stride(size_t index) const {
        return m_decoder->get_input_stride(m_node_idx, m_input_names[index]);
    }

    std::string get_output_name() const { return m_output_names[0]; }

    PartialShape get_output_shape() const { return m_decoder->get_output_shape(m_node_idx); }

    int32_t* get_input_op_params(size_t index) const {
        return m_decoder->get_input_op_params(m_node_idx, m_input_names[index]);
    }

    int32_t * get_output_op_params() const { return m_decoder->get_output_op_params(m_node_idx); }

    ov::element::Type get_output_type() const {
        return m_decoder->get_output_type(m_node_idx);
    }

    Output<Node> get_input(int idx) const override {
        // 1. Safely check the pointer map first (Physical Memory Address)
        if (idx < m_input_tensors.size() && m_input_tensors[idx] != nullptr) {
            auto it = m_tensor_ptr_map->find(m_input_tensors[idx]);
            if (it != m_tensor_ptr_map->end()) {
                // PROOF IT WORKS:
                // std::cout << "[DEBUG] Tensor found perfectly via Pointer Map!\n";
                return it->second; // Found it via exact pointer!
            }
        }

        // 2. Fallback to the string map (For OpenVINO synthetic tensors & static weights)
        if (idx < m_input_names.size()) {
            std::string target_name = m_input_names[idx];

            auto it = m_tensor_map->find(target_name);
            if (it != m_tensor_map->end()) {
                return it->second; // Found it via string name!
            }

            // Temporary fallback: Brute-Force Pointer Search
            // If the pointer mutated due to in-place optimization, scan all translated physical nodes!
            for (const auto& pair : *m_tensor_ptr_map) {
                if (pair.first != nullptr) {
                    std::string actual_name = ggml_get_name(pair.first);
                    
                    // IF WE ARE LOOKING FOR NORM-21, PRINT EVERYTHING WE HAVE!
                    if (target_name == "norm-21" || target_name == "ffn_inp-21") {
                        std::cout << "[DEBUG TRAP] In memory pointer name: '" << actual_name << "'\n";
                    }

                    if (actual_name == target_name) {
                        std::cerr << "[GGUFReaderV2] Recovered shifted tensor via brute-force: '" << target_name << "'\n";
                        return pair.second;
                    }
                }
            }

            // 🚨 THE GSOC FIX: NO MORE DUMMY NODES! 🚨
            // If we get here, the node is TRULY missing. We throw a hard error 
            // so we know if our Scheduler Capture Override worked or failed.
            throw std::runtime_error("[GGUFReaderV2] FATAL: Tensor completely lost during extraction: '" + target_name + "'");
        }

        throw std::runtime_error("CRITICAL: Input index out of bounds!");
    }

    Output<Node> get_input(const std::string& name) const override {
        if (m_tensor_map->find(name) == m_tensor_map->end()) {
            throw std::runtime_error("'" + name + "' not found in tensor map.");
        }
        return m_tensor_map->at(name);
    }

    bool has_input(const std::string& name) const {
        return m_tensor_map->find(name) != m_tensor_map->end();
    }

    const std::string& get_name() const override {
        return m_decoder->get_op_name(m_node_idx);
    }

    ov::Any get_attribute_as_any(const std::string& name) const override {
        return m_decoder->get_attribute(name);
    }

    int get_op_case() const {
        return m_decoder->get_op_case(m_node_idx);
    }

    bool is_static() const { return m_decoder->is_static(); }

    bool is_stateful() const { return m_decoder->is_stateful(); }

private:
    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap>& m_tensor_map;
    std::shared_ptr<TensorPtrMap>& m_tensor_ptr_map;
    int m_node_idx;
    TranslateSession* m_translate_session;
    std::vector<std::string> m_input_names;
    std::vector<std::string> m_output_names;
    std::vector<const struct ggml_tensor*> m_input_tensors;
    std::vector<const struct ggml_tensor*> m_output_tensors;
};

using CreatorFunction = std::function<ov::OutputVector(const ov::frontend::ggml::NodeContext&)>;

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
