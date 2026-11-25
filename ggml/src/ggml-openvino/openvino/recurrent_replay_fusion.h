#pragma once
#include "../ggml-decoder.h"
#include "node_context.h"

#include <set>

namespace ov::frontend::ggml {
class RecurrentReplayFusion {
public:
    RecurrentReplayFusion(const std::shared_ptr<GgmlDecoder> &, const std::shared_ptr<TensorMap> &);
    OutputVector translate(int index, const NodeContext & context);

    const std::map<std::string, Output<Node>> & updated_banks() const { return m_updated_banks; }
private:
    struct Group {
        int compute = -1;
        bool conv = false;
        const ggml_tensor * current = nullptr;
        std::vector<const ggml_tensor *> banks;
        std::set<const ggml_tensor *> consumed;
    };

    std::shared_ptr<GgmlDecoder> m_decoder;
    std::shared_ptr<TensorMap> m_tensors;
    const ggml_cgraph * m_graph = nullptr;
    std::vector<Group> m_groups;
    std::map<int, size_t> m_nodes;
    std::map<std::string, Output<Node>> m_updated_banks;
};
}  // namespace ov::frontend::ggml
