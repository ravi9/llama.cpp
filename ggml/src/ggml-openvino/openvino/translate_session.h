#pragma once

#include "input_model.h"
#include "node_context.h"

namespace ov {
namespace frontend {
namespace ggml {

/**
 * @brief Owns a GGML-to-OpenVINO translation session and caches the converted model
 */
class TranslateSession {
public:
    /**
     * @brief Constructs a translation session for a GGML input model
     * @param input_model - frontend input model to translate
     * @param translator_map - mapping from GGML operation names to translator functions
     * @param naive - true to skip non-naive preprocessing patterns during translation
     */
    TranslateSession(const frontend::InputModel::Ptr & input_model,
                     const std::unordered_map<std::string, CreatorFunction> & translator_map,
                     bool naive = false);

    /**
     * @brief Gets the converted OpenVINO model, translating it on the first call
     * @return cached or newly translated OpenVINO model
     */
    std::shared_ptr<Model> get_converted_model();

    /**
     * @brief Translates a GGML frontend input model into an OpenVINO model
     * @param input_model - frontend input model to translate
     * @return converted OpenVINO model
     */
    std::shared_ptr<Model> translate_graph(const frontend::InputModel::Ptr & input_model);

    /**
     * @brief Applies OpenVINO graph transformations required after GGML translation
     * @param model - OpenVINO model to transform
     * @return transformed OpenVINO model
     */
    std::shared_ptr<Model> apply_transformations(std::shared_ptr<Model> model);

private:
    const frontend::InputModel::Ptr m_input_model;
    const std::unordered_map<std::string, CreatorFunction> & m_translator_map;
    std::shared_ptr<Model> m_ov_model;
    bool m_naive;
};

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
