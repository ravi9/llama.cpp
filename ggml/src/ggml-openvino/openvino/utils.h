#pragma once

#include "node_context.h"

#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <utility>

namespace ov {
namespace frontend {
namespace ggml {

/**
 * @brief Validates that a translated GGML node has an input count within the expected range
 * @param context - node conversion context to inspect
 * @param min_inputs - minimum accepted number of inputs
 * @param max_inputs - maximum accepted number of inputs
 */
void num_inputs_check(const NodeContext & context, size_t min_inputs, size_t max_inputs);

/**
 * @brief Gathers selected dimensions from an existing ShapeOf node
 * @param shape - ShapeOf node that produces the source shape
 * @param dims - dimension indices to gather
 * @return node that contains the selected dimensions
 */
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         const std::vector<int> & dims);

/**
 * @brief Builds ShapeOf for a node and gathers selected dimensions from it
 * @param node - node whose output shape is queried
 * @param dims - dimension indices to gather
 * @return node that contains the selected dimensions
 */
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node> & node, const std::vector<int> & dims);

/**
 * @brief Appends a suffix to each output node friendly name
 * @param outputs - outputs whose nodes should be renamed
 * @param suffix - suffix to append to each friendly name
 * @return the same outputs after renaming their nodes
 */
OutputVector rename_outputs_with_suffix(const OutputVector & outputs, const std::string & suffix);

/**
 * @brief Builds ROPE sine and cosine tensors from GGML rope parameters and position inputs
 * @param rope_params - GGML ROPE parameter buffer
 * @param inp_pos - position input node
 * @param rope_freqs_weight - optional frequency scaling weight node
 * @param imrope - true when building IMROPE sine and cosine tensors
 * @param stateful - true when building tensors for a stateful model layout
 * @return pair of sine and cosine outputs
 */
std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight = nullptr,
                                                           bool imrope = false,
                                                           bool stateful = false);

/**
 * @brief Resolves a possibly VIEW-based input into the OpenVINO output consumed by translators
 * @param context - node conversion context that owns the input
 * @param input_index - input index to resolve
 * @return OpenVINO output for the resolved input
 */
ov::Output<ov::Node> process_view_input(const NodeContext & context, int input_index, int slice_len = 0, int axis = -1);
ov::Output<ov::Node> process_view_input_new(const NodeContext & context, int input_index);

namespace op {
/**
 * @brief Translates a binary GGML op to a matching OpenVINO op after VIEW input resolution
 * @tparam T - OpenVINO operation type to construct
 * @param context - node conversion context for the GGML op
 * @return translated OpenVINO outputs with renamed friendly names
 */
template <typename T> OutputVector translate_1to1_match_2_inputs(const NodeContext & context) {
    num_inputs_check(context, 2, 2);
    auto input_0 = process_view_input_new(context, 0);
    auto input_1 = process_view_input_new(context, 1);
    auto res = std::make_shared<T>(input_0, input_1);
    return rename_outputs_with_suffix({res}, context.get_name());
}

/**
 * @brief Translates a unary GGML op to a matching OpenVINO op after VIEW input resolution
 * @tparam T - OpenVINO operation type to construct
 * @param context - node conversion context for the GGML op
 * @return translated OpenVINO outputs with renamed friendly names
 */
template <typename T> OutputVector translate_1to1_match_1_input(const NodeContext & context) {
    num_inputs_check(context, 1, 1);
    auto input = process_view_input_new(context, 0);
    auto res = std::make_shared<T>(input);
    return rename_outputs_with_suffix({res}, context.get_name());
}
}  // namespace op

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
