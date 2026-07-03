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

void num_inputs_check(const NodeContext & context, size_t min_inputs, size_t max_inputs);

std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::op::v3::ShapeOf> & shape,
                                         const std::vector<int> & dims);
std::shared_ptr<ov::Node> get_dimensions(const std::shared_ptr<ov::Node> & node, const std::vector<int> & dims);

OutputVector rename_outputs_with_suffix(const OutputVector & outputs, const std::string & suffix);

std::pair<ov::Output<Node>, ov::Output<Node>> make_sin_cos(int32_t * rope_params,
                                                           std::shared_ptr<ov::Node> inp_pos,
                                                           std::shared_ptr<ov::Node> rope_freqs_weight = nullptr,
                                                           bool imrope = false,
                                                           bool stateful = false);

ov::Output<ov::Node> process_view_input_new(const NodeContext & context, int input_index);

namespace op {
template <typename T> OutputVector translate_1to1_match_2_inputs(const NodeContext & context) {
    num_inputs_check(context, 2, 2);
    auto input_0 = process_view_input_new(context, 0);
    auto input_1 = process_view_input_new(context, 1);
    auto res = std::make_shared<T>(input_0, input_1);
    return rename_outputs_with_suffix({res}, context.get_name());
}

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
