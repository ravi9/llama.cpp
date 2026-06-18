#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <openvino/core/except.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/equal.hpp>
#include <openvino/op/if.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/slice.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_scale(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    float scale;
    float bias;
    memcpy(&scale, (float *) context.get_output_op_params() + 0, sizeof(float));
    memcpy(&bias, (float *) context.get_output_op_params() + 1, sizeof(float));

    auto scale_node = std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{scale});

    if (context.get_op_case() == 1) {
        OPENVINO_ASSERT(context.has_input("cache_rs_reset"), "Missing input cache_rs_reset");
        auto cache_rs_reset = context.get_input("cache_rs_reset");
        auto cache_rs = context.get_input(0);
        // use v8::if, if reset == 1, then output = input * 0, else output = input
        auto one = std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
        auto reset = std::make_shared<ov::op::v1::Equal>(cache_rs_reset, one);
        auto if_node = std::make_shared<ov::op::v8::If>(reset);

        auto then_param =
            std::make_shared<ov::op::v0::Parameter>(cache_rs.get_element_type(), cache_rs.get_partial_shape());
        auto cleared_cache_rs = std::make_shared<ov::op::v1::Multiply>(then_param, scale_node);
        auto then_result = std::make_shared<ov::op::v0::Result>(cleared_cache_rs);
        auto then_body = std::make_shared<ov::Model>(ov::ResultVector{then_result}, ov::ParameterVector{then_param});

        auto else_param =
            std::make_shared<ov::op::v0::Parameter>(cache_rs.get_element_type(), cache_rs.get_partial_shape());
        auto else_result = std::make_shared<ov::op::v0::Result>(else_param);
        auto else_body = std::make_shared<ov::Model>(ov::ResultVector{else_result}, ov::ParameterVector{else_param});

        if_node->set_then_body(then_body);
        if_node->set_else_body(else_body);
        if_node->set_input(cache_rs, then_param, else_param);

        auto if_output = if_node->set_output(then_result, else_result);
        return rename_outputs_with_suffix({if_output}, context.get_name());
    }

    auto scaled = std::make_shared<ov::op::v1::Multiply>(context.get_input(0), scale_node);

    std::shared_ptr<ov::Node> res;
    if (bias != 0.0f) {
        auto bias_node =
            std::make_shared<ov::op::v0::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{bias});
        res = std::make_shared<ov::op::v1::Add>(scaled, bias_node);
    } else {
        res = scaled;
    }

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
