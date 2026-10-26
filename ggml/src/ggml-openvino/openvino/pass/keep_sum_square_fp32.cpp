#include "keep_sum_square_fp32.h"

#include <openvino/core/type/element_type.hpp>
#include <openvino/core/visibility.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#ifdef OPENVINO_STATIC_LIBRARY
#    define TRANSFORMATIONS_API
#else
#    ifdef IMPLEMENT_OPENVINO_API
#        define TRANSFORMATIONS_API OPENVINO_CORE_EXPORTS
#    else
#        define TRANSFORMATIONS_API OPENVINO_CORE_IMPORTS
#    endif  // IMPLEMENT_OPENVINO_API
#endif      // OPENVINO_STATIC_LIBRARY

namespace ov {
// from transformations/rt_info/disable_precision_conversion.hpp, which the OpenVINO runtime package does not ship
TRANSFORMATIONS_API void disable_conversion(const std::shared_ptr<Node> & node, const element::Type & to);
}  // namespace ov

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

namespace {
// the only consumer of node's output 0, or nullptr
std::shared_ptr<ov::Node> single_consumer(const std::shared_ptr<ov::Node> & node) {
    const auto targets = node->output(0).get_target_inputs();
    return targets.size() == 1 ? targets.begin()->get_node()->shared_from_this() : nullptr;
}

bool is_scalar_constant(const ov::Output<ov::Node> & output) {
    auto constant = ov::as_type_ptr<ov::op::v0::Constant>(output.get_node_shared_ptr());
    return constant && ov::shape_size(constant->get_shape()) == 1;
}
}  // namespace

KeepSumSquareFP32::KeepSumSquareFP32() {
    auto square = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>();
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v1::ReduceSum>({square, ov::pass::pattern::any_input()});

    const auto callback = [=](ov::pass::pattern::Matcher & m) {
        auto sum = m.get_match_root();
        auto square_node = m.get_pattern_value_map().at(square).get_node_shared_ptr();
        if (square_node->input_value(0) != square_node->input_value(1)) {
            return false;
        }
        // ConvertPrecision skips marked nodes, so the GPU plugin runs them in f32 under f16 inference precision
        ov::disable_conversion(square_node, ov::element::f16);
        ov::disable_conversion(sum, ov::element::f16);

        // optional ggml_scale (e.g. 1/(n-1) in a std), then Sqrt brings the result back into f16 range
        auto next = single_consumer(sum);
        if (ov::is_type<ov::op::v1::Multiply>(next) &&
            (is_scalar_constant(next->input_value(0)) || is_scalar_constant(next->input_value(1)))) {
            ov::disable_conversion(next, ov::element::f16);
            next = single_consumer(next);
        }
        if (ov::is_type<ov::op::v0::Sqrt>(next)) {
            ov::disable_conversion(next, ov::element::f16);
        }
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::KeepSumSquareFP32"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
