#include "rescale_norm_divide.h"

#include <openvino/core/graph_util.hpp>
#include <openvino/core/rt_info.hpp>
#include <openvino/op/abs.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reduce_l2.hpp>
#include <openvino/op/reduce_max.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/sqrt.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

namespace {
struct NormPattern {
    ov::Output<ov::Node> input;
    ov::Output<ov::Node> axes;
    bool keep_dims = false;
    std::vector<std::shared_ptr<ov::Node>> wrappers;
    std::vector<std::shared_ptr<ov::Node>> nodes;
};

bool get_norm_pattern(const ov::Output<ov::Node> & denominator, NormPattern & norm) {
    auto node = denominator.get_node_shared_ptr();
    while (true) {
        if (auto tile = ov::as_type_ptr<ov::op::v0::Tile>(node)) {
            norm.wrappers.push_back(node);
            node = tile->get_input_node_shared_ptr(0);
        } else if (auto broadcast = ov::as_type_ptr<ov::op::v3::Broadcast>(node)) {
            norm.wrappers.push_back(node);
            node = broadcast->get_input_node_shared_ptr(0);
        } else {
            break;
        }
    }

    if (auto l2 = ov::as_type_ptr<ov::op::v4::ReduceL2>(node)) {
        norm.input = l2->input_value(0);
        norm.axes = l2->input_value(1);
        norm.keep_dims = l2->get_keep_dims();
        norm.nodes.push_back(node);
        return true;
    }

    auto sqrt = ov::as_type_ptr<ov::op::v0::Sqrt>(node);
    auto sum = sqrt ? ov::as_type_ptr<ov::op::v1::ReduceSum>(sqrt->get_input_node_shared_ptr(0)) : nullptr;
    auto square = sum ? ov::as_type_ptr<ov::op::v1::Multiply>(sum->get_input_node_shared_ptr(0)) : nullptr;
    if (!square || square->input_value(0) != square->input_value(1)) {
        return false;
    }

    norm.input = square->input_value(0);
    norm.axes = sum->input_value(1);
    norm.keep_dims = sum->get_keep_dims();
    norm.nodes = {node, sum, square};
    return true;
}

// True when x is already Divide(y, [Maximum(]ReduceMax(Abs(y))[, eps)]), i.e. produced by these passes.
bool is_max_scaled(const ov::Output<ov::Node> & x) {
    auto div = ov::as_type_ptr<ov::op::v1::Divide>(x.get_node_shared_ptr());
    if (!div) {
        return false;
    }
    auto denom = div->get_input_node_shared_ptr(1);
    if (auto clamp = ov::as_type_ptr<ov::op::v1::Maximum>(denom)) {
        denom = clamp->get_input_node_shared_ptr(0);
    }
    auto max = ov::as_type_ptr<ov::op::v1::ReduceMax>(denom);
    auto abs = max ? ov::as_type_ptr<ov::op::v0::Abs>(max->get_input_node_shared_ptr(0)) : nullptr;
    return abs && abs->input_value(0) == div->input_value(0);
}
}  // namespace

RescaleNormDivide::RescaleNormDivide() {
    auto mul = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>();
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v1::Divide>({mul, ov::pass::pattern::any_input()});

    const auto callback = [](ov::pass::pattern::Matcher & m) {
        auto div = m.get_match_root();
        auto mul_node = div->get_input_node_shared_ptr(0);
        NormPattern norm;
        if (!get_norm_pattern(div->input_value(1), norm)) {
            return false;
        }
        const auto & x = norm.input;

        size_t x_idx;
        if (mul_node->input_value(0) == x) {
            x_idx = 0;
        } else if (mul_node->input_value(1) == x) {
            x_idx = 1;
        } else {
            return false;
        }
        if (is_max_scaled(x)) {
            return false;
        }

        auto eps = ov::op::v0::Constant::create(x.get_element_type(), ov::Shape{}, {1e-4f});
        auto abs = std::make_shared<ov::op::v0::Abs>(x);
        auto row_max = std::make_shared<ov::op::v1::Maximum>(std::make_shared<ov::op::v1::ReduceMax>(abs, norm.axes, true), eps);
        // Scaling both the numerator and norm input preserves the ratio and bounds the norm input.
        auto x_scaled = std::make_shared<ov::op::v1::Divide>(x, row_max);

        OutputVector mul_inputs = mul_node->input_values();
        mul_inputs[x_idx] = x_scaled;
        auto new_mul = mul_node->clone_with_new_inputs(mul_inputs);
        auto new_l2 = std::make_shared<ov::op::v4::ReduceL2>(x_scaled, norm.axes, norm.keep_dims);
        new_l2->set_friendly_name(norm.nodes.front()->get_friendly_name());

        ov::Output<ov::Node> new_denominator = new_l2;
        std::vector<std::shared_ptr<ov::Node>> new_nodes{abs, row_max, x_scaled, new_mul, new_l2};
        for (auto it = norm.wrappers.rbegin(); it != norm.wrappers.rend(); ++it) {
            auto inputs = (*it)->input_values();
            inputs[0] = new_denominator;
            auto wrapper = (*it)->clone_with_new_inputs(inputs);
            new_denominator = wrapper;
            new_nodes.push_back(wrapper);
        }
        auto new_div = div->clone_with_new_inputs({new_mul, new_denominator});

        new_div->set_friendly_name(div->get_friendly_name());
        std::vector<std::shared_ptr<ov::Node>> source_nodes{div, mul_node};
        source_nodes.insert(source_nodes.end(), norm.nodes.begin(), norm.nodes.end());
        source_nodes.insert(source_nodes.end(), norm.wrappers.begin(), norm.wrappers.end());
        new_nodes.push_back(new_div);
        ov::copy_runtime_info(source_nodes, new_nodes);
        ov::replace_node(div, new_div);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::RescaleNormDivide"),
                     callback);
}

FuseSquareSum::FuseSquareSum() {
    // Canonicalize standalone norms; RescaleNormDivide also handles unfused norms in full normalization patterns.
    auto square = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>();
    auto sum = ov::pass::pattern::wrap_type<ov::op::v1::ReduceSum>({square, ov::pass::pattern::any_input()});
    auto sqrt = ov::pass::pattern::wrap_type<ov::op::v0::Sqrt>({sum});
    const auto fuse_callback = [](ov::pass::pattern::Matcher & m) {
        auto sqrt_node = m.get_match_root();
        auto sum_node = ov::as_type_ptr<ov::op::v1::ReduceSum>(sqrt_node->get_input_node_shared_ptr(0));
        auto square_node = sum_node->get_input_node_shared_ptr(0);
        if (square_node->input_value(0) != square_node->input_value(1)) {
            return false;
        }

        auto l2 = std::make_shared<ov::op::v4::ReduceL2>(square_node->input_value(0), sum_node->input_value(1),
                                                         sum_node->get_keep_dims());
        l2->set_friendly_name(sqrt_node->get_friendly_name());
        ov::copy_runtime_info({sqrt_node, sum_node, square_node}, l2);
        ov::replace_node(sqrt_node, l2);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(sqrt, "ov::frontend::ggml::pass::FuseSquareSum"),
                     fuse_callback);
}

RescaleReduceL2::RescaleReduceL2() {
    // ||x|| = c * ||x / c|| for any c > 0; the GPU squares each element in fp16, so |x| > 256 overflows.
    auto pattern = ov::pass::pattern::wrap_type<ov::op::v4::ReduceL2>();

    const auto callback = [](ov::pass::pattern::Matcher & m) {
        auto l2 = ov::as_type_ptr<ov::op::v4::ReduceL2>(m.get_match_root());
        const auto x = l2->input_value(0);
        if (is_max_scaled(x)) {
            return false;
        }
        const auto axes = l2->input_value(1);
        const bool keep_dims = l2->get_keep_dims();

        // eps keeps an all-zero row at 0 instead of 0/0
        auto eps = ov::op::v0::Constant::create(x.get_element_type(), ov::Shape{}, {1e-4f});
        auto abs = std::make_shared<ov::op::v0::Abs>(x);
        auto c = std::make_shared<ov::op::v1::Maximum>(std::make_shared<ov::op::v1::ReduceMax>(abs, axes, true), eps);
        auto x_scaled = std::make_shared<ov::op::v1::Divide>(x, c);
        auto new_l2 = l2->clone_with_new_inputs({x_scaled, axes});

        ov::Output<ov::Node> c_out = c;
        if (!keep_dims) {
            c_out = std::make_shared<ov::op::v1::Maximum>(std::make_shared<ov::op::v1::ReduceMax>(abs, axes, false), eps);
        }
        auto res = std::make_shared<ov::op::v1::Multiply>(new_l2, c_out);

        res->set_friendly_name(l2->get_friendly_name());
        ov::copy_runtime_info(l2, {abs, c, x_scaled, new_l2, res});
        ov::replace_node(l2, res);
        return true;
    };

    register_matcher(std::make_shared<ov::pass::pattern::Matcher>(pattern, "ov::frontend::ggml::pass::RescaleReduceL2"),
                     callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
