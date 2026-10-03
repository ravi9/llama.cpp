#include "align_eltwise_ranks.h"

#include <numeric>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace pass {

AlignEltwiseOperandRanks::AlignEltwiseOperandRanks() {
    auto eltwise_m = ov::pass::pattern::wrap_type<ov::op::v1::Add, ov::op::v1::Multiply, ov::op::v1::Subtract>();

    const auto callback = [this](ov::pass::pattern::Matcher & m) {
        auto node = m.get_match_root();
        if (node->get_input_size() != 2) {
            return false;
        }

        auto lhs = node->input_value(0);
        auto rhs = node->input_value(1);

        // A tensor-vs-Constant mismatch is the norm's own eps / 1-over-sqrt arithmetic. That
        // folds into the `rms` primitive itself instead of becoming a fused eltwise post-op,
        // so it is not affected and is left alone.
        if (ov::is_type<ov::op::v0::Constant>(lhs.get_node()) ||
            ov::is_type<ov::op::v0::Constant>(rhs.get_node())) {
            return false;
        }

        const auto lhs_rank = lhs.get_partial_shape().rank();
        const auto rhs_rank = rhs.get_partial_shape().rank();
        if (lhs_rank.is_dynamic() || rhs_rank.is_dynamic() || lhs_rank == rhs_rank) {
            return false;
        }

        const size_t shorter_idx = lhs_rank.get_length() < rhs_rank.get_length() ? 0 : 1;
        const auto & shorter = shorter_idx == 0 ? lhs : rhs;
        const int64_t diff = std::abs(lhs_rank.get_length() - rhs_rank.get_length());

        std::vector<int64_t> axes(static_cast<size_t>(diff));
        std::iota(axes.begin(), axes.end(), 0);
        auto unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
            shorter, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{ axes.size() }, axes));

        node->input(shorter_idx).replace_source_output(unsqueeze->output(0));
        register_new_node(unsqueeze);
        return true;
    };

    register_matcher(
        std::make_shared<ov::pass::pattern::Matcher>(eltwise_m, "ov::frontend::ggml::pass::AlignEltwiseOperandRanks"),
        callback);
}

}  // namespace pass
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
