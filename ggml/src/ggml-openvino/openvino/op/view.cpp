#include "../op_table.h"
#include "../utils.h"
#include <openvino/op/constant.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <set>
namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_view(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    if (!context.is_static()) {
        // On the stateless/non-static path VIEW is normally a no-op (consumers re-slice).
        // EXCEPTION: the MoE expert aggregation slices each expert plane out of
        // ffn_moe_weighted [n_embd, n_expert_used, n_tokens] with ggml_view_2d and then
        // sums the planes with a chain of ADDs (llama-graph.cpp). Those ADDs read this
        // VIEW node directly from the tensor map and do NOT re-slice, so a no-op here
        // makes every plane the full tensor and the expert sum collapses. Materialize the
        // single-expert slice here. Gated by name (ffn_moe_weighted...view) so it can't
        // affect any other view.
        const std::string & vname = context.get_name();
        if (vname.find("ffn_moe_weighted") != std::string::npos) {
            auto src_ps = context.get_input_shape(0);
            auto dst_ps = context.get_output_shape();
            if (src_ps.rank().is_static() && dst_ps.rank().is_static() && src_ps.rank() == dst_ps.rank() &&
                src_ps.is_static() && dst_ps.is_static()) {
                auto sst = context.get_input_stride(0);
                auto dst = context.get_output_stride();
                size_t voff = context.get_output_op_offset();
                auto ss = src_ps.to_shape();
                auto dd = dst_ps.to_shape();
                const size_t nd = ss.size();
                if (sst.size() == nd && dst.size() == nd) {
                    // Map each dst axis of size>1 to a src axis with equal (size,stride);
                    // the unmatched src axis of size>1 is the indexed expert axis.
                    std::vector<bool> used(nd, false);
                    bool ok = true;
                    for (size_t d = 0; d < nd; ++d) {
                        if (dd[d] == 1) {
                            continue;
                        }
                        int found = -1;
                        for (size_t s = 0; s < nd; ++s) {
                            if (!used[s] && ss[s] == dd[d] && sst[s] == dst[d]) { found = (int) s; break; }
                        }
                        if (found < 0) { ok = false; break; }
                        used[found] = true;
                    }
                    int dropped = -1;
                    if (ok) {
                        for (size_t s = 0; s < nd; ++s) {
                            if (!used[s] && ss[s] > 1) {
                                if (dropped >= 0) { ok = false; break; }
                                dropped = (int) s;
                            }
                        }
                    }
                    if (ok && dropped >= 0) {
                        const size_t dstr = sst[dropped];
                        const int64_t dsz = (int64_t) ss[dropped];
                        if (dstr > 0 && voff % dstr == 0) {
                            const int64_t sel = (int64_t) (voff / dstr);
                            if (sel >= 0 && sel < dsz) {
                                ov::Output<ov::Node> sl = std::make_shared<ov::op::v8::Slice>(
                                    context.get_input(0),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {sel + 1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
                                    ov::op::v0::Constant::create(ov::element::i64, {1}, {dropped}));
                                auto dc = ov::op::v0::Constant::create(
                                    ov::element::i64, {nd}, std::vector<int64_t>(dd.begin(), dd.end()));
                                auto rs = std::make_shared<ov::op::v1::Reshape>(sl, dc, false);
                                return rename_outputs_with_suffix({rs}, context.get_name());
                            }
                        }
                    }
                }
            }
        }
        return {context.get_input(0)};
    }

    auto input = context.get_input(0);
    auto src_shape = context.get_input_shape(0);
    auto dst_shape = context.get_output_shape();

    if (src_shape.rank().is_dynamic() || dst_shape.rank().is_dynamic()) {
        return {input};
    }

    int64_t src_elems = 1, dst_elems = 1;
    for (int64_t i = 0; i < src_shape.rank().get_length(); ++i) {
        if (src_shape[i].is_dynamic()) return {input};
        src_elems *= src_shape[i].get_length();
    }
    for (int64_t i = 0; i < dst_shape.rank().get_length(); ++i) {
        if (dst_shape[i].is_dynamic()) return {input};
        dst_elems *= dst_shape[i].get_length();
    }

    if (dst_elems >= src_elems) {
        return {input};
    }

    auto src_stride = context.get_input_stride(0);
    auto dst_stride = context.get_output_stride();
    size_t view_offset = context.get_output_op_offset();

    bool same_stride = (src_stride.size() == dst_stride.size());
    if (same_stride) {
        for (size_t i = 0; i < src_stride.size(); ++i) {
            if (src_stride[i] != dst_stride[i]) {
                same_stride = false;
                break;
            }
        }
    }

    if (!same_stride) {
        return {input};
    }

    auto src_ov_shape = src_shape.to_shape();
    auto dst_ov_shape = dst_shape.to_shape();
    size_t ndims = src_ov_shape.size();
    if (dst_ov_shape.size() != ndims) {
        return {input};
    }

    std::vector<int> diff_dims;
    for (size_t i = 0; i < ndims; ++i) {
        if (src_ov_shape[i] != dst_ov_shape[i]) {
            diff_dims.push_back(static_cast<int>(i));
        }
    }

    if (diff_dims.size() != 1) {
        return {input};
    }

    int slice_dim = diff_dims[0];
    int64_t dim_size = static_cast<int64_t>(src_ov_shape[slice_dim]);

    size_t ov_stride_for_dim = 1;
    for (size_t i = slice_dim + 1; i < ndims; ++i) {
        ov_stride_for_dim *= src_ov_shape[i];
    }
    size_t elem_size = src_stride.back();
    if (elem_size == 0) elem_size = 1;

    int64_t begin_val = 0;
    if (ov_stride_for_dim > 0 && elem_size > 0) {
        begin_val = static_cast<int64_t>((view_offset / elem_size) / ov_stride_for_dim);
    }
    int64_t end_val = begin_val + static_cast<int64_t>(dst_ov_shape[slice_dim]);

    if (begin_val < 0 || end_val > dim_size) {
        return {input};
    }

    auto sliced = std::make_shared<ov::op::v8::Slice>(
        input,
        ov::op::v0::Constant::create(ov::element::i64, {1}, {begin_val}),
        ov::op::v0::Constant::create(ov::element::i64, {1}, {end_val}),
        ov::op::v0::Constant::create(ov::element::i64, {1}, {1}),
        ov::op::v0::Constant::create(ov::element::i64, {1}, {slice_dim}));

    sliced->set_friendly_name(context.get_output_name());
    return {sliced->output(0)};
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
