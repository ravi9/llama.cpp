#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml-openvino/ggml-openvino-extra.h"

#include <cassert>
#include <cstdint>
#include <memory>
#include <numeric>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/type.hpp>
#include <openvino/frontend/exception.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/mod.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/scatter_elements_update.hpp>
#include <openvino/op/scatter_update.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <vector>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

// Conv state rollback snapshots of build_conv_state_store (see GgmlOvDecoder::is_conv_snapshot_set_rows).
// conv_x is [1, n_seqs, channels, n_cols + n]. Slot p < Kw = min(n, K) gets the window starting at column n - p,
// and its row for sequence q is p * rs_snapshot_stride + rs_slot_begin + q. Kw comes from the conv_x shape, so
// slot_order (src[1]) is not read and one model serves every Kw.
static OutputVector translate_conv_snapshot_set_rows(const NodeContext & context) {
    using namespace ov::op;
    auto i64_const = [](const std::vector<int64_t> & values) {
        return v0::Constant::create(ov::element::i64, {values.size()}, values);
    };
    auto scalar = [](int64_t value) { return v0::Constant::create(ov::element::i64, {}, {value}); };

    FRONT_END_OP_CONVERSION_CHECK(context.get_view_input_size(0) > 0 && context.get_view_input_size(2) > 0,
                                  "SET_ROWS conv snapshot needs windows and slots views");
    const auto windows_shape = context.get_view_input_ggml_shape(0, 0);  // [n_seqs, channels, Kw, n_cols]
    const auto slots_shape = context.get_view_input_ggml_shape(2, 0);    // [n_seqs, channels, K, n_cols]
    FRONT_END_OP_CONVERSION_CHECK(windows_shape.size() == 4 && slots_shape.size() == 4,
                                  "SET_ROWS conv snapshot needs 4-D windows and slots views");
    const int64_t channels = (int64_t) windows_shape[1];
    const int64_t n_cols = (int64_t) windows_shape[3];
    const int64_t K = (int64_t) slots_shape[2];

    auto conv_x = context.get_input(0);
    auto base = context.get_input(2);
    const auto base_shape = base.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(base_shape.rank().is_static() && base_shape[base_shape.size() - 1].is_static(),
                                  "SET_ROWS conv snapshot needs a static cache row size");
    const int64_t full_feature_size = base_shape[base_shape.size() - 1].get_length();
    const int64_t update_feature_size = channels * n_cols;

    // n = number of new tokens; in a padded static chunk only the valid ones count
    Output<Node> n;
    if (context.has_input("chunk_valid_len")) {
        n = context.get_input("chunk_valid_len");
    } else {
        auto total_cols = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(conv_x, ov::element::i64),
                                                       i64_const({3}), scalar(0));
        n = std::make_shared<v1::Subtract>(total_cols, i64_const({n_cols}));
    }
    auto kw = std::make_shared<v1::Minimum>(n, i64_const({K}));
    auto slot = std::make_shared<v4::Range>(scalar(0), std::make_shared<v0::Squeeze>(kw), scalar(1), ov::element::i64);

    std::vector<int64_t> offsets(n_cols);
    std::iota(offsets.begin(), offsets.end(), 0);
    auto col_begin = std::make_shared<v1::Subtract>(n, slot);  // [Kw]
    auto columns = std::make_shared<v1::Add>(std::make_shared<v0::Unsqueeze>(col_begin, scalar(1)),
                                             v0::Constant::create(ov::element::i64, {1, (size_t) n_cols}, offsets));
    auto windows = std::make_shared<v8::Gather>(conv_x, columns, scalar(3));  // [1, n_seqs, channels, Kw, n_cols]
    auto slot_major = std::make_shared<v1::Transpose>(windows, i64_const({0, 3, 1, 2, 4}));
    Output<Node> states =
        std::make_shared<v1::Reshape>(slot_major, i64_const({1, 1, -1, update_feature_size}), false);
    if (states.get_element_type() != context.get_output_type()) {
        states = std::make_shared<v0::Convert>(states, context.get_output_type());
    }

    // cache rows of the [Kw * n_seqs] states, slot major
    auto rows = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(states, ov::element::i64), i64_const({2}),
                                             scalar(0));
    auto indices = std::make_shared<v4::Range>(scalar(0), std::make_shared<v0::Squeeze>(rows), scalar(1),
                                               ov::element::i64);
    auto seq_count = context.get_input("rs_seq_count_" + context.get_name());
    auto plane = std::make_shared<v1::Divide>(indices, seq_count);
    auto sequence = std::make_shared<v1::Mod>(indices, seq_count);
    auto target = std::make_shared<v1::Add>(
        std::make_shared<v1::Multiply>(plane, context.get_input("rs_snapshot_stride_" + context.get_name())),
        std::make_shared<v1::Add>(sequence, context.get_input("rs_slot_begin_" + context.get_name())));

    auto axis = scalar(2);
    auto cache_rows = std::make_shared<v1::Reshape>(base, i64_const({1, 1, -1, full_feature_size}), false);
    if (update_feature_size < full_feature_size) {
        // the snapshot is a sub-range of each cache row (e.g. one of the KDA q/k/v conv states)
        const size_t elem_size = context.get_output_type().size();
        FRONT_END_OP_CONVERSION_CHECK(elem_size > 0, "SET_ROWS conv snapshot has an invalid element size");
        const int64_t feature_begin = (int64_t) (context.get_view_input_offset(2, 0) / elem_size) % full_feature_size;
        const int64_t feature_end = feature_begin + update_feature_size;
        FRONT_END_OP_CONVERSION_CHECK(feature_end <= full_feature_size,
                                      "SET_ROWS conv snapshot feature range is out of bounds");
        auto existing = std::make_shared<v8::Gather>(cache_rows, target, axis);
        auto feature_axis = i64_const({3});
        auto head = std::make_shared<v8::Slice>(existing, i64_const({0}), i64_const({feature_begin}), i64_const({1}),
                                                feature_axis);
        auto tail = std::make_shared<v8::Slice>(existing, i64_const({feature_end}), i64_const({full_feature_size}),
                                                i64_const({1}), feature_axis);
        states = std::make_shared<v0::Concat>(OutputVector{head, states, tail}, 3);
    }
    auto updated = std::make_shared<v3::ScatterUpdate>(cache_rows, target, states, axis);
    auto res = std::make_shared<v1::Reshape>(updated, std::make_shared<v3::ShapeOf>(base, ov::element::i64), false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

// SET_ROWS into a column range of a plain 2-D buffer (see GgmlOvDecoder::is_column_set_rows). The result is the
// whole buffer: the indexed rows get the new columns, the other columns and rows keep their values, so the next
// write into another column range of the same buffer reads this one.
static OutputVector translate_column_set_rows(const NodeContext & context) {
    using namespace ov::op;
    auto i64_const = [](const std::vector<int64_t> & values) {
        return v0::Constant::create(ov::element::i64, {values.size()}, values);
    };

    const auto view_shape = context.get_view_input_ggml_shape(2, 0);  // [1, 1, rows, width]
    FRONT_END_OP_CONVERSION_CHECK(view_shape.size() == 4, "SET_ROWS column write needs a 4-D destination view");
    const int64_t width = (int64_t) view_shape[3];

    auto base = context.get_input(2);
    const auto base_shape = base.get_partial_shape();
    FRONT_END_OP_CONVERSION_CHECK(base_shape.rank().is_static() && base_shape[base_shape.size() - 1].is_static(),
                                  "SET_ROWS column write needs a static buffer row size");
    const int64_t row_size = base_shape[base_shape.size() - 1].get_length();
    const size_t elem_size = context.get_output_type().size();
    const int64_t col_begin = (int64_t) (context.get_view_input_offset(2, 0) / elem_size) % row_size;
    const int64_t col_end = col_begin + width;
    FRONT_END_OP_CONVERSION_CHECK(col_end <= row_size, "SET_ROWS column range is out of bounds");

    Output<Node> data = std::make_shared<v1::Reshape>(process_view_input_new(context, 0), i64_const({1, 1, -1, width}),
                                                      false);
    if (data.get_element_type() != context.get_output_type()) {
        data = std::make_shared<v0::Convert>(data, context.get_output_type());
    }
    Output<Node> rows = std::make_shared<v1::Reshape>(process_view_input_new(context, 1), i64_const({-1}), false);
    if (rows.get_element_type() != ov::element::i64) {
        rows = std::make_shared<v0::Convert>(rows, ov::element::i64);
    }

    auto axis = v0::Constant::create(ov::element::i64, {}, {2});
    auto buffer = std::make_shared<v1::Reshape>(base, i64_const({1, 1, -1, row_size}), false);
    Output<Node> updates = data;
    if (width < row_size) {
        auto existing = std::make_shared<v8::Gather>(buffer, rows, axis);
        auto col_axis = i64_const({3});
        OutputVector parts;
        if (col_begin > 0) {
            parts.push_back(std::make_shared<v8::Slice>(existing, i64_const({0}), i64_const({col_begin}),
                                                        i64_const({1}), col_axis));
        }
        parts.push_back(data);
        if (col_end < row_size) {
            parts.push_back(std::make_shared<v8::Slice>(existing, i64_const({col_end}), i64_const({row_size}),
                                                        i64_const({1}), col_axis));
        }
        updates = std::make_shared<v0::Concat>(parts, 3);
    }
    auto updated = std::make_shared<v3::ScatterUpdate>(buffer, rows, updates, axis);
    auto res = std::make_shared<v1::Reshape>(updated, std::make_shared<v3::ShapeOf>(base, ov::element::i64), false);
    return rename_outputs_with_suffix({res}, context.get_name());
}

OutputVector translate_set_rows(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    if (context.get_op_case() == 2) {
        return translate_conv_snapshot_set_rows(context);
    }
    if (context.get_op_case() == 3) {
        return translate_column_set_rows(context);
    }

    auto data = process_view_input_new(context, 0);
    auto indices = process_view_input_new(context, 1);

    data = std::make_shared<ov::op::v0::Convert>(data, context.get_output_type());

    const auto indices_shape = context.get_input_shape(1);
    const bool multidim_indices = indices_shape.rank().is_static() &&
                                  indices_shape.rank().get_length() == 4 &&
                                  ((indices_shape[1].is_static() && indices_shape[1].get_length() > 1) ||
                                   (indices_shape[2].is_static() && indices_shape[2].get_length() > 1));

    // On NPU ScatterUpdate is not in-place: it copies the whole destination to write
    // one KV row, which dominates decode once ctx grows. ScatterElementsUpdate is
    // measurably cheaper for the same single-row write.
    const bool use_elements_update =
        !multidim_indices && !context.is_stateful() && ggml_openvino_getenv_int("GGML_OPENVINO_KV_SCATTER_ELEMENTS");

    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});

    if (context.is_static() && !context.is_stateful() && !multidim_indices &&
        ggml_openvino_npu_kv_head_major_enabled()) {
        // Head-major cache [1, heads, rows, head_size]: write the new rows along axis 2 without reshaping the cache.
        ov::Output<Node> cache = context.get_input(2);
        while (auto reshape = ov::as_type_ptr<ov::op::v1::Reshape>(cache.get_node_shared_ptr())) {
            cache = reshape->input_value(0);
        }
        const auto cache_shape = cache.get_partial_shape();
        if (cache_shape.rank().is_static() && cache_shape.rank().get_length() == 4 && cache_shape[1].is_static() &&
            cache_shape[3].is_static() && cache_shape[1].get_length() > 1) {
            const int64_t heads = cache_shape[1].get_length();
            // Transposed V cache is [1, heads, head_size, rows]: rows go along axis 3.
            const bool v_transposed = cache.get_node()->get_friendly_name().rfind("cache_v_l", 0) == 0;
            const int64_t head_size = (v_transposed ? cache_shape[2] : cache_shape[3]).get_length();
            auto ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
                indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
            auto rows = std::make_shared<ov::op::v1::Reshape>(
                data,
                ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, heads, head_size}),
                false);
            auto rows_bhsd = std::make_shared<ov::op::v1::Transpose>(
                rows, ov::op::v0::Constant::create(ov::element::i64, {4},
                                                   v_transposed ? std::vector<int64_t>{0, 2, 3, 1} :
                                                                  std::vector<int64_t>{0, 2, 1, 3}));
            if (v_transposed && context.is_prefill() && ggml_openvino_npu_hfa_prefill_enabled()) {
                // On NPU a chunk-sized scatter along the innermost axis of [1, heads, head_size, rows] is very slow:
                // the HFA prefill outputs the chunk's V^T rows [1, heads, head_size, n] instead and the host copies
                // them into the cache (as GenAI does for its prefill KV).
                return rename_outputs_with_suffix({rows_bhsd}, context.get_name());
            }
            Output<Node> res = std::make_shared<ov::op::v3::ScatterUpdate>(
                cache, ind_squeezed, rows_bhsd,
                v_transposed ? ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {3}) : axes);
            return rename_outputs_with_suffix({res}, context.get_name());
        }
    }

    auto dst = process_view_input_new(context, 2);
    Output<Node> res;
    if (context.is_stateful()) {
        int concat_axis = 1;
        if (context.get_op_case() == 1) {
            // `-fa off` writes V transposed through a [1, n] reshape of the cache. The state keeps V token-major
            // like K, and the new values are already in token order, so append them to the state itself
            dst = dst.get_node()->input_value(0);
        }
        FRONT_END_OP_CONVERSION_CHECK(dst.get_partial_shape()[2].is_static() && dst.get_partial_shape()[3].is_static(),
                                      "Unsupported KV cache layout for stateful execution");
        int64_t dim2 = dst.get_partial_shape()[2].get_length();
        int64_t dim3 = dst.get_partial_shape()[3].get_length();
        data = std::make_shared<ov::op::v1::Reshape>(
            data, ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, dim2, dim3}), false);
        res = std::make_shared<ov::op::v0::Concat>(OutputVector{dst, data}, concat_axis);
    } else if (is_kvcache_4d_shape(dst.get_partial_shape())) {
        // cache [1, ctx, n_heads_kv, head_size]: write whole tokens on axis 1. This is the pattern of the GPU
        // plugin's StatelessKVFusion (1-D axis, read with VariadicSplit, see permute), which writes the new rows in
        // place instead of copying the whole cache first
        FRONT_END_OP_CONVERSION_CHECK(!multidim_indices, "SET_ROWS into a 4-D KV cache needs one index per token");
        const int64_t n_heads_kv = dst.get_partial_shape()[2].get_length();
        const int64_t head_size = dst.get_partial_shape()[3].get_length();
        auto data_tokens = std::make_shared<ov::op::v1::Reshape>(
            data,
            ov::op::v0::Constant::create(ov::element::i64, {4}, {(int64_t) 1, (int64_t) -1, n_heads_kv, head_size}),
            false);
        auto ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, data_tokens,
                                                          ov::op::v0::Constant::create(ov::element::i64, {1}, {1}));
        return rename_outputs_with_suffix({res}, context.get_name());
    } else if (multidim_indices) {
        auto updates_shape = std::make_shared<ov::op::v3::ShapeOf>(data, ov::element::i64);

        auto indices_rank3 = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {1}, {0}));
        auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
        auto indices_rank4_shape = std::make_shared<ov::op::v0::Concat>(OutputVector{get_dimensions(updates_shape, {0, 1, 2}), one}, 0);
        auto indices_rank4 = std::make_shared<ov::op::v1::Reshape>(indices_rank3, indices_rank4_shape, false);
        auto broadcasted_indices = std::make_shared<ov::op::v3::Broadcast>(indices_rank4, updates_shape);

        res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, data, axes);
    } else {
        auto row_size = context.get_input_shape(2)[3].get_length();
        auto ind_squeezed = std::make_shared<ov::op::v0::Squeeze>(
            indices, ov::op::v0::Constant::create(ov::element::i64, {3}, {0, 1, 2}));
        // skip the reshape when the rows already have that shape: it costs a GPU primitive per cache write
        const auto & data_shape = data.get_partial_shape();
        const bool data_is_rows = data_shape.rank().is_static() && data_shape.rank().get_length() == 4 &&
                                  data_shape[0] == 1 && data_shape[1] == 1 && data_shape[3] == row_size;
        ov::Output<ov::Node> data_reshaped = data;
        if (!data_is_rows) {
            data_reshaped = std::make_shared<ov::op::v1::Reshape>(
                data,
                ov::op::v0::Constant::create(ov::element::i64, {4},
                                             {(int64_t) 1, (int64_t) 1, (int64_t) -1, (int64_t) row_size}),
                false);
        }
        if (use_elements_update) {
            auto updates_shape = std::make_shared<ov::op::v3::ShapeOf>(data_reshaped, ov::element::i64);
            auto ind_rank4 = std::make_shared<ov::op::v1::Reshape>(
                ind_squeezed, ov::op::v0::Constant::create(ov::element::i64, {4}, {1, 1, -1, 1}), false);
            auto broadcasted_indices = std::make_shared<ov::op::v3::Broadcast>(ind_rank4, updates_shape);
            res = std::make_shared<ov::op::v3::ScatterElementsUpdate>(dst, broadcasted_indices, data_reshaped, axes);
        } else {
            res = std::make_shared<ov::op::v3::ScatterUpdate>(dst, ind_squeezed, data_reshaped, axes);
        }
    }

    auto dst_reshape = ov::as_type_ptr<ov::op::v1::Reshape>(dst.get_node_shared_ptr());
    if (!multidim_indices && dst_reshape) {
        // Fix the case of multiple sequences, reshape back to original shape [1, n_seq, ctx_per_seq, emb]
        // ctx_per_seq is not fixed due to llama-bench compatibility
        auto dst_shape_partial = dst_reshape->get_input_partial_shape(0);
        std::vector<int64_t> dst_shape = {dst_shape_partial[0].get_length(), dst_shape_partial[1].get_length(),
                                          dst_shape_partial[2].is_static() ? dst_shape_partial[2].get_length() : -1,
                                          dst_shape_partial[3].get_length()};
        res = std::make_shared<ov::op::v1::Reshape>(res, ov::op::v0::Constant::create(ov::element::i64, {4}, dst_shape),
                                                    false);
    }
    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
