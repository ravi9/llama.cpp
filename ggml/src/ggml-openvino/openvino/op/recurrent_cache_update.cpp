#include "recurrent_cache_update.hpp"

#include <algorithm>
#include <cstdio>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/maximum.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_min.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/strided_slice.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/util/precision_sensitive_attribute.hpp>
#include <string>

namespace ov::frontend::ggml::op {

namespace {

constexpr size_t STAGE_INPUTS = 6;

void validate_stage_inputs(const ov::Node * node, size_t first) {
    for (size_t i = first; i < first + STAGE_INPUTS; ++i) {
        NODE_VALIDATION_CHECK(node, node->get_input_partial_shape(i).rank().compatible(4),
                              "Recurrent cache inputs require rank 4");
    }
    const auto type = node->get_input_element_type(first);
    NODE_VALIDATION_CHECK(node, type == element::f32 || type == element::f16, "Recurrent cache requires f32 or f16");
}

// Custom kernel defines are emitted verbatim; keep float literals single precision.
std::string float_literal(float value) {
    char buf[32];
    snprintf(buf, sizeof(buf), "%.9ef", value);
    return buf;
}

Output<Node> i64_const(const std::vector<int64_t> & values, const Shape & shape) {
    return ov::op::v0::Constant::create(element::i64, shape, values);
}

Output<Node> dim_of(const Output<Node> & x, int64_t axis) {
    return std::make_shared<ov::op::v8::Gather>(std::make_shared<ov::op::v3::ShapeOf>(x, element::i64),
                                                i64_const({axis}, {1}), i64_const({0}, {}));
}

// Custom kernels compile once per input shape (dims are JIT constants). A dynamic token axis
// is padded to the next power of two and the real count passed as a [1,1,1,1] input.
struct TokenBucket {
    Output<Node> count;  // [1]
    Output<Node> pad;    // [1], tokens appended by pad_tokens()
    Output<Node> meta;   // [1,1,1,1]
    bool dynamic = false;
};

TokenBucket make_token_bucket(const Output<Node> & x, int64_t axis) {
    TokenBucket bucket;
    const auto & dim = x.get_partial_shape()[axis];
    bucket.dynamic = dim.is_dynamic();
    bucket.count = dim.is_dynamic() ? dim_of(x, axis) : i64_const({dim.get_length()}, {1});
    bucket.meta = std::make_shared<ov::op::v1::Reshape>(bucket.count, i64_const({1, 1, 1, 1}, {4}), false);
    if (bucket.dynamic) {
        // Smallest power of two >= count, using only arithmetic so the GPU plugin evaluates it
        // on the host as a shape subgraph (comparisons would run on the GPU and force a sync).
        std::vector<int64_t> sizes;
        for (int64_t i = 1; i <= (int64_t(1) << 20); i <<= 1) {
            sizes.push_back(i);
        }
        auto candidates = i64_const(sizes, {sizes.size()});
        auto short_by = std::make_shared<ov::op::v1::Subtract>(bucket.count, candidates);
        auto too_small = std::make_shared<ov::op::v1::Minimum>(
            std::make_shared<ov::op::v1::Maximum>(short_by, i64_const({0}, {1})), i64_const({1}, {1}));
        auto masked = std::make_shared<ov::op::v1::Add>(
            candidates, std::make_shared<ov::op::v1::Multiply>(too_small, i64_const({int64_t(1) << 24}, {1})));
        auto size = std::make_shared<ov::op::v1::ReduceMin>(masked, i64_const({0}, {1}), true);
        bucket.pad = std::make_shared<ov::op::v1::Subtract>(size, bucket.count);
    }
    return bucket;
}

Output<Node> pad_tokens(const Output<Node> & x, int64_t axis, const TokenBucket & bucket) {
    if (!bucket.dynamic) {
        return x;
    }
    const auto & shape = x.get_partial_shape();
    const auto rank = shape.rank().get_length();
    OutputVector dims;
    std::vector<int64_t> pattern(rank, 0);
    for (int64_t i = 0; i < rank; ++i) {
        dims.push_back(i == axis ? bucket.pad : dim_of(x, i));
        pattern[i] = i == axis ? -1 : (shape[i].is_static() ? shape[i].get_length() : 0);
    }
    auto zeros = std::make_shared<ov::op::v3::Broadcast>(ov::op::v0::Constant::create(x.get_element_type(), {}, {0}),
                                                         std::make_shared<ov::op::v0::Concat>(dims, 0));
    auto padded = std::make_shared<ov::op::v0::Concat>(OutputVector{x, zeros}, axis);
    // Keep the static dims of x for shape inference.
    return std::make_shared<ov::op::v1::Reshape>(padded, i64_const(pattern, {size_t(rank)}), true);
}

Output<Node> slice_tokens(const Output<Node> & x, int64_t axis, const Output<Node> & length, bool dynamic) {
    if (!dynamic) {
        return x;
    }
    // StridedSlice with end masks on the other axes: the GPU plugin skips it when nothing is cut.
    const auto rank = x.get_partial_shape().rank().get_length();
    OutputVector end;
    std::vector<int64_t> end_mask(rank, 1);
    for (int64_t i = 0; i < rank; ++i) {
        end.push_back(i == axis ? length : i64_const({0}, {1}));
    }
    end_mask[axis] = 0;
    return std::make_shared<ov::op::v1::StridedSlice>(
        x, i64_const(std::vector<int64_t>(rank, 0), {size_t(rank)}), std::make_shared<ov::op::v0::Concat>(end, 0),
        i64_const(std::vector<int64_t>(rank, 1), {size_t(rank)}), std::vector<int64_t>(rank, 1), end_mask);
}

int64_t static_dim(const ov::Node * node, size_t input, size_t axis) {
    if (node->get_input_size() <= input) {
        return 0;
    }
    const auto & shape = node->get_input_partial_shape(input);
    return shape.rank().is_static() && shape[axis].is_static() ? shape[axis].get_length() : 0;
}

}  // namespace

RecurrentCacheStage::RecurrentCacheStage(const OutputVector & args) : Op(args) {
    constructor_validate_and_infer_types();
}

void RecurrentCacheStage::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == STAGE_INPUTS, "Recurrent cache stage requires 6 inputs");
    validate_stage_inputs(this, 0);
    const auto & cache = get_input_partial_shape(0);
    const auto & active = get_input_partial_shape(1);
    const auto & extra = get_input_partial_shape(2);
    set_output_type(0, element::f32, PartialShape{1, 1, active[3] + extra[3], cache[3]});
}

std::shared_ptr<Node> RecurrentCacheStage::clone_with_new_inputs(const OutputVector & args) const {
    check_new_args_count(this, args);
    return std::make_shared<RecurrentCacheStage>(args);
}

RecurrentConvUpdate::RecurrentConvUpdate(const OutputVector & args, int64_t history) : Op(args), m_history(history) {
    constructor_validate_and_infer_types();
}

void RecurrentConvUpdate::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == STAGE_INPUTS + 5, "Recurrent convolution requires 11 inputs");
    NODE_VALIDATION_CHECK(this, m_history > 0, "History must be positive");
    validate_stage_inputs(this, 0);
    const auto & qkv = get_input_partial_shape(6);
    NODE_VALIDATION_CHECK(this, qkv.rank().compatible(4), "QKV requires [1,B,C,T]");
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(0)[3].compatible(qkv[2] * m_history),
                          "Cache features must equal channels times history");
    set_output_type(0, get_input_element_type(6), PartialShape{1, qkv[1], qkv[3], qkv[2]});
}

bool RecurrentConvUpdate::visit_attributes(AttributeVisitor & visitor) {
    visitor.on_attribute("history", m_history);
    return true;
}

std::shared_ptr<Node> RecurrentConvUpdate::clone_with_new_inputs(const OutputVector & args) const {
    check_new_args_count(this, args);
    return std::make_shared<RecurrentConvUpdate>(args, m_history);
}

RecurrentGdnUpdate::RecurrentGdnUpdate(const OutputVector & args,
                                       bool fuse_qk_l2norm,
                                       float q_eps,
                                       float k_eps,
                                       int64_t snapshot_count) :
    Op(args),
    m_fuse_qk_l2norm(fuse_qk_l2norm),
    m_q_eps(q_eps),
    m_k_eps(k_eps),
    m_snapshot_count(snapshot_count) {
    constructor_validate_and_infer_types();
}

void RecurrentGdnUpdate::validate_and_infer_types() {
    NODE_VALIDATION_CHECK(this, get_input_size() == 14, "Recurrent GDN requires 14 inputs");
    NODE_VALIDATION_CHECK(this, m_snapshot_count >= 1, "snapshot_count must be positive");
    const auto & q = get_input_partial_shape(0);
    const auto & v = get_input_partial_shape(2);
    NODE_VALIDATION_CHECK(this, q.rank().compatible(4) && v.rank().compatible(4), "Q/V require rank 4");
    NODE_VALIDATION_CHECK(this, q.rank().is_static() && q[3].is_static() && v.rank().is_static() && v[3].is_static(),
                          "Head sizes must be static");
    NODE_VALIDATION_CHECK(this, get_input_partial_shape(3)[3].compatible(v[2] * q[3] * v[3]),
                          "Cache features must match H*V*K");
    const auto type = get_input_element_type(3);
    NODE_VALIDATION_CHECK(this, type == element::f32 || type == element::f16, "Recurrent cache requires f32 or f16");
    set_output_type(0, get_input_element_type(2), PartialShape{q[0], q[1], v[2], v[3]});
}

bool RecurrentGdnUpdate::visit_attributes(AttributeVisitor & visitor) {
    int64_t k_head_dim = static_dim(this, 0, 3);
    int64_t v_head_dim = static_dim(this, 2, 3);
    std::string q_eps = float_literal(m_q_eps), k_eps = float_literal(m_k_eps);
    visitor.on_attribute("fuse_qk_l2norm", m_fuse_qk_l2norm);
    visitor.on_attribute("q_l2_norm_eps", q_eps);
    visitor.on_attribute("k_l2_norm_eps", k_eps);
    visitor.on_attribute("snapshot_count", m_snapshot_count);
    visitor.on_attribute("k_head_dim", k_head_dim);
    visitor.on_attribute("v_head_dim", v_head_dim);
    m_q_eps = std::stof(q_eps);
    m_k_eps = std::stof(k_eps);
    return true;
}

std::shared_ptr<Node> RecurrentGdnUpdate::clone_with_new_inputs(const OutputVector & args) const {
    check_new_args_count(this, args);
    return std::make_shared<RecurrentGdnUpdate>(args, m_fuse_qk_l2norm, m_q_eps, m_k_eps, m_snapshot_count);
}

RecurrentGdnSnapshots::RecurrentGdnSnapshots(const OutputVector & args,
                                             bool fuse_qk_l2norm,
                                             float q_eps,
                                             float k_eps,
                                             int64_t snapshot_count) :
    RecurrentGdnUpdate(args, fuse_qk_l2norm, q_eps, k_eps, snapshot_count) {
    constructor_validate_and_infer_types();
}

void RecurrentGdnSnapshots::validate_and_infer_types() {
    RecurrentGdnUpdate::validate_and_infer_types();
    const auto & q = get_input_partial_shape(0);
    Dimension rows = Dimension::dynamic();
    if (q[0].is_static() && q[1].is_static()) {
        rows = q[0].get_length() * std::min(q[1].get_length(), m_snapshot_count);
    }
    set_output_size(2);
    set_output_type(1, get_input_element_type(3), PartialShape{1, 1, rows, get_input_partial_shape(3)[3]});
}

std::shared_ptr<Node> RecurrentGdnSnapshots::clone_with_new_inputs(const OutputVector & args) const {
    check_new_args_count(this, args);
    return std::make_shared<RecurrentGdnSnapshots>(args, m_fuse_qk_l2norm, m_q_eps, m_k_eps, m_snapshot_count);
}

bool is_inplace_cache(const Output<Node> & cache) {
    // Reshape keeps the element count, so the chain always covers the whole allocation.
    auto node = cache.get_node_shared_ptr();
    while (ov::is_type<ov::op::v1::Reshape>(node)) {
        node = node->get_input_node_shared_ptr(0);
    }
    return ov::is_type<ov::op::v0::Parameter>(node);
}

void mark_inplace_cache(const Input<Node> & input) {
    // Precision-sensitive inputs keep their producer subgraph in the original precision.
    ov::mark_as_precision_sensitive(input);
}

Output<Node> make_cache_stage(const OutputVector & inputs) {
    auto stage = std::make_shared<RecurrentCacheStage>(OutputVector(inputs.begin(), inputs.begin() + STAGE_INPUTS));
    mark_inplace_cache(stage->input(0));
    return stage;
}

Output<Node> make_conv_update(OutputVector inputs, int64_t history, const std::string & name) {
    auto bucket = make_token_bucket(inputs[6], 3);
    inputs[6] = pad_tokens(inputs[6], 3, bucket);
    inputs.push_back(bucket.meta);
    auto conv = std::make_shared<RecurrentConvUpdate>(inputs, history);
    conv->set_friendly_name(name);
    mark_inplace_cache(conv->input(0));
    return slice_tokens(conv, 2, bucket.count, bucket.dynamic);
}

Output<Node> make_gdn_update(OutputVector inputs,
                             bool fuse_qk_l2norm,
                             float q_eps,
                             float k_eps,
                             int64_t snapshot_count,
                             const std::string & name) {
    auto bucket = make_token_bucket(inputs[0], 1);
    for (size_t i : {0, 1, 2, 4, 5}) {
        inputs[i] = pad_tokens(inputs[i], 1, bucket);
    }
    inputs.push_back(bucket.meta);
    auto gdn = std::make_shared<RecurrentGdnUpdate>(inputs, fuse_qk_l2norm, q_eps, k_eps, snapshot_count);
    gdn->set_friendly_name(name);
    mark_inplace_cache(gdn->input(3));
    return slice_tokens(gdn, 1, bucket.count, bucket.dynamic);
}

OutputVector make_gdn_snapshots(OutputVector inputs,
                                bool fuse_qk_l2norm,
                                float q_eps,
                                float k_eps,
                                int64_t snapshot_count) {
    auto bucket = make_token_bucket(inputs[0], 1);
    auto batch = dim_of(inputs[0], 0);
    for (size_t i : {0, 1, 2, 4, 5}) {
        inputs[i] = pad_tokens(inputs[i], 1, bucket);
    }
    inputs.push_back(bucket.meta);
    auto gdn = std::make_shared<RecurrentGdnSnapshots>(inputs, fuse_qk_l2norm, q_eps, k_eps, snapshot_count);
    // Snapshot rows are slot-major, so the rows of real tokens come first.
    auto rows = std::make_shared<ov::op::v1::Multiply>(
        batch, std::make_shared<ov::op::v1::Minimum>(bucket.count, i64_const({snapshot_count}, {1})));
    return {slice_tokens(gdn->output(0), 1, bucket.count, bucket.dynamic),
            slice_tokens(gdn->output(1), 2, rows, bucket.dynamic)};
}

}  // namespace ov::frontend::ggml::op
