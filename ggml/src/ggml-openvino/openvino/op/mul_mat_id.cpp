#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"

#include <memory>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/gather.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/shape_of.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/unsqueeze.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_mul_mat_id(const NodeContext & context) {
    num_inputs_check(context, 3, 3);

    auto expert_weights = process_view_input_new(context, 0);
    auto activations = process_view_input_new(context, 1);
    auto ids = process_view_input_new(context, 2);

    // OpenVINO sees GGML tensors in reversed dimension order:
    //   activations: [1, n_tokens, n_used_or_1, k]
    //   ids: [1, 1, n_tokens, n_used]
    // The expert weights node is built specially in GgmlOvDecoder::create_weight_node
    // as a rank-2 [n_expert, m*k] dequantization subgraph (Constant(u4)->Convert->
    // [Subtract]->Multiply->Reshape(3D->2D)->Convert). We MUST gather experts directly
    // on this rank-2 node so the CPU plugin can fold the Gather + dequant into a single
    // GatherCompressed op (keeping the weights compressed and decompressing only the
    // selected experts at runtime). Reshaping the weights to [n_expert,m,k] before the
    // Gather would break that fusion and cause the plugin to materialize all experts as
    // f32 at compile time → OOM. So we gather on [n_expert, m*k] and split m*k -> m,k on
    // the gathered result afterwards.
    auto activations_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape_4d = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);

    auto activations_shape_3d = get_dimensions(activations_shape_4d, {1, 2, 3});
    auto ids_shape_2d = get_dimensions(ids_shape_4d, {2, 3});

    activations = std::make_shared<ov::op::v1::Reshape>(activations, activations_shape_3d, false);
    ids = std::make_shared<ov::op::v1::Reshape>(ids, ids_shape_2d, false);

    if (ids.get_element_type() != ov::element::i32 && ids.get_element_type() != ov::element::i64) {
        ids = std::make_shared<ov::op::v0::Convert>(ids, ov::element::i32);
    }

    // m (output row dim) is static; k = (m*k) / m. Gather experts on axis 0 of the
    // rank-2 [n_expert, m*k] weight -> [n_tokens, n_used, m*k], then split to
    // [n_tokens, n_used, m, k].
    const auto output_type = context.get_output_type();
    const auto mm_output_shape = context.get_output_shape();
    FRONT_END_OP_CONVERSION_CHECK(mm_output_shape.rank().is_static() && mm_output_shape.rank().get_length() == 4,
                                  "Unexpected MUL_MAT_ID output rank");
    FRONT_END_OP_CONVERSION_CHECK(mm_output_shape[3].is_static(),
                                  "Expected static row dimension (m) for MUL_MAT_ID output");
    const int64_t m_value = mm_output_shape[3].get_length();

    // Normalize the weight to rank-2 [n_expert, m*k] so the expert Gather sits on a
    // 2D node (required for the GatherCompressed fusion). The quantized expert path in
    // GgmlOvDecoder::create_weight_node already produces [n_expert, m*k]. The
    // non-quantized path (f32/f16 experts, e.g. test-backend-ops) produces a rank-4
    // [1, n_expert, m, k] constant; collapse it to [n_expert, m*k] here.
    if (expert_weights.get_partial_shape().rank().is_static() &&
        expert_weights.get_partial_shape().rank().get_length() != 2) {
        auto w_shape = std::make_shared<ov::op::v3::ShapeOf>(expert_weights, ov::element::i64);
        auto n_expert_dim = get_dimensions(w_shape, {1});
        auto flat_w_dims = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{n_expert_dim, ov::op::v0::Constant::create(ov::element::i64, {1}, {-1})}, 0);
        expert_weights = std::make_shared<ov::op::v1::Reshape>(expert_weights, flat_w_dims, false);
    }

    auto gather_axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0});
    ov::Output<ov::Node> selected_weights = std::make_shared<ov::op::v8::Gather>(expert_weights, ids, gather_axis);

    if (selected_weights.get_element_type() != ov::element::f32) {
        selected_weights = std::make_shared<ov::op::v0::Convert>(selected_weights, ov::element::f32);
    }

    // Split the flattened m*k expert rows into [m, k]: reshape gathered
    // [n_tokens, n_used, m*k] -> [n_tokens, n_used, m, -1].
    auto sel_ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    auto split_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            get_dimensions(sel_ids_shape, {0, 1}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {m_value}),
            ov::op::v0::Constant::create(ov::element::i64, {1}, {-1}),
        },
        0);
    selected_weights = std::make_shared<ov::op::v1::Reshape>(selected_weights, split_target_dims, false);
    if (activations.get_element_type() != ov::element::f32) {
        activations = std::make_shared<ov::op::v0::Convert>(activations, ov::element::f32);
    }

    auto activations_shape = std::make_shared<ov::op::v3::ShapeOf>(activations, ov::element::i64);
    auto ids_shape = std::make_shared<ov::op::v3::ShapeOf>(ids, ov::element::i64);
    ov::Output<ov::Node> acts_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            get_dimensions(activations_shape, {0}),
            get_dimensions(ids_shape, {1}),
            get_dimensions(activations_shape, {2}),
        },
        0);
    ov::Output<ov::Node> acts_broadcasted = std::make_shared<ov::op::v3::Broadcast>(activations, acts_target_dims,
                                                                                     ov::op::BroadcastType::BIDIRECTIONAL);

    auto unsqueeze_axes = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});
    auto activations_expanded = std::make_shared<ov::op::v0::Unsqueeze>(acts_broadcasted, unsqueeze_axes);

    auto batch_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto row_dim = ov::op::v0::Constant::create(ov::element::i64, {1}, {m_value});

    ov::Output<ov::Node> result =
        std::make_shared<ov::op::v0::MatMul>(activations_expanded, selected_weights, false, true);

    auto result_target_dims = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{
            batch_dim,
            get_dimensions(ids_shape, {0, 1}),
            row_dim,
        },
        0);
    result = std::make_shared<ov::op::v1::Reshape>(result, result_target_dims, false);

    if (result.get_element_type() != output_type) {
        result = std::make_shared<ov::op::v0::Convert>(result, output_type);
    }

    return rename_outputs_with_suffix({result}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
