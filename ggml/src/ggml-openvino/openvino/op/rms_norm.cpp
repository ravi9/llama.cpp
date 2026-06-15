#include "../node_context.h"
#include "../op_table.h"
#include "../utils.h"
#include "ggml-openvino/ggml-openvino-extra.h"

#include <cstdlib>
#include <memory>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/power.hpp>
#include <openvino/op/reduce_mean.hpp>
#include <openvino/op/sqrt.hpp>

namespace ov {
namespace frontend {
namespace ggml {
namespace op {

OutputVector translate_rms_norm(const NodeContext & context) {
    num_inputs_check(context, 1, 1);

    auto input_node = process_view_input_new(context, 0);

    // Build the mean-of-squares numerator. Normally use Power(x, 2): the OpenVINO
    // rms_fusion pass matches that Power node and folds the whole decomposition into
    // the internal RMS op (a perf win, e.g. dense Llama on GPU), so we keep it by
    // default for every model and device.
    //
    // EXCEPTION — gemma4 26B MoE on GPU with GGML_OPENVINO_GPU_FULL_MOE: the fused GPU
    // RMS primitive's dynamic multi-token kernel writes only token 0 (tokens 1..N read
    // back as 0). That silently collapses the per-layer MoE router RMSNorm summed over
    // the prefill tokens (~7x), flattening the router softmax and flipping the top-8
    // expert selection, so the GPU output drifts from CPU (task #16). On that exact
    // path only, compute the square as Multiply(x, x) — algebraically identical, but it
    // does not match the rms_fusion pattern, so the GPU runs the unfused primitives and
    // writes every token. Gated strictly to GPU + FULL_MOE so it never affects other
    // models (CPU/NPU and non-MoE GPU models keep the fused fast path).
    static const bool dodge_rms_fusion =
        ggml_openvino_get_device_name() == "GPU" && getenv("GGML_OPENVINO_GPU_FULL_MOE") != nullptr;

    std::shared_ptr<ov::Node> square;
    if (dodge_rms_fusion) {
        square = std::make_shared<ov::op::v1::Multiply>(input_node, input_node);
    } else {
        square = std::make_shared<ov::op::v1::Power>(
            input_node, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {2.0f}));
    }

    auto mean = std::make_shared<ov::op::v1::ReduceMean>(
        square, ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1}), true);

    float eps;
    memcpy(&eps, context.get_output_op_params(), sizeof(float));

    auto rms = std::make_shared<ov::op::v0::Sqrt>(
        std::make_shared<ov::op::v1::Add>(mean, ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {eps})));

    auto reciprocal =
        std::make_shared<ov::op::v1::Divide>(ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {1.0f}), rms);

    auto res = std::make_shared<ov::op::v1::Multiply>(input_node, reciprocal);

    return rename_outputs_with_suffix({res}, context.get_name());
}

}  // namespace op
}  // namespace ggml
}  // namespace frontend
}  // namespace ov
