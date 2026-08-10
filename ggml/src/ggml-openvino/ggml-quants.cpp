#include "ggml-quants.h"

#include "ggml-common.h"
#include "ggml-impl.h"
#include "ggml-openvino-extra.h"
#include "ggml.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/core/node.hpp>
#include <openvino/core/node_output.hpp>
#include <openvino/core/parallel.hpp>
#include <openvino/core/shape.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/element_type_traits.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/core/type/float4_e2m1.hpp>
#include <openvino/core/type/float8_e8m0.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/util/attr_types.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>
#include <vector>

static ov::Shape make_weight_shape(const ggml_tensor * tensor) {
    return (tensor->ne[2] > 1) ? ov::Shape{static_cast<size_t>(tensor->ne[2]), static_cast<size_t>(tensor->ne[1]),
                                           static_cast<size_t>(tensor->ne[0])} :
                                  ov::Shape{static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
}

static ov::Shape make_mxfp4_weight_shape(const ggml_tensor * tensor) {
    if (tensor->ne[2] == 1 && tensor->ne[3] == 1) {
        return {static_cast<size_t>(tensor->ne[1]), static_cast<size_t>(tensor->ne[0])};
    }

    ov::Shape shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }
    return shape;
}

static ov::Shape make_packed_mxfp4_moe_shape(const ggml_tensor * tensor) {
    return {static_cast<size_t>(tensor->ne[3]),
            static_cast<size_t>(tensor->ne[2]),
            static_cast<size_t>(tensor->ne[1]),
            static_cast<size_t>(tensor->ne[0] / MXFP4_BLOCK_SIZE),
            MXFP4_BLOCK_BYTES};
}

// Extract quantized weights from tensor and create weight subgraph
std::shared_ptr<ov::Node> extract_quantized_weights(const ggml_tensor * tensor,
                                                    const void * data,
                                                    ov::Tensor & weights,
                                                    ov::Tensor & scales,
                                                    ov::Tensor & zp,
                                                    bool use_bias) {
    // Create a temporary tensor for extraction functions that read from tensor->data
    ggml_tensor temp_tensor = *tensor;
    temp_tensor.data = const_cast<void *>(data);

    if (tensor->type == GGML_TYPE_MXFP4) {
        extract_mxfp4_data(&temp_tensor, weights, scales);
        auto result = make_mxfp4_weights(weights, scales).get_node_shared_ptr();
        result->set_friendly_name(tensor->name);
        return result;
    }

    // Determine block size based on tensor type
    int64_t weights_per_block;
    bool is_u4;
    switch (tensor->type) {
    case GGML_TYPE_Q4_0:
    case GGML_TYPE_Q4_1:
    case GGML_TYPE_Q4_K:
        is_u4 = true;
        weights_per_block = 32;
        break;
    case GGML_TYPE_Q8_0:
    case GGML_TYPE_Q5_1:
    case GGML_TYPE_Q5_K:
        is_u4 = false;
        weights_per_block = 32;
        break;
    case GGML_TYPE_Q6_K:
        is_u4 = false;
        weights_per_block = 16;
        break;
    default:
        throw std::runtime_error("Unsupported quantized type for extraction: " +
                                 std::string(ggml_type_name(tensor->type)));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point extraction
    // (see make_int8_weights/make_int4_weights) rather than the rounded integer zero point --
    // round(min/scale) error is what corrupts Q4_K/Q5_1 experts, and the f16-zp form still fuses
    // into GatherMatmulCompressed since it stays a Subtract, not an Add.
    const bool for_gather_matmul = tensor->ne[2] > 1;
    use_bias = use_bias || for_gather_matmul;

    // Extract quantized data
    switch (tensor->type) {
    case GGML_TYPE_Q4_0:
        extract_q4_0_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q4_1:
        extract_q4_1_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    case GGML_TYPE_Q4_K:
        extract_q4_k_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    case GGML_TYPE_Q5_1:
        extract_q5_1_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    case GGML_TYPE_Q8_0:
        extract_q8_0_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q6_K:
        extract_q6_k_data(&temp_tensor, weights, scales, zp);
        break;
    case GGML_TYPE_Q5_K:
        extract_q5_k_data(&temp_tensor, weights, scales, zp, use_bias);
        break;
    default:
        throw std::runtime_error("Unsupported quantized type: " + std::string(ggml_type_name(tensor->type)));
    }

    // Create the OpenVINO weight subgraph. 3D expert weights (MoE) are routed through the
    // GatherMatmul-oriented path: dequantized in f16, with constant folding disabled on the chain.
    ov::Output<ov::Node> weight_node;
    if (is_u4) {
        weight_node = make_int4_weights(weights, scales, zp, weights_per_block, use_bias, for_gather_matmul);
    } else {
        weight_node = make_int8_weights(weights, scales, zp, weights_per_block, use_bias, for_gather_matmul);
    }

    auto result = weight_node.get_node_shared_ptr();
    result->set_friendly_name(tensor->name);
    return result;
}

OvWeight process_weight_tensor(const ggml_tensor * tensor, const void * data, void * output_base_ptr, bool use_bias) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(data != nullptr);

    OvWeight result;

    // Get shape for weights: [rows, cols], or [n_expert, rows, cols] for 3D MoE expert weights.
    ov::Shape node_shape = make_weight_shape(tensor);

    // Handle F16/F32/BF16 weights
    if (tensor->type == GGML_TYPE_F32 || tensor->type == GGML_TYPE_F16 || tensor->type == GGML_TYPE_BF16) {
        ov::element::Type element_type;
        switch (tensor->type) {
        case GGML_TYPE_F32:
            element_type = ov::element::f32;
            break;
        case GGML_TYPE_F16:
            element_type = ov::element::f16;
            break;
        case GGML_TYPE_BF16:
            element_type = ov::element::bf16;
            break;
        default:
            OPENVINO_THROW("Unexpected tensor type in F16/F32/BF16 path");
        }

        if (output_base_ptr && output_base_ptr != data) {
            // Using external buffer - copy data and create shared-memory constant
            size_t tensor_bytes = ggml_nbytes(tensor);
            memcpy(output_base_ptr, data, tensor_bytes);
            result.weights = ov::Tensor(element_type, node_shape, output_base_ptr);
        } else {
            result.weights = ov::Tensor(element_type, node_shape, data);
        }
        result.weight_node = std::make_shared<ov::op::v0::Constant>(result.weights);
        return result;
    }

    // Handle quantized weights
    if (!ggml_is_quantized(tensor->type)) {
        OPENVINO_THROW("Unsupported weight tensor type: ", ggml_type_name(tensor->type));
    }

    result.layout = ggml_openvino_get_extracted_layout(tensor, use_bias);
    const auto & layout = result.layout;
    if (layout.total_size == 0) {
        OPENVINO_THROW("Unsupported quantized type: ", ggml_type_name(tensor->type));
    }

    // 3D MoE expert weights (for_gather_matmul) always use the exact f16 zero-point path (see
    // extract_quantized_weights) -- must be kept in sync with the "use_bias || for_gather_matmul"
    // check in ggml_openvino_get_extracted_layout, which sizes/offsets the zp slot accordingly.
    // Requantized tensors (layout.is_requant) are handled by requantize_to_buffers instead, whose
    // zp sizing/type is unaffected by for_gather_matmul, so they are excluded here.
    const bool for_gather_matmul = tensor->ne[2] > 1;
    const bool zp_is_f16 = !layout.is_requant && (use_bias || for_gather_matmul);

    const bool is_3d_mxfp4_moe = tensor->type == GGML_TYPE_MXFP4 && (tensor->ne[2] > 1 || tensor->ne[3] > 1);
    if (is_3d_mxfp4_moe) {
        ov::Shape packed_shape = make_packed_mxfp4_moe_shape(tensor);
        const size_t tensor_bytes = ggml_nbytes(tensor);
        if (output_base_ptr) {
            auto * buf_base = static_cast<uint8_t *>(output_base_ptr);
            memcpy(buf_base + layout.weights_offset, data, tensor_bytes);
            result.weights = ov::Tensor(ov::element::u8, packed_shape, buf_base + layout.weights_offset);
        } else {
            result.weights = ov::Tensor(ov::element::u8, packed_shape);
            memcpy(result.weights.data(), data, tensor_bytes);
        }
        result.weight_node = make_mxfp4_moe_packed_weights(result.weights).get_node_shared_ptr();
        result.weight_node->set_friendly_name(tensor->name);
        return result;
    }

    if (use_bias) {
        OPENVINO_ASSERT(!layout.is_requant,
                        "use_bias is only used for test-backend-ops, which should not have requantization");
        // bias node will be created on the fly and not use backend buffer
        output_base_ptr = nullptr;
    }

    // F16 requant path - no separate scales/zp needed in result
    if (layout.is_requant && layout.requant_type.has_value() && layout.requant_type.value() == ExtraQuantType::F16) {
        if (output_base_ptr) {
            result.weights = ov::Tensor(ov::element::f16, node_shape,
                                        static_cast<uint8_t *>(output_base_ptr) + layout.weights_offset);
        } else {
            result.weights = ov::Tensor(ov::element::f16, node_shape);
        }
        ov::Tensor dummy_scales, dummy_zp;  // Not used for F16
        result.weight_node =
            requantize_to_buffers(tensor, data, ExtraQuantType::F16, 0, result.weights, dummy_scales, dummy_zp);
        return result;
    }

    // Quantized path (normal extraction or quantized requant)
    // Create weight/scale/zp tensors - shared between both paths
    // For symmetric quantization, use signed types (i4/i8) and no ZP tensor
    ov::element::Type weight_type = tensor->type == GGML_TYPE_MXFP4 ?
                                        ov::element::f4e2m1 :
                                        (layout.is_symmetric ? (layout.is_u4 ? ov::element::i4 : ov::element::i8) :
                                                               (layout.is_u4 ? ov::element::u4 : ov::element::u8));
    ov::Shape scale_shape = node_shape;
    scale_shape.back() /= layout.weights_per_block;

    if (tensor->type == GGML_TYPE_MXFP4) {
        node_shape = make_mxfp4_weight_shape(tensor);
        scale_shape = node_shape;
        scale_shape.back() /= layout.weights_per_block;
    }

    if (output_base_ptr) {
        uint8_t * buf_base = static_cast<uint8_t *>(output_base_ptr);
        result.weights = ov::Tensor(weight_type, node_shape, buf_base + layout.weights_offset);
        const ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
        result.scales = ov::Tensor(scale_type, scale_shape, buf_base + layout.scales_offset);
        if (!layout.is_symmetric) {
            ov::element::Type zp_type =
                zp_is_f16 ? ov::element::f16 : (layout.is_u4 ? ov::element::u4 : ov::element::u8);
            result.zp = ov::Tensor(zp_type, scale_shape, buf_base + layout.zp_offset);
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    } else {
        result.weights = ov::Tensor(weight_type, node_shape);
        const ov::element::Type scale_type = tensor->type == GGML_TYPE_MXFP4 ? ov::element::f8e8m0 : ov::element::f16;
        result.scales = ov::Tensor(scale_type, scale_shape);
        if (!layout.is_symmetric) {
            if (zp_is_f16) {
                result.zp = ov::Tensor(ov::element::f16, scale_shape);
            } else {
                ov::element::Type zp_type = layout.is_u4 ? ov::element::u4 : ov::element::u8;
                result.zp = ov::Tensor(zp_type, scale_shape);
            }
        }
        // else: result.zp remains default-constructed (empty) for symmetric
    }

    if (layout.is_requant && layout.requant_type.has_value()) {
        result.weight_node = requantize_to_buffers(tensor, data, layout.requant_type.value(), layout.weights_per_block,
                                                   result.weights, result.scales, result.zp);
    } else {
        result.weight_node =
            extract_quantized_weights(tensor, data, result.weights, result.scales, result.zp, use_bias);
    }

    return result;
}

