#include "ggml-quants.h"

#include <memory>
#include <openvino/core/except.hpp>
#include <openvino/core/type/element_type.hpp>
#include <openvino/core/type/float16.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/subtract.hpp>
#include <openvino/op/util/attr_types.hpp>
#include <openvino/runtime/tensor.hpp>

namespace {

enum class ZeroPointMode { None, Integer, ExactBiasAsF16ZeroPoint };

ov::Output<ov::Node> make_integer_weights(ov::Tensor & weight,
                                          ov::Tensor & scales,
                                          ov::Tensor & zp,
                                          size_t group_size,
                                          bool use_bias,
                                          bool for_gather_matmul,
                                          ov::element::Type signed_type,
                                          ov::element::Type unsigned_type) {
    ov::Shape orig_shape = weight.get_shape();
    bool is_signed = (weight.get_element_type() == signed_type);  // Symmetric: signed weights, no ZP
    const ZeroPointMode zp_mode = is_signed ? ZeroPointMode::None :
                                  (use_bias && zp.get_size() > 0) ? ZeroPointMode::ExactBiasAsF16ZeroPoint :
                                                                    ZeroPointMode::Integer;

    // Expand dimensions for scales and zp/bias
    auto scale_shape = scales.get_shape();

    // Group the innermost (last) dimension. For 2D weights [rows, cols] this yields
    // [rows, cols/group_size, group_size]; for 3D MoE experts [n_expert, rows, cols] this yields
    // [n_expert, rows, cols/group_size, group_size].
    ov::Shape packed_shape = orig_shape;
    packed_shape.back() /= group_size;
    packed_shape.push_back(group_size);
    const size_t group_dim = packed_shape.size() - 2;

    if (packed_shape[group_dim] == 1) {
        // Requantized channel-wise case
        packed_shape.erase(packed_shape.begin() + group_dim);
    } else {
        scale_shape.push_back(1);
        scales.set_shape(scale_shape);
        if (!is_signed && zp.get_size() > 0) {
            auto zp_shape = zp.get_shape();
            zp_shape.push_back(1);
            zp.set_shape(zp_shape);
        }
    }

    auto scales_f16 = std::make_shared<ov::op::v0::Constant>(scales);

    ov::Output<ov::Node> result;
    if (is_signed) {
        // Signed path: q * s (no zero point subtraction needed)
        auto weights_node = std::make_shared<ov::op::v0::Constant>(signed_type, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);
        auto mul = std::make_shared<ov::op::v1::Multiply>(weights_f16, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        result = mul;
    } else {
        // Unsigned path
        auto weights_node = std::make_shared<ov::op::v0::Constant>(unsigned_type, packed_shape,
                                                                   static_cast<uint8_t *>(weight.data()), nullptr);
        weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
        auto weights_f16 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f16);

        if (zp_mode == ZeroPointMode::ExactBiasAsF16ZeroPoint) {
            // Accurate dequant in the FUSABLE zero-point form: (w - zp) * s, where the zero
            // point is an exact f16 value zp = -bias/scale (the zp tensor holds bias values
            // coming in). Algebraically equal to w*s + bias, but unlike an Add(bias) graph this
            // matches CompressedWeightsBlock's pattern (Constant->Convert->Subtract->Multiply),
            // so for_gather_matmul weights still fuse into GatherMatmulCompressed. Also avoids
            // the round(min/scale) error of an integer zero point. Convert bias -> zero-point IN
            // PLACE in the (possibly buffer-backed) zp tensor to avoid a duplicate allocation.
            auto * bias_zp_data = zp.data<ov::float16>();
            const auto * scale_data = scales.data<ov::float16>();
            const size_t n = zp.get_size();
            for (size_t i = 0; i < n; i++) {
                float s = static_cast<float>(scale_data[i]);
                float b = static_cast<float>(bias_zp_data[i]);
                bias_zp_data[i] = ov::float16(s != 0.0f ? -b / s : 0.0f);
            }
            auto zero_point_f16 = std::make_shared<ov::op::v0::Constant>(zp);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
            result = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
        } else if (zp_mode == ZeroPointMode::Integer) {
            // Zero point path: (w - zp) * s
            auto zero_point = std::make_shared<ov::op::v0::Constant>(zp);
            float zp_value;
            if (ov::op::util::get_single_value(zero_point, zp_value)) {
                zero_point = ov::op::v0::Constant::create(zero_point->get_element_type(), {}, {zp_value});
            }
            auto zero_point_f16 = std::make_shared<ov::op::v0::Convert>(zero_point, ov::element::f16);
            auto w_zp =
                std::make_shared<ov::op::v1::Subtract>(weights_f16, zero_point_f16, ov::op::AutoBroadcastType::NUMPY);
            auto mul = std::make_shared<ov::op::v1::Multiply>(w_zp, scales_f16, ov::op::AutoBroadcastType::NUMPY);
            result = mul;
        } else {
            OPENVINO_THROW("Unexpected zero-point mode for unsigned integer weights");
        }
    }

    if (packed_shape.size() != orig_shape.size()) {
        // If not requantized channel-wise case, reshape back to original shape
        auto final_shape =
            std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{orig_shape.size()}, orig_shape);
        auto reshaped = std::make_shared<ov::op::v1::Reshape>(result, final_shape, false);
        result = reshaped;
    }

    if (for_gather_matmul) {
        return result;
    }
    return std::make_shared<ov::op::v0::Convert>(result, ov::element::f32);
}

}  // namespace

ov::Output<ov::Node> make_int8_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_size,
                                       bool use_bias,
                                       bool for_gather_matmul) {
    return make_integer_weights(weight, scales, zp, group_size, use_bias, for_gather_matmul, ov::element::i8,
                                ov::element::u8);
}

// See make_int8_weights for the meaning of for_gather_matmul.
ov::Output<ov::Node> make_int4_weights(ov::Tensor & weight,
                                       ov::Tensor & scales,
                                       ov::Tensor & zp,
                                       size_t group_size,
                                       bool use_bias,
                                       bool for_gather_matmul) {
    return make_integer_weights(weight, scales, zp, group_size, use_bias, for_gather_matmul, ov::element::i4,
                                ov::element::u4);
}

ov::Output<ov::Node> make_mxfp4_weights(ov::Tensor & weight, ov::Tensor & scales) {
    const ov::Shape final_shape = weight.get_shape();
    GGML_ASSERT(!final_shape.empty());
    GGML_ASSERT(final_shape.back() % MXFP4_BLOCK_SIZE == 0);

    ov::Shape packed_shape = final_shape;
    packed_shape.back() /= MXFP4_BLOCK_SIZE;
    packed_shape.push_back(MXFP4_BLOCK_SIZE);

    ov::Shape scale_shape = packed_shape;
    scale_shape.back() = 1;
    scales.set_shape(scale_shape);

    auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::f4e2m1, packed_shape,
                                                               static_cast<uint8_t *>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
    auto weights_f32 = std::make_shared<ov::op::v0::Convert>(weights_node, ov::element::f32);

    auto scales_node = std::make_shared<ov::op::v0::Constant>(scales);
    auto scales_f32 = std::make_shared<ov::op::v0::Convert>(scales_node, ov::element::f32);
    ov::Output<ov::Node> result =
        std::make_shared<ov::op::v1::Multiply>(weights_f32, scales_f32, ov::op::AutoBroadcastType::NUMPY);

    auto final_shape_node =
        std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{final_shape.size()}, final_shape);
    return std::make_shared<ov::op::v1::Reshape>(result, final_shape_node, false);
}

ov::Output<ov::Node> make_mxfp4_moe_packed_weights(ov::Tensor & weight) {
    auto weights_node = std::make_shared<ov::op::v0::Constant>(ov::element::u8, weight.get_shape(),
                                                               static_cast<uint8_t *>(weight.data()), nullptr);
    weights_node->get_rt_info()["__gguf_tensor_holder"] = weight;
    weights_node->get_rt_info()["__ggml_openvino_mxfp4_moe_packed"] = true;
    return weights_node;
}
