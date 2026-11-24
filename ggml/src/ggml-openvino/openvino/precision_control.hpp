#pragma once

#include <openvino/core/node.hpp>

#if __has_include(<transformations/rt_info/disable_precision_conversion.hpp>)
#    include <transformations/rt_info/disable_precision_conversion.hpp>
#else
// Runtime SDKs omit transformation headers. CMake checks that this exported helper is available.
namespace ov {
OPENVINO_API void disable_conversion(const std::shared_ptr<Node> & node, const element::Type & to);
}
#endif
