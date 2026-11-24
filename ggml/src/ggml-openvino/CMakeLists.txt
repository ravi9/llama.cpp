find_package(OpenVINO REQUIRED COMPONENTS Runtime Threading)
find_package(OpenCL REQUIRED)

message(STATUS "Found OpenVINO: ${OpenVINO_DIR} (found version \"${OpenVINO_VERSION}\")")

file(GLOB_RECURSE GGML_HEADERS_OPENVINO "*.h" "*.hpp")
file(GLOB_RECURSE GGML_SOURCES_OPENVINO "*.cpp")

ggml_add_backend_library(ggml-openvino
    ${GGML_SOURCES_OPENVINO}
    ${GGML_HEADERS_OPENVINO}
)

target_link_libraries(ggml-openvino PRIVATE openvino::runtime openvino::threading OpenCL::OpenCL)

# the OpenVINO RTTI macros take one argument and leave __VA_ARGS__ empty, which -Wpedantic reports
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "IntelLLVM")
    target_compile_options(ggml-openvino PRIVATE -Wno-gnu-zero-variadic-macro-arguments)
elseif (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(ggml-openvino PRIVATE -Wno-pedantic)
endif()

if (WIN32)
    target_link_libraries(ggml-openvino PRIVATE psapi)
endif()

if (GGML_OPENVINO)
    if (CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    elseif (CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "amd64" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "AMD64")
    else()
        message(FATAL_ERROR "OpenVINO: OpenVINO toolkit supports x86-64 and arm64 but not ${CMAKE_SYSTEM_PROCESSOR}")
    endif()
endif()

# Released runtimes retain the ordinary cache translators.
include(CheckCXXSourceCompiles)
unset(GGML_OPENVINO_HAS_CACHE_UPDATE CACHE)
set(CMAKE_REQUIRED_LIBRARIES openvino::runtime)
set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/openvino/op")
check_cxx_source_compiles("#include <recurrent_cache_update.hpp>
int main() {
    auto conv = std::make_shared<ov::op::internal::RecurrentConvCacheUpdate>(ov::OutputVector{}, 3);
    auto gdn = std::make_shared<ov::op::internal::RecurrentGatedDeltaNetUpdate>(ov::OutputVector{});
}"
    GGML_OPENVINO_HAS_CACHE_UPDATE)
unset(CMAKE_REQUIRED_LIBRARIES)
unset(CMAKE_REQUIRED_INCLUDES)
if(GGML_OPENVINO_HAS_CACHE_UPDATE)
    target_compile_definitions(ggml-openvino PRIVATE GGML_OPENVINO_HAS_CACHE_UPDATE=1)
endif()

# Core features that not every ggml tree has
unset(GGML_OPENVINO_HAS_BUFFER_TENSOR_N CACHE)
unset(GGML_OPENVINO_HAS_FA_INDEXED_BIAS CACHE)
set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/../../include" "${CMAKE_CURRENT_SOURCE_DIR}/..")
check_cxx_source_compiles("#include <ggml-backend-impl.h>
int main() { ggml_backend_buffer_i i{}; return i.set_tensor_n != nullptr; }"
    GGML_OPENVINO_HAS_BUFFER_TENSOR_N)
check_cxx_source_compiles("#include <ggml.h>
using fn = decltype(&ggml_flash_attn_ext_has_indexed_bias);
int main() { return 0; }"
    GGML_OPENVINO_HAS_FA_INDEXED_BIAS)
unset(CMAKE_REQUIRED_INCLUDES)
if(GGML_OPENVINO_HAS_BUFFER_TENSOR_N)
    target_compile_definitions(ggml-openvino PRIVATE GGML_OPENVINO_HAS_BUFFER_TENSOR_N=1)
endif()
if(GGML_OPENVINO_HAS_FA_INDEXED_BIAS)
    target_compile_definitions(ggml-openvino PRIVATE GGML_OPENVINO_HAS_FA_INDEXED_BIAS=1)
endif()
