find_package(OpenVINO REQUIRED COMPONENTS Runtime Threading)
find_package(OpenCL REQUIRED)

message(STATUS "Found OpenVINO: ${OpenVINO_DIR} (found version \"${OpenVINO_VERSION}\")")

# Two interchangeable translation paths, selected at configure time:
#   GGML_OPENVINO_GGUF_FRONTEND=OFF (default): the in-tree translator in this directory, unchanged. Needs only
#       the stock OpenVINO runtime.
#   GGML_OPENVINO_GGUF_FRONTEND=ON: the sources in ../ggml-openvino-ggml-frontend. GgmlOvDecoder is a bridge to
#       the OpenVINO GGUF frontend, which owns the op translations; needs an OpenVINO build that exports
#       openvino::frontend::gguf.
# This directory is deliberately left in place so changes to the in-tree translator keep applying here.
if (GGML_OPENVINO_GGUF_FRONTEND)
    if (NOT TARGET openvino::frontend::gguf)
        message(FATAL_ERROR "GGML_OPENVINO_GGUF_FRONTEND=ON requires the OpenVINO GGUF frontend target "
                            "openvino::frontend::gguf, which this OpenVINO (${OpenVINO_DIR}) does not provide. "
                            "Use an OpenVINO build that includes the GGUF frontend, or configure with "
                            "-DGGML_OPENVINO_GGUF_FRONTEND=OFF to use the in-tree translator.")
    endif()

    set(GGML_OPENVINO_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../ggml-openvino-ggml-frontend")
    file(GLOB_RECURSE GGML_HEADERS_OPENVINO "${GGML_OPENVINO_SRC_DIR}/*.h" "${GGML_OPENVINO_SRC_DIR}/*.hpp")
    file(GLOB_RECURSE GGML_SOURCES_OPENVINO "${GGML_OPENVINO_SRC_DIR}/*.cpp")
    message(STATUS "OpenVINO backend: translating with the OpenVINO GGUF frontend")
else()
    file(GLOB_RECURSE GGML_HEADERS_OPENVINO "*.h" "*.hpp")
    file(GLOB_RECURSE GGML_SOURCES_OPENVINO "*.cpp")
    message(STATUS "OpenVINO backend: translating with the in-tree translator")
endif()

ggml_add_backend_library(ggml-openvino
    ${GGML_SOURCES_OPENVINO}
    ${GGML_HEADERS_OPENVINO}
)

target_link_libraries(ggml-openvino PRIVATE openvino::runtime openvino::threading OpenCL::OpenCL)
if (GGML_OPENVINO_GGUF_FRONTEND)
    target_link_libraries(ggml-openvino PRIVATE openvino::frontend::gguf)
endif()

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

# Released runtimes retain the ordinary cache translators. Only the in-tree translator has the cache-update ops.
if (NOT GGML_OPENVINO_GGUF_FRONTEND)
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
endif()
