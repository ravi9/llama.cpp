# Included from ../ggml-openvino/CMakeLists.txt when GGML_OPENVINO_GGUF_FRONTEND=ON, after find_package(OpenVINO) and
# find_package(OpenCL). Builds the ggml-openvino backend from this directory: GgmlOvDecoder bridges ggml graphs to the
# OpenVINO GGUF frontend, which owns the op translations. Needs an OpenVINO build that exports openvino::frontend::gguf.
# Kept here so the in-tree translator's CMakeLists.txt only carries the hand-off block.

if (NOT TARGET openvino::frontend::gguf)
    message(FATAL_ERROR "GGML_OPENVINO_GGUF_FRONTEND=ON requires the OpenVINO GGUF frontend target "
                        "openvino::frontend::gguf, which this OpenVINO (${OpenVINO_DIR}) does not provide. "
                        "Use an OpenVINO build that includes the GGUF frontend, or configure with "
                        "-DGGML_OPENVINO_GGUF_FRONTEND=OFF to use the in-tree translator.")
endif()

set(GGML_OPENVINO_FE_DIR     "${CMAKE_CURRENT_LIST_DIR}")
set(GGML_OPENVINO_SHARED_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

file(GLOB_RECURSE GGML_HEADERS_OPENVINO "${GGML_OPENVINO_FE_DIR}/*.h" "${GGML_OPENVINO_FE_DIR}/*.hpp")
file(GLOB_RECURSE GGML_SOURCES_OPENVINO "${GGML_OPENVINO_FE_DIR}/*.cpp")
# Backend-independent files are compiled from the in-tree directory in both modes (this directory only holds
# forwarding headers for them), so a fix to quantization, the model cache or device selection lands once.
list(APPEND GGML_HEADERS_OPENVINO
    "${GGML_OPENVINO_SHARED_DIR}/ggml-quants.h"
    "${GGML_OPENVINO_SHARED_DIR}/ggml-openvino-extra.h"
    "${GGML_OPENVINO_SHARED_DIR}/model-cache.h")
list(APPEND GGML_SOURCES_OPENVINO
    "${GGML_OPENVINO_SHARED_DIR}/ggml-quants.cpp"
    "${GGML_OPENVINO_SHARED_DIR}/ggml-openvino-extra.cpp"
    "${GGML_OPENVINO_SHARED_DIR}/model-cache.cpp")
message(STATUS "OpenVINO backend: translating with the OpenVINO GGUF frontend")

ggml_add_backend_library(ggml-openvino
    ${GGML_SOURCES_OPENVINO}
    ${GGML_HEADERS_OPENVINO}
)

target_link_libraries(ggml-openvino PRIVATE openvino::runtime openvino::threading openvino::frontend::gguf OpenCL::OpenCL)

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
