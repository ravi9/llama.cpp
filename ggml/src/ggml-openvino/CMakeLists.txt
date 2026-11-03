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

# Embed the GPU custom-layer kernels; they are written to a temp directory at runtime.
set(GGML_OPENVINO_KERNEL_FILES
    recurrent_cache.xml
    recurrent_common.cl
    recurrent_cache_stage.cl
    recurrent_conv_update.cl
    recurrent_gdn_update.cl)
set(GGML_OPENVINO_KERNEL_INC "")
foreach(kernel_file ${GGML_OPENVINO_KERNEL_FILES})
    set(kernel_path "${CMAKE_CURRENT_SOURCE_DIR}/openvino/kernels/${kernel_file}")
    file(READ "${kernel_path}" kernel_content)
    string(APPEND GGML_OPENVINO_KERNEL_INC "{\"${kernel_file}\", R\"ggmlovk(${kernel_content})ggmlovk\"},\n")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${kernel_path}")
endforeach()
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/ggml-openvino-kernels.inc.tmp" "${GGML_OPENVINO_KERNEL_INC}")
configure_file("${CMAKE_CURRENT_BINARY_DIR}/ggml-openvino-kernels.inc.tmp"
               "${CMAKE_CURRENT_BINARY_DIR}/ggml-openvino-kernels.inc" COPYONLY)
target_include_directories(ggml-openvino PRIVATE "${CMAKE_CURRENT_BINARY_DIR}")
