#include "ggml-openvino-extra.h"

#include "ggml-impl.h"
#include "ggml.h"

#include <cstdlib>
#include <cstring>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/properties.hpp>
#include <optional>

ov::Core & ov_singleton_core() {
    static ov::Core core;
    return core;
}

// =====================================================
// Device Configuration Implementations
// =====================================================

void ggml_openvino_device_config::init() {
    if (initialized) {
        return;
    }

    // All recognized GGML_OPENVINO_* env vars. Their values are cached here
    // once at backend init time and read back via ggml_openvino_getenv_str()
    // (raw string) or ggml_openvino_getenv_int() (integer / boolean toggle).
    static constexpr const char * env_var_names[] = {
        // String values (use ggml_openvino_getenv_str)
        "GGML_OPENVINO_DEVICE",
        "GGML_OPENVINO_CACHE_DIR",
        "GGML_OPENVINO_DEBUG_NODE",
        // Integer values (use ggml_openvino_getenv_int)
        "GGML_OPENVINO_PREFILL_CHUNK_SIZE",
        // Boolean toggles (treated as int flags via ggml_openvino_getenv_int)
        "GGML_OPENVINO_STATEFUL_EXECUTION",
        "GGML_OPENVINO_PROFILING",
        "GGML_OPENVINO_DUMP_CGRAPH",
        "GGML_OPENVINO_DUMP_IR",
        "GGML_OPENVINO_DEBUG_INPUT",
        "GGML_OPENVINO_DEBUG_OUTPUT",
        "GGML_OPENVINO_PRINT_CGRAPH_TENSOR_ADDRESS",
        "GGML_OPENVINO_ENABLE_CACHE",
        "GGML_OPENVINO_DISABLE_CACHE",
        "GGML_OPENVINO_DISABLE_KV_SLICE",
        "GGML_OPENVINO_MANUAL_GQA_ATTN",
        "GGML_OPENVINO_MEMORY_OPTIMIZE",
        "GGML_OPENVINO_RELEASE_WEIGHTS",
        "GGML_OPENVINO_REDUCE_COMPILE_MEM",
        "GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR",
    };

    for (const char * const & env_var : env_var_names) {
        auto * env = getenv(env_var);
        if (env) {
            environment_variables[env_var] = env;
        }
    }

    device_name = ggml_openvino_getenv_str("GGML_OPENVINO_DEVICE", "CPU");
    auto available_devices = ov_singleton_core().get_available_devices();
    if (std::find(available_devices.begin(), available_devices.end(), device_name) == available_devices.end()) {
        GGML_LOG_WARN("GGML OpenVINO Backend: device %s is not available, fallback to CPU\n", device_name.c_str());
        device_name = "CPU";
    }
    is_npu = (device_name == "NPU");

    const char * cache_dir = ggml_openvino_getenv_str("GGML_OPENVINO_CACHE_DIR");
    if (device_name == "NPU") {
        compile_config = {
            {"NPU_COMPILER_DYNAMIC_QUANTIZATION", "YES"   },
            {"NPU_USE_NPUW",                      "YES"   },
            {"NPUW_DEVICES",                      "NPU"   },
            {"NPUW_FOLD",                         "YES"   },
            {"NPUW_WEIGHTS_BANK",                 "shared"},
            {"NPUW_FUNCALL_FOR_ALL",              "YES"   },
            {"NPUW_FUNCALL_ASYNC",                "YES"   },
            {"NPUW_DQ",                           "YES"   },
            {"NPUW_DQ_FULL",                      "NO"    },
        };
        if (cache_dir && strlen(cache_dir) > 0) {
            compile_config["NPUW_CACHE_DIR"] = cache_dir;
            compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
        }
    } else if (cache_dir && strlen(cache_dir) > 0) {
        compile_config.insert(ov::cache_dir(cache_dir));
        compile_config.insert(ov::cache_mode(ov::CacheMode::OPTIMIZE_SIZE));
    }

    // Initialize remote context with queue sharing for GPU
    if (device_name == "GPU") {
        // Create OpenCL context and queue
        cl_int err;
        cl_platform_id platform;
        err = clGetPlatformIDs(1, &platform, nullptr);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to get OpenCL platform: %d\n", err);
            return;
        }

        cl_device_id cl_device;
        err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &cl_device, nullptr);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to get OpenCL device: %d\n", err);
            return;
        }

        cl_context cl_ctx = clCreateContext(nullptr, 1, &cl_device, nullptr, nullptr, &err);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to create OpenCL context: %d\n", err);
            return;
        }

        cl_queue = clCreateCommandQueueWithProperties(cl_ctx, cl_device, nullptr, &err);
        if (err != CL_SUCCESS) {
            GGML_LOG_ERROR("Failed to create OpenCL command queue: %d\n", err);
            clReleaseContext(cl_ctx);
            return;
        }

        // Create OpenVINO remote context with queue sharing
        remote_context = ov::intel_gpu::ocl::ClContext(ov_singleton_core(), cl_queue);

        // Release the context (queue keeps a reference)
        clReleaseContext(cl_ctx);
    } else if (device_name == "NPU") {
        // remote tensor is not used for NPU yet
        // remote_context = ov_singleton_core().get_default_context(device_name);
    }

    initialized = true;
}

ggml_openvino_device_config::~ggml_openvino_device_config() {
    if (cl_queue != nullptr) {
        clReleaseCommandQueue(cl_queue);
        cl_queue = nullptr;
    }
}

// Get the global device config singleton
ggml_openvino_device_config & ggml_openvino_get_device_config() {
    static ggml_openvino_device_config config;
    return config;
}

// Initialize device config (call during backend init)
void ggml_openvino_init_device_config() {
    ggml_openvino_get_device_config().init();
}

// Get the device name
const std::string & ggml_openvino_get_device_name() {
    return ggml_openvino_get_device_config().device_name;
}

// Get the value of a GGML_OPENVINO_* env var as a string. Returns
// default_value when the var is unset or set to an empty string.
const char * ggml_openvino_getenv_str(const char * var, const char * default_value) {
    auto & env_map = ggml_openvino_get_device_config().environment_variables;
    auto it = env_map.find(var);
    return (it == env_map.end() || it->second.empty()) ? default_value : it->second.c_str();
}

// Get the value of a GGML_OPENVINO_* env var as an int (via std::atoi).
// Returns default_value (0) when the var is unset or empty. Used for both
// integer settings (e.g. GGML_OPENVINO_PREFILL_CHUNK_SIZE) and boolean
// toggles: "0" disables, any non-zero integer enables.
int ggml_openvino_getenv_int(const char * var, int default_value) {
    const char * v = ggml_openvino_getenv_str(var, nullptr);
    return v ? std::atoi(v) : default_value;
}

bool ggml_openvino_reduce_compile_mem_enabled() {
    const char * reduce_compile_mem = ggml_openvino_getenv_str("GGML_OPENVINO_REDUCE_COMPILE_MEM");
    if (reduce_compile_mem != nullptr) {
        return ggml_openvino_getenv_int("GGML_OPENVINO_REDUCE_COMPILE_MEM") != 0;
    }
    return ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

bool ggml_openvino_release_weights_enabled(const std::string & device) {
    const char * release_weights = ggml_openvino_getenv_str("GGML_OPENVINO_RELEASE_WEIGHTS");
    if (release_weights != nullptr) {
        return device == "GPU" && ggml_openvino_getenv_int("GGML_OPENVINO_RELEASE_WEIGHTS") != 0;
    }
    return device == "GPU" && ggml_openvino_getenv_int("GGML_OPENVINO_MEMORY_OPTIMIZE") != 0;
}

// Check if running on NPU
bool ggml_openvino_is_npu() {
    return ggml_openvino_get_device_config().is_npu;
}

// Get the remote context for the current device (returns empty optional for CPU)
std::optional<ov::RemoteContext> ggml_openvino_get_remote_context() {
    return ggml_openvino_get_device_config().remote_context;
}

// Get the compile config for the current device
const ov::AnyMap & ggml_openvino_get_compile_config() {
    return ggml_openvino_get_device_config().compile_config;
}

// Get the OpenCL command queue for GPU operations
cl_command_queue ggml_openvino_get_cl_queue() {
    return ggml_openvino_get_device_config().cl_queue;
}

// Get the clEnqueueMemFillINTEL function pointer (lazy load)
clEnqueueMemFillINTEL_fn ggml_openvino_get_clEnqueueMemFillINTEL() {
    static clEnqueueMemFillINTEL_fn fn = nullptr;
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) == CL_SUCCESS) {
            fn = (clEnqueueMemFillINTEL_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemFillINTEL");
        }
    }
    return fn;
}

// Get the clEnqueueMemcpyINTEL function pointer (lazy load)
clEnqueueMemcpyINTEL_fn ggml_openvino_get_clEnqueueMemcpyINTEL() {
    static clEnqueueMemcpyINTEL_fn fn = nullptr;
    static bool loaded = false;
    if (!loaded) {
        loaded = true;
        cl_platform_id platform;
        if (clGetPlatformIDs(1, &platform, nullptr) == CL_SUCCESS) {
            fn = (clEnqueueMemcpyINTEL_fn) clGetExtensionFunctionAddressForPlatform(platform, "clEnqueueMemcpyINTEL");
        }
    }
    return fn;
}

std::unique_ptr<ggml_openvino_tensor_extra> ggml_openvino_create_tensor_extra_unique(const ggml_tensor * tensor,
                                                                                     bool is_remote) {
    ov::Shape shape;
    for (int i = GGML_MAX_DIMS - 1; i >= 0; --i) {
        shape.push_back(static_cast<size_t>(tensor->ne[i]));
    }

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
    case GGML_TYPE_I32:
        element_type = ov::element::i32;
        break;
    case GGML_TYPE_I64:
        element_type = ov::element::i64;
        break;
    default:
        // GGML_LOG_WARN("%s: unsupported tensor type for ov::Tensor: %s\n", __func__, ggml_type_name(tensor->type));
        return nullptr;
    }

    const auto & device_name = ggml_openvino_get_device_name();
    auto remote_context = ggml_openvino_get_remote_context();

    std::shared_ptr<ov::Tensor> ov_tensor;
    if (is_remote) {
        GGML_ASSERT(device_name == "GPU");
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        auto usm_tensor = gpu_context.create_tensor(element_type, shape, tensor->data);
        ov_tensor = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    } else {
        ov_tensor = std::make_shared<ov::Tensor>(element_type, shape, tensor->data);
    }

    return std::make_unique<ggml_openvino_tensor_extra>(ov_tensor);
}
