#include "ggml-openvino.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "ggml-impl.h"
#include "ggml-openvino-buffer.h"
#include "ggml-openvino-extra.h"
#include "ggml-openvino-op-support.h"
#include "ggml-openvino-weight-buffer-release.h"
#include "ggml-openvino/utils.h"
#include "ggml-quants.h"
#include "ggml.h"

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <openvino/core/type/element_type.hpp>
#include <openvino/openvino.hpp>
#include <openvino/runtime/allocator.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/intel_npu/level_zero/level_zero.hpp>
#include <openvino/runtime/tensor.hpp>
#include <string>
#include <vector>

#if defined(_WIN32)
#    define WIN32_LEAN_AND_MEAN
#    ifndef NOMINMAX
#        define NOMINMAX
#    endif
#    include <windows.h>
#else
#    include <unistd.h>
#endif

// Buffer type context (per-device)
struct ggml_backend_openvino_buffer_type_context {
    int device;
    std::string name;
    bool is_host;
};

// Buffer type interface functions
static const char * ggml_backend_openvino_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_openvino_buffer_type_context * ctx = (ggml_backend_openvino_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_openvino_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft,
                                                                            size_t size) {
    ggml_backend_openvino_buffer_type_context * buft_ctx = (ggml_backend_openvino_buffer_type_context *) buft->context;
    return ggml_backend_openvino_buffer_alloc(buft, buft_ctx->device, size);
}

static size_t ggml_backend_openvino_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return TENSOR_ALIGNMENT;
}

static size_t ggml_backend_openvino_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return SIZE_MAX;
}

static size_t ggml_backend_openvino_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft,
                                                               const ggml_tensor * tensor) {
    GGML_UNUSED(buft);

    // For quantized weight tensors, we need extra space for extracted data.
    if (ggml_is_quantized(tensor->type) && tensor->ne[3] == 1) {
        ggml_openvino_extracted_layout layout = ggml_openvino_get_extracted_layout(tensor);
        if (layout.total_size > 0) {
            // GGML_LOG_DEBUG("%s: tensor %s needs %zu bytes (original %zu, extracted: weights=%zu scales=%zu zp=%zu)\n",
            //                __func__, tensor->name, layout.total_size, ggml_nbytes(tensor), layout.weights_size,
            //                layout.scales_size, layout.zp_size);
            return layout.total_size;
        }
    }

    return ggml_nbytes(tensor);
}

static const ggml_backend_buffer_type_i ggml_backend_openvino_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_openvino_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_openvino_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_openvino_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_openvino_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_openvino_buffer_type_get_alloc_size,
    /* .is_host          = */ nullptr,
};

// =====================================================
// OpenVINO Host Buffer Implementation
// =====================================================

static const char * ggml_backend_openvino_host_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_openvino_buffer_type_context * ctx = (ggml_backend_openvino_buffer_type_context *) buft->context;
    return ctx->name.c_str();
}

static bool ggml_backend_openvino_host_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(buft);
    return true;
}

static const ggml_backend_buffer_type_i ggml_backend_openvino_host_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_openvino_host_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_openvino_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_openvino_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_openvino_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_openvino_buffer_type_get_alloc_size,
    /* .is_host          = */ ggml_backend_openvino_host_buffer_type_is_host,
};

struct ggml_backend_openvino_buffer_type_state {
    std::mutex mutex;
    std::vector<ggml_backend_buffer_type> buffer_types;
    std::vector<ggml_backend_openvino_buffer_type_context> buffer_type_contexts;
};

static ggml_backend_buffer_type_t ggml_backend_openvino_get_buffer_type(
    ggml_backend_openvino_buffer_type_state & state,
    const ggml_backend_buffer_type_i & iface,
    int device,
    bool is_host) {
    GGML_ASSERT(device >= 0 && device < ggml_backend_openvino_get_device_count());

    std::lock_guard<std::mutex> lock(state.mutex);

    if (state.buffer_types.empty()) {
        int device_count = ggml_backend_openvino_get_device_count();
        state.buffer_types.resize(device_count);
        state.buffer_type_contexts.resize(device_count);

        for (int i = 0; i < device_count; i++) {
            state.buffer_type_contexts[i].device = i;
            state.buffer_type_contexts[i].name =
                std::string(GGML_OPENVINO_NAME) + std::to_string(i) + (is_host ? "_HOST" : "");
            state.buffer_type_contexts[i].is_host = is_host;

            state.buffer_types[i] = ggml_backend_buffer_type{
                /* .iface   = */ iface,
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_openvino_reg(), i),
                /* .context = */ &state.buffer_type_contexts[i],
            };
        }
    }

    return &state.buffer_types[device];
}

// Get buffer type for a specific device
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_buffer_type(int device) {
    static ggml_backend_openvino_buffer_type_state state;
    return ggml_backend_openvino_get_buffer_type(state, ggml_backend_openvino_buffer_type_interface, device, false);
}

GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_host_buffer_type(int device) {
    static ggml_backend_openvino_buffer_type_state state;
    return ggml_backend_openvino_get_buffer_type(state, ggml_backend_openvino_host_buffer_type_interface, device, true);
}

bool ggml_backend_buft_is_openvino(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr || buft->device == nullptr || buft->device->reg != ggml_backend_openvino_reg()) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_openvino_buffer_type_context *>(buft->context);
    return !ctx->is_host;
}

bool ggml_backend_buft_is_openvino_host(ggml_backend_buffer_type_t buft) {
    if (buft == nullptr || buft->device == nullptr || buft->device->reg != ggml_backend_openvino_reg()) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_openvino_buffer_type_context *>(buft->context);
    return ctx->is_host;
}

static void ggml_backend_openvino_free(ggml_backend_t backend) {
    ggml_backend_openvino_context * ctx = (ggml_backend_openvino_context *) backend->context;

    if (ctx->runtime_context) {
        auto r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
        if (--r_ctx->backend_count == 0) {
            // If host weight buffers were released (GGML_OPENVINO_RELEASE_WEIGHTS), the
            // dropped pages can never be repopulated, so a recompile is impossible. Keep
            // the compiled-model cache alive across backend teardown so the next context
            // reuses it instead of recompiling against zeroed weights.
            if (!ggml_openvino_weight_buffers_released()) {
                r_ctx->clear_caches();
            }
        }
    }

    delete ctx;
    delete backend;
}

static const char * ggml_backend_openvino_get_name(ggml_backend_t backend) {
    GGML_UNUSED(backend);
    return GGML_OPENVINO_NAME;
}

static enum ggml_status ggml_backend_openvino_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    GGML_UNUSED(backend);
    return ov_graph_compute(cgraph, backend);
}

static const ggml_backend_i ggml_backend_openvino_interface = {
    /* .get_name                = */ ggml_backend_openvino_get_name,
    /* .free                    = */ ggml_backend_openvino_free,
    /* .set_tensor_async        = */ NULL,
    /* .get_tensor_async        = */ NULL,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ NULL,
    /* .synchronize             = */ NULL,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_openvino_graph_compute,
    /* .event_record            = */ NULL,
    /* .event_wait              = */ NULL,
    /* .graph_optimize          = */ NULL,
};

int ggml_backend_openvino_get_device_count() {
    return 1;
}

static ggml_guid_t ggml_backend_openvino_guid(void) {
    static ggml_guid guid = {0x12, 0xa8, 0xae, 0xf4, 0xc0, 0x1e, 0x61, 0x97,
                             0x8f, 0xeb, 0x33, 0x04, 0xa1, 0x33, 0x51, 0x2d};
    return &guid;
}

static std::shared_ptr<ov_runtime_context> get_ov_runtime_context_ptr() {
    static std::shared_ptr<ov_runtime_context> r_ctx = [] {
        auto ctx = std::make_shared<ov_runtime_context>();
        ctx->device = ggml_openvino_get_device_name();
        ctx->stateful = ggml_openvino_is_stateful_enabled() && !ggml_openvino_is_npu();
        return ctx;
    }();
    return r_ctx;
}

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_openvino_init(int device) {
    if (device < 0 || device >= ggml_backend_openvino_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_openvino_context * ctx = new ggml_backend_openvino_context;
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ctx->runtime_context = get_ov_runtime_context_ptr();
    if (ctx->runtime_context == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate runtime context\n", __func__);
        delete ctx;
        return nullptr;
    }

    std::shared_ptr<ov_runtime_context> r_ctx = std::static_pointer_cast<ov_runtime_context>(ctx->runtime_context);
    r_ctx->backend_count++;

    ggml_backend_t openvino_backend = new ggml_backend{
        /* .guid      = */ ggml_backend_openvino_guid(),
        /* .interface = */ ggml_backend_openvino_interface,
        /* .device    = */ ggml_backend_reg_dev_get(ggml_backend_openvino_reg(), device),
        /* .context   = */ ctx,
    };

    return openvino_backend;
}

GGML_BACKEND_API bool ggml_backend_is_openvino(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_openvino_guid());
}

struct ggml_backend_openvino_device_context {
    int device;
    std::string name;
    std::string description;
};

static const char * ggml_backend_openvino_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_openvino_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ctx->description.c_str();
}

static void ggml_backend_openvino_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
#ifdef _WIN32
    MEMORYSTATUSEX status;
    status.dwLength = sizeof(status);
    GlobalMemoryStatusEx(&status);
    *total = status.ullTotalPhys;
    *free = status.ullAvailPhys;
#else
    long pages = sysconf(_SC_PHYS_PAGES);
    long page_size = sysconf(_SC_PAGE_SIZE);
    *total = pages * page_size;

    // "free" system memory is ill-defined, for practical purposes assume that all of it is free:
    *free = *total;
#endif  // _WIN32

    GGML_UNUSED(dev);
}

static enum ggml_backend_dev_type ggml_backend_openvino_device_get_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_openvino_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name = ggml_backend_openvino_device_get_name(dev);
    props->description = ggml_backend_openvino_device_get_description(dev);
    props->type = ggml_backend_openvino_device_get_type(dev);
    ggml_backend_openvino_device_get_memory(dev, &props->memory_free, &props->memory_total);

    props->caps = {
        /* .async                 = */ false,
        /* .host_buffer           = */ false,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ false,
    };
}

static ggml_backend_t ggml_backend_openvino_device_init(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ggml_backend_openvino_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_openvino_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ggml_backend_openvino_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_openvino_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_openvino_device_context * ctx = (ggml_backend_openvino_device_context *) dev->context;
    return ggml_backend_openvino_host_buffer_type(ctx->device);
}

static bool ggml_backend_openvino_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    return ggml_openvino_device_supports_op_impl(dev, op);
}

static bool ggml_backend_openvino_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    GGML_UNUSED(dev);
    return ggml_backend_buft_is_openvino(buft) || ggml_backend_buft_is_host(buft);
}

static const struct ggml_backend_device_i ggml_backend_openvino_device_interface = {
    /* .get_name             = */ ggml_backend_openvino_device_get_name,
    /* .get_description      = */ ggml_backend_openvino_device_get_description,
    /* .get_memory           = */ ggml_backend_openvino_device_get_memory,
    /* .get_type             = */ ggml_backend_openvino_device_get_type,
    /* .get_props            = */ ggml_backend_openvino_device_get_props,
    /* .init_backend         = */ ggml_backend_openvino_device_init,
    /* .get_buffer_type      = */ ggml_backend_openvino_device_get_buffer_type,
    /* .get_host_buffer_type = */ ggml_backend_openvino_device_get_host_buffer_type,
    /* .buffer_from_host_ptr = */ NULL,
    /* .supports_op          = */ ggml_backend_openvino_device_supports_op,
    /* .supports_buft        = */ ggml_backend_openvino_device_supports_buft,
    /* .offload_op           = */ NULL,
    /* .event_new            = */ NULL,
    /* .event_free           = */ NULL,
    /* .event_synchronize    = */ NULL,
};

struct ggml_backend_openvino_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_openvino_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_OPENVINO_NAME;
}

static size_t ggml_backend_openvino_reg_get_device_count(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return (size_t) ggml_backend_openvino_get_device_count();
}

static ggml_backend_dev_t ggml_backend_openvino_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_openvino_reg_context * ctx = (ggml_backend_openvino_reg_context *) reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static const struct ggml_backend_reg_i ggml_backend_openvino_reg_interface = {
    /* .get_name         = */ ggml_backend_openvino_reg_get_name,
    /* .get_device_count = */ ggml_backend_openvino_reg_get_device_count,
    /* .get_device       = */ ggml_backend_openvino_reg_get_device,
    /* .get_proc_address = */ NULL,
};

static void ggml_openvino_init() {
    // Initialize device config singleton from env var
    ggml_openvino_init_device_config();
    GGML_LOG_INFO("OpenVINO: using device %s\n", ggml_openvino_get_device_name().c_str());
}

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_openvino_reg(void) {
    static ggml_backend_reg reg;

    static bool initialized = false;
    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_openvino_init();

            ggml_backend_openvino_reg_context * ctx = new ggml_backend_openvino_reg_context;

            for (int i = 0; i < ggml_backend_openvino_get_device_count(); i++) {
                ggml_backend_openvino_device_context * dev_ctx = new ggml_backend_openvino_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_OPENVINO_NAME + std::to_string(i);

                dev_ctx->description = ov::get_openvino_version().description;

                ggml_backend_dev_t dev =
                    new ggml_backend_device{/* .interface = */ ggml_backend_openvino_device_interface,
                                            /* .reg       = */ &reg,
                                            /* .context   = */ dev_ctx};
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg{/* .api_version = */ GGML_BACKEND_API_VERSION,
                                   /* .iface       = */ ggml_backend_openvino_reg_interface,
                                   /* .context     = */ ctx};
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_openvino_reg)
