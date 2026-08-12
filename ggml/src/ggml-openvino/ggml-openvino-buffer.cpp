#include "ggml-openvino-buffer.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-openvino-op-support.h"
#include "ggml-openvino-weight-buffer-release.h"
#include "ggml-openvino/utils.h"
#include "ggml-openvino-quant-weights.h"
#include "ggml.h"

#include <atomic>
#include <cstdint>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/tensor.hpp>

class ggml_openvino_buffer_storage {
public:
    virtual ~ggml_openvino_buffer_storage() = default;

    virtual void * data() const noexcept = 0;
    virtual size_t size() const noexcept = 0;
};

class ggml_openvino_host_buffer_storage : public ggml_openvino_buffer_storage {
public:
    explicit ggml_openvino_host_buffer_storage(size_t size) :
        size_(size) {
        if (size_ == 0) {
            return;
        }

        data_ = ggml_aligned_malloc(size_);
        GGML_ASSERT(data_);
        memset(data_, 0, size_);
        ov_buffer_ = std::make_shared<ov::Tensor>(ov::element::u8, ov::Shape{size_}, data_);
    }

    ~ggml_openvino_host_buffer_storage() override {
        if (data_ != nullptr) {
            ggml_aligned_free(data_, size_);
        }
    }

    void * data() const noexcept override {
        return data_;
    }

    size_t size() const noexcept override {
        return size_;
    }

private:
    void * data_ = nullptr;
    size_t size_ = 0;
    std::shared_ptr<ov::Tensor> ov_buffer_;
};

class ggml_openvino_remote_buffer_storage : public ggml_openvino_buffer_storage {
public:
    explicit ggml_openvino_remote_buffer_storage(size_t size) :
        size_(size) {
        if (size_ == 0) {
            return;
        }

        GGML_ASSERT(ggml_openvino_get_device_name() == "GPU");
        auto remote_context = ggml_openvino_get_remote_context();
        auto gpu_context = remote_context->as<ov::intel_gpu::ocl::ClContext>();
        ov::intel_gpu::ocl::USMTensor usm_tensor =
            gpu_context.create_usm_device_tensor(ov::element::u8, ov::Shape{size_});
        data_ = usm_tensor.get();
        ov_buffer_ = std::make_shared<ov::intel_gpu::ocl::USMTensor>(std::move(usm_tensor));
    }

    void * data() const noexcept override {
        return data_;
    }

    size_t size() const noexcept override {
        return size_;
    }

private:
    void * data_ = nullptr;
    size_t size_ = 0;
    std::shared_ptr<ov::Tensor> ov_buffer_;
};

static std::unique_ptr<ggml_openvino_buffer_storage> ggml_openvino_create_buffer_storage(size_t size, bool is_remote) {
    if (is_remote) {
        return std::make_unique<ggml_openvino_remote_buffer_storage>(size);
    }

    return std::make_unique<ggml_openvino_host_buffer_storage>(size);
}

struct ggml_backend_openvino_buffer_context {
    int device;
    size_t id;

    bool is_remote;

    std::unique_ptr<ggml_openvino_buffer_storage> storage;

    std::map<ggml_tensor *, std::unique_ptr<ggml_openvino_extra_base>> tensor_extras;

    ggml_backend_openvino_buffer_context(int device, size_t size, bool is_remote = false) :
        device(device),
        id([]() {
            static std::atomic<size_t> next_id{1};
            return next_id.fetch_add(1);
        }()),
        is_remote(is_remote) {
        if (size == 0) {
            return;
        }

        const auto & device_name = ggml_openvino_get_device_name();

        storage = ggml_openvino_create_buffer_storage(size, is_remote);

        if (data() == nullptr) {
            GGML_LOG_ERROR("%s: failed to allocate %zu bytes\n", __func__, size);
            return;
        }

        if (reinterpret_cast<uintptr_t>(data()) % TENSOR_ALIGNMENT != 0) {
            GGML_LOG_ERROR("%s: %s buffer is not aligned to %d bytes\n", __func__, device_name.c_str(),
                           TENSOR_ALIGNMENT);
            GGML_ABORT("fatal error");
        }
    }

    void * data() const noexcept {
        return storage != nullptr ? storage->data() : nullptr;
    }

    size_t size() const noexcept {
        return storage != nullptr ? storage->size() : 0;
    }

    ~ggml_backend_openvino_buffer_context() {
        tensor_extras.clear();
    }
};

static void ggml_backend_openvino_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    delete ctx;
}

static void * ggml_backend_openvino_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    return ctx->data();
}

static enum ggml_status ggml_backend_openvino_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (strncmp(tensor->name, "cache_", 6) == 0 && !ctx->is_remote && ggml_openvino_get_device_name() == "GPU" &&
        !ggml_openvino_is_stateful_enabled()) {
        GGML_ASSERT(ctx->tensor_extras.empty());
        auto device = ctx->device;
        auto size = ctx->size();
        auto data_offset = (char *) tensor->data - (char *) ctx->data();
        delete ctx;
        ctx = new ggml_backend_openvino_buffer_context(device, size, true);
        buffer->context = ctx;
        tensor->data = (char *) ctx->data() + data_offset;
    }

    if (tensor->view_src != nullptr) {
        GGML_ASSERT(tensor->view_src->buffer->buft == buffer->buft);
        if (tensor->view_src->extra != nullptr) {
            tensor->extra = tensor->view_src->extra;
        }
        return GGML_STATUS_SUCCESS;
    }

    ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (tensor->data != nullptr && !ggml_is_quantized(tensor->type)) {
        auto extra = ggml_openvino_create_tensor_extra_unique(tensor, ctx->is_remote);
        if (extra != nullptr) {
            tensor->extra = extra.get();
            ctx->tensor_extras[tensor] = std::move(extra);
        }
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_openvino_buffer_memset_tensor(ggml_backend_buffer_t buffer,
                                                       ggml_tensor * tensor,
                                                       uint8_t value,
                                                       size_t offset,
                                                       size_t size) {
    GGML_ASSERT(tensor != nullptr && tensor->data != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (ctx->is_remote) {
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_fill_fn = ggml_openvino_get_clEnqueueMemFillINTEL();
        if (queue != nullptr && mem_fill_fn != nullptr) {
            uint8_t pattern = value;
            cl_int err = mem_fill_fn(queue, (char *) tensor->data + offset, &pattern, sizeof(pattern), size, 0, nullptr,
                                     nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("%s: clEnqueueMemFillINTEL failed with error %d\n", __func__, err);
            }
            clFinish(queue);
        } else {
            GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemFillINTEL not available for GPU buffer\n", __func__);
        }
    } else {
        memset((char *) tensor->data + offset, value, size);
    }
}

static void ggml_backend_openvino_buffer_set_tensor(ggml_backend_buffer_t buffer,
                                                    ggml_tensor * tensor,
                                                    const void * data,
                                                    size_t offset,
                                                    size_t size) {
    GGML_ASSERT(tensor != nullptr && tensor->data != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    bool is_weight_buffer = (buffer->usage == GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    bool is_full_tensor_set = (offset == 0 && size == ggml_nbytes(tensor) && tensor->view_src == nullptr);
    bool is_2d = (tensor->ne[2] == 1 && tensor->ne[3] == 1);
    bool is_supported_weight_shape = is_2d || (tensor->ne[3] == 1 && ggml_is_quantized(tensor->type));

    if (is_weight_buffer && is_full_tensor_set && is_supported_weight_shape) {
        try {
            auto result = process_weight_tensor(tensor, data, tensor->data);
            result.weight_node->set_friendly_name(tensor->name);

            std::unique_ptr<ggml_openvino_extra_base> extra;

            if (result.is_quantized()) {
                extra = std::make_unique<ggml_openvino_quantized_weight_extra>(
                    std::move(result.weights), std::move(result.scales), std::move(result.zp), result.weight_node);
            } else {
                extra = std::make_unique<ggml_openvino_weight_extra>(std::move(result.weights), result.weight_node);
            }

            tensor->extra = extra.get();
            ctx->tensor_extras[tensor] = std::move(extra);

            if (!ctx->is_remote) {
                if (ggml_openvino_weight_buffers_released()) {
                    GGML_ABORT(
                        "ggml-openvino: loading a new model while GGML_OPENVINO_RELEASE_WEIGHTS pinned a previous "
                        "model's compiled graph. This mode supports a single model per process; unset it for "
                        "multi-model runs.");
                }
                ggml_openvino_register_weight_buffer(ctx->data(), ctx->size());
            }

        } catch (const std::exception & e) {
            GGML_LOG_ERROR("%s: failed to process weight tensor for %s: %s\n", __func__, tensor->name, e.what());
            memcpy((char *) tensor->data + offset, data, size);
        }
    } else {
        if (ctx->is_remote) {
            cl_command_queue queue = ggml_openvino_get_cl_queue();
            auto mem_cpy_fn = ggml_openvino_get_clEnqueueMemcpyINTEL();
            if (queue != nullptr && mem_cpy_fn != nullptr) {
                cl_int err =
                    mem_cpy_fn(queue, CL_TRUE, (char *) tensor->data + offset, data, size, 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL failed with error %d\n", __func__, err);
                }
            } else {
                GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemcpyINTEL not available for GPU buffer\n", __func__);
            }
        } else {
            memcpy((char *) tensor->data + offset, data, size);
        }

        auto extra = ggml_openvino_create_tensor_extra_unique(tensor, ctx->is_remote);
        if (extra == nullptr) {
            return;
        }

        tensor->extra = extra.get();
        ctx->tensor_extras[tensor] = std::move(extra);
    }
}

static void ggml_backend_openvino_buffer_get_tensor(ggml_backend_buffer_t buffer,
                                                    const ggml_tensor * tensor,
                                                    void * data,
                                                    size_t offset,
                                                    size_t size) {
    GGML_ASSERT(tensor != nullptr && tensor->data != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (ctx->is_remote) {
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_cpy_fn = ggml_openvino_get_clEnqueueMemcpyINTEL();
        if (queue != nullptr && mem_cpy_fn != nullptr) {
            cl_int err =
                mem_cpy_fn(queue, CL_TRUE, data, (const char *) tensor->data + offset, size, 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL failed with error %d\n", __func__, err);
            }
        } else {
            GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemcpyINTEL not available for GPU buffer\n", __func__);
        }
    } else {
        memcpy(data, (const char *) tensor->data + offset, size);
    }
}

static bool ggml_backend_openvino_buffer_cpy_tensor(ggml_backend_buffer_t buffer,
                                                    const ggml_tensor * src,
                                                    ggml_tensor * dst) {
    GGML_ASSERT(src != nullptr && dst != nullptr);
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;

    if (ctx->is_remote) {
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_cpy_fn = ggml_openvino_get_clEnqueueMemcpyINTEL();
        if (queue == nullptr || mem_cpy_fn == nullptr) {
            GGML_LOG_ERROR("%s: no OpenCL queue or clEnqueueMemcpyINTEL not available for GPU buffer\n", __func__);
            return false;
        }
        if (ggml_backend_buffer_is_host(src->buffer)) {
            cl_int err = mem_cpy_fn(queue, CL_TRUE, dst->data, src->data, ggml_nbytes(src), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL (host-to-device) failed with error %d\n", __func__, err);
                return false;
            }
            return true;
        }
        if (ggml_backend_buffer_is_openvino(src->buffer)) {
            ggml_backend_openvino_buffer_context * src_ctx =
                (ggml_backend_openvino_buffer_context *) src->buffer->context;
            if (src_ctx->is_remote) {
                cl_int err = mem_cpy_fn(queue, CL_TRUE, dst->data, src->data, ggml_nbytes(src), 0, nullptr, nullptr);
                if (err != CL_SUCCESS) {
                    GGML_LOG_ERROR("%s: clEnqueueMemcpyINTEL (device-to-device) failed with error %d\n", __func__, err);
                    return false;
                }
                return true;
            }
        }
        return false;
    }

    if (ggml_backend_buffer_is_host(src->buffer)) {
        memcpy(dst->data, src->data, ggml_nbytes(src));
        return true;
    }
    return false;
}

static void ggml_backend_openvino_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    GGML_ASSERT(ctx->data() != nullptr);
    if (ctx->is_remote) {
        cl_command_queue queue = ggml_openvino_get_cl_queue();
        auto mem_fill_fn = ggml_openvino_get_clEnqueueMemFillINTEL();
        if (queue != nullptr && mem_fill_fn != nullptr) {
            uint8_t pattern = value;
            cl_int err = mem_fill_fn(queue, ctx->data(), &pattern, sizeof(pattern), ctx->size(), 0, nullptr, nullptr);
            if (err != CL_SUCCESS) {
                GGML_LOG_WARN("%s: clEnqueueMemFillINTEL failed with error %d\n", __func__, err);
            }
            clFinish(queue);
        } else {
            GGML_LOG_WARN("%s: no OpenCL queue or clEnqueueMemFillINTEL not available for GPU buffer clear\n",
                          __func__);
        }
    } else {
        memset(ctx->data(), value, ctx->size());
    }
}

static const ggml_backend_buffer_i ggml_backend_openvino_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_openvino_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_openvino_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_openvino_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_openvino_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_openvino_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_openvino_buffer_get_tensor,
    /* .set_tensor_2d   = */ NULL,
    /* .get_tensor_2d   = */ NULL,
    /* .cpy_tensor      = */ ggml_backend_openvino_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_openvino_buffer_clear,
    /* .reset           = */ NULL,
};

ggml_backend_buffer_t ggml_backend_openvino_buffer_alloc(ggml_backend_buffer_type_t buft, int device, size_t size) {
    ggml_backend_openvino_buffer_context * ctx = new ggml_backend_openvino_buffer_context(device, size);

    if (ctx->data() == nullptr && size > 0) {
        GGML_LOG_ERROR("%s: failed to allocate buffer of size %zu\n", __func__, size);
        delete ctx;
        return nullptr;
    }

    return ggml_backend_buffer_init(buft, ggml_backend_openvino_buffer_interface, ctx, size);
}

GGML_BACKEND_API bool ggml_backend_buffer_is_openvino(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_openvino_buffer_free_buffer;
}

GGML_BACKEND_API size_t ggml_backend_openvino_buffer_get_ctx_id(ggml_backend_buffer_t buffer) {
    if (!ggml_backend_buffer_is_openvino(buffer)) {
        return 0;
    }
    ggml_backend_openvino_buffer_context * ctx = (ggml_backend_openvino_buffer_context *) buffer->context;
    return ctx->id;
}

bool ggml_openvino_buffer_is_remote(const ggml_tensor * tensor) {
    if (tensor == nullptr || tensor->buffer == nullptr) {
        return false;
    }
    if (!ggml_backend_buffer_is_openvino(tensor->buffer)) {
        return false;
    }
    auto * ctx = static_cast<ggml_backend_openvino_buffer_context *>(tensor->buffer->context);
    return ctx->is_remote;
}

void ggml_openvino_buffer_register_extra(ggml_tensor * tensor, ggml_openvino_extra_base * extra) {
    GGML_ASSERT(tensor != nullptr);
    GGML_ASSERT(tensor->buffer != nullptr);
    GGML_ASSERT(ggml_backend_buffer_is_openvino(tensor->buffer));

    auto * ctx = static_cast<ggml_backend_openvino_buffer_context *>(tensor->buffer->context);

    ctx->tensor_extras[tensor].reset(extra);
    tensor->extra = extra;
}
