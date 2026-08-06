#include "ggml-openvino-buffer-storage.h"

#include "ggml-impl.h"
#include "ggml-openvino-extra.h"

#include <cstring>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/tensor.hpp>

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

    std::shared_ptr<ov::Tensor> ov_buffer() const noexcept override {
        return ov_buffer_;
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

    std::shared_ptr<ov::Tensor> ov_buffer() const noexcept override {
        return ov_buffer_;
    }

private:
    void * data_ = nullptr;
    size_t size_ = 0;
    std::shared_ptr<ov::Tensor> ov_buffer_;
};

std::unique_ptr<ggml_openvino_buffer_storage> ggml_openvino_create_buffer_storage(size_t size, bool is_remote) {
    if (is_remote) {
        return std::make_unique<ggml_openvino_remote_buffer_storage>(size);
    }

    return std::make_unique<ggml_openvino_host_buffer_storage>(size);
}
