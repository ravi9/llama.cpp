#include "ggml-openvino-buffer-storage.h"

#include "ggml-impl.h"
#include "ggml-openvino-extra.h"

#include <cerrno>
#include <climits>
#include <cstring>
#include <openvino/core/type/element_type.hpp>
#include <openvino/runtime/intel_gpu/ocl/ocl.hpp>
#include <openvino/runtime/tensor.hpp>

#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif

class ggml_openvino_host_buffer_storage : public ggml_openvino_buffer_storage {
public:
    explicit ggml_openvino_host_buffer_storage(size_t size) :
        size_(size) {
        if (size_ == 0) {
            return;
        }

#ifndef _WIN32
        if (const char * spill_dir = ggml_openvino_getenv_str("GGML_OPENVINO_SPILL_DIR")) {
            char path[PATH_MAX];
            snprintf(path, sizeof(path), "%s/ggml-ov-weights-%d-XXXXXX", spill_dir, (int) getpid());
            int fd = mkstemp(path);
            if (fd < 0) {
                GGML_LOG_ERROR("%s: mkstemp(%s) failed: %s\n", __func__, path, strerror(errno));
                return;
            }
            unlink(path);
            if (ftruncate(fd, (off_t) size_) != 0) {
                GGML_LOG_ERROR("%s: ftruncate(%zu) failed: %s\n", __func__, size_, strerror(errno));
                close(fd);
                return;
            }
            void * m = mmap(nullptr, size_, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
            close(fd);
            if (m == MAP_FAILED) {
                GGML_LOG_ERROR("%s: mmap(%zu) failed: %s\n", __func__, size_, strerror(errno));
                return;
            }
            data_ = m;
            spill_mapping_ = m;
            spill_size_ = size_;
            GGML_LOG_INFO("%s: weight buffer spilled to %s (%zu MB, file-backed)\n", __func__, spill_dir,
                          size_ / 1024 / 1024);
            ov_buffer_ = std::make_shared<ov::Tensor>(ov::element::u8, ov::Shape{size_}, data_);
        } else
#endif
        {
#ifdef _WIN32
            if (ggml_openvino_getenv_str("GGML_OPENVINO_SPILL_DIR")) {
                GGML_LOG_WARN("%s: GGML_OPENVINO_SPILL_DIR is not supported on Windows, ignoring\n", __func__);
            }
#endif
            data_ = ggml_aligned_malloc(size_);
            GGML_ASSERT(data_);
            memset(data_, 0, size_);
            ov_buffer_ = std::make_shared<ov::Tensor>(ov::element::u8, ov::Shape{size_}, data_);
        }
    }

    ~ggml_openvino_host_buffer_storage() override {
#ifndef _WIN32
        if (spill_mapping_ != nullptr) {
            munmap(spill_mapping_, spill_size_);
            return;
        }
#endif
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
#ifndef _WIN32
    void * spill_mapping_ = nullptr;
    size_t spill_size_ = 0;
#endif
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
