#pragma once

#include <cstddef>
#include <memory>

namespace ov {
class Tensor;
}

class ggml_openvino_buffer_storage {
public:
    virtual ~ggml_openvino_buffer_storage() = default;

    virtual void * data() const noexcept = 0;
    virtual size_t size() const noexcept = 0;
    virtual std::shared_ptr<ov::Tensor> ov_buffer() const noexcept = 0;
};

std::unique_ptr<ggml_openvino_buffer_storage> ggml_openvino_create_buffer_storage(size_t size, bool is_remote);
