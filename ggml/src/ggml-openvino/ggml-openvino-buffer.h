#pragma once

#include "ggml-backend.h"
#include "ggml-openvino-extra.h"
#include "ggml-openvino.h"

#include <cstddef>

/// \brief Allocate an OpenVINO backend buffer instance for the given buffer type and device.
/// \param buft OpenVINO backend buffer type used to initialize the buffer.
/// \param device OpenVINO device index associated with the buffer.
/// \param size Size in bytes to allocate for the buffer.
ggml_backend_buffer_t ggml_backend_openvino_buffer_alloc(ggml_backend_buffer_type_t buft, int device, size_t size);

/// \brief Return whether the tensor is backed by OpenVINO remote device memory.
/// \param tensor Tensor to query.
bool ggml_openvino_buffer_is_remote(const ggml_tensor * tensor);

/// \brief Register a tensor extra object with the owning OpenVINO buffer context.
/// \param tensor Tensor that owns the extra object through its OpenVINO buffer context.
/// \param extra Extra object to register and attach to the tensor.
void ggml_openvino_buffer_register_extra(ggml_tensor * tensor, ggml_openvino_extra_base * extra);
