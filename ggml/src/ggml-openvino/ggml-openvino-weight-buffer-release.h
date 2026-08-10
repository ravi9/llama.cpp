#pragma once

#include <cstddef>

/// \brief Record a host weight buffer for later release when GGML_OPENVINO_RELEASE_WEIGHTS is enabled.
/// \param data Base address of the host weight buffer.
/// \param size Size in bytes of the host weight buffer.
void ggml_openvino_register_weight_buffer(void * data, size_t size);

/// \brief Release registered host weight-buffer pages with madvise(MADV_DONTNEED).
void ggml_openvino_release_weight_buffers();

/// \brief Return whether registered host weight buffers have already been released.
/// \return True after ggml_openvino_release_weight_buffers has run.
bool ggml_openvino_weight_buffers_released();
