#pragma once

#include <cstddef>

// Host weight-buffer release (GGML_OPENVINO_RELEASE_WEIGHTS, GPU only).
// register: record a host weight buffer (idempotent per data pointer).
// release:  madvise(MADV_DONTNEED) all registered buffers, dropping their RSS.
// released: true once release has run (used to fail-fast on post-release recompile).
void ggml_openvino_register_weight_buffer(void * data, size_t size);
void ggml_openvino_release_weight_buffers();
bool ggml_openvino_weight_buffers_released();
