#pragma once

#include "ggml-backend.h"

#include <cstring>

#ifdef __cplusplus
extern "C" {
#endif

#define GGML_OPENVINO_NAME "OPENVINO"

// backend API
/// \brief Initialize an OpenVINO backend instance for the given device.
/// \param device OpenVINO device index to initialize.
/// \return OpenVINO backend instance, or nullptr if initialization fails.
GGML_BACKEND_API ggml_backend_t ggml_backend_openvino_init(int device);

/// \brief Return whether the backend instance is an OpenVINO backend.
/// \param backend Backend instance to query.
/// \return True if the backend is an OpenVINO backend.
GGML_BACKEND_API bool ggml_backend_is_openvino(ggml_backend_t backend);

/// \brief Return whether the buffer instance is an OpenVINO backend buffer.
/// \param buffer Backend buffer instance to query.
/// \return True if the buffer is an OpenVINO backend buffer.
GGML_BACKEND_API bool ggml_backend_buffer_is_openvino(ggml_backend_buffer_t buffer);

/// \brief Return whether the buffer type is an OpenVINO device buffer type.
/// \param buft Backend buffer type to query.
/// \return True if the buffer type is an OpenVINO device buffer type.
GGML_BACKEND_API bool ggml_backend_buft_is_openvino(ggml_backend_buffer_type_t buft);

/// \brief Return whether the buffer type is an OpenVINO host buffer type.
/// \param buft Backend buffer type to query.
/// \return True if the buffer type is an OpenVINO host buffer type.
GGML_BACKEND_API bool ggml_backend_buft_is_openvino_host(ggml_backend_buffer_type_t buft);

/// \brief Return the context identifier for an OpenVINO backend buffer.
/// \param buffer Backend buffer instance to query.
/// \return OpenVINO buffer context identifier, or 0 if the buffer is not an OpenVINO buffer.
GGML_BACKEND_API size_t ggml_backend_openvino_buffer_get_ctx_id(ggml_backend_buffer_t buffer);

// device buffer
/// \brief Return the OpenVINO device buffer type for the given device.
/// \param device OpenVINO device index.
/// \return OpenVINO device buffer type.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_buffer_type(int device);

/// \brief Return the OpenVINO host buffer type for the given device.
/// \param device OpenVINO device index.
/// \return OpenVINO host buffer type.
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_openvino_host_buffer_type(int device);

/// \brief Return the number of OpenVINO devices available to the backend.
/// \return Number of OpenVINO devices.
GGML_BACKEND_API int ggml_backend_openvino_get_device_count(void);

/// \brief Return the OpenVINO backend registry instance.
/// \return OpenVINO backend registry instance.
GGML_BACKEND_API ggml_backend_reg_t ggml_backend_openvino_reg(void);

#ifdef __cplusplus
}
#endif
