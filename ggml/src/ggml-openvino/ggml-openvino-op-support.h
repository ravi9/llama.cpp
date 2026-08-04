#pragma once

#include "ggml-backend.h"
#include "ggml.h"

bool ggml_openvino_device_supports_op_impl(ggml_backend_dev_t dev, const ggml_tensor * op);
