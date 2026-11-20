#pragma once

// Shared with the in-tree translator: one copy lives in ../ggml-openvino/ and is compiled into both builds.
#include "../ggml-openvino/ggml-openvino-extra.h"

// Settings below are read only by the OpenVINO GGUF frontend path, so they live here and not in the shared file.

// Apply a constant scale (Granite's residual and attention multipliers) before the MatMul it follows instead of
// after it, so an f16 plugin does not overflow on blocks whose unscaled result is out of range. The rewrite is exact
// but only useful when the device executes in f16 and is overhead on f32, so it defaults to on for GPU and NPU;
// GGML_OPENVINO_DISABLE_SCALE_REORDER=1 turns it off.
inline bool ggml_openvino_scale_reorder_enabled() {
    return (ggml_openvino_is_gpu() || ggml_openvino_is_npu()) &&
           ggml_openvino_getenv_int("GGML_OPENVINO_DISABLE_SCALE_REORDER") == 0;
}
