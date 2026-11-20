#pragma once

// NPU (NPUW) compile configuration. Everything that decides which OpenVINO properties an NPU graph is
// compiled with lives here, so "what does the NPU compile with" is answered in one place:
//   * the backend-wide base config (NPUW defaults, named env aliases, GGML_OPENVINO_NPU_CONFIG overrides)
//   * the per-graph variants of it (cacheless graphs, GATED_DELTA_NET graphs, static prefill, static decode)

#include <openvino/core/any.hpp>
#include <string>

class GgmlOvDecoder;

// Base config for every NPU graph, built once at backend init. GGML_OPENVINO_NPU_CONFIG ("K=V,...") is
// applied last, so it overrides the defaults and the named aliases.
ov::AnyMap ggml_openvino_npu_build_compile_config(const char * cache_dir);

// Cacheless (encoder) graphs compile on the plain NPU plugin: drop every NPUW key.
ov::AnyMap ggml_openvino_npu_without_npuw(const ov::AnyMap & config);

// Graphs with GATED_DELTA_NET (Qwen3.5). The driver compiler gives wrong output for them and the compiler bundled
// with OpenVINO does not, so they compile with NPU_COMPILER_TYPE=PLUGIN unless the environment sets the compiler
// type (GGML_OPENVINO_NPU_COMPILER_TYPE, or NPU_COMPILER_TYPE in GGML_OPENVINO_NPU_CONFIG).
ov::AnyMap ggml_openvino_npu_gdn_config(const ov::AnyMap & base);

// True when ggml_openvino_npu_gdn_config() switches the compiler. Part of the model cache key.
bool ggml_openvino_npu_gdn_uses_plugin_compiler();

// Graphs with a MoE block in the batched form (see translate_moe_tiled). NPUW isolates the
// experts into functions and runs only the routed experts per token, with the router on the host (the HOST_ROUTED mode of
// NPUW_LLM_*_MOE_HINT, which only the NPUW LLM pipeline sets itself), with one expert chunk size and no expert request
// pool. Keys the environment already sets (GGML_OPENVINO_NPU_CONFIG) are kept.
ov::AnyMap ggml_openvino_npu_moe_config(const ov::AnyMap & base);

// Static prefill graph. With GGML_OPENVINO_NPU_HFA_PREFILL, switches NPUW to the HFA attention pipeline.
ov::AnyMap ggml_openvino_npu_prefill_config(const ov::AnyMap & base, const GgmlOvDecoder & decoder);

// Static decode graph. With GGML_OPENVINO_NPU_DECODE_UNFOLD, unfolds NPUW function calls into separate infer
// requests, unless NPUW_UNFOLD_IREQS is already set.
ov::AnyMap ggml_openvino_npu_decode_config(const ov::AnyMap & base, const std::string & device);
