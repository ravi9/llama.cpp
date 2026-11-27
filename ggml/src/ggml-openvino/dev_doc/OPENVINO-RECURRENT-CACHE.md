# Current Qwen GPU implementation

The default Qwen3.5/Qwen3.6 MTP path uses pending-token replay, dynamic stateless execution, and stable pure-top-k sampling. When MTP discards a suffix, the next batch replays the retained tokens before processing new tokens. The implementation applies to `qwen35` and `qwen35moe` with one or multiple slots. The reference model is the UD Qwen3.6-35B-A3B MTP GGUF.

Configure `GGML_OPENVINO_DEVICE=GPU`, native softplus, and the documented requantization policy. The matching GPU plugin structurally fuses complete replay groups into its internal convolution and Gated Delta Net ops. Regenerate compiled blobs after switching to the writable-input replay interface. Historical measurements retain their original source revisions.

## Buffer ownership and operation boundaries

| State | Storage and lifetime | Update path |
|---|---|---|
| Attention K/V | Full caller-owned USM allocation, sized by context and KV layout | Native stateless KV fusion when eligible; SDPA consumes the live prefix |
| Convolution `cache_r` | Caller-owned USM row bank with history and pending columns | Fused reset, original-row gather/reorder, convolution and ordered history/pending writes |
| GDN `cache_s`, `cache_spk`, `cache_spv`, `cache_spgb` | Caller-owned USM row banks with committed state and pending token inputs | Fused reset, original-row gather/reorder, GDN and ordered state/pending writes |
| Runtime controls | Reused USM host storage on GPU | Row maps, offsets and write controls remain runtime inputs |
| Protected source rows | Private device scratch | Staged before writes to protect cycles, duplicate sources and overlaps |
| Inactive sampling results | Private device-USM scratch | Discarded without touching inactive GGML output buffers |

Recurrent-cache size depends on slot capacity and pending-token capacity. Each fused cache input Parameter binds a caller-owned writable allocation.

The GPU plugin uses two stages: first preserve required original rows in scratch while applying logical reset and precision rounding; then compute and perform ordered writes. This avoids overwriting a source before another destination has read it. Identity reorder sources need no scratch writes. In-place execution removes the spare full output bank, not every cache read or internal memory transaction.

`RecurrentConvCacheReplayUpdate` returns convolution `[1,B,T,C]` and writes history and pending columns. `RecurrentGatedDeltaNetReplayUpdate` returns attention and writes committed state and pending K, V, gate and beta. Both kernels write their cache inputs directly.

## Convolution cache stores

The structural replay graph gathers history using the retained-token count, concatenates live input, and uses one grouped convolution. A slice selects history and pending columns for the next batch, and ScatterNDUpdate produces the updated bank for the Assign sink. Row maps, replay counts and reset rows are runtime inputs.

## Replay graph contract

Qwen selects the recurrent graph when the target context is created. `n_rs_seq=0` selects the ordinary graph. With `--spec-type draft-mtp` and a positive draft limit, `n_rs_seq` is the draft limit and `rs_replay` selects the replay graph even for a fresh prefill. Both forms use `GGML_OP_GATED_DELTA_NET`. The replay form sets `op_params[1]` to the pending-token capacity and adds inputs for pending K, V, gate/beta, and the per-sequence replay count.

A fresh prompt supplies a replay count of zero. Prefill microbatches longer than `n_rs_seq + 1` tokens per sequence commit their final state. The unfused OpenVINO graph processes current tokens in a `Loop` and uses a bounded unrolled recurrence for the retained prefix.

The backend emits a standard-op graph for each complete dynamic stateless replay group. The computation feeds later layers and the updated banks feed standard `Assign` sinks. Fusion requires each bank to be a direct FP32 `Parameter` of shape `[1, 1, slots, features]` backed by caller-owned writable GPU USM. The backend opts each bank into in-place fusion by giving its sink the variable ID `__ggml_replay_cache_<Parameter friendly name>`.

The GPU plugin requires the transformed model to preserve its `Result` count. Sinks express writeback separately from that output interface. The pass can replace the computation and remove its matched sinks and variables while keeping the Results. The internal op then writes through the original input binding, which avoids the GPU runtime changes that would be needed to alias a cache output to its input.

The variable ID supplies the write permission. `RecurrentReplayFusion` still matches the computation, row controls, reset, reordering and writeback structure, and requires exclusive bank readers and a live computation output. The kernel checks for overlapping buffer ranges. When a sink remains, including on a GPU plugin without this pass, llama.cpp uses `query_state()` to find it and copies its state to the original GGML bank after inference. Mixed-precision execution of that graph can round bank values.

The backend builds the graph in [`openvino/op/recurrent_replay_graph.hpp`](../openvino/op/recurrent_replay_graph.hpp). The GPU matcher in `src/plugins/intel_gpu/src/plugin/transformations/recurrent_replay_fusion.cpp` compares it to a reference built from the OpenVINO copy of `recurrent_replay_graph.hpp` in the same directory. When upstream modeling changes the GGML graph, adapt the backend recognition and check what it emits. If the emitted replay pattern or its semantics change, update the reference and the relevant matcher and kernel logic with the translation. Verify that the graph still fuses and that the standard-op writeback still works.

## Attention KV layout

`GGML_OPENVINO_ENABLE_KV_4D=1` enables eligible native stateless in-place attention updates. The current eligibility covers one physical stream, including unified KV. Explicit split multi-slot KV retains the full-cache path and can be much slower even with only one active slot. Use `--kv-unified` for multi-slot OV GPU serving.

On dynamic CPU and GPU, `GGML_OPENVINO_KV_SLICE` is a binding switch and defaults to `1`. For an eligible contiguous ordinary attention K/V root with one active sequence at offset zero, it binds a view of the allocation sized to the attended prefix. Recurrent banks and sliding-window attention use the full binding. The 4-D in-place attention path binds the full allocation first and exposes the attended prefix inside the graph. Dynamic compiled blobs can accept either eligible binding.

`GGML_OPENVINO_GPU_KV_ROUND_UP` controls the attended width separately on dynamic stateless GPU. With `0`, the decoder uses the GGML mask width, which ordinary GGML attention KV already pads in steps of 256 rows. The default `1` uses power-of-two windows when the graph input has more than 16 tokens and coarser aligned steps for smaller batches. The step starts at 256 rows and doubles whenever the width reaches eight times the current step. At 33,024 GGML rows, the added rounding can expand attention to 40,960 rows for decode or 65,536 for a larger prefill input, subject to the context cap. Added mask columns contain negative infinity. The current cache-only batched measurements used `GGML_OPENVINO_ENABLE_KV_4D=1`, the default `KV_SLICE` setting and `GGML_OPENVINO_GPU_KV_ROUND_UP=0`.

On static NPU, `GGML_OPENVINO_KV_SLICE` also selects the base of the compiled KV windows. The default base is 1024 and an explicit multiple of 256 changes it. The window is the smallest `base * 2^k` that covers the attended length, capped by the context.

## Stable MTP sampling

The [graph walkthrough](MTP-SAMPLING-GRAPH.md) describes the fixed sampling interface. The backend keeps pure top-k branches for all configured slots, packs runtime row indices, and uses scratch for inactive outputs. It does not change the original cgraph or execute inactive stateful sampler chains. Only the small sampling branches stay at configured capacity; the expensive draft layer processes active rows.

One compatible draft graph is shared across active subsets. Trunk and genuinely different graph stages/configurations remain separate. Prepare compatible blobs once, then use cache-only mode. The first import and runtime preparation still have a cost.

## Validation

Validation checks the caller's mutated convolution bank against the expected reset, reorder and history writes. The four-layer Qwen3.6 model exercises replay at one and four active sequences.

## Debugging and source map

- `ggml-decoder.cpp`: dynamic dimensions, control inputs, output names and `cgraph_ov.txt` dump.
- `utils.cpp`: compute-branch filtering, compiled reuse and input/output binding.
- `openvino/recurrent_replay_fusion.cpp`: GGML group recognition and standard-op replay translation.
- `openvino/translate_session.cpp`: runtime sampling aliases and model construction.
- OV `recurrent_replay_fusion.cpp` and `recurrent_cache_replay`: structural matching, writable-input validation, source staging and GPU computation.

`GGML_OPENVINO_DUMP_CGRAPH=1` dumps the raw graph before filtering. It includes COMPUTE/OUTPUT flags and byte offsets and overwrites `cgraph_ov.txt` on each call. Enable dumps only for diagnosis; use the captured examples for a static view. GPU profiles and OpenCL copy traces add overhead, and lack of an explicit copy call does not prove zero traffic inside a kernel.
