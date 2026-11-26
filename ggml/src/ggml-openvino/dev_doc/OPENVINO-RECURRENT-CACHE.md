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

Recurrent-cache size depends on slot capacity and pending-token capacity. Each fused cache input Parameter binds a caller-owned writable allocation. llama.cpp represents writeback as standard `Assign` sinks. The GPU pass replaces matching sinks and their computation with one internal op, preserving the model's Results.

The GPU plugin uses two stages: first preserve required original rows in scratch while applying logical reset and precision rounding; then compute and perform ordered writes. This avoids overwriting a source before another destination has read it. Identity reorder sources need no scratch writes. In-place execution removes the spare full output bank, not every cache read or internal memory transaction.

`RecurrentConvCacheReplayUpdate` returns convolution `[1,B,T,C]` and writes history and pending columns. `RecurrentGatedDeltaNetReplayUpdate` returns attention and writes committed state and pending K, V, gate and beta. Both kernels write their cache inputs directly.

## Convolution cache stores

The structural replay graph gathers history using the retained-token count, concatenates live input, and uses one grouped convolution. A slice selects history and pending columns for the next batch, and ScatterNDUpdate produces the updated bank for the Assign sink. Row maps, replay counts and reset rows are runtime inputs.

## Generic fallback and restrictions

Fusion requires direct writable GPU USM Parameters with exclusive bank readers. The GPU implementation checks that bank ranges do not overlap other inputs or the computation output. When the translated graph runs without a matching fusion, the Assign sinks retain the updated banks and llama.cpp copies their states into the GGML buffers after inference. Mixed-precision GPU execution of the unfused graph can round bank values. The GPU replay translator requires complete dynamic stateless groups.

## Attention KV layout

`GGML_OPENVINO_ENABLE_KV_4D=1` enables eligible native stateless in-place attention updates. The current eligibility covers one physical stream, including unified KV. Explicit split multi-slot KV retains the full-cache path and can be much slower even with only one active slot. Use `--kv-unified` for multi-slot OV GPU serving.

Leave `GGML_OPENVINO_KV_SLICE` at its default (`1`). Live-prefix slicing applies to eligible ordinary attention K/V roots, not recurrent caches. In-place KV fusion can bind the full allocation while exposing a live prefix internally. KV-slice on/off is a runtime binding choice and does not require different compiled blobs; incompatible graph layouts/capacities can still do so.

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
