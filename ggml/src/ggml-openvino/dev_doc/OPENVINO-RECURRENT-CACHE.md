# Current Qwen GPU implementation

As of 2026-11-03, the default Qwen3.5/Qwen3.6 path is dynamic stateless execution with llama.cpp custom recurrent ops and stable pure-top-k MTP translation. It supports one or multiple slots and rollback. The implementation applies to `qwen35` and `qwen35moe`; the reference model is the UD Qwen3.6-35B-A3B MTP GGUF.

The recurrent ops run as OpenVINO GPU custom layers: `openvino/kernels/recurrent_cache.xml` and its `.cl` files are embedded at build time, written to a temporary directory at runtime and passed as the GPU `CONFIG_FILE`. No OpenVINO runtime change is required; GPU plugin fix openvinotoolkit/openvino#39354 is still recommended. Configure `GGML_OPENVINO_DEVICE=GPU`, native softplus, and the documented requantization policy. The current cache translation revision is 15.

## Buffer ownership and operation boundaries

| State | Storage and lifetime | Update path |
|---|---|---|
| Attention K/V | Full caller-owned USM allocation, sized by context and KV layout | Native stateless KV fusion when eligible; SDPA consumes the live prefix |
| Convolution `cache_r` | Full caller-owned USM row bank, sized by slot/snapshot capacity | Fused reset, original-row gather/reorder, convolution and ordered history/snapshot writes |
| GDN `cache_s` | Full caller-owned USM row bank, sized by slot/snapshot capacity | Fused reset, original-row gather/reorder, GDN and ordered final/snapshot writes |
| Runtime controls | Reused USM host storage on GPU | Row maps, offsets and write controls remain runtime inputs |
| Protected source rows | Private device scratch | Staged before writes to protect cycles, duplicate sources and overlaps |
| Inactive sampling results | Private device-USM scratch | Discarded without touching inactive GGML output buffers |

Attention context capacity does not set recurrent-cache size. The update ops write the bound cache allocation in place and do not return it, so the graph has no cache Result. There are no rotation banks or pointer swaps.

Each recurrence uses two custom ops, because a custom op runs a single kernel. `GgmlRecurrentCacheStage` copies the source rows the update reads (active rows plus non-identity reorder rows, with reset applied) into f32 scratch. The update op reads only that scratch and writes the cache, so cycles, duplicate sources and overlaps are safe.

| Op | Inputs | Output |
|---|---|---|
| `GgmlRecurrentCacheStage` | cache, active rows, extra rows, reset begin/length, extra offsets | staged rows `[1,1,B+N,F]` f32 |
| `GgmlRecurrentConvUpdate` | stage inputs, QKV `[1,B,C,T]`, snapshot windows, depthwise weights, staged rows, token count | convolution `[1,B,T,C]` |
| `GgmlRecurrentGdnUpdate` | Q/K/V, cache, gate/beta, stage controls, snapshot write control, staged rows, token count | attention `[B,T,H,V]` |
| `GgmlRecurrentGdnSnapshots` | as GDN update, with the initial state as cache and staged rows | attention and newest states |

GDN keeps the newest `min(T,snapshot_count)` states, newest first, and requires a key head size that is a multiple of 16 and at most 256. Cache inputs are marked precision-sensitive so FP16 compression does not insert a converted copy for the kernel to write into. Op contracts are in `openvino/op/recurrent_cache_update.hpp`.

Custom kernels are compiled per layer and input shape: dims are JIT constants and the plugin adds the layer name to the source. The builders pad a dynamic token axis to a power of two and pass the real count as an input, then slice the outputs, which bounds the shape count at about 5% prefill cost on small models. First use of a bucket or batch size compiles every layer; the driver's persistent cache (`~/.cache/neo_compiler_cache`) serves later runs.

## Generic fallback and restrictions

Fusion requires a remote (USM) cache root, a cache input that is a whole Parameter through Reshapes only, and no later reader of the root or its views. Otherwise the group keeps the ordinary GGML decomposition with a warning.

Custom ops are disabled with `GGML_OPENVINO_CACHE_DIR` or `GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR`, because the GPU plugin does not export the shape-inference op and work-size rules of dynamic custom layers. Fusion is then off and snapshot GDN is reported unsupported, so the scheduler runs it on another backend; that CPU snapshot fallback currently fails `recurrent_slots` rollback (also before this change). `GGML_OPENVINO_DISABLE_FUSED_RECURRENCE=1` retains the ordinary cache decomposition for diagnosis; unmatched snapshot GDN still uses `GgmlRecurrentGdnSnapshots`. Stateful and NPU MTP are not the supported default; this validation does not claim those paths.

## Attention KV layout

`GGML_OPENVINO_ENABLE_KV_4D=1` enables eligible native stateless in-place attention updates. The current eligibility covers one physical stream, including unified KV. Explicit split multi-slot KV retains the full-cache path and can be much slower even with only one active slot. Use `--kv-unified` for multi-slot OV GPU serving.

Leave `GGML_OPENVINO_KV_SLICE` at its default (`1`). Live-prefix slicing applies to eligible ordinary attention K/V roots, not recurrent caches. In-place KV fusion can bind the full allocation while exposing a live prefix internally. KV-slice on/off is a runtime binding choice and does not require different compiled blobs; incompatible graph layouts/capacities can still do so.

## Stable MTP sampling

The [graph walkthrough](MTP-SAMPLING-GRAPH.md) describes the fixed sampling interface. The backend keeps pure top-k branches for all configured slots, packs runtime row indices, and uses scratch for inactive outputs. It does not change the original cgraph or execute inactive stateful sampler chains. Only the small sampling branches stay at configured capacity; the expensive draft layer processes active rows.

One compatible draft graph is shared across active subsets. Trunk and genuinely different graph stages/configurations remain separate. With compiled-model caching (which disables the recurrent custom ops), prepare a revision-15 cache once, then use cache-only mode. Exhaustive active-subset priming is no longer required, though first-use model import and runtime preparation still have a cost.

## Validation

The custom kernels are checked end to end on Qwen3.5 models: greedy output against the unfused path (`GGML_OPENVINO_DISABLE_FUSED_RECURRENCE=1`), multi-slot serving with `-np 4 --kv-unified`, and speculative rollback through the snapshot op. The reference top-k is computed on CPU from raw logits produced by the same model backend; this is not a claim of full-model CPU/GPU numerical equivalence.

## Debugging and source map

- `ggml-decoder.cpp`: dynamic dimensions, control inputs, output names and `cgraph_ov.txt` dump.
- `utils.cpp`: compute-branch filtering, compiled reuse and input/output binding.
- `openvino/recurrent_cache_fusion.cpp`: structural pattern matching and custom op translation.
- `openvino/op/recurrent_cache_update.*`: custom op classes, in-place checks and token bucketing.
- `openvino/kernels/`: custom-layer XML and OpenCL kernels, embedded by `CMakeLists.txt`.
- `openvino/translate_session.cpp`: runtime sampling aliases and model construction.

`GGML_OPENVINO_DUMP_CGRAPH=1` dumps the raw graph before filtering. It includes COMPUTE/OUTPUT flags and byte offsets and overwrites `cgraph_ov.txt` on each call. Enable dumps only for diagnosis; use the captured examples for a static view. GPU profiles and OpenCL copy traces add overhead, and lack of an explicit copy call does not prove zero traffic inside a kernel.
