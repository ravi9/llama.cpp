# Qwen3.5 0.8B F16: OpenVINO GPU investigation and optimization plan

Date: 2026-10-16. This document records the original investigation and plan. Steps 1–3 are now implemented locally; measurements and validation are recorded in [OPENVINO-QWEN35-08B-PERF-RESULTS.md](OPENVINO-QWEN35-08B-PERF-RESULTS.md). The baseline findings below describe the pre-optimization implementation. The current target has since changed to `~/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, keeping GPU, `q4_asym64_all`, native softplus, and disabled KV slicing; the results report records its retest separately.

OpenVINO GPU currently reaches **43.5–43.8 tokens/s**, compared with **57.91 tokens/s** for a new Vulkan Release build from the same working tree. The next priority is eliminating runtime shape synchronization: an ordinary single-token inference calls `clFinish` **164 times**, of which **163 are shape-inference synchronizations: 108 Reshapes and 55 Slices**. Cache allocations and bindings already use device USM. Native cache fusion helps, but leaves this synchronization problem and matrix-state writeback/layout work intact.

The user's 18.99 tokens/s OpenVINO example selected **CPU**, so it is not the GPU baseline. The user's expectation of further improvement is reasonable; reaching a particular throughput on the 35B model still needs measurement after the small-model changes.

## Reproducible baseline

Hardware: Intel Arc B390 integrated GPU / Core Ultra X7 358H. Model: `~/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf`. This dense, unquantized model removes MoE and requantization from the comparison.

- OpenVINO frontend: Release `build/MtpOVPerf`, llama.cpp HEAD `73f61e1cb`, with the existing uncommitted native recurrent-cache implementation.
- Runtime: `/home/zijun/openvino-build-mtp`, OpenVINO HEAD `21cacac33d`, with the existing native-op changes.
- Matched Vulkan: new Release `build/VulkanPerf08`, same current llama.cpp working tree, Vulkan SDK 1.4.357.1. The existing Debug `build/vulkan` was left unchanged.
- GPU workloads ran serially. No stateful tests or execution. No backend implementation files changed during this investigation.
- Environment is scrubbed of inherited OpenVINO/Vulkan controls before each run. Requantization and compiled-model disk caching are unset. Throughput excludes initial loading/compilation through llama-bench's warmup.

```sh
GGML_OPENVINO_DEVICE=GPU \
GGML_OPENVINO_NATIVE_SOFTPLUS=1 \
GGML_OPENVINO_DISABLE_KV_SLICE=1 \
build/MtpOVPerf/bin/llama-bench \
  -m "$HOME/models/Qwen3.5-0.8B-GGUF/Qwen3.5-0.8B-F16.gguf" \
  -p 0 -n 128 -r 3
```

| Run | tokens/s, benchmark mean ± SD | Approx. ms/token |
| --- | ---: | ---: |
| OV GPU, native cache ops | 43.76 ± 0.10 | 22.85 |
| OV GPU, independent repeat | 43.51 ± 0.11 | 22.98 |
| OV GPU, `GGML_OPENVINO_DISABLE_RECURRENT_CACHE=1` | 39.14 ± 0.12 | 25.55 |
| OV GPU, `GGML_OPENVINO_DISABLE_REMOTE_OUTPUTS=1` | 43.26 ± 0.11 | 23.12 |
| Existing Vulkan Debug, older build | 55.69 ± 0.25 | 17.96 |
| Matched Vulkan Release | 57.91 ± 0.17 | 17.27 |

Native fusion saves approximately **2.70 ms/token**, or 11.8% higher throughput, on this model. Approximately **5.6–5.7 ms/token** remains to match Vulkan Release. Disabling remote outputs changes latency by only about 0.26 ms relative to the first native run; this is a small, order-sensitive control, not a complete measurement of memory-copy cost.

Timing instrumentation is separate from these baselines. One profiling-level-1 run measured 41.24 tokens/s; a later verbose level-1 run measured 43.57. Its warmed medians were 0.100 ms for graph decoding, 22.637 ms for the inference phase including bindings, and 22.193 ms inside `InferRequest::infer`. Most latency is inside the runtime, rather than llama.cpp graph construction or tensor binding. Do not use the noisier instrumented result as a new baseline.

## Controlled token-count and snapshot comparison

The existing `tmp/mtp-gap-20261014/batch-bench.cpp` harness was reused, including a newly linked Vulkan Release executable. Each case uses a 128-token prefix, eight warmups, 16 measured calls, context 4096, one active sequence, hidden output enabled, and all input rows requesting logits. Means below are milliseconds per complete synchronized decode call.

| Case | OV GPU | Vulkan Release |
| --- | ---: | ---: |
| 1 token, snapshots disabled | 23.49 | 17.07 |
| 3 tokens, snapshots disabled | 24.44 | 18.40 |
| 1 token, two rollback positions configured | 29.84 | 17.85 |
| 3 tokens, two rollback positions configured | 31.82 | 19.32 |
| 3 tokens, reject two after each call | 32.01 | 18.59 |

On OV, ordinary T1→T3 adds only **0.95 ms**. Enabling snapshot storage adds **7.38 ms** to T3. Actual rejection adds approximately **0.19 ms** in this run. The slightly lower Vulkan reject-case timing is not evidence that rejection accelerates inference; the cases advance context length differently and are separate short measurements.

This isolates a substantial generic/snapshot-path penalty without any MoE or quantization. These are target-call measurements, not an MTP throughput prediction: draft generation, catch-up, acceptance, and host sampling must be measured separately on the target model.

## What the runtime actually does

### USM and model I/O

A scratch `LD_PRELOAD` probe inspects bound tensors and intercepts OpenCL submissions and USM copies. A warmed native T1 call has:

| Item | Count / size |
| --- | ---: |
| OV Parameters | 184 |
| OV Results | 51 |
| Cache inputs, all USM | 48 |
| Cache outputs, all USM | 48 |
| Host inputs | 136, mostly scalar metadata |
| Cache-r allocation represented at I/O | 18 × 73,728 bytes |
| Cache-s allocation represented at I/O | 18 × 1,048,576 bytes |
| Cache-k/cache-v allocation represented at I/O | 12 × 262,144 bytes |
| Host→device USM copies per inference | 136 calls, **1,588 bytes total** |
| USM-host→ordinary-host output copies | 3 calls, **1,001,472 bytes total** |
| OpenCL NDRange submissions | **789** |
| `clFinish` calls | **164** |

The three host outputs are logits (993,280 bytes), `h_nextn` (4,096 bytes), and `result_norm` (4,096 bytes). The cache byte counts are bound capacities, not host-transfer volumes. No full recurrent-cache host round trip appears in the intercepted copies. Device kernels still read, convert, transpose, and write cache data.

Code confirms the binding behavior: `ggml-openvino.cpp` allocates GPU `cache_*` buffers with `create_usm_device_tensor`; `ggml-openvino-extra.cpp` wraps their pointers as USMTensors; `utils.cpp` reuses them for both inputs and outputs. There is already **one final Result per cache per layer**. Reducing duplicate cache Results is therefore not the immediate opportunity.

The small host inputs are different: `get_ov_input_tensor()` allocates a fresh tensor for each extra scalar on every invocation. The runtime uploads them individually. Per-layer `rs_slot_begin_*` and `rs_src_begin_*` parameters are combined into small device-side Concat/Reshape expressions even for the native cache ops.

### Synchronization and shape inference

The probe measures roughly **12.9–13.3 ms spent inside `clFinish`** per warmed call, with raw inference around 22.4 ms. This includes completion of useful GPU work; it is **not** 13 ms of independently removable overhead.

An additional GPU verbose trace (`OV_VERBOSE=4`, `OV_VERBOSE_COLOR=0`) confirms three consecutive calls each contain exactly:

- **108 Reshape shape-inference synchronizations**, six per recurrent layer: QKV mixed, alpha, beta, gate, z, and final-output reshapes.
- **55 StridedSlice shape-inference synchronizations**.
- The remaining finish is inference completion; the probe records a separate final event wait as well.

`primitive_inst::update_shape()` waits for runtime value dependencies before calculating output layouts. The in-order shared OpenCL queue takes the `finish()` path. In `openvino/op/reshape.cpp`, cases 6 and 10 build shape vectors using **the tensor value** of `s_copy_active_slot_len`. This introduces synchronization even though the active sequence count is already known by the frontend on every call. `recurrent_cache_fusion.cpp` also slices the row map using that value; other active-state views and cache writeback slices retain similar dependencies.

This explains why fusing many GGML nodes does not eliminate the gap: arithmetic fusion does not automatically remove the surrounding shape computations or their queue drains. Removing waits blindly would be incorrect; their data dependencies must first be eliminated or satisfied safely.

### Device execution profile

Level-2 profiling substantially changes synchronization/timing (the short OV run reports only 17.34 tokens/s). Use it to locate kernels, not as an end-to-end benchmark. A warmed native profile has 1,180 entries marked EXECUTED, versus 1,777 with native cache fusion disabled. These entries include CPU/shape nodes and are not equivalent to kernel dispatches.

| Profiled work | Native OV device-time sum |
| --- | ---: |
| FullyConnected, 187 nodes | 14.697 ms |
| Transpose, 48 nodes | 1.032 ms |
| Convert, 180 entries | 0.377 ms |
| GatedDeltaNet, 18 | 0.274 ms |
| RecurrentStateGather, 18 | 0.246 ms |
| RecurrentConvCache, 18 | approximately 0.073 ms |
| All recorded executed entries | 17.837 ms |

The matched Vulkan instrumented profile sums to 17.858 ms, including approximately 15.573 ms of matmul work. The huge F16 vocabulary projection alone is approximately 4.45 ms in OV and 4.41 ms in Vulkan. Different profilers change scheduling, so these sums cannot be subtracted from plain wall time to calculate an exact overhead budget. They do suggest that dense matrix computation is not the first issue to address.

The matrix-state path still materializes gather output, transposes between GGML `[B,H,V,K]` and native GDN `[B,H,K,V]`, transposes results back, and assembles cache writeback. Snapshot routing adds more state-selection/writeback work. The current `RecurrentStateGather` also produces a complete prepared-cache copy before the final update. These are more promising fusion targets than micro-tuning the already small convolution-cache kernel.

## Implementation plan, in order

### 1. Remove active-count value dependencies from shapes

**Files:** `ggml-decoder.{h,cpp}`, `utils.cpp`, `openvino/op/reshape.cpp`, `openvino/op/view.cpp`, `openvino/utils.cpp`, `openvino/recurrent_cache_fusion.cpp`.

1. Expose a runtime active-row-map input whose **shape** is `[1,1,1,B]`, where B comes from the actual active recurrent-state view each compute. Bind the corresponding prefix of `rs_s_copy`; retain the full map separately for remainder reordering. Do not assume B=1 or equate B with configured slot capacity, total tokens, or snapshot count.
2. Derive `s_copy_active_slot_len` for shape expressions from `ShapeOf(active_rows)` through the existing shape-source mechanism. Use constant reshape patterns with `-1`/copied dimensions wherever that expresses the same shape. This targets all 108 recurrent reshape waits while preserving changing B and T.
3. Feed the native cache ops the already-shaped active map, removing their value-dependent prefix Slice. Remove redundant active-prefix slices only when the producer structurally guarantees that exact active output, for every supported B/T.
4. Inspect the executed graph to ensure the new expressions remain shape-only through GPU lowering. A frontend ShapeOf is insufficient if the plugin still makes the dependency value-driven.

**Gate:** changing-slot/rollback correctness tests pass; the 108 identified reshape waits disappear or any surviving dependency is individually explained. Repeat uninstrumented tg128. The first performance target is **at least 50 tokens/s** (≤20 ms/token), not a promised gain.

### 2. Pack and reuse recurrent control inputs

**Files:** decoder runtime-parameter collection and binding, `recurrent_cache_fusion.cpp`, native op metadata interfaces if needed.

1. Materialize reset range, remainder offsets, and convolution snapshot windows on the host into persistent, correctly shaped metadata tensors. Feed native ops directly instead of building their metadata with GPU Concats/Reshapes.
2. Share metadata across layers only when graph structure and runtime provenance guarantee equality. Never deduplicate merely because values match in the first batch. Keep per-node descriptors for genuinely different writes.
3. Use reusable USM-host storage for small host-produced controls where supported; measure against one packed device upload. Keep caches in device USM. Avoid moving shape metadata to device-only storage and forcing a readback for shape inference.
4. Reuse bindings when tensor identity, address, type, and shape are unchanged. Rebind after allocation/view/shape changes. Increment the compiled-model cache ABI/revision for any input-interface change.

**Gate:** reduce the current 136 tiny uploads to a small number of packed controls, verify no stale metadata across reset/reorder/rejection, and record input count, upload count, and uninstrumented latency. Binding alone has only about a 0.44 ms observed budget; a larger gain must come from removing device metadata work/synchronization.

### 3. Fuse matrix-state writeback and eliminate state-layout traffic

**Files:** `openvino/op/cpy.cpp`, `openvino/op/gated_delta_net.cpp`, native recurrent-cache/GDN core ops, GPU primitive and kernel implementations.

1. Add a generic native **RecurrentStateUpdate** operation, or extend the current cache operation, to combine the prepared full cache with active final states/snapshots in one kernel. Output shape equals cache shape; runtime offsets and snapshot selections affect indexing, not shape inference. Preserve inactive slots, reset-before-gather semantics, cyclic/duplicate sources, and ordered overlapping-write precedence.
2. Accept GDN's native state layout in the update kernel and transpose while writing GGML cache layout. Fold active-state gather and input-layout conversion together where possible. This avoids standalone full-state transpose intermediates and Slice/Concat/Range writeback chains.
3. If profiling still justifies it, extend GDN to consume the original cache plus active-row/reset metadata and emit active results/snapshots directly in cache-compatible layout. A complete fused gather→GDN→writeback path can avoid the prepared-cache intermediate. Keep old source storage immutable until all potentially aliasing reads finish; arbitrary multi-slot cycles forbid naive in-place writes.
4. Preserve the existing external cache precision and numerical contract. Do not trade throughput for a silent F32→F16 cache change. Compare native-on/off logits and replay behavior after fusion.

**Gate:** remove the identified state writeback shape waits and standalone state transposes; reduce T3 snapshot overhead substantially from **7.38 ms toward ≤2 ms** on 0.8B. This is a target, subject to measurement. Retain all multislot/snapshot regressions and compiled-model export/import coverage.

### 4. Optimize remaining dispatches based on a fresh profile

Only after steps 1–3, reassess kernel submission and dense compute:

- Consider fusing `RecurrentConvCache` with SSM convolution and SiLU, avoiding materialized history-plus-QKV and extra launches. Its current cache kernel is only ~0.073 ms aggregate; prioritize measured dispatch/synchronization savings, not an assumed large bandwidth saving.
- Remove unused `h_nextn`/`result_norm` host Results when the caller does not request them, while retaining the exact MTP/embedding/sampling contract. They are only 8 KiB here, so this is a secondary I/O cleanup.
- Compare remaining FP16 projection/gating kernels with Vulkan. Preserve accuracy and multi-token performance when adjusting fusions or kernel selection.
- If translated shape-only expressions still wait unnecessarily, make a narrowly justified GPU-plugin shape-dependency fix. Do not globally bypass `finish()` or change queue ordering without dependency correctness tests.

**Gate:** aim for **≥55 tokens/s** on 0.8B, within roughly 5% of the current 57.91 t/s Vulkan Release reference. Report gains and counter changes separately for each retained patch.

### 5. Validate generality, then measure the target model

For every functional patch, exercise the existing recurrent-slots and hidden/rollback regressions with active subsets, nonzero-slot resets, copies/branches, inactive-state preservation, unequal rejection lengths, zero requested outputs, and changing B/T on a reused graph. For native-op changes, extend the existing f16/f32, i32/i64, dynamic/static, cycle/overlap, and export/import tests. Statefulness remains out of scope.

After the small-model improvements:

1. Run 0.8B `llama-server -np 4` reset/reuse/concurrent waves and a longer-context `-p 1024 -n 512 -d 0,1024` performance check.
2. Rebuild/reuse compiled caches with the new revision. For functional tests, unset requantization when supported; the previously observed 35B no-requant fusion failure may still require the target requant config.
3. Run the target Qwen3.6-35B-A3B with `q4_asym64_all`, GPU, native softplus, and KV slicing disabled. Repeat `-np 4` server checks without and with two-token MTP.
4. Repeat the established target bench, controlled T1/T3/snapshot matrix, and matched CLI/MTP traces. Report draft, catch-up, verification, acceptance, and complete generation throughput.
5. Compare against the current generic/native target baselines: **23.05 t/s non-MTP tg512**, **22.32 at depth 1024**, and **27.93 t/s MTP on the prior greeting workload**. The user's **≥30 t/s** goal is a target; small-model gains cannot simply be multiplied into the 35B result.

Keep useful patches independently reviewable across llama.cpp and OpenVINO, with measured results and relevant files only. The existing native implementation remains uncommitted; this investigation does not create additional implementation commits.

## Evidence and limitations

All commands, environments, logs, profiles, and the scratch probe are under [`tmp/qwen08-perf-plan-20261016`](../../../../tmp/qwen08-perf-plan-20261016/). Useful entry points:

- `run.py`, per-run `command.json`, and `summary.json`.
- `ov-native`, `ov-native-repeat`, `ov-unfused`, `ov-host-outputs`, `vulkan-release`: plain benchmark evidence.
- `ov-matrix`, `vulkan-matrix`: controlled target calls; `batch-vulkan` is linked to the new Release build.
- `ov-profile`, `ov-profile-unfused`, `vulkan-profile`: instrumented operation profiles.
- `probe.cpp`, `ov-probe/probe-1-io.tsv`, `ov-probe/probe-1-exec.xml`, and `ov-probe/run.log`: actual bindings, runtime graph, OpenCL call/copy counts.
- `ov-sync-trace/run.log`: `primitive_inst.cpp:481:update_shape` records proving the 108 Reshape + 55 Slice waits per inference. Do not benchmark with this verbose tracing enabled.

The probe is investigative tooling, not production code. It counts intercepted NDRange calls and Intel USM memcpy calls, not every possible driver-internal transfer. An initial debug-input attempt failed because that helper calls `Tensor::data()` on a remote tensor; the final probe inspects remote handles safely. Early probe build/instrumentation attempts were corrected; reported probe results come from the final successful run.

No new model-correctness claim is made from throughput tests alone. The prior native implementation's functional coverage is documented in [OPENVINO-RECURRENT-CACHE.md](OPENVINO-RECURRENT-CACHE.md); the planned changes must repeat the relevant coverage. No 35B performance rerun was performed in this investigation-only round.
