# Generic recurrent-cache baseline for Qwen MTP

Date: 2026-10-15. Branch: `qwen-openvino-generic-cache`. Model: Qwen3.6-35B-A3B-Q4_K_M_yamini.gguf.

Removing the single-slot shortcuts changes the matched non-MTP `tg512` baseline from **35.54 to 19.72 tokens/s** (33.62 to 19.10 at depth 1024). Four-slot serving passes both with and without MTP. On the same greeting workload, MTP now measures **23.21 tokens/s versus 19.45 without MTP**, approximately a 19% improvement against the generic path.

The previous non-MTP baseline used single-slot shortcuts that were unavailable to MTP. Consequently, the earlier 36.1 versus 80.7 ms three-token comparison mixed the cost of general cache maintenance with the cost of retaining rollback snapshots. It did not isolate snapshot capture alone. The baseline below removes that confound.

## Changes

Stateless execution now uses the same reset, gather/reorder, and partial-writeback translations for one or multiple slots. Removed the host-side single-slot cache clear, reset/gather identity cases, and full-cache replacement/no-op CPY cases 7/8/9. Native GatedDeltaNet direct output routing, convolution-window Gather, and the TopK improvement remain: these support multiple slots.

Exercising the generic paths also exposed and fixed three correctness problems:

- Reset offsets were divided by elements per row instead of bytes per row, selecting the wrong row for nonzero offsets.
- Views of active recurrent states could retain inactive rows from the combined gather when the active sequence count changed. They now select the runtime active prefix before reshaping.
- An initially empty reset or reorder remainder could disappear from the translated in-place alias chain even when a later batch made it nonempty. Dynamic recurrent writes now remain in that chain, and remainder source/destination offsets are runtime inputs.

The compiled-model cache translation revision is incremented. Existing blobs must be regenerated once with `GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY` unset. The user's `tmp/qwen36_ud_cache` directory was not present in this workspace; measurements use cold model loading and exclude it from token-throughput timing.

## Benchmark method and results

Hardware: Intel Arc B390 iGPU / Core Ultra X7 358H, 62 GiB RAM. Both locally matched runs use Release `build/MtpOVPerf`, the same existing unrelated working-tree changes, llama.cpp starting at `4f92304c2`, and the OpenVINO runtime containing `21cacac33d`. No stateful execution is used. The appendix used `build/ReleaseOV`, whose configured runtime is a different 2026.4.1 toolkit; treat that column as a reference, not the matched control.

Both matched benchmark runs use:

```sh
GGML_OPENVINO_DEVICE=GPU \
GGML_OPENVINO_NATIVE_SOFTPLUS=1 \
GGML_OPENVINO_DISABLE_KV_SLICE=1 \
GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all \
build/MtpOVPerf/bin/llama-bench \
  -m "$HOME/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-Q4_K_M_yamini.gguf" \
  -p 1024 -n 512 -d 0,1024
```

Default five repetitions; values are reported tokens/s with the benchmark's standard deviation. GPU workloads run serially.

| Workload | User appendix | Local before removal | Generic cache path |
| --- | ---: | ---: | ---: |
| pp1024 | 1051.29 +/- 5.21 | 1151.77 +/- 8.85 | 1058.21 +/- 7.29 |
| tg512 | 36.98 +/- 0.10 | 35.54 +/- 0.06 | 19.72 +/- 0.03 |
| pp1024 at depth 1024 | 979.10 +/- 61.42 | 1089.73 +/- 48.93 | 1016.84 +/- 17.64 |
| tg512 at depth 1024 | 34.77 +/- 0.05 | 33.62 +/- 0.05 | 19.10 +/- 0.03 |

### Controlled target calls

One loaded model, identical forced-token stream, context 4096, one slot, two CPU threads, hidden-state extraction enabled. Each case has a 128-token prefix, eight warmup calls, and 32 measured calls. Values are median milliseconds per target call. The previous OpenVINO and Vulkan columns are from the 2026-10-14 experiment; Vulkan was not rerun for this OpenVINO-only cleanup.

| Target call | Previous OV path | Generic OV path | Prior Vulkan reference |
| --- | ---: | ---: | ---: |
| 1 token, no snapshots | 28.31 | 49.41 | 43.21 |
| 3 tokens, no snapshots | 36.10 | 62.55 | 54.29 |
| 1 token, snapshots | 62.01 | 76.49 | 33.54 |
| 3 tokens, snapshots, no rejection | 80.74 | 95.57 | 52.59 |
| 3 tokens, snapshots, reject 2 | 79.37 | 95.67 | 52.72 |

With the generic baseline, moving from one to three tokens without snapshots adds **13.14 ms**; enabling snapshots at three tokens adds **33.02 ms**. Along this comparison, the snapshot-enabled path accounts for about **72%** of the increase from one-token ordinary decode to three-token verification, rather than the earlier approximately 85%. These deltas include the full graph's layout, cache, scheduling, and storage changes; they are not timings of the native GDN kernel alone.

Rejecting two tokens changes the target median by only 0.10 ms here. The host `seq_rm` median is 0.0075 ms; the following target call includes the deferred restore. The large penalty is present even with no rejection. In the final rejection-case call, the full target call was 96.10 ms and OpenVINO `infer()` was 94.40 ms. This places most latency inside OpenVINO execution, which includes scheduling and synchronization as well as kernels.

The old claim that a one-token call returns to roughly 28-30 ms applies to the removed shortcuts. The generic path now measures 49.41 ms without snapshots and 76.49 ms with them.

### End-to-end generation

Same `Hi` prompt, context 4096, one slot, two CPU threads, temperature zero, and a 192-token cap. These runs use the target requantization and the same Release build. Per-call medians exclude positions below 64. Generation rates exclude model loading and prompt processing.

| Measurement | Generic non-MTP | Generic MTP, two proposals |
| --- | ---: | ---: |
| Generation throughput | 19.45 tokens/s | 23.21 tokens/s |
| Generated tokens | 159 | 154 |
| One-token target call, median | 49.85 ms | - |
| Three-token verification, median | - | 95.62 ms |
| One draft token, median | - | 6.09 ms |
| Three-token draft catch-up, median | - | 1.91 ms |
| Accepted proposals | - | 95 / 116 (81.90%) |
| Mean output length per round | - | 2.64 |

MTP is approximately **19.3% faster** than the generic non-MTP path on this greeting. The generated continuations have different lengths, so use the controlled target-call table for causal comparisons of batch width and snapshots. These are one process run per mode with many warmed calls, rather than repeated independent trials.

Compared with the earlier TopK-optimized MTP run, throughput falls from 27.55 to 23.21 tokens/s while output length and acceptance remain 154 and 95/116. Median verification rises from 78.08 to 95.62 ms; draft latency changes only from 5.93 to 6.09 ms. The generic reset/reorder correctness fixes affect snapshot-enabled graphs too, particularly operations that were empty in the first captured batch. The absolute MTP slowdown therefore accompanies removal of the misleadingly fast non-MTP baseline.

A speculative round costs approximately `verification + catch-up + 2 * draft + host work`, divided by `1 + accepted proposals`, not always three output tokens. The measured median compute subtotal is 109.71 ms/round; at 2.64 output tokens this is 41.56 ms/token before remaining host work, close to the reported 43.09 ms/token.

The previous Vulkan greeting reference was 49.83 tokens/s, with 42.56 ms verification, 4.99 ms per draft, and 2.31 ms catch-up. Vulkan was not rerun here. The remaining OpenVINO gap is concentrated in target verification; the controlled results still identify snapshot-enabled cache maintenance as the largest added cost, while ordinary generic cache maintenance is also substantial. Separate kernel profiling is needed to divide that cost among dispatch, shape work, layout conversion, and memory traffic.

## Validation

- Release build and `git diff --check` pass.
- New `test-backend-sampler --test recurrent_slots`: four slots with changing active subsets, nonzero-slot reset, sequence copy/branch, inactive-slot preservation, and unequal rejection lengths. Both zero and two rollback positions are compared with independently replayed single-sequence logits on Qwen3.5-2B Q4_0. Maximum relative logit error: 0.006757 across 40 sequence comparisons.
- Existing hidden-state / zero-output / changing-sequence / rollback sampler check passes.
- Selected GPU operator tests: 266/266 supported cases pass.
- Selected CPU operator tests: 262/262 supported cases pass.
- Full 35B model: 12/12 server requests pass without MTP and 12/12 with two-token MTP, using `-np 4 -c 16384`, ordinary separate KV streams (`kv_unified=false`), temperature zero, and the target requantization. Cases include two concurrent four-slot waves, reset/reuse of selected slots, and cached continuation of an inactive slot. The MTP run generated 64 draft tokens and accepted 43. These short requests include graph compilation and are functional checks, not throughput comparisons.

The optional full-model functional attempt with requantization unset hit an OpenVINO `FullyConnectedHorizontalFusion` failure in the unchanged runtime: projection scale shapes `[8192,1]`, `[512,64]`, and `[512,64]` cannot be concatenated. Full-model validation therefore uses the target `q4_asym64_all` configuration. That runtime fusion issue is outside this cache cleanup.

A stateless forced-static CPU smoke test (Qwen3.5-2B, prefill chunks of 8, 16 generated tokens) matches dynamic CPU generation. Stateful execution and NPU execution were not tested. CPU recurrent snapshots are not supported by the native OpenVINO runtime.

## Generic cache fusion opportunities

`cache_r` and `cache_s` are the convolution and recurrent matrix state of the model's linear-attention blocks; they are separate from the ordinary flash-attention K/V cache.

The proposed graph groups are useful fusion boundaries. Their main benefit would be eliminating intermediate kernels, shape work, and repeated cache copying. The current full-model translator already publishes the final cache root after chaining in-place writes; the existing target IRs have one `cache_r` and one `cache_s` Result per recurrent layer. Older IRs with duplicate cache Results should not be used to count current output transfers.

| Proposed native operation | GGML subgraph from cgraph_ov.txt | Inputs | Outputs |
| --- | --- | --- | --- |
| RecurrentConvCache | 7-14 and 17-27 | old convolution cache; new QKV; reset range; source/destination row maps; active count; valid-token count; snapshot window starts and destination rows | convolution input for SSM_CONV; final updated convolution cache |
| RecurrentStateGather | 28-35 | old matrix-state cache; reset range; source/destination row maps; active count | active matrix states for GatedDeltaNet; reset/reordered cache feeding the later state writeback |

`RecurrentConvCache` must depend on the QKV projection at node 15/16. Its first output replaces node 18, while its second replaces the final cache write. A kernel can read each history/new-token element directly for both outputs without constructing separate reset, gather, concatenation, and snapshot-copy intermediates. The convolution itself can remain outside this first implementation.

`RecurrentStateGather` alone cannot perform final GDN writeback: that data does not exist until GatedDeltaNet runs. A subsequent generic cache-update fusion, or a cache-aware GatedDeltaNet variant, would remove the remaining snapshot layout/scatter chain. Only the final cache becomes a model Result; the prepared-cache output remains internal.

Both operations must preserve arbitrary active subsets, inactive slots, repeated gather indices/sequence copies, nonzero destination rows, reset-before-gather ordering, valid snapshot prefixes when T<K, and rollback-plane selection. Cache capacity, active count, token count, reset range, and write offsets must not be frozen from the first batch. Match topology and layouts rather than fixed node numbers or names; retain the existing translations as fallback.

Arbitrary reorders can contain cycles or duplicate sources. A single kernel cannot safely overwrite a buffer that it is still gathering from unless alias safety is proven. Start with an immutable input and separate output allocation; optimize storage reuse only with explicit read-before-write guarantees. Batch metadata can be shared across layers instead of generating redundant per-layer scalar bindings. A frontend custom op without a native GPU implementation would simply expand back into the same operations and would not demonstrate the intended speedup.

First benchmark the two native operations against their existing subgraphs at 1 and 4 slots, 1/2/3 input tokens, with and without snapshots; then rerun the regression and server cases above. These operations were proposals in the 2026-10-15 baseline commit; the implementation and measurements follow below.

## Artifacts

Local scratch artifacts are in `tmp/mtp-generic-20261015/`: `run.py`, exact command/environment JSON files, before/after benchmark logs, numerical regression logs, server responses, and synchronized target/draft trace logs. The controlled target and CLI trace harnesses are in `tmp/mtp-gap-20261014/`.

## Native implementation (2026-10-16)

`RecurrentStateGather` and `RecurrentConvCache` now have native OpenVINO GPU implementations. They share a GPU primitive and OpenCL kernel, selected by the history attribute (zero for matrix-state gather, positive for convolution history). Each output element has one writer; all cache reads come from an immutable input allocation. The kernel honors the reset range before reading mapped rows, preserves inactive rows, and applies convolution snapshot windows in the original write order.

The frontend recognizes complete combined-gather `build_rs` groups from their topology. It removes the reset/reorder/remainder chain and routes the convolution CONCAT and all its snapshot CPYs to the native op's two outputs. The convolution-cache gather becomes unreachable and is eliminated. Matrix-state output is still followed by the existing GDN state/snapshot writeback; fusing that later chain is a separate opportunity. Runtime row maps, reset ranges, remainder source/destination offsets, QKV dimensions, and snapshot windows remain dynamic. Short batches use the original runtime window starts, including repeated initial-history windows when fewer than K tokens are available.

The metadata interface uses rank-four integer tensors. The six shared inputs are old cache `[1,1,S,F]`, active source rows `[1,1,1,B]`, extra source row map `[1,1,1,N]`, reset begin `[1,1,1,1]`, reset length `[1,1,1,1]`, and extra offsets `[1,1,1,2]` containing destination begin and source-map begin. Using a suffix offset permits an empty remainder without binding a zero-sized GPU input. Convolution adds QKV `[1,B,C,T]` and window pairs `[1,1,K,2]` containing start column and destination row; history H is an attribute and F=C*H. Outputs are active state `[1,1,B,F]` or convolution input `[1,B,C,H+T]`, followed by the full updated cache `[1,1,S,F]`.

Released runtimes lacking these native classes use the existing translators through a CMake symbol check. CPU, forced-static, stateful, and split graphs retain their previous translations. Set `GGML_OPENVINO_DISABLE_RECURRENT_CACHE=1` to compare against the generic decomposition. The disk-cache translation revision is 3 and includes this switch, so old blobs need regeneration.

Validation completed:

- Native GPU exact-reference checks: f16/f32 data, i32/i64 metadata, dynamic and fixed shapes, changing batch/token counts, empty/nonempty remainder, nonzero reset, duplicate/cyclic maps, inactive rows, overlapping and short snapshot windows, cloned operations, and compiled-model export/import.
- Large remote-buffer cyclic reorder with caller-bound input/output sharing, repeated four times.
- Core type/shape validation, including rejection of dynamic rank, incorrect metadata types, incompatible history/channel dimensions, and invalid history.
- Four-slot model reset/copy/rollback: 40 comparisons, maximum relative logit error 0.006757, matching the generic baseline. Hidden-state/zero-output/rollback sampler check passes with no logit mismatches.
- Full 35B `llama-server -np 4`: all 12 requests pass without MTP and all 12 pass with two-token MTP under `q4_asym64_all`; MTP accepted 43 of 64 proposals. Cases include concurrent waves, selected-slot reuse, and cached continuation.
- GPU execution profiles confirm exactly one native convolution-cache op and one native matrix-state gather per recurrent layer. On the small Qwen3.5-2B model, the convolution gather is eliminated. Device profiling here establishes execution and graph structure; its summed durations are not full inference latency.
- CPU fallback generation passes. The GPU disable switch also executes the original decomposition with no native recurrent-cache ops. Stateful and NPU execution were not tested.

Build/test commands and raw results are in `tmp/mtp-native-20261016/`. Native standalone tests use the permanent OpenVINO sources under `src/core/tests/type_prop/recurrent_cache.cpp` and `src/plugins/intel_gpu/tests/functional/single_layer_tests/recurrent_cache.cpp`.

### Matched full-model benchmark

Same target environment, Release `build/MtpOVPerf`, five repetitions and `-p 1024 -n 512 -d 0,1024` as the generic baseline:

| Workload | Generic cache | Native cache | Change |
| --- | ---: | ---: | ---: |
| pp1024 | 1058.21 +/- 7.29 | 1079.26 +/- 7.49 | +2.0% |
| tg512 | 19.72 +/- 0.03 | 23.05 +/- 0.27 | +16.9% |
| pp1024 at depth 1024 | 1016.84 +/- 17.64 | 1031.68 +/- 18.71 | +1.5% |
| tg512 at depth 1024 | 19.10 +/- 0.03 | 22.32 +/- 0.03 | +16.9% |

The native kernels recover part of the generic cache overhead without reintroducing the removed single-slot shortcuts. They do not recover the shortcut baseline's 35.54 tokens/s. Full-cache output writes and the later GDN writeback still remain; changes to those need equally generic slot and rollback semantics.

Controlled forced-token target calls (same prefix 128, eight warmups, 32 measured calls and hidden outputs as the baseline):

| Call | Generic median ms | Native median ms | Latency reduction |
| --- | ---: | ---: | ---: |
| 1 token, no snapshots | 49.41 | 42.11 | 14.8% |
| 3 tokens, no snapshots | 62.55 | 55.43 | 11.4% |
| 1 token, snapshots | 76.49 | 60.86 | 20.4% |
| 3 tokens, snapshots | 95.57 | 79.27 | 17.1% |
| 3 tokens, snapshots, reject 2 | 95.67 | 79.58 | 16.8% |

Actual rejection still adds little: host `seq_rm` median is 0.00746 ms, and rejecting two tokens changes the target median by 0.31 ms. Three-token snapshot-enabled inference is 23.84 ms slower than three-token ordinary inference, down from 33.02 ms before fusion. Increasing ordinary batch width from one to three adds 13.33 ms, essentially unchanged from the earlier 13.14 ms. The speedup therefore primarily removes cache maintenance overhead rather than changing MoE scaling with token count. The final rejection call took 78.55 ms, including 76.91 ms in raw OpenVINO inference.

Against the previously measured Vulkan reference, native OV remains slower on snapshot-enabled calls (79.27 versus 52.59 ms for controlled three-token verification). Remaining opportunities include fusing GDN snapshot transpose/scatter with final cache writeback, reducing repeated full-cache copies while preserving arbitrary maps, and reducing metadata/layout dispatches. The new kernels are one dispatch per cache group, but do not fuse the convolution itself or GDN computation.

End-to-end greeting (`Hi`, temperature zero, one slot, cap 192, same target environment):

| Measurement | Generic | Native |
| --- | ---: | ---: |
| Non-MTP generation | 19.45 tokens/s | 22.95 tokens/s |
| MTP generation, two proposals | 23.21 tokens/s | 27.93 tokens/s |
| Non-MTP one-token target, median | 49.85 ms | 42.39 ms |
| MTP three-token verification, median | 95.62 ms | 77.10 ms |
| MTP draft token, median | 6.09 ms | 6.02 ms |
| MTP three-token catch-up, median | 1.91 ms | 1.79 ms |
| MTP proposals accepted | 95/116 | 95/116 |

The native non-MTP and MTP runs generated 159 and 154 tokens, respectively, matching the generic runs. MTP improves by 20.3% over its generic path and runs 21.7% faster than native non-MTP on this greeting. Per-round compute is approximately `(77.10 + 2*6.02 + 1.79)/2.64 = 34.44 ms/token`, before remaining host work; measured generation is 35.81 ms/token. These CLI values are one process run per mode with warmed call medians, not independent repeated trials. Startup/loading and prompt compilation are excluded from generation throughput.

The previous Vulkan greeting reference remains 49.83 tokens/s with 42.56 ms verification, 4.99 ms draft token, and 2.31 ms catch-up. Native OV reduces its verification deficit from approximately 53 to 35 ms per round; it does not yet close the gap. Acceptance differs across the backend runs, so use the forced-token controlled matrix for causal latency comparisons. Vulkan was not rerun for this implementation.

## Further shape/control/state-update optimization (2026-10-16)

The subsequent [small-model investigation and plan](OPENVINO-QWEN35-08B-PERF-PLAN.md) identified runtime shape-inference synchronization around the native kernels. The [implementation results](OPENVINO-QWEN35-08B-PERF-RESULTS.md) document active-row shapes, reusable USM host controls, native `RecurrentStateUpdate`, folded matrix layouts, current cache revision 6, and updated target-model measurements. The earlier revision and throughput tables above remain historical baselines.

The current target is now `~/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` with the same OpenVINO configuration. Updated tests and performance are in the [performance report](OPENVINO-QWEN35-08B-PERF-RESULTS.md); all yamini results above remain historical.


## Native operator contracts (2026-10-20 review)

The historical graph-node ranges above explain the first prototype. The current native API has three internal OpenVINO operations. They fuse cache data movement; GDN attention and SSM convolution arithmetic remain separate operations. Each cache is a full `[1,1,S,F]` tensor, where `S` is the allocated slot count and `F` is the flattened per-slot state size. Cache/QKV/state inputs support F16/F32; each metadata input supports I32/I64. All input ranks are four, while dimensions and metadata values may change between requests.

| Operation | Ordered input ports (zero-based) | Output ports |
| --- | --- | --- |
| `RecurrentStateGather` | `cache`, `active_rows`, `extra_rows`, `reset_begin`, `reset_length`, `extra_offsets` | 0: active states `[1,1,B,F]`; 1: full reset/reordered cache |
| `RecurrentConvCache` | the same six inputs, then `qkv [1,B,C,T]`, `windows [1,1,K,2]`; attribute `history=H>0`, `F=C*H` | 0: convolution input `[1,B,C,H+T]`; 1: full cache after window writes |
| `RecurrentStateUpdate` | `prepared_cache`, `states [1,1,N,F]`, `control [1,1,1,4]` | 0: full updated cache |

Gather logically clears the source interval `[reset_begin, reset_begin+reset_length)` before reading either row map. Active rows are selected from that cleared source, before reorder. `extra_offsets=(destination_begin, source_begin)` selects the suffix `extra_rows[source_begin:]` and copies it to consecutive destination rows. The input allocation stays immutable, which preserves cyclic maps such as `[1,0]` and duplicate sources. Rows outside the reset/reorder ranges are preserved. The active output feeds GDN; the full prepared-cache output feeds StateUpdate.

Convolution-cache output 0 concatenates gathered per-channel history with projected QKV. Each window `(start_column,destination_row)` copies `H` consecutive columns into `B` cache rows. Valid starts lie in `[0,T]`; the caller supplies valid-token windows for short or padded batches. Window writes occur after reset/reorder and later overlapping windows win. Output 0 uses QKV precision; output 1 uses cache precision. A mixed-precision window write rounds through output-0 precision before converting to cache precision, matching a materialized convolution-input slice.

StateUpdate control is `(destination_begin, sequence_count, snapshot_stride, source_begin)`. Selected source row `source_begin+r` writes destination row `destination_begin + (r/sequence_count)*snapshot_stride + r%sequence_count`, using integer division. Later source rows win for overlapping planes, including `snapshot_stride=0`; `source_begin=N` is an empty update. For example, two sequences with stride 4 and destination begin 1 write six source rows to `[1,2,5,6,9,10]`.

Positive `key_size`/`value_size` attributes fold the matrix-layout conversion into gather and update. Gather converts `[head,value,key]` cache matrices to native GDN `[head,key,value]`; StateUpdate converts back. Both attributes default to zero, which disables transpose. They must either both be zero or both positive, with their product dividing `F`.

GDN has two outputs. Output 0 is attention. Output 1 is `[B,H_v,D_k,D_v]` for `snapshot_count=1`, or `[K,B,H_v,D_k,D_v]` for `K>1`. Snapshot entry `i` is the state after zero-based token `T-1-i`; only `min(K,T)` entries are valid. Entry 0 already contains the final state, so no additional final-state output is needed. Snapshot mode leaves the input matrix state unchanged.

All runtime indices, reset ranges, windows, and destination rows must be in bounds. The OpenCL kernels use distinct source/output allocations; the GPU runtime may stage a caller's shared remote input/output tensor to preserve immutable reads. This supports multi-slot reset/reorder and speculative rollback without single-slot assumptions.

The OpenVINO operation specifications are in `docs/articles_en/documentation/openvino-ir-format/operation-sets/operation-specs/internal/`: `recurrent-state-gather.rst`, `recurrent-conv-cache.rst`, `recurrent-state-update.rst`, and `gated-delta-net.rst`. The corresponding C++ developer headers also document every port, attribute, and precedence rule.
