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

First benchmark the two native operations against their existing subgraphs at 1 and 4 slots, 1/2/3 input tokens, with and without snapshots; then rerun the regression and server cases above. These are proposed operations, not implemented or measured speedups in this commit.

## Artifacts

Local scratch artifacts are in `tmp/mtp-generic-20261015/`: `run.py`, exact command/environment JSON files, before/after benchmark logs, numerical regression logs, server responses, and synchronized target/draft trace logs. The controlled target and CLI trace harnesses are in `tmp/mtp-gap-20261014/`.
