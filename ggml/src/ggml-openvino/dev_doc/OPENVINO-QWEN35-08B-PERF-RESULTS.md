# Qwen recurrent-cache optimization results

Date: 2026-10-16. Implementation was measured from the local changes in llama.cpp (`qwen-openvino-generic-cache`, HEAD `73f61e1cb`) and `/home/zijun/openvino` (`qwen-mtp-openvino-checkpoint`, HEAD `21cacac33d`). The matching native OpenVINO changes are committed as `9e08e52ad5` (`intel_gpu: fuse generic recurrent cache maintenance`). Builds use Release `build/MtpOVPerf` and `/home/zijun/openvino-build-mtp`. Unrelated working-tree changes were preserved. Results below exclude load/requantization/graph compilation from generation throughput. No stateful execution is involved.


From this retest onward, the target is `~/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`. GPU, `q4_asym64_all`, native softplus, and disabled KV slicing remain unchanged. The yamini-model tables below retain their original model identity; new measurements are reported separately.
## Implemented changes

1. Added an active recurrent-row-map prefix input with dynamic length. Active sequence counts now come from its runtime shape instead of an uploaded scalar value. This preserves changing slot counts and removes all 108 recurrent Reshape shape-inference waits in the 0.8B graph.
2. Reuse per-request runtime control tensors in USM host allocations. This removes almost all tiny scalar uploads while keeping shape metadata CPU-readable. Ordinary cache buffers remain in device USM, as before.
3. Materialize convolution windows and remainder offsets as packed host controls. Metadata varies per actual graph write; it is not deduplicated by coincidentally equal first-batch values.
4. Added native `RecurrentStateUpdate`: a full-cache output with runtime active/snapshot offsets and ordered overlap handling. It replaces Slice/Concat/Range state writeback, removing the remaining 36 ordinary-decode shape-inference waits.
5. StateUpdate converts native GDN state/snapshot matrices to GGML cache layout while writing. RecurrentStateGather converts gathered matrices into native GDN layout while reading. Standalone recurrent input/output state transposes are eliminated on the structurally matched path. Original source cache reads remain immutable, including arbitrary row-map cycles and duplicates.
6. Added native-op capability probing and updated the compiled-model cache revision. Dynamic CPU, static, and released-runtime paths retain their primitive translations. Existing disk-cache blobs need regeneration.

## 0.8B F16 results

The same small-model command and matched Vulkan Release reference as the [investigation](OPENVINO-QWEN35-08B-PERF-PLAN.md) are used: `llama-bench -p 0 -n 128 -r 3`, GPU, native softplus, KV slicing disabled, requant unset. Changes were measured incrementally; small differences between adjacent runs are subject to run order and machine variability.

| Stage | Generation tokens/s |
| --- | ---: |
| Initial native cache implementation | 43.76 ± 0.10 |
| Active count from row-map shape | 46.82 ± 0.05 |
| Reusable USM host scalar controls | 50.23 ± 0.07 |
| Native full-cache state update | 51.15 ± 0.15 |
| Output state transpose folded into update | 52.60 ± 0.07 |
| Packed window/remainder metadata | 52.50 ± 0.07 |
| Input state transpose folded into gather | 53.75 ± 0.08 |
| Final independent optimized repeat | **56.21 ± 0.06** |
| Matched Vulkan Release reference | 57.91 ± 0.17 |

The final independent repeat reaches **56.21 ± 0.06 tokens/s**, **28.4%** above the initial native path and **2.9%** below Vulkan Release. An earlier final-build repeat measured 55.51 ± 0.17 tokens/s. Both meet the ≥55 tokens/s small-model target. The independent repeat ran after the target MTP process exited, with no overlapping GPU workload.

A longer three-repeat check (`-p 1024 -n 512 -d 0,1024`) records:

| 0.8B workload | Optimized tokens/s |
| --- | ---: |
| pp1024 | 6781.63 ± 418.28 |
| tg512 | **55.46 ± 0.07** |
| pp1024 at depth 1024 | 5330.30 ± 1286.93 |
| tg512 at depth 1024 | **53.04 ± 0.18** |

Prefill has high variance in this short run; the decode measurements are more stable. No matched Vulkan longer-context baseline was rerun.

The intercepted warmed ordinary-decode work changes as follows:

| Counter | Initial native | Optimized |
| --- | ---: | ---: |
| Model Parameters | 184 | 130 |
| Model Results | 51 | 51 |
| Remote USM cache inputs/outputs | 48 / 48 | 48 / 48 |
| Tiny host→device copies | 136, 1588 bytes | 8, 560 bytes |
| Host output copies | 3, 1001472 bytes | 3, 1001472 bytes |
| `clFinish` calls | 164 | 1 |
| Kernel submissions | 789 | 606 |
| Profile entries marked EXECUTED | 1180 | 945 |

The sole remaining finish is inference completion. The prior 163 queue drains were 108 Reshapes and 55 Slices; deriving shapes and native writeback removes them instead of bypassing synchronization. Profiling-level-2 device times are diagnostic only: the optimized trace records 16.982 ms total, of which 14.835 ms (87.4%) is FullyConnected. Profiling changes scheduling and is not the throughput baseline.

The existing controlled target-call harness uses a 128-token prefix, 8 warmups, 16 measured calls, hidden states enabled, and all rows requesting logits:

| Case | Initial OV ms/call | Optimized OV ms/call | Vulkan reference ms/call |
| --- | ---: | ---: | ---: |
| T1, no snapshots | 23.49 | 19.15 | 17.07 |
| T3, no snapshots | 24.44 | 19.93 | 18.40 |
| T1, rollback storage enabled | 29.84 | 22.28 | 17.85 |
| T3, rollback storage enabled | 31.82 | 24.46 | 19.32 |
| T3, reject two | 32.01 | 24.59 | 18.59 |

Snapshot storage still adds about 4.53 ms to the optimized T3 call, compared with 7.38 ms initially. The ≤2 ms snapshot-overhead target is not met. Rejection itself remains small. Single-token throughput and snapshot verification are different performance cases.

## Current target: Qwen3.6 UD-Q4_K_M retest

Model: `~/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` (22,663,387,424 bytes; 21.10 GiB). Same architecture metadata and one `nextn_predict_layers` MTP layer as the previous GGUF. The build and runtime are unchanged from the optimized measurements above. GPU workloads run serially; compiled-model disk caching and stateful execution are unset.

```sh
GGML_OPENVINO_DEVICE=GPU \
GGML_OPENVINO_NATIVE_SOFTPLUS=1 \
GGML_OPENVINO_DISABLE_KV_SLICE=1 \
GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all \
build/MtpOVPerf/bin/llama-bench \
  -m "$HOME/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf" \
  -p 1024 -n 512 -d 0,1024
```

Five repetitions; mean ± benchmark standard deviation:

| Workload | Previous optimized yamini tokens/s | Current UD tokens/s |
| --- | ---: | ---: |
| pp1024 | 1125.19 ± 8.25 | **1127.99 ± 7.25** |
| tg512 | 29.91 ± 0.47 | **27.41 ± 0.07** |
| pp1024 at depth 1024 | 1074.27 ± 33.48 | **1081.64 ± 19.96** |
| tg512 at depth 1024 | 28.47 ± 0.44 | **26.19 ± 0.04** |

UD decode is 8.4% lower at depth zero and 8.0% lower at depth 1024. This compares different GGUF quantizations under the same optimized backend; it is not a before/after optimization measurement on UD. No pre-optimization or Vulkan UD baseline has been measured.

The quantization mix differs substantially: non-expert/non-MTP weights change from 95 Q6_K, 53 Q4_K, 94 Q8_0 and 10 Q5_K tensors to 1 Q6_K and 251 Q8_0 tensors. Those weights grow from 1.87 GB to 2.56 GB. Trunk MoE expert weights change from 20 Q6_K/100 Q4_K to 3 Q6_K/80 Q4_K/37 Q5_K. The full MTP block changes from 11 Q8_0/7 F32/2 BF16 (898 MB) to 8 Q8_0/7 F32/2 BF16/1 Q5_K/2 Q4_K (529 MB): its three expert tensors changed from Q8_0 to K-quants, while its non-expert tensor types remain the same. The same requantization setting does not make all weights identical: K-quants are processed by that setting while existing Q8_0 tensors retain their separate translation. The denser non-expert weights are a plausible contributor to slower decode, but per-op profiling is needed to attribute the gap quantitatively.

The forced-token matrix uses a 128-token prefix, eight warmups, 32 measured calls, hidden output enabled, and logits on all input tokens. Median milliseconds:

| Target call | Optimized yamini ms | Current UD ms |
| --- | ---: | ---: |
| T1, no snapshots | 29.64 | **35.19** |
| T3, no snapshots | 38.75 | **44.10** |
| T1, snapshots | 37.51 | **43.42** |
| T3, snapshots | 53.95 | **58.83** |
| T3, reject two | 53.90 | **59.42** |

UD ordinary T1→T3 costs **8.91 ms**, and snapshot storage adds **14.73 ms** to T3. The latter is close to yamini's 15.20 ms: the extra UD cost is already present in ordinary decode and does not look like a new snapshot-path regression. Rejection raises the T3 median by 0.58 ms in this run; host sequence removal is 0.0072 ms. These differences include full inference scheduling, layout and compute, not only MoE or cache kernels.

The `Hi` greeting runs use context 4096, one slot, two CPU threads, temperature zero and a 192-token cap. Warmed medians exclude positions below 64. These are one process run per mode, not independent repeated trials.

| Measurement | Optimized yamini | Current UD |
| --- | ---: | ---: |
| Non-MTP generation tokens/s | 32.23 | **27.15** |
| MTP generation tokens/s | 38.74 | **35.93** |
| Non-MTP target T1 median ms | 29.76 | **35.88** |
| MTP verification T3 median ms | 51.55 | **57.74** |
| Draft token median ms | 6.010 | **5.980** |
| Draft catch-up T3 median ms | 1.808 | **1.790** |
| Accepted proposals | 95/116 (81.90%) | **114/136 (83.82%)** |
| Mean output tokens/round | 2.64 | **2.68** |

MTP is **32.3% faster** than non-MTP on UD. Both UD continuations have 182 tokens; the yamini continuations had 162 without MTP and 154 with MTP, so cross-model generation rates do not represent an identical-output trial. The changed MTP expert quantization does not show a material draft-latency change in this run. Verification rises by about 6.2 ms, consistent with the additional UD latency observed in ordinary target inference.

The median round compute subtotal is approximately `57.74 + 2*5.980 + 1.790 = 71.49 ms`. At the measured mean 2.68 output tokens/round this is **26.68 ms/token**, before remaining host work, close to **27.83 ms/token** reported end-to-end. Cold prompt processing is 3.57 s with MTP and 0.40 s without it; graph compilation/startup work is excluded from generation throughput.

The UD non-MTP `llama-server -np 4 -c 16384` check passes **12/12 requests**, including concurrent four-slot waves, selected-slot reset/reuse, and cached continuation. The UD MTP server also passes **12/12 requests** with two proposals, accepting **42/64 draft tokens**. These short concurrent requests include compilation and test functionality, not warmed serving throughput. No stateful tests were run. A detached clean source checkout containing only the proposed llama.cpp commit files builds `llama-bench`, `llama-cli`, `llama-server`, and `test-backend-sampler` successfully. Its 0.8B slot/rollback and hidden-row regressions pass, and all seven native tests pass again. This confirms the changes build without the unrelated dirty files in the main workspace. Exact commands, metadata/type counts, logs and responses are in `tmp/qwen36-ud-20261016/`.

## Historical target 35B results (yamini)

Model: Qwen3.6-35B-A3B-Q4_K_M_yamini.gguf. Same Release build and runtime, `q4_asym64_all`, GPU, native softplus, KV slicing disabled. Baseline is the previous native implementation, rather than the removed single-slot shortcut path. Five benchmark repetitions: `-p 1024 -n 512 -d 0,1024`.

| Workload | Previous native tokens/s | Optimized tokens/s |
| --- | ---: | ---: |
| pp1024 | 1079.26 ± 7.49 | 1125.19 ± 8.25 |
| tg512 | 23.05 ± 0.27 | 29.91 ± 0.47 |
| pp1024 at depth 1024 | 1031.68 ± 18.71 | 1074.27 ± 33.48 |
| tg512 at depth 1024 | 22.32 ± 0.03 | 28.47 ± 0.44 |

Generation improves **29.8%** at depth zero and **27.6%** at depth 1024. The 30 tokens/s target is approached at depth zero; the five-repeat mean remains below it. These are generic stateless cache paths.

Controlled target calls use the same forced-token stream, 128-token prefix, eight warmups, and 32 measured calls. Values below are median milliseconds; hidden states are enabled and all token rows request logits. The Vulkan column is the prior target-model reference, not a new Vulkan rerun.

| Target call | Previous native OV ms | Optimized OV ms | Prior Vulkan ms |
| --- | ---: | ---: | ---: |
| T1, no snapshots | 42.11 | 29.64 | 43.21 |
| T3, no snapshots | 55.43 | 38.75 | 54.29 |
| T1, snapshots | 60.86 | 37.51 | 33.54 |
| T3, snapshots | 79.27 | 53.95 | 52.59 |
| T3, reject two | 79.58 | 53.90 | 52.72 |

Three-token verification falls by **31.9%**, from 79.27 to 53.95 ms. The snapshot-enabled T3 penalty falls from 23.84 to **15.20 ms** relative to ordinary T3. Increasing ordinary token width from one to three costs 9.11 ms; rejection adds no measurable penalty in this short comparison, and host sequence removal takes about 0.0072 ms. Restoring state is included in the following target inference. These measurements distinguish snapshot preparation/storage from actual rollback and from multi-token computation; they do not isolate individual kernels.

Optimized controlled snapshot-enabled T3 is within 2.6% of the previous Vulkan reference, while the ordinary T1 path reaches approximately 30 ms. The greeting workload uses `Hi`, context 4096, one slot, two CPU threads, temperature zero, a 192-token cap, and two draft proposals for MTP. Warmed call medians exclude positions below 64. These are one process run per mode, with many warmed calls, rather than repeated independent trials.

| Measurement | Previous native OV | Optimized OV | Prior Vulkan reference |
| --- | ---: | ---: | ---: |
| Non-MTP generation tokens/s | 22.95 | **32.23** | — |
| MTP generation tokens/s | 27.93 | **38.74** | 49.83 |
| Non-MTP target T1 median ms | 42.39 | 29.76 | — |
| MTP verification T3 median ms | 77.10 | **51.55** | 42.56 |
| Draft token median ms | 6.015 | 6.010 | 4.99 |
| Draft catch-up T3 median ms | 1.799 | 1.808 | 2.31 |
| Accepted proposals | 95/116 | **95/116** | differing acceptance |

MTP improves **38.7%** over the previous native path and runs **20.2%** faster than optimized non-MTP on this greeting. The new MTP continuation still has 154 tokens and identical acceptance, so the improvement is not explained by increased acceptance. Non-MTP produces 162 tokens versus 159 previously; use the forced-token table for causal comparisons.

A round's median compute subtotal is approximately `51.55 + 2*6.010 + 1.808 = 65.38 ms`. Divide by the mean 2.64 output tokens, not always three: this gives **24.76 ms/token** before remaining host work, close to the measured **25.81 ms/token**. Draft generation/catch-up are essentially unchanged; the throughput gain comes from target verification.

MTP remains **22.3% below** the prior 49.83 tokens/s Vulkan greeting reference. Its 51.55 ms verification is about 9 ms slower than the previous Vulkan greeting, and two draft tokens add about 2 ms more than that reference. The close controlled T3 results and larger greeting gap are different context/output/acceptance workloads; they should not be mixed into one exact overhead estimate. Vulkan's target-model reference was not rerun in this implementation round. Cold prompt time is 3.54 s for MTP versus 0.28 s without it, including first-use graph work; these values are separate from warmed generation.

## Original optimization validation and artifacts (0.8B and yamini)

Native GPU tests cover dynamic active counts/token widths, f16/f32, i32/i64, reset, duplicate/cyclic maps, preserved inactive rows, overlapping snapshot destinations, native matrix-layout gather/update, cloning, and compiled export/import. Model regressions replay each active sequence independently to check changing slot subsets, nonzero reset, copied sequences, inactive-state preservation and unequal rejection. Hidden-output tests cover changing selected rows and zero requested outputs. A CPU smoke run passes on 0.8B.

The final native/type-propagation suite passes **7/7 tests**. The small-model replay regression passes **40/40 sequence comparisons**, with maximum relative logit error **0.002724** against independently replayed sequences. Hidden-state/selected-row/zero-output rollback tests pass with no sampled-token mismatches. The 0.8B `llama-server -np 4 -c 16384` run passes **12/12 requests**, covering concurrent waves, selected-slot reset/reuse, and cached continuation. The 35B non-MTP `llama-server -np 4 -c 16384` run also passes **12/12 requests** under `q4_asym64_all`. The 35B MTP server also passes **12/12 requests** with two proposals, accepting **43/64 draft tokens**. These short server requests include cold graph work and validate functionality; their timings are not warmed throughput baselines. Qwen3.5-0.8B has no MTP tensors, so MTP server validation uses the 35B target. Stateful testing remains outside this work's scope.

A native-on/off crosscheck repeats the recurrent-slot regression and captures every compared logit vector: **80/80 records match exactly**, with maximum relative error 0 and no argmax mismatches. This checks the new native cache/state-layout translations against the retained generic decomposition.

The full 0.8B compiled-model cache exports and reloads successfully with `GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY=1`; GPU cold/export, GPU import, and CPU fallback produce the same eight-token greedy continuation. Native export/import tests additionally cover dynamic metadata and matrix attributes. Release rebuild and `git diff --check` pass for both repositories. CPU fallback is smoke-tested; static, stateful, NPU, and a released runtime without the new native ops were not tested in this implementation round.

Commands, environments, profiles, model responses, and intermediate/final logs are in `tmp/qwen08-opt-20261016/`. The original baseline evidence is in `tmp/qwen08-perf-plan-20261016/`; the prior 35B native baseline is in `tmp/mtp-native-20261016/`. Early StateUpdate execution attempts exposed a missing GPU factory registration; it was corrected before the successful native/model tests and reported benchmarks.

## Remaining opportunities

- Snapshot-enabled verification still has additional state traffic and graph work. A complete native gather→GDN→cache-update operation could eliminate prepared-cache and snapshot intermediates, but requires careful multi-slot alias and overlap semantics.
- Further cache kernel tuning has a limited budget on ordinary 0.8B decode; dense matmul dominates the current profile. Remaining kernel/reorder fusion should be driven by measured latency rather than node count alone.
- Logits still cross to host for sampling, and hidden/norm outputs remain per the caller's output contract. Removing them requires caller-aware output selection; their presence is not a full-cache host-copy problem.
- Vulkan's 57.91 t/s remains ahead; the final 56.21 t/s small-model repeat meets the ≥55 t/s target. Target-model MTP also includes draft/catch-up and acceptance effects that cannot be inferred from the 0.8B target-call matrix.
