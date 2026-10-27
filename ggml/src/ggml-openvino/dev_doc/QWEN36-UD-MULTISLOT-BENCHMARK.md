# Qwen3.6 UD multi-slot benchmark (2026-10-23)

## In-place KV and recurrent convolution rerun (2026-10-27)

This is a separate campaign; earlier tables use the previous translation and
`-b 2048 -ub 512`. The current code merges `origin/dev_backend_openvino`
`0974aef91`, removes recurrent rotation, and compares fused convolution against
history/QKV output plus ordinary GroupConvolution. No new server or Vulkan
measurements are part of this rerun.

- llama.cpp implementation: `9ca7965d6` (`qwen-openvino-inplace-perf`, isolated
  worktree `/home/zijun/llama.cpp-inplace-perf`). Merge commit: `26870e70d`;
  recurrent implementation: `bdaf4818f`.
- OpenVINO: `397f7a2c71` (`qwen-mtp-inplace-primary`). Release GPU build at
  `/home/zijun/openvino-build-mtp`.
- Timed binaries include the cache-only fixes subsequently committed as llama.cpp
  `036829535` and OpenVINO `40d69210bf`. The llama.cpp build identifier stayed
  `9ca7965d6` during the measurements. Obsolete rotation-setting cleanup is
  `2188bc23f`; it does not change the graph or cache fingerprint used here.
- Same target UD GGUF, GPU, native softplus, `q4_asym64_all`.
  `GGML_OPENVINO_ENABLE_KV_4D=1`; KV slicing is not disabled.
- `-b 8192 -ub 8192`, 1024/8192/32768 prompt tokens per sequence, 512 decode
  tokens per sequence. One-slot context is 131072; four-slot contexts are 524288
  total. Split allocates 131072 per slot; unified shares the total capacity.
- In-place recurrent caches are primary in both variants. Fused convolution is
  the default; `GGML_OPENVINO_SEPARATE_RECURRENT_CONV=1` selects the separate
  variant. Both use the same reset, reorder and ordered writeback semantics.
- Full-capacity 4D attention KV is bound where the upstream eligibility check
  permits one physical stream (one-slot and unified). The GPU StatelessKV fusion
  can expose the live prefix to attention. Explicit split four-slot KV is still
  outside that upstream eligibility check and retains ordinary full bindings.
- All GPU workloads are serial. Cache population and functional checks precede
  the 30-minute timed matrix. Timed runs use compiled-model cache-only mode.
  Each batched timing row is one pass; batched-bench does not test MTP acceptance.
  The upstream merge, cache implementation and batch sizes all changed from the
  earlier campaign. These results do not isolate the effect of any one change.

Reproducible runner: [qwen36_ud_inplace_batched.py](benchmarks/qwen36_ud_inplace_batched.py).
Raw commands, outputs, memory samples and status:
`tmp/inplace-kv-retest-20261027/target/` in the original llama.cpp workspace.
The six compiled blobs total approximately 108 GiB on disk. They are shared
across prompt lengths; convolution mode and KV layout use distinct signatures.

```sh
BINARY=/home/zijun/llama.cpp-inplace-perf/build/MtpOVPerf/bin/llama-batched-bench
MODEL="$HOME/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
OUT=/home/zijun/llama.cpp/tmp/inplace-kv-retest-20261027/target
RUNNER=ggml/src/ggml-openvino/dev_doc/benchmarks/qwen36_ud_inplace_batched.py
python3 "$RUNNER" --binary "$BINARY" --model "$MODEL" --output "$OUT" --phase prime
python3 "$RUNNER" --binary "$BINARY" --model "$MODEL" --output "$OUT" --phase timed --minutes 30
python3 ggml/src/ggml-openvino/dev_doc/benchmarks/qwen36_ud_inplace_summary.py \
  --output "$OUT" --report ggml/src/ggml-openvino/dev_doc/QWEN36-UD-MULTISLOT-BENCHMARK.md --finished
```

The equivalent raw commands for either variant are below. Set the separate-conv
flag only for the separate variant; unset it for the fused default. Set cache-only
only after population using the same compilation settings.

```sh
export GGML_OPENVINO_DEVICE=GPU
export GGML_OPENVINO_NATIVE_SOFTPLUS=1
export GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all
export GGML_OPENVINO_ENABLE_KV_4D=1
export GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR="$OUT/model-cache"
export GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY=1
"$BINARY" -m "$MODEL" -c 131072 -npp 1024,8192,32768 -ntg 512 -npl 1 -b 8192 -ub 8192
"$BINARY" -m "$MODEL" -c 524288 -npp 1024,8192,32768 -ntg 512 -npl 1,4 -b 8192 -ub 8192
"$BINARY" -m "$MODEL" -c 524288 -npp 1024,8192,32768 -ntg 512 -npl 1,4 --kv-unified -b 8192 -ub 8192
```

### Validation

All 13 native shape, attribute and GPU test cases pass, including history-only
convolution, cyclic/duplicate reorders, overlapping windows/snapshots, mixed
precision, dynamic shapes, host/device USM, alias/distinct output bindings,
exclusive Reshapes, shared readers, unrelated overlaps and export/import.
Convolution uses an independent host reference; GDN uses ordinary GDN with
independent host cache mapping. Both llama.cpp convolution variants pass the
0.8B four-slot reset/copy/checkpoint/rollback test (`rollback=0,2`).

These checks also caught a post-GDN reshape that kept sequence and token axes
separate at the projection/residual boundary; it now flattens back into the
shared token stream. Full UD preparation exposed matching empty token axes in
DIV during no-logits MoE warmup; their repeat factor now remains one. The 0.8B
split four-slot 8K prompt smoke test with `b=ub=8192` completed (peak process RSS
12.77 GiB); its short 8-token decode is a functional check, not a target result.

### Cache-only and runtime binding checks

Cache preparation itself passed, but the first cache-only validation exposed two
compatibility bugs. The llama.cpp imported-output check compared 4D KV Results
against their old flat GGML shape; it now validates the same head-split layout
used for input/output binding. In the GPU plugin, SetOutput tried to bind a dynamic
Result's arguments while its producer still had the empty allocation from the
previous no-logits batch. Dynamic argument binding is deferred to inference,
after shape inference and allocation. These fixes preserve the compiled graph
format; the six prepared blobs are reused.

Both new GPU output-binding regression tests pass with directly compiled and
exported/imported models. They cover empty-to-nonempty producer allocations and
a new output destination with unchanged input storage and shape. The 13 native
shape, attribute and GPU test cases also pass after the runtime fix.

The one-slot and unified runtime dumps each contain 20 `stateless_kv` primitives,
60 `recurrent_cache_update` primitives and 10 attention primitives. All 80 cache
Parameter/Result pairs bind the same USM pointers. Short cache-only runs pass
no-logits warmup, prefill and decode, including four active unified sequences.
The pointer check establishes caller bindings; the OpenCL trace also checks
explicit copy calls, and does not measure traffic inside a kernel.

### Convolution kernel control

This isolated control uses 8192 channels, history 3, four cache rows, one active
sequence, FP16 data/weights and an FP32 cache. Cache input and output share device
USM. Eight warmups precede 20 measured calls per variant; profiling is enabled,
so these are operator controls, not predicted full-model token rates.

| Input tokens | Fused request time (ms) | Separate request time (ms) |
|---:|---:|---:|
| 1 | 0.068 | 0.158 |
| 1024 | 2.179 | 3.243 |
| 8192 | 18.862 | 26.746 |

At 8192 tokens, the ordinary convolution itself takes 2.133 ms in the last
profiled call, against 16.269 ms for the fused cache/convolution primitive.
However, the separate path also needs 3.104 ms to materialize history/QKV,
2.461 ms input layout conversion, 2.419 ms output reorder and 16.364 ms for the
final transpose. The full separate chain is slower in this control. Full-model
consumers may allow different layout optimizations, so the model matrix remains
the primary comparison. Improving the fused convolution's output access and
eliminating separate-convolution transposes are concrete kernel/layout targets.

### Native GenAI prefill scheduling

Native GenAI supports chunked prefill. In local source `25899f70`,
[SchedulerConfig](https://github.com/openvinotoolkit/openvino.genai/blob/25899f70/src/cpp/include/openvino/genai/scheduler_config.hpp)
defaults to `max_num_batched_tokens=256` and `dynamic_split_fuse=true`. Its
[latency-oriented client configuration](https://github.com/openvinotoolkit/openvino.genai/blob/25899f70/src/cpp/src/utils.cpp)
sets the token budget to `size_t::max`, so that client path does not impose this
small batch limit. With chunking disabled, the scheduler requires a full prompt
to fit the budget. These modes use different scheduling choices; native GenAI
performance alone does not prove the best llama.cpp microbatch size. This rerun
uses an 8192-token budget and equal batch/microbatch sizes, so 32K prompts still
span several graph executions.

<!-- inplace-results:start -->
Finished: 23/30 timing rows retained. A row is retained only after its prefill and all 512 decode steps finish.

Decode throughput below excludes prefill. Values are aggregate tokens/s, using `speed_tg = pl * tg / t_tg`. In four-slot configurations, `A / B` means one active sequence / four active sequences; divide B by four for the simultaneous per-sequence rate.

| OV convolution / KV configuration | 1K prompt | 8K prompt | 32K prompt |
|---|---:|---:|---:|
| Fused conv, `-npl 1` | 38.82 | 34.31 | 24.66 |
| Fused conv, `-npl 1,4` split | 4.56 / 27.36 | 5.17 / 22.47 | 4.47 / Skipped |
| Fused conv, `-npl 1,4 --kv-unified` | 38.40 / 93.43 | 33.98 / 71.90 | Skipped / Skipped |
| Separate conv, `-npl 1` | 38.22 | 33.90 | 24.44 |
| Separate conv, `-npl 1,4` split | 4.52 / 27.53 | 5.13 / 23.07 | Skipped / Skipped |
| Separate conv, `-npl 1,4 --kv-unified` | 37.92 / 90.75 | 33.40 / 71.07 | Skipped / Skipped |

Prefill duration in seconds (not streaming TTFT). `t_pp` ends after all active prompts finish and synchronization completes. It is a prefill-latency proxy, and includes first-use runtime shape/kernel preparation; it does not include process startup or loading compiled blobs.

| OV convolution / KV configuration | 1K prompt | 8K prompt | 32K prompt |
|---|---:|---:|---:|
| Fused conv, `-npl 1` | 1.052 | 5.596 | 24.133 |
| Fused conv, `-npl 1,4` split | 2.768 / 4.113 | 12.157 / 88.200 | 37.953 / Skipped |
| Fused conv, `-npl 1,4 --kv-unified` | 1.069 / 2.421 | 6.290 / 29.394 | Skipped / Skipped |
| Separate conv, `-npl 1` | 1.096 | 5.457 | 30.124 |
| Separate conv, `-npl 1,4` split | 3.344 / 4.388 | 8.760 / 27.930 | Skipped / Skipped |
| Separate conv, `-npl 1,4 --kv-unified` | 1.336 / 2.839 | 7.074 / 24.270 | Skipped / Skipped |

Aggregate prefill throughput in tokens/s. Prompt tokens are independently evaluated for each sequence; no shared-prompt optimization is enabled.

| OV convolution / KV configuration | 1K prompt | 8K prompt | 32K prompt |
|---|---:|---:|---:|
| Fused conv, `-npl 1` | 973.71 | 1464.03 | 1357.78 |
| Fused conv, `-npl 1,4` split | 370.00 / 995.92 | 673.87 / 371.52 | 863.39 / Skipped |
| Fused conv, `-npl 1,4 --kv-unified` | 957.77 / 1691.56 | 1302.47 / 1114.77 | Skipped / Skipped |
| Separate conv, `-npl 1` | 934.72 | 1501.26 | 1087.77 |
| Separate conv, `-npl 1,4` split | 306.26 / 933.49 | 935.13 / 1173.20 | Skipped / Skipped |
| Separate conv, `-npl 1,4 --kv-unified` | 766.60 / 1442.88 | 1158.11 / 1350.12 | Skipped / Skipped |

Peak process RSS in GiB for each process (both active-sequence rows together). This is a host process measurement, not dedicated GPU memory or total system consumption.

| OV convolution / KV configuration | 1K prompt | 8K prompt | 32K prompt |
|---|---:|---:|---:|
| Fused conv, `-npl 1` | 21.23 | 21.31 | 21.31 |
| Fused conv, `-npl 1,4` split | 21.31 | 21.31 | 20.00 |
| Fused conv, `-npl 1,4 --kv-unified` | 21.31 | 21.31 | Skipped |
| Separate conv, `-npl 1` | 21.31 | 21.31 | 21.21 |
| Separate conv, `-npl 1,4` split | 21.31 | 21.31 | Skipped |
| Separate conv, `-npl 1,4 --kv-unified` | 21.31 | 21.31 | Skipped |

The 30-minute limit interrupted `timed-fused-split-32768` after 267.8 seconds; 1 completed row(s) from that process are retained. Remaining cells are marked Skipped.


<!-- inplace-results:end -->

### Findings and next opportunities

All 1K and 8K cases completed. Both one-slot 32K variants and the fused split-KV
one-active-sequence row completed. The 30-minute limit interrupted the fused
split-KV four-active-sequence case; seven remaining rows were skipped. No further
GPU benchmark is running. RSS shown for the interrupted process covers only its
observed lifetime. System memory and process RSS measure different allocations;
the minimum system `MemAvailable` in completed processes ranged from 8.07 to
34.52 GiB, so the approximately 21 GiB process RSS is not total GPU/system usage.

Decode rates are close between convolution variants. Fused versus separate
one-slot decode is 38.82 versus 38.22 t/s at 1K, 34.31 versus 33.90 at 8K, and
24.66 versus 24.44 at 32K. Four-active unified decode is 93.43 versus 90.75 at 1K
and 71.90 versus 71.07 at 8K. These single-pass differences do not establish a
repeatable small speedup.

Separate convolution improves some prefill cases, but not all. Four-active split
8K prefill takes 27.930 seconds versus fused 88.200; unified 8K takes 24.270
versus 29.394. Conversely, one-slot 32K takes 30.124 seconds versus fused 24.133.
First-use shape/kernel preparation and different memory-pressure samples remain
in these timings. Keep fused convolution as the default and retain the explicit
separate mode for controlled follow-up; a universal long-prefill selection rule
is not justified by this matrix.

The largest decode gap remains split attention KV. With one active sequence in
four slots, fused 1K decode is 4.56 t/s split versus 38.40 unified, although both
use the same in-place recurrent-cache implementation. At 8K it is 5.17 versus
33.98. The upstream attention fusion excludes split roots because their stream
dimension exceeds one. This comparison points to the attention-cache path and
its layout as the next priority; it does not isolate individual copies or
kernels. Extend the 4D KV translation and StatelessKV fusion to split streams,
then measure actual binding/copy traffic before assigning the residual gap.

For recurrent computation, optimize fused convolution output stores and layouts,
or remove the separate path's materialization/reorder costs. The kernel control
shows why ordinary convolution arithmetic alone cannot predict full-chain
latency. Both variants retain staged in-place handling for cycles and duplicate
sources, with private output/copy-back for incompatible caller bindings.

---

## Default-batch completion of the OV 32K cases (2026-10-27)

This continuation measures the four unified-KV rows left unfinished by the
30-minute large-batch campaign. The initial plan also included three split-KV
rows, but the user skipped those comparisons until the split KV cache bottleneck
is addressed. Vulkan and 64K cases are outside this continuation. Model,
context allocation, in-place caches and requantization are unchanged. The
llama.cpp source is `3d9b2fb78` and OpenVINO is `40d69210bf`; only the benchmark
runner and reporting options change. Default logical batch size is 2048 and
microbatch size is 512. No `-b` or `-ub` flag is passed, and their environment
overrides are cleared. The compiled-model blobs from the earlier campaign are
reused in cache-only mode. GPU workloads run serially.

Raw commands, logs, timing rows and sampled memory:
`tmp/inplace-kv-default-batch-20261027/target/` in the original llama.cpp workspace.

```sh
BINARY=/home/zijun/llama.cpp-inplace-perf/build/MtpOVPerf/bin/llama-batched-bench
MODEL="$HOME/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
OUT=/home/zijun/llama.cpp/tmp/inplace-kv-default-batch-20261027/target
CACHE=/home/zijun/llama.cpp/tmp/inplace-kv-retest-20261027/target/model-cache
RUNNER=ggml/src/ggml-openvino/dev_doc/benchmarks/qwen36_ud_inplace_batched.py
python3 "$RUNNER" --binary "$BINARY" --model "$MODEL" --output "$OUT" \
  --cache-dir "$CACHE" --phase timed --default-batch --prompts 32768 \
  --mode fused --layout unified
python3 "$RUNNER" --binary "$BINARY" --model "$MODEL" --output "$OUT" \
  --cache-dir "$CACHE" --phase timed --default-batch --prompts 32768 \
  --mode separate --layout unified
python3 ggml/src/ggml-openvino/dev_doc/benchmarks/qwen36_ud_inplace_summary.py \
  --output "$OUT" --report ggml/src/ggml-openvino/dev_doc/QWEN36-UD-MULTISLOT-BENCHMARK.md \
  --default-continuation --finished
```

The equivalent raw benchmark commands use the same environment as the earlier
in-place rerun: GPU, native softplus, `q4_asym64_all`, `ENABLE_KV_4D=1` and
compiled-model cache-only loading. Only the separate variant sets
`GGML_OPENVINO_SEPARATE_RECURRENT_CONV=1`.

```sh
"$BINARY" -m "$MODEL" -c 524288 -npp 32768 -ntg 512 -npl 1,4 --kv-unified
```

Each of the two processes measures one then four active sequences. Batched-bench
uses random token IDs and does not test MTP acceptance. Split-KV batch/convolution
comparisons are deferred because full-cache handling dominates their decode;
this does not establish that convolution or batch size has no prefill cost.

<!-- default-continuation:start -->
Finished: 4/4 previously unfinished OV unified-KV 32K rows measured with default batch sizes. The three split-KV continuation rows were skipped at user request.

These are separate continuation measurements with `-b 2048 -ub 512` defaults; neither flag is passed. They do not replace the historical `-b 8192 -ub 8192` rows. Each sequence has 32,768 prompt tokens and 512 decode tokens; four slots and 524,288 total context tokens are allocated in every case. MTP is off. All rates below are aggregate; prefill seconds exclude startup and are not streaming TTFT.

| Convolution | KV | Active sequences | Prefill seconds | Prefill t/s | Decode t/s | Peak process RSS GiB | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Fused | Unified | 1 | 31.029 | 1056.05 | 24.57 | 21.29 | Complete |
| Fused | Unified | 4 | 175.468 | 746.98 | 40.79 | 21.29 | Complete |
| Separate | Unified | 1 | 31.405 | 1043.40 | 24.40 | 21.31 | Complete |
| Separate | Unified | 4 | 184.804 | 709.25 | 40.72 | 21.31 | Complete |

Peak process RSS is shared by both rows from a process. It does not include all GPU/system allocations.

<!-- default-continuation:end -->

All four retained rows finished successfully in approximately 10.2 minutes,
including both process starts. The three split-KV continuation rows are deferred
at user request; none were measured. No Vulkan tests were run and no benchmark
is running. Completed rows confirm the effective defaults `n_batch=2048` and
`n_ubatch=512`, with exactly 512 decode tokens per sequence. Both processes reused
the existing compiled cache without recompilation. Minimum system `MemAvailable`
was 27.19 GiB for fused convolution and 25.45 GiB for separate convolution; peak
process RSS was 21.29 and 21.31 GiB respectively.

At 32K, separate convolution does not improve this default-batch sample. One-active
decode is 24.57 versus 24.40 t/s, and four-active aggregate decode is 40.79 versus
40.72 t/s for fused versus separate. Fused prefill takes 31.03 versus 31.40 seconds
with one active sequence and 175.47 versus 184.80 with four. These are single
passes, so the small decode differences are not a repeatability claim.

Against the earlier default-batch unified-KV campaign, the current fused path's
32K four-active decode improves from 7.72 to 40.79 aggregate t/s, approximately
5.28 times, and prefill falls from 1279.25 to 175.47 seconds. This measures the
combined upstream/backend changes, not the in-place KV update in isolation.
The existing Vulkan split-KV four-active reference is 41.53 t/s, about 1.8% above
the current OV unified result. Vulkan's unified four-active 32K row remains
unmeasured, so this is a comparison with its split layout and older source.

---


Status of the earlier 2026-10-23 campaign: finished. All 16 original server cases and all eight
Vulkan full-GPU server reruns completed. Of 40 batched timing rows, 32 completed
and eight were skipped at user request. No benchmark remains pending.

## Model and runtime

- Model: `~/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
- Hardware: Intel Arc B390 integrated GPU, approximately 62 GiB usable system RAM.
- llama.cpp source: `dd06588b1`; OpenVINO source: `38aaf24a07` on
  `qwen-mtp-openvino-checkpoint`.
- Release builds, default batch 2048 and microbatch 512; no batch-size overrides.
  Both use the default two inference and batch threads on this machine.
- OpenVINO uses GPU, native softplus and `q4_asym64_all`. Recurrent caches use the
  default rotation implementation. Attention KV slicing is enabled; it currently
  applies to a single physical attention-cache stream and therefore supports
  unified KV but falls back to full-capacity bindings for split four-stream KV.
- Vulkan uses the stored GGUF quantization. These are backend configuration
  comparisons, not comparisons of identical compressed weight representations.
- GPU workloads run serially. Cache population is excluded from timed tests;
  OpenVINO measurements use compiled-model cache-only loading.

The Vulkan build uses a clean detached checkout of the same llama.cpp commit,
with `CMAKE_BUILD_TYPE=Release` and `GGML_VULKAN=ON`. This excludes local,
unconditional graph-dump instrumentation and the Debug build at `build/vulkan`.
The OpenVINO binary was rebuilt after reverting an experimental input fallback.

<!-- compact-matrices:start -->
## Result matrices

The campaign is finished: 32 batched timing rows, 16 original server cases and
8 Vulkan server reruns completed. Eight remaining batched rows were canceled.
No benchmark is pending. All completed cases decode 512 tokens per sequence.

### Batched-bench: prefill time and decode throughput

Each cell is **prefill seconds @ aggregate decode tokens/s**. For `-npl 1,4`,
the two entries are **one active / four active sequences** with four slots
allocated. `Skipped` means canceled, not zero throughput. `1K/8K/32K/64K`
mean 1024/8192/32768/65536 prompt tokens per sequence.

Batched-bench does not sample or stream output tokens, so it does not measure
TTFT. Its synchronized prefill duration is shown as a prefill-only proxy: it
excludes model loading, sampling, server scheduling and transport. With four
active sequences it is the time to prefill all four, not a per-request TTFT.

| Backend/configuration | 1K prompt | 8K prompt | 32K prompt | 64K prompt |
|---|---:|---:|---:|---:|
| OV `-npl 1` | 1.244 @ 34.75 | 9.062 @ 27.71 | 44.871 @ 14.19 | Skipped |
| OV `-npl 1,4` split | 2.527 @ 4.55 / 6.431 @ 26.70 | 11.767 @ 5.08 / 41.013 @ 20.93 | 55.671 @ 4.04 / 208.992 @ 11.10 | 143.794 @ 2.73 / 584.334 @ 5.20 |
| OV `-npl 1,4 --kv-unified` | 1.218 @ 35.51 / 3.734 @ 83.10 | 8.469 @ 27.95 / 47.204 @ 43.12 | 50.438 @ 14.32 / 1279.251 @ 7.72 | Skipped / Skipped |
| Vulkan `-npl 1` | 1.498 @ 32.49 | 18.785 @ 30.59 | 255.503 @ 25.99 | 1013.906 @ 19.80 |
| Vulkan `-npl 1,4` split | 1.508 @ 32.45 / 5.931 @ 72.39 | 18.901 @ 30.31 / 75.542 @ 65.11 | 258.362 @ 25.99 / 1021.572 @ 41.53 | Skipped / Skipped |
| Vulkan `-npl 1,4 --kv-unified` | 1.492 @ 32.44 / 5.964 @ 71.82 | 18.785 @ 30.69 / 75.933 @ 55.31 | 254.584 @ 25.42 / Skipped | Skipped / Skipped |

Memory usage was not recorded for these batched timing rows. The one saved
`/proc/meminfo` snapshot describes system memory before a run, so it is not
reported as benchmark memory usage. No additional tests were run for this table.

### Server: TTFT and end-to-end output throughput (includes prefill)

Each cell is **mean TTFT seconds @ end-to-end aggregate output tokens/s** over eight
MMLU questions (35-73 templated tokens, mean 53.25). Every server allocates
four slots and 524288 total context tokens. Columns select one or four requests
in flight; these are short natural prompts, not the long batched prompt matrix.

End-to-end output throughput is total generated tokens divided by the total
benchmark elapsed time. Its denominator includes prefill, decode, scheduling
and client overhead; its numerator counts generated tokens only. It is **not
decode throughput** and should not be compared directly with the decode t/s in
the batched-bench matrix. The server's decode-only rates are in the next table.

TTFT is the client-observed interval from request submission to the first
counted token arrival, including template application, scheduling, prompt
processing and transport. It is the harness's saved "Average prompt latency",
checked against per-request timestamps when those were saved. Stream buffering
can delay the first observed token. Aggregate throughput includes prompt time
and scheduling; native decode rates are shown separately below.

The primary Vulkan rows use the verified `-fit off -ngl 99` reruns. OV uses the
original completed cases with cache-only loading. OV requantization and stored
Vulkan GGUF weights differ, as described under Model and runtime.

| Backend/configuration | 1 request: TTFT s @ end-to-end t/s | 4 requests: TTFT s @ end-to-end aggregate t/s |
|---|---:|---:|
| OV, split, MTP off | 2.888 @ 4.43 | 10.355 @ 22.90 |
| OV, split, MTP on | 7.757 @ 7.62 | 39.164 @ 9.11 |
| OV, unified, MTP off | 0.582 @ 35.07 | 1.888 @ 80.55 |
| OV, unified, MTP on | 0.674 @ 48.11 | 2.102 @ 90.98 |
| Vulkan full GPU, split, MTP off | 0.812 @ 30.26 | 2.615 @ 61.41 |
| Vulkan full GPU, split, MTP on | 0.891 @ 44.79 | 1.692 @ 52.05 |
| Vulkan full GPU, unified, MTP off | 0.800 @ 30.33 | 5.961 @ 55.71 |
| Vulkan full GPU, unified, MTP on | 0.915 @ 44.00 | 2.903 @ 48.59 |

### Server: decode throughput per request (excludes prefill)

These values are the mean server-reported tokens/s per request during decode.
Prompt processing time is excluded. Four-request values here are **per request,
not aggregate**. Multiplying the mean by four gives only an approximate
aggregate decode rate: requests do not necessarily start and finish decoding
together, and this is an arithmetic mean of per-request rates.

| Backend/configuration | 1 request: decode t/s per request | 4 requests: decode t/s per request |
|---|---:|---:|
| OV, split, MTP off | 4.54 | 6.47 |
| OV, split, MTP on | 8.62 | 3.88 |
| OV, unified, MTP off | 36.47 | 21.73 |
| OV, unified, MTP on | 51.35 | 25.91 |
| Vulkan full GPU, split, MTP off | 31.74 | 16.63 |
| Vulkan full GPU, split, MTP on | 48.54 | 13.87 |
| Vulkan full GPU, unified, MTP off | 31.79 | 16.60 |
| Vulkan full GPU, unified, MTP on | 47.71 | 13.29 |

MTP comparison using the same **decode-only tokens/s per request**, with
**MTP off → MTP on** in each cell. Vulkan uses the verified full-GPU runs.
All cases allocate four server slots; the columns select one or four requests
in flight.

| Backend/layout | One request: off → on | Four requests: off → on |
|---|---:|---:|
| OV split | 4.54 → 8.62 | 6.47 → 3.88 |
| OV unified | 36.47 → 51.35 | 21.73 → 25.91 |
| Vulkan split | 31.74 → 48.54 | 16.63 → 13.87 |
| Vulkan unified | 31.79 → 47.71 | 16.60 → 13.29 |

MTP improves decode throughput in every one-request case. With four requests
in flight, it improves OV unified throughput but lowers OV split and Vulkan
throughput in these measurements.

For reference, historical Vulkan default-fit results in the same TTFT @
aggregate-throughput format. Expert placement was not controlled in these runs;
they are excluded from the primary Vulkan comparison above.

| Backend/configuration | 1 request in flight | 4 requests in flight |
|---|---:|---:|
| Vulkan default fit, split, MTP off | 1.025 @ 16.28 | 3.439 @ 31.00 |
| Vulkan default fit, split, MTP on | 1.126 @ 23.23 | 2.773 @ 31.08 |
| Vulkan default fit, unified, MTP off | 1.052 @ 16.13 | 6.886 @ 31.06 |
| Vulkan default fit, unified, MTP on | 1.148 @ 22.62 | 3.080 @ 46.84 |

### Available server memory measurements

Each cell is **main-context device / host allocations in GiB**, from llama.cpp's
shutdown memory breakdown (model + main cache + main compute buffers). This is
an allocation tally, not peak process or whole-system memory usage. Host and
device storage both occupy DRAM on this integrated GPU. `N/R` means not recorded
in the timed run; diagnostic measurements from other runs are not substituted.

The MTP rows omit the separate draft context, output buffers and allocations
outside that main-context breakdown. In the Vulkan MTP startup logs, the draft
context additionally allocates 1024 MiB of attention KV, 170.02 / 554.02 MiB
of device compute and 136.02 / 520.02 MiB of host compute for split / unified,
plus its output buffer. Repeated reserve messages are not additional allocations.
Thus these cells must not be interpreted as total MTP memory or compared with
unrecorded OV peaks.

| Backend/configuration | 1 request in flight | 4 requests in flight |
|---|---:|---:|
| OV, split, MTP off | N/R | N/R |
| OV, split, MTP on | N/R | N/R |
| OV, unified, MTP off | N/R | N/R |
| OV, unified, MTP on | N/R | N/R |
| Vulkan full GPU, split, MTP off | 30.54 / 0.64 | 30.54 / 0.64 |
| Vulkan full GPU, split, MTP on | 31.55 / 0.64 | 31.55 / 0.64 |
| Vulkan full GPU, unified, MTP off | 30.92 / 1.02 | 30.92 / 1.02 |
| Vulkan full GPU, unified, MTP on | 31.93 / 1.02 | 31.93 / 1.02 |

<!-- compact-matrices:end -->

## Batched benchmark

`llama-batched-bench` uses random token IDs and does not run MTP. Each sequence
gets the stated prompt length and 512 decode steps. No prompt sharing is enabled.
All throughput values in its output are aggregate across the active sequences.

The executable has no repetition loop or `llama-bench`-style `-r` repetitions.
It runs each `(prompt length, decode length, active sequence count)` combination
once, after one 16-token warmup per process. Thus the `-npl 1` command contains
four measured cases, and `-npl 1,4` contains eight. `-ntg 512` means 512 decode
steps per sequence; with four active sequences, each step processes four
inputs and the measured decode totals 2048 tokens. Default microbatch 512
makes 64K prefill require 128 microbatches per sequence, or 512 microbatches
for four sequences. These are required model evaluations, not benchmark
repetitions that can be removed while retaining the specified prompt length.

Prefill dominates the slow Vulkan cases: four independent 32K sequences took
1021.572 seconds to prefill versus 49.309 seconds to decode. Reducing decode
from 512 to 128 would save roughly 37 seconds in that case, leaving the
17-minute prefill. The requested 512 decode steps are retained. On interruption,
the former whole-matrix runner repeated its completed shorter rows; those
restarts are not intentional repetitions. Results are now checkpointed per
prompt length, and remaining long-prompt batched cases are skipped at user request.
The full-GPU Vulkan server reruns then completed within the half-hour time budget.

```bash
MODEL="$HOME/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
export GGML_OPENVINO_DEVICE=GPU
export GGML_OPENVINO_NATIVE_SOFTPLUS=1
export GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all
export GGML_OPENVINO_COMPILED_MODEL_CACHE_DIR="$PWD/tmp/mtp-regression-20261022/rotation-cache"
export GGML_OPENVINO_COMPILED_MODEL_CACHE_ONLY=1

build/MtpOVPerf/bin/llama-batched-bench -m "$MODEL" -c 131072 \
  -npp 1024,8192,32768,65536 -ntg 512 -npl 1
build/MtpOVPerf/bin/llama-batched-bench -m "$MODEL" -c 524288 \
  -npp 1024,8192,32768,65536 -ntg 512 -npl 1,4
build/MtpOVPerf/bin/llama-batched-bench -m "$MODEL" -c 524288 \
  -npp 1024,8192,32768,65536 -ntg 512 -npl 1,4 --kv-unified
```

`GGML_OPENVINO_DISABLE_KV_SLICE` and
`GGML_OPENVINO_RECURRENT_CACHE_INPLACE` are unset. The split one-slot allocation
is 128K; split four-slot allocation is 128K per slot. Unified KV uses one 512K
pool, allowing the same aggregate capacity without reserving equal slot regions.
Vulkan uses the same commands with the clean Release binary path. These commands
describe the original full matrix; rows labeled Skipped were canceled before
completion and are not measured results.

<!-- batch-results:start -->
Batched matrix status. Values are aggregate tokens/s; `Skipped` marks a
user-canceled long-prompt case. All other rows completed.
Each completed timing row finished its prefill and all 512
synchronized decode steps. Completed rows from an interrupted matrix are
retained; they do not imply that its other prompt lengths completed.

| Backend | KV | Allocated sequences | Prompt/sequence | Active sequences | Prefill t/s | Decode t/s | Status |
|---|---|---:|---:|---:|---:|---:|---|
| OpenVINO | Split | 1 | 1,024 | 1 | 823.14 | 34.75 | Complete |
| OpenVINO | Split | 1 | 8,192 | 1 | 904.00 | 27.71 | Complete |
| OpenVINO | Split | 1 | 32,768 | 1 | 730.27 | 14.19 | Complete |
| OpenVINO | Split | 1 | 65,536 | 1 | - | - | Skipped |
| OpenVINO | Split | 4 | 1,024 | 1 | 405.21 | 4.55 | Complete |
| OpenVINO | Split | 4 | 1,024 | 4 | 636.96 | 26.70 | Complete |
| OpenVINO | Split | 4 | 8,192 | 1 | 696.17 | 5.08 | Complete |
| OpenVINO | Split | 4 | 8,192 | 4 | 798.97 | 20.93 | Complete |
| OpenVINO | Split | 4 | 32,768 | 1 | 588.60 | 4.04 | Complete |
| OpenVINO | Split | 4 | 32,768 | 4 | 627.16 | 11.10 | Complete |
| OpenVINO | Split | 4 | 65,536 | 1 | 455.76 | 2.73 | Complete |
| OpenVINO | Split | 4 | 65,536 | 4 | 448.62 | 5.20 | Complete |
| OpenVINO | Unified | 4 | 1,024 | 1 | 840.45 | 35.51 | Complete |
| OpenVINO | Unified | 4 | 1,024 | 4 | 1097.04 | 83.10 | Complete |
| OpenVINO | Unified | 4 | 8,192 | 1 | 967.34 | 27.95 | Complete |
| OpenVINO | Unified | 4 | 8,192 | 4 | 694.17 | 43.12 | Complete |
| OpenVINO | Unified | 4 | 32,768 | 1 | 649.66 | 14.32 | Complete |
| OpenVINO | Unified | 4 | 32,768 | 4 | 102.46 | 7.72 | Complete |
| OpenVINO | Unified | 4 | 65,536 | 1 | - | - | Skipped |
| OpenVINO | Unified | 4 | 65,536 | 4 | - | - | Skipped |
| Vulkan | Split | 1 | 1,024 | 1 | 683.62 | 32.49 | Complete |
| Vulkan | Split | 1 | 8,192 | 1 | 436.10 | 30.59 | Complete |
| Vulkan | Split | 1 | 32,768 | 1 | 128.25 | 25.99 | Complete |
| Vulkan | Split | 1 | 65,536 | 1 | 64.64 | 19.80 | Complete |
| Vulkan | Split | 4 | 1,024 | 1 | 679.07 | 32.45 | Complete |
| Vulkan | Split | 4 | 1,024 | 4 | 690.61 | 72.39 | Complete |
| Vulkan | Split | 4 | 8,192 | 1 | 433.42 | 30.31 | Complete |
| Vulkan | Split | 4 | 8,192 | 4 | 433.77 | 65.11 | Complete |
| Vulkan | Split | 4 | 32,768 | 1 | 126.83 | 25.99 | Complete |
| Vulkan | Split | 4 | 32,768 | 4 | 128.30 | 41.53 | Complete |
| Vulkan | Split | 4 | 65,536 | 1 | - | - | Skipped |
| Vulkan | Split | 4 | 65,536 | 4 | - | - | Skipped |
| Vulkan | Unified | 4 | 1,024 | 1 | 686.38 | 32.44 | Complete |
| Vulkan | Unified | 4 | 1,024 | 4 | 686.81 | 71.82 | Complete |
| Vulkan | Unified | 4 | 8,192 | 1 | 436.08 | 30.69 | Complete |
| Vulkan | Unified | 4 | 8,192 | 4 | 431.54 | 55.31 | Complete |
| Vulkan | Unified | 4 | 32,768 | 1 | 128.71 | 25.42 | Complete |
| Vulkan | Unified | 4 | 32,768 | 4 | - | - | Skipped |
| Vulkan | Unified | 4 | 65,536 | 1 | - | - | Skipped |
| Vulkan | Unified | 4 | 65,536 | 4 | - | - | Skipped |

Original OpenVINO split four-slot log:
`tmp/mtp-regression-20261022/pool-v2-np4.log`. Vulkan one-slot log:
`tmp/ud-backend-matrix-20261023/batch-vulkan-split-npl1.log`. Its split
four-slot rows through 32K are in `batch-vulkan-split-npl1-4.log` in the same
folder. New runs save one prompt length per process as `batch-*-pp*.log` and
checkpoint it separately. This avoids repeating completed lengths on restart.
It also introduces a fresh 16-token warmup for each remaining prompt length;
these are labeled by per-case provenance rather than silently pooled as repeats.
Vulkan one-slot 64K prefill took 1,013.906 seconds versus 25.865 for decode.

<!-- batch-results:end -->

## Server benchmark

The matrix has two backends, split/unified KV, MTP off/on, and request concurrency
1/4. All server cases allocate four slots and 524288 total attention-cache tokens.
Each case sends the first eight question texts from `cais/mmlu`, `all`, test,
using the native chat template and `/completion` endpoint. Greedy generation
requests exactly 512 tokens with `ignore_eos=true` and `cache_prompt=false`.
The question lengths are measured with `/tokenize`; they are not padded to
1K/8K/32K/64K. The long-context prompt matrix belongs to the batched tests above.
This uses MMLU as natural input text, not as an accuracy evaluation.

Server arguments, in addition to the common model/configuration above:

```bash
llama-server -m "$MODEL" -np 4 -c 524288 --temp 0
llama-server -m "$MODEL" -np 4 -c 524288 --temp 0 --kv-unified
llama-server -m "$MODEL" -np 4 -c 524288 --temp 0 \
  --spec-type draft-mtp --spec-draft-n-max 2
llama-server -m "$MODEL" -np 4 -c 524288 --temp 0 --kv-unified \
  --spec-type draft-mtp --spec-draft-n-max 2
```

The wrapper imports `scripts/server-bench.py`, loads the saved local MMLU parquet,
and separates server slot count (always four) from client concurrency (one or
four). Native request timing JSON and stream arrival times are saved. Server SSE
errors fail the run immediately. A completed case must have eight responses,
each with native `predicted_n=512` and a final cumulative
`tokens_predicted=512`, and no draft-computation failure or missing compiled-model signature in the server log.
The first concurrent unified-MTP attempt completed output by falling back after
missing draft signatures; that attempt is invalid and excluded. Cache population
exercises explicit active-slot subsets and full-length requests before repeating the test. Backend draft sampling adds ports for the
active sequence IDs, so a four-request startup alone does not populate every
valid intermediate graph signature.

- Native decode t/s/request is the mean of the eight server-reported decode rates.
- End-to-end aggregate t/s is 4096 native generated tokens divided by the
  benchmark's client-observed duration, including prompt processing and scheduling.
- Native streams may withhold individual token-ID events while accumulating an
  incomplete UTF-8 character. `server-bench.py` uses the cumulative
  `tokens_predicted` counter, including the final event, to count this work.
  Buffered tokens share the observation time when the counter advances; this
  cannot recover their individual GPU completion times. One saved Vulkan
  unified non-MTP sequential run contains 4086 partial token IDs but all 4096
  generated tokens are confirmed by final counters and native timing JSON. Its
  raw arrivals are preserved, and aggregate throughput uses 4096. Earlier runs
  whose counts already match are unaffected.
- Draft acceptance is the sum of accepted/generated drafts in per-request logs.

A reusable wrapper is saved as
[`benchmarks/qwen36_ud_mmlu.py`](benchmarks/qwen36_ud_mmlu.py). It needs the
Python dependencies used by `scripts/server-bench.py` plus `pyarrow`, and a local
MMLU `all` test parquet at `tmp/mmlu-dataset/all/test-00000-of-00001.parquet`
(or the path in `UD_MMLU_PARQUET`). Example full server matrix:

```bash
export UD_VULKAN_BIN_DIR=/home/zijun/llama-benchmark-clean/build/vulkan/bin
export UD_LLAMA_COMMIT=dd06588b1
export UD_OPENVINO_COMMIT=38aaf24a07
for backend in ov vulkan; do
  for layout in split unified; do
    for mode in off mtp; do
      for concurrency in 1 4; do
        python ggml/src/ggml-openvino/dev_doc/benchmarks/qwen36_ud_mmlu.py \
          "$backend" "$layout" "$mode" "$concurrency"
      done
    done
  done
done
```

OV defaults to cache-only loading and the cache path above. Populate missing
signatures separately with `UD_PRIME_SUBSETS=1 UD_CACHE_ONLY=0 UD_TEST_CTX=8192`
and a distinct `UD_RUN_LABEL`, then rerun the full case in cache-only mode.
`UD_OV_BIN_DIR`, `UD_VULKAN_BIN_DIR`, `UD_MODEL`, `UD_COMPILED_CACHE`, and
`UD_BENCH_DIR` can redirect builds, model, cache and output locations. Use a clean
Release Vulkan build; the local `build/vulkan` used for diagnostics is unsuitable
for reproducing this comparison.

Raw artifacts and the serial matrix runner: `tmp/ud-backend-matrix-20261023/`.
The runner records effective environment and commit provenance for each server
case. `summary.md`, `server-summary.json`, and `batch-summary.json` are generated
from completed responses and raw logs.

Original default-fit server matrix (historical Vulkan rows; use the full-GPU
reruns for the primary comparison):

| Backend | KV | MTP | Requests in flight | Native decode t/s/request | End-to-end aggregate t/s | Draft acceptance |
|---|---|---|---:|---:|---:|---:|
| OpenVINO | Split | On | 1 | 8.62 | 7.62 | 2632/2905 (90.6%) |
| OpenVINO | Split | On | 4 | 3.88 | 9.11 | 2409/3348 (72.0%) |
| OpenVINO | Split | Off | 1 | 4.54 | 4.43 | - |
| OpenVINO | Split | Off | 4 | 6.47 | 22.90 | - |
| OpenVINO | Unified | On | 1 | 51.35 | 48.11 | 2623/2922 (89.8%) |
| OpenVINO | Unified | On | 4 | 25.91 | 90.98 | 2622/2923 (89.7%) |
| OpenVINO | Unified | Off | 1 | 36.47 | 35.07 | - |
| OpenVINO | Unified | Off | 4 | 21.73 | 80.55 | - |
| Vulkan | Split | On | 1 | 24.48 | 23.23 | 2617/2933 (89.2%) |
| Vulkan | Split | On | 4 | 8.27 | 31.08 | 2607/2955 (88.2%) |
| Vulkan | Split | Off | 1 | 16.81 | 16.28 | - |
| Vulkan | Split | Off | 4 | 8.16 | 31.00 | - |
| Vulkan | Unified | On | 1 | 23.88 | 22.62 | 2625/2919 (89.9%) |
| Vulkan | Unified | On | 4 | 12.88 | 46.84 | 2611/2945 (88.7%) |
| Vulkan | Unified | Off | 1 | 16.66 | 16.13 | - |
| Vulkan | Unified | Off | 4 | 8.72 | 31.06 | - |

## Interpretation and limits

The following comparisons describe the original default-fit server matrix.
Use the full-GPU Vulkan reruns below for the primary backend comparison, since
the original Vulkan weight placement was not controlled.

For these short MMLU prompts in a large unified allocation, OpenVINO MTP increases
sequential aggregate throughput by 37.2% (35.07 to 48.11 t/s) and four-request
aggregate throughput by 12.9% (80.55 to 90.98 t/s). With Vulkan, unified MTP raises
sequential throughput from 16.13 to 22.62 t/s and four-request throughput from
31.06 to 46.84 t/s. Neither comparison controls the model's numerical weights
across backends, since OpenVINO requantizes them.

OpenVINO's split four-slot MTP result is much lower than its unified result at
this allocation. Its sequential split run accepts 90.6% of drafts, close to
the unified 89.8%, yet reaches only 8.62 native decode t/s versus 51.35.
Thus poor acceptance does not explain the sequential gap. Full-capacity
attention-cache binding remains a candidate cost: slicing currently handles only one physical stream. Recurrent state size
is determined by sequence count and snapshot capacity, not attention-context
length, so the large context does not itself enlarge the recurrent caches.
The split concurrent run also accepts 72.0% of drafts versus 89.7% for unified;
this changes the amount of verification work. System memory during the split
concurrent run varied around 41–54 GiB. These measurements do not isolate copying,
recurrent kernels, attention, memory pressure or numerical/scheduling differences
well enough to assign the entire layout gap to one of them. At four-request
concurrency, split MTP reaches 9.11 aggregate t/s versus non-MTP 22.90; unified
MTP reaches 90.98 versus non-MTP 80.55. MTP therefore helps unified serving but
hurts the split serving configuration in this large-allocation sample.

Vulkan's two layouts have almost identical non-MTP aggregate throughput with four
concurrent requests (31.00 versus 31.06 t/s). Its concurrent MTP case gains from
unified KV (31.08 to 46.84 t/s), while its sequential MTP results are close
(23.23 split and 22.62 unified). Vulkan's large-allocation rates are below the
earlier small-context controls.
Untimed verbose diagnostics expose a major confound: default `--fit on` can
move expert tensors to `Vulkan_Host` even while reporting all 42/42 layers as
GPU-offloaded. The diagnostic unified-MTP startup reported 43,108 MiB free and
kept 21,087.70 MiB in Vulkan device model storage plus 515.31 MiB host storage.
Subsequent diagnostic starts reported only about 26,700–26,850 MiB free and fit
6,402–8,511 MiB of model tensors into host storage. The split-MTP diagnostic
explicitly overrides expert down/gate/up tensors to `Vulkan_Host`. A layer-count
message therefore does not establish that every expert matmul stays on the GPU.
The Vulkan backend's `supports_buft` accepts its device buffer type rather than
`Vulkan_Host`, so expert host placement can change scheduler placement as well
as memory access costs.

The memory fitting decision uses the Vulkan backend's sum of
`heapBudget[i] - heapUsage[i]`, rather than the physical 64 GiB system-memory
capacity. Khronos describes this budget as a changing guideline for process
allocations; it may respond to other system activity and is not a fixed hardware
limit. See the [Vulkan memory-budget specification](https://github.com/KhronosGroup/Vulkan-Docs/blob/main/chapters/memory.adoc#memory-device).

The installed Mesa package is `26.2.4~kisak1~n`. In the corresponding upstream
Mesa 26.2.4 source, Intel ANV normally caps its system-memory heap at 75%
of the kernel-reported system memory on this class of machine (older Xe kernels
with an already restricted heap have a separate path). The diagnostic reports
47,898 MiB total Vulkan heap, consistent with this restriction. It then updates
available system-heap memory, distributes it over those heaps, reserves 10%,
and reports
`heapBudget = min(heapSize, heapUsage + 0.9 * availableShare)`, rounded down to
MiB. Both the heap-size cap and the available-memory reserve apply. The Intel
device-info path bounds the kernel's available-memory estimate
by Linux `/proc/meminfo`'s `MemAvailable`. This explains why total installed RAM
is not the budget used by the fitter. It does not establish the cause of the
particular 43-to-27 GiB change between diagnostic starts. See
[Mesa's Intel budget calculation](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.2.4/src/intel/vulkan/anv_physical_device.c),
[available-memory adjustment](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.2.4/src/intel/dev/intel_device_info.c),
and [Linux available-memory query](https://gitlab.freedesktop.org/mesa/mesa/-/blob/mesa-26.2.4/src/util/os_misc.c).

In the split non-MTP diagnostic, the estimate was 31,278 MiB against 26,739 MiB
free. Including the default 1024 MiB margin, the fitter needed to reduce GPU
allocations by 5563 MiB. Context was explicitly set to 524288, so it left context
unchanged and moved expert tensors to `Vulkan_Host`: the final model allocation
was 14,696.44 MiB device plus 6402.22 MiB host. The all-GPU non-MTP model is
about 21 GiB, attention KV is 10 GiB at this allocation, and recurrent/compute
buffers add more; comparing model size alone with total RAM misses these costs.
MTP adds its attention cache and recurrent snapshots as well.

This is a conservative software placement decision, not evidence that the UMA
hardware cannot accommodate an all-GPU run. The exact reason the driver budget
fell from roughly 43 GiB to 27 GiB across starts has not been isolated. Host and
GPU allocations both use system DRAM on this integrated GPU, but they remain
different backend allocation types; expert host placement can change where
matmuls execute. The separately labeled `-fit off -ngl 99` controls avoid that
placement decision; their measured results are shown below.

These diagnostics occurred after the timed server matrix, so their individual
placement values cannot be retroactively assigned to timed cases. They do show
that default fitting can change the workload with the driver-reported budget,
which varied substantially across successive process starts. The lower Vulkan
server rates must be interpreted as configured-default end-to-end results, not
a controlled all-GPU kernel comparison. `llama-batched-bench` loads the model
and context directly and does not call `common_fit_params`; its results do not
have this server-specific automatic-fit confound. Separate server reruns
with `UD_FORCE_GPU=1` (explicit `-ngl 99 -fit off`) are reported below, separately
from the original default-fit results.

This is an eight-request throughput sample, with one pass per case. MMLU question
texts average 53.25 templated tokens in this sample. It tests MTP on natural short
prompts and separately tests long-context non-MTP with random token IDs; it does
not measure long-context MTP, MMLU accuracy, or statistical variation across
repeated benchmark passes.

## Vulkan server reruns with full GPU weight placement

These reruns keep the same model, four slots, 524288 context, MMLU inputs,
concurrency and 512 generated tokens. They add `-ngl 99 -fit off` and llama
info logging (`-lv 4`) to record actual model storage. All eight original Vulkan
server configurations are rerun because their timed logs did not retain enough
placement detail to identify individually which had host-offloaded experts.
The small host embedding allocation is expected; each included rerun retains
more than 20,000 MiB of device weights and less than 1024 MiB of host weights.
The original default-fit table is retained as historical configuration data.

Reproduce a full-GPU server case with the reusable wrapper, for example:

```bash
UD_FORCE_GPU=1 UD_LOG_VERBOSITY=4 UD_RUN_LABEL=gpu-vulkan-unified-mtp-c4 \
UD_VULKAN_BIN_DIR=/home/zijun/llama-benchmark-clean/build/vulkan/bin \
UD_LLAMA_COMMIT=dd06588b1 UD_OPENVINO_COMMIT=38aaf24a07 \
python ggml/src/ggml-openvino/dev_doc/benchmarks/qwen36_ud_mmlu.py \
  vulkan unified mtp 4
```

| KV | MTP | Requests in flight | Native decode t/s/request | End-to-end aggregate t/s | Draft acceptance | Device / host model MiB | Status |
|---|---|---:|---:|---:|---:|---:|---|
| Split | Off | 1 | 31.74 | 30.26 | - | 20583.34 / 515.31 | Complete |
| Split | Off | 4 | 16.63 | 61.41 | - | 20583.34 / 515.31 | Complete |
| Split | On | 1 | 48.54 | 44.79 | 2626/2911 (90.2%) | 21087.70 / 515.31 | Complete |
| Split | On | 4 | 13.87 | 52.05 | 2621/2927 (89.5%) | 21087.70 / 515.31 | Complete |
| Unified | Off | 1 | 31.79 | 30.33 | - | 20583.34 / 515.31 | Complete |
| Unified | Off | 4 | 16.60 | 55.71 | - | 20583.34 / 515.31 | Complete |
| Unified | On | 1 | 47.71 | 44.00 | 2626/2911 (90.2%) | 21087.70 / 515.31 | Complete |
| Unified | On | 4 | 13.29 | 48.59 | 2629/2908 (90.4%) | 21087.70 / 515.31 | Complete |

With expert weights kept on the GPU, Vulkan sequential rates are close across
KV layouts. The original default-fit rates therefore mixed backend performance
with weight-placement effects. The full-GPU table is the primary Vulkan server
comparison; the historical default-fit table documents the earlier workload.

Split KV: MTP changes sequential aggregate throughput from 30.26 to
44.79 t/s (+48.0%) and four-request aggregate throughput from
61.41 to 52.05 t/s (-15.2%).

Unified KV: MTP changes sequential aggregate throughput from 30.33 to
44.00 t/s (+45.1%) and four-request aggregate throughput from
55.71 to 48.59 t/s (-12.8%).

The user stopped remaining long-prompt batched tests to finish this campaign
within half an hour. The Vulkan unified 32K one-active row finished before
cancellation and is retained; its four-active row did not finish. OpenVINO
unified four-active 32K took 1279.251 seconds prefill and 265.141 seconds decode
(102.46 and 7.72 aggregate t/s). That measured slowdown also made its remaining
64K case unsuitable for the revised time budget; no 64K values are extrapolated.
