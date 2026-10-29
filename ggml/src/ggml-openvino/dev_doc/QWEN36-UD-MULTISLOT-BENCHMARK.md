# Qwen3.6 UD benchmark results

The current Qwen GPU implementation uses in-place fused recurrent ops and stable MTP sampling. See [implementation and validation](OPENVINO-RECURRENT-CACHE.md) and the [sampling graph](MTP-SAMPLING-GRAPH.md). The tables below retain their actual measurement dates and configurations; newer implementation decisions do not relabel historical results.

## Non-MTP batched-bench: default and large batches (2026-10-28)

Larger batches improve OV prefill in the completed one-slot and unified-KV cases. They do not materially improve decode. Unified 32K with four active sequences and an 8192-token microbatch remains unresolved; use default batch sizes as the general baseline.

<!-- vulkan-default-reference:2026-10-23 -->
<!-- batch-comparison:start -->
47/60 planned rows were measured; the selected campaign is complete. Missing values remain explicit.

This campaign compares default `-b 2048 -ub 512` with large `-b 8192 -ub 8192` on the same llama.cpp source. Default batch flags are omitted. OV uses fused in-place recurrent caches, GPU, native softplus, `q4_asym64_all`, `ENABLE_KV_4D=1`, enabled KV slicing and compiled-model cache-only loading. Vulkan uses the stored GGUF weights. GPU workloads run serially, with no elapsed-time cap.

Each cell is **prefill seconds @ aggregate decode tokens/s**; four-slot rows show **one active / four active sequences**. Prompt lengths are 1024/8192/32768 tokens per sequence, with 512 generated tokens each. One-slot context is 131072 and four-slot context is 524288 total. MTP and shared-prompt processing are off; batched-bench uses random token IDs. Each timing row is one pass. Prefill includes first-use runtime shape/kernel preparation, excludes model loading and is not streaming TTFT. Decode excludes prefill.

The default-batch Vulkan rows below use the 2026-10-23 Release build at `dd06588b1`, selected as the reference because the current-build results were similar. OV and large-batch Vulkan rows use the current campaign. This displayed comparison spans source revisions; the 47 measured campaign rows and their original values remain in the raw artifacts. Historical reference rows do not increase the campaign completion count.

### Default batch: `-b / -ub = 2048 / 512`

| Backend/configuration | 1K prompt | 8K prompt | 32K prompt |
|---|---:|---:|---:|
| OV fused `-npl 1` | 1.197 @ 34.61 | 7.703 @ 33.77 | 32.539 @ 24.51 |
| OV fused `-npl 1,4` split | 8.959 @ 4.54 / 7.095 @ 27.22 | 10.842 @ 5.19 / 38.769 @ 23.06 | 43.922 @ 4.47 / 170.131 @ 14.54 |
| OV fused `-npl 1,4 --kv-unified` | 1.158 @ 37.46 / 3.479 @ 91.95 | 7.224 @ 33.51 / 30.785 @ 71.41 | 30.326 @ 24.42 / 179.282 @ 40.84 |
| Vulkan reference `-npl 1` | 1.498 @ 32.49 | 18.785 @ 30.59 | 255.503 @ 25.99 |
| Vulkan reference `-npl 1,4` split | 1.508 @ 32.45 / 5.931 @ 72.39 | 18.901 @ 30.31 / 75.542 @ 65.11 | 258.362 @ 25.99 / 1021.572 @ 41.53 |
| Vulkan reference `-npl 1,4 --kv-unified` | 1.492 @ 32.44 / 5.964 @ 71.82 | 18.785 @ 30.69 / 75.933 @ 55.31 | 254.584 @ 25.42 / Skipped |

OpenVINO recorded-process peak RSS: 21.30-21.31 GiB. Lowest sampled system MemAvailable: 11.45 GiB.

Memory usage was not recorded for the historical Vulkan reference rows. Current-build memory records remain in the raw artifacts.

### Large batch: `-b / -ub = 8192 / 8192`

| Backend/configuration | 1K prompt | 8K prompt | 32K prompt |
|---|---:|---:|---:|
| OV fused `-npl 1` | 1.072 @ 37.85 | 5.147 @ 33.68 | 22.989 @ 24.31 |
| OV fused `-npl 1,4` split | 2.640 @ 4.53 / 4.896 @ 27.05 | 9.554 @ 5.10 / 41.408 @ 22.61 | 28.137 @ 4.44 / 181.780 @ 14.51 |
| OV fused `-npl 1,4 --kv-unified` | 1.089 @ 37.33 / 2.402 @ 92.00 | 6.498 @ 33.46 / 25.233 @ 71.52 | 25.083 @ 24.38 / Canceled |
| Vulkan `-npl 1` | 1.548 @ 32.42 | 24.700 @ 30.68 | Skipped |
| Vulkan `-npl 1,4` split | 1.533 @ 32.41 / 9.641 @ 74.00 | 24.446 @ 30.81 / Skipped | Skipped / Skipped |
| Vulkan `-npl 1,4 --kv-unified` | 1.562 @ 32.44 / 9.658 @ 72.78 | Skipped / Skipped | Skipped / Skipped |

OpenVINO recorded-process peak RSS: 21.27-21.31 GiB. Lowest sampled system MemAvailable: 6.89 GiB.

Vulkan recorded-process peak RSS: 0.55-0.66 GiB. Lowest sampled system MemAvailable: 19.64 GiB.

The unified four-active OV 32K large-batch run was canceled after 2147.7s without completing prefill. Remaining expensive Vulkan tests were stopped after sufficient preliminary evidence; their completed rows remain above. No missing value is extrapolated.

Process RSS excludes some GPU/system allocations. Raw commands, per-process memory samples, source patches and binary identities: `/home/zijun/llama.cpp/tmp/batch-size-matrix-20261028`.

<!-- batch-comparison:end -->

## Stable MTP sampling graph (2026-10-28)

At `-np 4 -c 8192 --kv-unified --no-warmup`, a single slot-3 request prepares two blobs, one trunk and one draft. A new cache-only process serves all 15 requested slot subsets without another model: 33 validation requests / 768 generated tokens. These short requests check correctness and cache reuse; throughput uses the continuous streaming harness below. Validation artifacts and the validation-only `target_validate.py`: `tmp/stable-sampling-20261028`.

## Matched-context and streaming-harness follow-up (2026-10-29)

All rows use four slots and eight 512-token MMLU outputs. Larger-context runs retain `-c 524288` with fitting disabled and full GPU offload. Logs confirm 10240 MiB trunk and 1024 MiB draft attention KV buffers. The same revision-13 cache prepared at 8192 is reused; actual prompt lengths stay short.

| Implementation / measurement | Total context | Request scheduling | Request warmup | Native decode t/s per request |
|---|---:|---|---|---:|
| Earlier translation, stored result | 524288 | Streaming server-bench, continuous replacement | None | 22.79 |
| Stable translation, original harness | 524288 | Streaming server-bench, continuous replacement | Yes | 27.71 |
| Stable translation, original harness | 524288 | Streaming server-bench, continuous replacement | None | 27.03 |

All retained four-request MTP throughput tests use `scripts/server-bench.py`: eight MMLU questions, exactly 512 outputs per request, and four requests in flight with immediate replacement when a request completes. Incoming prefill can delay ongoing decode. This is the default serving benchmark approach.

The closest observed comparison is **22.79 -> 27.03 t/s per request (+18.6%)**. This supports stable translation and removal of active-subset-specific models/first-use work as the main contributor. It does not establish a raw-kernel speedup or an exact causal decomposition. These are single-pass comparisons with a stored baseline; the new runs disable the ordinary startup probe, while the earlier run kept it. Request warmup is separate from that startup probe.

The new fresh streaming run imports only the draft and trunk models, at 3.899s and 28.026s, before generation. Earlier variant imports at 2:43.605, 2:47.778 and 2:53.037 overlap generation and can lower native decode rate. Warm/fresh stable streaming aggregate end-to-end rates are 102.65/63.42 t/s; acceptance is 2634/2897 and 2627/2909. Each run completes eight requests and 4096 outputs.

Commands, effective environments, logs, responses and `comparison.json`: `/home/zijun/llama.cpp/tmp/stable-sampling-ctx524288-20261029`. Its local harness copies retain their original paths as provenance, not as instructions for the current checkout.

### Stable translation: MTP off -> on (continuous requests)

Native decode tokens/s per request, excluding prompt evaluation. Fresh servers use four slots, total context 524288, unified KV, default batches, the first eight MMLU questions and 512 generated tokens per request. The streaming harness replaces completed requests immediately, with one or four requests in flight.

| Requests in flight | MTP off -> on | Change |
|---|---:|---:|
| 1 | 37.22 -> 49.13 | +32.0% |
| 4 | 21.58 -> 27.03 | +25.2% |

MTP-off values reuse the 2026-10-28 baseline and were not rerun. Stable MTP values are fresh-request streaming measurements from 2026-10-29; the new one-request run is in `tmp/stable-sampling-ctx524288-20261029/streaming-fresh-c1`, and the four-request run is in `streaming-fresh`. Both use cache-only mode with no request priming. The stable runs disable the ordinary startup probe with `--no-warmup`; the stored baseline kept it. These are observed improvements against the retained baseline, not a fully matched fresh A/B experiment. Exact values and source paths are in `mtp-off-on.json` in the same artifact directory.

## MMLU before stable sampling: fresh servers (2026-10-28)

This stored baseline uses `-np 4 -c 524288 --kv-unified`, default batches, cache-only loading, and the original streaming harness with continuous request replacement. No request warmup precedes measurement. The first MTP prompt evaluations include graph imports (77.291s at concurrency 1; about 110.689s for the first four concurrent requests).

<!-- mmlu-current:start -->
| MTP | Requests in flight | TTFT mean / median (s) | Decode t/s per request, excludes prefill | End-to-end aggregate t/s, includes prefill | Draft accepted / generated |
|---|---:|---:|---:|---:|---:|
| Off | 1 | 0.555 / 0.471 | 37.22 | 35.83 | - |
| Off | 4 | 1.800 / 1.781 | 21.58 | 80.36 | - |
| On | 1 | 10.249 / 0.667 | 48.62 | 24.61 | 2626/2913 (90.1%) |
| On | 4 | 57.373 / 57.412 | 22.79 | 25.40 | 2619/2935 (89.2%) |

<!-- mmlu-current:end -->

TTFT is client-observed time to the first counted token. Native decode excludes each request's prompt evaluation but can include stalls during generation. End-to-end aggregate throughput includes prefill, scheduling and client overhead. Four-request native values are per request, not aggregate. First-use requests have not been removed from this baseline.

Artifacts: `/home/zijun/llama.cpp/tmp/mmlu-inplace-20261028`. Source identifiers: fused convolution `da95d1b1c`, sampler fixes `c1ae62002`, OpenVINO `9299cc3d78`; metadata records binary hashes and the then-uncommitted patches. The historical revision-12 cache contained slot-specific draft variants; it is not the current cache format.

## Historical Vulkan / OV server reference (2026-10-23)

These use the same eight MMLU questions and 512 outputs per request, four slots and total context 524288. The OV rows use the earlier rotation implementation. Vulkan rows are the verified full-GPU reruns (`-ngl 99 --fit off`); runs that offloaded experts to host are excluded. OV requantizes weights, while Vulkan uses the stored GGUF weights. These are historical references, not matched current-backend measurements.

Native decode t/s per request; each cell is MTP off -> on:

| Backend/layout | One request: off -> on | Four requests: off -> on |
|---|---:|---:|
| OV split | 4.54 -> 8.62 | 6.47 -> 3.88 |
| OV unified | 36.47 -> 51.35 | 21.73 -> 25.91 |
| Vulkan split | 31.74 -> 48.54 | 16.63 -> 13.87 |
| Vulkan unified | 31.79 -> 47.71 | 16.60 -> 13.29 |

Client TTFT and end-to-end aggregate throughput, including prefill:

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

Main-context Vulkan device/host allocation tallies are 30.54/0.64 GiB split without MTP, 31.55/0.64 split with MTP, 30.92/1.02 unified without MTP, and 31.93/1.02 unified with MTP. They exclude the separate draft context and are not total peak memory. Batched Vulkan 64K single-slot reference: 1013.906s prefill and 19.80 t/s decode; four-active 64K and unified four-active 32K were not completed.

Full raw provenance is retained in `tmp/ud-backend-matrix-20261023` and the original report at commit `71c0bbc4d`. No historical value is substituted for a missing current measurement.

## Reproduction

Run from the main checkout. Set the device, precision and cache settings from [the backend README](../README.md). Populate caches once without cache-only mode, then enable it for timed runs. Use a new output directory rather than overwriting historical results.

```sh
MODEL="$HOME/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf"
BINARY=build/MtpOVPerf/bin/llama-batched-bench
"$BINARY" -m "$MODEL" -c 131072 -npp 1024,8192,32768 -ntg 512 -npl 1
"$BINARY" -m "$MODEL" -c 524288 -npp 1024,8192,32768 -ntg 512 -npl 1,4
"$BINARY" -m "$MODEL" -c 524288 -npp 1024,8192,32768 -ntg 512 -npl 1,4 --kv-unified
```

Add `-b 8192 -ub 8192` for the large-batch comparison. `llama-batched-bench` uses random token IDs and has no MTP mode. Each timing row is a single pass; use natural text and the server for MTP.

`benchmarks/qwen36_ud_batch_size_bench.py` runs/resumes the batch matrix and records memory and provenance. `benchmarks/qwen36_ud_mmlu.py` wraps `scripts/server-bench.py`, reads the local MMLU parquet, keeps four server slots, and selects one/four concurrent clients. Its `UD_PRIME_SUBSETS` option is retained for historical cache preparation and explicit warmup experiments; exhaustive subset priming is not required by stable sampling. `qwen36_ud_inplace_batched.py` and `qwen36_ud_inplace_summary.py` retain the earlier fused-runner and archived-result compatibility. Generated report blocks are snapshots; preserve their provenance when regenerating them.

## Remaining opportunities and limits

- Extend native in-place attention KV fusion to split multi-stream layouts; this remains the dominant split-KV bottleneck.
- Profile long-context decode. At default-batch 32K, single-slot OV is 24.51 t/s versus historical Vulkan 25.99; OV unified four-active is 40.84 aggregate versus Vulkan split 41.53. Vulkan unified four-active 32K is unmeasured.
- Optimize fused convolution computation and memory layouts. Separate convolution showed no consistent unified-KV model advantage and was removed.
- Diagnose unified 32K x four-active large-microbatch prefill before recommending an 8192 microbatch generally.
- Measure further sampling batching/fusion and warmed old/new serving comparisons only when needed. Keeping inactive top-k branches has a small computation cost.

Native GenAI supports chunked prefill; its latency-oriented scheduling can use larger token budgets. That is not evidence that an arbitrarily large llama.cpp microbatch is optimal. Kernel profiles include instrumentation overhead and must not be presented as unprofiled throughput.
