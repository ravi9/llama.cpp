# Qwen3.6 UD multi-slot benchmark (2026-10-23)

Status: all 16 OpenVINO/Vulkan split/unified server cases and the split
OpenVINO four-slot batched matrix are complete. Remaining matched long-prompt
batched matrices are running. Results are included only after successful
completion and token-count validation.

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

## Batched benchmark

`llama-batched-bench` uses random token IDs and does not run MTP. Each sequence
gets the stated prompt length and 512 decode steps. No prompt sharing is enabled.
All throughput values in its output are aggregate across the active sequences.

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
Vulkan uses the same commands with the clean Release binary path.

Completed split OpenVINO four-slot allocation, including one-active controls:

| Prompt/sequence | Active sequences | Aggregate prefill t/s | Aggregate decode t/s |
|---:|---:|---:|---:|
| 1,024 | 1 | 405.21 | 4.55 |
| 1,024 | 4 | 636.96 | 26.70 |
| 8,192 | 1 | 696.17 | 5.08 |
| 8,192 | 4 | 798.97 | 20.93 |
| 32,768 | 1 | 588.60 | 4.04 |
| 32,768 | 4 | 627.16 | 11.10 |
| 65,536 | 1 | 455.76 | 2.73 |
| 65,536 | 4 | 448.62 | 5.20 |

Raw log: `tmp/mtp-regression-20261022/pool-v2-np4.log`. These low one-active
results are measured with four physical split cache streams allocated and the
current full-cache binding fallback. They are not the one-slot baseline.

Completed Vulkan one-slot matrix (`-npl 1 -c 131072`):

| Prompt/sequence | Active sequences | Aggregate prefill t/s | Aggregate decode t/s |
|---:|---:|---:|---:|
| 1,024 | 1 | 683.62 | 32.49 |
| 8,192 | 1 | 436.10 | 30.59 |
| 32,768 | 1 | 128.25 | 25.99 |
| 65,536 | 1 | 64.64 | 19.80 |

Raw log: `tmp/ud-backend-matrix-20261023/batch-vulkan-split-npl1.log`.
The 64K prefill takes 1,013.906 seconds, versus 25.865 seconds for decode.
The four-slot Vulkan split/unified matrices and remaining OpenVINO matrices
are still running; partial rows are excluded from final matrix tables.

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

Completed server matrix:

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

These diagnostics occurred after the timed server matrix, so their individual
placement values cannot be retroactively assigned to timed cases. They do show
that default fitting can change the workload with the driver-reported budget,
which varied substantially across successive process starts. The lower Vulkan
server rates must be interpreted as configured-default end-to-end results, not
a controlled all-GPU kernel comparison. `llama-batched-bench` loads the model
and context directly and does not call `common_fit_params`; its results do not
have this server-specific automatic-fit confound. A separate server control
with `UD_FORCE_GPU=1` (explicit `-ngl 99 -fit off`) is queued after the requested
matrix; it will be reported separately from the default-fit results.

This is an eight-request throughput sample, with one pass per case. MMLU question
texts average 53.25 templated tokens in this sample. It tests MTP on natural short
prompts and separately tests long-context non-MTP with random token IDs; it does
not measure long-context MTP, MMLU accuracy, or statistical variation across
repeated benchmark passes.
