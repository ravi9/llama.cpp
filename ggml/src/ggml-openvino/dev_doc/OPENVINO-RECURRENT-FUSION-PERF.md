# Optimization history and retained evidence

These are historical controls explaining the current design. They are not current default-performance claims. Use [the implementation guide](OPENVINO-RECURRENT-CACHE.md) and [UD benchmark report](QWEN36-UD-MULTISLOT-BENCHMARK.md) for current behavior and measurements. Detailed original reports are in Git at `71c0bbc4d`; raw artifacts remain under the main checkout's `tmp/`.

## Decisions retained in the default

| Investigation | Evidence / outcome |
|---|---|
| Remove single-slot no-rollback shortcuts | Made reset, reorder and writeback generic for multi-slot/MTP; earlier shortcut-based non-MTP rates were not a fair generic baseline |
| Reduce dynamic shape/control overhead | Derive active count from row-map shape, reuse USM-host controls, and pack metadata rather than repeatedly uploading scalars |
| Fuse cache handling and computation | Two Update ops replace separate cache preparation/computation/writeback chains and their intermediate model I/O |
| In-place instead of rotation | Stable caller allocation and staged original-row reads support cycles/duplicates without pointer-swapped cache banks; private output/copy-back remains the compatibility fallback |
| Keep fused convolution | Separate convolution's faster computation did not overcome materialization/reorder overhead in the measured complete path |
| Enable eligible in-place attention KV | Unified KV avoids the split-stream full-cache bottleneck; recurrent cache capacity remains independent of context length |
| Stable sampling translation | One draft graph across active subsets; fixed pure top-k branches and packed row controls eliminate variant-specific first-use work |

## Small-model control (2026-10-16)

Qwen3.5-0.8B F16, Release GPU, `llama-bench -p 0 -n 128 -r 3`, native softplus, requant unset and KV slicing disabled. These measurements used the former three-op implementation; retain them as evidence for control/shape overhead, not an active implementation option.

| Stage | Decode t/s |
|---|---:|
| Initial native cache path | 43.76 +/- 0.10 |
| Active count from row-map shape | 46.82 +/- 0.05 |
| Reusable USM-host scalar controls | 50.23 +/- 0.07 |
| Native full-cache state update | 51.15 +/- 0.15 |
| Output transpose folded into update | 52.60 +/- 0.07 |
| Input transpose folded into gather | 53.75 +/- 0.08 |
| Independent optimized repeat | 56.21 +/- 0.06 |
| Matched Vulkan Release reference | 57.91 +/- 0.17 |

Parameters fell from 184 to 130, kernel submissions from 789 to 606, and intercepted clFinish calls from 164 to 1. Tiny host-to-device uploads fell from 136 calls / 1588 bytes to 8 / 560 bytes. Both versions used 48 remote-USM cache inputs and outputs; merely using USM did not remove shape-induced synchronization. The remaining finish was inference completion. Profiled timings are diagnostic, not throughput baselines.

## Fused versus separate convolution control (2026-10-27)

8192 channels, history 3, four stored rows, one active sequence, FP16 computation/weights and FP32 cache, aliased device-USM input/output. Eight warmups and 20 profiled calls:

| Input tokens | Fused request (ms) | Separate request (ms) |
|---:|---:|---:|
| 1 | 0.068 | 0.158 |
| 1024 | 2.179 | 3.243 |
| 8192 | 18.862 | 26.746 |

At 8192 tokens, ordinary convolution alone took 2.133ms versus 16.269ms for the fused primitive. The separate path also incurred history/QKV materialization and input/output layout conversions, including a 16.364ms final transpose. This supports optimizing the fused kernel/layout rather than assuming that a faster isolated GroupConvolution gives a faster model. The unified 32K default-batch full-model fused/separate four-active decode rates were 40.79/40.72 t/s; fused prefill was faster in that sample. These small decode differences are not a demonstrated repeatable speedup.

## Interpretation rules

- Separate prefill, native decode, end-to-end aggregate throughput and client TTFT. Four-request native decode is per request.
- Preserve model identity. Yamini and UD differ in stored quantization. `q4_asym64_all` follows the INT4/group-64 plus INT8 embedding/output backup policy for supported quantized tensors, but requantizes GGUF values rather than the original floats. They do not become numerically identical models.
- Earlier UD tests kept some dense Q8_0 weights at INT8/group-32; those precision-policy results are superseded by the corrected dense INT4 policy.
- A large context allocation is not a long actual prompt. With eligible in-place/live-prefix KV, allocation size has little effect on the short MMLU workload; long-context attention still has additional work.
- Do not infer internal memory traffic from absence of explicit memcpy, or total GPU/system usage from process RSS alone.
- Do not present historical rotation/separate-convolution flags or closed-PR branch layouts as current configuration choices.

## Provenance

The archived reports contain incremental measurements at `73f61e1cb`, precision-policy work on 2026-10-19, fused-cache investigations on 2026-10-21/22, and the final convolution cleanup `da95d1b1c` / OV `9299cc3d78`. Relevant raw roots include `tmp/recurrent-inplace-20261022`, `tmp/inplace-kv-retest-20261027`, `tmp/fused-conv-cleanup-20261027`, `tmp/batch-size-matrix-20261028`, `tmp/stable-sampling-20261028`, and `tmp/stable-sampling-ctx524288-20261029`. The pre-consolidation documents, including the latest uncommitted follow-up, are copied to `tmp/worktree-cleanup-20261029/dev_doc-before`.
