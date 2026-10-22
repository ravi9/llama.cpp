# Qwen3.5/Qwen3.6 generic recurrent fusion performance

2026-10-21. This work supports Qwen3.5 and Qwen3.6 (`qwen35`/`qwen35moe`) in
llama.cpp's OpenVINO backend, including multi-slot llama-server and draft-MTP
verification/rollback. It implements the two computation/cache boundaries proposed
in [the recurrent-cache design](OPENVINO-RECURRENT-CACHE.md). No stateful or
single-slot/no-rollback shortcut is used.

## Implementation

Each recurrent layer uses two native GPU operations on the matched graph:

1. Nine-input `RecurrentConvCache`: logical reset, active history read, depthwise
   convolution, reorder, and ordered history-window writes. It returns convolution
   output plus the full updated `cache_r`. The existing eight-input history-only
   form is retained.
2. `RecurrentGatedDeltaNet`: logical reset, active state read, GDN computation,
   reorder, and ordered final-state/MTP snapshot writes. It returns attention plus
   the full updated `cache_s`. The existing six-input GDN is retained.

All metadata remains runtime input: active/extra row maps, reset ranges, source and
destination offsets, and ordered write controls. Original cache reads are immutable;
cyclic/duplicate maps are valid. Later overlapping writes have the original ordered
precedence. Only the initialized newest `min(snapshot_count,T)` states participate
in MTP writeback. The frontend requires the exact valid-prefix selection before
fusing snapshot mode. Cache layout is `[head,value,key]`, avoiding standalone state
transpose/gather/update intermediates.

A GPU precision pass keeps F32 cache inputs and outputs while selecting the normal
FP16 computation policy. Kernels explicitly retain the former computation-precision
rounding, including preserved rows. Convolution accumulates in FP32. For long
prefill, adjacent convolution lanes read adjacent tokens from channel-major QKV;
short batches favor contiguous channel output writes. GDN vectorization and moving
snapshot destination checks out of per-element
writes were also evaluated. They improved the isolated graph but the first UD
non-MTP retest slowed down, so they were reverted because a full-model benefit was not established.
Grouping multiple preserved rows into each subgroup
was tried and removed because it provided no clear additional improvement.

`RecurrentStateGather(prepare_cache=false)` and seven-input `RecurrentStateUpdate`
provide the preceding, independently usable prepared-cache elimination. Capability
checks keep older runtimes on the existing path. CPU/static/stateful/split translation
paths are outside this fusion. Compiled-model cache revision is **8**.

## Matched server measurements

Device: Intel Arc B390 (PTL), Core Ultra X7 358H. Release `build/MtpOVPerf`,
OpenVINO `/home/zijun/openvino-build-mtp`. llama.cpp base HEAD `049f55449` plus
local backend changes; unrelated existing worktree changes were preserved.

Target: `~/models/Qwen3.6-35B-A3B-MTP-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`.
Environment:

```sh
GGML_OPENVINO_DEVICE=GPU
GGML_OPENVINO_NATIVE_SOFTPLUS=1
GGML_OPENVINO_DISABLE_KV_SLICE=1
GGML_OPENVINO_REQUANT_KQUANT=q4_asym64_all
```

Server arguments are `-np 4 -c 8192 -t 2 -tb 2`. One active `/completion` request
uses exactly **1,024 saved prompt token IDs**, `temperature=0`, `top_k=1`,
`n_predict=512`, `ignore_eos=true`, `cache_prompt=false`, and nonstreaming output.
An eight-token warmup uses the same prompt. MTP adds
`--spec-type draft-mtp --spec-draft-n-max 2`. GPU workloads run serially. Disk
compiled-model caching and stateful execution are unset. Load/requantization,
compilation and warmup are excluded from server generation timings.

The final comparison uses a control from the same current build with both
`GGML_OPENVINO_DISABLE_CACHE_PREPARE_FUSION=1` and
`GGML_OPENVINO_DISABLE_FUSED_RECURRENCE=1`. This preserves the original native
cache ops, GDN and convolution, while disabling this round's transformations.
The repeated control averages **26.497 tokens/s** decode and **905.70 tokens/s**
prompt processing. The final fused/device-USM build averages **27.631 tokens/s**
decode and **912.09 tokens/s** prompt processing: **4.28% faster decode** than the
repeated control, reducing the average token cost from about **37.74 to 36.19 ms**.
This remains below the requested 30 tokens/s target. The primary content
matches the control exactly, as do the two fused repeats. All four concurrent
functional requests completed 96 output tokens.

| Final build / mode | Decode tokens/s, three requests | Mean decode tokens/s | Mean prompt tokens/s |
| --- | ---: | ---: | ---: |
| Both new fusions disabled | 26.529 / 26.509 / 26.453 | 26.497 | 905.70 |
| Two-op fusion, device-USM cache staging | 27.702 / 27.614 / 27.577 | 27.631 | 912.09 |
| Two-op fusion, device-USM, two-token MTP | 43.450 / 43.353 / 42.528 | 43.110 | 651.87 |

The final non-MTP comparison keeps the same pre-vector GDN kernel. The disabled
control and fused run share the new runtime allocation policy, but the control
has no fused cache port to select it. This comparison measures the combined
translation/kernel/allocation change; it does not isolate the device-USM policy's
individual contribution. MTP is 56.0% faster than final non-MTP on this particular
prompt, with **340/340 draft tokens accepted in each request**. This is a
high-acceptance example, not a general MTP speedup estimate. Its primary content
matches non-MTP. The four varied concurrent cases complete 96 tokens each and
accept 47/94, 58/74, 55/79 and 54/81 proposals. The whole-model transfer probe
passed with a rebuilt probe linked to the current OpenVINO 2026.5.0 runtime;
its transfer evidence is detailed below. An earlier probe attempt linked the stale
2026.4.1 runtime alongside the server's current runtime and crashed during startup;
it is excluded from performance evidence.

Earlier stages are recorded below to expose variation across kernel/runtime changes;
they are not substitutes for the final repeated control.

| Earlier stage | Decode tokens/s | Prompt tokens/s |
| --- | ---: | ---: |
| Early disabled-fusion control, one request | 25.216 | 889.18 |
| Initial computation fusion, before coalesced convolution reads | 27.662 | 622.18 |
| Coalesced convolution, three requests | 27.836 / 27.659 / 27.650 | 925.43 / 914.67 / 913.28 |
| MTP, coalesced convolution, three requests | 39.974 / 40.240 / 40.086 | 480.96 / 674.80 / 583.10 |
| Vectorization experiment, non-MTP | 25.818 / 25.893 / 25.908 | 945.45 / 939.82 / 935.92 |
| Vectorization experiment, MTP | 40.427 / 40.098 / 40.866 | 511.95 / 672.78 / 668.92 |
| Vectorization reverted, before device-USM staging | 23.608 / 23.540 / 23.621 | 882.17 / 864.90 / 870.09 |

Coalesced convolution restored prompt processing after the initial fusion's prefill
regression. The early repeated non-MTP mean was 27.715 tokens/s. The subsequent
vectorization retest averaged 25.873, but reverting it then measured 23.590: the
continued drop does not establish vectorization as its cause. The retained build
uses the pre-vector GDN kernel; the final repeated control is the basis for assessing
its combined changes. Initial fused and early disabled-fusion content is identical.

Every measured MTP request in the table accepted **340/340 draft tokens**. The
saved prompt is repetitive, so these are high-acceptance results, not a representative
MTP speedup claim. MTP prompt rates remain lower and more variable than non-MTP;
they include draft-context processing and snapshot paths. Four varied concurrent
prompts exercised rejection: the coalesced run accepted 47/94, 58/72, 54/80, 54/81,
and the vectorized experiment accepted 47/94, 58/74, 55/79, 54/81. No current-build
control with MTP and these transformations disabled was measured.

Historical 2026-10-20 server results were 31.107 tokens/s at explicit `-np 1`,
26.4955 at explicit `-np 4`, and 27.2554 with implicit four slots/unified KV.
They explain the original investigation but are not matched controls for this
worktree. Do not compare those directly to attribute the entire difference to this
patch. Context/slot cache storage and the number of active sequences are separate
variables. The logs allocate four recurrent rows (251.25 MiB cache_r/cache_s)
without MTP and twelve rows (753.75 MiB) for two rollback tokens with MTP.
Thus snapshot mode triples recurrent cache storage even at one active request.

## What profiling establishes

The full-model execution graph contains **30 `RecurrentConvCache` and 30
`RecurrentGatedDeltaNet` operations**, replacing separate gather/GDN/update cache
stages. The 0.8B graph has 18 of each. Fewer nodes alone do not establish lower
latency or zero-copy execution.

The full-model CSV diagnostics below are sums of OpenVINO **cumulative average**
operator times, including preceding prefill calls. In particular, prefill strongly
contaminates fused computation times. These are graph/traffic clues, **not pure
decode costs**. Profiling substantially slows throughput and was disabled for the
server results.

| Operator class, summed diagnostic ms | Before np1 | Before unified np4 | Prepared-cache elimination np4 | Initial computation fusion np4 |
| --- | ---: | ---: | ---: | ---: |
| RecurrentStateGather | 1.139 | 2.220 | 0.929 | 0 |
| RecurrentStateUpdate | 0.902 | 1.815 | 2.075 | 0 |
| RecurrentConvCache | 0.585 | 0.682 | 0.694 | 4.084 |
| GatedDeltaNet | 3.771 | 3.842 | 3.802 | 0 |
| RecurrentGatedDeltaNet | 0 | 0 | 0 | 6.613 |
| Convert | 1.251 | 2.487 | 2.484 | 0.124 |
| Result | 0.468 | 1.105 | 1.110 | 0.005 |

The before np1/np4 profiles show similar GDN arithmetic times but substantially
larger gather/update/precision traffic at four slots. The generic full-cache
contract causes work proportional to stored rows even when one sequence is active.
The fused profile's Result average is not proof that runtime output copying vanished.

### Isolated GDN/cache experiment

A standalone dynamic graph uses the target state geometry: 32 value heads,
16 key heads, K=V=128, F32 device-USM cache, FP16 Q/K/V/gate/beta, one active
sequence, no reset/reorder, and either 4 or 12 allocated rows. The legacy reference
explicitly materializes the original FP16 cache/gather/GDN/update/F32 writeback.
The row-count comparison isolates storage work, excluding MoE, quantization, full attention
or llama-server scheduling. Fifty inferences follow eight warmups; all data tensors
are device USM and metadata is constant. These microbenchmarks are not full-model
latency estimates.

| Snapshot count / tokens / rows | Materialized, caller alias ms | Initial fused, caller alias ms | Vectorized fused, caller alias ms |
| --- | ---: | ---: | ---: |
| 1 / 1 / 4 | 0.402 | 0.319 | 0.313 |
| 1 / 1 / 12 | 1.081 | 0.881 | 0.840 |
| 3 / 3 / 4 | 0.575 | 0.324 | 0.278 |
| 3 / 3 / 12 | 1.260 | 0.869 | 0.800 |

A separate profiling run at 3 snapshots / 3 tokens / 12 rows records about
**0.30 ms in fused GDN and 0.38 ms in the cache Result reorder/copy kernel**.
A distinct output allocation has similar costs to caller input/output aliasing.
Thus this remaining copy cannot be attributed solely to caller aliasing. Runtime
execution-graph port metadata reports both GDN outputs using the first port's
precision/shape, while the actual caller cache output is F32 with the full cache
size. Changing the cache row dimension from static to dynamic eliminates the Result
kernel in a follow-up micrograph, while end-to-end latency remains approximately
0.86 ms with caller aliasing. This establishes a static/dynamic output-ownership
difference, not a universal Result-copy cause in llama.cpp (whose caches use
dynamic rows). A per-port precision override in the op factory did not remove the
static-graph copy and was reverted.

OpenCL interception in this isolated graph finds two kernel submissions and one
small host-USM→device upload (24,576 bytes at T=3); it finds no explicit large
cache `clEnqueueMemcpyINTEL` transfer. That does **not** mean no cache copy: the
static-graph Result kernel performs a full device-memory pass. The dynamic-rows
case shows **two host-USM→device copies totaling 25,190,400 bytes per call**:
24,576 bytes of ordinary staging plus the 25,165,824-byte cache. Thus the dynamic
output's internal allocation is host USM even with a device-USM caller destination.
This is direct evidence of cache staging traffic, missed by operator-only profiling.
Ordinary llama.cpp caller cache buffers already use device USM. The runtime output
policy is changed for the fused cache port to
allocate device USM, retaining lockable memory when CPU consumers require it.
A rebuilt probe linked only to OpenVINO 2026.5.0 confirms one 25,165,824-byte
**device→device** copy plus the 24,576-byte ordinary upload. This removes host
cache staging but retains the full-cache copy to the caller. The current-runtime
micrograph with caller aliasing averages **0.779 ms** for 50 measured calls;
Using a distinct output averages 0.885 ms in the same diagnostic run. The final
whole-model comparison above measures the combined patch; it does not isolate
this allocation change.

### Whole-model device-memory transfer check

A separate shortened UD non-MTP run uses the final backend, `-np 4 -c 8192`,
1,024 saved prompt tokens, an eight-token warmup, and 16 output tokens. It uses
OpenCL/InferRequest interception, not operator profiling. Throughput from this
short instrumented run is not used in the performance table.

All 60 recurrent cache input ports and 60 recurrent cache output ports bind USM
buffers. Exact cache-root names account for **263,454,720 bytes (251.25 MiB)**
on each side: 11,796,480 bytes for `cache_r` and 251,658,240 bytes for `cache_s`.
Each warmed one-token call records:

| Intercepted operation | Calls | Bytes |
| --- | ---: | ---: |
| Device→device memcpy | 60 | 263,454,720 |
| Ordinary host→device metadata uploads | 8 | 2,608 |
| Host-USM→device uploads | 20 | 320 |
| Host-USM→host output copies | 3 | 1,009,664 |
| NDRange submissions | 1,020 | — |
| `clFinish` | 21 | — |

The large memcpy volume matches the 60 final cache outputs exactly. There is no
large host-USM→device cache transfer in the current-runtime probe. The execution
graph contains 30 recurrent-cache primitives and 30 GDN primitives for the fused
ops. It has no separate RecurrentStateGather/StateUpdate stage. This confirms
that reducing those nodes does not remove runtime full-output copying.

In the last twelve instrumented calls, `infer()` takes about 33.5–33.6 ms,
including 14.2–14.8 ms spent inside `clFinish`. Those waits synchronize queued
GPU work, so their durations are **not additional kernel costs** and cannot be
added to operator times or attributed solely to cache copies. The graph still
submits 1,020 kernels for all layers, projections, MoE, attention and conversions;
two recurrent ops per layer does not mean two kernels for the whole layer.

Artifacts are `device-cache-probe-rebuilt/`, `micro-probe-current/`, and
`transfer-summary.json`. The initial full-model probe used a stale shared-library
soname and crashed before inference; the successful rebuilt probe supersedes it.
The earlier host-staging microprobe also used that older soname and should be
regarded as diagnostic evidence of the allocation path; the final D2D copy counts
above are verified with the current runtime.

## Validation

Both repositories' final builds and `git diff --check` pass. Native core
shape/type/clone/attribute suite: **9/9 pass**. Native GPU suite: **8/8 pass**,
including original APIs, deferred preparation, fused convolution versus GroupConv,
fused GDN versus materialized reference, export/import, mixed FP16/F32, cyclic and
duplicate maps, reset-before-gather, source offsets, ordered overlaps, empty window
lists and device-USM caller input/output aliasing. Both K128 and fallback K8/V7
GDN paths are covered. Final prefill extensions exercise T=16/17/33, and short
T=1/2 snapshot batches select only initialized states. An additional checked-in
transformation regression passes, ensuring six-input GDN fusion passes leave the
cache-aware runtime-input contract intact.

Small-model `test-backend-sampler --test recurrent_slots --device gpu` passes
40 comparisons across four sequences with reset/copy/reuse and rollback 0/2;
maximum reported relative logit error is below 0.003. Full UD non-MTP and MTP
server runs each include four concurrent varied prompts, all completing 96 tokens.
The final MTP acceptance is 47/94, 58/74, 55/79 and 54/81 in those cases,
exercising rejection and rollback. Their per-request concurrent rates are functional evidence and are
not comparable to single-active throughput.

CPU/static/stateful execution was not tested in this round. The fusion is GPU
stateless only; no new support for those modes is claimed.

## Remaining optimization opportunities

1. The dynamic full-cache output copy is addressed by the follow-up below;
   the initial opportunity was to eliminate the remaining copy to the caller.
   Static cache output uses
   a Result kernel; dynamic cache output uses runtime memcpy after inference.
   Internal staging is now device USM, but both paths still move the full cache.
   Caller aliasing still requires
   immutable original reads: use device buffer rotation or correct staging rather
   than unsafe in-place cyclic reorder. Validate changing shapes, exported graphs,
   overlapping windows and four-slot MTP rollback before accepting a speedup.
   A concrete implementation needs output-port ownership, not just an op-wide
   output-buffer flag: keep cache port 1 in the caller's device allocation when
   it is distinct from all reads; for aliases, rotate two device cache buffers and
   bind the previous result as the next input. The latter also requires updating
   GGML root/view ownership so checkpoints and sequence operations read the current
   allocation. Multiple live cache readers must finish before storage reuse.
2. Reduce full-cache preservation traffic. Fusion removes intermediates but still
   reads/writes inactive rows. A generic runtime/cache representation that maintains
   untouched rows without a full materialization could save more bandwidth; it must
   support arbitrary runtime reset/reorder and snapshots, not a single-slot exception.
3. Remove repeated precision roundtrips only after defining and validating the desired
   cache numerical contract. Preserving F32 inactive rows changes the old FP16 GPU
   behavior and can alter logits/acceptance, so it is not a free optimization.
4. Improve long-token GDN/snapshot prefill and measure draft-prefill separately.
   MTP prompt timings are still lower than non-MTP. Use fresh steady-shape profiling,
   not mixed prefill/decode cumulative averages, to select the next kernel change.
5. Benchmark representative MTP prompts and compare synchronized T1/T3 trunk and
   draft latencies after cache fixes. Remaining MoE/weight/attention work also matters;
   node fusion alone does not guarantee the non-MTP 30 tokens/s target.

## Reproduction and artifacts

Scratch directory: `tmp/recurrent-fusion-20261021/`. `bench.py` wraps the saved
server harness in `tmp/server-ud-20261020/run_benchmark.py`; `kernel_bench.cpp`
and its before/vector/profile logs contain the isolated experiment. `profile-summary.json`
and per-stage CSV45 files contain the cumulative-average diagnostics. Correctness
logs and final build logs are in the same directory.

Use response `timings.predicted_per_second` and `timings.prompt_per_second`.
The harness's outer `wall_seconds` includes the two repeat requests and four-slot
checks before returning the primary response; it must not be used as that request's
throughput. Metrics are annotated with this caveat.

## Follow-up: direct cache output binding (2026-10-21)

The subsequent optimization implements caller-owned cache output port 1 and
current/spare device-USM rotation. GGML root addresses stay stable; tensor I/O,
views, checkpoints and clears resolve buffer-owned current storage. A successful
inference commits the swap. The plugin only borrows a compatible dynamic Result
destination disjoint from its inputs; switching to an aliased destination detaches
the borrowed bank and restores staging. The backend opt-out is
`GGML_OPENVINO_DISABLE_CACHE_ROTATION=1`. Static/stateful/split and partial-root
cases retain the existing path. Extra device capacity is one full recurrent cache
set: 251.25 MiB without MTP or 753.75 MiB for two rollback tokens at four slots.
This storage replaces the need for private cache-output staging on the direct path.

The native suites pass 9 core, 8 GPU, and 1 transformation tests. Extended GPU
cases alternate distinct and aliased caller outputs on one request and check
unchanged distinct-input cache contents, dynamic B/T, cyclic/duplicate maps, reset,
ordered overlaps, short snapshots, long prefill, and export/import. The small-model
slot test now passes **50 comparisons**, including checkpoint export/import and
full cache clear with rollback 0/2.

A current-runtime intercepted dynamic GDN micrograph (T=3, S=12, one active
sequence, K=V=128, 32 value heads) has **zero large D2D copies** for distinct
outputs. Aliased fallback still makes one 25,165,824-byte copy. An unprofiled run
measures 0.488 ms distinct and 0.763 ms aliased; this isolated comparison does not
include other model operations. The small 0.8B server likewise has zero large D2D
cache copies in warmed calls, with 474 NDRange submissions and one clFinish.
Its instrumented short output is functional/traffic evidence, not a benchmark.

### Computation kernel comparison

The fused GDN uses the ordinary `gated_delta_net_ref.cl` recurrence, including
K128 float8 per-lane state and four value columns per subgroup. It does **not**
use the newer paged GDN optimized subgroup block-read/write path. The latter
distributes keys interleaved across subgroup lanes for contiguous transactions;
changing the key distribution also changes floating-point reduction order.
Vector cache load/store and hoisting snapshot destination calculations can preserve
the existing arithmetic while reducing addressing and memory instructions.

Fused convolution computes a direct FP32 dot product over H+1 taps per output.
The separate GroupConvolution selects a device convolution implementation; paged
causal convolution instead keeps history/weights in registers across a sequence's
tokens. Those are different schedules. Register reuse across short token batches
and explicit expansion of the JIT-constant tap loop are candidate improvements;
long prefill must retain sufficient parallelism and coalesced channel-major reads.
Neither fused kernel should be described as already equivalent in performance to
the optimized paged operators. Full-cache preserved-row traffic remains after
copy removal, alongside MoE/projection and ordinary attention costs.

Follow-up artifacts: `tmp/recurrent-rotation-20261021/`. The matched server harness
uses the same UD model/config, saved 1,024-token prompt, 512 outputs, warmup,
three repeated single-active requests and four varied concurrent slot checks.

The direct-binding follow-up's matched unprofiled non-MTP control disables only
rotation, keeping the same fused arithmetic and runtime build. All requests
evaluate exactly 1,024 prompt tokens and emit 512 outputs.

| Rotation setting | Decode tokens/s, three requests | Mean decode tokens/s | Mean prompt tokens/s |
| --- | ---: | ---: | ---: |
| Disabled, private device-USM staging | 28.107 / 28.098 / 28.027 | 28.077 | 758.09 |
| Enabled, direct cache port 1 | 29.148 / 28.965 / 28.651 | 28.921 | 924.84 |

This is **3.01% higher decode throughput**, about 35.62 to 34.58 ms per
output token. It remains below 30 tokens/s. All primary/repeated content matches
across modes; four concurrent varied requests complete 96 tokens each. Prompt
timings in the control vary considerably and should not be used to isolate
prompt-processing benefit from this short sample.

Rotation-enabled MTP measures **47.873 / 48.218 / 47.715 tokens/s**, mean
**47.935**, with mean prompt processing 804.76 tokens/s. All three requests
accept 340/340 draft tokens and their content matches non-MTP. Four concurrent
varied requests complete 96 tokens each with acceptance 49/90, 54/82, 55/79,
and 54/81, exercising rejection and rollback. This is not a matched MTP
rotation-disabled comparison; prior staging MTP averaged 43.110 in an earlier
round, so that historical difference alone does not isolate rotation's effect.

The full UD transfer probe confirms **zero large D2D cache copies** in each
warmed one-token call, removing all 60 copies / 263,454,720 bytes. Ordinary
uploads remain 8/2,608 bytes plus 20/320 bytes, with 1,020 NDRange submissions
and 21 clFinish calls unchanged. Twelve calls average 30.629 ms raw instrumented
infer versus about 33.6 ms in the prior staging probe; these separate shortened
instrumented runs are traffic evidence, not the matched throughput comparison.
The independent-input alias guard also passes the 8 GPU tests and leaves
the isolated distinct GDN latency unchanged (0.491 ms versus 0.491 ms before).

An unprofiled standalone convolution/cache micrograph (C=8192,H=3, one active
sequence, F32 cache and FP16 computation, distinct device outputs) measures:

| T / stored rows | Cache preparation + GroupConv ms | Fused convolution/cache ms |
| --- | ---: | ---: |
| 1 / 4 | 0.066 | 0.067 |
| 1 / 12 | 0.125 | 0.089 |
| 3 / 4 | 0.119 | 0.063 |
| 3 / 12 | 0.092 | 0.070 |
| 128 / 4 | 0.226 | 0.302 |
| 128 / 12 | 0.251 | 0.311 |

The fused three-token path is competitive, but long-prefill arithmetic scheduling
needs improvement. This comparison includes cache preparation/output ownership
differences and is not a pure convolution-kernel comparison.

A follow-up compute experiment retains GDN's per-lane key distribution and
arithmetic/reduction order, using contiguous eight-element cache loads/stores
and calculating each snapshot destination once per timestep. It passes the
eight native GPU tests with exact GDN/reference state comparison. The same
isolated T3/S12 graph falls from 0.491 to **0.419 ms** for distinct outputs and
0.765 to 0.714 ms for aliased outputs. Explicit unrolling of the JIT-constant
convolution tap loop helps some short shapes (T1/S4 0.067 to 0.052 ms);
T128 stays near 0.302/0.313 ms. This does not resolve long-prefill convolution
scheduling. These short microbenchmarks motivate a full-model retest; they are
not evidence of a proportional model throughput gain.

The full UD non-MTP compute experiment subsequently measures
28.454 / 27.903 / 27.460 tokens/s (mean **27.939**) with unchanged primary
content, versus rotation-only mean 28.921. Prompt mean is 962.45 tokens/s.
Throughput declines across repetitions, so this does not isolate the experiment
as the cause; however, it does not establish a full-model decode benefit. The
experiment is excluded from the retained optimization pending stronger evidence.

The MTP compute experiment also provides no established decode benefit:
47.677 / 47.564 / 46.849 tokens/s, mean **47.364**, versus rotation-only
47.935. Output matches and all requests accept 340/340 tokens. Both vector
GDN and convolution-unrolling experiments are **reverted** in the retained
build. Their source and measurements remain under the scratch directory.
A different GDN memory schedule and a long-prefill convolution schedule remain
useful opportunities, but isolated speedups must be confirmed in matched model
runs before adoption.

Final retained sources restore the original computation kernels, retain direct
cache binding with whole-network input-overlap checks, and correct shared-view
wrapper resolution through the base tensor. Both builds and diff checks pass;
the final native GPU suite passes 8/8 and the final small-model slot/checkpoint
suite passes all 50 comparisons.

The final restored-kernel confirmation (including shared-view correction and
whole-network input-overlap guard) measures **29.013 / 27.983 / 27.103
tokens/s**, mean **28.033**, with mean prompt processing 933.60 tokens/s.
All primary/repeated output matches rotation-only/control, and four concurrent
requests complete 96 tokens. It shows the same declining-throughput pattern as
the compute experiment after its kernels were reverted. Thus the earlier 3.01%
rotation comparison is a measured sample, **not an established sustained gain**
for the final build; nor can the compute experiment's decline be confidently
attributed to its kernel changes. Copy removal is independently verified, but
the desired stable 30 tokens/s non-MTP target is not achieved. GPU power/clock,
thermal state and other run variation were not captured; those are possible
confounders, not proven causes. Keep the simpler original compute kernels until
a controlled repeated comparison establishes a model-scale benefit.

Next concrete kernel work should compare fresh steady T1/T3/T128 profiles after
copy removal, separate active recurrence from preserved-row traffic, then test
a coalesced GDN state schedule and a tiled convolution prefill schedule. Capture
GPU frequency/power alongside interleaved controls. Retain arbitrary reset,
reorder, snapshot and immutable-read semantics throughout.

## 2026-10-22: MTP prefill path and Vulkan comparison

The retained implementation still uses two native operations per recurrent layer,
with 30 of each in the target trunk. `RecurrentConvCache` fuses reset/history
read/convolution/reorder/window writeback; `RecurrentGatedDeltaNet` fuses
reset/state read/GDN/reorder/snapshot writeback. Their second outputs are the
complete updated caches. Current/spare device-USM binding removes the separate
large output-to-caller cache copies, but the kernels still read and materialize
preserved cache rows. This is generic stateless execution, including multiple
slots, arbitrary runtime metadata and rollback.

The computation schedules have not been replaced with the optimized paged
kernels: GDN uses the reference K=128, float8, four-value-block recurrence with
a sequential token loop, and convolution uses direct FP32 accumulation over
the history taps. Long-prefill convolution has coalesced token reads. The later
vector cache load/store and convolution-unrolling experiments remain reverted;
their isolated improvements did not establish a whole-model decode benefit.

### What enabling MTP adds during prefill

For Qwen3.5/Qwen3.6 in llama.cpp, the single trained MTP block has a separate
context and attention cache. It is reused autoregressively to produce two
proposals; there are not two independent trained MTP blocks. During prefill:

1. The target evaluates the prompt and exports every `h_nextn` hidden row.
   `llama_set_embeddings_nextn(ctx_tgt, true, false)` requests unmasked rows.
   `src/llama-context.cpp` extracts them into a host F32 buffer through
   `ggml_backend_tensor_get_async`. At width 2048, 1024 tokens total **8 MiB**.
2. `common_speculative_impl_draft_mtp::process` pairs each token with the
   preceding target hidden row, copies these rows into the draft batch, and runs
   `llama_process(ctx_dft, DECODE, ...)` over the processed prompt. This catches
   up the separate MTP cache and executes its projection, attention and MoE.
   Prompt catch-up does not request vocabulary logits for every position.
3. The target recurrent graph also enables rollback storage throughout prefill.
   With two draft tokens, `build_recurrent_attn` requests three snapshot slots
   and writes only the newest `min(T,3)` states; convolution similarly keeps
   three windows. Four allocated recurrent rows become twelve, increasing the
   cache payload from **251.25 to 753.75 MiB**. The fused full-cache contract
   therefore retains extra preservation/writeback work even during prefill.

Both modes currently default to fused GDN. The slowdown should not be described
as non-MTP choosing chunked GDN while MTP alone switches to sequential GDN.
Nor does snapshot mode save all 1024 intermediate states: it keeps only the
newest three. Export/catch-up and snapshot/cache work are established extra
operations; their individual contributions to prompt latency have not yet been
isolated with phase-specific timing.

The response JSON `timings.prompt_ms` ends at first-token sampling:
`server-context.cpp` updates `slot.stats` after `common_speculative_process`.
It includes target prefill, draft catch-up, hidden transfers, checkpoint work
and first sampling. The earlier `metrics_post_decode` update precedes draft
catch-up, so aggregate server metrics have a different timing boundary. Use
response timings consistently for this comparison. Cold compilation/first-use
effects are separate from warmed inference; one short warmup does not prove all
subsequent shape/state costs are warm.

### Measurements

Vulkan was rebuilt as **Release** in `build/VulkanPerf08` from the current
worktree (base HEAD `049f55449`). The matched server uses the same UD GGUF,
saved 1024-token prompt, `-np 4 -c 8192 -t 2 -tb 2`, batch 2048/ubatch 512,
automatic flash attention, one active request, eight-token warmup and three
512-output requests. Every measured response reports `cache_n=0`,
`prompt_n=1024`, `predicted_n=512`. GPU tests run serially.

The local Vulkan source contains an unconditional `dump_cgraph` call on every
graph execution. A benchmark-only `LD_PRELOAD` shim replaces that function with
a no-op, leaving source files unchanged. No graph dump is produced in these
primary runs. This avoids including diagnostic file formatting/I/O in the
comparison. Initial runs with dumping enabled are retained as diagnostics:
non-MTP averaged 504.64 prompt / 21.60 decode tokens/s, MTP 500.96 / 37.05.
Decode varied substantially without the shim; these separate runs cannot
attribute that entire difference solely to dumping.

| Backend / mode | Prompt tokens/s, three requests | Mean prompt tokens/s | Mean prompt ms | Mean decode tokens/s |
| --- | ---: | ---: | ---: | ---: |
| OV, rotation-only non-MTP, previous round | 932.34 / 922.52 / 919.67 | 924.84 | 1107.25 | 28.921 |
| OV, rotation-only MTP, previous round | 675.62 / 865.54 / 873.12 | 804.76 | 1290.51 | 47.935 |
| OV, final restored-kernel non-MTP, previous round | 940.44 / 934.08 / 926.27 | 933.60 | 1096.88 | 28.033 |
| Vulkan, graph dump disabled, non-MTP | 523.23 / 522.04 / 521.28 | 522.18 | 1961.00 | 31.281 |
| Vulkan, graph dump disabled, MTP | 499.93 / 502.76 / 504.07 | 502.25 | 2038.85 | 46.325 |

Rates are arithmetic means of per-request rates. OV's rotation-only comparison
has about **13.0% lower mean prompt throughput** with MTP; comparing just its
later two requests gives about **5.6%**, with approximately 66 ms extra mean
prompt time. The first measured MTP request takes 1515.64 ms versus
1183.08/1172.81 ms later, so it materially affects the three-request average;
its precise cause was not instrumented. Vulkan's mean prompt throughput drops
about **3.8%**, adding about 78 ms. Thus additional MTP prefill work exists in
both backends, while this sample does not establish how much of OV's larger
average drop belongs to kernels, transfers, or residual first-use costs.

Vulkan MTP accepts **323/376** proposals in each measured request versus
OV's **340/340** on this repetitive passage. Each mode's repeated content is
identical. These are the same GGUF but different internal weights: OV uses
`q4_asym64_all` requantization and Vulkan uses the original mixed GGUF types.
They also use different arithmetic. Decode and acceptance are therefore not
a controlled comparison of exactly the same numerical model. OV numbers are
from the preceding round, not newly interleaved controls.

A separate Vulkan `llama-bench -p 1024 -n 0 -r 3` with the same Release build,
model, default batch/ubatch and graph-dump shim measures **703.79 +/- 6.55
tokens/s** (samples 711.19/698.74/701.44). This is raw non-MTP trunk prefill
with llama-bench's prompt and single-sequence context, without server checkpoint
and first-sampling overhead; it is not the server's 522 tokens/s metric.
The server creates two context checkpoints per request. Their logged durations
can include waiting for asynchronous inference, so they must not be interpreted
as isolated checkpoint-copy costs.

Artifacts: `tmp/prefill-comparison-20261022/`, including `comparison.json`,
effective server commands, response JSON, repeated results, logs, raw benchmark
output and the benchmark-only graph-dump shim. The next prefill investigation
should time target inference, hidden extraction, draft catch-up and checkpoint
synchronization separately before assigning the gap to a particular kernel.


## Independent in-place cache ops (2026-10-22)

The opt-in `GGML_OPENVINO_RECURRENT_CACHE_INPLACE=1` adds
`RecurrentConvCacheUpdate` and `RecurrentGatedDeltaNetUpdate` for
Qwen3.5/Qwen3.6 in llama.cpp. The corrected non-MTP path measures **29.001
rather than 27.875 decode tokens/s (+4.0%)**, with practically unchanged prompt
throughput. With two-token MTP it measures **45.014 rather than 46.345 tokens/s
(-2.9%)**, despite eliminating full-cache copies. The default remains
immutable-source cache rotation.

This round adds independent op classes, registration, a GPU primitive and
kernels, type/serialization tests, and operation specifications. Existing
OpenVINO op contracts and computation kernels receive no further modifications.
Shared plugin code adds safe output binding and cache precision preservation
for the new primitive; earlier local work on existing ops is preserved.

### Implementation and buffer ownership

Each recurrent layer emits two ops with computation output 0 and full-cache
output 1. Eligible Parameter and Result bindings share one caller-owned **GPU
device-USM allocation**. The new ops write that allocation directly, eliminating
the spare output bank and final full-cache copy. An exclusive chain of optimized,
memory-identical cache views/Reshapes is allowed. Independent readers of the
cache root, incompatible layouts, or unrelated/partial input overlaps reject
alias binding and use private output storage with any required final copy.
Static/stateful/split paths retain their existing handling.

Two ordered kernels execute for each aliased op:

1. Gather active original rows and nonidentity reorder sources into private
   device scratch, applying logical source reset and computation-precision
   rounding. All reads finish before cache writes begin. Identity reorder
   rows require no scratch write. This supports arbitrary cycles and duplicate
   sources without imposing single-slot or no-rollback assumptions.
2. Compute convolution/GDN and perform reset, reorder, and ordered writeback in
   the original cache. Each destination element has one writer; GDN retained-row
   workers skip snapshot destinations written by recurrence owners. Later
   windows/snapshots retain their original precedence. Distinct input/output
   allocations use one immutable-source computation/writeback kernel.

Preserved values still round through computation precision. Untouched rows
therefore remain **read**, while already canonical values avoid stores. This
is not yet an implementation that touches only modified rows. Scratch capacity
is conservatively based on active-map plus full extra-map lengths; removing
one spare bank does not by itself establish lower total allocation size.

The computation retains the earlier K128/vector-four GDN with a sequential
loop over tokens, and fused FP32 convolution accumulation. It does not yet
substitute the optimized paged GDN or causal-convolution kernels. The plugin
already has input-mutating implementations for paged GDN, paged causal
convolution, and KV updates; this work supplies the generic llama.cpp map/cache
contract and binding. Command replay is conservatively disabled for these new
ops because stage selection depends on runtime aliasing; the normal dynamic
graph already cannot use full command replay. A failed aliased inference may
partly update the cache, so callers must restore or clear it before retrying.

### Matched target-model measurements

All runs use `Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`, GPU, `q4_asym64_all`, native
softplus and disabled KV slicing; `llama-server -np 4 -c 8192 -t 2 -tb 2` receives
the same saved 1,024 prompt token IDs and generates exactly 512 greedy tokens.
Prompt caching is disabled and EOS ignored. An eight-token request warms the
same prompt length, followed by three sequential timed requests and four varied
concurrent 96-token requests. These are single-active-request throughput results
on a four-slot server; the concurrent requests are functional checks.

| Final path | Decode tokens/s, three requests | Mean decode tokens/s | Mean prompt tokens/s |
| --- | ---: | ---: | ---: |
| Default rotation, non-MTP | 28.565 / 27.901 / 27.157 | 27.875 | 925.35 |
| Corrected in-place, non-MTP | 29.044 / 28.945 / 29.013 | 29.001 | 922.12 |
| Default rotation, two-token MTP | 46.593 / 47.063 / 45.380 | 46.345 | 865.50 |
| Corrected in-place, two-token MTP | 44.975 / 44.963 / 45.102 | 45.014 | 839.60 |

MTP acceptance is **340/340** in all three requests in both paths, and greedy
content matches the rotation control. Mean MTP prompt latency is 1,219.66 ms
in-place versus 1,183.14 ms rotation (+3.1%). Both modes complete the four
concurrent requests. The in-place path remains opt-in because its modest
non-MTP benefit is accompanied by a small MTP regression.

Cold model loading and
requantization take several minutes and are excluded from these response
timings. The runs are separate process-level samples, so the modest non-MTP
gain should not be treated as an exact kernel-only speedup. **30 non-MTP
tokens/s has not been reached.**

### Transfer evidence and alias-view correction

The corrected full-model decode probe finds **zero device-to-device cache
copies**, compared with **60 copies / 263,454,720 bytes (251.25 MiB)** when the
new ops fall back to private output storage. It observes 1,080 kernel submissions
per trunk decode rather than 1,020: removing the copies adds 60 gather kernels
(two recurrent ops in each of 30 recurrent layers). Small metadata transfers
and 21 `clFinish` calls remain. Ordinary warmed non-MTP decode uploads are eight
host metadata copies totaling 2,608 bytes and twenty host-USM copies totaling
320 bytes; three host output copies total 1,009,664 bytes. Eliminating cache
copies does not eliminate all transfers or synchronization.

The MTP probe also finds **zero device-to-device cache copies**, across all 26
instrumented trunk/draft inferences. All 60 recurrent cache roots have identical
input/output USM addresses in both modes. The runtime graph contains 60 Update
primitives in the trunk; the Qwen MTP layer uses full attention and has no
recurrent Update primitive. Warmed three-token trunk verification observes
1,131 kernel submissions and about 46.0–46.9 ms raw inference; each single-token
draft inference observes 68 submissions and about 6.2–6.4 ms. The interleaved
58-submission / 1.8–1.9 ms draft graph catches up the accepted hidden states.
These diagnostic timings come from a short probe request and are not substituted
for the three measured response timings.

The extra gather launches, retained-row reads, and scratch traffic mean that
zero final copies need not yield higher MTP t/s. The isolated larger-cache GDN
gain below does not explain or predict the full-model regression by itself;
the current measurements do not isolate its exact cause.

The first full-model opt-in samples were **fallback execution**, despite native
tests exercising real aliasing. llama.cpp inserts an optimized cache Reshape;
the initial overlap check incorrectly classified its root Parameter as an
unrelated overlapping input. The corrected check follows only exclusive,
memory-identical optimized alias steps and still vetoes independent root
readers. Native tests and a full-model device transfer probe cover this case.

For provenance, the pre-fix samples were 28.625 non-MTP and 42.600 MTP decode
tokens/s, and a later pre-fix armed non-MTP sample was 27.774. Their measurements
remain in the artifacts, but **they do not measure true in-place execution**
and are excluded from the final comparison table.

### Correctness and isolated kernel checks

Ten new native/core/visitor tests pass. They compare against separate immutable
reference graphs and cover cyclic maps, duplicates, identity suffixes,
reset-before-gather, overlapping windows/snapshots, snapshot counts 1/3,
changing batches/token counts, I32/I64 controls, F16/F32 caches and computation,
alternating aliased/distinct bindings, export/import, unrelated input overlaps,
and exclusive cache Reshapes with shared-root rejection. Twelve existing native
regression tests also pass after the final shared binding change.

The four-slot small-model replay suite passes **50 comparisons**, including
reset/copy, rollback depths 0/2, checkpoint export/import, and clears. Maximum
relative logit error is **0.003044** against the existing 0.01 threshold. The
corrected full-model non-MTP and MTP samples have identical greedy content to
their rotation controls and all four concurrent 96-token requests complete.

An isolated GDN sweep uses H32/K128/V128, device-USM input/output, separate
output storage for rotation and identical input/output storage for Update,
eight warmups and 200 inferences. Values begin in the computation-precision
canonical form. The large model was loading on CPU, with no concurrent model
generation during the sweep.

| Stored rows / tokens / snapshot count | Rotation ms | In-place Update ms |
| --- | ---: | ---: |
| 4 / 1 / 1 | 0.1904 | 0.1711 |
| 4 / 3 / 1 | 0.1807 | 0.1790 |
| 12 / 3 / 3 | 0.4309 | 0.3278 |

The twelve-row snapshot case improves about 24%; the four-row/three-token case
has little gain. This single synthetic wall-time sweep supports the expected
memory-traffic benefit for larger caches, without predicting whole-model t/s.

### Remaining opportunities and reproduction

The next independent-op improvements should target full retained-row scans and
scratch capacity, coalesced state accesses/register pressure in GDN, and
prefill-appropriate computation. Any precision change must explicitly account
for retained-row rounding and MTP acceptance. Small metadata transfers and host
synchronization remain separate costs. Improvements to the new kernels should
be benchmarked with reset/reorder/rollback intact.

Artifacts are in `tmp/recurrent-inplace-20261022/`: `bench.py`, effective commands,
response/repeat/parallel JSON, logs, `comparison.json`, native/replay tests,
`microbench-final.log`, and transfer probes/runtime graphs. The probe is armed
only after all three timed requests; primary timings contain no transfer
recording or runtime graph dumps. Reproduce the corrected measurements with:

```sh
python3 tmp/recurrent-inplace-20261022/bench.py fixed-inplace-normal-armed
python3 tmp/recurrent-inplace-20261022/bench.py fixed-inplace-mtp-armed
python3 tmp/recurrent-inplace-20261022/summary.py
```

For direct use, retain the established model/environment settings and add:

```sh
export GGML_OPENVINO_RECURRENT_CACHE_INPLACE=1
```

Capability detection and translation cache revision 9 require regeneration of
older compiled-model blobs. Detailed operation contracts are in OpenVINO's
`recurrent-conv-cache-update.rst` and `recurrent-gated-delta-net-update.rst`.

## Direct frontend integration and branch split (2026-10-22)

The current backend emits the two fused recurrent operations directly from
complete GGML reset/reorder/computation/writeback groups. It no longer emits
`RecurrentStateGather`, `RecurrentStateUpdate`, or the eight-input history-only
`RecurrentConvCache`, and the prepared-cache elimination pass is removed.
The dated measurements above describe the preceding implementations, not new
measurements of this cleanup.

The matcher verifies exclusive state consumers, depthwise convolution geometry,
ordered history windows and canonical GDN state writeback. Unmatched groups,
CPU/static/stateful/split execution and runtimes without the fused operations
use the ordinary GGML op translations. `GGML_OPENVINO_DISABLE_FUSED_RECURRENCE=1`
selects that decomposition for debugging. The old `DISABLE_RECURRENT_CACHE` and
`DISABLE_CACHE_PREPARE_FUSION` switches have been removed. Rotation is the default;
`GGML_OPENVINO_RECURRENT_CACHE_INPLACE=1` selects the separate Update operations
when the runtime supports them. Translation revision **10** invalidates old blobs.

OpenVINO branches start at clean upstream master `178d03e210`:

- `qwen-mtp-recurrent-fused-rotation`: `8568b33cf4`, rotation and its snapshot foundation.
- `qwen-mtp-recurrent-fused-inplace`: `3023daedef`, stacked on rotation, adding Update ops.
- `qwen-mtp-openvino-checkpoint`: the combined in-place tip used by `build/MtpOVPerf`.

The in-place change depends on rotation's core shape validation and reference
operations; it is a stacked alternative rather than an independent master branch.
The earlier checkpoint is preserved as
`backup/qwen-mtp-openvino-checkpoint-20261022` (`934cd6492f`).

After rebasing, the Release CPU/GPU plugins build and all **27** core, attribute
and GPU tests pass (17 rotation, 10 in-place). Both frontend paths pass **50**
four-slot small-model comparisons with reset, copy, checkpoint export/import,
clear, changing batches and rollback depths 0/2. The rotation IR contains 18
`RecurrentConvCache` and 18 `RecurrentGatedDeltaNet` operations and no standalone
native gather/update operations. Logs are in `tmp/ov-branch-split-20261022/` and
`tmp/qwen-finalize-20261022/`.
