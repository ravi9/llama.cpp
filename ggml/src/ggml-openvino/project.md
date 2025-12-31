## Goal
Implement an OpenVINO backend for llama.cpp.

## Current Status
The following tools work with the OpenVINO backend on CPU and GPU: `llama-simple`, `llama-run`, `llama-cli`, `llama-server`, `llama-bench`, `llama-perplexity`.

On GPU, llama-bench results are close to Vulkan performance for larger models. See [llama.cpp-ov bench (backend buffer)](https://intel-my.sharepoint.com/:x:/p/zijun_yu/IQBSyjB9vhBWS6FHFsNp1yi-Ae-K43rcttfl-xUFiz0nkdM?e=pe15Th).

**NPU limitations:**
- Does not support `llama-server -np > 1` (multiple parallel sequences)
- Only supports `llama-perplexity -b 512` or smaller

## Key Problems

Other backends operate at the kernel level, while the OpenVINO backend operates at the graph level.

**Root cause:** OpenVINO is an AOT (ahead-of-time) framework, but llama.cpp doesn't have a graph compilation step.

### Problem 1: Static ggml cgraph vs. Dynamic OpenVINO IR

For each token, llama.cpp builds a cgraph and delegates execution to backends. See [appendix](#cgraph-example) for an example.

Each cgraph has fixed shapes, but the graph structure changes every inference step:
1. Input shapes change
2. Ops change dynamically (e.g., `VIEW` offsets for KV cache depend on past token count, see [appendix](#view-dimensions-of-cache_kcache_v-change-based-on-past-token-length) for details)

Other backends execute the ops in the cgraph, reading/writing directly to tensor data pointers (addresses in buffers allocated by the backend). The OpenVINO backend must convert the cgraph to an OpenVINO IR graph. Since OpenVINO is AOT, the IR must be compiled before execution, and compilation is expensive—we can't afford to recompile at every inference step.

**Current solution:** Build a dynamic IR with symbolic shapes (e.g., `inp_tokens` shape `[1,1,1,-1]`) for CPU and GPU, and extract changing values as extra inputs (e.g., `attention_size` for slicing KV cache). The compiled graph is then cached.

**Limitations:**
1. This approach works but is fragile and may not scale. The current conversion logic is ad-hoc and only guaranteed to work for llama-like models.
2. Makes fallback to CPU harder. The conversion code is designed to convert the entire cgraph to OpenVINO IR. If the cgraph is split (e.g., some ops not supported by OpenVINO), it's unclear how to convert a partial cgraph to a dynamic IR.

### Problem 2: Buffer Management

Other backends allocate buffers for weights, KV cache, and compute tensors; kernels read/write directly to these buffers.

In the OpenVINO backend:
1. We allocate buffers and load weights (with extraction for quantized weights to match OpenVINO's expected quantization format)
2. We create `ov::Constant` nodes pointing to the weight buffers and use them in the IR graph
3. The compiled blob or inference request likely contains a copy of the weights, effectively doubling memory usage


## Appendix

### cgraph example
```
 nodes          shape                  op                name                                                            stride         buffer_type
 -   0: [  2048,     1,     1,     1] GET_ROWS             inp_embd                                     [ 4,  8192,  8192,  8192]       OPENVINO0
        [  2048, 128256,     1,     1]            0: NONE        token_embd.weight                           [ 210,  1680, 215470080, 215470080]      CPU_Mapped
        [     1,     1,     1,     1]            1: NONE        inp_tokens                                  [ 4,     4,     4,     4]  OPENVINO0_HOST
 -   1: [  2048,     1,     1,     1] RMS_NORM             norm-0                                       [ 4,  8192,  8192,  8192]       OPENVINO0
        [  2048,     1,     1,     1]            0: GET_ROWS    inp_embd                                    [ 4,  8192,  8192,  8192]       OPENVINO0
 -   2: [  2048,     1,     1,     1] MUL                  attn_norm-0                                  [ 4,  8192,  8192,  8192]       OPENVINO0
        [  2048,     1,     1,     1]            0: RMS_NORM    norm-0                                      [ 4,  8192,  8192,  8192]       OPENVINO0
        [  2048,     1,     1,     1]            1: NONE        blk.0.attn_norm.weight                      [ 4,  8192,  8192,  8192]       OPENVINO0
 -   3: [  2048,     1,     1,     1] MUL_MAT              Qcur-0                                       [ 4,  8192,  8192,  8192]       OPENVINO0
        [  2048,  2048,     1,     1]            0: NONE        blk.0.attn_q.weight                         [ 18,  1152, 2359296, 2359296]       OPENVINO0
        [  2048,     1,     1,     1]            1: MUL         attn_norm-0                                 [ 4,  8192,  8192,  8192]       OPENVINO0
```

### View dimensions of cache_k/cache_v change based on past token length
```
-  19: [    64,     8,   256,     1] VIEW                 cache_v_l0 (view)                            [ 2,   128,  1024, 14336]       OPENVINO0
        [   512,  4096,     1,     1]            0: NONE        cache_v_l0                                  [ 2,  1024, 14336, 14336]       OPENVINO0
-  20: [    64,   256,     8,     1] PERMUTE              cache_v_l0 (view) (permuted)                 [ 2,  1024,   128, 14336]       OPENVINO0
        [    64,     8,   256,     1]            0: VIEW        cache_v_l0 (view)                           [ 2,   128,  1024, 14336]       OPENVINO0
-  22: [    64,    32,     1,     1] FLASH_ATTN_EXT       __fattn__-0                                  [ 4,   256,  8192,  8192]       OPENVINO0
        [    64,     1,    32,     1]            0: PERMUTE     Qcur-0 (view) (permuted)                    [ 4,  8192,   256,  8192]       OPENVINO0
        [    64,   256,     8,     1]            1: PERMUTE     cache_k_l0 (view) (permuted)                [ 2,  1024,   128, 14336]       OPENVINO0
        [    64,   256,     8,     1]            2: PERMUTE     cache_v_l0 (view) (permuted)                [ 2,  1024,   128, 14336]       OPENVINO0
        [   256,    64,     1,     1]            3: CPY         KQ_mask (copy)                              [ 2,    28,  1792,  1792]       OPENVINO0
```
When the past token length crosses 256 tokens, the shapes of `cache_k_l0` and `cache_v_l0` change from `[64, 8, 256, 1]` to `[64, 8, 512, 1]`.
