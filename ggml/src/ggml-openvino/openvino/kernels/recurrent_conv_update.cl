// Qwen3.5/3.6 short convolution with in-place cache maintenance.
// Reads cache sources only from the stage output, so writes to the cache (input 0) cannot race
// with reads. Cache writes are distributed over all work items with a grid-stride loop.
__kernel void ggml_recurrent_conv_update(__global INPUT0_TYPE * cache,
                                         const __global INPUT1_TYPE * active_rows,
                                         const __global INPUT2_TYPE * extra_rows,
                                         const __global INPUT3_TYPE * reset_begin,
                                         const __global INPUT4_TYPE * reset_len,
                                         const __global INPUT5_TYPE * offsets,
                                         const __global INPUT6_TYPE * qkv,
                                         const __global INPUT7_TYPE * windows,
                                         const __global INPUT8_TYPE * weights,
                                         const __global INPUT9_TYPE * staged,
                                         const __global INPUT10_TYPE * token_count,
                                         __global OUTPUT0_TYPE * output) {
    const int S = INPUT0_DIMS[2], F = INPUT0_DIMS[3];
    const int B = INPUT1_DIMS[3], N = INPUT2_DIMS[3];
    // QKV may be padded to a token bucket; only the first T tokens are real.
    const int C = INPUT6_DIMS[2], T = RC_META(10, token_count, 0);
    const int n_windows = INPUT7_DIMS[2];
    const int i = get_global_id(0);
    const int total = get_global_size(0);

    if (i < B * C * T) {
        // Prefill QKV is channel-major: adjacent lanes read adjacent tokens. Short decode
        // batches favor contiguous [token,channel] output stores.
        const int c = T >= 16 ? (i / T) % C : i % C;
        const int t = T >= 16 ? i % T : (i / C) % T;
        const int b = i / (T * C);
        float sum = 0.0f;
        for (int j = 0; j <= HISTORY; ++j) {
            const int x = t + j;
            const float v = x < HISTORY ? (float) staged[RC_IN(9, 0, 0, b, c * HISTORY + x)] :
                                          (float) qkv[RC_IN(6, 0, b, c, x - HISTORY)];
            sum = fma(v, (float) weights[RC_IN(8, 0, 0, c, j)], sum);
        }
        output[RC_OUT(0, 0, b, t, c)] = (OUTPUT0_TYPE) sum;
    }

    const int rb = RC_META(3, reset_begin, 0), re = rb + RC_META(4, reset_len, 0);
    const int eb = RC_META(5, offsets, 0), es = RC_META(5, offsets, 1), E = N - es;
    for (int dst = 0; dst < S; ++dst) {
        // Later windows take precedence, matching the original ordered CPYs.
        int start = -1, dest = 0;
        for (int k = 0; k < n_windows; ++k) {
            const int d = (int) windows[RC_IN(7, 0, 0, k, 1)];
            if (dst >= d && dst < d + B) {
                start = (int) windows[RC_IN(7, 0, 0, k, 0)];
                dest = d;
            }
        }
        const int row = dst >= eb && dst < eb + E ? RC_META(2, extra_rows, dst - eb + es) : dst;
        if (start < 0 && row == dst && !(dst >= rb && dst < re)) {
            continue;  // unchanged row
        }
        for (int f = i; f < F; f += total) {
            float v = 0.0f;
            if (start >= 0) {
                const int b = dst - dest, c = f / HISTORY, t = start + f % HISTORY;
                v = t < HISTORY ? (float) staged[RC_IN(9, 0, 0, b, c * HISTORY + t)] :
                                  (float) qkv[RC_IN(6, 0, b, c, t - HISTORY)];
            } else if (row != dst) {
                v = (float) staged[RC_IN(9, 0, 0, B + dst - eb, f)];
            }
            cache[RC_IN(0, 0, 0, dst, f)] = (INPUT0_TYPE) v;
        }
    }
}
