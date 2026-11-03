// Qwen3.5/3.6 gated delta net with in-place cache maintenance.
// One subgroup owns (sequence, head, 4 value columns); each lane holds K_HEAD_DIM/SUBGROUP_SIZE
// key elements. Initial states come from the stage output, so the cache (input 3) is only written.
// SNAPSHOT_OUTPUT writes the newest states to output 1 instead of the cache (standalone graphs).
#define RC_KPL (K_HEAD_DIM / SUBGROUP_SIZE)
#define RC_VB  4

// Cache row for the state of active row b after `slot` steps from the end, or -1.
// Rows are written plane by plane; a later plane wins an overlapping destination.
inline int rc_snapshot_row(const __global INPUT11_TYPE * control, int b, int slot, int B, int valid) {
    const int begin = RC_META(11, control, 0), seqs = RC_META(11, control, 1);
    const int stride = RC_META(11, control, 2), source = RC_META(11, control, 3);
    const int r = slot * B + b - source, n = valid * B - source;
    if (r < 0 || r >= n) {
        return -1;
    }
    const int dst = begin + (r / seqs) * stride + r % seqs;
    for (int plane = r / seqs + 1; plane * seqs < n; ++plane) {
        const int offset = dst - begin - plane * stride;
        if (offset >= 0 && offset < seqs && plane * seqs + offset < n) {
            return -1;
        }
    }
    return dst;
}

inline bool rc_snapshot_owned(const __global INPUT11_TYPE * control, int dst, int B, int valid) {
    const int begin = RC_META(11, control, 0), seqs = RC_META(11, control, 1);
    const int stride = RC_META(11, control, 2), source = RC_META(11, control, 3);
    const int n = valid * B - source;
    for (int plane = 0; plane * seqs < n; ++plane) {
        const int offset = dst - begin - plane * stride;
        if (offset >= 0 && offset < seqs && plane * seqs + offset < n) {
            return true;
        }
    }
    return false;
}

__attribute__((intel_reqd_sub_group_size(SUBGROUP_SIZE)))
__kernel void ggml_recurrent_gdn_update(const __global INPUT0_TYPE * q,
                                        const __global INPUT1_TYPE * k,
                                        const __global INPUT2_TYPE * v,
                                        __global INPUT3_TYPE * cache,
                                        const __global INPUT4_TYPE * g,
                                        const __global INPUT5_TYPE * beta,
                                        const __global INPUT6_TYPE * active_rows,
                                        const __global INPUT7_TYPE * extra_rows,
                                        const __global INPUT8_TYPE * reset_begin,
                                        const __global INPUT9_TYPE * reset_len,
                                        const __global INPUT10_TYPE * offsets,
                                        const __global INPUT11_TYPE * control,
                                        const __global INPUT12_TYPE * staged,
                                        const __global INPUT13_TYPE * token_count,
                                        __global OUTPUT0_TYPE * output
#if SNAPSHOT_OUTPUT
                                        ,
                                        __global OUTPUT1_TYPE * snapshots
#endif
) {
    // Token inputs may be padded to a bucket; only the first T tokens are real.
    const int B = INPUT0_DIMS[0], T = RC_META(13, token_count, 0), HK = INPUT0_DIMS[2];
    const int H = INPUT2_DIMS[2];
    const int S = INPUT3_DIMS[2], F = INPUT3_DIMS[3];
    const int b = get_global_id(0), h = get_global_id(1);
    const int start_v = get_group_id(2) * RC_VB;
    const int lid = get_sub_group_local_id();
    const int hk = h % HK;  // GGML maps GQA heads in tiled order
    const int valid = min(SNAPSHOT_COUNT, T);
    const float scale = rsqrt((float) K_HEAD_DIM);

    float state[RC_VB][RC_KPL];
    for (int iv = 0; iv < RC_VB; ++iv) {
        for (int j = 0; j < RC_KPL; ++j) {
            const int col = start_v + iv;
            state[iv][j] = col < V_HEAD_DIM ?
                               (float) staged[RC_IN(12, 0, 0, b, (h * V_HEAD_DIM + col) * K_HEAD_DIM + lid * RC_KPL + j)] :
                               0.0f;
        }
    }

    for (int t = 0; t < T; ++t) {
        const float decay = exp((float) g[RC_IN(4, b, t, h, 0)]);
        const float beta_t = (float) beta[RC_IN(5, b, t, h, 0)];
        float kv[RC_KPL], qv[RC_KPL];
        float k_sum = 0.0f, q_sum = 0.0f;
        for (int j = 0; j < RC_KPL; ++j) {
            kv[j] = (float) k[RC_IN(1, b, t, hk, lid * RC_KPL + j)];
            qv[j] = (float) q[RC_IN(0, b, t, hk, lid * RC_KPL + j)];
            k_sum = fma(kv[j], kv[j], k_sum);
            q_sum = fma(qv[j], qv[j], q_sum);
        }
#if FUSE_QK_L2NORM
        const float k_scale = rsqrt(sub_group_reduce_add(k_sum) + K_L2_NORM_EPS);
        const float q_scale = rsqrt(sub_group_reduce_add(q_sum) + Q_L2_NORM_EPS) * scale;
#else
        const float k_scale = 1.0f;
        const float q_scale = scale;
#endif
        for (int j = 0; j < RC_KPL; ++j) {
            kv[j] *= k_scale;
            qv[j] *= q_scale;
        }

        const int slot = T - 1 - t;
        const int dst = slot < SNAPSHOT_COUNT ?
#if SNAPSHOT_OUTPUT
                                              (slot < valid ? slot * B + b : -1)
#else
                                              rc_snapshot_row(control, b, slot, B, valid)
#endif
                                              : -1;
        float out[RC_VB];
        for (int iv = 0; iv < RC_VB; ++iv) {
            float dot = 0.0f;
            for (int j = 0; j < RC_KPL; ++j) {
                state[iv][j] *= decay;
                dot = fma(state[iv][j], kv[j], dot);
            }
            const int col = start_v + iv;
            const float v_t = col < V_HEAD_DIM ? (float) v[RC_IN(2, b, t, h, col)] : 0.0f;
            const float update = (v_t - sub_group_reduce_add(dot)) * beta_t;
            float dq = 0.0f;
            for (int j = 0; j < RC_KPL; ++j) {
                state[iv][j] = fma(kv[j], update, state[iv][j]);
                dq = fma(state[iv][j], qv[j], dq);
            }
            out[iv] = sub_group_reduce_add(dq);
            if (dst >= 0 && col < V_HEAD_DIM) {
                const int f = (h * V_HEAD_DIM + col) * K_HEAD_DIM + lid * RC_KPL;
                for (int j = 0; j < RC_KPL; ++j) {
#if SNAPSHOT_OUTPUT
                    snapshots[RC_OUT(1, 0, 0, dst, f + j)] = (OUTPUT1_TYPE) state[iv][j];
#else
                    cache[RC_IN(3, 0, 0, dst, f + j)] = (INPUT3_TYPE) state[iv][j];
#endif
                }
            }
        }
        if (lid == 0) {
            for (int iv = 0; iv < RC_VB; ++iv) {
                if (start_v + iv < V_HEAD_DIM) {
                    output[RC_OUT(0, b, t, h, start_v + iv)] = (OUTPUT0_TYPE) out[iv];
                }
            }
        }
    }

#if !SNAPSHOT_OUTPUT
    // Reset and reorder rows that the recurrence does not own, spread over the whole grid.
    const int N = INPUT7_DIMS[3];
    const int rb = RC_META(8, reset_begin, 0), re = rb + RC_META(9, reset_len, 0);
    const int eb = RC_META(10, offsets, 0), es = RC_META(10, offsets, 1), E = N - es;
    const int lin = (get_global_id(0) * get_global_size(1) + get_global_id(1)) * get_global_size(2) + get_global_id(2);
    const int total = get_global_size(0) * get_global_size(1) * get_global_size(2);
    for (int dst = 0; dst < S; ++dst) {
        const int row = dst >= eb && dst < eb + E ? RC_META(7, extra_rows, dst - eb + es) : dst;
        if ((row == dst && !(dst >= rb && dst < re)) || rc_snapshot_owned(control, dst, B, valid)) {
            continue;
        }
        for (int f = lin; f < F; f += total) {
            cache[RC_IN(3, 0, 0, dst, f)] =
                row != dst ? (INPUT3_TYPE) staged[RC_IN(12, 0, 0, B + dst - eb, f)] : (INPUT3_TYPE) 0;
        }
    }
#endif
}
