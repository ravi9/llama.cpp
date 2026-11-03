// Copy every cache row read by the following update kernel before it writes the cache in place.
// Rows [0,B) are the active sources; rows [B,B+E) are nonidentity reorder sources for
// destinations [eb,eb+E). Reset rows read as zero.
__kernel void ggml_recurrent_cache_stage(const __global INPUT0_TYPE * cache,
                                         const __global INPUT1_TYPE * active_rows,
                                         const __global INPUT2_TYPE * extra_rows,
                                         const __global INPUT3_TYPE * reset_begin,
                                         const __global INPUT4_TYPE * reset_len,
                                         const __global INPUT5_TYPE * offsets,
                                         __global OUTPUT0_TYPE * staged) {
    const int F = INPUT0_DIMS[3];
    const int B = INPUT1_DIMS[3];
    const int N = INPUT2_DIMS[3];
    const int i = get_global_id(0);
    if (i >= (B + N) * F) {
        return;
    }
    const int row = i / F, f = i % F;
    const int rb = RC_META(3, reset_begin, 0), re = rb + RC_META(4, reset_len, 0);
    const int eb = RC_META(5, offsets, 0), es = RC_META(5, offsets, 1);
    int src;
    if (row < B) {
        src = RC_META(1, active_rows, row);
    } else {
        const int e = row - B;
        if (e >= N - es) {
            return;
        }
        src = RC_META(2, extra_rows, e + es);
        if (src == eb + e) {
            return;  // identity rows are not rewritten
        }
    }
    staged[RC_OUT(0, 0, 0, row, f)] =
        src >= rb && src < re ? (OUTPUT0_TYPE) 0 : (OUTPUT0_TYPE) cache[RC_IN(0, 0, 0, src, f)];
}
