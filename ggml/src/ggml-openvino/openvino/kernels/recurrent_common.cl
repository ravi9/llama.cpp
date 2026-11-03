#ifdef cl_khr_fp16
#    pragma OPENCL EXTENSION cl_khr_fp16 : enable
#endif

// Shared helpers for the llama.cpp recurrent cache custom kernels.
// The GPU plugin defines INPUTn_/OUTPUTn_ DIMS, PITCHES and OFFSET (in elements, bfyx order).
#define RC_IN(n, b, f, y, x)                                                                              \
    (INPUT##n##_OFFSET + (b) * INPUT##n##_PITCHES[0] + (f) * INPUT##n##_PITCHES[1] + (y) * INPUT##n##_PITCHES[2] + \
     (x) * INPUT##n##_PITCHES[3])
#define RC_OUT(n, b, f, y, x)                                                                                 \
    (OUTPUT##n##_OFFSET + (b) * OUTPUT##n##_PITCHES[0] + (f) * OUTPUT##n##_PITCHES[1] + (y) * OUTPUT##n##_PITCHES[2] + \
     (x) * OUTPUT##n##_PITCHES[3])
// Element x of a [1,1,1,N] metadata tensor.
#define RC_META(n, ptr, x) ((int) (ptr)[RC_IN(n, 0, 0, 0, x)])
