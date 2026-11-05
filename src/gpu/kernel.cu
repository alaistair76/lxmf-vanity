// CUDA entry points for the GPU search; the work is in common.h, which the
// host splices in where the #include is (NVRTC gets one source string).

typedef int i32;
typedef unsigned int u32;
typedef long long i64;
typedef unsigned long long u64;

#define FN __device__ __forceinline__
#define TABLE __device__ __constant__
#define GLOBAL
#define LOCAL_U32_PTR u32*
#define LOCAL_I64_PTR i64*
#define HIT_COUNTER u32
#define ATOMIC_INC(p) atomicAdd((p), 1u)
#define rotr(x, n) __funnelshift_r((x), (x), (n))
#define bswap(x) __byte_perm((x), 0, 0x0123)

#include "common.h"

extern "C" __global__ void init_walk(i32* state, const u32* u_prev_cur, u32 threads) {
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < threads) init_thread(t, state, u_prev_cur);
}

extern "C" __global__ void walk(i32* state, u64* ns, u32 threads, u32 iters,
                                const u32* ed, u32 nh0, u32 nh1, u32 nh2,
                                const u32* want, const u32* mask,
                                u32* hit_count, u64* hits, u32 max_hits,
                                u32* dump, u32 dump_threads, const u32* q_words) {
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t < threads) {
        walk_thread(t, state, ns, iters, ed, nh0, nh1, nh2, want, mask, hit_count, hits, max_hits,
                    dump, dump_threads, q_words);
    }
}
