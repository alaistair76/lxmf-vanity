// Metal entry points for the GPU search; the work is in common.h, which the
// host splices in where the #include is (Metal gets one source string).

#include <metal_stdlib>
using namespace metal;

typedef int i32;
typedef uint u32;
typedef long i64;
typedef ulong u64;

#define FN inline
#define TABLE constant
#define GLOBAL device
#define LOCAL_U32_PTR thread u32*
#define LOCAL_I64_PTR thread i64*
#define HIT_COUNTER atomic_uint
#define ATOMIC_INC(p) atomic_fetch_add_explicit((p), 1u, memory_order_relaxed)
#define rotr(x, n) rotate((u32)(x), (u32)(32 - (n)))
#define bswap(x) ((((x) >> 24) & 0xFFu) | (((x) >> 8) & 0xFF00u) | (((x) << 8) & 0xFF0000u) | ((x) << 24))

#include "common.h"

// The kernels' scalar arguments, set with setBytes.
struct InitArgs {
    u32 threads;
};

struct WalkArgs {
    u32 threads, iters, nh0, nh1, nh2, max_hits, dump_threads;
};

kernel void init_walk(device i32* state [[buffer(0)]],
                      device const u32* u_prev_cur [[buffer(1)]],
                      constant InitArgs& args [[buffer(2)]],
                      uint t [[thread_position_in_grid]]) {
    if (t < args.threads) init_thread(t, state, u_prev_cur);
}

kernel void walk(device i32* state [[buffer(0)]],
                 device u64* ns [[buffer(1)]],
                 constant WalkArgs& args [[buffer(2)]],
                 device const u32* ed [[buffer(3)]],
                 device const u32* want [[buffer(4)]],
                 device const u32* mask [[buffer(5)]],
                 device atomic_uint* hit_count [[buffer(6)]],
                 device u64* hits [[buffer(7)]],
                 device u32* dump [[buffer(8)]],
                 device const u32* q_words [[buffer(9)]],
                 uint t [[thread_position_in_grid]]) {
    if (t < args.threads) {
        walk_thread(t, state, ns, args.iters, ed, args.nh0, args.nh1, args.nh2, want, mask, hit_count, hits,
                    args.max_hits, dump, args.dump_threads, q_words);
    }
}
