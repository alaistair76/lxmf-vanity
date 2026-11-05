// GPU half of lxmf-vanity: each thread walks X25519 private keys k, k+8,
// k+16, ... (k chosen on the host), derives each LXMF address, and records
// those that match the prefix. The host rechecks every hit on the CPU.
//
// Field elements mod 2^255-19 use ref10's 10-limb signed radix 2^25.5
// (limb i at bit ceil(25.5 i)), so products are native 32x32->64 multiplies.
// "Tight" values come out of fe_mul; adding or subtracting two tight values
// gives a "loose" value that fe_mul accepts without overflow.

typedef int i32;
typedef unsigned int u32;
typedef long long i64;
typedef unsigned long long u64;

#ifndef BATCH
#define BATCH 64  // keys per batch inversion; the host passes -DBATCH
#endif

struct fe { i32 v[10]; };

__device__ __constant__ int LIMB_BIT[10] = {0, 26, 51, 77, 102, 128, 153, 179, 204, 230};

__device__ __forceinline__ fe fe_add(const fe& f, const fe& g) {
    fe h;
#pragma unroll
    for (int i = 0; i < 10; i++) h.v[i] = f.v[i] + g.v[i];
    return h;
}

__device__ __forceinline__ fe fe_sub(const fe& f, const fe& g) {
    fe h;
#pragma unroll
    for (int i = 0; i < 10; i++) h.v[i] = f.v[i] - g.v[i];
    return h;
}

__device__ __forceinline__ fe fe_one() {
    fe h;
#pragma unroll
    for (int i = 0; i < 10; i++) h.v[i] = 0;
    h.v[0] = 1;
    return h;
}

// ref10's carry chain, from 64-bit column sums to a tight element.
__device__ __forceinline__ fe fe_carry(i64 h[10]) {
    i64 c;
    c = (h[0] + (1LL << 25)) >> 26; h[1] += c; h[0] -= c << 26;
    c = (h[4] + (1LL << 25)) >> 26; h[5] += c; h[4] -= c << 26;
    c = (h[1] + (1LL << 24)) >> 25; h[2] += c; h[1] -= c << 25;
    c = (h[5] + (1LL << 24)) >> 25; h[6] += c; h[5] -= c << 25;
    c = (h[2] + (1LL << 25)) >> 26; h[3] += c; h[2] -= c << 26;
    c = (h[6] + (1LL << 25)) >> 26; h[7] += c; h[6] -= c << 26;
    c = (h[3] + (1LL << 24)) >> 25; h[4] += c; h[3] -= c << 25;
    c = (h[7] + (1LL << 24)) >> 25; h[8] += c; h[7] -= c << 25;
    c = (h[4] + (1LL << 25)) >> 26; h[5] += c; h[4] -= c << 26;
    c = (h[8] + (1LL << 25)) >> 26; h[9] += c; h[8] -= c << 26;
    c = (h[9] + (1LL << 24)) >> 25; h[0] += c * 19; h[9] -= c << 25;
    c = (h[0] + (1LL << 25)) >> 26; h[1] += c; h[0] -= c << 26;
    fe out;
#pragma unroll
    for (int i = 0; i < 10; i++) out.v[i] = (i32)h[i];
    return out;
}

// Limb i times limb j lands at limb (i+j) mod 10, times 2 when both are odd
// (their half bits add up) and times 19 when it wraps (2^255 = 19).
__device__ __forceinline__ fe fe_mul(const fe& f, const fe& g) {
    i64 h[10];
#pragma unroll
    for (int k = 0; k < 10; k++) h[k] = 0;
#pragma unroll
    for (int i = 0; i < 10; i++) {
#pragma unroll
        for (int j = 0; j < 10; j++) {
            const i32 a = ((i & 1) && (j & 1)) ? 2 * f.v[i] : f.v[i];
            const i32 b = (i + j >= 10) ? 19 * g.v[j] : g.v[j];
            h[(i + j) % 10] += (i64)a * b;
        }
    }
    return fe_carry(h);
}

// As fe_mul(f, f), but each cross term f_i f_j (i < j) once, doubled.
__device__ __forceinline__ fe fe_sq(const fe& f) {
    i64 h[10];
#pragma unroll
    for (int k = 0; k < 10; k++) h[k] = 0;
#pragma unroll
    for (int i = 0; i < 10; i++) {
#pragma unroll
        for (int j = i; j < 10; j++) {
            const int scale = (i < j ? 2 : 1) * (((i & 1) && (j & 1)) ? 2 : 1);
            const i32 a = scale * f.v[i];
            const i32 b = (i + j >= 10) ? 19 * f.v[j] : f.v[j];
            h[(i + j) % 10] += (i64)a * b;
        }
    }
    return fe_carry(h);
}

__device__ __forceinline__ fe fe_sq_n(fe f, int n) {
    for (int i = 0; i < n; i++) f = fe_sq(f);
    return f;
}

// z^(p-2), ref10's addition chain.
__device__ fe fe_invert(const fe& z) {
    fe z2 = fe_sq(z);
    fe z9 = fe_mul(fe_sq_n(z2, 2), z);
    fe z11 = fe_mul(z9, z2);
    fe z2_5_0 = fe_mul(fe_sq(z11), z9);
    fe z2_10_0 = fe_mul(fe_sq_n(z2_5_0, 5), z2_5_0);
    fe z2_20_0 = fe_mul(fe_sq_n(z2_10_0, 10), z2_10_0);
    fe z2_40_0 = fe_mul(fe_sq_n(z2_20_0, 20), z2_20_0);
    fe z2_50_0 = fe_mul(fe_sq_n(z2_40_0, 10), z2_10_0);
    fe z2_100_0 = fe_mul(fe_sq_n(z2_50_0, 50), z2_50_0);
    fe z2_200_0 = fe_mul(fe_sq_n(z2_100_0, 100), z2_100_0);
    fe z2_250_0 = fe_mul(fe_sq_n(z2_200_0, 50), z2_50_0);
    return fe_mul(fe_sq_n(z2_250_0, 5), z11);
}

// 32 little-endian bytes (as 8 little-endian words) to limbs; bit 255 dropped.
__device__ fe fe_from_words(const u32 w[8]) {
    fe h;
#pragma unroll
    for (int i = 0; i < 10; i++) {
        const int bit = LIMB_BIT[i], word = bit >> 5, shift = bit & 31;
        u64 v = w[word];
        if (word + 1 < 8) v |= (u64)w[word + 1] << 32;
        const u32 mask = (i & 1) ? 0x1FFFFFF : 0x3FFFFFF;
        h.v[i] = (i32)((v >> shift) & mask);
    }
    return h;
}

// A tight element to its canonical 32 bytes, as 8 little-endian words (ref10).
__device__ void fe_to_words(fe f, u32 out[8]) {
    i32* h = f.v;
    i32 q = (19 * h[9] + (1 << 24)) >> 25;
    q = (h[0] + q) >> 26; q = (h[1] + q) >> 25; q = (h[2] + q) >> 26; q = (h[3] + q) >> 25;
    q = (h[4] + q) >> 26; q = (h[5] + q) >> 25; q = (h[6] + q) >> 26; q = (h[7] + q) >> 25;
    q = (h[8] + q) >> 26; q = (h[9] + q) >> 25;
    h[0] += 19 * q;
    i32 c;
    c = h[0] >> 26; h[1] += c; h[0] -= c << 26;
    c = h[1] >> 25; h[2] += c; h[1] -= c << 25;
    c = h[2] >> 26; h[3] += c; h[2] -= c << 26;
    c = h[3] >> 25; h[4] += c; h[3] -= c << 25;
    c = h[4] >> 26; h[5] += c; h[4] -= c << 26;
    c = h[5] >> 25; h[6] += c; h[5] -= c << 25;
    c = h[6] >> 26; h[7] += c; h[6] -= c << 26;
    c = h[7] >> 25; h[8] += c; h[7] -= c << 25;
    c = h[8] >> 26; h[9] += c; h[8] -= c << 26;
    c = h[9] >> 25; h[9] -= c << 25;
#pragma unroll
    for (int i = 0; i < 8; i++) out[i] = 0;
#pragma unroll
    for (int i = 0; i < 10; i++) {
        const int bit = LIMB_BIT[i], word = bit >> 5, shift = bit & 31;
        const u64 v = (u64)(u32)h[i] << shift;
        out[word] |= (u32)v;
        if (word + 1 < 8) out[word + 1] |= (u32)(v >> 32);
    }
}

// ---- SHA-256

__device__ __constant__ u32 K256[64] = {
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2};

__device__ __forceinline__ u32 rotr(u32 x, int n) { return __funnelshift_r(x, x, n); }

// One compression of `w` (16 big-endian words) into `s`.
__device__ __forceinline__ void sha256_block(u32 s[8], const u32 w_in[16]) {
    u32 w[16];
#pragma unroll
    for (int i = 0; i < 16; i++) w[i] = w_in[i];
    u32 a = s[0], b = s[1], c = s[2], d = s[3], e = s[4], f = s[5], g = s[6], h = s[7];
#pragma unroll
    for (int i = 0; i < 64; i++) {
        u32 wi;
        if (i < 16) {
            wi = w[i];
        } else {
            const u32 w15 = w[(i - 15) & 15], w2 = w[(i - 2) & 15];
            const u32 s0 = rotr(w15, 7) ^ rotr(w15, 18) ^ (w15 >> 3);
            const u32 s1 = rotr(w2, 17) ^ rotr(w2, 19) ^ (w2 >> 10);
            wi = w[i & 15] = w[i & 15] + s0 + w[(i - 7) & 15] + s1;
        }
        const u32 S1 = rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25);
        const u32 ch = (e & f) ^ (~e & g);
        const u32 t1 = h + S1 + ch + K256[i] + wi;
        const u32 S0 = rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22);
        const u32 maj = (a & b) ^ (a & c) ^ (b & c);
        const u32 t2 = S0 + maj;
        h = g; g = f; f = e; e = d + t1; d = c; c = b; b = a; a = t1 + t2;
    }
    s[0] += a; s[1] += b; s[2] += c; s[3] += d; s[4] += e; s[5] += f; s[6] += g; s[7] += h;
}

__device__ __forceinline__ void sha256_init(u32 s[8]) {
    s[0] = 0x6a09e667; s[1] = 0xbb67ae85; s[2] = 0x3c6ef372; s[3] = 0xa54ff53a;
    s[4] = 0x510e527f; s[5] = 0x9b05688c; s[6] = 0x1f83d9ab; s[7] = 0x5be0cd19;
}

__device__ __forceinline__ u32 bswap(u32 x) { return __byte_perm(x, 0, 0x0123); }

// The first 4 big-endian words of the LXMF address for X25519 public key u.
// ed[8]: the Ed25519 public key as big-endian words. nh[3]: the name hash's
// 10 bytes as big-endian words, the last holding two bytes in its top half.
__device__ void lxmf_address(const u32 u[8], const u32 ed[8], const u32 nh[3], u32 out[4]) {
    u32 w[16];
    u32 s[8];
    // identity hash = SHA-256(u || ed)[0..16]: a full block, then padding.
    sha256_init(s);
#pragma unroll
    for (int i = 0; i < 8; i++) w[i] = bswap(u[i]);
#pragma unroll
    for (int i = 0; i < 8; i++) w[8 + i] = ed[i];
    sha256_block(s, w);
    w[0] = 0x80000000;
#pragma unroll
    for (int i = 1; i < 15; i++) w[i] = 0;
    w[15] = 512;
    sha256_block(s, w);
    // address = SHA-256(name hash (10) || identity hash (16))[0..16]: one block.
    w[0] = nh[0];
    w[1] = nh[1];
    w[2] = nh[2] | (s[0] >> 16);
    w[3] = (s[0] << 16) | (s[1] >> 16);
    w[4] = (s[1] << 16) | (s[2] >> 16);
    w[5] = (s[2] << 16) | (s[3] >> 16);
    w[6] = (s[3] << 16) | 0x8000;
#pragma unroll
    for (int i = 7; i < 15; i++) w[i] = 0;
    w[15] = 26 * 8;
    u32 t[8];
    sha256_init(t);
    sha256_block(t, w);
#pragma unroll
    for (int i = 0; i < 4; i++) out[i] = t[i];
}

// ---- Kernels

// Per-thread state: X(n-1), Z(n-1), X(n), Z(n) as 4 x 10 limbs.
#define STATE_INTS 40

// Sets up each thread from u(k-8) and u(k): 16 little-endian words each.
extern "C" __global__ void init_walk(i32* state, const u32* u_prev_cur, u32 threads) {
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= threads) return;
    const fe xp = fe_from_words(u_prev_cur + t * 16);
    const fe xc = fe_from_words(u_prev_cur + t * 16 + 8);
    const fe one = fe_one();
    i32* st = state + (u64)t * STATE_INTS;
#pragma unroll
    for (int i = 0; i < 10; i++) {
        st[i] = xp.v[i];
        st[10 + i] = one.v[i];
        st[20 + i] = xc.v[i];
        st[30 + i] = one.v[i];
    }
}

// Walks `iters` batches per thread. Thread t's next key is k_t + 8 n[t].
// A key whose address matches (addr & mask) == want is recorded in `hits`
// as (t, n). With `dump`, the first `dump_threads` threads also write the
// public keys of their first batch there (for the self-test).
extern "C" __global__ void walk(i32* state, u64* ns, u32 threads, u32 iters,
                                const u32* ed, u32 nh0, u32 nh1, u32 nh2,
                                const u32* want, const u32* mask,
                                u32* hit_count, u64* hits, u32 max_hits,
                                u32* dump, u32 dump_threads,
                                const u32* q_words) {
    const u32 t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= threads) return;

    u32 edl[8];
#pragma unroll
    for (int i = 0; i < 8; i++) edl[i] = ed[i];
    const u32 nh[3] = {nh0, nh1, nh2};
    u32 wantl[4], maskl[4];
#pragma unroll
    for (int i = 0; i < 4; i++) { wantl[i] = want[i]; maskl[i] = mask[i]; }

    // u(8B) + 1 and u(8B) - 1, for the differential addition.
    const fe q = fe_from_words(q_words);
    const fe q_plus = fe_add(q, fe_one());
    const fe q_minus = fe_sub(q, fe_one());

    i32* st = state + (u64)t * STATE_INTS;
    fe xp, zp, xc, zc;
#pragma unroll
    for (int i = 0; i < 10; i++) {
        xp.v[i] = st[i]; zp.v[i] = st[10 + i]; xc.v[i] = st[20 + i]; zc.v[i] = st[30 + i];
    }
    u64 n = ns[t];

    for (u32 it = 0; it < iters; it++) {
        fe xs[BATCH], zs[BATCH];
        for (int i = 0; i < BATCH; i++) {
            xs[i] = xc;
            zs[i] = zc;
            // xADD: P(n+1) = P(n) + Q, where P(n) - Q = P(n-1) and Q is affine.
            const fe t1 = fe_mul(fe_sub(xc, zc), q_plus);
            const fe t2 = fe_mul(fe_add(xc, zc), q_minus);
            const fe xn = fe_mul(zp, fe_sq(fe_add(t1, t2)));
            const fe zn = fe_mul(xp, fe_sq(fe_sub(t1, t2)));
            xp = xc; zp = zc; xc = xn; zc = zn;
        }
        // Montgomery's trick: one inversion for the batch. prefix[i] is the
        // product of zs[0..i].
        fe acc = fe_one();
        fe prefix[BATCH];
        for (int i = 0; i < BATCH; i++) {
            prefix[i] = acc;
            acc = fe_mul(acc, zs[i]);
        }
        fe inv = fe_invert(acc);
        for (int i = BATCH - 1; i >= 0; i--) {
            const fe z_inv = fe_mul(inv, prefix[i]);
            inv = fe_mul(inv, zs[i]);
            u32 u[8];
            fe_to_words(fe_mul(xs[i], z_inv), u);
            if (dump != 0 && it == 0 && t < dump_threads) {
#pragma unroll
                for (int k = 0; k < 8; k++) dump[((u64)t * BATCH + i) * 8 + k] = u[k];
            }
            u32 addr[4];
            lxmf_address(u, edl, nh, addr);
            bool match = true;
#pragma unroll
            for (int k = 0; k < 4; k++) match = match && ((addr[k] & maskl[k]) == wantl[k]);
            if (match) {
                const u32 slot = atomicAdd(hit_count, 1);
                if (slot < max_hits) {
                    hits[slot * 2] = t;
                    hits[slot * 2 + 1] = n + i;
                }
            }
        }
        n += BATCH;
    }

#pragma unroll
    for (int i = 0; i < 10; i++) {
        st[i] = xp.v[i]; st[10 + i] = zp.v[i]; st[20 + i] = xc.v[i]; st[30 + i] = zc.v[i];
    }
    ns[t] = n;
}
