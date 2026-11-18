// Artificer CUDA kernels — compiled at runtime with NVRTC (see cuda_quant.rs).
// Quantized GGUF blocks are read as-is from device memory and dequantized
// inside the matvec loop. Layouts match crates/core/src/quant/dequant.rs
// (llama.cpp / ggml block structs). One warp computes one output row; each
// lane walks 32-element chunks of the row.
typedef unsigned char u8;
typedef signed char i8;
typedef unsigned short u16;
typedef unsigned int u32;

__device__ __forceinline__ float h2f(u16 h) {
    float f;
    asm("cvt.f32.f16 %0, %1;" : "=f"(f) : "h"(h));
    return f;
}
__device__ __forceinline__ u16 ld16(const u8* p) { return (u16)p[0] | ((u16)p[1] << 8); }
__device__ __forceinline__ u32 ld32(const u8* p) {
    return (u32)p[0] | ((u32)p[1] << 8) | ((u32)p[2] << 16) | ((u32)p[3] << 24);
}

// ---- 32-element blocks (j ignored) ----
__device__ __forceinline__ float dot_q4_0(const u8* b, int, const float* x) {
    float d = h2f(ld16(b)); const u8* qs = b + 2; float s = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j)
        s += (float)((int)(qs[j] & 15) - 8) * x[j] + (float)((int)(qs[j] >> 4) - 8) * x[j + 16];
    return d * s;
}
__device__ __forceinline__ float dot_q4_1(const u8* b, int, const float* x) {
    float d = h2f(ld16(b)), m = h2f(ld16(b + 2)); const u8* qs = b + 4; float s = 0.f, sx = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        s += (float)(qs[j] & 15) * x[j] + (float)(qs[j] >> 4) * x[j + 16];
        sx += x[j] + x[j + 16];
    }
    return d * s + m * sx;
}
__device__ __forceinline__ float dot_q5_0(const u8* b, int, const float* x) {
    float d = h2f(ld16(b)); u32 qh = ld32(b + 2); const u8* qs = b + 6; float s = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        int x0 = ((qs[j] & 15) | (((qh >> j) << 4) & 0x10)) - 16;
        int x1 = ((qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16;
        s += (float)x0 * x[j] + (float)x1 * x[j + 16];
    }
    return d * s;
}
__device__ __forceinline__ float dot_q5_1(const u8* b, int, const float* x) {
    float d = h2f(ld16(b)), m = h2f(ld16(b + 2)); u32 qh = ld32(b + 4); const u8* qs = b + 8;
    float s = 0.f, sx = 0.f;
    #pragma unroll
    for (int j = 0; j < 16; ++j) {
        int x0 = (qs[j] & 15) | (((qh >> j) << 4) & 0x10);
        int x1 = (qs[j] >> 4) | ((qh >> (j + 12)) & 0x10);
        s += (float)x0 * x[j] + (float)x1 * x[j + 16];
        sx += x[j] + x[j + 16];
    }
    return d * s + m * sx;
}
__device__ __forceinline__ float dot_q8_0(const u8* b, int, const float* x) {
    float d = h2f(ld16(b)); const i8* qs = (const i8*)(b + 2); float s = 0.f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) s += (float)qs[j] * x[j];
    return d * s;
}

// ---- 256-element super-blocks: chunk j in 0..8 covers elements 32j..32j+31 ----
__device__ __forceinline__ float dot_q2_k(const u8* sb, int j, const float* x) {
    const u8* sc = sb; const u8* q = sb + 16 + 32 * (j >> 2); int shift = 2 * (j & 3);
    float d = h2f(ld16(sb + 80)), dmin = h2f(ld16(sb + 82));
    float r = 0.f;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        u8 s8 = sc[2 * j + h]; float sq = 0.f, sx = 0.f;
        #pragma unroll
        for (int l = 0; l < 16; ++l) {
            float xv = x[16 * h + l];
            sq += (float)((q[16 * h + l] >> shift) & 3) * xv; sx += xv;
        }
        r += d * (float)(s8 & 15) * sq - dmin * (float)(s8 >> 4) * sx;
    }
    return r;
}
__device__ __forceinline__ int q3k_scale(const u8* raw, int i) {
    u32 a0 = ld32(raw), a1 = ld32(raw + 4), t = ld32(raw + 8);
    const u32 km1 = 0x03030303u, km2 = 0x0f0f0f0fu;
    u32 a;
    switch (i >> 2) {
        case 0: a = (a0 & km2) | (((t >> 0) & km1) << 4); break;
        case 1: a = (a1 & km2) | (((t >> 2) & km1) << 4); break;
        case 2: a = ((a0 >> 4) & km2) | (((t >> 4) & km1) << 4); break;
        default: a = ((a1 >> 4) & km2) | (((t >> 6) & km1) << 4); break;
    }
    return (int)(i8)((a >> (8 * (i & 3))) & 0xFF);
}
__device__ __forceinline__ float dot_q3_k(const u8* sb, int j, const float* x) {
    const u8* hm = sb; const u8* q = sb + 32 + 32 * (j >> 2); int shift = 2 * (j & 3);
    u8 m = (u8)(1u << j); float d = h2f(ld16(sb + 108)); float r = 0.f;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        float dl = d * (float)(q3k_scale(sb + 96, 2 * j + h) - 32); float s = 0.f;
        #pragma unroll
        for (int l = 16 * h; l < 16 * h + 16; ++l) {
            int qv = (int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4);
            s += (float)qv * x[l];
        }
        r += dl * s;
    }
    return r;
}
__device__ __forceinline__ void scale_min_k4(int j, const u8* q, int* sc, int* mn) {
    if (j < 4) { *sc = q[j] & 63; *mn = q[j + 4] & 63; }
    else { *sc = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4); *mn = (q[j + 4] >> 4) | ((q[j] >> 6) << 4); }
}
__device__ __forceinline__ float dot_q4_k(const u8* sb, int j, const float* x) {
    float d = h2f(ld16(sb)), dmin = h2f(ld16(sb + 2)); int sc, mn; scale_min_k4(j, sb + 4, &sc, &mn);
    const u8* q = sb + 16 + 32 * (j >> 1); int hi = j & 1; float sq = 0.f, sx = 0.f;
    #pragma unroll
    for (int l = 0; l < 32; ++l) {
        int v = hi ? (q[l] >> 4) : (q[l] & 15);
        sq += (float)v * x[l]; sx += x[l];
    }
    return d * (float)sc * sq - dmin * (float)mn * sx;
}
__device__ __forceinline__ float dot_q5_k(const u8* sb, int j, const float* x) {
    float d = h2f(ld16(sb)), dmin = h2f(ld16(sb + 2)); int sc, mn; scale_min_k4(j, sb + 4, &sc, &mn);
    const u8* qh = sb + 16; const u8* ql = sb + 48 + 32 * (j >> 1); int hi = j & 1; u8 bit = (u8)(1u << j);
    float sq = 0.f, sx = 0.f;
    #pragma unroll
    for (int l = 0; l < 32; ++l) {
        int v = (hi ? (ql[l] >> 4) : (ql[l] & 15)) + ((qh[l] & bit) ? 16 : 0);
        sq += (float)v * x[l]; sx += x[l];
    }
    return d * (float)sc * sq - dmin * (float)mn * sx;
}
__device__ __forceinline__ float dot_q6_k(const u8* sb, int j, const float* x) {
    int half = j >> 2, r = j & 3;
    const u8* ql = sb + 64 * half + 32 * (r & 1); const u8* qh = sb + 128 + 32 * half;
    const i8* sc = (const i8*)(sb + 192) + 8 * half; float d = h2f(ld16(sb + 208));
    float out = 0.f;
    #pragma unroll
    for (int h = 0; h < 2; ++h) {
        float s = 0.f;
        #pragma unroll
        for (int l = 16 * h; l < 16 * h + 16; ++l) {
            int nib = (r >= 2) ? (ql[l] >> 4) : (ql[l] & 15);
            int q = (nib | (((qh[l] >> (2 * r)) & 3) << 4)) - 32;
            s += (float)q * x[l];
        }
        out += (float)sc[2 * r + h] * s;
    }
    return d * out;
}

#define MV_KERNEL(NAME, BLK_BYTES, BLK_ELEMS, CHUNKS, FN)                                     \
extern "C" __global__ void NAME(const u8* __restrict__ W, const float* __restrict__ x,       \
                                float* __restrict__ y, int k, int n) {                       \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                          \
    int lane = threadIdx.x & 31;                                                             \
    if (row >= n) return;                                                                    \
    int nblk = k / BLK_ELEMS;                                                                \
    const u8* r = W + (size_t)row * (size_t)nblk * BLK_BYTES;                               \
    int nch = nblk * CHUNKS; float s = 0.f;                                                  \
    for (int c = lane; c < nch; c += 32) {                                                   \
        int b = c / CHUNKS, j = c % CHUNKS;                                                  \
        s += FN(r + (size_t)b * BLK_BYTES, j, x + (size_t)b * BLK_ELEMS + 32 * j);           \
    }                                                                                        \
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);              \
    if (lane == 0) y[row] = s;                                                               \
}

MV_KERNEL(mv_q4_0, 18, 32, 1, dot_q4_0)
MV_KERNEL(mv_q4_1, 20, 32, 1, dot_q4_1)
MV_KERNEL(mv_q5_0, 22, 32, 1, dot_q5_0)
MV_KERNEL(mv_q5_1, 24, 32, 1, dot_q5_1)
MV_KERNEL(mv_q8_0, 34, 32, 1, dot_q8_0)
MV_KERNEL(mv_q2_k, 84, 256, 8, dot_q2_k)
MV_KERNEL(mv_q3_k, 110, 256, 8, dot_q3_k)
MV_KERNEL(mv_q4_k, 144, 256, 8, dot_q4_k)
MV_KERNEL(mv_q5_k, 176, 256, 8, dot_q5_k)
MV_KERNEL(mv_q6_k, 210, 256, 8, dot_q6_k)

// f16 / bf16 weights (2 bytes per element, any k).
extern "C" __global__ void mv_f16(const u8* __restrict__ W, const float* __restrict__ x,
                                  float* __restrict__ y, int k, int n, int is_bf16) {
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (row >= n) return;
    const u16* r = (const u16*)(W + (size_t)row * (size_t)k * 2);
    float s = 0.f;
    for (int i = lane; i < k; i += 32) {
        u16 h = r[i];
        float w = is_bf16 ? __uint_as_float(((u32)h) << 16) : h2f(h);
        s += w * x[i];
    }
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
    if (lane == 0) y[row] = s;
}

// ---- Numerically stable softmax over `rows` rows of length n (one block per row). ----
// Applies v = s*scale, optional softcap (cap*tanh(v/cap)), then max-subtract exp / sum.
__device__ float block_reduce(float v, int is_max) {
    __shared__ float sh[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    for (int o = 16; o > 0; o >>= 1) {
        float t = __shfl_xor_sync(0xffffffffu, v, o);
        v = is_max ? fmaxf(v, t) : v + t;
    }
    __syncthreads();
    if (lane == 0) sh[w] = v;
    __syncthreads();
    float ident = is_max ? -3.402823466e38f : 0.f;
    v = (threadIdx.x < nw) ? sh[threadIdx.x] : ident;
    if (w == 0) {
        for (int o = 16; o > 0; o >>= 1) {
            float t = __shfl_xor_sync(0xffffffffu, v, o);
            v = is_max ? fmaxf(v, t) : v + t;
        }
        if (lane == 0) sh[0] = v;
    }
    __syncthreads();
    v = sh[0];
    __syncthreads();
    return v;
}
extern "C" __global__ void softmax_rows(float* s, int n, float scale, float softcap) {
    float* row = s + (size_t)blockIdx.x * n;
    float mx = -3.402823466e38f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float v = row[i] * scale;
        if (softcap > 0.f) v = softcap * tanhf(v / softcap);
        row[i] = v;
        mx = fmaxf(mx, v);
    }
    mx = block_reduce(mx, 1);
    float sum = 0.f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        float e = expf(row[i] - mx);
        row[i] = e;
        sum += e;
    }
    sum = block_reduce(sum, 0);
    float inv = 1.f / sum;
    for (int i = threadIdx.x; i < n; i += blockDim.x) row[i] *= inv;
}

// ===== DEVICE-FORWARD KERNELS (device KV cache + attention) =====
// f32 -> f16, IEEE-754 round to nearest, ties to even (PTX cvt.rn). Bit-identical to the host
// crate::quant::f32_to_f16 on every non-NaN input, so device f16 KV rows equal the host KvCache.
__device__ __forceinline__ u16 f2h_rn(float f) {
    u16 h;
    asm("cvt.rn.f16.f32 %0, %1;" : "=h"(h) : "f"(f));
    return h;
}

// dst[i] = f16(src[i]) for i < n.
extern "C" __global__ void store_f16(const float* __restrict__ src, u16* __restrict__ dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = f2h_rn(src[i]);
}

__device__ __forceinline__ float kv_ld(const float* p, size_t i) { return p[i]; }
__device__ __forceinline__ float kv_ld(const u16* p, size_t i) { return h2f(p[i]); }

// Multi-head attention over the device KV cache, one block per (head, row).
// Row r is query position pos0 + r and attends [start, pos0 + r] (causal; sliding window if window > 0).
// q/out: [rows][q_stride]; K/V: [n_ctx][kv_dim] (f32 or f16); att scratch: [rows][n_head][n_ctx].
// Same math/order as host multi_head_attn: s = dot(q,k) * scale -> softcap -> max-subtracted softmax -> sum p*v.
template <typename KT>
__device__ void attn_rows_impl(const float* __restrict__ q, const KT* __restrict__ K, const KT* __restrict__ V,
                               float* __restrict__ out, float* __restrict__ att, int head_dim, int kv_dim,
                               int kv_mul, int q_stride, int n_ctx, int pos0, int window, float scale,
                               float softcap) {
    int h = blockIdx.x, r = blockIdx.y, n_head = gridDim.x;
    int pos = pos0 + r;
    int start = (window > 0) ? max(0, pos - window + 1) : 0;
    int len = pos + 1 - start;
    int hkv = h / kv_mul;
    const float* qh = q + (size_t)r * q_stride + (size_t)h * head_dim;
    float* a = att + ((size_t)r * n_head + h) * (size_t)n_ctx;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = blockDim.x >> 5;
    for (int ti = w; ti < len; ti += nw) {
        size_t base = (size_t)(start + ti) * kv_dim + (size_t)hkv * head_dim;
        float s = 0.f;
        for (int d = lane; d < head_dim; d += 32) s += qh[d] * kv_ld(K, base + d);
        for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);
        if (lane == 0) {
            float v = s * scale;
            if (softcap > 0.f) v = softcap * tanhf(v / softcap);
            a[ti] = v;
        }
    }
    __syncthreads();
    float mx = -3.402823466e38f;
    for (int i = threadIdx.x; i < len; i += blockDim.x) mx = fmaxf(mx, a[i]);
    mx = block_reduce(mx, 1);
    float sum = 0.f;
    for (int i = threadIdx.x; i < len; i += blockDim.x) {
        float e = expf(a[i] - mx);
        a[i] = e;
        sum += e;
    }
    sum = block_reduce(sum, 0);
    float inv = (sum == 0.f) ? 0.f : 1.f / sum;
    for (int i = threadIdx.x; i < len; i += blockDim.x) a[i] *= inv;
    __syncthreads();
    float* oh = out + (size_t)r * q_stride + (size_t)h * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int ti = 0; ti < len; ++ti)
            acc += a[ti] * kv_ld(V, (size_t)(start + ti) * kv_dim + (size_t)hkv * head_dim + d);
        oh[d] = acc;
    }
}
extern "C" __global__ void attn_rows_f32(const float* q, const float* K, const float* V, float* out, float* att,
                                         int head_dim, int kv_dim, int kv_mul, int q_stride, int n_ctx, int pos0,
                                         int window, float scale, float softcap) {
    attn_rows_impl<float>(q, K, V, out, att, head_dim, kv_dim, kv_mul, q_stride, n_ctx, pos0, window, scale, softcap);
}
extern "C" __global__ void attn_rows_f16(const float* q, const u16* K, const u16* V, float* out, float* att,
                                         int head_dim, int kv_dim, int kv_mul, int q_stride, int n_ctx, int pos0,
                                         int window, float scale, float softcap) {
    attn_rows_impl<u16>(q, K, V, out, att, head_dim, kv_dim, kv_mul, q_stride, n_ctx, pos0, window, scale, softcap);
}

// ---- mmvq-style matvec (Q8 activations + dp4a integer dots) ----
// Activations are quantized once per matvec input into 32-element blocks:
//   xq[k] int8 (row-major), xds[k/32] = (d, d * sum(q)) with d = amax / 127, q = round(x / d).
// Weight blocks are read as-is; each lane handles one 32-element sub-block per iteration.
extern "C" __global__ void quant_q8(const float* __restrict__ x, signed char* __restrict__ xq,
                                    float2* __restrict__ xds, int k) {
    int b = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    int lane = threadIdx.x & 31;
    if (b * 32 >= k) return;
    float v = x[b * 32 + lane];
    float a = fabsf(v);
    for (int o = 16; o > 0; o >>= 1) a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, o));
    float d = a / 127.f;
    int q = (d == 0.f) ? 0 : __float2int_rn(v / d);
    int sum = q;
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    xq[b * 32 + lane] = (signed char)q;
    if (lane == 0) xds[b] = make_float2(d, d * (float)sum);
}

__device__ __forceinline__ void ld_xq(const signed char* p, int* v) {
    const int4* q = (const int4*)p;
    int4 a = __ldg(q), b = __ldg(q + 1);
    v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
}
__device__ __forceinline__ int ld2x16(const u16* p) { return (int)((u32)__ldg(p) | ((u32)__ldg(p + 1) << 16)); }

// Q8_0: 34 B = f16 d + 32 int8.
__device__ __forceinline__ float vd_q8_0(const u8* blk, int, const int* xv, float2 xd) {
    const u16* p = (const u16*)blk;
    float d = h2f(__ldg(p));
    int acc = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc = __dp4a(ld2x16(p + 1 + 2 * i), xv[i], acc);
    return d * xd.x * (float)acc;
}
// spread 4 bits of t into bit 4 of each byte
__device__ __forceinline__ int spread4(u32 t) {
    return (int)(((t & 1u) << 4) | ((t & 2u) << 11) | ((t & 4u) << 18) | ((t & 8u) << 25));
}
// Q5_0: 22 B = f16 d + u32 qh + 16 B nibbles; value = (nib | hbit << 4) - 16.
__device__ __forceinline__ float vd_q5_0(const u8* blk, int, const int* xv, float2 xd) {
    const u16* p = (const u16*)blk;
    float d = h2f(__ldg(p));
    u32 qh = (u32)ld2x16(p + 1);
    int acc = 0;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int q = ld2x16(p + 3 + 2 * i);
        int lo = (q & 0x0F0F0F0F) | spread4((qh >> (4 * i)) & 0xFu);
        int hi = ((q >> 4) & 0x0F0F0F0F) | spread4((qh >> (16 + 4 * i)) & 0xFu);
        acc = __dp4a((int)__vsub4(lo, 0x10101010), xv[i], acc);
        acc = __dp4a((int)__vsub4(hi, 0x10101010), xv[4 + i], acc);
    }
    return d * xd.x * (float)acc;
}
// Q4_K: 144 B super-block (16-B aligned): d, dmin, 12 B scales, 128 B nibbles. Sub-block j of 8.
__device__ __forceinline__ float vd_q4_k(const u8* sb, int j, const int* xv, float2 xd) {
    const u16* p = (const u16*)sb;
    float d = h2f(__ldg(p)), dmin = h2f(__ldg(p + 1));
    int sc, mn;
    scale_min_k4(j, sb + 4, &sc, &mn);
    const int4* q4 = (const int4*)(sb + 16 + 32 * (j >> 1));
    int4 a = __ldg(q4), b = __ldg(q4 + 1);
    int q[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    int sh = (j & 1) * 4;
    int acc = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) acc = __dp4a((q[i] >> sh) & 0x0F0F0F0F, xv[i], acc);
    return d * (float)sc * xd.x * (float)acc - dmin * (float)mn * xd.y;
}
// Q6_K: 210 B super-block: ql[128], qh[64], int8 scales[16], f16 d. Sub-block j = (half, r).
__device__ __forceinline__ float vd_q6_k(const u8* sb, int j, const int* xv, float2 xd) {
    int half = j >> 2, r = j & 3;
    const u16* ql = (const u16*)(sb + 64 * half + 32 * (r & 1));
    const u16* qh = (const u16*)(sb + 128 + 32 * half);
    const i8* sc = (const i8*)(sb + 192) + 8 * half + 2 * r;
    float d = h2f(__ldg((const u16*)(sb + 208)));
    int nsh = (r >= 2) ? 4 : 0, hsh = 2 * r;
    int acc0 = 0, acc1 = 0;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        int l = (ld2x16(ql + 2 * i) >> nsh) & 0x0F0F0F0F;
        int h = ((ld2x16(qh + 2 * i) >> hsh) & 0x03030303) << 4;
        int v = (int)__vsub4(l | h, 0x20202020);
        if (i < 4) acc0 = __dp4a(v, xv[i], acc0);
        else acc1 = __dp4a(v, xv[i], acc1);
    }
    return d * xd.x * ((float)sc[0] * (float)acc0 + (float)sc[1] * (float)acc1);
}

// One warp per output row (blockDim.x/32 rows per block); lane c walks 32-element sub-blocks.
#define MMVQ_KERNEL(NAME, BLK_BYTES, SUBS, FN)                                                   \
extern "C" __global__ void NAME(const u8* __restrict__ W, const signed char* __restrict__ xq,   \
                                const float2* __restrict__ xds, float* __restrict__ y, int k,   \
                                int n) {                                                        \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                             \
    int lane = threadIdx.x & 31;                                                                \
    if (row >= n) return;                                                                       \
    int nsub = k >> 5;                                                                          \
    const u8* r = W + (size_t)row * (size_t)(nsub / SUBS) * BLK_BYTES;                          \
    float s = 0.f;                                                                              \
    for (int c = lane; c < nsub; c += 32) {                                                     \
        int xv[8];                                                                              \
        ld_xq(xq + 32 * c, xv);                                                                 \
        s += FN(r + (size_t)(c / SUBS) * BLK_BYTES, c % SUBS, xv, xds[c]);                      \
    }                                                                                           \
    for (int o = 16; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffffu, s, o);                 \
    if (lane == 0) y[row] = s;                                                                  \
}
MMVQ_KERNEL(mmvq_q8_0, 34, 1, vd_q8_0)
MMVQ_KERNEL(mmvq_q5_0, 22, 1, vd_q5_0)
MMVQ_KERNEL(mmvq_q4_k, 144, 8, vd_q4_k)
MMVQ_KERNEL(mmvq_q6_k, 210, 8, vd_q6_k)

// ---- Batched prefill: dequant weight tiles to f16 + activation f16 rows (cuBLAS GemmEx) ----
// deq_*: chunk j (32 elements) of one block -> v[32] f32 (layouts as the dot_* helpers above).
__device__ __forceinline__ void deq_q4_0(const u8* b, int, float* v) {
    float d = h2f(ld16(b)); const u8* qs = b + 2;
    for (int j = 0; j < 16; ++j) { v[j] = d * (float)((int)(qs[j] & 15) - 8); v[j + 16] = d * (float)((int)(qs[j] >> 4) - 8); }
}
__device__ __forceinline__ void deq_q4_1(const u8* b, int, float* v) {
    float d = h2f(ld16(b)), m = h2f(ld16(b + 2)); const u8* qs = b + 4;
    for (int j = 0; j < 16; ++j) { v[j] = d * (float)(qs[j] & 15) + m; v[j + 16] = d * (float)(qs[j] >> 4) + m; }
}
__device__ __forceinline__ void deq_q5_0(const u8* b, int, float* v) {
    float d = h2f(ld16(b)); u32 qh = ld32(b + 2); const u8* qs = b + 6;
    for (int j = 0; j < 16; ++j) {
        v[j] = d * (float)((int)((qs[j] & 15) | (((qh >> j) << 4) & 0x10)) - 16);
        v[j + 16] = d * (float)((int)((qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) - 16);
    }
}
__device__ __forceinline__ void deq_q5_1(const u8* b, int, float* v) {
    float d = h2f(ld16(b)), m = h2f(ld16(b + 2)); u32 qh = ld32(b + 4); const u8* qs = b + 8;
    for (int j = 0; j < 16; ++j) {
        v[j] = d * (float)(int)((qs[j] & 15) | (((qh >> j) << 4) & 0x10)) + m;
        v[j + 16] = d * (float)(int)((qs[j] >> 4) | ((qh >> (j + 12)) & 0x10)) + m;
    }
}
__device__ __forceinline__ void deq_q8_0(const u8* b, int, float* v) {
    float d = h2f(ld16(b)); const i8* qs = (const i8*)(b + 2);
    for (int j = 0; j < 32; ++j) v[j] = d * (float)qs[j];
}
__device__ __forceinline__ void deq_q2_k(const u8* sb, int j, float* v) {
    const u8* sc = sb; const u8* q = sb + 16 + 32 * (j >> 2); int shift = 2 * (j & 3);
    float d = h2f(ld16(sb + 80)), dmin = h2f(ld16(sb + 82));
    for (int h = 0; h < 2; ++h) {
        u8 s8 = sc[2 * j + h]; float dl = d * (float)(s8 & 15), ml = dmin * (float)(s8 >> 4);
        for (int l = 0; l < 16; ++l) v[16 * h + l] = dl * (float)((q[16 * h + l] >> shift) & 3) - ml;
    }
}
__device__ __forceinline__ void deq_q3_k(const u8* sb, int j, float* v) {
    const u8* hm = sb; const u8* q = sb + 32 + 32 * (j >> 2); int shift = 2 * (j & 3);
    u8 m = (u8)(1u << j); float d = h2f(ld16(sb + 108));
    for (int h = 0; h < 2; ++h) {
        float dl = d * (float)(q3k_scale(sb + 96, 2 * j + h) - 32);
        for (int l = 16 * h; l < 16 * h + 16; ++l)
            v[l] = dl * (float)((int)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }
}
__device__ __forceinline__ void deq_q4_k(const u8* sb, int j, float* v) {
    float d = h2f(ld16(sb)), dmin = h2f(ld16(sb + 2)); int sc, mn; scale_min_k4(j, sb + 4, &sc, &mn);
    const u8* q = sb + 16 + 32 * (j >> 1); int hi = j & 1; float dl = d * (float)sc, ml = dmin * (float)mn;
    for (int l = 0; l < 32; ++l) v[l] = dl * (float)(hi ? (q[l] >> 4) : (q[l] & 15)) - ml;
}
__device__ __forceinline__ void deq_q5_k(const u8* sb, int j, float* v) {
    float d = h2f(ld16(sb)), dmin = h2f(ld16(sb + 2)); int sc, mn; scale_min_k4(j, sb + 4, &sc, &mn);
    const u8* qh = sb + 16; const u8* ql = sb + 48 + 32 * (j >> 1); int hi = j & 1; u8 bit = (u8)(1u << j);
    float dl = d * (float)sc, ml = dmin * (float)mn;
    for (int l = 0; l < 32; ++l)
        v[l] = dl * (float)((hi ? (ql[l] >> 4) : (ql[l] & 15)) + ((qh[l] & bit) ? 16 : 0)) - ml;
}
__device__ __forceinline__ void deq_q6_k(const u8* sb, int j, float* v) {
    int half = j >> 2, r = j & 3;
    const u8* ql = sb + 64 * half + 32 * (r & 1); const u8* qh = sb + 128 + 32 * half;
    const i8* sc = (const i8*)(sb + 192) + 8 * half; float d = h2f(ld16(sb + 208));
    for (int h = 0; h < 2; ++h) {
        float dl = d * (float)sc[2 * r + h];
        for (int l = 16 * h; l < 16 * h + 16; ++l) {
            int nib = (r >= 2) ? (ql[l] >> 4) : (ql[l] & 15);
            v[l] = dl * (float)((nib | (((qh[l] >> (2 * r)) & 3) << 4)) - 32);
        }
    }
}
__device__ __forceinline__ u32 pack_h2(float a, float b) { return (u32)f2h_rn(a) | ((u32)f2h_rn(b) << 16); }

// out[r][0..k) = f16(dequant(W row row0 + r)), r < nrows. One thread per 32-element chunk.
#define DEQ_KERNEL(NAME, BLK_BYTES, BLK_ELEMS, CHUNKS, FN)                                          \
extern "C" __global__ void NAME(const u8* __restrict__ W, u16* __restrict__ out, int k, int row0,  \
                                int nrows) {                                                       \
    int cpr = k >> 5;                                                                              \
    long long c = (long long)blockIdx.x * blockDim.x + threadIdx.x;                                \
    if (c >= (long long)nrows * cpr) return;                                                       \
    int rr = (int)(c / cpr), cc = (int)(c % cpr);                                                  \
    const u8* r = W + (size_t)(row0 + rr) * (size_t)(k / BLK_ELEMS) * BLK_BYTES;                   \
    float v[32];                                                                                   \
    FN(r + (size_t)(cc / CHUNKS) * BLK_BYTES, cc % CHUNKS, v);                                     \
    uint4* o = (uint4*)(out + (size_t)rr * k + (size_t)cc * 32);                                   \
    _Pragma("unroll") for (int q = 0; q < 4; ++q)                                                  \
        o[q] = make_uint4(pack_h2(v[8 * q], v[8 * q + 1]), pack_h2(v[8 * q + 2], v[8 * q + 3]),     \
                          pack_h2(v[8 * q + 4], v[8 * q + 5]), pack_h2(v[8 * q + 6], v[8 * q + 7]));\
}
DEQ_KERNEL(deq_f16_q4_0, 18, 32, 1, deq_q4_0)
DEQ_KERNEL(deq_f16_q4_1, 20, 32, 1, deq_q4_1)
DEQ_KERNEL(deq_f16_q5_0, 22, 32, 1, deq_q5_0)
DEQ_KERNEL(deq_f16_q5_1, 24, 32, 1, deq_q5_1)
DEQ_KERNEL(deq_f16_q8_0, 34, 32, 1, deq_q8_0)
DEQ_KERNEL(deq_f16_q2_k, 84, 256, 8, deq_q2_k)
DEQ_KERNEL(deq_f16_q3_k, 110, 256, 8, deq_q3_k)
DEQ_KERNEL(deq_f16_q4_k, 144, 256, 8, deq_q4_k)
DEQ_KERNEL(deq_f16_q5_k, 176, 256, 8, deq_q5_k)
DEQ_KERNEL(deq_f16_q6_k, 210, 256, 8, deq_q6_k)

// Activation rows -> f16 with a per-row power-of-two scale keeping |x·s| <= 8192 (f16 range
// headroom for large post-activation values, e.g. ReLU²). One block per row; inv[r] = 1/s.
extern "C" __global__ void rows_to_f16(const float* __restrict__ x, u16* __restrict__ out,
                                       float* __restrict__ inv, int k) {
    const float* xr = x + (size_t)blockIdx.x * k;
    float a = 0.f;
    for (int i = threadIdx.x; i < k; i += blockDim.x) a = fmaxf(a, fabsf(xr[i]));
    a = block_reduce(a, 1);
    float s = 1.f;
    if (a > 8192.f) s = exp2f(-ceilf(log2f(a / 8192.f)));
    u16* o = out + (size_t)blockIdx.x * k;
    for (int i = threadIdx.x; i < k; i += blockDim.x) o[i] = f2h_rn(xr[i] * s);
    if (threadIdx.x == 0) inv[blockIdx.x] = 1.f / s;
}
// y[r][0..n) *= inv[r] for r < rows (undo the rows_to_f16 scale after the GEMM).
extern "C" __global__ void rows_scale(float* __restrict__ y, const float* __restrict__ inv, int n, int rows) {
    long long total = (long long)n * rows;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        float f = inv[i / n];
        if (f != 1.f) y[i] *= f;
    }
}

// Multi-column mmvq (batched prefill, small N): y[c][row] = W[row] · x[c] for up to 8 activation
// rows c per blockIdx.y slice; each weight sub-block is fetched once per slice (L1 for the rest).
// xq/xds: per-row Q8 activations as produced by quant_q8 over [ncols × k] (k % 32 == 0).
#define MMVQ_MC_KERNEL(NAME, BLK_BYTES, SUBS, FN)                                                 \
extern "C" __global__ void NAME(const u8* __restrict__ W, const signed char* __restrict__ xq,   \
                                const float2* __restrict__ xds, float* __restrict__ y, int k,   \
                                int n, int ncols) {                                             \
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);                             \
    int lane = threadIdx.x & 31;                                                                \
    if (row >= n) return;                                                                       \
    int nsub = k >> 5;                                                                          \
    int c0 = blockIdx.y * 8, nc = min(8, ncols - c0);                                           \
    const u8* r = W + (size_t)row * (size_t)(nsub / SUBS) * BLK_BYTES;                          \
    float s[8];                                                                                 \
    _Pragma("unroll") for (int q = 0; q < 8; ++q) s[q] = 0.f;                                   \
    for (int c = lane; c < nsub; c += 32) {                                                     \
        const u8* blk = r + (size_t)(c / SUBS) * BLK_BYTES;                                     \
        _Pragma("unroll") for (int q = 0; q < 8; ++q) {                                         \
            if (q < nc) {                                                                       \
                int xv[8];                                                                      \
                size_t col = (size_t)(c0 + q);                                                  \
                ld_xq(xq + col * k + 32 * c, xv);                                               \
                s[q] += FN(blk, c % SUBS, xv, xds[col * nsub + c]);                             \
            }                                                                                   \
        }                                                                                       \
    }                                                                                           \
    _Pragma("unroll") for (int q = 0; q < 8; ++q) {                                             \
        float v = s[q];                                                                         \
        for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);             \
        if (lane == 0 && q < nc) y[(size_t)(c0 + q) * n + row] = v;                             \
    }                                                                                           \
}
MMVQ_MC_KERNEL(mmvq8_q8_0, 34, 1, vd_q8_0)
MMVQ_MC_KERNEL(mmvq8_q5_0, 22, 1, vd_q5_0)
MMVQ_MC_KERNEL(mmvq8_q4_k, 144, 8, vd_q4_k)
MMVQ_MC_KERNEL(mmvq8_q6_k, 210, 8, vd_q6_k)
// ---- Split-K flash-decode attention (single query row) ----
// grid (n_head_kv, n_split), FD_WARPS warps per block. Block (hkv, s) covers KV positions
// [start + s*chunk, start + min((s+1)*chunk, len)) for ALL kv_mul query heads of KV head hkv,
// so each K/V row is read once per KV head (GQA). Lane l owns dims [l*vec, l*vec + vec).
// Each warp keeps an online softmax (m, l, o[vec]) per query head over its keys (t = w, w+NW, …)
// and writes one partial: part[((h * n_part) + p) * (head_dim + 2) + {0..head_dim-1 = o, hd = m, hd+1 = l}],
// p = s * FD_WARPS + w. fd_reduce merges the n_part partials of each query head.
#define FD_WARPS 4
#define FD_MAXG 8
#define FD_MAXVEC 8
template <typename KT>
__device__ void fd_partial_impl(const float* __restrict__ q, const KT* __restrict__ K, const KT* __restrict__ V,
                                float* __restrict__ part, int head_dim, int kv_dim, int kv_mul, int start, int len,
                                int chunk, float scale, float softcap) {
    __shared__ float qs[FD_MAXG * FD_MAXVEC * 32];
    int hkv = blockIdx.x, s = blockIdx.y, n_split = gridDim.y;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    int vec = head_dim >> 5;
    int h0 = hkv * kv_mul;
    for (int i = threadIdx.x; i < kv_mul * head_dim; i += blockDim.x) qs[i] = q[(size_t)h0 * head_dim + i];
    __syncthreads();
    float m[FD_MAXG], l[FD_MAXG], o[FD_MAXG][FD_MAXVEC];
    #pragma unroll
    for (int g = 0; g < FD_MAXG; ++g) {
        m[g] = -3.402823466e38f; l[g] = 0.f;
        #pragma unroll
        for (int j = 0; j < FD_MAXVEC; ++j) o[g][j] = 0.f;
    }
    int t0 = s * chunk, t1 = min(len, t0 + chunk);
    int d0 = lane * vec;
    for (int t = t0 + w; t < t1; t += FD_WARPS) {
        size_t base = (size_t)(start + t) * kv_dim + (size_t)hkv * head_dim + d0;
        float kr[FD_MAXVEC], vr[FD_MAXVEC];
        #pragma unroll
        for (int j = 0; j < FD_MAXVEC; ++j) {
            kr[j] = j < vec ? kv_ld(K, base + j) : 0.f;
            vr[j] = j < vec ? kv_ld(V, base + j) : 0.f;
        }
        #pragma unroll
        for (int g = 0; g < FD_MAXG; ++g) {
            if (g < kv_mul) {
                const float* qg = qs + g * head_dim + d0;
                float dot = 0.f;
                #pragma unroll
                for (int j = 0; j < FD_MAXVEC; ++j) if (j < vec) dot = fmaf(qg[j], kr[j], dot);
                for (int off = 16; off > 0; off >>= 1) dot += __shfl_xor_sync(0xffffffffu, dot, off);
                float sc = dot * scale;
                if (softcap > 0.f) sc = softcap * tanhf(sc / softcap);
                float mn = fmaxf(m[g], sc);
                float corr = expf(m[g] - mn), p = expf(sc - mn);
                l[g] = l[g] * corr + p;
                #pragma unroll
                for (int j = 0; j < FD_MAXVEC; ++j) o[g][j] = o[g][j] * corr + p * vr[j];
                m[g] = mn;
            }
        }
    }
    int n_part = n_split * FD_WARPS, pidx = s * FD_WARPS + w;
    #pragma unroll
    for (int g = 0; g < FD_MAXG; ++g) {
        if (g < kv_mul) {
            float* pp = part + ((size_t)(h0 + g) * n_part + pidx) * (head_dim + 2);
            #pragma unroll
            for (int j = 0; j < FD_MAXVEC; ++j) if (j < vec) pp[d0 + j] = o[g][j];
            if (lane == 0) { pp[head_dim] = m[g]; pp[head_dim + 1] = l[g]; }
        }
    }
}
extern "C" __global__ void fd_partial_f32(const float* q, const float* K, const float* V, float* part, int head_dim,
                                          int kv_dim, int kv_mul, int start, int len, int chunk, float scale,
                                          float softcap) {
    fd_partial_impl<float>(q, K, V, part, head_dim, kv_dim, kv_mul, start, len, chunk, scale, softcap);
}
extern "C" __global__ void fd_partial_f16(const float* q, const u16* K, const u16* V, float* part, int head_dim,
                                          int kv_dim, int kv_mul, int start, int len, int chunk, float scale,
                                          float softcap) {
    fd_partial_impl<u16>(q, K, V, part, head_dim, kv_dim, kv_mul, start, len, chunk, scale, softcap);
}
// Merge: grid n_head, head_dim threads. out[h*head_dim + d] = Σ_p o_p[d] e^{m_p - M} / Σ_p l_p e^{m_p - M}.
extern "C" __global__ void fd_reduce(const float* __restrict__ part, float* __restrict__ out, int head_dim,
                                     int n_part) {
    int h = blockIdx.x, d = threadIdx.x;
    const float* ph = part + (size_t)h * n_part * (head_dim + 2);
    float M = -3.402823466e38f;
    for (int p = 0; p < n_part; ++p) M = fmaxf(M, ph[(size_t)p * (head_dim + 2) + head_dim]);
    float L = 0.f, acc = 0.f;
    for (int p = 0; p < n_part; ++p) {
        const float* pp = ph + (size_t)p * (head_dim + 2);
        float lp = pp[head_dim + 1];
        if (lp == 0.f) continue;  // empty partial (no keys)
        float wgt = expf(pp[head_dim] - M);
        L = fmaf(lp, wgt, L);
        if (d < head_dim) acc = fmaf(pp[d], wgt, acc);
    }
    if (d < head_dim) out[(size_t)h * head_dim + d] = (L > 0.f) ? acc / L : 0.f;
}
// ---- Flash-attention prefill, scalar-tiled (fp32 math) ----
// FlashAttention-2 style causal prefill over the device KV cache, no score materialisation.
// Grid (n_head, ceil(n_rows / FA_BR)); block FA_WARPS warps. Block (h, qb) owns query rows
// q0 = qb * FA_BR .. q0 + FA_BR - 1 (positions pos0 + row) of head h and streams the K/V tiles
// [t * FA_BC, (t + 1) * FA_BC) of kv head h / kv_mul from 0 up to its last query position.
// Smem: the Q tile in f32 (32 KB) + one f16 K/V tile buffer (16 KB, K then V) = 48 KB static.
// f16 KV is copied as-is; f32 KV is rounded to f16 per tile (the same operand format the
// tensor-core version below uses). All arithmetic (QK dot, scale/softcap, row max,
// row sum, P*V accumulation) is fp32. Online softmax across K tiles; only tiles that cross the
// diagonal (last key > the block's first query position) are masked. Tail rows (>= n_rows) are
// computed on zero Q and never written. Each warp owns FA_RPW query rows: in QK a lane owns
// keys (lane, lane + 32), in PV a lane owns dims (2*lane, 2*lane+1, 64+2*lane, 65+2*lane).
// Host model of the tile plan + math: `fa_block_tiles` / `fa_reference` in cuda_attention.rs.
#define FA_BR 64
#define FA_BC 64
#define FA_D 128
#define FA_WARPS 4
#define FA_RPW (FA_BR / FA_WARPS)
#define FA_DW (FA_D / 2)

__device__ __forceinline__ float2 fa_h2f2(u32 x) { return make_float2(h2f((u16)(x & 0xFFFFu)), h2f((u16)(x >> 16))); }
// Two consecutive K/V elements (i even) as a packed f16 pair.
__device__ __forceinline__ u32 fa_ld2(const u16* p, size_t i) { return *(const u32*)(p + i); }
__device__ __forceinline__ u32 fa_ld2(const float* p, size_t i) {
    return pack_h2(p[i], p[i + 1]);  // f2h_rn / pack_h2: fd section above
}
// Swizzled word index of (key j, word w) in the K/V tile: lanes reading key j = lane (QK) or
// word w = lane (PV) hit 32 distinct banks.
__device__ __forceinline__ int fa_sw(int j, int w) { return j * FA_DW + (w ^ (j & 31)); }

template <typename KT>
__device__ void fa_prefill_impl(const float* __restrict__ q, const KT* __restrict__ K, const KT* __restrict__ V,
                                float* __restrict__ out, int n_rows, int kv_dim, int kv_mul, int q_stride,
                                int pos0, float scale, float softcap) {
    __shared__ float qs[FA_BR * FA_D];
    __shared__ u32 kvs[FA_BC * FA_DW];
    const float NEG_INF = __int_as_float(0xff800000);
    const int h = blockIdx.x;
    const int qb = gridDim.y - 1 - blockIdx.y;  // latest (longest) query tiles first
    const int q0 = qb * FA_BR;
    const int rows_here = min(FA_BR, n_rows - q0);
    const size_t kvh = (size_t)(h / kv_mul) * FA_D;
    const int tid = threadIdx.x, lane = tid & 31, rbase = (tid >> 5) * FA_RPW;
    for (int i = tid; i < FA_BR * FA_D; i += blockDim.x) {
        int r = i / FA_D, d = i - r * FA_D;
        qs[i] = (r < rows_here) ? q[(size_t)(q0 + r) * q_stride + (size_t)h * FA_D + d] : 0.f;
    }
    const int kv_end = pos0 + n_rows;  // keys [0, kv_end) are in the cache
    const int qp_min = pos0 + q0;      // the block's first query position
    const int n_tiles = (qp_min + rows_here - 1) / FA_BC + 1;
    float m[FA_RPW], l[FA_RPW], o[FA_RPW][4];
#pragma unroll
    for (int r = 0; r < FA_RPW; ++r) {
        m[r] = NEG_INF;
        l[r] = 0.f;
        o[r][0] = o[r][1] = o[r][2] = o[r][3] = 0.f;
    }
    for (int t = 0; t < n_tiles; ++t) {
        const int k0 = t * FA_BC;
        const bool need_mask = k0 + FA_BC - 1 > qp_min;  // diagonal tile(s) only
        __syncthreads();  // Q stores visible / previous V tile fully read
        for (int i = tid; i < FA_BC * FA_DW; i += blockDim.x) {
            int j = i / FA_DW, w = i - j * FA_DW, kp = k0 + j;
            kvs[fa_sw(j, w)] = (kp < kv_end) ? fa_ld2(K, (size_t)kp * kv_dim + kvh + 2 * w) : 0u;
        }
        __syncthreads();
        float s[FA_RPW][2];
#pragma unroll
        for (int r = 0; r < FA_RPW; ++r) s[r][0] = s[r][1] = 0.f;
        for (int w = 0; w < FA_DW; ++w) {
            float2 ka = fa_h2f2(kvs[fa_sw(lane, w)]), kb = fa_h2f2(kvs[fa_sw(lane + 32, w)]);
#pragma unroll
            for (int r = 0; r < FA_RPW; ++r) {
                float2 qv = *(const float2*)&qs[(rbase + r) * FA_D + 2 * w];
                s[r][0] = fmaf(qv.y, ka.y, fmaf(qv.x, ka.x, s[r][0]));
                s[r][1] = fmaf(qv.y, kb.y, fmaf(qv.x, kb.x, s[r][1]));
            }
        }
        // Online softmax (fp32): scale -> softcap -> mask -> running max / sum rescale.
#pragma unroll
        for (int r = 0; r < FA_RPW; ++r) {
            const int qp = qp_min + rbase + r;
            float v0 = s[r][0] * scale, v1 = s[r][1] * scale;
            if (softcap > 0.f) {
                v0 = softcap * tanhf(v0 / softcap);
                v1 = softcap * tanhf(v1 / softcap);
            }
            if (need_mask) {
                const int ka = k0 + lane, kb = k0 + lane + 32;
                if (ka > qp || ka >= kv_end) v0 = NEG_INF;
                if (kb > qp || kb >= kv_end) v1 = NEG_INF;
            }
            float mx = fmaxf(v0, v1);
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, off));
            const float mn = fmaxf(m[r], mx);
            float p0 = 0.f, p1 = 0.f, corr = 1.f;
            if (mn != NEG_INF) {
                corr = expf(m[r] - mn);  // m = -inf -> 0
                p0 = expf(v0 - mn);
                p1 = expf(v1 - mn);
            }
            float ps = p0 + p1;
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) ps += __shfl_xor_sync(0xffffffffu, ps, off);
            l[r] = l[r] * corr + ps;
            m[r] = mn;
            o[r][0] *= corr;
            o[r][1] *= corr;
            o[r][2] *= corr;
            o[r][3] *= corr;
            s[r][0] = p0;
            s[r][1] = p1;
        }
        __syncthreads();  // every warp is done with K
        for (int i = tid; i < FA_BC * FA_DW; i += blockDim.x) {
            int j = i / FA_DW, w = i - j * FA_DW, kp = k0 + j;
            kvs[fa_sw(j, w)] = (kp < kv_end) ? fa_ld2(V, (size_t)kp * kv_dim + kvh + 2 * w) : 0u;
        }
        __syncthreads();
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            for (int jj = 0; jj < 32; ++jj) {
                const int j = half * 32 + jj;
                float2 va = fa_h2f2(kvs[fa_sw(j, lane)]), vb = fa_h2f2(kvs[fa_sw(j, lane + 32)]);
#pragma unroll
                for (int r = 0; r < FA_RPW; ++r) {
                    float p = __shfl_sync(0xffffffffu, s[r][half], jj);
                    o[r][0] = fmaf(p, va.x, o[r][0]);
                    o[r][1] = fmaf(p, va.y, o[r][1]);
                    o[r][2] = fmaf(p, vb.x, o[r][2]);
                    o[r][3] = fmaf(p, vb.y, o[r][3]);
                }
            }
        }
    }
#pragma unroll
    for (int r = 0; r < FA_RPW; ++r) {
        const int row = rbase + r;
        if (row < rows_here) {
            const float inv = (l[r] > 0.f) ? 1.f / l[r] : 0.f;
            float* oh = out + (size_t)(q0 + row) * q_stride + (size_t)h * FA_D;
            oh[2 * lane] = o[r][0] * inv;
            oh[2 * lane + 1] = o[r][1] * inv;
            oh[64 + 2 * lane] = o[r][2] * inv;
            oh[65 + 2 * lane] = o[r][3] * inv;
        }
    }
}
extern "C" __global__ void __launch_bounds__(FA_WARPS * 32) fa_prefill_f16(const float* q, const u16* K, const u16* V,
        float* out, int n_rows, int kv_dim, int kv_mul, int q_stride, int pos0, float scale, float softcap) {
    fa_prefill_impl<u16>(q, K, V, out, n_rows, kv_dim, kv_mul, q_stride, pos0, scale, softcap);
}
extern "C" __global__ void __launch_bounds__(FA_WARPS * 32) fa_prefill_f32kv(const float* q, const float* K,
        const float* V, float* out, int n_rows, int kv_dim, int kv_mul, int q_stride, int pos0, float scale,
        float softcap) {
    fa_prefill_impl<float>(q, K, V, out, n_rows, kv_dim, kv_mul, q_stride, pos0, scale, softcap);
}
// ---- Flash-attention prefill, tensor cores (mma.sync) ----
// Same tiling / masking / GQA / pos0 semantics as fa_prefill_* above (Br = Bc = 64, d = 128,
// 4 warps, grid (n_head, ceil(n_rows / 64)), latest q-blocks first, diagonal-only masking, tail rows
// never written). S = Q K^T and O += P V run on mma.sync.m16n8k16 (f16 x f16 -> f32).
// Precision: S (the mma output), the scale/softcap/mask, m, l and the O accumulator are fp32;
// Q is rounded to f16 once (register A fragments), K/V are f16 (cache) or rounded to f16 per tile,
// P is rounded to f16 only as the PV A operand (l sums the fp32 P).
// Warp w owns query rows 16w..16w+15 of the block; lane = 4*g + t holds rows g and g+8.
// Fragments (PTX ISA m16n8k16 .f16): A a0 (g, 2t..2t+1) a1 (g+8, 2t..) a2 (g, 2t+8..) a3 (g+8, 2t+8..);
// B b0 (k 2t..2t+1, n g) b1 (k 2t+8.., n g); C c0,c1 (g, 2t..2t+1) c2,c3 (g+8, 2t..2t+1).
// K B-fragments: ldmatrix.x4 (K rows = keys are B columns); V B-fragments: ldmatrix.x4.trans.
// Smem: K tile + V tile, 64 keys x 128 dims f16 each = 32 KB static; 16-byte chunks XOR-swizzled by
// (key & 7) so the 8 row addresses of every ldmatrix phase hit distinct banks.
// qflag (debug, may be null): set to 1 if any |Q| element is not representable in f16.
#define FAM_ROWB 128  // f16 per smem row (= FA_D)
__device__ __forceinline__ int fam_idx(int key, int dim) {
    return key * FAM_ROWB + ((((dim >> 3) ^ (key & 7))) << 3) + (dim & 7);
}
__device__ __forceinline__ u32 fam_smem(const void* p) {
    u32 a;
    asm("{ .reg .u64 t; cvta.to.shared.u64 t, %1; cvt.u32.u64 %0, t; }" : "=r"(a) : "l"(p));
    return a;
}
__device__ __forceinline__ void fam_ldm_x4(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void fam_ldm_x4_t(u32 addr, u32& r0, u32& r1, u32& r2, u32& r3) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3) : "r"(addr));
}
__device__ __forceinline__ void fam_mma(float* d, u32 a0, u32 a1, u32 a2, u32 a3, u32 b0, u32 b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}
// 8 consecutive K/V elements (i % 8 == 0) as 8 f16.
__device__ __forceinline__ uint4 fam_ld8(const u16* p, size_t i) { return *(const uint4*)(p + i); }
__device__ __forceinline__ uint4 fam_ld8(const float* p, size_t i) {
    float4 a = *(const float4*)(p + i), b = *(const float4*)(p + i + 4);
    return make_uint4(pack_h2(a.x, a.y), pack_h2(a.z, a.w), pack_h2(b.x, b.y), pack_h2(b.z, b.w));
}

template <typename KT>
__device__ void fa_mma_prefill_impl(const float* __restrict__ q, const KT* __restrict__ K, const KT* __restrict__ V,
                                    float* __restrict__ out, int n_rows, int kv_dim, int kv_mul, int q_stride,
                                    int pos0, float scale, float softcap, int* qflag) {
    __shared__ __align__(16) u16 ks[FA_BC * FAM_ROWB];
    __shared__ __align__(16) u16 vs[FA_BC * FAM_ROWB];
    const float NEG_INF = __int_as_float(0xff800000);
    const int h = blockIdx.x;
    const int qb = gridDim.y - 1 - blockIdx.y;  // latest (longest) query tiles first
    const int q0 = qb * FA_BR;
    const int rows_here = min(FA_BR, n_rows - q0);
    const size_t kvh = (size_t)(h / kv_mul) * FA_D;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int wr0 = warp * 16;  // warp's first row within the block
    const int kv_end = pos0 + n_rows;
    const int qp_min = pos0 + q0;
    const int n_tiles = (qp_min + rows_here - 1) / FA_BC + 1;
    // Q A-fragments (f16), 8 k-steps of 16 dims; rows >= rows_here are zero.
    u32 qa[8][4];
    {
        const int ra = wr0 + g, rb = wr0 + g + 8;
        const float* qra = q + (size_t)(q0 + ra) * q_stride + (size_t)h * FA_D;
        const float* qrb = q + (size_t)(q0 + rb) * q_stride + (size_t)h * FA_D;
        float amax = 0.f;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const int d0 = 16 * s + 2 * t;
            float2 x0 = make_float2(0.f, 0.f), x1 = x0, x2 = x0, x3 = x0;
            if (ra < rows_here) { x0 = *(const float2*)(qra + d0); x2 = *(const float2*)(qra + d0 + 8); }
            if (rb < rows_here) { x1 = *(const float2*)(qrb + d0); x3 = *(const float2*)(qrb + d0 + 8); }
            if (qflag) amax = fmaxf(amax, fmaxf(fmaxf(fmaxf(fabsf(x0.x), fabsf(x0.y)), fmaxf(fabsf(x1.x), fabsf(x1.y))),
                                                fmaxf(fmaxf(fabsf(x2.x), fabsf(x2.y)), fmaxf(fabsf(x3.x), fabsf(x3.y)))));
            qa[s][0] = pack_h2(x0.x, x0.y);
            qa[s][1] = pack_h2(x1.x, x1.y);
            qa[s][2] = pack_h2(x2.x, x2.y);
            qa[s][3] = pack_h2(x3.x, x3.y);
        }
        // 65504 = f16 max (cvt.rn rounds |x| >= 65520 to inf); inf trips it, NaN is not detected (fmaxf).
        if (qflag && !(amax < 65504.f)) atomicOr(qflag, 1);
    }
    float o[16][4];
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.f;
    float m[2] = {NEG_INF, NEG_INF}, l[2] = {0.f, 0.f};  // l: this thread's partial row sums
    const int qp_a = qp_min + wr0 + g, qp_b = qp_a + 8;    // positions of rows g and g+8
    const bool warp_rows = wr0 < rows_here;
    const int warp_qp_max = qp_min + wr0 + 15;
    // ldmatrix lane addresses (key, dim) offsets: K (non-trans) and V (trans), see the header comment.
    const int k_key = (lane & 7) + ((lane >> 4) << 3), k_dim = ((lane >> 3) & 1) << 3;
    const int v_key = (lane & 7) + (((lane >> 3) & 1) << 3), v_dim = (lane >> 4) << 3;
    for (int tt = 0; tt < n_tiles; ++tt) {
        const int k0 = tt * FA_BC;
        const bool need_mask = k0 + FA_BC - 1 > qp_min;  // block rule, same as the scalar kernel
        __syncthreads();  // previous K/V tile fully read
        for (int i = tid; i < FA_BC * (FA_D / 8); i += blockDim.x) {
            const int j = i >> 4, c = (i & 15) << 3, kp = k0 + j;
            uint4 kx = make_uint4(0u, 0u, 0u, 0u), vx = kx;
            if (kp < kv_end) {
                const size_t off = (size_t)kp * kv_dim + kvh + c;
                kx = fam_ld8(K, off);
                vx = fam_ld8(V, off);
            }
            *(uint4*)&ks[fam_idx(j, c)] = kx;
            *(uint4*)&vs[fam_idx(j, c)] = vx;
        }
        __syncthreads();
        // Warps whose rows are all tail rows or all before this tile's first key skip the math
        // (every score would be masked: m, l, O unchanged).
        if (!warp_rows || k0 > warp_qp_max) continue;
        // S = Q K^T: 16 rows x 64 keys (8 n-tiles), fp32 accumulators.
        float s[8][4];
#pragma unroll
        for (int j = 0; j < 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.f;
#pragma unroll
        for (int kd = 0; kd < 8; ++kd) {
#pragma unroll
            for (int np = 0; np < 4; ++np) {
                u32 b0, b1, b2, b3;
                fam_ldm_x4(fam_smem(&ks[fam_idx(16 * np + k_key, 16 * kd + k_dim)]), b0, b1, b2, b3);
                fam_mma(s[2 * np], qa[kd][0], qa[kd][1], qa[kd][2], qa[kd][3], b0, b1);
                fam_mma(s[2 * np + 1], qa[kd][0], qa[kd][1], qa[kd][2], qa[kd][3], b2, b3);
            }
        }
        // fp32 scale -> softcap -> mask -> online softmax (rows g: e = 0,1; g+8: e = 2,3).
        float mx[2] = {NEG_INF, NEG_INF};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                float v = s[j][e] * scale;
                if (softcap > 0.f) v = softcap * tanhf(v / softcap);
                if (need_mask) {
                    const int kp = k0 + 8 * j + 2 * t + (e & 1);
                    const int qp = (e < 2) ? qp_a : qp_b;
                    if (kp > qp || kp >= kv_end) v = NEG_INF;
                }
                s[j][e] = v;
                mx[e >> 1] = fmaxf(mx[e >> 1], v);
            }
        }
        float corr[2];
#pragma unroll
        for (int r = 0; r < 2; ++r) {
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 1));
            mx[r] = fmaxf(mx[r], __shfl_xor_sync(0xffffffffu, mx[r], 2));
            const float mn = fmaxf(m[r], mx[r]);
            corr[r] = (mn == NEG_INF) ? 1.f : expf(m[r] - mn);  // m = -inf -> 0
            m[r] = mn;
        }
        float ls[2] = {0.f, 0.f};
#pragma unroll
        for (int j = 0; j < 8; ++j) {
#pragma unroll
            for (int e = 0; e < 4; ++e) {
                const int r = e >> 1;
                const float p = (m[r] == NEG_INF) ? 0.f : expf(s[j][e] - m[r]);
                s[j][e] = p;
                ls[r] += p;
            }
        }
#pragma unroll
        for (int r = 0; r < 2; ++r) l[r] = l[r] * corr[r] + ls[r];
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            o[j][0] *= corr[0];
            o[j][1] *= corr[0];
            o[j][2] *= corr[1];
            o[j][3] *= corr[1];
        }
        // O += P V: P (fp32 C layout of S) -> f16 A fragments, 4 k-steps of 16 keys, 16 dim n-tiles.
#pragma unroll
        for (int kk = 0; kk < 4; ++kk) {
            const u32 a0 = pack_h2(s[2 * kk][0], s[2 * kk][1]);
            const u32 a1 = pack_h2(s[2 * kk][2], s[2 * kk][3]);
            const u32 a2 = pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            const u32 a3 = pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
#pragma unroll
            for (int dp = 0; dp < 8; ++dp) {
                u32 b0, b1, b2, b3;
                fam_ldm_x4_t(fam_smem(&vs[fam_idx(16 * kk + v_key, 16 * dp + v_dim)]), b0, b1, b2, b3);
                fam_mma(o[2 * dp], a0, a1, a2, a3, b0, b1);
                fam_mma(o[2 * dp + 1], a0, a1, a2, a3, b2, b3);
            }
        }
    }
    // Final row sums across the quad, normalise, write valid rows (row g: o[.][0..1], g+8: o[.][2..3]).
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 1);
        l[r] += __shfl_xor_sync(0xffffffffu, l[r], 2);
    }
#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int row = wr0 + g + 8 * r;
        if (row < rows_here) {
            const float inv = (l[r] > 0.f) ? 1.f / l[r] : 0.f;
            float* oh = out + (size_t)(q0 + row) * q_stride + (size_t)h * FA_D + 2 * t;
#pragma unroll
            for (int j = 0; j < 16; ++j)
                *(float2*)(oh + 8 * j) = make_float2(o[j][2 * r] * inv, o[j][2 * r + 1] * inv);
        }
    }
}
extern "C" __global__ void __launch_bounds__(FA_WARPS * 32) fa_mma_prefill_f16(const float* q, const u16* K,
        const u16* V, float* out, int n_rows, int kv_dim, int kv_mul, int q_stride, int pos0, float scale,
        float softcap, int* qflag) {
    fa_mma_prefill_impl<u16>(q, K, V, out, n_rows, kv_dim, kv_mul, q_stride, pos0, scale, softcap, qflag);
}
extern "C" __global__ void __launch_bounds__(FA_WARPS * 32) fa_mma_prefill_f32kv(const float* q, const float* K,
        const float* V, float* out, int n_rows, int kv_dim, int kv_mul, int q_stride, int pos0, float scale,
        float softcap, int* qflag) {
    fa_mma_prefill_impl<float>(q, K, V, out, n_rows, kv_dim, kv_mul, q_stride, pos0, scale, softcap, qflag);
}
// ---- Int8 MMQ prefill (ARTIFICER_PREFILL_MATMUL=mmq) ----
// Y[rows x n] (f32, row-major, ld = n) = X[rows x k] . W^T with X quantized by quant_q8 (xq int8 [rows][k],
// xds[rows][k/32] = (d, d*sum q)) and W the raw GGUF blocks [n][k]. Block tile: MMQ_BN = 128 weight rows x
// MMQ_BM = 64 activation rows; 8 warps = 4 (weight) x 2 (activation), each warp 32 x 32 = 2 x 4 m16n8 tiles.
// One k stage = MMQ_KB = 4 blocks of 32. Weights are unpacked to int8 in smem (same bit tricks as vd_*);
// per 32-k block the int32 tile sum is exact and folded into f32 as acc += (dw * dx) * isum (the vd_* expression).
//   kind 0 (Q5_0): one scale per 32 (offset -16 applied in the unpack).
//   kind 1 (Q4_K): scale d*sc and min dmin*mn per 32: acc -= dmin*mn * xds.y.
//   kind 2 (Q6_K): two scales per 32 (per 16): two m16n8k16 mma, acc += dx * (d*sc0*isum_lo + d*sc1*isum_hi).
#define MMQ_BN 128
#define MMQ_BM 64
#define MMQ_KB 4
#define MMQ_LDS (MMQ_KB * 8 + 4)  // int words per smem row; +4 pad -> conflict-free fragment loads

// Q5_0 / Q6_K: exact int32 -> f32 of the mma tile sum without I2F. The accumulator is seeded with
// MMQ_BIAS = 0x4B400000 (1.5 * 2^23 as f32 bits); the mma adds the block sum v exactly (int32, no overflow),
// and __int_as_float(MMQ_BIAS + v) - 1.5 * 2^23 == (float)v bit for bit for abs(v) < 2^22 (block sums are
// < 32 * 32 * 127 < 2^17). So the fold sees the same f32 values as with (float)v: bit-identical.
// Q4_K keeps (float)c and its inline min term (no bias trick, so Q4_K results are unchanged).
#define MMQ_BIAS 0x4B400000
__device__ __forceinline__ float mmq_unbias(int biased) { return __int_as_float(biased) - 12582912.0f; }

__device__ __forceinline__ void mma_s8_16832(int* c, const int* a, int b0, int b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_s8_16816(int* c, int a0, int a1, int b0) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0));
}

// 4-byte-aligned word at or just below a 2-byte-aligned address.
__device__ __forceinline__ const u32* mmq_al4(const u8* p) { return (const u32*)((size_t)p & ~(size_t)3); }
// Unpackers: 32-element block kb of one weight row -> 8 int words (word i = elements 4i..4i+3), s0/s1 scales.
struct MmqQ5_0 {
    static constexpr int KIND = 0;
    __device__ static size_t row_bytes(int k) { return (size_t)(k >> 5) * 22; }
    __device__ static void load(const u8* row, int kb, int* w, float* s0, float* s1) {
        // The 22-byte block is 2-byte aligned; read it as 6 aligned words (5 + the last half word when the block
        // starts on a 4-byte boundary, so nothing past the block is read) and realign with funnel shifts:
        // 6-7 LDG.32 instead of 11 LDG.U16. sh = 16 when the block starts at 2 mod 4 (the aligned word then starts
        // 2 bytes before the block, inside the same cudaMalloc'd, 256-aligned weight buffer).
        const size_t ad = (size_t)(row + (size_t)kb * 22);
        const u32* a = (const u32*)(ad & ~(size_t)3);
        const u32 sh = (u32)(ad & 2) << 3, fs = 16 + sh;
        u32 W[6];
        #pragma unroll
        for (int i = 0; i < 5; ++i) W[i] = __ldg(a + i);
        W[5] = sh ? __ldg(a + 5) : (u32)__ldg((const u16*)(a + 5));
        const u32 qh = __funnelshift_rc(W[0], W[1], fs);
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int q = (int)__funnelshift_rc(W[1 + i], W[2 + i], fs);
            int lo = (q & 0x0F0F0F0F) | spread4((qh >> (4 * i)) & 0xFu);
            int hi = ((q >> 4) & 0x0F0F0F0F) | spread4((qh >> (16 + 4 * i)) & 0xFu);
            w[i] = (int)__vsub4(lo, 0x10101010);
            w[4 + i] = (int)__vsub4(hi, 0x10101010);
        }
        *s0 = h2f((u16)(W[0] >> sh));
        *s1 = 0.f;
    }
};
struct MmqQ4_K {
    static constexpr int KIND = 1;
    __device__ static size_t row_bytes(int k) { return (size_t)(k >> 8) * 144; }
    __device__ static void load(const u8* row, int kb, int* w, float* s0, float* s1) {
        const u8* sb = row + (size_t)(kb >> 3) * 144;
        int j = kb & 7;
        const u16* p = (const u16*)sb;
        int sc, mn;
        scale_min_k4(j, sb + 4, &sc, &mn);
        const int4* q4 = (const int4*)(sb + 16 + 32 * (j >> 1));
        int4 a = __ldg(q4), b = __ldg(q4 + 1);
        int sh = (j & 1) * 4;
        w[0] = (a.x >> sh) & 0x0F0F0F0F; w[1] = (a.y >> sh) & 0x0F0F0F0F;
        w[2] = (a.z >> sh) & 0x0F0F0F0F; w[3] = (a.w >> sh) & 0x0F0F0F0F;
        w[4] = (b.x >> sh) & 0x0F0F0F0F; w[5] = (b.y >> sh) & 0x0F0F0F0F;
        w[6] = (b.z >> sh) & 0x0F0F0F0F; w[7] = (b.w >> sh) & 0x0F0F0F0F;
        *s0 = h2f(__ldg(p)) * (float)sc;
        *s1 = h2f(__ldg(p + 1)) * (float)mn;
    }
};
struct MmqQ6_K {
    static constexpr int KIND = 2;
    __device__ static size_t row_bytes(int k) { return (size_t)(k >> 8) * 210; }
    __device__ static void load(const u8* row, int kb, int* w, float* s0, float* s1) {
        const u8* sb = row + (size_t)(kb >> 3) * 210;
        int j = kb & 7, half = j >> 2, r = j & 3;
        const u32* ql = mmq_al4(sb + 64 * half + 32 * (r & 1));
        const u32* qh = mmq_al4(sb + 128 + 32 * half);
        // The superblock is 2-byte aligned (210 B): read 9 aligned words per 32-byte run and realign (fs = 0 or 16).
        // The 9th word stays inside the superblock (ql runs are followed by qh, qh runs by the scales).
        const u32 fs = (u32)((size_t)sb & 2) << 3;
        u32 L[9], H[9];
        #pragma unroll
        for (int i = 0; i < 9; ++i) { L[i] = __ldg(ql + i); H[i] = __ldg(qh + i); }
        const i8* sc = (const i8*)(sb + 192) + 8 * half + 2 * r;
        int nsh = (r >= 2) ? 4 : 0, hsh = 2 * r;
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            int l = ((int)__funnelshift_r(L[i], L[i + 1], fs) >> nsh) & 0x0F0F0F0F;
            int h = (((int)__funnelshift_r(H[i], H[i + 1], fs) >> hsh) & 0x03030303) << 4;
            w[i] = (int)__vsub4(l | h, 0x20202020);
        }
        float d = h2f(__ldg((const u16*)(sb + 208)));
        *s0 = d * (float)__ldg(sc);
        *s1 = d * (float)__ldg(sc + 1);
    }
};

template <class WT>
__device__ __forceinline__ void mmq_impl(const u8* __restrict__ W, const signed char* __restrict__ xq,
                                         const float2* __restrict__ xds, float* __restrict__ y, int k, int n,
                                         int rows) {
    __shared__ __align__(16) int Ws[MMQ_BN * MMQ_LDS];
    __shared__ __align__(16) int Xs[MMQ_BM * MMQ_LDS];
    __shared__ float S0[MMQ_BN * MMQ_KB], S1[MMQ_BN * MMQ_KB];
    __shared__ float2 Xd[MMQ_BM * MMQ_KB];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, t = lane & 3;
    const int r0 = blockIdx.x * MMQ_BM, j0 = blockIdx.y * MMQ_BN;
    const int wj = (warp & 3) * 32, wr = (warp >> 2) * 32;
    const int nbk = k >> 5;
    const size_t wrow = WT::row_bytes(k);
    float acc[2][4][4];
    #pragma unroll
    for (int a = 0; a < 2; ++a)
        #pragma unroll
        for (int b = 0; b < 4; ++b)
            #pragma unroll
            for (int c = 0; c < 4; ++c) acc[a][b][c] = 0.f;

    for (int kb0 = 0; kb0 < nbk; kb0 += MMQ_KB) {
        const int nkb = min(MMQ_KB, nbk - kb0);
        // Weights: 128 rows x 4 blocks = 512 unpacks, 2 per thread.
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int idx = tid + s * 256, jj = idx >> 2, b = idx & 3;
            int w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
            float s0 = 0.f, s1 = 0.f;
            if (j0 + jj < n && b < nkb) WT::load(W + (size_t)(j0 + jj) * wrow, kb0 + b, w, &s0, &s1);
            int4* dst = (int4*)(Ws + jj * MMQ_LDS + b * 8);
            dst[0] = make_int4(w[0], w[1], w[2], w[3]);
            dst[1] = make_int4(w[4], w[5], w[6], w[7]);
            S0[jj * MMQ_KB + b] = s0;
            S1[jj * MMQ_KB + b] = s1;
        }
        // Activations: 64 rows x 128 B = 512 int4, 2 per thread.
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            int idx = tid + s * 256, rr = idx >> 3, q = idx & 7;
            int4 v = make_int4(0, 0, 0, 0);
            if (r0 + rr < rows && (q >> 1) < nkb)
                v = __ldg((const int4*)(xq + (size_t)(r0 + rr) * k + (size_t)kb0 * 32) + q);
            *(int4*)(Xs + rr * MMQ_LDS + q * 4) = v;
        }
        {
            int rr = tid >> 2, b = tid & 3;
            float2 v = make_float2(0.f, 0.f);
            if (r0 + rr < rows && b < nkb) v = __ldg(xds + (size_t)(r0 + rr) * nbk + kb0 + b);
            Xd[rr * MMQ_KB + b] = v;
        }
        __syncthreads();
        for (int b = 0; b < nkb; ++b) {
            int a[2][4];
            float s0[2][2], s1[2][2];
            #pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                const int rw = wj + mt * 16 + g;
                const int* p = Ws + rw * MMQ_LDS + b * 8 + t;
                a[mt][0] = p[0];
                a[mt][1] = p[8 * MMQ_LDS];
                a[mt][2] = p[4];
                a[mt][3] = p[8 * MMQ_LDS + 4];
                #pragma unroll
                for (int h = 0; h < 2; ++h) {
                    s0[mt][h] = S0[(rw + 8 * h) * MMQ_KB + b];
                    s1[mt][h] = S1[(rw + 8 * h) * MMQ_KB + b];
                }
            }
            #pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int cb = wr + nt * 8;
                const int* q = Xs + (cb + g) * MMQ_LDS + b * 8 + t;
                const int b0 = q[0], b1 = q[4];
                const float2 dx0 = Xd[(cb + 2 * t) * MMQ_KB + b], dx1 = Xd[(cb + 2 * t + 1) * MMQ_KB + b];
                #pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    float* o = acc[mt][nt];
                    if (WT::KIND == 2) {
                        int lo[4] = {MMQ_BIAS, MMQ_BIAS, MMQ_BIAS, MMQ_BIAS};
                        int hi[4] = {MMQ_BIAS, MMQ_BIAS, MMQ_BIAS, MMQ_BIAS};
                        mma_s8_16816(lo, a[mt][0], a[mt][1], b0);
                        mma_s8_16816(hi, a[mt][2], a[mt][3], b1);
                        o[0] += dx0.x * (s0[mt][0] * mmq_unbias(lo[0]) + s1[mt][0] * mmq_unbias(hi[0]));
                        o[1] += dx1.x * (s0[mt][0] * mmq_unbias(lo[1]) + s1[mt][0] * mmq_unbias(hi[1]));
                        o[2] += dx0.x * (s0[mt][1] * mmq_unbias(lo[2]) + s1[mt][1] * mmq_unbias(hi[2]));
                        o[3] += dx1.x * (s0[mt][1] * mmq_unbias(lo[3]) + s1[mt][1] * mmq_unbias(hi[3]));
                    } else {
                        if (WT::KIND == 0) {
                            int c[4] = {MMQ_BIAS, MMQ_BIAS, MMQ_BIAS, MMQ_BIAS};
                            mma_s8_16832(c, a[mt], b0, b1);
                            o[0] += s0[mt][0] * dx0.x * mmq_unbias(c[0]);
                            o[1] += s0[mt][0] * dx1.x * mmq_unbias(c[1]);
                            o[2] += s0[mt][1] * dx0.x * mmq_unbias(c[2]);
                            o[3] += s0[mt][1] * dx1.x * mmq_unbias(c[3]);
                        } else {  // Q4_K: original MMQ tile layout
                        int c[4] = {0, 0, 0, 0};
                        mma_s8_16832(c, a[mt], b0, b1);
                        o[0] += s0[mt][0] * dx0.x * (float)c[0];
                        o[1] += s0[mt][0] * dx1.x * (float)c[1];
                        o[2] += s0[mt][1] * dx0.x * (float)c[2];
                        o[3] += s0[mt][1] * dx1.x * (float)c[3];
                        if (WT::KIND == 1) {
                            o[0] -= s1[mt][0] * dx0.y;
                            o[1] -= s1[mt][0] * dx1.y;
                            o[2] -= s1[mt][1] * dx0.y;
                            o[3] -= s1[mt][1] * dx1.y;
                        }
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int mt = 0; mt < 2; ++mt)
        #pragma unroll
        for (int nt = 0; nt < 4; ++nt)
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const int jr = j0 + wj + mt * 16 + g + (i >= 2 ? 8 : 0);
                const int r = r0 + wr + nt * 8 + 2 * t + (i & 1);
                if (jr < n && r < rows) y[(size_t)r * n + jr] = acc[mt][nt][i];
            }
}
// Launch bounds (min blocks/SM for ptxas). Q5_0 (31744 B smem + 1 KB reserved -> 3 x 32 KB fits the 100 KB SM) needs
// <= 80 registers for 3 blocks/SM; the MMQ_BIAS seeding alone would take it to 92 (2 blocks/SM), so cap it.
// Q4_K/Q6_K (33792 B smem) cannot fit 3 blocks/SM whatever their register count: no cap, the original bounds
// (a plain (256): an explicit min-blocks of 1 already changes ptxas's Q4_K schedule).
#define MMQ_LB_DEFAULT __launch_bounds__(256)
#define MMQ_LB_3PERSM __launch_bounds__(256, 3)
#define MMQ_KERNEL(NAME, WT, LB)                                                                          \
extern "C" __global__ void LB NAME(const u8* __restrict__ W,                                             \
        const signed char* __restrict__ xq, const float2* __restrict__ xds, float* __restrict__ y, int k, \
        int n, int rows) {                                                                                \
    mmq_impl<WT>(W, xq, xds, y, k, n, rows);                                                              \
}
MMQ_KERNEL(mmq_q5_0, MmqQ5_0, MMQ_LB_3PERSM)
MMQ_KERNEL(mmq_q4_k, MmqQ4_K, MMQ_LB_DEFAULT)
MMQ_KERNEL(mmq_q6_k, MmqQ6_K, MMQ_LB_DEFAULT)
// ===== END DEVICE-FORWARD KERNELS =====

// ===== HOST-OP KERNELS (below) =====
// SSM / norm / element-wise kernels for the device "host ops", names prefixed `ho_`.
//
// Numerics: element-wise ops use explicit round-to-nearest intrinsics (__fmul_rn/__fadd_rn/…) so
// NVRTC's default FMA contraction cannot change the host (Rust, non-contracted) result, and
// transcendental functions are evaluated in double and rounded once to f32 (≈ correctly rounded,
// like glibc's expf/tanhf/cosf). Result: add / bias / scale / relu² / embed are bit-identical to the
// host; silu/gelu/softcap/rope differ by ≤ 1 ulp at most; norms use a parallel reduction (different
// summation order than the host's sequential loop → ~1e-6 relative).

__device__ float ho_block_sum(float v) {
    __shared__ float ho_sh[32];
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5, nw = (blockDim.x + 31) >> 5;
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if (lane == 0) ho_sh[w] = v;
    __syncthreads();
    if (w == 0) {
        v = (lane < nw) ? ho_sh[lane] : 0.f;
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
        if (lane == 0) ho_sh[0] = v;
    }
    __syncthreads();
    v = ho_sh[0];
    __syncthreads();
    return v;
}
__device__ __forceinline__ float ho_expf(float x) { return (float)exp((double)x); }
// ops::silu: x / (1 + exp(-x))
__device__ __forceinline__ float ho_silu(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, ho_expf(-x))); }
// ops::gelu (tanh form): 0.5*x*(1 + tanh(C*(x + 0.044715*x^3)))
__device__ __forceinline__ float ho_gelu(float x) {
    const float C = 0.7978845608f;
    float x3 = __fmul_rn(__fmul_rn(x, x), x);
    float t = (float)tanh((double)__fmul_rn(C, __fadd_rn(x, __fmul_rn(0.044715f, x3))));
    return __fmul_rn(__fmul_rn(0.5f, x), __fadd_rn(1.0f, t));
}
// ssm::softplus: x > 20 ? x : ln(1 + exp(x))
__device__ __forceinline__ float ho_softplus(float x) {
    return x > 20.f ? x : (float)log((double)__fadd_rn(1.0f, ho_expf(x)));
}
// ARTIFICER_PREFILL_SSM=fp32: SSM-prefill-only SiLU with single-precision expf
// (full-range IEEE expf, max 2 ulp; NOT __expf). ho_expf / ho_silu above are unchanged and stay
// the default everywhere; only the *_f32 SSM prefill kernels below call this.
__device__ __forceinline__ float ho_ssm_silu_f32(float x) { return __fdiv_rn(x, __fadd_rn(1.0f, expf(-x))); }
// Compile-time SiLU choice for the templated SSM prefill kernels (F32 = false -> ho_silu, i.e. the
// original fp64-expf SiLU arithmetic).
template <bool F32> __device__ __forceinline__ float ho_ssm_silu_sel(float x) {
    if constexpr (F32) return ho_ssm_silu_f32(x); else return ho_silu(x);
}

// ---- norms (one block per row) ----
// RMS: out = x * (1/sqrt(mean(x²)+eps)) * w      gemma != 0: * (1 + w)
extern "C" __global__ void ho_rms_norm(const float* __restrict__ x, const float* __restrict__ w,
                                       float* __restrict__ out, int dim, float eps, int gemma) {
    const float* xr = x + (size_t)blockIdx.x * dim;
    float* o = out + (size_t)blockIdx.x * dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) ss += xr[i] * xr[i];
    ss = ho_block_sum(ss);
    float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)dim), eps)));
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float wi = gemma ? __fadd_rn(1.0f, w[i]) : w[i];
        o[i] = __fmul_rn(__fmul_rn(xr[i], inv), wi);
    }
}
// LayerNorm: (x - mean) * (1/sqrt(var+eps)) * w (+ b)
extern "C" __global__ void ho_layer_norm(const float* __restrict__ x, const float* __restrict__ w,
                                         const float* __restrict__ b, float* __restrict__ out, int dim,
                                         float eps, int has_b) {
    const float* xr = x + (size_t)blockIdx.x * dim;
    float* o = out + (size_t)blockIdx.x * dim;
    float s = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) s += xr[i];
    float mean = __fdiv_rn(ho_block_sum(s), (float)dim);
    float v = 0.f;
    for (int i = threadIdx.x; i < dim; i += blockDim.x) { float d = __fsub_rn(xr[i], mean); v += d * d; }
    v = ho_block_sum(v);
    float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(v, (float)dim), eps)));
    for (int i = threadIdx.x; i < dim; i += blockDim.x) {
        float y = __fmul_rn(__fmul_rn(__fsub_rn(xr[i], mean), inv), w[i]);
        if (has_b) y = __fadd_rn(y, b[i]);
        o[i] = y;
    }
}

// ---- element-wise (grid-stride over n) ----
extern "C" __global__ void ho_add(float* __restrict__ a, const float* __restrict__ b, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        a[i] = __fadd_rn(a[i], b[i]);
}
// out[r*dim + c] += bias[c] for c < nb (host add_bias adds min(bias.len, out.len) elements)
extern "C" __global__ void ho_add_bias(float* __restrict__ out, const float* __restrict__ bias, int rows,
                                       int dim, int nb) {
    int n = rows * dim;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        int c = i % dim;
        if (c < nb) out[i] = __fadd_rn(out[i], bias[c]);
    }
}
extern "C" __global__ void ho_scale(float* __restrict__ x, float s, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = __fmul_rn(x[i], s);
}
extern "C" __global__ void ho_softcap(float* __restrict__ x, float cap, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        x[i] = __fmul_rn(cap, (float)tanh((double)__fdiv_rn(x[i], cap)));
}
// Gated FFN: hb = act(hb) * hb2   (kind 0 = SiLU, 1 = GELU)
extern "C" __global__ void ho_glu(float* __restrict__ hb, const float* __restrict__ hb2, int n, int kind) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float h = kind == 0 ? ho_silu(hb[i]) : ho_gelu(hb[i]);
        hb[i] = __fmul_rn(h, hb2[i]);
    }
}
extern "C" __global__ void ho_relu_sqr(float* __restrict__ x, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        float r = fmaxf(x[i], 0.f);
        x[i] = __fmul_rn(r, r);
    }
}
// ops::apply_rope (interleaved pairs) on `rows` rows (stride row_stride) at positions pos0 + r.
// One thread per (row, head, pair).
extern "C" __global__ void ho_rope(float* __restrict__ x, int rows, int row_stride, int n_head, int head_dim,
                                   int pos0, float freq_base, int n_rot) {
    int pairs = (n_rot + 1) / 2;
    int total = rows * n_head * pairs;
    for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < total; t += gridDim.x * blockDim.x) {
        int p = t % pairs, h = (t / pairs) % n_head, r = t / (pairs * n_head);
        int i = 2 * p;
        if (i + 1 >= head_dim) continue;
        float e = __fdiv_rn((float)i, (float)n_rot);
        float freq = __fdiv_rn(1.0f, (float)pow((double)freq_base, (double)e));
        float val = __fmul_rn((float)(pos0 + r), freq);
        float c = (float)cos((double)val), s = (float)sin((double)val);
        float* hp = x + (size_t)r * row_stride + (size_t)h * head_dim;
        float x0 = hp[i], x1 = hp[i + 1];
        hp[i] = __fsub_rn(__fmul_rn(x0, c), __fmul_rn(x1, s));
        hp[i + 1] = __fadd_rn(__fmul_rn(x0, s), __fmul_rn(x1, c));
    }
}

// ---- embedding gather + dequant (matches quant/dequant.rs) ----
// ty: 0 F32, 1 F16, 2 BF16, 3 Q8_0, 4 Q5_0, 5 Q4_K, 6 Q6_K, 7 Q4_0
__device__ __forceinline__ float ho_deq(const u8* t, size_t e, int ty) {
    switch (ty) {
    case 0: return __uint_as_float(ld32(t + 4 * e));
    case 1: return h2f(ld16(t + 2 * e));
    case 2: return __uint_as_float((u32)ld16(t + 2 * e) << 16);
    case 3: { const u8* b = t + (e / 32) * 34; int j = (int)(e % 32);
              return __fmul_rn((float)(i8)b[2 + j], h2f(ld16(b))); }
    case 4: { const u8* b = t + (e / 32) * 22; int j = (int)(e % 32);
              float d = h2f(ld16(b)); u32 qh = ld32(b + 2); const u8* qs = b + 6; int x;
              if (j < 16) x = (int)((qs[j] & 0x0F) | (((qh >> j) << 4) & 0x10)) - 16;
              else { int jj = j - 16; x = (int)((qs[jj] >> 4) | ((qh >> (jj + 12)) & 0x10)) - 16; }
              return __fmul_rn((float)x, d); }
    case 5: { const u8* b = t + (e / 256) * 144; int j = (int)(e % 256);
              float d = h2f(ld16(b)), mn = h2f(ld16(b + 2)); const u8* sc = b + 4; const u8* q = b + 16;
              int chunk = j / 64, w = j % 64, l = w % 32, is = chunk * 2 + (w >= 32 ? 1 : 0);
              int s, m;
              if (is < 4) { s = sc[is] & 63; m = sc[is + 4] & 63; }
              else { s = (sc[is + 4] & 0xF) | ((sc[is - 4] >> 6) << 4); m = (sc[is + 4] >> 4) | ((sc[is] >> 6) << 4); }
              u8 qb = q[chunk * 32 + l]; int nib = w < 32 ? (qb & 0xF) : (qb >> 4);
              return __fsub_rn(__fmul_rn(__fmul_rn(d, (float)s), (float)nib), __fmul_rn(mn, (float)m)); }
    case 6: { const u8* b = t + (e / 256) * 210; int j = (int)(e % 256);
              int half = j / 128, jj = j % 128, quarter = jj / 32, l = jj % 32, is = l / 16;
              const u8* ql = b + half * 64; const u8* qh = b + 128 + half * 32;
              const i8* sc = (const i8*)(b + 192) + half * 8; float d = h2f(ld16(b + 208)); int q, s;
              switch (quarter) {
              case 0: q = (ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4); s = sc[is]; break;
              case 1: q = (ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4); s = sc[is + 2]; break;
              case 2: q = (ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4); s = sc[is + 4]; break;
              default: q = (ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4); s = sc[is + 6]; break;
              }
              return __fmul_rn(__fmul_rn(d, (float)s), (float)(q - 32)); }
    default: { const u8* b = t + (e / 32) * 18; int j = (int)(e % 32); float d = h2f(ld16(b));
               int x = j < 16 ? (int)(b[2 + j] & 15) - 8 : (int)(b[2 + j - 16] >> 4) - 8;
               return __fmul_rn((float)x, d); }
    }
}
// out[i] = dequant(table row `token`)[i] (* scale when do_scale)
extern "C" __global__ void ho_embed(const u8* __restrict__ table, float* __restrict__ out, int token, int n_embd,
                                    int ty, float scale, int do_scale) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_embd; i += gridDim.x * blockDim.x) {
        float v = ho_deq(table, (size_t)token * n_embd + i, ty);
        out[i] = do_scale ? __fmul_rn(v, scale) : v;
    }
}

// ---- Mamba-2 mixer, one token (matches ssm::mamba2_mix_host) ----
// Causal depthwise conv + SiLU, then shift the conv state and append the input.
// conv_w layout [d_conv][conv_dim] (host SsmCache statics), state [(d_conv-1)][conv_dim].
extern "C" __global__ void ho_ssm_conv(const float* __restrict__ xin, float* __restrict__ st,
                                       const float* __restrict__ w, const float* __restrict__ bias,
                                       float* __restrict__ out, int conv_dim, int d_conv) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    float acc = bias[c];
    for (int k = 0; k < d_conv - 1; ++k) acc = fmaf(st[(size_t)k * conv_dim + c], w[(size_t)k * conv_dim + c], acc);
    float xc = xin[c];
    acc = fmaf(xc, w[(size_t)(d_conv - 1) * conv_dim + c], acc);
    out[c] = ho_silu(acc);
    for (int k = 0; k + 2 < d_conv; ++k) st[(size_t)k * conv_dim + c] = st[(size_t)(k + 1) * conv_dim + c];
    if (d_conv > 1) st[(size_t)(d_conv - 2) * conv_dim + c] = xc;
}
// Selective scan, one warp per (head, channel) row; lanes stride d_state.
// xbc = conv output [x (d_inner) | B (n_group*d_state) | C (n_group*d_state)]; dt raw (pre-bias).
extern "C" __global__ void ho_ssm_scan(const float* __restrict__ xbc, const float* __restrict__ dt_raw,
                                       const float* __restrict__ A, const float* __restrict__ D,
                                       const float* __restrict__ dt_b, float* __restrict__ state,
                                       float* __restrict__ y, int n_head, int head_dim, int d_state, int n_group) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_head * head_dim) return;
    int h = row / head_dim;
    int d_inner = n_head * head_dim;
    float dt = ho_softplus(__fadd_rn(dt_raw[h], dt_b[h]));
    float dA = ho_expf(__fmul_rn(dt, A[h]));
    int g = h / (n_head / n_group);
    const float* B = xbc + d_inner + (size_t)g * d_state;
    const float* C = xbc + d_inner + (size_t)n_group * d_state + (size_t)g * d_state;
    float xv = xbc[row];
    float x_dt = __fmul_rn(xv, dt);
    float* s = state + (size_t)row * d_state;
    float sum = 0.f;
    for (int i = lane; i < d_state; i += 32) {
        float v = fmaf(s[i], dA, B[i] * x_dt);
        s[i] = v;
        sum = fmaf(v, C[i], sum);
    }
    for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
    if (lane == 0) y[row] = fmaf(xv, D[h], sum);
}
// y = y * silu(z), then grouped RMSNorm (one block per group of group_dim channels).
extern "C" __global__ void ho_ssm_gate_norm(float* __restrict__ y, const float* __restrict__ z,
                                            const float* __restrict__ w, int group_dim, float eps) {
    float* yg = y + (size_t)blockIdx.x * group_dim;
    const float* zg = z + (size_t)blockIdx.x * group_dim;
    const float* wg = w + (size_t)blockIdx.x * group_dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < group_dim; i += blockDim.x) {
        float v = __fmul_rn(yg[i], ho_silu(zg[i]));
        yg[i] = v;
        ss += v * v;
    }
    ss = ho_block_sum(ss);
    float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)group_dim), eps)));
    for (int i = threadIdx.x; i < group_dim; i += blockDim.x) yg[i] = __fmul_rn(__fmul_rn(yg[i], inv), wg[i]);
}

// ---- Mamba-2 mixer, n_tok rows in 3 launches (prefill / multi-token) ----
// Same arithmetic (same FMA chains, same reduction order, same block size) as the single-row
// ho_ssm_conv / ho_ssm_scan / ho_ssm_gate_norm above, so a multi-row call is bit-identical to
// looping the single-row kernels over the rows.
#define HO_CONV_MAX 8   // d_conv <= 8
#define HO_SCAN_NS 8    // d_state <= 32 * 8

// Launch 1: causal depthwise conv over n_tok rows, one thread per channel walking the rows with
// the (d_conv-1)-tap window in registers; writes out[t][c] = silu(conv) and the final window
// back to the conv state. Threads past conv_dim compute dt = softplus(dt_raw + dt_b) and
// dA = exp(dt * A) once per (t, head) into dtA[t][0..n_head | n_head..2*n_head].
extern "C" __global__ void ho_ssm_conv_rows(const float* __restrict__ zx, int zx_stride, int xbc_off,
                                            int dt_off, float* __restrict__ st, const float* __restrict__ w,
                                            const float* __restrict__ bias, float* __restrict__ out,
                                            const float* __restrict__ A, const float* __restrict__ dt_b,
                                            float* __restrict__ dtA, int conv_dim, int d_conv, int n_head,
                                            int n_tok) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= conv_dim) {
        int k = idx - conv_dim;
        if (k >= n_tok * n_head) return;
        int t = k / n_head, h = k % n_head;
        float dt = ho_softplus(__fadd_rn(zx[(size_t)t * zx_stride + dt_off + h], dt_b[h]));
        dtA[(size_t)t * 2 * n_head + h] = dt;
        dtA[(size_t)t * 2 * n_head + n_head + h] = ho_expf(__fmul_rn(dt, A[h]));
        return;
    }
    int c = idx, dm1 = d_conv - 1;
    float win[HO_CONV_MAX - 1], wk[HO_CONV_MAX - 1];
    #pragma unroll
    for (int k = 0; k < HO_CONV_MAX - 1; ++k) {
        win[k] = k < dm1 ? st[(size_t)k * conv_dim + c] : 0.f;
        wk[k] = k < dm1 ? w[(size_t)k * conv_dim + c] : 0.f;
    }
    float wl = w[(size_t)dm1 * conv_dim + c];
    float b = bias[c];
    for (int t = 0; t < n_tok; ++t) {
        float xc = zx[(size_t)t * zx_stride + xbc_off + c];
        float acc = b;
        #pragma unroll
        for (int k = 0; k < HO_CONV_MAX - 1; ++k)
            if (k < dm1) acc = fmaf(win[k], wk[k], acc);
        acc = fmaf(xc, wl, acc);
        out[(size_t)t * conv_dim + c] = ho_silu(acc);
        #pragma unroll
        for (int k = 0; k < HO_CONV_MAX - 2; ++k)
            if (k + 1 < dm1) win[k] = win[k + 1];
        #pragma unroll
        for (int k = 0; k < HO_CONV_MAX - 1; ++k)
            if (k == dm1 - 1) win[k] = xc;
    }
    #pragma unroll
    for (int k = 0; k < HO_CONV_MAX - 1; ++k)
        if (k < dm1) st[(size_t)k * conv_dim + c] = win[k];
}

// Launch 2: selective scan, sequential over the n_tok rows inside the kernel. One warp per
// (head, channel) row; each lane carries d_state/32 state values in registers across all rows
// (loaded once, stored once). xbc = conv output [n_tok][conv_dim]; dtA from launch 1.
extern "C" __global__ void ho_ssm_scan_rows(const float* __restrict__ xbc, const float* __restrict__ dtA,
                                            const float* __restrict__ D, float* __restrict__ state,
                                            float* __restrict__ y, int n_head, int head_dim, int d_state,
                                            int n_group, int n_tok) {
    int lane = threadIdx.x & 31;
    int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (row >= n_head * head_dim) return;
    int h = row / head_dim;
    int d_inner = n_head * head_dim;
    int conv_dim = d_inner + 2 * n_group * d_state;
    int g = h / (n_head / n_group);
    float Dh = D[h];
    float* sp = state + (size_t)row * d_state;
    float s[HO_SCAN_NS];
    #pragma unroll
    for (int j = 0; j < HO_SCAN_NS; ++j) {
        int i = lane + 32 * j;
        s[j] = i < d_state ? sp[i] : 0.f;
    }
    for (int t = 0; t < n_tok; ++t) {
        const float* xr = xbc + (size_t)t * conv_dim;
        float dt = dtA[(size_t)t * 2 * n_head + h];
        float dA = dtA[(size_t)t * 2 * n_head + n_head + h];
        const float* B = xr + d_inner + (size_t)g * d_state;
        const float* C = xr + d_inner + (size_t)n_group * d_state + (size_t)g * d_state;
        float xv = xr[row];
        float x_dt = __fmul_rn(xv, dt);
        float sum = 0.f;
        #pragma unroll
        for (int j = 0; j < HO_SCAN_NS; ++j) {
            int i = lane + 32 * j;
            if (i < d_state) {
                float v = fmaf(s[j], dA, B[i] * x_dt);
                s[j] = v;
                sum = fmaf(v, C[i], sum);
            }
        }
        for (int o = 16; o > 0; o >>= 1) sum += __shfl_xor_sync(0xffffffffu, sum, o);
        if (lane == 0) y[(size_t)t * d_inner + row] = fmaf(xv, Dh, sum);
    }
    #pragma unroll
    for (int j = 0; j < HO_SCAN_NS; ++j) {
        int i = lane + 32 * j;
        if (i < d_state) sp[i] = s[j];
    }
}

// Launch 3: y[t] *= silu(z[t]) + grouped RMSNorm; grid (n_group, n_tok).
// Templated on the SiLU: <false> = ho_ssm_gate_norm_rows (decode + old prefill, unchanged
// arithmetic), <true> = ho_ssm_gate_norm_rows_f32 (ARTIFICER_PREFILL_SSM=fp32, tiled prefill only).
template <bool F32>
__device__ __forceinline__ void ho_ssm_gate_norm_rows_impl(float* __restrict__ y, const float* __restrict__ zx,
                                                           int zx_stride, const float* __restrict__ w,
                                                           int group_dim, int d_inner, float eps) {
    size_t t = blockIdx.y;
    float* yg = y + t * d_inner + (size_t)blockIdx.x * group_dim;
    const float* zg = zx + t * zx_stride + (size_t)blockIdx.x * group_dim;
    const float* wg = w + (size_t)blockIdx.x * group_dim;
    float ss = 0.f;
    for (int i = threadIdx.x; i < group_dim; i += blockDim.x) {
        float v = __fmul_rn(yg[i], ho_ssm_silu_sel<F32>(zg[i]));
        yg[i] = v;
        ss += v * v;
    }
    ss = ho_block_sum(ss);
    float inv = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(__fdiv_rn(ss, (float)group_dim), eps)));
    for (int i = threadIdx.x; i < group_dim; i += blockDim.x) yg[i] = __fmul_rn(__fmul_rn(yg[i], inv), wg[i]);
}
extern "C" __global__ void ho_ssm_gate_norm_rows(float* __restrict__ y, const float* __restrict__ zx,
                                                 int zx_stride, const float* __restrict__ w, int group_dim,
                                                 int d_inner, float eps) {
    ho_ssm_gate_norm_rows_impl<false>(y, zx, zx_stride, w, group_dim, d_inner, eps);
}
extern "C" __global__ void ho_ssm_gate_norm_rows_f32(float* __restrict__ y, const float* __restrict__ zx,
                                                     int zx_stride, const float* __restrict__ w, int group_dim,
                                                     int d_inner, float eps) {
    ho_ssm_gate_norm_rows_impl<true>(y, zx, zx_stride, w, group_dim, d_inner, eps);
}

// ---- Mamba-2 mixer, prefill-tiled conv + scan ----
// Drop-in replacements for ho_ssm_conv_rows / ho_ssm_scan_rows (those two stay as the naive
// reference kernels, selectable with ARTIFICER_SSM_NAIVE=1). Every output is produced by the
// SAME per-element FMA chain and the SAME reduction tree as the naive kernels, so the results
// are bit-identical to them (and therefore to the per-row ho_ssm_conv / ho_ssm_scan loop):
//  - conv: out[t][c] = silu(fma chain over the d_conv window) is independent per (t, c); the
//    window is re-read from zx (or the conv state for t < d_conv-1) instead of carried.
//  - scan: the state update fmaf(s, dA, B*x_dt) is per element; y's per-lane partial sum keeps
//    the lane + 32j order and the cross-lane sum keeps the xor-16/8/4/2/1 butterfly tree, done
//    as a reduce-scatter over a tile of tokens (each node is still "a + b" of the same two
//    subtrees; fp add is commutative, so every node is bit-identical).
#define HO_CONV_TT 8    // positions per conv thread; must be >= HO_CONV_MAX - 1 (state race-free)
#define HO_SCAN_T 16    // tokens per scan smem tile (reduce-scatter width; must be 16)
#define HO_SCAN_W 8     // warps (= channels of one head) per scan block

// Launch 1 (tiled): grid (conv_blocks + dt_blocks, ceil(n_tok / HO_CONV_TT)), block 256.
// blockIdx.x < conv_blocks: one thread per (channel c, HO_CONV_TT consecutive positions); loads
// and stores are coalesced across channels. blockIdx.x >= conv_blocks: dt / dA for the tile's
// (t, head) pairs (dt_blocks = ceil(HO_CONV_TT * n_head / 256)). Only the tile-0 thread of a
// channel reads the old conv state (HO_CONV_TT >= d_conv-1), and it writes the new state after
// all of its reads, so there is no cross-thread race on the state.
// Templated on the SiLU: <false> = ho_ssm_conv_tiled (old, unchanged arithmetic), <true> =
// ho_ssm_conv_tiled_f32 (ARTIFICER_PREFILL_SSM=fp32). dt = softplus / dA = exp stay on the
// fp64 helpers in both (cheap: n_tok * n_head per launch; dA feeds the long scan recurrence).
template <bool F32>
__device__ __forceinline__ void ho_ssm_conv_tiled_impl(const float* __restrict__ zx, int zx_stride, int xbc_off,
                                                       int dt_off, float* __restrict__ st,
                                                       const float* __restrict__ w, const float* __restrict__ bias,
                                                       float* __restrict__ out, const float* __restrict__ A,
                                                       const float* __restrict__ dt_b, float* __restrict__ dtA,
                                                       int conv_dim, int d_conv, int n_head, int n_tok,
                                                       int conv_blocks) {
    int t0 = blockIdx.y * HO_CONV_TT;
    int nt = min(HO_CONV_TT, n_tok - t0);
    if ((int)blockIdx.x >= conv_blocks) {
        int k = ((int)blockIdx.x - conv_blocks) * blockDim.x + threadIdx.x;
        if (k >= nt * n_head) return;
        int t = t0 + k / n_head, h = k % n_head;
        float dt = ho_softplus(__fadd_rn(zx[(size_t)t * zx_stride + dt_off + h], dt_b[h]));
        dtA[(size_t)t * 2 * n_head + h] = dt;
        dtA[(size_t)t * 2 * n_head + n_head + h] = ho_expf(__fmul_rn(dt, A[h]));
        return;
    }
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= conv_dim) return;
    int dm1 = d_conv - 1;
    const float* xin = zx + xbc_off + c;
    float win[HO_CONV_MAX - 1], wk[HO_CONV_MAX - 1], nst[HO_CONV_MAX - 1];
    // win[k] = input at position t0 - dm1 + k (position p < 0 -> conv state row dm1 + p).
    // nst[k] = new conv state = input at position n_tok - dm1 + k (tile 0 only; read before writes).
    #pragma unroll
    for (int k = 0; k < HO_CONV_MAX - 1; ++k) {
        wk[k] = k < dm1 ? w[(size_t)k * conv_dim + c] : 0.f;
        int p = t0 - dm1 + k;
        win[k] = k < dm1 ? (p < 0 ? st[(size_t)(dm1 + p) * conv_dim + c] : xin[(size_t)p * zx_stride]) : 0.f;
        int q = n_tok - dm1 + k;
        nst[k] = (blockIdx.y == 0 && k < dm1)
                     ? (q < 0 ? st[(size_t)(dm1 + q) * conv_dim + c] : xin[(size_t)q * zx_stride]) : 0.f;
    }
    float wl = w[(size_t)dm1 * conv_dim + c];
    float b = bias[c];
    #pragma unroll
    for (int tt = 0; tt < HO_CONV_TT; ++tt) {
        if (tt < nt) {
            float xc = xin[(size_t)(t0 + tt) * zx_stride];
            float acc = b;
            #pragma unroll
            for (int k = 0; k < HO_CONV_MAX - 1; ++k)
                if (k < dm1) acc = fmaf(win[k], wk[k], acc);
            acc = fmaf(xc, wl, acc);
            out[(size_t)(t0 + tt) * conv_dim + c] = ho_ssm_silu_sel<F32>(acc);
            #pragma unroll
            for (int k = 0; k < HO_CONV_MAX - 2; ++k)
                if (k + 1 < dm1) win[k] = win[k + 1];
            #pragma unroll
            for (int k = 0; k < HO_CONV_MAX - 1; ++k)
                if (k == dm1 - 1) win[k] = xc;
        }
    }
    if (blockIdx.y == 0) {
        #pragma unroll
        for (int k = 0; k < HO_CONV_MAX - 1; ++k)
            if (k < dm1) st[(size_t)k * conv_dim + c] = nst[k];
    }
}
extern "C" __global__ void ho_ssm_conv_tiled(const float* __restrict__ zx, int zx_stride, int xbc_off,
                                             int dt_off, float* __restrict__ st, const float* __restrict__ w,
                                             const float* __restrict__ bias, float* __restrict__ out,
                                             const float* __restrict__ A, const float* __restrict__ dt_b,
                                             float* __restrict__ dtA, int conv_dim, int d_conv, int n_head,
                                             int n_tok, int conv_blocks) {
    ho_ssm_conv_tiled_impl<false>(zx, zx_stride, xbc_off, dt_off, st, w, bias, out, A, dt_b, dtA, conv_dim,
                                  d_conv, n_head, n_tok, conv_blocks);
}
extern "C" __global__ void ho_ssm_conv_tiled_f32(const float* __restrict__ zx, int zx_stride, int xbc_off,
                                                 int dt_off, float* __restrict__ st, const float* __restrict__ w,
                                                 const float* __restrict__ bias, float* __restrict__ out,
                                                 const float* __restrict__ A, const float* __restrict__ dt_b,
                                                 float* __restrict__ dtA, int conv_dim, int d_conv, int n_head,
                                                 int n_tok, int conv_blocks) {
    ho_ssm_conv_tiled_impl<true>(zx, zx_stride, xbc_off, dt_off, st, w, bias, out, A, dt_b, dtA, conv_dim,
                                 d_conv, n_head, n_tok, conv_blocks);
}

// Reduce-scatter step of the xor butterfly: lanes with bit O set keep the upper half of the
// 2*O live partials, the others the lower half; each kept partial becomes mine + partner's.
template <int O>
__device__ __forceinline__ void ho_rs_step(float (&p)[HO_SCAN_T], int lane) {
    bool up = (lane & O) != 0;
    #pragma unroll
    for (int i = 0; i < O; ++i) {
        float send = up ? p[i] : p[i + O];
        float keep = up ? p[i + O] : p[i];
        p[i] = keep + __shfl_xor_sync(0xffffffffu, send, O);
    }
}

// Launch 2 (tiled): selective scan (ho_ssm_scan_tiled / _ns4). grid (ceil(head_dim / HO_SCAN_W),
// n_head), block 32 * HO_SCAN_W, dynamic smem 4 * (2*HO_SCAN_T*d_state + 2*HO_SCAN_T*HO_SCAN_W + 2*HO_SCAN_T)
// bytes. All warps of a block are channels of one head (same group, same dt / dA). Per tile of
// HO_SCAN_T tokens the block stages B and C of its group (coalesced), dt / dA of its head and x of
// its channels in shared memory; each warp then runs the serial state update over the tile (state
// in registers, as before), keeps one per-lane partial per token (independent chains -> ILP), and
// reduces all HO_SCAN_T partials with one 31-shuffle reduce-scatter (vs 5 dependent shuffles per
// token). y is staged in smem and stored per token row. Warps past head_dim only help load/sync.
// NS = d_state / 32 register slots per lane (instantiated for 4 = Nemotron's d_state 128, and
// HO_SCAN_NS = 8); slots past d_state are inactive in the naive kernel too, so dropping them
// changes nothing numerically.
template <int NS>
__device__ __forceinline__ void ho_ssm_scan_tiled_impl(const float* __restrict__ xbc, const float* __restrict__ dtA,
                                                       const float* __restrict__ D, float* __restrict__ state,
                                                       float* __restrict__ y, int n_head, int head_dim,
                                                       int d_state, int n_group, int n_tok) {
    extern __shared__ float ho_scan_sh[];
    float* sB = ho_scan_sh;                    // [HO_SCAN_T][d_state]
    float* sC = sB + HO_SCAN_T * d_state;      // [HO_SCAN_T][d_state]
    float* sx = sC + HO_SCAN_T * d_state;      // [HO_SCAN_T][HO_SCAN_W]
    float* sy = sx + HO_SCAN_T * HO_SCAN_W;    // [HO_SCAN_T][HO_SCAN_W]
    float* sdt = sy + HO_SCAN_T * HO_SCAN_W;   // [HO_SCAN_T]
    float* sdA = sdt + HO_SCAN_T;              // [HO_SCAN_T]
    int tid = threadIdx.x, nthr = blockDim.x;
    int lane = tid & 31, warp = tid >> 5;
    int h = blockIdx.y;
    int ch0 = blockIdx.x * HO_SCAN_W;
    int nw = min(HO_SCAN_W, head_dim - ch0);
    bool valid = warp < nw;
    int d_inner = n_head * head_dim;
    int conv_dim = d_inner + 2 * n_group * d_state;
    int g = h / (n_head / n_group);
    int row0 = h * head_dim + ch0;
    float Dh = D[h];
    float* sp = state + (size_t)(row0 + warp) * d_state;
    float s[NS];
    #pragma unroll
    for (int j = 0; j < NS; ++j) {
        int i = lane + 32 * j;
        s[j] = (valid && i < d_state) ? sp[i] : 0.f;
    }
    const float* Bg = xbc + d_inner + (size_t)g * d_state;
    const float* Cg = Bg + (size_t)n_group * d_state;
    for (int t0 = 0; t0 < n_tok; t0 += HO_SCAN_T) {
        int nt = min(HO_SCAN_T, n_tok - t0);
        for (int k = tid; k < nt * d_state; k += nthr) {
            int tt = k / d_state, i = k - tt * d_state;
            size_t off = (size_t)(t0 + tt) * conv_dim + i;
            sB[k] = Bg[off];
            sC[k] = Cg[off];
        }
        for (int k = tid; k < nt * HO_SCAN_W; k += nthr) {
            int tt = k / HO_SCAN_W, wi = k - tt * HO_SCAN_W;
            sx[k] = wi < nw ? xbc[(size_t)(t0 + tt) * conv_dim + row0 + wi] : 0.f;
        }
        if (tid < nt) {
            sdt[tid] = dtA[(size_t)(t0 + tid) * 2 * n_head + h];
            sdA[tid] = dtA[(size_t)(t0 + tid) * 2 * n_head + n_head + h];
        }
        __syncthreads();
        float part[HO_SCAN_T];
        #pragma unroll
        for (int tt = 0; tt < HO_SCAN_T; ++tt) {
            float sum = 0.f;
            if (tt < nt) {
                float dt = sdt[tt], dA = sdA[tt];
                float xv = sx[tt * HO_SCAN_W + warp];
                float x_dt = __fmul_rn(xv, dt);
                const float* B = sB + tt * d_state;
                const float* C = sC + tt * d_state;
                #pragma unroll
                for (int j = 0; j < NS; ++j) {
                    int i = lane + 32 * j;
                    if (i < d_state) {
                        float v = fmaf(s[j], dA, B[i] * x_dt);
                        s[j] = v;
                        sum = fmaf(v, C[i], sum);
                    }
                }
            }
            part[tt] = sum;
        }
        // Butterfly level xor-16 (HO_SCAN_T = 16 < 32: every lane keeps all 16 partials), then
        // reduce-scatter xor-8/4/2/1: afterwards lane l (and l ^ 16) holds token (l & 15).
        #pragma unroll
        for (int tt = 0; tt < HO_SCAN_T; ++tt) part[tt] += __shfl_xor_sync(0xffffffffu, part[tt], 16);
        ho_rs_step<8>(part, lane);
        ho_rs_step<4>(part, lane);
        ho_rs_step<2>(part, lane);
        ho_rs_step<1>(part, lane);
        int tl = lane & (HO_SCAN_T - 1);
        if (lane < HO_SCAN_T && tl < nt) sy[tl * HO_SCAN_W + warp] = fmaf(sx[tl * HO_SCAN_W + warp], Dh, part[0]);
        __syncthreads();
        for (int k = tid; k < nt * HO_SCAN_W; k += nthr) {
            int tt = k / HO_SCAN_W, wi = k - tt * HO_SCAN_W;
            if (wi < nw) y[(size_t)(t0 + tt) * d_inner + row0 + wi] = sy[k];
        }
    }
    if (valid) {
        #pragma unroll
        for (int j = 0; j < NS; ++j) {
            int i = lane + 32 * j;
            if (i < d_state) sp[i] = s[j];
        }
    }
}
extern "C" __global__ void
ho_ssm_scan_tiled(const float* __restrict__ xbc, const float* __restrict__ dtA, const float* __restrict__ D,
                  float* __restrict__ state, float* __restrict__ y, int n_head, int head_dim, int d_state,
                  int n_group, int n_tok) {
    ho_ssm_scan_tiled_impl<HO_SCAN_NS>(xbc, dtA, D, state, y, n_head, head_dim, d_state, n_group, n_tok);
}
// d_state <= 128 (Nemotron-H): 4 state registers per lane.
extern "C" __global__ void
ho_ssm_scan_tiled_ns4(const float* __restrict__ xbc, const float* __restrict__ dtA, const float* __restrict__ D,
                      float* __restrict__ state, float* __restrict__ y, int n_head, int head_dim, int d_state,
                      int n_group, int n_tok) {
    ho_ssm_scan_tiled_impl<4>(xbc, dtA, D, state, y, n_head, head_dim, d_state, n_group, n_tok);
}

// ---- Mamba-2 selective scan, prefill fast path (ARTIFICER_PREFILL_SSM=fp32) ----
// Same math as ho_ssm_scan_tiled (above, unchanged), restructured:
//  * d_state is a compile-time constant (DS = 32 * NS; instantiated for NS = 4, d_state 128):
//    no runtime division in the staging, constant smem strides.
//  * HO_SCAN2_C = 2 channels per warp (16 channels per block): each B / C element read from smem
//    feeds both channels, and every block stages B / C for twice as many channels (half the
//    redundant L2 -> smem traffic of the per-head blocks).
//  * B / C staged with 16-byte cp.async (x, dt, dA with 4-byte cp.async) into a double buffer:
//    tile k+1 loads while tile k computes; ONE __syncthreads per tile (y of tile k-1 is stored
//    from a second smem y buffer during tile k).
// Staging: blockDim (256) == HO_SCAN_T * HO_SCAN2_CH, so the staging is straight-line: each thread owns
// 4 B / C 16-byte chunks (rows warp and warp + 8, B and C), one x element, one y element, and threads 0..31
// one dt / dA value; pointers are computed once and advanced by HO_SCAN_T rows per tile. Full tiles run a
// branch-free compute (template <bool FULL>); only the last (partial) tile keeps the per-token guards.
// dt / dA are read as float4 (broadcast), the warp's 2 x values as one float2. Same arithmetic as v1.
// Per channel and token the state update, the per-lane partial (FMA chain over j = 0..NS-1, lane
// slot i = lane + 32 j), the 32-lane xor tree (16, 8, 4, 2, 1 via the same reduce-scatter) and the
// final fmaf(x, D, sum) are exactly those of ho_ssm_scan_tiled / ho_ssm_scan_rows, so y and the
// state are bit-identical to them. Requires conv_dim % 4 == 0 and d_inner % 4 == 0 (16-byte
// aligned B / C rows; the host checks and otherwise uses ho_ssm_scan_tiled).
// grid (ceil(head_dim / 16), n_head), block 32 * HO_SCAN2_W, dynamic smem ho_scan2_smem<NS> bytes.
#define HO_SCAN2_W 8                         // warps per block
#define HO_SCAN2_C 2                         // channels per warp
#define HO_SCAN2_CH (HO_SCAN2_W * HO_SCAN2_C) // channels per block
__device__ __forceinline__ void ho_cp_async16(float* smem, const float* g) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void ho_cp_async4(float* smem, const float* g) {
    unsigned s = (unsigned)__cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(s), "l"(g) : "memory");
}
__device__ __forceinline__ void ho_cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void ho_cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// One smem stage: B [T][DS], C [T][DS], x [T][CH], dt [T], dA [T] (floats; 16-byte multiple).
template <int NS> struct HoScan2 {
    static constexpr int DS = 32 * NS;
    static constexpr int T = HO_SCAN_T;
    static constexpr int CH = HO_SCAN2_CH;
    static constexpr int STAGE = 2 * T * DS + T * CH + 2 * T;
    static constexpr int SMEM_FLOATS = 2 * STAGE + 2 * T * CH; // 2 stages + 2 y buffers
};

template <bool FULL>
__device__ __forceinline__ void ho_scan2_compute(const float* st, float* sy, float (&s)[HO_SCAN2_C][4], int nt,
                                                 int warp, int lane, float Dh) {
    using K = HoScan2<4>;
    constexpr int DS = K::DS, T = K::T, CH = K::CH, C2 = HO_SCAN2_C, NS = 4;
    const float* sB = st;
    const float* sC = st + T * DS;
    const float* sx = sC + T * DS;
    const float* sdt = sx + T * CH;
    const float* sdA = sdt + T;
    float part[C2][T];
    #pragma unroll
    for (int q = 0; q < T / 4; ++q) {
        float4 dt4 = reinterpret_cast<const float4*>(sdt)[q];
        float4 dA4 = reinterpret_cast<const float4*>(sdA)[q];
        float dts[4] = {dt4.x, dt4.y, dt4.z, dt4.w}, dAs[4] = {dA4.x, dA4.y, dA4.z, dA4.w};
        #pragma unroll
        for (int u = 0; u < 4; ++u) {
            int tt = 4 * q + u;
            float sum[C2] = {0.f, 0.f};
            if (FULL || tt < nt) {
                float2 x2 = *reinterpret_cast<const float2*>(sx + tt * CH + warp * C2);
                float x_dt[C2] = {__fmul_rn(x2.x, dts[u]), __fmul_rn(x2.y, dts[u])};
                const float* B = sB + tt * DS;
                const float* C = sC + tt * DS;
                #pragma unroll
                for (int j = 0; j < NS; ++j) {
                    int i = lane + 32 * j;
                    float b = B[i], cc = C[i];
                    #pragma unroll
                    for (int k = 0; k < C2; ++k) {
                        float v = fmaf(s[k][j], dAs[u], b * x_dt[k]);
                        s[k][j] = v;
                        sum[k] = fmaf(v, cc, sum[k]);
                    }
                }
            }
            #pragma unroll
            for (int k = 0; k < C2; ++k) part[k][tt] = sum[k];
        }
    }
    int tl = lane & (T - 1);
    #pragma unroll
    for (int k = 0; k < C2; ++k) {
        #pragma unroll
        for (int tt = 0; tt < T; ++tt) part[k][tt] += __shfl_xor_sync(0xffffffffu, part[k][tt], 16);
        ho_rs_step<8>(part[k], lane);
        ho_rs_step<4>(part[k], lane);
        ho_rs_step<2>(part[k], lane);
        ho_rs_step<1>(part[k], lane);
        int c = warp * C2 + k;
        if (lane < T && (FULL || tl < nt)) sy[tl * CH + c] = fmaf(sx[tl * CH + c], Dh, part[k][0]);
    }
}

// d_state == 128 (Nemotron-H). Dynamic smem: 4 * HoScan2<4>::SMEM_FLOATS = 37120 bytes.
// grid (ceil(head_dim / 16), n_head), block 256 (required: one x / y element per thread).
extern "C" __global__ void __launch_bounds__(32 * HO_SCAN2_W, 2)
ho_ssm_scan_fast_ns4(const float* __restrict__ xbc, const float* __restrict__ dtA, const float* __restrict__ D,
                     float* __restrict__ state, float* __restrict__ y, int n_head, int head_dim, int n_group,
                     int n_tok) {
    using K = HoScan2<4>;
    constexpr int DS = K::DS, T = K::T, CH = K::CH, C2 = HO_SCAN2_C, NS = 4;
    static_assert(32 * HO_SCAN2_W == T * CH, "one x / y element per thread");
    static_assert(HO_SCAN2_W == 8 && T == 16 && C2 == 2, "B / C staging assumes rows warp and warp + 8");
    extern __shared__ __align__(16) float ho_scan2_sh[];
    float* sy0 = ho_scan2_sh + 2 * K::STAGE;     // [2][T][CH]
    int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    int h = blockIdx.y;
    int ch0 = blockIdx.x * CH;
    int nch = min(CH, head_dim - ch0);
    int d_inner = n_head * head_dim;
    int conv_dim = d_inner + 2 * n_group * DS;
    int g = h / (n_head / n_group);
    int row0 = h * head_dim + ch0;
    float Dh = D[h];
    // Per-thread staging pointers at token 0 (advanced by t0 rows per tile).
    const float* gB = xbc + (size_t)warp * conv_dim + d_inner + g * DS + 4 * lane; // B row `warp`, chunk `lane`
    size_t row8 = (size_t)8 * conv_dim, coff = (size_t)n_group * DS;              // +8 rows; B -> C
    int xtt = tid >> 4, xc = tid & 15;                                             // this thread's x / y element
    bool xok = xc < nch;
    const float* gx = xbc + (size_t)xtt * conv_dim + row0 + xc;
    int dtt = tid & 15;                                                            // tid < 16: dt, 16..31: dA
    const float* gd = dtA + (size_t)dtt * 2 * n_head + (tid >> 4) * n_head + h;
    float* gy = y + (size_t)xtt * d_inner + row0 + xc;
    int sBo = warp * DS + 4 * lane;
    float s[C2][NS];
    #pragma unroll
    for (int k = 0; k < C2; ++k) {
        int c = warp * C2 + k;
        const float* sp = state + (size_t)(row0 + c) * DS;
        #pragma unroll
        for (int j = 0; j < NS; ++j) s[k][j] = c < nch ? sp[lane + 32 * j] : 0.f;
    }
    auto stage = [&](float* st, int t0) {
        int nt = min(T, n_tok - t0);
        bool full = nt == T;
        const float* b = gB + (size_t)t0 * conv_dim;
        if (full || warp < nt) {
            ho_cp_async16(st + sBo, b);
            ho_cp_async16(st + T * DS + sBo, b + coff);
        }
        if (full || warp + 8 < nt) {
            ho_cp_async16(st + 8 * DS + sBo, b + row8);
            ho_cp_async16(st + T * DS + 8 * DS + sBo, b + row8 + coff);
        }
        float* sx = st + 2 * T * DS;
        if (xok && (full || xtt < nt)) ho_cp_async4(sx + tid, gx + (size_t)t0 * conv_dim);
        else sx[tid] = 0.f;
        if (tid < 32 && (full || dtt < nt)) ho_cp_async4(sx + T * CH + tid, gd + (size_t)t0 * 2 * n_head);
        ho_cp_async_commit();
    };
    stage(ho_scan2_sh, 0);
    int it = 0;
    for (int t0 = 0; t0 < n_tok; t0 += T, ++it) {
        float* st = ho_scan2_sh + (it & 1) * K::STAGE;
        float* sy = sy0 + (it & 1) * T * CH;
        ho_cp_async_wait_all();
        __syncthreads(); // tile `it` visible; every thread is done with tile it-1 and with sy[(it-1) & 1]'s writes
        if (t0 + T < n_tok) stage(ho_scan2_sh + ((it + 1) & 1) * K::STAGE, t0 + T);
        // y of tile it-1 (always a full tile: only the last tile can be partial)
        if (it > 0 && xok) gy[(size_t)(t0 - T) * d_inner] = sy0[((it + 1) & 1) * T * CH + tid];
        int nt = min(T, n_tok - t0);
        if (nt == T) ho_scan2_compute<true>(st, sy, s, nt, warp, lane, Dh);
        else ho_scan2_compute<false>(st, sy, s, nt, warp, lane, Dh);
    }
    __syncthreads();
    if (it > 0) {
        int t0 = (it - 1) * T;
        if (xok && xtt < n_tok - t0) gy[(size_t)t0 * d_inner] = sy0[((it + 1) & 1) * T * CH + tid];
    }
    #pragma unroll
    for (int k = 0; k < C2; ++k) {
        int c = warp * C2 + k;
        if (c < nch) {
            float* sp = state + (size_t)(row0 + c) * DS;
            #pragma unroll
            for (int j = 0; j < NS; ++j) sp[lane + 32 * j] = s[k][j];
        }
    }
}
// ---- Split-K flash-decode, new partial + reduce (ARTIFICER_FD_KERNEL=new) ----
// Same split-K algorithm, partition and per-key math as fd_partial_f16 / fd_reduce above (those are the
// original kernels and are what ARTIFICER_FD_KERNEL=old launches). Bit-identical by construction (same
// split ranges, per-key op order and reduction tree; only launch shape and register use differ):
//  * fd_partial_f16_g5d128: fd_partial_f16 specialised to kv_mul = 5, head_dim = 128 (vec 4) f16
//    KV, launched with the same grid (n_head_kv, n_split), block 32 * FD_WARPS and arguments (head_dim, kv_mul
//    are ignored; the host only picks it for that shape, kv_dim % 4 == 0 and 8-byte aligned K/V). Differences
//    are mechanical: o[5][4] / m[5] / l[5] instead of [8][8] arrays (fewer registers, __launch_bounds__),
//    one 8-byte load per lane per K / V row (the same 4 halves, unpacked with the same h2f), the next key of the
//    warp's sequence prefetched into registers before the current key is processed, q read from smem as float4.
//    Key order per warp (t = t0 + w, += FD_WARPS), split ranges, the 5-step xor shuffle tree, the per-key op
//    sequence and the partial layout are unchanged. Every FP op is written as the explicit intrinsic matching
//    what NVRTC/ptxas emit for fd_partial_f16 (checked in its PTX + SASS: FFMA dot chain from +0, FADD per
//    shuffle step, FMUL scale, IEEE div + tanhf + FMUL for softcap, FMNMX, FADD subtractions into the same
//    expf, FFMA l = l*corr + p, FMUL p*v then FFMA o = o*corr + p*v), so no contraction choice is left to
//    the compiler.
//  * fd_reduce_smem: grid (n_head, ceil(head_dim / FDR_THREADS)), FDR_THREADS threads; one thread per
//    output element as before, with the per-partial m / l loads hoisted into smem and expf(m_p - M) computed
//    once per partial instead of once per thread. M is the same left-to-right fmaxf chain, and each element is
//    accumulated over p = 0 .. n_part-1 in order with the same skip of empty partials (l_p == 0), the same
//    fmaf(l_p, w_p, L), fmaf(o_p[d], w_p, acc) and the same acc / L. Works for any fd_supported shape
//    (layout is that of fd_partial_*), so it is used with either partial kernel.
#define FD2_KVM 5
#define FD2_HD 128
#ifndef FD2_MIN_BLOCKS
#define FD2_MIN_BLOCKS 6
#endif
__device__ __forceinline__ void fd2_unpack4(uint2 r, float (&f)[4]) {
    f[0] = h2f((u16)(r.x & 0xffffu));
    f[1] = h2f((u16)(r.x >> 16));
    f[2] = h2f((u16)(r.y & 0xffffu));
    f[3] = h2f((u16)(r.y >> 16));
}
extern "C" __global__ void __launch_bounds__(32 * FD_WARPS, FD2_MIN_BLOCKS)
fd_partial_f16_g5d128(const float* __restrict__ q, const u16* __restrict__ K, const u16* __restrict__ V,
                      float* __restrict__ part, int head_dim, int kv_dim, int kv_mul, int start, int len,
                      int chunk, float scale, float softcap) {
    (void)head_dim;  // == FD2_HD (host guard)
    (void)kv_mul;    // == FD2_KVM (host guard)
    __shared__ __align__(16) float qs[FD2_KVM * FD2_HD];
    const int hkv = blockIdx.x, s = blockIdx.y, n_split = gridDim.y;
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int h0 = hkv * FD2_KVM;
    for (int i = threadIdx.x; i < FD2_KVM * FD2_HD; i += blockDim.x) qs[i] = q[(size_t)h0 * FD2_HD + i];
    __syncthreads();
    float m[FD2_KVM], l[FD2_KVM], o[FD2_KVM][4];
    #pragma unroll
    for (int g = 0; g < FD2_KVM; ++g) {
        m[g] = -3.402823466e38f; l[g] = 0.f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) o[g][j] = 0.f;
    }
    const int t0 = s * chunk, t1 = min(len, t0 + chunk);
    const int d0 = lane * 4;
    // Element index (start + t) * kv_dim + hkv * 128 + d0 is a multiple of 4 (kv_dim % 4 == 0): uint2 index.
    const uint2* __restrict__ K2 = reinterpret_cast<const uint2*>(K);
    const uint2* __restrict__ V2 = reinterpret_cast<const uint2*>(V);
    const size_t row4 = (size_t)kv_dim >> 2;
    const size_t col4 = ((size_t)hkv * FD2_HD + d0) >> 2;
    int t = t0 + w;
    uint2 kc = make_uint2(0u, 0u), vc = make_uint2(0u, 0u);
    if (t < t1) {
        const size_t b = (size_t)(start + t) * row4 + col4;
        kc = __ldg(K2 + b);
        vc = __ldg(V2 + b);
    }
    for (; t < t1; t += FD_WARPS) {
        // Prefetch this warp's next key (t + FD_WARPS) before computing key t; processing order is unchanged.
        uint2 kn = make_uint2(0u, 0u), vn = make_uint2(0u, 0u);
        if (t + FD_WARPS < t1) {
            const size_t b = (size_t)(start + t + FD_WARPS) * row4 + col4;
            kn = __ldg(K2 + b);
            vn = __ldg(V2 + b);
        }
        float kr[4], vr[4];
        fd2_unpack4(kc, kr);
        fd2_unpack4(vc, vr);
        #pragma unroll
        for (int g = 0; g < FD2_KVM; ++g) {
            const float4 qv = *reinterpret_cast<const float4*>(qs + g * FD2_HD + d0);
            float dot = 0.f;
            dot = __fmaf_rn(qv.x, kr[0], dot);
            dot = __fmaf_rn(qv.y, kr[1], dot);
            dot = __fmaf_rn(qv.z, kr[2], dot);
            dot = __fmaf_rn(qv.w, kr[3], dot);
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) dot = __fadd_rn(dot, __shfl_xor_sync(0xffffffffu, dot, off));
            float sc = __fmul_rn(dot, scale);
            if (softcap > 0.f) sc = __fmul_rn(softcap, tanhf(__fdiv_rn(sc, softcap)));
            const float mn = fmaxf(m[g], sc);
            const float corr = expf(__fsub_rn(m[g], mn)), p = expf(__fsub_rn(sc, mn));
            l[g] = __fmaf_rn(l[g], corr, p);
            #pragma unroll
            for (int j = 0; j < 4; ++j) o[g][j] = __fmaf_rn(o[g][j], corr, __fmul_rn(p, vr[j]));
            m[g] = mn;
        }
        kc = kn;
        vc = vn;
    }
    const int n_part = n_split * FD_WARPS, pidx = s * FD_WARPS + w;
    #pragma unroll
    for (int g = 0; g < FD2_KVM; ++g) {
        float* pp = part + ((size_t)(h0 + g) * n_part + pidx) * (FD2_HD + 2);
        #pragma unroll
        for (int j = 0; j < 4; ++j) pp[d0 + j] = o[g][j];
        if (lane == 0) { pp[FD2_HD] = m[g]; pp[FD2_HD + 1] = l[g]; }
    }
}
#define FDR_THREADS 32
#define FDR_MAXPART 256  // FD_MAX_SPLIT (64) * FD_WARPS
extern "C" __global__ void __launch_bounds__(FDR_THREADS)
fd_reduce_smem(const float* __restrict__ part, float* __restrict__ out, int head_dim, int n_part) {
    __shared__ float ls[FDR_MAXPART], ws[FDR_MAXPART];
    const int h = blockIdx.x, d = blockIdx.y * FDR_THREADS + threadIdx.x;
    const size_t stride = (size_t)head_dim + 2;
    const float* __restrict__ ph = part + (size_t)h * n_part * stride;
    for (int p = threadIdx.x; p < n_part; p += FDR_THREADS) {
        ws[p] = ph[(size_t)p * stride + head_dim];      // m_p (replaced by its weight below)
        ls[p] = ph[(size_t)p * stride + head_dim + 1];  // l_p
    }
    __syncthreads();
    float M = -3.402823466e38f;
    for (int p = 0; p < n_part; ++p) M = fmaxf(M, ws[p]);
    __syncthreads();
    for (int p = threadIdx.x; p < n_part; p += FDR_THREADS) ws[p] = expf(__fsub_rn(ws[p], M));
    __syncthreads();
    const bool act = d < head_dim;
    const float* __restrict__ pd = ph + (act ? d : 0);
    float L = 0.f, acc = 0.f;
    #pragma unroll 8
    for (int p = 0; p < n_part; ++p) {
        const float lp = ls[p], wgt = ws[p];
        const float x = act ? __ldg(pd + (size_t)p * stride) : 0.f;  // independent of lp: loads pipeline
        if (lp != 0.f) {  // fd_reduce: `if (lp == 0.f) continue;` (empty partial)
            L = fmaf(lp, wgt, L);
            acc = fmaf(x, wgt, acc);
        }
    }
    if (act) out[(size_t)h * head_dim + d] = (L > 0.f) ? acc / L : 0.f;
}
