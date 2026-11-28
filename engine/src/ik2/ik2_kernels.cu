// Based on Project Maya (https://github.com/mw00/project-maya)
// Copyright (c) 2026 Peasant Smith, OscarSheng, alesha-pro, cobra91 and the Project Maya contributors
// Licensed under the MIT License.

// src/ik2/ik2_kernels.cu - strata-ik2's kernels (the interface and the conventions: ik2_kernels.cuh).
#include "ik2_kernels.cuh"
#include "ik2_iq_tables.h"

#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace ik2::k {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ik2::k::%s: launch failed: %s\n", what, cudaGetErrorString(e));
        std::abort();
    }
}

__device__ __forceinline__ float h2f(uint16_t h) { return __half2float(__ushort_as_half(h)); }
__device__ __forceinline__ uint16_t f2h(float f) { return __half_as_ushort(__float2half_rn(f)); }
__device__ __forceinline__ int kv_int_nib(unsigned n) { return (int) n - ((n & 8u) ? 16 : 0); }   // int4 sign-extend (shared with the WMMA flash int staging)
__device__ __forceinline__ float2 u32_to_f2(uint32_t u) {
    const __half2 h2 = __halves2half2(__ushort_as_half((uint16_t) (u & 0xFFFFu)),
                                      __ushort_as_half((uint16_t) (u >> 16)));
    return __half22float2(h2);
}
__host__ __device__ __forceinline__ float bits_to_float(uint32_t u) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __uint_as_float(u);
#else
    float f;
    std::memcpy(&f, &u, 4);
    return f;
#endif
}
__host__ __device__ __forceinline__ uint32_t float_to_bits(float f) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __float_as_uint(f);
#else
    uint32_t u;
    std::memcpy(&u, &f, 4);
    return u;
#endif
}
__host__ __device__ inline float half_bits_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t) (h >> 15) << 31, e = (h >> 10) & 31, m = h & 1023;
    uint32_t u;
    if (e == 0) {
        if (m == 0) {
            u = sign;
        } else {   // subnormal
            int sh = 0;
            uint32_t mm = m;
            while (!(mm & 1024)) mm <<= 1, ++sh;
            u = sign | ((uint32_t) (113 - sh) << 23) | ((mm & 1023) << 13);
        }
    } else if (e == 31) {
        u = sign | 0x7f800000u | (m << 13);
    } else {
        u = sign | ((e + 112) << 23) | (m << 13);
    }
    return bits_to_float(u);
}
__host__ __device__ __forceinline__ uint8_t f32_to_e4m3fn(float v) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return (uint8_t) __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
#else
    const uint32_t u = float_to_bits(v);
    const uint8_t sign = (uint8_t) ((u >> 24) & 0x80u);
    const uint32_t ax = u & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) return sign | 0x7Fu;
    if (ax >= 0x43E00000u) return sign | 0x7Eu;
    if (ax < 0x3C800000u) {
#if defined(__CUDA_ARCH__)
        const uint32_t sub = float_to_bits(__fadd_rn(bits_to_float(ax), 16384.0f)) - 0x46800000u;
#else
        const uint32_t sub = float_to_bits(bits_to_float(ax) + 16384.0f) - 0x46800000u;
#endif
        return sign | (uint8_t) sub;
    }
    const uint32_t lsb = (ax >> 20) & 1u;
    return sign | (uint8_t) (((ax + 0x7FFFFu + lsb) >> 20) - (120u << 3));
#endif
}
__device__ __forceinline__ uint16_t f32x2_to_e4m3fnx2(float a, float b) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return (uint16_t) __nv_cvt_float2_to_fp8x2(make_float2(a, b), __NV_SATFINITE, __NV_E4M3);
#else
    return (uint16_t) f32_to_e4m3fn(a) | ((uint16_t) f32_to_e4m3fn(b) << 8);
#endif
}
__host__ __device__ __forceinline__ uint16_t e4m3fn_to_f16(uint8_t q) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __half_as_ushort(__half(__nv_cvt_fp8_to_halfraw((__nv_fp8_storage_t) q, __NV_E4M3)));
#else
    const uint16_t sign = (uint16_t) (q & 0x80u) << 8;
    const uint32_t mag = q & 0x7Fu;
    if (mag == 0x7Fu) return sign | 0x7FFFu;
    if (mag < 8u) {
#if defined(__CUDA_ARCH__)
        return sign | (uint16_t) (__byte_perm(0x1E1C1800u, 0x23222120u, mag << 4) & 0xFF00u);
#else
        static constexpr uint16_t kSub[8] = {0x0000u, 0x1800u, 0x1C00u, 0x1E00u, 0x2000u, 0x2100u, 0x2200u, 0x2300u};
        return sign | kSub[mag];
#endif
    }
    return sign | (uint16_t) ((mag << 7) + 0x2000u);
#endif
}
__device__ __forceinline__ uint32_t e4m3fnx2_to_h2_u32(uint16_t p2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    const __half2 h2 = __half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) p2, __NV_E4M3));
    return (uint32_t) __half_as_ushort(__low2half(h2)) | ((uint32_t) __half_as_ushort(__high2half(h2)) << 16);
#else
    return (uint32_t) e4m3fn_to_f16((uint8_t) (p2 & 0xFFu)) |
           ((uint32_t) e4m3fn_to_f16((uint8_t) (p2 >> 8)) << 16);
#endif
}
__host__ __device__ __forceinline__ float e4m3fn_to_f32(uint8_t q) {
#if defined(__CUDA_ARCH__)
    return h2f(e4m3fn_to_f16(q));
#else
    return half_bits_to_float(e4m3fn_to_f16(q));
#endif
}
__device__ __forceinline__ float2 e4m3fnx2_to_f2(uint16_t p2) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2((__nv_fp8x2_storage_t) p2, __NV_E4M3)));
#else
    return u32_to_f2(e4m3fnx2_to_h2_u32(p2));
#endif
}

// the block's sum / max of v (blockDim.x a power of two, sh [blockDim.x]); every thread gets it
__device__ float bsum(float v, float* sh) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (t < o) sh[t] += sh[t + o];
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}
__device__ float bmax(float v, float* sh) {
    const int t = threadIdx.x;
    sh[t] = v;
    __syncthreads();
    for (int o = blockDim.x >> 1; o > 0; o >>= 1) {
        if (t < o) sh[t] = fmaxf(sh[t], sh[t + o]);
        __syncthreads();
    }
    const float r = sh[0];
    __syncthreads();
    return r;
}

constexpr int kBlk = 256;

__device__ __forceinline__ uint32_t f2_to_u32(float a, float b) {
    const __half2 h2 = __float22half2_rn(make_float2(a, b));
    return (uint32_t) __half_as_ushort(__low2half(h2)) | ((uint32_t) __half_as_ushort(__high2half(h2)) << 16);
}

template <bool WRITE_F16, bool WRITE_F32>
__global__ void rms_impl_kernel(const float* x, const float* w, int D, float eps, uint16_t* y16, float* y32) {
    __shared__ float sh[kBlk];
    const float* xr = x + (size_t) blockIdx.x * D;
    uint16_t* y16r = WRITE_F16 ? y16 + (size_t) blockIdx.x * D : nullptr;
    float* y32r = WRITE_F32 ? y32 + (size_t) blockIdx.x * D : nullptr;
    const bool vec4 = (((uintptr_t) xr & 15) == 0) && (!w || ((uintptr_t) w & 15) == 0) &&
                      (!WRITE_F32 || ((uintptr_t) y32r & 15) == 0) &&
                      (!WRITE_F16 || ((uintptr_t) y16r & 7) == 0) && ((D & 3) == 0);
    float a = 0.0f;
    if (vec4) {
        const int n4 = D >> 2;
        const float4* x4 = (const float4*) xr;
        for (int d4 = threadIdx.x; d4 < n4; d4 += kBlk) {
            const float4 v = __ldg(x4 + d4);
            a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float rs = rsqrtf(bsum(a, sh) / (float) D + eps);
        const float4* w4 = (const float4*) w;
        float4* y32_4 = (float4*) y32r;
        uint2* y16_4 = (uint2*) y16r;
        for (int d4 = threadIdx.x; d4 < n4; d4 += kBlk) {
            const float4 v = __ldg(x4 + d4);
            const float4 ww = w4 ? __ldg(w4 + d4) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float r0 = v.x * rs * ww.x, r1 = v.y * rs * ww.y, r2 = v.z * rs * ww.z, r3 = v.w * rs * ww.w;
            if (WRITE_F32) y32_4[d4] = make_float4(r0, r1, r2, r3);
            if (WRITE_F16) y16_4[d4] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
        }
    } else {
        for (int d = threadIdx.x; d < D; d += kBlk) a += xr[d] * xr[d];
        const float rs = rsqrtf(bsum(a, sh) / (float) D + eps);
        for (int d = threadIdx.x; d < D; d += kBlk) {
            const float r = xr[d] * rs * (w ? w[d] : 1.0f);
            if (WRITE_F32) y32r[d] = r;
            if (WRITE_F16) y16r[d] = f2h(r);
        }
    }
}
// rms of up to three vectors, each with its own weights, in one launch (a block a vector; rms_impl_kernel's sums)
__global__ void rms_multi_kernel(RmsJobs jobs, int D, float eps) {
    __shared__ float sh[kBlk];
    const float* xr = jobs.x[blockIdx.x];
    const float* w = jobs.w[blockIdx.x];
    float* y32r = jobs.y[blockIdx.x];
    const bool vec4 = (((uintptr_t) xr & 15) == 0) && (!w || ((uintptr_t) w & 15) == 0) &&
                      (((uintptr_t) y32r & 15) == 0) && ((D & 3) == 0);
    float a = 0.0f;
    if (vec4) {
        const int n4 = D >> 2;
        const float4* x4 = (const float4*) xr;
        for (int d4 = threadIdx.x; d4 < n4; d4 += kBlk) {
            const float4 v = __ldg(x4 + d4);
            a += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
        const float rs = rsqrtf(bsum(a, sh) / (float) D + eps);
        const float4* w4 = (const float4*) w;
        float4* y32_4 = (float4*) y32r;
        for (int d4 = threadIdx.x; d4 < n4; d4 += kBlk) {
            const float4 v = __ldg(x4 + d4);
            const float4 ww = w4 ? __ldg(w4 + d4) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float r0 = v.x * rs * ww.x, r1 = v.y * rs * ww.y, r2 = v.z * rs * ww.z, r3 = v.w * rs * ww.w;
            y32_4[d4] = make_float4(r0, r1, r2, r3);
        }
    } else {
        for (int d = threadIdx.x; d < D; d += kBlk) a += xr[d] * xr[d];
        const float rs = rsqrtf(bsum(a, sh) / (float) D + eps);
        for (int d = threadIdx.x; d < D; d += kBlk) y32r[d] = xr[d] * rs * (w ? w[d] : 1.0f);
    }
}

__device__ __forceinline__ float wsum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
// IK2_ATTN_DPP (opt-in): bit-exact wave32 xor-butterfly reduction using RDNA3 register ops.
// __shfl_xor_sync lowers to ds_bpermute on gfx1100; DPP row_xmask (M=1,2,4,8) and
// v_permlanex16 (M=16) exchange the SAME lane partner at each step, in the SAME order
// (16,8,4,2,1), so the float additions happen in exactly the same sequence as wsum().
// Builtins mirrored from repos/strata-ik2/engine/src/kernels/cuda/qsa_decode_attn.cu:510-528
// and repos/Strata-rdna3/include/strata/hip_compat/intrinsics.hpp:116-130 (gfx11 dpp_xor32).
template <int M>
__device__ __forceinline__ float dpp_xor_exchange(float v) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    const int x = __float_as_int(v);
    if constexpr (M == 16)
        return __int_as_float(__builtin_amdgcn_permlanex16(0, x, 0x76543210, 0xfedcba98, false, false));
    else
        return __int_as_float(__builtin_amdgcn_update_dpp(0, x, 0x160 | M, 0xf, 0xf, false));
#else
    return __shfl_xor_sync(0xffffffffu, v, M);
#endif
}
__device__ __forceinline__ float wsum_dpp(float v) {
    v += dpp_xor_exchange<16>(v);
    v += dpp_xor_exchange<8>(v);
    v += dpp_xor_exchange<4>(v);
    v += dpp_xor_exchange<2>(v);
    v += dpp_xor_exchange<1>(v);
    return v;
}
__device__ __forceinline__ float wmax(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}

// Block reductions over NW warps (blockDim.x = NW*32): warp reduce, then one tiny shared pass.
// gfx1100 wave32: 5 shuffle steps + 2 barriers versus bmax/bsum's 8-step tree (16 barriers).
template <int NW>
__device__ __forceinline__ float block_max(float v, float* sh) {
    v = wmax(v);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) sh[w] = v;
    __syncthreads();
    if (w == 0) {
        float x = (lane < NW) ? sh[lane] : -INFINITY;
        x = wmax(x);
        if (lane == 0) sh[0] = x;
    }
    __syncthreads();
    return sh[0];
}
template <int NW>
__device__ __forceinline__ float block_sum(float v, float* sh) {
    v = wsum(v);
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (lane == 0) sh[w] = v;
    __syncthreads();
    if (w == 0) {
        float x = (lane < NW) ? sh[lane] : 0.0f;
        x = wsum(x);
        if (lane == 0) sh[0] = x;
    }
    __syncthreads();
    return sh[0];
}

__device__ __forceinline__ void l2clip_warp(const float* src, const float* sc, float* dst32, uint16_t* dst16,
                                            int D, float th, float tau) {
    const int lane = threadIdx.x & 31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (!sc || ((uintptr_t) sc & 15) == 0) &&
        (!dst32 || ((uintptr_t) dst32 & 15) == 0) && (!dst16 || ((uintptr_t) dst16 & 7) == 0)) {
        const float4 xv = ((const float4*) src)[lane];
        const float4 sv = sc ? __ldg((const float4*) sc + lane) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        const float v0 = xv.x * sv.x, v1 = xv.y * sv.y, v2 = xv.z * sv.z, v3 = xv.w * sv.w;
        const float nrm = sqrtf(wsum(v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3));
        const float f = fminf(1.0f, th / (nrm + 1e-6f)) * tau;
        const float o0 = v0 * f, o1 = v1 * f, o2 = v2 * f, o3 = v3 * f;
        if (dst32) ((float4*) dst32)[lane] = make_float4(o0, o1, o2, o3);
        if (dst16) ((uint2*) dst16)[lane] = make_uint2(f2_to_u32(o0, o1), f2_to_u32(o2, o3));
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d] * (sc ? sc[d] : 1.0f);
        a += v * v;
    }
    const float nrm = sqrtf(wsum(a));
    const float f = fminf(1.0f, th / (nrm + 1e-6f)) * tau;
    for (int d = lane; d < D; d += 32) {
        const float o = src[d] * (sc ? sc[d] : 1.0f) * f;
        if (dst32) dst32[d] = o;
        if (dst16) dst16[d] = f2h(o);
    }
}

__device__ __forceinline__ void rms_f16_warp(const float* src, uint16_t* dst16, int D, float eps) {
    const int lane = threadIdx.x & 31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (((uintptr_t) dst16 & 7) == 0)) {
        const float4 xv = ((const float4*) src)[lane];
        const float rs = rsqrtf(wsum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w) / (float) D + eps);
        ((uint2*) dst16)[lane] = make_uint2(f2_to_u32(xv.x * rs, xv.y * rs), f2_to_u32(xv.z * rs, xv.w * rs));
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float rs = rsqrtf(wsum(a) / (float) D + eps);
    for (int d = lane; d < D; d += 32) dst16[d] = f2h(src[d] * rs);
}

__device__ __forceinline__ void l2clip_fp8_warp(const float* src, uint8_t* dst8, int D, float th, float k_scale) {
    const int lane = threadIdx.x & 31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (((uintptr_t) dst8 & 3) == 0)) {
        const float4 xv = ((const float4*) src)[lane];
        const float nrm = sqrtf(wsum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w));
        const float f = fminf(1.0f, th / (nrm + 1e-6f)) * k_scale;
        const uint32_t p0 = f32x2_to_e4m3fnx2(xv.x * f, xv.y * f);
        const uint32_t p1 = f32x2_to_e4m3fnx2(xv.z * f, xv.w * f);
        ((uint32_t*) dst8)[lane] = p0 | (p1 << 16);
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float nrm = sqrtf(wsum(a));
    const float f = fminf(1.0f, th / (nrm + 1e-6f)) * k_scale;
    for (int d = lane; d < D; d += 32) dst8[d] = f32_to_e4m3fn(src[d] * f);
}

__device__ __forceinline__ void rms_fp8_warp(const float* src, uint8_t* dst8, int D, float eps, float v_scale) {
    const int lane = threadIdx.x & 31;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (((uintptr_t) dst8 & 3) == 0)) {
        const float4 xv = ((const float4*) src)[lane];
        const float rs = rsqrtf(wsum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w) / (float) D + eps) *
                         v_scale;
        const uint32_t p0 = f32x2_to_e4m3fnx2(xv.x * rs, xv.y * rs);
        const uint32_t p1 = f32x2_to_e4m3fnx2(xv.z * rs, xv.w * rs);
        ((uint32_t*) dst8)[lane] = p0 | (p1 << 16);
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float rs = rsqrtf(wsum(a) / (float) D + eps) * v_scale;
    for (int d = lane; d < D; d += 32) dst8[d] = f32_to_e4m3fn(src[d] * rs);
}

__device__ __forceinline__ void store_k_warp(const float* kp, uint16_t* kdst, int D, float qk_clip, float) {
    l2clip_warp(kp, nullptr, nullptr, kdst, D, qk_clip, 1.0f);
}
__device__ __forceinline__ void store_k_warp(const float* kp, uint8_t* kdst, int D, float qk_clip, float k_scale) {
    l2clip_fp8_warp(kp, kdst, D, qk_clip, k_scale);
}
__device__ __forceinline__ void store_v_warp(const float* vp, uint16_t* vdst, int D, float eps, float) {
    rms_f16_warp(vp, vdst, D, eps);
}
__device__ __forceinline__ void store_v_warp(const float* vp, uint8_t* vdst, int D, float eps, float v_scale) {
    rms_fp8_warp(vp, vdst, D, eps, v_scale);
}

// 1 warp (32 threads) per group across all q, k, v, r groups of T tokens
template <class KV>
__global__ void prep_qkv_r_impl_kernel(float* q, const float* q_clip, const float* k_in, KV* kc, const float* v_in,
                                       KV* vc, float* r, const float* r_clip, int T, int H, int KVh, int D, int d_rel,
                                       float qk_clip, float rel_clip, float eps, const int* pos, float alpha,
                                       float n_floor, int pos0, int kv_cap, float k_scale, float v_scale,
                                       const int* pos0_d, uint16_t* q16, const uint16_t* sink_w, const float* sink_x,
                                       int sink_k, float* sink_y) {
    if (pos0_d) pos0 = *pos0_d;   // (a token of a replayed graph: its position read here)
    const int gid = blockIdx.x;
    const int per_tok = 2 * H + 2 * KVh;
    if (gid >= T * per_tok) {
        // (one token's sinks, sink_y [H] = sink_w [H][sink_k] . sink_x - mv_f16_kernel's sums, a launch less)
        const int h = gid - T * per_tok, lane = threadIdx.x;
        if (!sink_w || h >= H) return;
        const uint16_t* w = sink_w + (size_t) h * sink_k;
        float a = 0.0f;
        for (int i = lane; i < sink_k; i += 32) a += h2f(w[i]) * sink_x[i];
        a = wsum(a);
        if (lane == 0) sink_y[h] = a;
        return;
    }
    const int t = gid / per_tok, rem = gid % per_tok;
    const float tau = pos ? (1.0f + alpha * logf(fmaxf(1.0f, ((float) pos[t] + 1.0f) / n_floor))) : 1.0f;
    if (rem < H) {
        const int h = rem;
        float* qp = q + ((size_t) t * H + h) * D;
        l2clip_warp(qp, q_clip ? q_clip + (size_t) h * D : nullptr, qp, q16 ? q16 + ((size_t) t * H + h) * D : nullptr, D,
                    qk_clip, tau);
    } else if (rem < H + KVh) {
        const int kv = rem - H;
        const int slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t);
        const float* kp = k_in + ((size_t) t * KVh + kv) * D;
        KV* kdst = kc + ((size_t) slot * KVh + kv) * D;
        store_k_warp(kp, kdst, D, qk_clip, k_scale);
    } else if (rem < H + 2 * KVh) {
        const int kv = rem - H - KVh;
        const int slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t);
        const float* vp = v_in + ((size_t) t * KVh + kv) * D;
        KV* vdst = vc + ((size_t) slot * KVh + kv) * D;
        store_v_warp(vp, vdst, D, eps, v_scale);
    } else {
        const int h = rem - H - 2 * KVh;
        float* rp = r + ((size_t) t * H + h) * d_rel;
        l2clip_warp(rp, r_clip ? r_clip + (size_t) h * d_rel : nullptr, rp, nullptr, d_rel, rel_clip, tau);
    }
}

__global__ void l2clip_kernel(const float* x, const float* scale, int G, int D, float th, const int* pos, float alpha,
                              float n_floor, float* y) {
    __shared__ float sh[128];
    const int row = blockIdx.x / G, g = blockIdx.x % G;
    const float* xr = x + (size_t) blockIdx.x * D;
    float* yr = y + (size_t) blockIdx.x * D;
    const float* sc = scale ? scale + (size_t) g * D : nullptr;
    float a = 0.0f;
    for (int d = threadIdx.x; d < D; d += 128) {
        const float v = xr[d] * (sc ? sc[d] : 1.0f);
        a += v * v;
    }
    const float nrm = sqrtf(bsum(a, sh));
    float f = fminf(1.0f, th / (nrm + 1e-6f));
    if (pos) f *= 1.0f + alpha * logf(fmaxf(1.0f, ((float) pos[row] + 1.0f) / n_floor));
    for (int d = threadIdx.x; d < D; d += 128) yr[d] = xr[d] * (sc ? sc[d] : 1.0f) * f;
}

constexpr int kConvFusedMaxT = 16;
constexpr int kConvTileT = 16;

// Single-launch channel-mapped conv4 for small T (T <= kConvFusedMaxT): y must not alias x.
__global__ void conv4_fused_kernel(const float* x, float* hist, const float* w, const float* dyn, int heads, int T,
                                   int C, const float* residual, float* y) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c];
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float* dc_base = nullptr;
    int d_stride = 0;
    if (dyn) {
        const int head = c / (C / heads);
        dc_base = dyn + (size_t) head * 4;
        d_stride = heads * 4;
    }
    for (int t = 0; t < T; ++t) {
        const size_t tc = (size_t) t * C + c;
        const float x_new = x[tc];
        float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
        if (dc_base) {
            const float* dc = dc_base + (size_t) t * d_stride;
            a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        }
        if (residual) a += residual[tc];
        y[tc] = a;
        h0 = h1;
        h1 = h2;
        h2 = x_new;
    }
    hist[c] = h0;
    hist[(size_t) C + c] = h1;
    hist[(size_t) 2 * C + c] = h2;
}

// Channel-mapped tiled conv4 for large T (prefill): reads w and head once per tile, no hist writes.
__global__ void conv4_tiled_kernel(const float* x, const float* hist, const float* w, const float* dyn, int heads,
                                   int T, int C, const float* residual, float* y) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int t0 = blockIdx.y * kConvTileT;
    const int t1 = min(T, t0 + kConvTileT);
    auto xp_at = [&](int r) -> float { return r < 3 ? hist[(size_t) r * C + c] : x[(size_t) (r - 3) * C + c]; };
    float h0 = xp_at(t0), h1 = xp_at(t0 + 1), h2 = xp_at(t0 + 2);
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float* dc_base = nullptr;
    int d_stride = 0;
    if (dyn) {
        const int head = c / (C / heads);
        dc_base = dyn + (size_t) head * 4;
        d_stride = heads * 4;
    }
    for (int t = t0; t < t1; ++t) {
        const size_t tc = (size_t) t * C + c;
        const float x_new = x[tc];
        float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
        if (dc_base) {
            const float* dc = dc_base + (size_t) t * d_stride;
            a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        }
        if (residual) a += residual[tc];
        y[tc] = a;
        h0 = h1;
        h1 = h2;
        h2 = x_new;
    }
}

__global__ void conv4_hist_kernel(const float* x, float* hist, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    for (int j = 0; j < 3; ++j) {   // (row T + j of xp; in place: it reads rows past the ones written)
        const int r = T + j;
        hist[(size_t) j * C + c] = r < 3 ? hist[(size_t) r * C + c] : x[(size_t) (r - 3) * C + c];
    }
}

// True batched conv: grid.z is the row.  hist_tab[row] is that row's own [3][C] history (oldest first), x / y /
// residual are [n_rows][T][C], and dyn (when given) is shared [T][heads * 4].  The per-thread loop is copied from
// conv4_fused_kernel / conv4_tiled_kernel, so each row is bit-identical to one scalar conv4_res call.
__global__ void conv4_fused_rows_kernel(const float* x, float* const* hist_tab, const float* w, const float* dyn,
                                        int heads, int T, int C, const float* residual, float* y) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int row = blockIdx.z;
    const size_t row_off = (size_t) row * T * C;
    const float* xr = x + row_off;
    float* yr = y + row_off;
    const float* resr = residual ? residual + row_off : nullptr;
    float* hist = hist_tab[row];
    float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c];
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float* dc_base = nullptr;
    int d_stride = 0;
    if (dyn) {
        const int head = c / (C / heads);
        dc_base = dyn + (size_t) head * 4;
        d_stride = heads * 4;
    }
    for (int t = 0; t < T; ++t) {
        const size_t tc = (size_t) t * C + c;
        const float x_new = xr[tc];
        float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
        if (dc_base) {
            const float* dc = dc_base + (size_t) t * d_stride;
            a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        }
        if (resr) a += resr[tc];
        yr[tc] = a;
        h0 = h1;
        h1 = h2;
        h2 = x_new;
    }
    hist[c] = h0;
    hist[(size_t) C + c] = h1;
    hist[(size_t) 2 * C + c] = h2;
}

__global__ void conv4_tiled_rows_kernel(const float* x, float* const* hist_tab, const float* w, const float* dyn,
                                        int heads, int T, int C, const float* residual, float* y) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int row = blockIdx.z;
    const size_t row_off = (size_t) row * T * C;
    const float* xr = x + row_off;
    float* yr = y + row_off;
    const float* resr = residual ? residual + row_off : nullptr;
    float* hist = hist_tab[row];
    const int t0 = blockIdx.y * kConvTileT;
    const int t1 = min(T, t0 + kConvTileT);
    auto xp_at = [&](int r) -> float { return r < 3 ? hist[(size_t) r * C + c] : xr[(size_t) (r - 3) * C + c]; };
    float h0 = xp_at(t0), h1 = xp_at(t0 + 1), h2 = xp_at(t0 + 2);
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    const float* dc_base = nullptr;
    int d_stride = 0;
    if (dyn) {
        const int head = c / (C / heads);
        dc_base = dyn + (size_t) head * 4;
        d_stride = heads * 4;
    }
    for (int t = t0; t < t1; ++t) {
        const size_t tc = (size_t) t * C + c;
        const float x_new = xr[tc];
        float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
        if (dc_base) {
            const float* dc = dc_base + (size_t) t * d_stride;
            a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
        }
        if (resr) a += resr[tc];
        yr[tc] = a;
        h0 = h1;
        h1 = h2;
        h2 = x_new;
    }
}

__global__ void conv4_hist_rows_kernel(const float* x, float* const* hist_tab, int T, int C) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    const int row = blockIdx.z;
    const float* xr = x + (size_t) row * T * C;
    float* hist = hist_tab[row];
    for (int j = 0; j < 3; ++j) {   // (row T + j of xp; in place: it reads rows past the ones written)
        const int r = T + j;
        hist[(size_t) j * C + c] = r < 3 ? hist[(size_t) r * C + c] : xr[(size_t) (r - 3) * C + c];
    }
}

template <int NUM_WARPS>
__device__ __forceinline__ void attn_res_warp0_softmax(float sh_red[][kResMax * 4], float sc[3][kResMax], int n, int nq,
                                                       int C, float res_eps, int lane) {
    float ss = 0.0f, d0 = 0.0f, d1 = 0.0f, d2 = 0.0f;
    if (lane < n) {
#pragma unroll
        for (int w = 0; w < NUM_WARPS; ++w) {
            ss += sh_red[w][lane * 4 + 0];
            d0 += sh_red[w][lane * 4 + 1];
            d1 += sh_red[w][lane * 4 + 2];
            d2 += sh_red[w][lane * 4 + 3];
        }
    }
    const float r = (lane < n) ? rsqrtf(ss / (float) C + res_eps) : 0.0f;
    const float inv_sqrt_c = rsqrtf((float) C);
    const float raw[3] = {d0 * r * inv_sqrt_c, d1 * r * inv_sqrt_c, d2 * r * inv_sqrt_c};
#pragma unroll
    for (int qi = 0; qi < 3; ++qi) {
        if (qi >= nq) break;
        const float s_i = (lane < n) ? raw[qi] : -INFINITY;
        const float m = wmax(s_i);
        const float e_i = (lane < n) ? expf(s_i - m) : 0.0f;
        const float sum = wsum(e_i);
        if (lane < n) sc[qi][lane] = (e_i / sum) * r;
    }
}

constexpr int kResGroup = 4;
template <bool FUSED_RMS, int BLOCK_THREADS>
__global__ void attn_res_vec4_kernel(ResStates st, int n, const float* q, int nq, int T, int C, float res_eps,
                                     float rms_eps, float* out, RmsOuts outs, size_t job_step) {
    q += blockIdx.y * job_step, out += blockIdx.y * job_step;   // (attn_res_jobs: a job a block of the launch)
    constexpr int NUM_WARPS = BLOCK_THREADS / 32;
    __shared__ float sh_red[NUM_WARPS][kResMax * 4];
    __shared__ float sc[3][kResMax];
    __shared__ float rs_rms[3];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int c4 = C >> 2, d4_0 = tid, d4_1 = tid + BLOCK_THREADS;
    float4 q0[3] = {}, q1[3] = {};
    if (d4_0 < c4) {
        q0[0] = __ldg((const float4*) q + d4_0);
        if (nq > 1) q0[1] = __ldg((const float4*) (q + C) + d4_0);
        if (nq > 2) q0[2] = __ldg((const float4*) (q + 2 * C) + d4_0);
    }
    if (d4_1 < c4) {
        q1[0] = __ldg((const float4*) q + d4_1);
        if (nq > 1) q1[1] = __ldg((const float4*) (q + C) + d4_1);
        if (nq > 2) q1[2] = __ldg((const float4*) (q + 2 * C) + d4_1);
    }
    // (kResGroup states' values asked for at once, then used in the states' order: a thread waits for a load before
    //  it issues the next one's, and a token's launch is those waits - the sums are the one-by-one loop's, bit for bit)
    const bool in0 = d4_0 < c4, in1 = d4_1 < c4;
    const float4 zero4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int i0 = 0; i0 < n; i0 += kResGroup) {
        float4 xa[kResGroup], xb[kResGroup];
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const float4* x4 = (const float4*) (st.st[min(i0 + j, n - 1)] + (size_t) t * C);
            xa[j] = in0 ? __ldg(x4 + d4_0) : zero4;
            xb[j] = in1 ? __ldg(x4 + d4_1) : zero4;
        }
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const int i = i0 + j;
            if (i >= n) break;
            float sum_sq = 0.0f, qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f;
            if (in0) {
                const float4 xv0 = xa[j];
                sum_sq += xv0.x * xv0.x + xv0.y * xv0.y + xv0.z * xv0.z + xv0.w * xv0.w;
                qd0 += xv0.x * q0[0].x + xv0.y * q0[0].y + xv0.z * q0[0].z + xv0.w * q0[0].w;
                if (nq > 1) qd1 += xv0.x * q0[1].x + xv0.y * q0[1].y + xv0.z * q0[1].z + xv0.w * q0[1].w;
                if (nq > 2) qd2 += xv0.x * q0[2].x + xv0.y * q0[2].y + xv0.z * q0[2].z + xv0.w * q0[2].w;
            }
            if (in1) {
                const float4 xv1 = xb[j];
                sum_sq += xv1.x * xv1.x + xv1.y * xv1.y + xv1.z * xv1.z + xv1.w * xv1.w;
                qd0 += xv1.x * q1[0].x + xv1.y * q1[0].y + xv1.z * q1[0].z + xv1.w * q1[0].w;
                if (nq > 1) qd1 += xv1.x * q1[1].x + xv1.y * q1[1].y + xv1.z * q1[1].z + xv1.w * q1[1].w;
                if (nq > 2) qd2 += xv1.x * q1[2].x + xv1.y * q1[2].y + xv1.z * q1[2].z + xv1.w * q1[2].w;
            }
            sum_sq = wsum(sum_sq);
            qd0 = wsum(qd0);
            if (nq > 1) qd1 = wsum(qd1);
            if (nq > 2) qd2 = wsum(qd2);
            if (lane == 0) {
                sh_red[warp][i * 4 + 0] = sum_sq;
                sh_red[warp][i * 4 + 1] = qd0;
                sh_red[warp][i * 4 + 2] = qd1;
                sh_red[warp][i * 4 + 3] = qd2;
            }
        }
    }
    __syncthreads();
    if (warp == 0) attn_res_warp0_softmax<NUM_WARPS>(sh_red, sc, n, nq, C, res_eps, lane);
    __syncthreads();

    float4 o0[3] = {}, o1[3] = {};
    for (int i0 = 0; i0 < n; i0 += kResGroup) {
        float4 xa[kResGroup], xb[kResGroup];
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const float4* x4 = (const float4*) (st.st[min(i0 + j, n - 1)] + (size_t) t * C);
            xa[j] = in0 ? __ldg(x4 + d4_0) : zero4;
            xb[j] = in1 ? __ldg(x4 + d4_1) : zero4;
        }
#pragma unroll
        for (int j = 0; j < kResGroup; ++j) {
            const int i = i0 + j;
            if (i >= n) break;
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) {
                if (qi >= nq) break;
                const float w_qi = sc[qi][i];
                if (in0) {
                    o0[qi].x += w_qi * xa[j].x;
                    o0[qi].y += w_qi * xa[j].y;
                    o0[qi].z += w_qi * xa[j].z;
                    o0[qi].w += w_qi * xa[j].w;
                }
                if (in1) {
                    o1[qi].x += w_qi * xb[j].x;
                    o1[qi].y += w_qi * xb[j].y;
                    o1[qi].z += w_qi * xb[j].z;
                    o1[qi].w += w_qi * xb[j].w;
                }
            }
        }
    }
    if constexpr (!FUSED_RMS) {
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            float4* y4 = (float4*) (out + ((size_t) qi * T + t) * C);
            if (d4_0 < c4) y4[d4_0] = o0[qi];
            if (d4_1 < c4) y4[d4_1] = o1[qi];
        }
    } else {
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            float sq = 0.0f;
            if (d4_0 < c4) sq += o0[qi].x * o0[qi].x + o0[qi].y * o0[qi].y + o0[qi].z * o0[qi].z + o0[qi].w * o0[qi].w;
            if (d4_1 < c4) sq += o1[qi].x * o1[qi].x + o1[qi].y * o1[qi].y + o1[qi].z * o1[qi].z + o1[qi].w * o1[qi].w;
            sq = wsum(sq);
            if (lane == 0) sh_red[warp][qi] = sq;
        }
        __syncthreads();
        if (warp == 0) {
#pragma unroll
            for (int qi = 0; qi < 3; ++qi) {
                if (qi >= nq) break;
                float s_tot = (lane < NUM_WARPS) ? sh_red[lane][qi] : 0.0f;
                s_tot = wsum(s_tot);
                if (lane == 0) rs_rms[qi] = rsqrtf(s_tot / (float) C + rms_eps);
            }
        }
        __syncthreads();
#pragma unroll
        for (int qi = 0; qi < 3; ++qi) {
            if (qi >= nq) break;
            const float rs = rs_rms[qi];
            const float4* w4 = (const float4*) outs.w[qi];
            float4* y32_4 = (float4*) (outs.y32[qi] ? outs.y32[qi] + (size_t) t * C : nullptr);
            uint2* y16_4 = (uint2*) (outs.y16[qi] ? outs.y16[qi] + (size_t) t * C : nullptr);
            if (d4_0 < c4) {
                const float4 ww = w4 ? __ldg(w4 + d4_0) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                const float r0 = o0[qi].x * rs * ww.x, r1 = o0[qi].y * rs * ww.y;
                const float r2 = o0[qi].z * rs * ww.z, r3 = o0[qi].w * rs * ww.w;
                if (y32_4) y32_4[d4_0] = make_float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_0] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
            if (d4_1 < c4) {
                const float4 ww = w4 ? __ldg(w4 + d4_1) : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                const float r0 = o1[qi].x * rs * ww.x, r1 = o1[qi].y * rs * ww.y;
                const float r2 = o1[qi].z * rs * ww.z, r3 = o1[qi].w * rs * ww.w;
                if (y32_4) y32_4[d4_1] = make_float4(r0, r1, r2, r3);
                if (y16_4) y16_4[d4_1] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
            }
        }
    }
}

template <bool FUSED_RMS>
__global__ void attn_res_scalar_kernel(ResStates st, int n, const float* q, int nq, int T, int C, float res_eps,
                                       float rms_eps, float* out, RmsOuts outs, size_t job_step) {
    q += blockIdx.y * job_step, out += blockIdx.y * job_step;
    constexpr int NUM_WARPS = kBlk / 32;
    __shared__ float sh_red[NUM_WARPS][kResMax * 4];
    __shared__ float sc[3][kResMax];
    __shared__ float rs_rms[3];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    for (int i = 0; i < n; ++i) {
        const float* x = st.st[i] + (size_t) t * C;
        float sum_sq = 0.0f, qd0 = 0.0f, qd1 = 0.0f, qd2 = 0.0f;
        for (int d = tid; d < C; d += kBlk) {
            const float xv = x[d];
            sum_sq += xv * xv;
            qd0 += xv * q[d];
            if (nq > 1) qd1 += xv * q[C + d];
            if (nq > 2) qd2 += xv * q[2 * C + d];
        }
        sum_sq = wsum(sum_sq);
        qd0 = wsum(qd0);
        if (nq > 1) qd1 = wsum(qd1);
        if (nq > 2) qd2 = wsum(qd2);
        if (lane == 0) {
            sh_red[warp][i * 4 + 0] = sum_sq;
            sh_red[warp][i * 4 + 1] = qd0;
            sh_red[warp][i * 4 + 2] = qd1;
            sh_red[warp][i * 4 + 3] = qd2;
        }
    }
    __syncthreads();
    if (warp == 0) attn_res_warp0_softmax<NUM_WARPS>(sh_red, sc, n, nq, C, res_eps, lane);
    __syncthreads();
    if constexpr (!FUSED_RMS) {
        for (int qi = 0; qi < nq; ++qi) {
            float* o = out + ((size_t) qi * T + t) * C;
            for (int d = tid; d < C; d += kBlk) {
                float a = 0.0f;
                for (int i = 0; i < n; ++i) a += sc[qi][i] * st.st[i][(size_t) t * C + d];
                o[d] = a;
            }
        }
    } else {
        for (int qi = 0; qi < nq; ++qi) {
            float sq = 0.0f;
            for (int d = tid; d < C; d += kBlk) {
                float a = 0.0f;
                for (int i = 0; i < n; ++i) a += sc[qi][i] * st.st[i][(size_t) t * C + d];
                sq += a * a;
            }
            sq = wsum(sq);
            if (lane == 0) sh_red[warp][qi] = sq;
        }
        __syncthreads();
        if (warp == 0) {
            for (int qi = 0; qi < nq; ++qi) {
                float s_tot = (lane < NUM_WARPS) ? sh_red[lane][qi] : 0.0f;
                s_tot = wsum(s_tot);
                if (lane == 0) rs_rms[qi] = rsqrtf(s_tot / (float) C + rms_eps);
            }
        }
        __syncthreads();
        for (int qi = 0; qi < nq; ++qi) {
            const float rs = rs_rms[qi];
            const float* wq = outs.w[qi];
            uint16_t* y16 = outs.y16[qi] ? outs.y16[qi] + (size_t) t * C : nullptr;
            float* y32 = outs.y32[qi] ? outs.y32[qi] + (size_t) t * C : nullptr;
            for (int d = tid; d < C; d += kBlk) {
                float a = 0.0f;
                for (int i = 0; i < n; ++i) a += sc[qi][i] * st.st[i][(size_t) t * C + d];
                const float r = a * rs * (wq ? wq[d] : 1.0f);
                if (y32) y32[d] = r;
                if (y16) y16[d] = f2h(r);
            }
        }
    }
}

__global__ void attn_probs_kernel(float* lg, int nk, int qpos0, int kpos0, float qk_scale, const float* r, int r_ld,
                                  int d_rel, const float* proj, int extent, int window, const float* sink,
                                  int sink_ld) {
    __shared__ float sh[kBlk];
    const int i = blockIdx.x;
    float* row = lg + (size_t) i * nk;
    const float* ri = r + (size_t) i * r_ld;
    const int qp = qpos0 + i;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < nk; j += kBlk) {
        const int dist = qp - (kpos0 + j);
        float l = -INFINITY;
        if (dist >= 0 && dist < window) {
            l = row[j] * qk_scale;
            if (dist < extent) {
                const float* p = proj + (size_t) dist * d_rel;
                float b = 0.0f;
                for (int e = 0; e < d_rel; ++e) b += ri[e] * p[e];
                l += b;
            }
        }
        row[j] = l;
        mx = fmaxf(mx, l);
    }
    const float sk = sink[(size_t) i * sink_ld];
    const float m = fmaxf(bmax(mx, sh), sk);
    float a = 0.0f;
    for (int j = threadIdx.x; j < nk; j += kBlk) {
        const float e = expf(row[j] - m);   // (a masked key: exp(-inf) = 0)
        row[j] = e;
        a += e;
    }
    const float inv = 1.0f / (bsum(a, sh) + expf(sk - m));
    for (int j = threadIdx.x; j < nk; j += kBlk) row[j] *= inv;
}

__global__ void attn_probs_f16_kernel(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0, int kpos0,
                                      float qk_scale, const float* r, int d_rel, const float* proj, int extent,
                                      int window, const float* sink) {
    __shared__ float sh[kBlk];
    __shared__ float rs[64];
    const int i = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const float* row = lg + ((size_t) h * Tq + i) * nk;
    uint16_t* row_out = out16 + ((size_t) h * Tq + i) * nk;
    const float* ri_g = r + ((size_t) i * H + h) * d_rel;
    if (d_rel <= 64) {
        for (int e = tid; e < d_rel; e += kBlk) rs[e] = ri_g[e];
        __syncthreads();
    }
    const int qp = qpos0 + i;
    const int j_lo = min(nk, max(0, qp - kpos0 - window + 1));
    const int j_hi = max(j_lo, min(nk, qp - kpos0 + 1));
    float mx = -INFINITY, sum = 0.0f;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        const int dist = qp - (kpos0 + j);
        float l = __ldg(row + j) * qk_scale;
        if (dist < extent) {
            const float* p = proj + (size_t) dist * d_rel;
            float b = 0.0f;
            for (int e = 0; e < d_rel; ++e) b += (e < 64 ? rs[e] : ri_g[e]) * __ldg(p + e);
            l += b;
        }
        if (l > mx) {
            sum = (mx == -INFINITY ? 0.0f : sum * expf(mx - l)) + 1.0f;
            mx = l;
        } else {
            sum += expf(l - mx);
        }
    }
    const float sk = sink[(size_t) i * H + h];
    const float m = fmaxf(block_max<kBlk / 32>(mx, sh), sk);
    const float s_scaled = (mx == -INFINITY) ? 0.0f : sum * expf(mx - m);
    const float inv = 1.0f / (block_sum<kBlk / 32>(s_scaled, sh) + expf(sk - m));
    for (int j = tid; j < j_lo; j += kBlk) row_out[j] = 0;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        const int dist = qp - (kpos0 + j);
        float l = __ldg(row + j) * qk_scale;
        if (dist < extent) {
            const float* p = proj + (size_t) dist * d_rel;
            float b = 0.0f;
            for (int e = 0; e < d_rel; ++e) b += (e < 64 ? rs[e] : ri_g[e]) * __ldg(p + e);
            l += b;
        }
        row_out[j] = f2h(expf(l - m) * inv);
    }
    for (int j = j_hi + tid; j < nk; j += kBlk) row_out[j] = 0;
}

// attn_probs_f16 with the relative bias already in a table: bias [Tq * H rows][extent] (row i * H + h, column the
// distance), so an element costs a load instead of a d_rel-long dot product (done twice) - the prefill's hot kernel.
__global__ void attn_probs_bias_f16_kernel(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0,
                                           int kpos0, float qk_scale, const float* bias, int extent, int window,
                                           const float* sink) {
    __shared__ float sh[kBlk];
    const int i = blockIdx.x, h = blockIdx.y, tid = threadIdx.x;
    const float* row = lg + ((size_t) h * Tq + i) * nk;
    uint16_t* row_out = out16 + ((size_t) h * Tq + i) * nk;
    const float* brow = bias + ((size_t) i * H + h) * extent;
    const int qp = qpos0 + i;
    const int j_lo = min(nk, max(0, qp - kpos0 - window + 1));
    const int j_hi = max(j_lo, min(nk, qp - kpos0 + 1));
    const int j_b = max(j_lo, qp - kpos0 - extent + 1);   // from here on the distance is < extent
    float mx = -INFINITY, sum = 0.0f;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        float l = __ldg(row + j) * qk_scale;
        if (j >= j_b) l += __ldg(brow + (qp - kpos0 - j));
        if (l > mx) {
            sum = (mx == -INFINITY ? 0.0f : sum * __expf(mx - l)) + 1.0f;
            mx = l;
        } else {
            sum += __expf(l - mx);
        }
    }
    const float sk = sink[(size_t) i * H + h];
    const float m = fmaxf(block_max<kBlk / 32>(mx, sh), sk);
    const float s_scaled = (mx == -INFINITY) ? 0.0f : sum * __expf(mx - m);
    const float inv = 1.0f / (block_sum<kBlk / 32>(s_scaled, sh) + __expf(sk - m));
    for (int j = tid; j < j_lo; j += kBlk) row_out[j] = 0;
    for (int j = j_lo + tid; j < j_hi; j += kBlk) {
        float l = __ldg(row + j) * qk_scale;
        if (j >= j_b) l += __ldg(brow + (qp - kpos0 - j));
        row_out[j] = f2h(__expf(l - m) * inv);
    }
    for (int j = j_hi + tid; j < nk; j += kBlk) row_out[j] = 0;
}

// Prefill attention in one pass (FlashAttention-2 style, tensor cores via mma.sync m16n8k16, FP16 in / F32
// accumulate): the logits never leave registers.  A block: 64 queries (4 warps x 16) of one head against one split of
// its keys, in tiles of 64; the online softmax in the log2 domain.  Each split writes its unnormalised output and its
// (max, sum); attn_flash_combine merges the splits and the sink.  The relative bias from the table [Tq * H][extent].
constexpr int kFaBN = 64, kFaD = 128, kFaLd = kFaD + 8;   // (row pad: conflict-free fragments/ldmatrix)
constexpr float kLog2e = 1.4426950408889634f;

// NVIDIA tensor-core PTX (mma.sync / ldmatrix / cp.async). On HIP these kernels
// are never launched (IK2_FLASH=0 / IK2_ATTN_MMA=0 / IK2_EXPERT_MMA=0 select the
// scalar kernels); stub the helpers so the kernels still compile and the symbols
// exist for the host occupancy probe. Phase B replaces them with gfx1100 WMMA.
#if defined(__HIPCC__)
__device__ __forceinline__ void mma16816(float*, const uint32_t*, uint32_t, uint32_t) { __builtin_trap(); }
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void*) {
    r0 = r1 = r2 = r3 = 0; __builtin_trap();
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void*) {
    r0 = r1 = r2 = r3 = 0; __builtin_trap();
}
__device__ __forceinline__ void cp_async16(void*, const void*, bool) { __builtin_trap(); }
__device__ __forceinline__ void cp_async4(void*, const void*, bool) { __builtin_trap(); }
__device__ __forceinline__ void cp_async_commit() { __builtin_trap(); }
template <int N>
__device__ __forceinline__ void cp_async_wait() { __builtin_trap(); }
#else
__device__ __forceinline__ void mma16816(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, "
                 "{%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void ldsm_x4_t(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void ldsm_x4(uint32_t& r0, uint32_t& r1, uint32_t& r2, uint32_t& r3, const void* p) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
                 : "r"(a));
}
__device__ __forceinline__ void cp_async16(void* dst, const void* src, bool valid) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(dst);
    const int n = valid ? 16 : 0;   // (0: the 16 bytes zero-filled)
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(a), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async4(void* dst, const void* src, bool valid) {
    const unsigned a = (unsigned) __cvta_generic_to_shared(dst);
    const int n = valid ? 4 : 0;
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(a), "l"(src), "r"(n));
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
#endif  // __HIPCC__

// one 64-key tile of K and V (rows kt .. kt+63; rows at or past k_end zero-filled) into smem, NT threads
template <int NT>
__device__ __forceinline__ void fa_load_tile(uint16_t* Ks, uint16_t* Vs, const uint16_t* kc, const uint16_t* vc,
                                             int kv_cap, int KD, int g, int kt, int k_end, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 8) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 4, col = (c & 15) * 8;
        const int kp = kt + row;
        const bool ok = kp < k_end;
        const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
        const size_t off = (size_t) pr * KD + (size_t) g * kFaD + col;
        cp_async16(Ks + row * kFaLd + col, kc + off, ok);
        cp_async16(Vs + row * kFaLd + col, vc + off, ok);
    }
}

// ... from an FP8 (E4M3) cache: the tile's bytes are copied as they are (cp.async, a tile ahead, as the FP16 tile)
// and turned into FP16 in shared memory once they have arrived.  (Turning them as they were loaded had every load
// wait for its bytes - a branch between it and the next - and the kernel ran at a sixth of the memory's speed.)
constexpr int kFaRaw = 2 * kFaBN * kFaD;   // the bytes of a tile: K's [kFaBN][kFaD], then V's
template <int NT>
__device__ __forceinline__ void fa_load_tile_raw(uint8_t* Rs, const uint8_t* kc, const uint8_t* vc, int kv_cap, int KD,
                                                 int g, int kt, int k_end, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 16) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 3, col = (c & 7) * 16;
        const int kp = kt + row;
        const bool ok = kp < k_end;
        const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
        const size_t off = (size_t) pr * KD + (size_t) g * kFaD + col;
        cp_async16(Rs + row * kFaD + col, kc + off, ok);
        cp_async16(Rs + kFaBN * kFaD + row * kFaD + col, vc + off, ok);
    }
}
// four E4M3 values as FP16: lo the first two (two halves a register), hi the other two.  A value is sign, 4 bits
// of exponent, 3 of mantissa, and as FP16 its magnitude is (mag << 7) + 0x2000: with a value the high byte of its
// half (x = sign 2^15 + mag 2^8) the FP16 is x - mag 2^7 + 0x2000.  A subnormal (exponent 0: under 1/64) is that
// less an amount by its mantissa, looked up for the four at once (a byte permutation) - no branch, the same cost
// whatever the values.  (0x7F, the NaN, comes out finite: f32_to_e4m3fn saturates, a cache has none.)
__device__ __forceinline__ void e4m3x4_h4(uint32_t w, uint32_t& lo, uint32_t& hi) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    lo = e4m3fnx2_to_h2_u32((uint16_t) w), hi = e4m3fnx2_to_h2_u32((uint16_t) (w >> 16));
#else
    const uint32_t t = w & 0x78787878u;
    uint32_t z = (t - 0x01010101u) & ~t & 0x80808080u;   // 0x80 a byte whose exponent is 0 ...
    z -= z >> 7;                                         // ... 0x7F there
    const uint32_t v = w & 0x07070707u, u = v | (v >> 4);
    // (the normal formula - the subnormal's FP16) / 0x80 by the mantissa: 64, 17, 10, 7, 4, 3, 2, 1
    const uint32_t dd = __byte_perm(0x070A1140u, 0x01020304u, __byte_perm(u, 0u, 0x4420u)) & z;
    const uint32_t x0 = __byte_perm(w, 0u, 0x1404u), x1 = __byte_perm(w, 0u, 0x3424u);
    const uint32_t y0 = __byte_perm(dd, 0u, 0x1404u), y1 = __byte_perm(dd, 0u, 0x3424u);
    lo = x0 - (((x0 & 0x7F007F00u) + y0) >> 1) + 0x20002000u;
    hi = x1 - (((x1 & 0x7F007F00u) + y1) >> 1) + 0x20002000u;
#endif
}
__device__ __forceinline__ uint4 e4m3x8_h8(uint2 p) {
    uint4 h;
    e4m3x4_h4(p.x, h.x, h.y);
    e4m3x4_h4(p.y, h.z, h.w);
    return h;
}
// the FP16 tile (Ks, Vs: [kFaBN][kFaLd]) of the bytes Rs, NT threads
template <int NT>
__device__ __forceinline__ void fa_tile_from_raw(uint16_t* Ks, uint16_t* Vs, const uint8_t* Rs, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 8) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 4, col = (c & 15) * 8;
        const uint2 k8 = *(const uint2*) (Rs + row * kFaD + col);
        const uint2 v8 = *(const uint2*) (Rs + kFaBN * kFaD + row * kFaD + col);
        *(uint4*) (Ks + row * kFaLd + col) = e4m3x8_h8(k8);
        *(uint4*) (Vs + row * kFaLd + col) = e4m3x8_h8(v8);
    }
}

// The relative bias: a tile within the extent of a warp's queries has the warp copy its 16 rows x 64 distances of the
// table into shared memory (cp.async, coalesced, issued before the tile's Q K^T) - loading it element by element at
// the softmax cost ~3x the whole kernel in the sliding layers (every tile there is within the extent).
constexpr int kFaBld = 72;   // (a bias row in shared memory; 72: 2-way bank conflicts at most for the float2 reads)
constexpr int kFaNW = 4;     // warps a block (8: the same speed in a test, and past sm_89/sm_120's shared memory)
constexpr int kFaSmem = 4 * kFaBN * kFaLd * (int) sizeof(uint16_t) + kFaNW * 16 * kFaBld * (int) sizeof(float);
// Hopper (sm_90a) warpgroup MMA: the four warps of a 128-thread block as one warpgroup, 64 rows.
// D[64 x 64] (+)= A[64 x 16] (registers, the mma.m16n8k16 A fragments of each warp's 16 rows) x B (smem
// descriptor; tnsp: B stored N-major), F16 in, F32 accumulate; d[j * 4 + i] as mma's C fragment of n8 block j
template <int TNSP>
__device__ __forceinline__ void wgmma_m64n64k16_rs(float* d, const uint32_t* a, uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %37, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31}, "
        "{%32, %33, %34, %35}, %36, p, 1, 1, %38;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d), "n"(TNSP));
#else
    __trap();
#endif
}

// D[64 x 128] (+)= A[64 x 16] (registers, the mma.m16n8k16 A fragments of each warp's 16 rows) x B (smem
// descriptor; tnsp: B stored N-major), F16 in, F32 accumulate; d[j * 4 + i] as mma's C fragment of n8 block j
template <int TNSP>
__device__ __forceinline__ void wgmma_m64n128k16_rs(float* d, const uint32_t* a, uint64_t desc, int scale_d) {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile(
        "{\n.reg .pred p;\nsetp.ne.b32 p, %69, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n128k16.f32.f16.f16 "
        "{%0, %1, %2, %3, %4, %5, %6, %7, %8, %9, %10, %11, %12, %13, %14, %15, %16, %17, %18, %19, %20, %21, %22, %23, %24, %25, %26, %27, %28, %29, %30, %31, %32, %33, %34, %35, %36, %37, %38, %39, %40, %41, %42, %43, %44, %45, %46, %47, %48, %49, %50, %51, %52, %53, %54, %55, %56, %57, %58, %59, %60, %61, %62, %63}, "
        "{%64, %65, %66, %67}, %68, p, 1, 1, %70;\n}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]), "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]), "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]), "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]), "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31]), "+f"(d[32]), "+f"(d[33]), "+f"(d[34]), "+f"(d[35]), "+f"(d[36]), "+f"(d[37]), "+f"(d[38]), "+f"(d[39]), "+f"(d[40]), "+f"(d[41]), "+f"(d[42]), "+f"(d[43]), "+f"(d[44]), "+f"(d[45]), "+f"(d[46]), "+f"(d[47]), "+f"(d[48]), "+f"(d[49]), "+f"(d[50]), "+f"(d[51]), "+f"(d[52]), "+f"(d[53]), "+f"(d[54]), "+f"(d[55]), "+f"(d[56]), "+f"(d[57]), "+f"(d[58]), "+f"(d[59]), "+f"(d[60]), "+f"(d[61]), "+f"(d[62]), "+f"(d[63])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "l"(desc), "r"(scale_d), "n"(TNSP));
#else
    __trap();
#endif
}

__device__ __forceinline__ void wg_fence() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.fence.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_commit() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.commit_group.sync.aligned;\n" ::: "memory");
#endif
}
__device__ __forceinline__ void wg_wait0() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("wgmma.wait_group.sync.aligned 0;\n" ::: "memory");
#endif
}
// (keeps the compiler from moving an accumulator's use across the asynchronous MMA's issue or wait)
#if defined(__HIPCC__)
__device__ __forceinline__ void wg_fence_operand(float&) {}   // no async MMA on HIP (wgmma kernels trap)
#else
__device__ __forceinline__ void wg_fence_operand(float& r) { asm volatile("" : "+f"(r)::"memory"); }
#endif
__device__ __forceinline__ void fence_proxy_async() {
#if defined(__CUDA_ARCH_FEAT_SM90_ALL)
    asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
#endif
}
// shared-memory matrix descriptor with the 128-byte swizzle: start, leading / stride byte offsets
__device__ __forceinline__ uint64_t wg_desc(const void* p, uint32_t lbo, uint32_t sbo) {
#if defined(__HIPCC__)
    (void) p; (void) lbo; (void) sbo;
    return 0;   // wgmma descriptors are NVIDIA-only; the wgmma kernels trap on HIP
#else
    const uint32_t a = (uint32_t) __cvta_generic_to_shared(p);
    return (uint64_t) ((a & 0x3FFFF) >> 4) | ((uint64_t) ((lbo & 0x3FFFF) >> 4) << 16) |
           ((uint64_t) ((sbo & 0x3FFFF) >> 4) << 32) | ((uint64_t) 1 << 62);
#endif
}
// A 64 x 128 tile (rows: keys, columns: dims), 1024-byte aligned, in the 128-byte-swizzle layout: two atoms of
// 64 rows x 64 columns (128 bytes a row), the 16-byte chunks of row r XOR-permuted by r % 8. K is then B K-major
// (SBO 1024 B between 8-row groups; a k-step 32 bytes further in the atom), V B MN-major (transposed: LBO 8192 B
// between the atoms along the dims, SBO 1024 B between 8-key groups).
constexpr int kWgTile = kFaBN * kFaD;   // halves a tile
__device__ __forceinline__ int wg_tile_off(int r, int c) {
    return (c >> 6) * (kFaBN * 64) + r * 64 + ((((c >> 3) & 7) ^ (r & 7)) << 3);
}
template <int NT>
__device__ __forceinline__ void fa_load_tile_wg(uint16_t* Ks, uint16_t* Vs, const uint16_t* kc, const uint16_t* vc,
                                                int kv_cap, int KD, int g, int kt, int k_end, int tid) {
#pragma unroll
    for (int it = 0; it < (kFaBN * kFaD / 8) / NT; ++it) {
        const int c = it * NT + tid, row = c >> 4, col = (c & 15) * 8;
        const int kp = kt + row;
        const bool ok = kp < k_end;
        const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
        const size_t off = (size_t) pr * KD + (size_t) g * kFaD + col;
        cp_async16(Ks + wg_tile_off(row, col), kc + off, ok);
        cp_async16(Vs + wg_tile_off(row, col), vc + off, ok);
    }
}

template <int NW, bool WG, class KVT = uint16_t>   // (KVT: the cache's values, FP16 or FP8 - an FP8 tile's bytes are
__global__ void __launch_bounds__(NW * 32, 2)        //  made FP16 in shared memory; the warpgroup path is FP16's)
attn_flash_f16_kernel(const uint16_t* __restrict__ q16, int q_ld, const KVT* __restrict__ kc,
                      const KVT* __restrict__ vc, int kv_cap, int H, int KV, int Tq, int qpos0, float qk_scale,
                      const float* __restrict__ bias, int extent, int window, int n_split, float* __restrict__ o_part,
                      float2* __restrict__ ml_part) {
    extern __shared__ __align__(16) uint16_t fa_sm[];
    constexpr bool F8 = sizeof(KVT) == 1;        // (FP8: one FP16 tile, and the bytes of two [2][kFaRaw] after it)
    uint16_t* Ks = fa_sm;                        // [2][kFaBN][kFaLd]
    uint16_t* Vs = fa_sm + (F8 ? 1 : 2) * kFaBN * kFaLd;   // [2][kFaBN][kFaLd]
    uint8_t* Rs = (uint8_t*) (fa_sm + 2 * kFaBN * kFaLd);
    constexpr int kTileLd = WG ? kWgTile : F8 ? 0 : kFaBN * kFaLd;   // (halves from one buffer to the other)
    float* Bw = (float*) (fa_sm + 4 * kFaBN * kFaLd) + (threadIdx.x >> 5) * 16 * kFaBld;   // this warp's [16][kFaBld]
    if constexpr (WG) {   // [2][kWgTile] each, 1024-byte aligned (the swizzle's atoms), inside the same 68 KB
        uint16_t* a = (uint16_t*) (((uintptr_t) fa_sm + 1023) & ~(uintptr_t) 1023);
        Ks = a, Vs = a + 2 * kWgTile;
    }
    constexpr int kFaBM = 16 * NW;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, gr = lane >> 2, tq = lane & 3;
    const int h = blockIdx.y, split = blockIdx.z, g = h / (H / KV), KD = KV * kFaD;
    const int i0 = blockIdx.x * kFaBM;                     // the block's first query (index in the launch)
    const int i_last = min(Tq, i0 + kFaBM) - 1;
    // the block's keys: [k_lo, k_hi), cut into tiles; this split's tiles [t0, t1)
    const int k_hi = qpos0 + i_last + 1;
    const int k_lo = max(0, qpos0 + i0 - window + 1);
    const int n_tiles = (k_hi - k_lo + kFaBN - 1) / kFaBN;
    const int per = (n_tiles + n_split - 1) / n_split;
    const int t0 = min(n_tiles, split * per), t1 = min(n_tiles, t0 + per);

    // this warp's 16 queries: rows r0 = gr, r1 = gr + 8
    const int wi = i0 + warp * 16;
    const int ia = wi + gr, ib = wi + gr + 8;
    const int qa = qpos0 + ia, qb = qpos0 + ib;
    uint32_t qf[kFaD / 16][4];
    {
        const uint16_t* qra = q16 + (size_t) ia * q_ld + (size_t) h * kFaD;
        const uint16_t* qrb = q16 + (size_t) ib * q_ld + (size_t) h * kFaD;
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            const int d = kk * 16 + tq * 2;
            qf[kk][0] = ia < Tq ? *(const uint32_t*) (qra + d) : 0u;
            qf[kk][1] = ib < Tq ? *(const uint32_t*) (qrb + d) : 0u;
            qf[kk][2] = ia < Tq ? *(const uint32_t*) (qra + d + 8) : 0u;
            qf[kk][3] = ib < Tq ? *(const uint32_t*) (qrb + d + 8) : 0u;
        }
    }
    float o[kFaD / 8][4];
#pragma unroll
    for (int j = 0; j < kFaD / 8; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    const float sc = qk_scale * kLog2e;
    const int lo_a = qa - window + 1, lo_b = qb - window + 1;

    if (t0 < t1) {
        if constexpr (WG) fa_load_tile_wg<NW * 32>(Ks, Vs, kc, vc, kv_cap, KD, g, k_lo + t0 * kFaBN, k_hi, tid);
        else if constexpr (F8) fa_load_tile_raw<NW * 32>(Rs, kc, vc, kv_cap, KD, g, k_lo + t0 * kFaBN, k_hi, tid);
        else fa_load_tile<NW * 32>(Ks, Vs, kc, vc, kv_cap, KD, g, k_lo + t0 * kFaBN, k_hi, tid);
        cp_async_commit();
    }
    for (int t = t0; t < t1; ++t) {
        const int buf = (t - t0) & 1;
        const int kt = k_lo + t * kFaBN;
        // does this warp's rows see any key of the tile, and is any distance within the bias's extent?
        const bool any = !(kt > qpos0 + wi + 15 || kt + kFaBN - 1 < qpos0 + wi - window + 1) && wi < Tq;
        const bool near = qpos0 + wi - (kt + kFaBN - 1) < extent;
        // cp.async groups in flight here: K/V(t), then this tile's bias, then K/V(t+1) (empty groups keep the count)
        if (any && near) {
#pragma unroll 4
            for (int rr = 0; rr < 16; ++rr) {
                const int i = wi + rr, qp = qpos0 + i;
                const float* brow = bias + ((size_t) i * H + h) * extent;
#pragma unroll
                for (int hf = 0; hf < 2; ++hf) {
                    const int c = hf * 32 + lane, dist = qp - kt - c;
                    const bool ok = i < Tq && dist >= 0 && dist < extent;
                    cp_async4(Bw + rr * kFaBld + c, ok ? brow + dist : bias, ok);
                }
            }
        }
        cp_async_commit();
        if (t + 1 < t1) {
            if constexpr (WG)
                fa_load_tile_wg<NW * 32>(Ks + (buf ^ 1) * kTileLd, Vs + (buf ^ 1) * kTileLd, kc, vc, kv_cap,
                                         KD, g, k_lo + (t + 1) * kFaBN, k_hi, tid);
            else if constexpr (F8)
                fa_load_tile_raw<NW * 32>(Rs + (buf ^ 1) * kFaRaw, kc, vc, kv_cap, KD, g, k_lo + (t + 1) * kFaBN, k_hi,
                                          tid);
            else
                fa_load_tile<NW * 32>(Ks + (buf ^ 1) * kFaBN * kFaLd, Vs + (buf ^ 1) * kFaBN * kFaLd, kc, vc, kv_cap,
                                      KD, g, k_lo + (t + 1) * kFaBN, k_hi, tid);
        }
        cp_async_commit();
        cp_async_wait<2>();
        if constexpr (WG) fence_proxy_async();   // (the copies visible to the MMA's reads of shared memory)
        __syncthreads();
        if constexpr (F8) {   // (the tile's bytes are in: FP16 of them)
            fa_tile_from_raw<NW * 32>(Ks, Vs, Rs + buf * kFaRaw, tid);
            __syncthreads();
        }
        const uint16_t* K = Ks + buf * kTileLd;
        const uint16_t* V = Vs + buf * kTileLd;
        if (any || WG) {   // (the warpgroup's MMA takes all four warps)
            float s[kFaBN / 8][4];
            if constexpr (WG) {
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j) s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j)
                    for (int e = 0; e < 4; ++e) wg_fence_operand(s[j][e]);
                wg_fence();
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk)
                    wgmma_m64n64k16_rs<0>(&s[0][0], qf[kk], wg_desc(K + (kk >> 2) * (kFaBN * 64) + (kk & 3) * 16, 16, 1024), 1);
                wg_commit();
                wg_wait0();
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j)
                    for (int e = 0; e < 4; ++e) wg_fence_operand(s[j][e]);
            } else
#pragma unroll
            for (int j = 0; j < kFaBN / 8; ++j) {
                s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
                const uint16_t* kr = K + (j * 8 + (lane & 7)) * kFaLd + (lane >> 3) * 8;   // (two k-steps an ldmatrix)
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; kk += 2) {
                    uint32_t b0, b1, b2, b3;
                    ldsm_x4(b0, b1, b2, b3, kr + kk * 16);
                    mma16816(s[j], qf[kk], b0, b1);
                    mma16816(s[j], qf[kk + 1], b2, b3);
                }
            }
            // logits (log2 units): scale, bias, mask - a tile inside every row's keys and past the bias's extent
            // needs only the scale
            cp_async_wait<1>();   // (this tile's bias)
            __syncwarp();
            const bool full = !near && kt + kFaBN - 1 <= qpos0 + wi && kt >= qpos0 + wi + 15 - window + 1 &&
                              wi + 15 < Tq;
            float mxa = -INFINITY, mxb = -INFINITY;
            if (full) {
#pragma unroll
                for (int j = 0; j < kFaBN / 8; ++j) {
                    s[j][0] *= sc, s[j][1] *= sc, s[j][2] *= sc, s[j][3] *= sc;
                    mxa = fmaxf(mxa, fmaxf(s[j][0], s[j][1]));
                    mxb = fmaxf(mxb, fmaxf(s[j][2], s[j][3]));
                }
            } else
#pragma unroll
            for (int j = 0; j < kFaBN / 8; ++j) {
#pragma unroll
                for (int e = 0; e < 2; ++e) {
                    const int c = j * 8 + tq * 2 + e, kp = kt + c;
                    const int da = qa - kp, db = qb - kp;
                    float va = s[j][e] * sc, vb = s[j][2 + e] * sc;
                    if (near) {   // (zero where out of the extent)
                        va += Bw[gr * kFaBld + c] * kLog2e;
                        vb += Bw[(gr + 8) * kFaBld + c] * kLog2e;
                    }
                    va = (da < 0 || kp < lo_a || ia >= Tq) ? -INFINITY : va;
                    vb = (db < 0 || kp < lo_b || ib >= Tq) ? -INFINITY : vb;
                    s[j][e] = va, s[j][2 + e] = vb;
                    mxa = fmaxf(mxa, va), mxb = fmaxf(mxb, vb);
                }
            }
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 1));
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, 2));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 1));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, 2));
            const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
            const float ca = mna == -INFINITY ? 1.0f : exp2f(ma - mna);   // (ma = -inf: exp2f(-inf) = 0)
            const float cb = mnb == -INFINITY ? 1.0f : exp2f(mb - mnb);
            const float ua = mna == -INFINITY ? 0.0f : mna, ub = mnb == -INFINITY ? 0.0f : mnb;
            ma = mna, mb = mnb;
            la *= ca, lb *= cb;
#pragma unroll
            for (int j = 0; j < kFaD / 8; ++j) o[j][0] *= ca, o[j][1] *= ca, o[j][2] *= cb, o[j][3] *= cb;
            // P (FP16) as the A fragments of P V, two 8-key tiles per 16-key step
            uint32_t pa[kFaBN / 16][4];
#pragma unroll
            for (int kk = 0; kk < kFaBN / 16; ++kk) {
                const float p00 = exp2f(s[2 * kk][0] - ua), p01 = exp2f(s[2 * kk][1] - ua);
                const float p02 = exp2f(s[2 * kk][2] - ub), p03 = exp2f(s[2 * kk][3] - ub);
                const float p10 = exp2f(s[2 * kk + 1][0] - ua), p11 = exp2f(s[2 * kk + 1][1] - ua);
                const float p12 = exp2f(s[2 * kk + 1][2] - ub), p13 = exp2f(s[2 * kk + 1][3] - ub);
                la += p00 + p01 + p10 + p11;
                lb += p02 + p03 + p12 + p13;
                pa[kk][0] = f2_to_u32(p00, p01);
                pa[kk][1] = f2_to_u32(p02, p03);
                pa[kk][2] = f2_to_u32(p10, p11);
                pa[kk][3] = f2_to_u32(p12, p13);
            }
            if constexpr (WG) {
#pragma unroll
                for (int j = 0; j < kFaD / 8; ++j)
                    for (int e = 0; e < 4; ++e) wg_fence_operand(o[j][e]);
                wg_fence();
#pragma unroll
                for (int kk = 0; kk < kFaBN / 16; ++kk)
                    wgmma_m64n128k16_rs<1>(&o[0][0], pa[kk], wg_desc(V + kk * 16 * 64, kFaBN * 128, 1024), 1);
                wg_commit();
                wg_wait0();
#pragma unroll
                for (int j = 0; j < kFaD / 8; ++j)
                    for (int e = 0; e < 4; ++e) wg_fence_operand(o[j][e]);
            } else {
#pragma unroll
                for (int kk = 0; kk < kFaBN / 16; ++kk) {
                    const uint16_t* vr = V + (kk * 16 + (lane & 7) + ((lane >> 3) & 1) * 8) * kFaLd + (lane >> 4) * 8;
#pragma unroll
                    for (int j = 0; j < kFaD / 8; j += 2) {
                        uint32_t b0, b1, b2, b3;
                        ldsm_x4_t(b0, b1, b2, b3, vr + j * 8);
                        mma16816(o[j], pa[kk], b0, b1);
                        mma16816(o[j + 1], pa[kk], b2, b3);
                    }
                }
            }
        }
        if constexpr (!F8) __syncthreads();   // (the buffer is refilled next iteration; FP8's FP16 tile is written
    }                                         //  past the next iteration's barrier)
    la += __shfl_xor_sync(0xffffffffu, la, 1);
    la += __shfl_xor_sync(0xffffffffu, la, 2);
    lb += __shfl_xor_sync(0xffffffffu, lb, 1);
    lb += __shfl_xor_sync(0xffffffffu, lb, 2);
    // the split's partials: o_part [split][Tq][H][D], ml_part [split][Tq][H] = (max in log2 units, sum)
    if (ia < Tq) {
        float* op = o_part + (((size_t) split * Tq + ia) * H + h) * kFaD + tq * 2;
#pragma unroll
        for (int j = 0; j < kFaD / 8; ++j) *(float2*) (op + j * 8) = make_float2(o[j][0], o[j][1]);
        if (tq == 0) ml_part[((size_t) split * Tq + ia) * H + h] = make_float2(ma, la);
    }
    if (ib < Tq) {
        float* op = o_part + (((size_t) split * Tq + ib) * H + h) * kFaD + tq * 2;
#pragma unroll
        for (int j = 0; j < kFaD / 8; ++j) *(float2*) (op + j * 8) = make_float2(o[j][2], o[j][3]);
        if (tq == 0) ml_part[((size_t) split * Tq + ib) * H + h] = make_float2(mb, lb);
    }
}

// ===========================================================================
// gfx1100 WMMA prefill attention (RDNA3, wave32). One wave = 16 queries of one head
// against all its keys in tiles of BN, FlashAttention-2 online softmax in the
// log2 domain, logits stay in registers (no W.lg materialisation). Reuses
// attn_flash_combine_kernel for the split merge + sink. The relative bias comes
// from the [Tq*H][extent] table exactly as the scalar attn_probs_bias_f16_kernel.
//
// WMMA fragment convention (proven in src/prefill/wmma_gemm.cu):
//   __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c), a/b = 16 x _Float16, c = 8 x float.
//   lane_lo = lane & 15 holds ROW lane_lo of A (16 elems along K) and COLUMN lane_lo of
//   B (16 elems along K); lanes 16..31 duplicate. C: lane holds COLUMN lane_lo, 8 elems
//   at rows m = 2*i + lane_hi (lane_hi = lane >> 4).
// QK^T: A = Q[query, dim] (M=queries, K=dims), B = K[key, dim] (N=keys, K=dims) ->
//   C[query, key]: lane holds key=lane_lo, 8 alternating queries. Softmax per query
//   reduces across the 16 lane_lo (shfl_xor 1/2/4/8).
// PV: A = P[query, key] (M=queries, K=keys), B = V[key, dim] (N=dims, K=keys) ->
//   C[query, dim]: lane holds dim=lane_lo, 8 alternating queries. V[key, dim=lane_lo]
//   is strided, so V is staged transposed in LDS (Vt[dim][key], contiguous in key).
// ===========================================================================
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
namespace {
using fw_vec16 = _Float16 __attribute__((ext_vector_type(16)));
using fw_v8f = float __attribute__((ext_vector_type(8)));
__device__ __forceinline__ fw_v8f fw_mma(fw_vec16 a, fw_vec16 b, fw_v8f c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    (void) a; (void) b; (void) c; __builtin_trap();
#endif
}
__device__ __forceinline__ fw_vec16 fw_load16(const uint16_t* p) {
    fw_vec16 v;
    __builtin_memcpy(&v, p, sizeof(v));
    return v;
}
}   // namespace

constexpr int kFwD = 128;     // head dim
constexpr int kFwLd = 136;    // K tile row stride (pad, avoids LDS bank conflicts)
constexpr int kFwNW = 4;      // waves a block (64 queries share each K/V tile load)
constexpr int kFwBM = 16 * kFwNW;
// dynamic LDS: Ks[BN][kFwLd] + Vt[kFwD][BN] shared by the block + Ps[kFwNW][16][BN] per wave.
// BN=64: 37376 B -> 1 block a CU (4 waves); BN=32: 20992 B -> 3 blocks a CU (12 waves). Bias from global (L2).
constexpr int kFwSmem64 = (64 * kFwLd + kFwD * 64 + kFwNW * 16 * 64) * (int) sizeof(uint16_t);
constexpr int kFwSmem32 = (32 * kFwLd + kFwD * 32 + kFwNW * 16 * 32) * (int) sizeof(uint16_t);

// stage 8 int codes (int8: 8 bytes; int4: 4 bytes = 8 nibbles) x one FP16 row scale -> 8 FP16 (uint4),
// the int counterpart of e4m3x8_h8 for the WMMA flash tile loader.
template <int BITS>
__device__ __forceinline__ uint4 kv_intx8_h8(const uint8_t* p, float scale) {
    uint16_t h[8];
    if constexpr (BITS == 8) {
        const uint2 c = *(const uint2*) p;
        const uint32_t lo = c.x, hi = c.y;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            h[i] = f2h((float) (int8_t) ((lo >> (8 * i)) & 0xFFu) * scale);
            h[4 + i] = f2h((float) (int8_t) ((hi >> (8 * i)) & 0xFFu) * scale);
        }
    } else {
        const uint32_t c = *(const uint32_t*) p;   // 8 nibbles
#pragma unroll
        for (int i = 0; i < 8; ++i) h[i] = f2h((float) kv_int_nib((c >> (4 * i)) & 0xFu) * scale);
    }
    return *(const uint4*) h;
}

template <int BN, int NW, class KVT, int IBITS = 0>
__global__ void __launch_bounds__(NW * 32, 1)
attn_flash_wmma_kernel(const uint16_t* __restrict__ q16, int q_ld, const KVT* __restrict__ kc,
                       const KVT* __restrict__ vc, int kv_cap, int H, int KV, int Tq, int qpos0, float qk_scale,
                       const float* __restrict__ bias, int extent, int window, int n_split, float* __restrict__ o_part,
                       float2* __restrict__ ml_part, const uint16_t* __restrict__ ks = nullptr,
                       const uint16_t* __restrict__ vs = nullptr) {
    constexpr int BM = 16 * NW;
    extern __shared__ __align__(16) uint16_t fw_sm[];
    uint16_t* Ks = fw_sm;                              // [BN][kFwLd] keys x dims (block shared)
    uint16_t* Vt = fw_sm + BN * kFwLd;                 // [kFwD][BN] dims x keys (transposed, block shared)
    const int tid = threadIdx.x, warp = tid >> 5;
    uint16_t* Ps = fw_sm + BN * kFwLd + kFwD * BN + warp * 16 * BN;   // [16][BN] this wave's P
    const int lane = tid & 31, lane_lo = lane & 15, lane_hi = lane >> 4;
    const int h = blockIdx.y, split = blockIdx.z;
    const int i0 = blockIdx.x * BM;                    // the block's first query
    const int wi = i0 + warp * 16;                     // this wave's first query
    const int g = h / (H / KV), KD = KV * kFwD;
    const int i_last = min(Tq, i0 + BM) - 1;
    const int k_hi = qpos0 + i_last + 1;
    const int k_lo = max(0, qpos0 + i0 - window + 1);
    const int n_tiles = (k_hi - k_lo + BN - 1) / BN;
    const int per = (n_tiles + n_split - 1) / n_split;
    const int t0 = min(n_tiles, split * per), t1 = min(n_tiles, t0 + per);
    const float sc = qk_scale * kLog2e;

    // Q fragments: qf[kk] = Q[query = wi + lane_lo, dims kk*16 .. +15] (M=queries, K=dims).
    fw_vec16 qf[kFwD / 16];
    {
        const int qrow = wi + lane_lo;
        const uint16_t* qp = q16 + (size_t) qrow * q_ld + (size_t) h * kFwD;
#pragma unroll
        for (int kk = 0; kk < kFwD / 16; ++kk) qf[kk] = qrow < Tq ? fw_load16(qp + kk * 16) : fw_vec16{};
    }
    // per-query online softmax state (lane holds queries 2*i + lane_hi)
    float m[8], l[8];
    fw_v8f o[kFwD / 16];
#pragma unroll
    for (int i = 0; i < 8; ++i) { m[i] = -INFINITY; l[i] = 0.0f; }
#pragma unroll
    for (int j = 0; j < kFwD / 16; ++j) o[j] = fw_v8f{0, 0, 0, 0, 0, 0, 0, 0};

    for (int t = t0; t < t1; ++t) {
        const int kt = k_lo + t * BN;
        const bool near = qpos0 + wi - (kt + BN - 1) < extent;
        // stage K tile (contiguous dims) and V tile transposed (dims x keys), shared by the block's waves
#pragma unroll 4
        for (int c = tid; c < BN * kFwD / 8; c += NW * 32) {
            const int row = c >> 4, col = (c & 15) * 8;     // key row, dim col (8 halves)
            const int kp = kt + row;
            const bool ok = kp < k_hi;
            const int pr = ok ? (kv_cap > 0 ? kp % kv_cap : kp) : 0;
            uint4 kv, vv;
            if constexpr (IBITS != 0) {         // INT8 / INT4 cache: per-(token,head) FP16 scale, dequant to FP16 tile
                const int cb = (IBITS == 8) ? kFwD : kFwD / 2;     // code bytes per head
                const size_t offi = (size_t) pr * KV * cb + (size_t) g * cb + col / (IBITS == 8 ? 1 : 2);
                const size_t sidx = (size_t) pr * KV + g;
                const float ksc = ok ? h2f(__ldg(ks + sidx)) : 0.0f;
                const float vsc = ok ? h2f(__ldg(vs + sidx)) : 0.0f;
                kv = ok ? kv_intx8_h8<IBITS>(kc + offi, ksc) : make_uint4(0, 0, 0, 0);
                vv = ok ? kv_intx8_h8<IBITS>(vc + offi, vsc) : make_uint4(0, 0, 0, 0);
            } else {
                const size_t off = (size_t) pr * KD + (size_t) g * kFwD + col;
                if constexpr (sizeof(KVT) == 1) {   // FP8 E4M3 cache: 8 bytes -> 8 FP16, branchless (e4m3x8_h8)
                    const uint2 k8 = ok ? *(const uint2*) (kc + off) : make_uint2(0, 0);
                    const uint2 v8 = ok ? *(const uint2*) (vc + off) : make_uint2(0, 0);
                    kv = e4m3x8_h8(k8);
                    vv = e4m3x8_h8(v8);
                } else {                            // FP16 cache: 16 bytes straight
                    kv = ok ? *(const uint4*) (kc + off) : make_uint4(0, 0, 0, 0);
                    vv = ok ? *(const uint4*) (vc + off) : make_uint4(0, 0, 0, 0);
                }
            }
            *(uint4*) (Ks + row * kFwLd + col) = kv;
            const uint16_t* vh = (const uint16_t*) &vv;
#pragma unroll
            for (int e = 0; e < 8; ++e) Vt[(col + e) * BN + row] = vh[e];   // Vt[dim][key]
        }
        __syncthreads();
        // QK^T: S[query, key] = sum_d Q[query,d] K[key,d]; C lane = key lane_lo, queries 2*i+lane_hi
        fw_v8f s[BN / 16];
#pragma unroll
        for (int j = 0; j < BN / 16; ++j) s[j] = fw_v8f{0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int j = 0; j < BN / 16; ++j) {
            const uint16_t* kr = Ks + (j * 16 + lane_lo) * kFwLd;
#pragma unroll
            for (int kk = 0; kk < kFwD / 16; ++kk) s[j] = fw_mma(qf[kk], fw_load16(kr + kk * 16), s[j]);
        }
        // scale + rel-bias (global, L2) + causal/window mask, then per-query row max across the 16 keys
        float mx[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) mx[i] = -INFINITY;
#pragma unroll
        for (int j = 0; j < BN / 16; ++j) {
            const int key = kt + j * 16 + lane_lo;
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int q = 2 * i + lane_hi, qp = qpos0 + wi + q;
                float v = s[j][i] * sc;
                if (near) {
                    const int dist = qp - key;
                    if (dist >= 0 && dist < extent) v += __ldg(bias + ((size_t) (wi + q) * H + h) * extent + dist) * kLog2e;
                }
                const bool valid = (wi + q < Tq) && (key <= qp) && (qp - key < window) && (key < k_hi);
                v = valid ? v : -INFINITY;
                s[j][i] = v;
                mx[i] = fmaxf(mx[i], v);
            }
        }
#pragma unroll
        for (int off = 1; off < 16; off <<= 1)
#pragma unroll
            for (int i = 0; i < 8; ++i) mx[i] = fmaxf(mx[i], __shfl_xor_sync(0xffffffffu, mx[i], off));
        // online softmax rescale of the running output and sum
        float corr[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const float mn = fmaxf(m[i], mx[i]);
            corr[i] = (mn == -INFINITY) ? 1.0f : exp2f(m[i] - mn);
            l[i] *= corr[i];
            m[i] = mn;
        }
#pragma unroll
        for (int j = 0; j < kFwD / 16; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) o[j][i] *= corr[i];
        // P = exp2(S - m) in FP16, staged transposed as the A fragments of PV: Ps[query][key]
        // (C of QK holds key=lane_lo, queries 2*i+lane_hi; A of PV needs query=lane_lo, keys kk*16+e.)
        float ls[8];
#pragma unroll
        for (int i = 0; i < 8; ++i) ls[i] = 0.0f;
#pragma unroll
        for (int j = 0; j < BN / 16; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int q = 2 * i + lane_hi, key = j * 16 + lane_lo;
                const float p = (m[i] == -INFINITY) ? 0.0f : exp2f(s[j][i] - m[i]);
                Ps[q * BN + key] = f2h(p);
                ls[i] += p;
            }
#pragma unroll
        for (int off = 1; off < 16; off <<= 1)
#pragma unroll
            for (int i = 0; i < 8; ++i) ls[i] += __shfl_xor_sync(0xffffffffu, ls[i], off);
#pragma unroll
        for (int i = 0; i < 8; ++i) l[i] += ls[i];
        __syncwarp();
        // PV: O[query, dim] += sum_key P[query,key] V[key,dim]; B = Vt[dim, key] (N=dims, K=keys)
#pragma unroll
        for (int kk = 0; kk < BN / 16; ++kk) {
            const fw_vec16 pa = fw_load16(Ps + lane_lo * BN + kk * 16);   // A[query=lane_lo, keys kk*16..]
#pragma unroll
            for (int j = 0; j < kFwD / 16; ++j) {
                const fw_vec16 vb = fw_load16(Vt + (j * 16 + lane_lo) * BN + kk * 16);   // B[dim=lane_lo, keys]
                o[j] = fw_mma(pa, vb, o[j]);
            }
        }
        __syncthreads();
    }
    // write this split's unnormalised output and (max, sum); combine merges splits + sink
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int q = 2 * i + lane_hi, gi = wi + q;
        if (gi < Tq) {
            float* op = o_part + (((size_t) split * Tq + gi) * H + h) * kFwD + lane_lo;
#pragma unroll
            for (int j = 0; j < kFwD / 16; ++j) op[j * 16] = o[j][i];
            if (lane_lo == 0) ml_part[((size_t) split * Tq + gi) * H + h] = make_float2(m[i], l[i]);
        }
    }
}
#endif   // STRATA_USE_HIP && STRATA_WMMA_GFX11

// the splits merged with the sink: out [Tq] rows of o_ld, head h at h * D; a block a (query, head), D threads
__global__ void attn_flash_combine_kernel(const float* __restrict__ o_part, const float2* __restrict__ ml_part,
                                          int n_split, int Tq, int H, const float* __restrict__ sink, float* out,
                                          int o_ld, float v_inv) {
    const int i = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    float m = sink[(size_t) i * H + h] * kLog2e;
    for (int s = 0; s < n_split; ++s) m = fmaxf(m, ml_part[((size_t) s * Tq + i) * H + h].x);
    float den = exp2f(sink[(size_t) i * H + h] * kLog2e - m), acc = 0.0f;
    for (int s = 0; s < n_split; ++s) {
        const float2 ml = ml_part[((size_t) s * Tq + i) * H + h];
        if (ml.x == -INFINITY) continue;
        const float c = exp2f(ml.x - m);
        den += ml.y * c;
        acc += o_part[(((size_t) s * Tq + i) * H + h) * kFaD + d] * c;
    }
    out[(size_t) i * o_ld + (size_t) h * kFaD + d] = acc / den * v_inv;   // (v_inv: an FP8 cache's V scale)
}

// ---- one token's attention on the tensor cores (long contexts: the scalar kernels are bound by their arithmetic)
// A warp at a time and straight from the cache into the registers: no shared memory, so a dozen and more warps an
// SM run it, and a warp's wait for its rows is another warp's arithmetic - the kernel reads at the memory's speed,
// an FP8 cache too (its bytes made FP16 in the registers).  (attn_flash's way - tiles in shared memory, one block
// of four warps an SM, one of them computing - read an FP16 cache at the memory's speed and an FP8 one at a third
// of it.)
// A block's warps share a K / V head; a warp takes a run of its keys, 16 a step, for the R = H / Hkv <= 8 heads of
// that K / V head - a key's K and V rows are read once for them all:
//   Q K^T   C [16 keys][8 heads] += A [16 keys][16 dims] B [16 dims][8 heads].  A is K as it lies in the cache: a
//           lane loads 4 consecutive values of two keys, and the 16 dims of a step are taken in the order that makes
//           those the lane's fragment; B is the heads' q in the same order.
//   P V     C [16 dims][8 heads] += A [16 dims][16 keys] B [16 keys][8 heads].  A pairs two keys' values of a dim:
//           a lane loads 16 consecutive values of four keys and interleaves them; B is P, shuffled from the lanes
//           that hold a (key, head) logit to the lanes of that head's fragment.
// The online softmax in the log2 domain: a lane has the logits of two heads (the columns 2 tq, 2 tq + 1 of both
// C's), keeps their (max, sum) and 16 dims of their output.  A warp writes its unnormalised output and (max, sum);
// attn_decode_mma_combine merges the warps and the sink (and takes the output through the layer's gate: one launch
// less).  The relative bias r . proj is taken where a step is within the extent of the token, by the lanes that hold
// the logits.

constexpr int kWaNW = 4;    // warps a block
constexpr int kWaMin = 1;   // steps a warp before the next warp has keys (1: a short context or a window is bound by
                            // a warp's latency, not by the partials to merge)
// the keys of a token at qpos: steps of 16 from k_lo; of nw warps, the ones that have keys
__device__ __forceinline__ int wa_active(int qpos, int window, int nw, int& k_lo, int& n_steps) {
    k_lo = window < (1 << 30) ? max(0, qpos - window + 1) : 0;
    n_steps = (qpos + 1 - k_lo + 15) >> 4;
    return min(nw, (n_steps + kWaMin - 1) / kWaMin);
}

template <class KVT>
__global__ void __launch_bounds__(kWaNW * 32, 4)
attn_decode_mma_kernel(const uint16_t* __restrict__ q16, const KVT* __restrict__ kc, const KVT* __restrict__ vc,
                       int kv_cap, int H, int Hkv, const int* pos_d, int qpos, float qk_scale,
                       const float* __restrict__ r, const float* __restrict__ proj, int d_rel, bool rel4, int extent,
                       int window, int n_split, float* __restrict__ o_part, float2* __restrict__ ml_part) {
    constexpr bool F8 = sizeof(KVT) == 1;
    if (pos_d) qpos = *pos_d;   // (a token of a replayed graph: its position read here)
    const int lane = threadIdx.x & 31, gr = lane >> 2, tq = lane & 3;
    const int g = blockIdx.x, R = H / Hkv, KD = Hkv * kFaD;
    const int w = blockIdx.y * kWaNW + (threadIdx.x >> 5);   // this warp of the K / V head's
    int k_lo, n_steps;
    const int n_act = wa_active(qpos, window, n_split * kWaNW, k_lo, n_steps);
    if (w >= n_act) return;
    const int per = (n_steps + n_act - 1) / n_act;
    const int st0 = min(n_steps, w * per), st1 = min(n_steps, st0 + per);
    // the lane's part: head gr's q and P (B's column gr), the logits and the output of the heads qa and qb
    const int qa = 2 * tq, qb = qa + 1;
    const bool q_ok = gr < R, a_ok = qa < R, b_ok = qb < R;
    uint32_t qf[kFaD / 16][2];
    {
        const uint2* qr = (const uint2*) (q16 + (size_t) (g * R + (q_ok ? gr : 0)) * kFaD);
#pragma unroll
        for (int kk = 0; kk < kFaD / 16; ++kk) {
            const uint2 v = qr[kk * 4 + tq];   // (the dims 16 kk + 4 tq .. + 3)
            qf[kk][0] = q_ok ? v.x : 0u, qf[kk][1] = q_ok ? v.y : 0u;
        }
    }
    float o[kFaD / 16][4];
#pragma unroll
    for (int j = 0; j < kFaD / 16; ++j) o[j][0] = o[j][1] = o[j][2] = o[j][3] = 0.0f;
    float ma = -INFINITY, mb = -INFINITY, la = 0.0f, lb = 0.0f;
    const float sc = qk_scale * kLog2e;
    const KVT* kg = kc + (size_t) g * kFaD + tq * 4;    // the lane's values of a K row (+ 16 kk) ...
    const KVT* vg = vc + (size_t) g * kFaD + gr * 16;   // ... and of a V row
    const float* ra = r + (size_t) (g * R + (a_ok ? qa : 0)) * d_rel;   // (the bias: the heads' r)
    const float* rb = r + (size_t) (g * R + (b_ok ? qb : 0)) * d_rel;
    const int src = tq * 8 + (gr >> 1);   // the lane with head gr's logits of the key 2 tq (+ 4: of the key 2 tq + 1)
    const uint32_t psel = (gr & 1) ? 0x7632u : 0x5410u;

    for (int st = st0; st < st1; ++st) {
        const int kt = k_lo + st * 16, kmax = qpos - kt;
        const int row0 = kv_cap > 0 ? kt % kv_cap : kt;
        // a key's row in the cache (a key past the token's: the token's row, its logits masked below)
        auto row = [&](int key) {
            int rw = row0 + min(key, kmax);
            if (kv_cap > 0 && rw >= kv_cap) rw -= kv_cap;
            return (size_t) rw * KD;
        };
        // Q K^T: the keys gr and gr + 8
        float sa[4] = {0.0f, 0.0f, 0.0f, 0.0f}, sb[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        {
            const KVT* p0 = kg + row(gr);
            const KVT* p1 = kg + row(gr + 8);
            uint32_t ka[kFaD / 16][4];
            if constexpr (F8) {
                uint32_t w0[kFaD / 16], w1[kFaD / 16];
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    w0[kk] = __ldg((const uint32_t*) (p0 + kk * 16));
                    w1[kk] = __ldg((const uint32_t*) (p1 + kk * 16));
                }
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    e4m3x4_h4(w0[kk], ka[kk][0], ka[kk][2]);
                    e4m3x4_h4(w1[kk], ka[kk][1], ka[kk][3]);
                }
            } else {
#pragma unroll
                for (int kk = 0; kk < kFaD / 16; ++kk) {
                    const uint2 v0 = __ldg((const uint2*) (p0 + kk * 16)), v1 = __ldg((const uint2*) (p1 + kk * 16));
                    ka[kk][0] = v0.x, ka[kk][2] = v0.y, ka[kk][1] = v1.x, ka[kk][3] = v1.y;
                }
            }
#pragma unroll
            for (int kk = 0; kk < kFaD / 16; ++kk) mma16816((kk & 1) ? sb : sa, ka[kk], qf[kk][0], qf[kk][1]);
        }
        // the logits (log2 units): scale, bias, mask
        const int k0 = kt + gr, k1 = k0 + 8;
        float c0 = (sa[0] + sb[0]) * sc, c1 = (sa[1] + sb[1]) * sc, c2 = (sa[2] + sb[2]) * sc, c3 = (sa[3] + sb[3]) * sc;
        if (qpos - (kt + 15) < extent) {   // (a step within the bias's extent of the token; zero past the extent)
            const int d0 = qpos - k0, d1 = qpos - k1;
            const bool e0 = d0 >= 0 && d0 < extent, e1 = d1 >= 0 && d1 < extent;
            const float* pj0 = proj + (size_t) (e0 ? d0 : 0) * d_rel;
            const float* pj1 = proj + (size_t) (e1 ? d1 : 0) * d_rel;
            float b0a = 0.0f, b0b = 0.0f, b1a = 0.0f, b1b = 0.0f;   // (key gr | gr + 8, head qa | qb)
            if (rel4) {
                for (int e = 0; e < d_rel; e += 4) {
                    const float4 x0 = __ldg((const float4*) (pj0 + e)), x1 = __ldg((const float4*) (pj1 + e));
                    const float4 ya = __ldg((const float4*) (ra + e)), yb = __ldg((const float4*) (rb + e));
                    b0a += x0.x * ya.x + x0.y * ya.y + x0.z * ya.z + x0.w * ya.w;
                    b0b += x0.x * yb.x + x0.y * yb.y + x0.z * yb.z + x0.w * yb.w;
                    b1a += x1.x * ya.x + x1.y * ya.y + x1.z * ya.z + x1.w * ya.w;
                    b1b += x1.x * yb.x + x1.y * yb.y + x1.z * yb.z + x1.w * yb.w;
                }
            } else {
                for (int e = 0; e < d_rel; ++e) {
                    const float x0 = pj0[e], x1 = pj1[e], ya = ra[e], yb = rb[e];
                    b0a += x0 * ya, b0b += x0 * yb, b1a += x1 * ya, b1b += x1 * yb;
                }
            }
            if (e0) c0 += b0a * kLog2e, c1 += b0b * kLog2e;
            if (e1) c2 += b1a * kLog2e, c3 += b1b * kLog2e;
        }
        c0 = (k0 > qpos || !a_ok) ? -INFINITY : c0;
        c1 = (k0 > qpos || !b_ok) ? -INFINITY : c1;
        c2 = (k1 > qpos || !a_ok) ? -INFINITY : c2;
        c3 = (k1 > qpos || !b_ok) ? -INFINITY : c3;
        float mxa = fmaxf(c0, c2), mxb = fmaxf(c1, c3);   // (over the lanes of the same heads: the same tq)
#pragma unroll
        for (int sh = 4; sh < 32; sh <<= 1) {
            mxa = fmaxf(mxa, __shfl_xor_sync(0xffffffffu, mxa, sh));
            mxb = fmaxf(mxb, __shfl_xor_sync(0xffffffffu, mxb, sh));
        }
        if (__ballot_sync(0xffffffffu, mxa > ma || mxb > mb)) {   // a new maximum (few steps have one): rescale
            const float mna = fmaxf(ma, mxa), mnb = fmaxf(mb, mxb);
            const float ca = mna == -INFINITY ? 1.0f : exp2f(ma - mna);   // (ma = -inf: exp2f(-inf) = 0)
            const float cb = mnb == -INFINITY ? 1.0f : exp2f(mb - mnb);
            ma = mna, mb = mnb;
            la *= ca, lb *= cb;
#pragma unroll
            for (int j = 0; j < kFaD / 16; ++j) o[j][0] *= ca, o[j][1] *= cb, o[j][2] *= ca, o[j][3] *= cb;
        }
        const float ua = ma == -INFINITY ? 0.0f : ma, ub = mb == -INFINITY ? 0.0f : mb;
        const float p0 = exp2f(c0 - ua), p1 = exp2f(c1 - ub), p2 = exp2f(c2 - ua), p3 = exp2f(c3 - ub);
        la += p0 + p2, lb += p1 + p3;
        // P (FP16) to the lanes of its fragment: head gr's of the keys 2 tq, 2 tq + 1 and 2 tq + 8, 2 tq + 9
        const uint32_t plo = f2_to_u32(p0, p1), phi = f2_to_u32(p2, p3);   // (key gr, key gr + 8: heads qa | qb)
        const uint32_t alo = __shfl_sync(0xffffffffu, plo, src), blo = __shfl_sync(0xffffffffu, plo, src + 4);
        const uint32_t ahi = __shfl_sync(0xffffffffu, phi, src), bhi = __shfl_sync(0xffffffffu, phi, src + 4);
        const uint32_t pb0 = q_ok ? __byte_perm(alo, blo, psel) : 0u, pb1 = q_ok ? __byte_perm(ahi, bhi, psel) : 0u;
        // P V: the dims 16 gr .. + 15 of the keys 2 tq, 2 tq + 1, 2 tq + 8, 2 tq + 9; o[j]: the dims 16 gr + 2 j, + 1
        {
            const KVT* v0 = vg + row(2 * tq);
            const KVT* v1 = vg + row(2 * tq + 1);
            const KVT* v8 = vg + row(2 * tq + 8);
            const KVT* v9 = vg + row(2 * tq + 9);
            if constexpr (F8) {
                const uint4 r0 = __ldg((const uint4*) v0), r1 = __ldg((const uint4*) v1);
                const uint4 r8 = __ldg((const uint4*) v8), r9 = __ldg((const uint4*) v9);
                const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
                const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
#pragma unroll
                for (int i = 0; i < 4; ++i) {   // (four bytes: two dims of two keys)
                    uint32_t a[4];
                    e4m3x4_h4(__byte_perm(x0[i], x1[i], 0x5140u), a[0], a[1]);
                    e4m3x4_h4(__byte_perm(x8[i], x9[i], 0x5140u), a[2], a[3]);
                    mma16816(o[2 * i], a, pb0, pb1);
                    e4m3x4_h4(__byte_perm(x0[i], x1[i], 0x7362u), a[0], a[1]);
                    e4m3x4_h4(__byte_perm(x8[i], x9[i], 0x7362u), a[2], a[3]);
                    mma16816(o[2 * i + 1], a, pb0, pb1);
                }
            } else {
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    const uint4 r0 = __ldg((const uint4*) v0 + i), r1 = __ldg((const uint4*) v1 + i);
                    const uint4 r8 = __ldg((const uint4*) v8 + i), r9 = __ldg((const uint4*) v9 + i);
                    const uint32_t x0[4] = {r0.x, r0.y, r0.z, r0.w}, x1[4] = {r1.x, r1.y, r1.z, r1.w};
                    const uint32_t x8[4] = {r8.x, r8.y, r8.z, r8.w}, x9[4] = {r9.x, r9.y, r9.z, r9.w};
#pragma unroll
                    for (int jj = 0; jj < 4; ++jj) {
                        const uint32_t a[4] = {__byte_perm(x0[jj], x1[jj], 0x5410u), __byte_perm(x0[jj], x1[jj], 0x7632u),
                                               __byte_perm(x8[jj], x9[jj], 0x5410u), __byte_perm(x8[jj], x9[jj], 0x7632u)};
                        mma16816(o[4 * i + jj], a, pb0, pb1);
                    }
                }
            }
        }
    }
#pragma unroll
    for (int sh = 4; sh < 32; sh <<= 1) {
        la += __shfl_xor_sync(0xffffffffu, la, sh);
        lb += __shfl_xor_sync(0xffffffffu, lb, sh);
    }
    // the warp's partials: o_part [warp][H][D], ml_part [warp][H] = (max in log2 units, sum)
    if (a_ok) {
        float* op = o_part + ((size_t) w * H + g * R + qa) * kFaD + gr * 16;
#pragma unroll
        for (int j = 0; j < kFaD / 16; ++j) *(float2*) (op + 2 * j) = make_float2(o[j][0], o[j][2]);
        if (gr == 0) ml_part[(size_t) w * H + g * R + qa] = make_float2(ma, la);
    }
    if (b_ok) {
        float* op = o_part + ((size_t) w * H + g * R + qb) * kFaD + gr * 16;
#pragma unroll
        for (int j = 0; j < kFaD / 16; ++j) *(float2*) (op + 2 * j) = make_float2(o[j][1], o[j][3]);
        if (gr == 0) ml_part[(size_t) w * H + g * R + qb] = make_float2(mb, lb);
    }
}
// the warps merged with the sink: out [H][D] (times v_inv: an FP8 cache's V scale; times the sigmoid of gate [H][D]
// if given; out16: an FP16 copy); a block a head, D threads.  A warp's weight (the exp2 of its max less the overall
// one) is found once and kept in shared memory.
__global__ void attn_decode_mma_combine_kernel(const float* __restrict__ o_part, const float2* __restrict__ ml_part,
                                               int n_split, int H, const float* __restrict__ sink, float v_inv,
                                               const int* pos_d, int qpos, int window, const float* __restrict__ gate,
                                               float* out, uint16_t* out16) {
    extern __shared__ float cm_sm[];
    float* red = cm_sm;         // [kFaD] the block's reductions
    float* cw = cm_sm + kFaD;   // [n_split * kWaNW] the warps' weights
    if (pos_d) qpos = *pos_d;
    int k_lo, n_steps;
    const int n_act = wa_active(qpos, window, n_split * kWaNW, k_lo, n_steps);
    const int h = blockIdx.x, d = threadIdx.x;
    const float sk = sink[h] * kLog2e;
    float m = d == 0 ? sk : -INFINITY;
    for (int s = d; s < n_act; s += kFaD) m = fmaxf(m, ml_part[(size_t) s * H + h].x);
    m = bmax(m, red);
    float den = d == 0 ? exp2f(sk - m) : 0.0f;
    for (int s = d; s < n_act; s += kFaD) {
        const float2 ml = ml_part[(size_t) s * H + h];
        const float c = ml.x == -INFINITY ? 0.0f : exp2f(ml.x - m);
        cw[s] = c;
        den += ml.y * c;
    }
    den = bsum(den, red);   // (and its barriers: the weights are all there)
    float acc = 0.0f;
    for (int s = 0; s < n_act; ++s) acc += o_part[((size_t) s * H + h) * kFaD + d] * cw[s];
    float res = acc / den * v_inv;
    if (gate) res = res * (1.0f / (1.0f + expf(-gate[(size_t) h * kFaD + d])));
    out[(size_t) h * kFaD + d] = res;
    if (out16) out16[(size_t) h * kFaD + d] = f2h(res);
}

template <bool WRITE_F16, bool WRITE_F32>
__global__ void gate_mul_impl_kernel(float* o, const float* g, size_t n, uint16_t* o16) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (((uintptr_t) o & 15) == 0 && ((uintptr_t) g & 15) == 0 && (!WRITE_F16 || ((uintptr_t) o16 & 7) == 0)) {
        const size_t n4 = n >> 2;
        if (i < n4) {
            const float4 ov = ((const float4*) o)[i];
            const float4 gv = __ldg((const float4*) g + i);
            const float r0 = ov.x * (1.0f / (1.0f + expf(-gv.x)));
            const float r1 = ov.y * (1.0f / (1.0f + expf(-gv.y)));
            const float r2 = ov.z * (1.0f / (1.0f + expf(-gv.z)));
            const float r3 = ov.w * (1.0f / (1.0f + expf(-gv.w)));
            if (WRITE_F32) ((float4*) o)[i] = make_float4(r0, r1, r2, r3);
            if (WRITE_F16) ((uint2*) o16)[i] = make_uint2(f2_to_u32(r0, r1), f2_to_u32(r2, r3));
        }
        const size_t tail = (n4 << 2) + i;
        if (tail < n) {
            const float r = o[tail] * (1.0f / (1.0f + expf(-g[tail])));
            if (WRITE_F32) o[tail] = r;
            if (WRITE_F16) o16[tail] = f2h(r);
        }
        return;
    }
    if (i < n) {
        const float r = o[i] * (1.0f / (1.0f + expf(-g[i])));
        if (WRITE_F32) o[i] = r;
        if (WRITE_F16) o16[i] = f2h(r);
    }
}

constexpr int kRouteMax = 16;
// Single-warp zero-barrier router for E <= 512: each lane holds <= 16 experts in registers.
__global__ void route_warp_kernel(const float* logits, const float* bias, int T, int E, int K, float scale, int* top,
                                  float* w) {
    const int t = blockIdx.x, lane = threadIdx.x;   // 32 threads
    const float* l = logits + (size_t) t * E;
    float val[16];
#pragma unroll
    for (int m = 0; m < 16; ++m) {
        const int e = lane + m * 32;
        val[m] = (e < E) ? (1.0f / (1.0f + expf(-__ldg(l + e))) + __ldg(bias + e)) : -INFINITY;
    }
    int ch[kRouteMax];
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
#pragma unroll
        for (int m = 0; m < 16; ++m) {
            const int e = lane + m * 32;
            if (val[m] > best || (val[m] == best && e < idx)) {
                best = val[m];
                idx = e;
            }
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi < idx)) {
                best = ob;
                idx = oi;
            }
        }
        ch[k] = idx;
        if (idx < E && (idx & 31) == lane) val[idx >> 5] = -INFINITY;
    }
    if (lane == 0) {
        for (int a = 1; a < K; ++a) {   // ascending expert order
            const int x = ch[a];
            int b = a - 1;
            for (; b >= 0 && ch[b] > x; --b) ch[b + 1] = ch[b];
            ch[b + 1] = x;
        }
        float sc[kRouteMax], sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += (sc[k] = 1.0f / (1.0f + expf(-__ldg(l + ch[k]))));
        for (int k = 0; k < K; ++k) {
            top[(size_t) t * K + k] = ch[k];
            w[(size_t) t * K + k] = sc[k] / (sum + 1e-20f) * scale;
        }
    }
}

// a block a token: the scores in shared memory, then K rounds of the block's argmax (ties: the lower expert)
__global__ void route_kernel(const float* logits, const float* bias, int T, int E, int K, float scale, int* top,
                             float* w) {
    extern __shared__ float v[];   // [E]
    __shared__ float bv[32];
    __shared__ int bi[32];
    __shared__ int ch[kRouteMax];
    const int t = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, nw = blockDim.x >> 5;
    const float* l = logits + (size_t) t * E;
    for (int e = tid; e < E; e += blockDim.x) v[e] = 1.0f / (1.0f + expf(-l[e])) + bias[e];
    __syncthreads();
    for (int k = 0; k < K; ++k) {
        float best = -INFINITY;
        int idx = 0x7fffffff;
        for (int e = tid; e < E; e += blockDim.x)
            if (v[e] > best || (v[e] == best && e < idx)) best = v[e], idx = e;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            if (ob > best || (ob == best && oi < idx)) best = ob, idx = oi;
        }
        if (lane == 0) bv[warp] = best, bi[warp] = idx;
        __syncthreads();
        if (tid == 0) {
            float b = bv[0];
            int i = bi[0];
            for (int j = 1; j < nw; ++j)
                if (bv[j] > b || (bv[j] == b && bi[j] < i)) b = bv[j], i = bi[j];
            ch[k] = i;
            v[i] = -INFINITY;
        }
        __syncthreads();
    }
    if (tid == 0) {
        int c[kRouteMax];
        for (int k = 0; k < K; ++k) c[k] = ch[k];
        for (int a = 1; a < K; ++a) {   // ascending expert order
            const int x = c[a];
            int b = a - 1;
            for (; b >= 0 && c[b] > x; --b) c[b + 1] = c[b];
            c[b + 1] = x;
        }
        float sc[kRouteMax], sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += (sc[k] = 1.0f / (1.0f + expf(-l[c[k]])));
        for (int k = 0; k < K; ++k) {
            top[(size_t) t * K + k] = c[k];
            w[(size_t) t * K + k] = sc[k] / (sum + 1e-20f) * scale;
        }
    }
}

__global__ void gather_f16_kernel(const float* x, const int* idx, int n, int C, uint16_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * C) return;
    y[i] = f2h(x[(size_t) idx[i / C] * C + i % C]);
}

__global__ void swiglu_cap_kernel(const uint16_t* gu, int n, int F, const float* sg, const float* su, float gcap,
                                  float ucap, uint16_t* h) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * F) return;
    const size_t r = i / F, c = i % F;
    const float g = h2f(gu[r * 2 * F + c]) * sg[r];
    const float u = h2f(gu[r * 2 * F + F + c]) * su[r];
    h[i] = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
}

__global__ void moe_combine_kernel(const uint16_t* y, const int* inv, const float* wt, int T, int K, int C, float* out,
                                   bool add) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * C) return;
    const size_t t = i / C, c = i % C;
    float a = 0.0f;
    for (int k = 0; k < K; ++k) a += wt[t * K + k] * h2f(y[(size_t) inv[t * K + k] * C + c]);
    out[i] = add ? out[i] + a : a;
}

__global__ void swiglu_cap_f32_kernel(const float* gu, int n, int F, const float* sg, const float* su, float gcap,
                                      float ucap, uint16_t* h) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) n * F) return;
    const size_t r = i / F, c = i % F;
    const float g = gu[r * 2 * F + c] * sg[r];
    const float u = gu[r * 2 * F + F + c] * su[r];
    h[i] = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
}

// one token's moe_combine_f32 and the conv4_res after it in one launch: out = conv(sum_k wt[k] y[inv[k]]) + res
__global__ void moe_combine_conv_kernel(const float* __restrict__ y, const int* __restrict__ inv,
                                        const float* __restrict__ wt, int K, int C, float* hist,
                                        const float* __restrict__ w, const float* res, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float a = 0.0f;
    for (int k = 0; k < K; ++k) a += wt[k] * y[(size_t) inv[k] * C + c];
    const float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c];
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float b = w0 * h0 + w1 * h1 + w2 * h2 + w3 * a;
    if (res) b += res[c];
    out[c] = b;
    hist[c] = h1;
    hist[(size_t) C + c] = h2;
    hist[(size_t) 2 * C + c] = a;
}
__global__ void moe_combine_f32_kernel(const float* y, const int* inv, const float* wt, int T, int K, int C, float* out,
                                       bool add) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * C) return;
    const size_t t = i / C, c = i % C;
    float a = 0.0f;
    for (int k = 0; k < K; ++k) a += wt[t * K + k] * y[(size_t) inv[t * K + k] * C + c];
    out[i] = add ? out[i] + a : a;
}

__global__ void add_kernel(const float* a, const float* b, size_t n, float* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a[i] + b[i];
}
__global__ void to_f16_kernel(const float* x, size_t n, uint16_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2h(x[i]);
}

__global__ void to_f16_limbs_kernel(const float* x, size_t n, uint16_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const uint16_t hi = f2h(x[i]);
    y[i] = hi, y[n + i] = f2h(x[i] - h2f(hi));
}
__global__ void to_f16_limbs_tokens_kernel(const float* x, int T, int C, uint16_t* y) {
    const size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = (size_t) T * C;
    if (idx >= total) return;
    const int t = (int) (idx / C);
    const int d = (int) (idx % C);
    const uint16_t hi = f2h(x[idx]);
    y[(size_t) t * 2 * C + d] = hi;
    y[(size_t) t * 2 * C + C + d] = f2h(x[idx] - h2f(hi));
}
__global__ void softcap_kernel(float* x, size_t n, float cap) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = cap * tanhf(x[i] / cap);
}

// The loop's message gates, up to three (a layer's q, k and v inputs) in two launches that spread over the SMs.
// sc [n][T][2 R + 2]: per gate and token, the two halves of each g before the norms (D's x columns . x, its m columns
// . m), then the two sums of squares.
struct LoopGateArgs {
    const float *x[3], *m[3], *D[3], *F[3], *G[3];
    float* y[3];
};
constexpr int kLoopRankMax = 128;
// (1) a block a rank (128 threads): its two dots; rank 0's block also stores the sums of squares
__global__ void loop_gate_dots_kernel(LoopGateArgs a, int T, int C, int R, float* sc) {
    __shared__ float sh[4][4];
    const int r = blockIdx.x, j = blockIdx.y, t = blockIdx.z, tid = threadIdx.x, warp = tid >> 5;
    const float* xr = a.x[j] + (size_t) t * C;
    const float* mr = a.m[j] + (size_t) t * C;
    const float* dr = a.D[j] + (size_t) r * 2 * C;
    float ax = 0.0f, am = 0.0f, sx = 0.0f, sm = 0.0f;
    for (int c = tid; c < C; c += 128) {
        const float xv = xr[c], mv = mr[c];
        ax += dr[c] * xv, am += dr[C + c] * mv;
        sx += xv * xv, sm += mv * mv;
    }
    ax = wsum(ax), am = wsum(am), sx = wsum(sx), sm = wsum(sm);
    if ((tid & 31) == 0) sh[warp][0] = ax, sh[warp][1] = am, sh[warp][2] = sx, sh[warp][3] = sm;
    __syncthreads();
    if (tid == 0) {
        float* o = sc + ((size_t) j * T + t) * (2 * R + 2);
        o[2 * r] = sh[0][0] + sh[1][0] + sh[2][0] + sh[3][0];
        o[2 * r + 1] = sh[0][1] + sh[1][1] + sh[2][1] + sh[3][1];
        if (r == 0) {
            o[2 * R] = sh[0][2] + sh[1][2] + sh[2][2] + sh[3][2];
            o[2 * R + 1] = sh[0][3] + sh[1][3] + sh[2][3] + sh[3][3];
        }
    }
}
// (2) y = (1 - tanh(F g)) x + tanh(G g) m: a block is kBlk channels of a token, a thread a channel; F and G by rank
// ([R][C]), so that a warp's threads read side by side.  y may be x (x's sums were taken in (1)).
__global__ void loop_gate_apply_kernel(LoopGateArgs a, int T, int C, int R, float eps, const float* __restrict__ sc) {
    __shared__ float sh_g[kLoopRankMax];
    const int j = blockIdx.y, t = blockIdx.z, tid = threadIdx.x;
    const float* o = sc + ((size_t) j * T + t) * (2 * R + 2);
    if (tid < R) {
        const float rx = rsqrtf(o[2 * R] / (float) C + eps), rm = rsqrtf(o[2 * R + 1] / (float) C + eps);
        sh_g[tid] = rx * o[2 * tid] + rm * o[2 * tid + 1];
    }
    __syncthreads();
    const int c = (int) blockIdx.x * kBlk + tid;
    if (c >= C) return;
    const float* F = a.F[j];
    const float* G = a.G[j];
    float f = 0.0f, gn = 0.0f;
    for (int r = 0; r < R; ++r) {
        const float gv = sh_g[r];
        f += F[(size_t) r * C + c] * gv, gn += G[(size_t) r * C + c] * gv;
    }
    const size_t i = (size_t) t * C + c;
    a.y[j][i] = (1.0f - tanhf(f)) * a.x[j][i] + tanhf(gn) * a.m[j][i];
}
// ---- sampling: the largest logits found on the device - the host gets a thousand or two candidates and not the
// vocabulary (copying 200,000 logits out and sorting them was 0.8 ms a token).  A histogram of the values a block,
// the bin from which the bins hold K values at least, the values of those bins gathered.
constexpr int kCandBins = 1024;
__device__ __forceinline__ int cand_bin(float x, float lo, float inv_w) {
    return min(kCandBins - 1, max(0, (int) ((x - lo) * inv_w)));
}
__global__ void cand_hist_kernel(const float* __restrict__ x, int n, float lo, float inv_w, int per,
                                 int* __restrict__ part) {
    __shared__ int h[kCandBins];
    for (int j = threadIdx.x; j < kCandBins; j += blockDim.x) h[j] = 0;
    __syncthreads();
    const int i0 = blockIdx.x * per, i1 = min(n, i0 + per);
    for (int i = i0 + threadIdx.x; i < i1; i += blockDim.x) atomicAdd(&h[cand_bin(x[i], lo, inv_w)], 1);
    __syncthreads();
    for (int j = threadIdx.x; j < kCandBins; j += blockDim.x) part[(size_t) blockIdx.x * kCandBins + j] = h[j];
}
// a thread a bin (kCandBins threads): head[1] = the bin from which the bins hold K values (0 if they all hold
// fewer), head[0] = 0 (the gathered count starts there)
__global__ void cand_threshold_kernel(const int* __restrict__ part, int n_part, int K, int* head) {
    __shared__ int cnt[kCandBins];
    __shared__ int wcnt[kCandBins / 32];
    const int j = threadIdx.x, lane = j & 31, warp = j >> 5;
    int c = 0;
    for (int b = 0; b < n_part; ++b) c += part[(size_t) b * kCandBins + j];
    cnt[j] = c;
    int ws = c;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) ws += __shfl_xor_sync(0xffffffffu, ws, o);
    if (lane == 0) wcnt[warp] = ws;
    __syncthreads();
    int above = 0;   // the values of the bins over this one
    for (int w = warp + 1; w < kCandBins / 32; ++w) above += wcnt[w];
    for (int i = j + 1; i < (warp + 1) * 32; ++i) above += cnt[i];
    if (above < K && above + c >= K) head[1] = j;
    if (j == 0) {
        head[0] = 0;
        if (above + c < K) head[1] = 0;
    }
}
__global__ void cand_gather_kernel(const float* __restrict__ x, int n, float lo, float inv_w, int cap, int* head,
                                   int* idx, float* val) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = x[i];
    if (cand_bin(v, lo, inv_w) < head[1]) return;
    const int at = atomicAdd(head, 1);
    if (at < cap) idx[at] = i, val[at] = v;
}

// The gates of a chunk of tokens: their two products (g = D [rms(x); rms(m)], then F g and G g) are matrix products
// taken outside, and these are what is left.  (1) g [T][R] = a1 / rms(x) + a2 / rms(m), a1 = x D_x^T and
// a2 = m D_m^T; a block a token.
__global__ void loop_gate_mix_kernel(const float* __restrict__ x, const float* __restrict__ m,
                                     const float* __restrict__ a1, const float* __restrict__ a2, int C, int R, float eps,
                                     float* __restrict__ g) {
    __shared__ float sh[kBlk];
    const int t = blockIdx.x, tid = threadIdx.x;
    const float* xr = x + (size_t) t * C;
    const float* mr = m + (size_t) t * C;
    float sx = 0.0f, sm = 0.0f;
    for (int c = tid; c < C; c += kBlk) {
        const float xv = xr[c], mv = mr[c];
        sx += xv * xv, sm += mv * mv;
    }
    sx = bsum(sx, sh), sm = bsum(sm, sh);
    if (tid < R) {
        const float rx = rsqrtf(sx / (float) C + eps), rm = rsqrtf(sm / (float) C + eps);
        g[(size_t) t * R + tid] = rx * a1[(size_t) t * R + tid] + rm * a2[(size_t) t * R + tid];
    }
}
// (2) y = (1 - tanh f) x + tanh(gn) m, f = g F and gn = g G; y may be x
__global__ void loop_gate_blend_kernel(const float* x, const float* m, const float* __restrict__ f,
                                       const float* __restrict__ gn, size_t n, float* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = (1.0f - tanhf(f[i])) * x[i] + tanhf(gn[i]) * m[i];
}

// ---- the decodes: value v of a run of blocks, ggml's dequantize_row_* arithmetic

__host__ __device__ __forceinline__ float ue4m3(uint8_t x) {   // ggml_ue4m3_to_fp32 (branchless bit-ops)
    const uint32_t e = (x >> 3) & 0xFu;
    const uint32_t m = x & 7u;
    const float norm_f = bits_to_float(((e + 119u) << 23) | (m << 20));
    const float sub_f = (float) m * 0x1.0p-10f;
    const float v = (e == 0u) ? sub_f : norm_f;
    return (x == 0x7Fu) ? 0.0f : v;
}
// IQ3_XXS / IQ4_XS tables: a device copy (read with __ldg: lanes index them divergently) and a host copy.
__device__ const uint32_t d_iq3xxs_grid[256] = {IK2_IQ3XXS_GRID};
static const uint32_t h_iq3xxs_grid[256] = {IK2_IQ3XXS_GRID};
__device__ const int8_t d_iq4nl[16] = {IK2_IQ4NL_VALUES};
static const int8_t h_iq4nl[16] = {IK2_IQ4NL_VALUES};
__host__ __device__ __forceinline__ uint32_t iq3xxs_grid_at(int i) {
#ifdef __CUDA_ARCH__
    return __ldg(d_iq3xxs_grid + i);
#else
    return h_iq3xxs_grid[i];
#endif
}
__host__ __device__ __forceinline__ int iq4nl_at(int i) {
#ifdef __CUDA_ARCH__
    return d_iq4nl[i];
#else
    return h_iq4nl[i];
#endif
}
// ggml's ksigns_iq2xs[s]: the 7 sign bits plus an 8th that makes the count of set bits even
__host__ __device__ __forceinline__ uint32_t iq_signs8(uint32_t s7) {
#ifdef __CUDA_ARCH__
    return s7 | ((uint32_t) (__popc(s7) & 1) << 7);
#else
    uint32_t p = s7 ^ (s7 >> 4);   // (the parity by folding: no GNU builtin, no C++20 <bit> - MSVC builds this too)
    p ^= p >> 2;
    p ^= p >> 1;
    return s7 | (p & 1) << 7;
#endif
}

__host__ __device__ inline float decode_value(int type, const uint8_t* blocks, size_t v) {
    if (type == kNVFP4) {
        const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
        const uint8_t* b = blocks + (v / 64) * 36;
        const int i = (int) (v % 64), sb = i / 16, j = i % 16;
        const uint8_t byte = b[4 + sb * 8 + (j & 7)];
        return (float) kv[j < 8 ? (byte & 0xF) : (byte >> 4)] * ue4m3(b[sb]);
    }
    if (type == kQ8_0) {
        const uint8_t* b = blocks + (v / 32) * 34;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        return d * (float) (int8_t) b[2 + v % 32];
    }
    if (type == kQ6_K) {   // ql [128], qh [64], scales int8 [16], d
        const uint8_t* b = blocks + (v / 256) * 210;
        const int i = (int) (v % 256), n = i / 128, lq = i % 128, qd = lq / 32, l = lq % 32;
        const uint8_t* ql = b + n * 64;
        const uint8_t* qh = b + 128 + n * 32;
        const int8_t* sc = (const int8_t*) b + 192 + n * 8;
        const float d = half_bits_to_float((uint16_t) (b[208] | (b[209] << 8)));
        const uint8_t lo = ql[l + ((qd & 1) ? 32 : 0)];
        const int nib = qd >= 2 ? (lo >> 4) : (lo & 0xF);
        const int q = (int8_t) (nib | (((qh[l] >> (2 * qd)) & 3) << 4)) - 32;
        return d * (float) sc[l / 16 + 2 * qd] * (float) q;
    }
    if (type == kIQ3_XXS) {   // d, qs [64] grid indices, [8] u32 scale (top 4 bits) + 4 x 7 sign bits
        const uint8_t* b = blocks + (v / 256) * 98;
        const int i = (int) (v % 256), ib32 = i / 32, l = (i % 32) / 8, j = i % 8;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const uint8_t* q = b + 2 + 64 + 4 * ib32;
        const uint32_t aux = (uint32_t) q[0] | ((uint32_t) q[1] << 8) | ((uint32_t) q[2] << 16) | ((uint32_t) q[3] << 24);
        const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
        const uint32_t signs = iq_signs8((aux >> (7 * l)) & 127u);
        const uint32_t g = iq3xxs_grid_at(b[2 + 8 * ib32 + 2 * l + (j >> 2)]);
        const float w = (float) ((g >> (8 * (j & 3))) & 0xFFu);
        return db * w * ((signs >> j) & 1u ? -1.0f : 1.0f);
    }
    if (type == kIQ4_XS) {   // d, scales_h, scales_l [4], qs [128]
        const uint8_t* b = blocks + (v / 256) * 136;
        const int i = (int) (v % 256), ib = i / 32, j = i % 32;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const uint32_t sh = (uint32_t) b[2] | ((uint32_t) b[3] << 8);
        const int ls = ((b[4 + ib / 2] >> (4 * (ib % 2))) & 0xF) | (int) (((sh >> (2 * ib)) & 3u) << 4);
        const uint8_t q = b[8 + 16 * ib + (j & 15)];
        return d * (float) (ls - 32) * (float) iq4nl_at(j < 16 ? (q & 0xF) : (q >> 4));
    }
    if (type == kQ5_K) {   // d, dmin, scales [12], qh [32], qs [128]
        const uint8_t* b = blocks + (v / 256) * 176;
        const int i = (int) (v % 256), j64 = i / 64, half = (i % 64) / 32, l = i % 32;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const float mn = half_bits_to_float((uint16_t) (b[2] | (b[3] << 8)));
        const uint8_t* q = b + 4;
        const int is = 2 * j64 + half;
        int sc, m;
        if (is < 4) {
            sc = q[is] & 63, m = q[is + 4] & 63;
        } else {
            sc = (q[is + 4] & 0xF) | ((q[is - 4] >> 6) << 4), m = (q[is + 4] >> 4) | ((q[is] >> 6) << 4);
        }
        const uint8_t lo = b[48 + 32 * j64 + l];
        const int hb = (b[16 + l] >> (2 * j64 + half)) & 1;
        const int val = (half ? (lo >> 4) : (lo & 0xF)) + (hb ? 16 : 0);
        return d * (float) sc * (float) val - mn * (float) m;
    }
    return 0.0f;
}

// The input vector in shared memory, padded: value i at xp(i) = (i >> 5) * 36 + (i & 31), so every 16-float
// NVFP4 sub-block and 32-float Q8_0 block is 16-byte (float4 / LDS.128) aligned with 0 bank conflicts.
__device__ __forceinline__ int xp(int i) { return (i >> 5) * 36 + (i & 31); }
__host__ __device__ constexpr size_t xp_floats(int K) {
    return K <= 0 ? 0 : (size_t) ((K - 1) >> 5) * 36 + ((K - 1) & 31) + 1;
}

// Fast register E2M1 pair decode via FP16 high-byte LUT in __byte_perm -> __half22float2:
// FP16 representations of {0, 1, 2, 3, 4, 6, 8, 12} all have low byte 0x00 and high bytes {0x00, 0x3c, 0x40, 0x42, 0x44, 0x46, 0x48, 0x4a}.
__device__ __forceinline__ float2 e2m1_pair_lo(uint32_t s) {
    const uint32_t sel = (s << 4) & 0x7070u;
    const uint32_t sgn = ((s & 8u) << 12) | ((s & 0x800u) << 20);
    return u32_to_f2(__byte_perm(0x42403c00u, 0x4a484644u, sel) | sgn);
}
__device__ __forceinline__ float2 e2m1_pair_hi(uint32_t s) {
    const uint32_t sel = s & 0x7070u;
    const uint32_t sgn = ((s & 0x80u) << 8) | ((s & 0x8000u) << 16);
    return u32_to_f2(__byte_perm(0x42403c00u, 0x4a484644u, sel) | sgn);
}

__device__ __forceinline__ float nvfp4_sub_dot(uint32_t q0, uint32_t q1, float4 x0, float4 x1, float4 x2, float4 x3) {
    const float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    return (w01.x * x0.x + w01.y * x0.y + w23.x * x0.z + w23.y * x0.w) +
           (w45.x * x1.x + w45.y * x1.y + w67.x * x1.z + w67.y * x1.w) +
           (w89.x * x2.x + w89.y * x2.y + wAB.x * x2.z + wAB.y * x2.w) +
           (wCD.x * x3.x + wCD.y * x3.y + wEF.x * x3.z + wEF.y * x3.w);
}

__device__ __forceinline__ float dot_u32_f4(uint32_t q4, float4 f) {
    return (float) (int8_t) (q4 & 0xFFu) * f.x +
           (float) (int8_t) ((q4 >> 8) & 0xFFu) * f.y +
           (float) (int8_t) ((q4 >> 16) & 0xFFu) * f.z +
           (float) (int8_t) (q4 >> 24) * f.w;
}

__device__ __forceinline__ float q8_blk0_dot(const uint32_t* p, float4 f0, float4 f1, float4 f2, float4 f3,
                                             float4 f4, float4 f5, float4 f6, float4 f7, float& d1_out) {
    const uint32_t u0 = __ldg(p + 0), u1 = __ldg(p + 1), u2 = __ldg(p + 2), u3 = __ldg(p + 3), u4 = __ldg(p + 4);
    const uint32_t u5 = __ldg(p + 5), u6 = __ldg(p + 6), u7 = __ldg(p + 7), u8 = __ldg(p + 8);
    const float d0 = half_bits_to_float((uint16_t) (u0 & 0xFFFFu));
    float a0 = (float) (int8_t) ((u0 >> 16) & 0xFFu) * f0.x + (float) (int8_t) (u0 >> 24) * f0.y;
    a0 += dot_u32_f4(u1, make_float4(f0.z, f0.w, f1.x, f1.y));
    a0 += dot_u32_f4(u2, make_float4(f1.z, f1.w, f2.x, f2.y));
    a0 += dot_u32_f4(u3, make_float4(f2.z, f2.w, f3.x, f3.y));
    a0 += dot_u32_f4(u4, make_float4(f3.z, f3.w, f4.x, f4.y));
    a0 += dot_u32_f4(u5, make_float4(f4.z, f4.w, f5.x, f5.y));
    a0 += dot_u32_f4(u6, make_float4(f5.z, f5.w, f6.x, f6.y));
    a0 += dot_u32_f4(u7, make_float4(f6.z, f6.w, f7.x, f7.y));
    a0 += (float) (int8_t) (u8 & 0xFFu) * f7.z + (float) (int8_t) ((u8 >> 8) & 0xFFu) * f7.w;
    d1_out = half_bits_to_float((uint16_t) (u8 >> 16));
    return d0 * a0;
}

__device__ __forceinline__ float q8_blk1_dot(const uint32_t* p, float d1, float4 g0, float4 g1, float4 g2, float4 g3,
                                             float4 g4, float4 g5, float4 g6, float4 g7) {
    return d1 * (dot_u32_f4(__ldg(p + 9), g0) + dot_u32_f4(__ldg(p + 10), g1) + dot_u32_f4(__ldg(p + 11), g2) +
                 dot_u32_f4(__ldg(p + 12), g3) + dot_u32_f4(__ldg(p + 13), g4) + dot_u32_f4(__ldg(p + 14), g5) +
                 dot_u32_f4(__ldg(p + 15), g6) + dot_u32_f4(__ldg(p + 16), g7));
}


// ---- IQ3_XXS / IQ4_XS: one 32-value sub-block against 32 inputs (xv: 8 float4, the padded xs of that sub-block)

__device__ __forceinline__ float sdot4_u8(uint32_t g, uint32_t s, float4 x) {   // bytes of g, sign bit k of s
    return __int_as_float(__float_as_int((float) (g & 0xFFu) * x.x) ^ ((s & 1u) << 31)) +
           __int_as_float(__float_as_int((float) ((g >> 8) & 0xFFu) * x.y) ^ (((s >> 1) & 1u) << 31)) +
           __int_as_float(__float_as_int((float) ((g >> 16) & 0xFFu) * x.z) ^ (((s >> 2) & 1u) << 31)) +
           __int_as_float(__float_as_int((float) (g >> 24) * x.w) ^ (((s >> 3) & 1u) << 31));
}
// block b (98 bytes, 2-aligned), sub-block ib32
__device__ __forceinline__ float iq3xxs_sub_dot(const uint8_t* b, int ib32, const float4 (&xv)[8]) {
    const float d = half_bits_to_float(__ldg((const uint16_t*) b));
    const uint16_t* qs = (const uint16_t*) (b + 2 + 8 * ib32);
    const uint16_t* sp = (const uint16_t*) (b + 66 + 4 * ib32);
    const uint32_t aux = (uint32_t) __ldg(sp) | ((uint32_t) __ldg(sp + 1) << 16);
    float a = 0.0f;
#pragma unroll
    for (int l = 0; l < 4; ++l) {
        const uint32_t qq = __ldg(qs + l);
        const uint32_t g1 = __ldg(d_iq3xxs_grid + (qq & 0xFFu)), g2 = __ldg(d_iq3xxs_grid + (qq >> 8));
        const uint32_t sg = iq_signs8((aux >> (7 * l)) & 127u);
        a += sdot4_u8(g1, sg, xv[2 * l]) + sdot4_u8(g2, sg >> 4, xv[2 * l + 1]);
    }
    return d * (0.5f + (float) (aux >> 28)) * 0.5f * a;
}
// IQ4_NL values of the 4 low (hi = false) or high nibbles of the bytes of w, as 4 int8 in a u32
__device__ __forceinline__ uint32_t iq4nl_lut4(uint32_t w, bool hi) {
    const uint32_t t = (hi ? (w >> 4) : w) & 0x0F0F0F0Fu;
    const uint32_t x = t | (t >> 4);
    const uint32_t sel = (x & 0xFFu) | ((x >> 8) & 0xFF00u);
    const uint32_t T0 = 0xBFAD9881u, T1 = 0xF6EADDCFu, T2 = 0x26190D01u, T3 = 0x71594535u;
    const uint32_t lo = __byte_perm(T0, T1, sel & 0x7777u), hv = __byte_perm(T2, T3, sel & 0x7777u);
    const uint32_t m = ((t >> 3) & 0x01010101u) * 0xFFu;
    return (lo & ~m) | (hv & m);
}
// block b (136 bytes, 4-aligned), sub-block ib
__device__ __forceinline__ float iq4xs_sub_dot(const uint8_t* b, int ib, const float4 (&xv)[8]) {
    const uint32_t h = __ldg((const uint32_t*) b), sl = __ldg((const uint32_t*) (b + 4));
    const float d = half_bits_to_float((uint16_t) (h & 0xFFFFu));
    const int ls = (int) ((sl >> (4 * ib)) & 0xFu) | (int) (((h >> (16 + 2 * ib)) & 3u) << 4);
    const uint32_t* q = (const uint32_t*) (b + 8 + 16 * ib);
    float a = 0.0f;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const uint32_t w = __ldg(q + k);
        a += dot_u32_f4(iq4nl_lut4(w, false), xv[k]) + dot_u32_f4(iq4nl_lut4(w, true), xv[4 + k]);
    }
    return d * (float) (ls - 32) * a;
}
__device__ __forceinline__ void load_xv8(const float* x, float4 (&xv)[8]) {
#pragma unroll
    for (int k = 0; k < 8; ++k) xv[k] = *(const float4*) (x + 4 * k);
}

// a warp's dot of one row of K values (blocks of `type` from w) with xs (shared, padded: xp), every lane gets it
__device__ float row_dot(int type, const uint8_t* w, const float* xs, int K) {
    const int lane = threadIdx.x & 31;
    float acc = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const uint8_t* b = w + (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const float d = ue4m3((uint8_t) (__ldg((const uint32_t*) b) >> (8 * sub)));
            const uint32_t q0 = __ldg((const uint32_t*) (b + 4 + sub * 8));
            const uint32_t q1 = __ldg((const uint32_t*) (b + 8 + sub * 8));
            const float* x = xs + (sb >> 1) * 36 + (sb & 1) * 16;
            const float4 x0 = *(const float4*) (x + 0), x1 = *(const float4*) (x + 4);
            const float4 x2 = *(const float4*) (x + 8), x3 = *(const float4*) (x + 12);
            acc += d * nvfp4_sub_dot(q0, q1, x0, x1, x2, x3);
        }
    } else if (type == kQ8_0) {
        const int nb = K / 32;
        if (((uintptr_t) w & 3) == 0) {
            for (int bi = 2 * lane; bi + 1 < nb; bi += 64) {
                const uint32_t* p = (const uint32_t*) (w + (size_t) bi * 34);
                const float* x0 = xs + bi * 36;
                const float4 f0 = *(const float4*) (x0 + 0), f1 = *(const float4*) (x0 + 4);
                const float4 f2 = *(const float4*) (x0 + 8), f3 = *(const float4*) (x0 + 12);
                const float4 f4 = *(const float4*) (x0 + 16), f5 = *(const float4*) (x0 + 20);
                const float4 f6 = *(const float4*) (x0 + 24), f7 = *(const float4*) (x0 + 28);
                float d1;
                acc += q8_blk0_dot(p, f0, f1, f2, f3, f4, f5, f6, f7, d1);
                const float4 g0 = *(const float4*) (x0 + 36), g1 = *(const float4*) (x0 + 40);
                const float4 g2 = *(const float4*) (x0 + 44), g3 = *(const float4*) (x0 + 48);
                const float4 g4 = *(const float4*) (x0 + 52), g5 = *(const float4*) (x0 + 56);
                const float4 g6 = *(const float4*) (x0 + 60), g7 = *(const float4*) (x0 + 64);
                acc += q8_blk1_dot(p, d1, g0, g1, g2, g3, g4, g5, g6, g7);
            }
            if (nb & 1) {
                const int bi = nb - 1;
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (__ldg(b) | (__ldg(b + 1) << 8)));
                acc += d * (float) (int8_t) __ldg(b + 2 + lane) * xs[xp(bi * 32 + lane)];
            }
        } else {
            for (int bi = lane; bi < nb; bi += 32) {
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (__ldg(b) | (__ldg(b + 1) << 8)));
                const float* x0 = xs + bi * 36;
                float a = 0.0f;
#pragma unroll 8
                for (int j = 0; j < 32; ++j) a += (float) (int8_t) __ldg(b + 2 + j) * x0[j];
                acc += d * a;
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 210;
            const float d = half_bits_to_float((uint16_t) (__ldg(b + 208) | (__ldg(b + 209) << 8)));
            const int8_t* sc = (const int8_t*) (b + 192);
            const int x0 = bi * 256;
            float a = 0.0f;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const uint8_t* ql = b + n * 64;
                const uint8_t qh = __ldg(b + 128 + n * 32 + lane);
                const uint8_t l0 = __ldg(ql + lane), l1 = __ldg(ql + lane + 32);
                const int is = n * 8 + (lane >> 4);
                const int xn = x0 + n * 128 + lane;
                a += (float) __ldg(sc + is + 0) * (float) ((int) ((l0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32) * xs[xp(xn)];
                a += (float) __ldg(sc + is + 2) * (float) ((int) ((l1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32) * xs[xp(xn + 32)];
                a += (float) __ldg(sc + is + 4) * (float) ((int) ((l0 >> 4) | (((qh >> 4) & 3) << 4)) - 32) * xs[xp(xn + 64)];
                a += (float) __ldg(sc + is + 6) * (float) ((int) ((l1 >> 4) | (((qh >> 6) & 3) << 4)) - 32) * xs[xp(xn + 96)];
            }
            acc += d * a;
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        const bool hdr4 = (((uintptr_t) w & 3) == 0);
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 176;
            float d, mn;
            if (hdr4) {
                const uint32_t dm = __ldg((const uint32_t*) b);
                d = half_bits_to_float((uint16_t) (dm & 0xFFFFu));
                mn = half_bits_to_float((uint16_t) (dm >> 16));
            } else {
                d = half_bits_to_float((uint16_t) (__ldg(b) | (__ldg(b + 1) << 8)));
                mn = half_bits_to_float((uint16_t) (__ldg(b + 2) | (__ldg(b + 3) << 8)));
            }
            const uint8_t* q = b + 4;
            const uint8_t qh = __ldg(b + 16 + lane);
            const int x0 = bi * 256 + lane;
            float a = 0.0f;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t lo = __ldg(b + 48 + 32 * j64 + lane);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int is = 2 * j64 + half;
                    int sc, m;
                    if (is < 4) {
                        sc = __ldg(q + is) & 63, m = __ldg(q + is + 4) & 63;
                    } else {
                        const uint8_t q_hi = __ldg(q + is + 4);
                        sc = (q_hi & 0xF) | ((__ldg(q + is - 4) >> 6) << 4);
                        m = (q_hi >> 4) | ((__ldg(q + is) >> 6) << 4);
                    }
                    const int val = (half ? (lo >> 4) : (lo & 0xF)) + (((qh >> is) & 1) ? 16 : 0);
                    a += (d * (float) sc * (float) val - mn * (float) m) * xs[xp(x0 + 64 * j64 + 32 * half)];
                }
            }
            acc += a;
        }
    } else if (type == kIQ3_XXS || type == kIQ4_XS) {
        const bool iq3 = type == kIQ3_XXS;
        for (int sb = lane; sb < K / 32; sb += 32) {
            float4 xv[8];
            load_xv8(xs + sb * 36, xv);
            acc += iq3 ? iq3xxs_sub_dot(w + (size_t) (sb >> 3) * 98, sb & 7, xv)
                       : iq4xs_sub_dot(w + (size_t) (sb >> 3) * 136, sb & 7, xv);
        }
    } else {
        for (int i = lane; i < K; i += 32) acc += decode_value(type, w, (size_t) i) * xs[xp(i)];
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
    return acc;
}

// Single-pass dot of two weight rows (w0, w1) of the same quant type against xs (reading xs once for both rows).
__device__ void row_dot_2(int type, const uint8_t* wg, const uint8_t* wu, const float* xs, int K, float& og, float& ou) {
    const int lane = threadIdx.x & 31;
    float ag = 0.0f, au = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const float* x = xs + (sb >> 1) * 36 + (sb & 1) * 16;
            const float4 x0 = *(const float4*) (x + 0), x1 = *(const float4*) (x + 4);
            const float4 x2 = *(const float4*) (x + 8), x3 = *(const float4*) (x + 12);
            const size_t blk_off = (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const uint8_t* bg = wg + blk_off;
            const float dg = ue4m3((uint8_t) (__ldg((const uint32_t*) bg) >> (8 * sub)));
            const uint32_t gq0 = __ldg((const uint32_t*) (bg + 4 + sub * 8));
            const uint32_t gq1 = __ldg((const uint32_t*) (bg + 8 + sub * 8));
            ag += dg * nvfp4_sub_dot(gq0, gq1, x0, x1, x2, x3);

            const uint8_t* bu = wu + blk_off;
            const float du = ue4m3((uint8_t) (__ldg((const uint32_t*) bu) >> (8 * sub)));
            const uint32_t uq0 = __ldg((const uint32_t*) (bu + 4 + sub * 8));
            const uint32_t uq1 = __ldg((const uint32_t*) (bu + 8 + sub * 8));
            au += du * nvfp4_sub_dot(uq0, uq1, x0, x1, x2, x3);
        }
    } else if (type == kQ8_0) {
        const int nb = K / 32;
        if (((uintptr_t) wg & 3) == 0 && ((uintptr_t) wu & 3) == 0) {
            for (int bi = 2 * lane; bi + 1 < nb; bi += 64) {
                const float* x0 = xs + bi * 36;
                const float4 f0 = *(const float4*) (x0 + 0), f1 = *(const float4*) (x0 + 4);
                const float4 f2 = *(const float4*) (x0 + 8), f3 = *(const float4*) (x0 + 12);
                const float4 f4 = *(const float4*) (x0 + 16), f5 = *(const float4*) (x0 + 20);
                const float4 f6 = *(const float4*) (x0 + 24), f7 = *(const float4*) (x0 + 28);
                const uint32_t* pg = (const uint32_t*) (wg + (size_t) bi * 34);
                const uint32_t* pu = (const uint32_t*) (wu + (size_t) bi * 34);
                float dg1, du1;
                ag += q8_blk0_dot(pg, f0, f1, f2, f3, f4, f5, f6, f7, dg1);
                au += q8_blk0_dot(pu, f0, f1, f2, f3, f4, f5, f6, f7, du1);
                const float4 g0 = *(const float4*) (x0 + 36), g1 = *(const float4*) (x0 + 40);
                const float4 g2 = *(const float4*) (x0 + 44), g3 = *(const float4*) (x0 + 48);
                const float4 g4 = *(const float4*) (x0 + 52), g5 = *(const float4*) (x0 + 56);
                const float4 g6 = *(const float4*) (x0 + 60), g7 = *(const float4*) (x0 + 64);
                ag += q8_blk1_dot(pg, dg1, g0, g1, g2, g3, g4, g5, g6, g7);
                au += q8_blk1_dot(pu, du1, g0, g1, g2, g3, g4, g5, g6, g7);
            }
            if (nb & 1) {
                const int bi = nb - 1;
                const float xv = xs[xp(bi * 32 + lane)];
                const uint8_t* bg = wg + (size_t) bi * 34;
                const uint8_t* bu = wu + (size_t) bi * 34;
                ag += half_bits_to_float((uint16_t) (__ldg(bg) | (__ldg(bg + 1) << 8))) *
                      (float) (int8_t) __ldg(bg + 2 + lane) * xv;
                au += half_bits_to_float((uint16_t) (__ldg(bu) | (__ldg(bu + 1) << 8))) *
                      (float) (int8_t) __ldg(bu + 2 + lane) * xv;
            }
        } else {
            for (int bi = lane; bi < nb; bi += 32) {
                const float* x0 = xs + bi * 36;
                const uint8_t* bg = wg + (size_t) bi * 34;
                const uint8_t* bu = wu + (size_t) bi * 34;
                const float dg = half_bits_to_float((uint16_t) (__ldg(bg) | (__ldg(bg + 1) << 8)));
                const float du = half_bits_to_float((uint16_t) (__ldg(bu) | (__ldg(bu + 1) << 8)));
                float sg = 0.0f, su = 0.0f;
#pragma unroll 8
                for (int j = 0; j < 32; ++j) {
                    const float xv = x0[j];
                    sg += (float) (int8_t) __ldg(bg + 2 + j) * xv;
                    su += (float) (int8_t) __ldg(bu + 2 + j) * xv;
                }
                ag += dg * sg;
                au += du * su;
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 210;
            const uint8_t* bu = wu + (size_t) bi * 210;
            const float dg = half_bits_to_float((uint16_t) (__ldg(bg + 208) | (__ldg(bg + 209) << 8)));
            const float du = half_bits_to_float((uint16_t) (__ldg(bu + 208) | (__ldg(bu + 209) << 8)));
            const int8_t* scg = (const int8_t*) (bg + 192);
            const int8_t* scu = (const int8_t*) (bu + 192);
            const int x0 = bi * 256;
            float sg = 0.0f, su = 0.0f;
#pragma unroll
            for (int n = 0; n < 2; ++n) {
                const int xn = x0 + n * 128 + lane;
                const float xv0 = xs[xp(xn)], xv1 = xs[xp(xn + 32)], xv2 = xs[xp(xn + 64)], xv3 = xs[xp(xn + 96)];
                const int is = n * 8 + (lane >> 4);

                const uint8_t* qlg = bg + n * 64;
                const uint8_t qhg = __ldg(bg + 128 + n * 32 + lane);
                const uint8_t lg0 = __ldg(qlg + lane), lg1 = __ldg(qlg + lane + 32);
                sg += (float) __ldg(scg + is + 0) * (float) ((int) ((lg0 & 0xF) | (((qhg >> 0) & 3) << 4)) - 32) * xv0;
                sg += (float) __ldg(scg + is + 2) * (float) ((int) ((lg1 & 0xF) | (((qhg >> 2) & 3) << 4)) - 32) * xv1;
                sg += (float) __ldg(scg + is + 4) * (float) ((int) ((lg0 >> 4) | (((qhg >> 4) & 3) << 4)) - 32) * xv2;
                sg += (float) __ldg(scg + is + 6) * (float) ((int) ((lg1 >> 4) | (((qhg >> 6) & 3) << 4)) - 32) * xv3;

                const uint8_t* qlu = bu + n * 64;
                const uint8_t qhu = __ldg(bu + 128 + n * 32 + lane);
                const uint8_t lu0 = __ldg(qlu + lane), lu1 = __ldg(qlu + lane + 32);
                su += (float) __ldg(scu + is + 0) * (float) ((int) ((lu0 & 0xF) | (((qhu >> 0) & 3) << 4)) - 32) * xv0;
                su += (float) __ldg(scu + is + 2) * (float) ((int) ((lu1 & 0xF) | (((qhu >> 2) & 3) << 4)) - 32) * xv1;
                su += (float) __ldg(scu + is + 4) * (float) ((int) ((lu0 >> 4) | (((qhu >> 4) & 3) << 4)) - 32) * xv2;
                su += (float) __ldg(scu + is + 6) * (float) ((int) ((lu1 >> 4) | (((qhu >> 6) & 3) << 4)) - 32) * xv3;
            }
            ag += dg * sg;
            au += du * su;
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        const bool hdr4 = (((uintptr_t) wg & 3) == 0) && (((uintptr_t) wu & 3) == 0);
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 176;
            const uint8_t* bu = wu + (size_t) bi * 176;
            float dg, mng, du, mnu;
            if (hdr4) {
                const uint32_t dmg = __ldg((const uint32_t*) bg);
                const uint32_t dmu = __ldg((const uint32_t*) bu);
                dg = half_bits_to_float((uint16_t) (dmg & 0xFFFFu));
                mng = half_bits_to_float((uint16_t) (dmg >> 16));
                du = half_bits_to_float((uint16_t) (dmu & 0xFFFFu));
                mnu = half_bits_to_float((uint16_t) (dmu >> 16));
            } else {
                dg = half_bits_to_float((uint16_t) (__ldg(bg) | (__ldg(bg + 1) << 8)));
                mng = half_bits_to_float((uint16_t) (__ldg(bg + 2) | (__ldg(bg + 3) << 8)));
                du = half_bits_to_float((uint16_t) (__ldg(bu) | (__ldg(bu + 1) << 8)));
                mnu = half_bits_to_float((uint16_t) (__ldg(bu + 2) | (__ldg(bu + 3) << 8)));
            }
            const uint8_t* qg = bg + 4;
            const uint8_t* qu = bu + 4;
            const uint8_t qhg = __ldg(bg + 16 + lane), qhu = __ldg(bu + 16 + lane);
            const int x0 = bi * 256 + lane;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t log_ = __ldg(bg + 48 + 32 * j64 + lane);
                const uint8_t lou = __ldg(bu + 48 + 32 * j64 + lane);
#pragma unroll
                for (int half = 0; half < 2; ++half) {
                    const int is = 2 * j64 + half;
                    const float xv = xs[xp(x0 + 64 * j64 + 32 * half)];
                    int scg, mg, scu, mu;
                    if (is < 4) {
                        scg = __ldg(qg + is) & 63, mg = __ldg(qg + is + 4) & 63;
                        scu = __ldg(qu + is) & 63, mu = __ldg(qu + is + 4) & 63;
                    } else {
                        const uint8_t qgh = __ldg(qg + is + 4), quh = __ldg(qu + is + 4);
                        scg = (qgh & 0xF) | ((__ldg(qg + is - 4) >> 6) << 4), mg = (qgh >> 4) | ((__ldg(qg + is) >> 6) << 4);
                        scu = (quh & 0xF) | ((__ldg(qu + is - 4) >> 6) << 4), mu = (quh >> 4) | ((__ldg(qu + is) >> 6) << 4);
                    }
                    const int vg = (half ? (log_ >> 4) : (log_ & 0xF)) + (((qhg >> is) & 1) ? 16 : 0);
                    const int vu = (half ? (lou >> 4) : (lou & 0xF)) + (((qhu >> is) & 1) ? 16 : 0);
                    ag += (dg * (float) scg * (float) vg - mng * (float) mg) * xv;
                    au += (du * (float) scu * (float) vu - mnu * (float) mu) * xv;
                }
            }
        }
    } else if (type == kIQ3_XXS || type == kIQ4_XS) {
        const bool iq3 = type == kIQ3_XXS;
        const size_t bb = iq3 ? 98 : 136;
        for (int sb = lane; sb < K / 32; sb += 32) {
            float4 xv[8];
            load_xv8(xs + sb * 36, xv);
            const size_t off = (size_t) (sb >> 3) * bb;
            ag += iq3 ? iq3xxs_sub_dot(wg + off, sb & 7, xv) : iq4xs_sub_dot(wg + off, sb & 7, xv);
            au += iq3 ? iq3xxs_sub_dot(wu + off, sb & 7, xv) : iq4xs_sub_dot(wu + off, sb & 7, xv);
        }
    } else {
        for (int i = lane; i < K; i += 32) {
            const float xv = xs[xp(i)];
            ag += decode_value(type, wg, (size_t) i) * xv;
            au += decode_value(type, wu, (size_t) i) * xv;
        }
    }
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ag += __shfl_xor_sync(0xffffffffu, ag, o);
        au += __shfl_xor_sync(0xffffffffu, au, o);
    }
    og = ag;
    ou = au;
}
__device__ __forceinline__ void row_dot_pair(int type, const uint8_t* wg, const uint8_t* wu, const float* xs, int K,
                                             float& og, float& ou) {
    row_dot_2(type, wg, wu, xs, K, og, ou);
}

__device__ __forceinline__ void load_xs_f32(const float* x, int K, float* xs) {
    if (((uintptr_t) x & 15) == 0 && (K & 3) == 0) {
        const int n4 = K >> 2;
        const float4* x4 = (const float4*) x;
        for (int i4 = threadIdx.x; i4 < n4; i4 += blockDim.x)
            *(float4*) (xs + xp(i4 << 2)) = __ldg(x4 + i4);
    } else {
        for (int i = threadIdx.x; i < K; i += blockDim.x) xs[xp(i)] = x[i];
    }
}

__device__ __forceinline__ void load_xs_f16(const uint16_t* h, int K, float* xs) {
    if (((uintptr_t) h & 7) == 0 && (K & 3) == 0) {
        const int n4 = K >> 2;
        const uint2* h4 = (const uint2*) h;
        for (int i4 = threadIdx.x; i4 < n4; i4 += blockDim.x) {
            const uint2 u = __ldg(h4 + i4);
            const float2 f0 = u32_to_f2(u.x), f1 = u32_to_f2(u.y);
            *(float4*) (xs + xp(i4 << 2)) = make_float4(f0.x, f0.y, f1.x, f1.y);
        }
    } else {
        for (int i = threadIdx.x; i < K; i += blockDim.x) xs[xp(i)] = h2f(h[i]);
    }
}

size_t mv_row_bytes(int type, int K) {
    return (size_t) (K / block_values(type)) * block_bytes(type);
}

// ---- one token's NVFP4 experts through the tensor cores, straight from their blocks into the registers.
// A warp is 16 rows (gate / up: 8 gate rows and the same 8 of up, so that SwiGLU is a lane's; down: 16 rows) over
// the whole K, no shared memory.  Lane (g, t): sub-block t of rows g and g + 8 of each 64-value block - four MMA a
// block, MMA m the sub-block's bytes 2 m and 2 m + 1 (low nibbles values 2 m, 2 m + 1 in k slots 2 t, 2 t + 1; high
// nibbles values 2 m + 8, 2 m + 9 in slots 2 t + 8, 2 t + 9).  B: the input in FP16 where it is, the same in all
// eight columns.  A code times its sub-block's scale is exact in FP16 (a code is at most 12, ue4m3 at most 240 and
// at least 2^-10); the sum is the tensor cores' FP32.
__device__ __forceinline__ uint32_t ue4m3_h2(uint32_t x) {   // FP16 bits of ue4m3(x) in both halves
    const uint32_t e = (x >> 3) & 15u, m = x & 7u;
    const uint32_t sub = __byte_perm(0x1a181400u, 0x1f1e1d1cu, m) << 8;   // m 2^-10
    uint32_t h = e ? (((e + 7u) << 10) | (m << 7)) : sub;
    h = (x == 0x7Fu) ? 0u : h;
    return h | (h << 16);
}
__device__ __forceinline__ uint32_t nv_pair_h2(uint32_t s, uint32_t scale) {   // bytes 0 and 1's low nibbles
    const uint32_t mag = __byte_perm(0x42403c00u, 0x4a484644u, (s & 0x0707u) << 4);
    const uint32_t sign = ((s & 0x0808u) * 0x101000u) & 0x80008000u;
    uint32_t r;
#if defined(__HIPCC__)
    // mul.rn.f16x2 is round-to-nearest-even packed-half multiply; HIP __hmul2 matches.
    const uint32_t ms = mag | sign;
    const __half2 m = *reinterpret_cast<const __half2*>(&ms);
    const __half2 sc = *reinterpret_cast<const __half2*>(&scale);
    const __half2 pr = __hmul2(m, sc);
    r = *reinterpret_cast<const uint32_t*>(&pr);
#else
    asm("mul.rn.f16x2 %0, %1, %2;" : "=r"(r) : "r"(mag | sign), "r"(scale));
#endif
    return r;
}
struct NvStep {   // a lane's part of a 64-value block of its two rows: the sub-block's scales and codes
    uint32_t sa, sb, wa0, wa1, wb0, wb1;
};
__device__ __forceinline__ NvStep nv_step_load(const uint8_t* a, const uint8_t* b, int k, int t) {
    const uint32_t* ap = (const uint32_t*) (a + (size_t) k * 36);
    const uint32_t* bp = (const uint32_t*) (b + (size_t) k * 36);
    NvStep r;
    r.sa = __ldg(ap), r.sb = __ldg(bp);
    r.wa0 = __ldg(ap + 1 + 2 * t), r.wa1 = __ldg(ap + 2 + 2 * t);
    r.wb0 = __ldg(bp + 1 + 2 * t), r.wb1 = __ldg(bp + 2 + 2 * t);
    return r;
}
// A block is a group of 16 rows, its SPLIT warps each a stride of the K blocks (more rows of weights in flight than
// a warp alone keeps: a warp waits for a block's bytes before it asks for more than the next one's).
template <bool Gate, int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_mma_kernel(const uint8_t* const* __restrict__ blob, size_t off, int C, int F, const uint16_t* __restrict__ x16,
                   int x_ld, uint16_t* hidden, float* y, const float* sg, const float* su, float gcap, float ucap,
                   const int* __restrict__ tok = nullptr) {
    const int slot = blockIdx.y;
    const uint8_t* bslot = blob[slot];
    if (!bslot) return;
    __shared__ float2 part[SPLIT][8];
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5, g = lane >> 2, t = lane & 3;
    const int K = Gate ? C : F;
    const int r = blockIdx.x * (Gate ? 8 : 16) + g;
    const size_t rb = (size_t) (K / 64) * 36;
    const uint8_t* a = blob[slot] + (Gate ? 0 : off) + (size_t) r * rb;
    const uint8_t* b = Gate ? a + off : a + 8 * rb;
    // (gate / up: the input as two FP16 limbs, x = hi + lo, [2][C] - column 0 takes hi, column 1 lo, and a row's
    //  two columns add up to the product with the FP32 input; down: h is FP16 itself, every column the same)
    const int tok_idx = (Gate && tok) ? tok[slot] : 0;
    const uint4* xp = (const uint4*) (x16 + (Gate ? ((size_t) tok_idx * 2 * C + (g == 1 ? (size_t) C : 0)) : (size_t) slot * x_ld)) + 2 * t;
    float v[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    const int nk = K / 64;
    NvStep cur = nv_step_load(a, b, split, t);
    for (int k = split; k < nk; k += SPLIT) {
        NvStep nxt = cur;
        if (k + SPLIT < nk) nxt = nv_step_load(a, b, k + SPLIT, t);
        const uint32_t sa = ue4m3_h2((cur.sa >> (8 * t)) & 0xFFu), sb = ue4m3_h2((cur.sb >> (8 * t)) & 0xFFu);
        const uint32_t wa[2] = {cur.wa0, cur.wa1}, wb[2] = {cur.wb0, cur.wb1};
        const uint4 x0 = __ldg(xp + k * 8), x1 = __ldg(xp + k * 8 + 1);
        const uint32_t xx[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
        for (int m = 0; m < 4; ++m) {
            const uint32_t qa = wa[m >> 1] >> (16 * (m & 1)), qb = wb[m >> 1] >> (16 * (m & 1));
            const uint32_t af[4] = {nv_pair_h2(qa, sa), nv_pair_h2(qb, sb), nv_pair_h2(qa >> 4, sa),
                                    nv_pair_h2(qb >> 4, sb)};
            mma16816(v, af, xx[m], xx[4 + m]);
        }
        cur = nxt;
    }
    if (t == 0) part[split][g] = Gate ? make_float2(v[0] + v[1], v[2] + v[3]) : make_float2(v[0], v[2]);
    __syncthreads();
    if (split || t) return;
    float2 tot = part[0][g];
#pragma unroll
    for (int i = 1; i < SPLIT; ++i) tot.x += part[i][g].x, tot.y += part[i][g].y;
    if (Gate) {
        const float gg = tot.x * (sg ? sg[slot] : 1.0f), u = tot.y * (su ? su[slot] : 1.0f);
        hidden[(size_t) slot * F + r] =
            f2h(gcap * tanhf(gg / gcap) * (1.0f / (1.0f + expf(-gg))) * ucap * tanhf(u / ucap));
    } else {
        y[(size_t) slot * C + r] = tot.x;
        y[(size_t) slot * C + r + 8] = tot.y;
    }
}

template <int WARPS, int ROWS>
__global__ void experts_gate_up_kernel(const uint8_t* const* blob, const int* tok, int ty_gate, int ty_up, size_t eb0,
                                       const float* x, int C, int F, size_t rb_gate, size_t rb_up, float* gu) {
    const int s = blockIdx.y;
    const uint8_t* bl = blob[s];
    if (!bl) return;
    extern __shared__ float xs[];
    load_xs_f32(x + (size_t) tok[s] * C, C, xs);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    for (int j = 0; j < ROWS; ++j) {
        const int row = (blockIdx.x * WARPS + warp) * ROWS + j;   // of 2 F: gate's then up's
        if (row >= 2 * F) break;
        const bool up = row >= F;
        const int r = up ? row - F : row;
        const float v = row_dot(up ? ty_up : ty_gate, bl + (up ? eb0 + r * rb_up : r * rb_gate), xs, C);
        if ((threadIdx.x & 31) == 0) gu[(size_t) s * 2 * F + row] = v;
    }
}

template <int WARPS, int ROWS>
__global__ void experts_gate_up_swiglu_kernel(const uint8_t* const* blob, const int* tok, int ty_gate, int ty_up,
                                              size_t eb0, const float* x, int C, int F, size_t rb_gate, size_t rb_up,
                                              const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                                              const int* orow, int h_ld) {
    const int s = blockIdx.y;
    const uint8_t* bl = blob[s];
    if (!bl) return;
    extern __shared__ float xs[];
    load_xs_f32(x + (size_t) tok[s] * C, C, xs);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const float sgs = sg ? sg[s] : 1.0f;
    const float sus = su ? su[s] : 1.0f;
    for (int j = 0; j < ROWS; ++j) {
        const int r = (blockIdx.x * WARPS + warp) * ROWS + j;
        if (r >= F) break;
        float vg, vu;
        if (ty_gate == ty_up) {
            row_dot_pair(ty_gate, bl + (size_t) r * rb_gate, bl + eb0 + (size_t) r * rb_up, xs, C, vg, vu);
        } else {
            vg = row_dot(ty_gate, bl + (size_t) r * rb_gate, xs, C);
            vu = row_dot(ty_up, bl + eb0 + (size_t) r * rb_up, xs, C);
        }
        if ((threadIdx.x & 31) == 0) {
            const float g = vg * sgs, u = vu * sus;
            h[(size_t) (orow ? orow[s] : s) * h_ld + r] =
                f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
        }
    }
}

template <int WARPS, int ROWS>
__global__ void experts_down_kernel(const uint8_t* const* blob, int ty_down, size_t off_down, const uint16_t* h,
                                    int C, int F, size_t rb_down, float* y) {
    const int s = blockIdx.y;
    const uint8_t* bl = blob[s];
    if (!bl) return;
    extern __shared__ float xs[];
    load_xs_f16(h + (size_t) s * F, F, xs);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    const uint8_t* w = bl + off_down;
    for (int j = 0; j < ROWS;) {
        const int row = (blockIdx.x * WARPS + warp) * ROWS + j;
        if (row >= C) break;
        if (j + 1 < ROWS && row + 1 < C) {
            float v0, v1;
            row_dot_2(ty_down, w + (size_t) row * rb_down, w + (size_t) (row + 1) * rb_down, xs, F, v0, v1);
            if ((threadIdx.x & 31) == 0) {
                y[(size_t) s * C + row] = v0;
                y[(size_t) s * C + row + 1] = v1;
            }
            j += 2;
        } else {
            const float v = row_dot(ty_down, w + (size_t) row * rb_down, xs, F);
            if ((threadIdx.x & 31) == 0) y[(size_t) s * C + row] = v;
            j += 1;
        }
    }
}

// ---- grouped: an expert's rows read once for every token that picked it (a verify window's few tokens)

// a warp's dots of one row (blocks of `type`) with n <= N_CAP inputs (each K values, F32 or FP16 rows in global memory)
template <class X>
__device__ __forceinline__ float xval(const X* p, int i);
template <>
__device__ __forceinline__ float xval<float>(const float* p, int i) { return __ldg(p + i); }
template <>
__device__ __forceinline__ float xval<uint16_t>(const uint16_t* p, int i) { return h2f(__ldg(p + i)); }

template <class X>
__device__ __forceinline__ float dot16_vec(const X* p, const float* wv);

template <>
__device__ __forceinline__ float dot16_vec<float>(const float* p, const float* wv) {
    if (((uintptr_t) p & 15) == 0) {
        const float4* p4 = (const float4*) p;
        const float4 v0 = __ldg(p4 + 0), v1 = __ldg(p4 + 1), v2 = __ldg(p4 + 2), v3 = __ldg(p4 + 3);
        return wv[0] * v0.x + wv[1] * v0.y + wv[2] * v0.z + wv[3] * v0.w +
               wv[4] * v1.x + wv[5] * v1.y + wv[6] * v1.z + wv[7] * v1.w +
               wv[8] * v2.x + wv[9] * v2.y + wv[10] * v2.z + wv[11] * v2.w +
               wv[12] * v3.x + wv[13] * v3.y + wv[14] * v3.z + wv[15] * v3.w;
    }
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a += wv[i] * __ldg(p + i);
    return a;
}

template <>
__device__ __forceinline__ float dot16_vec<uint16_t>(const uint16_t* p, const float* wv) {
    if (((uintptr_t) p & 15) == 0) {
        const uint4* p4 = (const uint4*) p;
        const uint4 u0 = __ldg(p4 + 0), u1 = __ldg(p4 + 1);
        const float2 f0 = u32_to_f2(u0.x), f1 = u32_to_f2(u0.y), f2 = u32_to_f2(u0.z), f3 = u32_to_f2(u0.w);
        const float2 f4 = u32_to_f2(u1.x), f5 = u32_to_f2(u1.y), f6 = u32_to_f2(u1.z), f7 = u32_to_f2(u1.w);
        return wv[0] * f0.x + wv[1] * f0.y + wv[2] * f1.x + wv[3] * f1.y +
               wv[4] * f2.x + wv[5] * f2.y + wv[6] * f3.x + wv[7] * f3.y +
               wv[8] * f4.x + wv[9] * f4.y + wv[10] * f5.x + wv[11] * f5.y +
               wv[12] * f6.x + wv[13] * f6.y + wv[14] * f7.x + wv[15] * f7.y;
    }
    float a = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i) a += wv[i] * h2f(__ldg(p + i));
    return a;
}

__device__ __forceinline__ void decode_nvfp4_wv(uint32_t q0, uint32_t q1, float* wv) {
    const float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    wv[0] = w01.x, wv[1] = w01.y, wv[2] = w23.x, wv[3] = w23.y;
    wv[4] = w45.x, wv[5] = w45.y, wv[6] = w67.x, wv[7] = w67.y;
    wv[8] = w89.x, wv[9] = w89.y, wv[10] = wAB.x, wv[11] = wAB.y;
    wv[12] = wCD.x, wv[13] = wCD.y, wv[14] = wEF.x, wv[15] = wEF.y;
}

__device__ __forceinline__ void unpack_u32_i8(uint32_t q4, float* wv) {
    wv[0] = (float) (int8_t) (q4 & 0xFFu);
    wv[1] = (float) (int8_t) ((q4 >> 8) & 0xFFu);
    wv[2] = (float) (int8_t) ((q4 >> 16) & 0xFFu);
    wv[3] = (float) (int8_t) (q4 >> 24);
}

template <class X, int N_CAP>
__device__ void row_dot_multi(int type, const uint8_t* w, const X* const* xs, int n, int K, float* out) {
    const int lane = threadIdx.x & 31;
    float acc[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) acc[t] = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const uint8_t* b = w + (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const float d = ue4m3((uint8_t) (*(const uint32_t*) b >> (8 * sub)));
            const uint32_t q0 = *(const uint32_t*) (b + 4 + sub * 8), q1 = *(const uint32_t*) (b + 8 + sub * 8);
            float wv[16];
            decode_nvfp4_wv(q0, q1, wv);
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                acc[t] += d * dot16_vec(xs[t] + sb * 16, wv);
            }
        }
    } else if (type == kQ8_0) {
        const int nb = K / 32;
        if (((uintptr_t) w & 3) == 0) {
            for (int bi = 2 * lane; bi + 1 < nb; bi += 64) {
                const uint32_t* p = (const uint32_t*) (w + (size_t) bi * 34);
                const uint32_t u0 = p[0], u1 = p[1], u2 = p[2], u3 = p[3], u4 = p[4];
                const uint32_t u5 = p[5], u6 = p[6], u7 = p[7], u8 = p[8];
                const float d0 = half_bits_to_float((uint16_t) (u0 & 0xFFFFu));
                float a0[N_CAP];
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) a0[t] = 0.0f;
                {
                    float wv[16];
                    wv[0] = (float) (int8_t) ((u0 >> 16) & 0xFFu);
                    wv[1] = (float) (int8_t) (u0 >> 24);
                    unpack_u32_i8(u1, wv + 2);
                    unpack_u32_i8(u2, wv + 6);
                    unpack_u32_i8(u3, wv + 10);
                    wv[14] = (float) (int8_t) (u4 & 0xFFu);
                    wv[15] = (float) (int8_t) ((u4 >> 8) & 0xFFu);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a0[t] += dot16_vec(xs[t] + bi * 32, wv);
                    }
                    wv[0] = (float) (int8_t) ((u4 >> 16) & 0xFFu);
                    wv[1] = (float) (int8_t) (u4 >> 24);
                    unpack_u32_i8(u5, wv + 2);
                    unpack_u32_i8(u6, wv + 6);
                    unpack_u32_i8(u7, wv + 10);
                    wv[14] = (float) (int8_t) (u8 & 0xFFu);
                    wv[15] = (float) (int8_t) ((u8 >> 8) & 0xFFu);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a0[t] += dot16_vec(xs[t] + bi * 32 + 16, wv);
                        acc[t] += d0 * a0[t];
                    }
                }
                const float d1 = half_bits_to_float((uint16_t) (u8 >> 16));
                const uint32_t v0 = p[9], v1 = p[10], v2 = p[11], v3 = p[12];
                const uint32_t v4 = p[13], v5 = p[14], v6 = p[15], v7 = p[16];
                float a1[N_CAP];
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) a1[t] = 0.0f;
                {
                    float wv[16];
                    unpack_u32_i8(v0, wv);
                    unpack_u32_i8(v1, wv + 4);
                    unpack_u32_i8(v2, wv + 8);
                    unpack_u32_i8(v3, wv + 12);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a1[t] += dot16_vec(xs[t] + bi * 32 + 32, wv);
                    }
                    unpack_u32_i8(v4, wv);
                    unpack_u32_i8(v5, wv + 4);
                    unpack_u32_i8(v6, wv + 8);
                    unpack_u32_i8(v7, wv + 12);
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        a1[t] += dot16_vec(xs[t] + bi * 32 + 48, wv);
                        acc[t] += d1 * a1[t];
                    }
                }
            }
            if (nb & 1) {
                const int bi = nb - 1;
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
                const float wv = d * (float) (int8_t) b[2 + lane];
                const int off = bi * 32 + lane;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    acc[t] += wv * xval(xs[t], off);
                }
            }
        } else {
            for (int bi = lane; bi < nb; bi += 32) {
                const uint8_t* b = w + (size_t) bi * 34;
                const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
                for (int h = 0; h < 2; ++h) {
                    float wv[16];
#pragma unroll
                    for (int j = 0; j < 16; ++j) wv[j] = (float) (int8_t) b[2 + h * 16 + j];
#pragma unroll
                    for (int t = 0; t < N_CAP; ++t) {
                        if (t >= n) break;
                        acc[t] += d * dot16_vec(xs[t] + bi * 32 + h * 16, wv);
                    }
                }
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 210;
            const float d = half_bits_to_float((uint16_t) (b[208] | (b[209] << 8)));
            const int8_t* sc = (const int8_t*) (b + 192);
            const int x0 = bi * 256;
            float a[N_CAP];
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) a[t] = 0.0f;
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                const uint8_t* ql = b + nn * 64;
                const uint8_t qh = b[128 + nn * 32 + lane];
                const uint8_t l0 = ql[lane], l1 = ql[lane + 32];
                const int is = nn * 8 + (lane >> 4);
                const float w0 = (float) sc[is + 0] * (float) ((int) ((l0 & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
                const float w1 = (float) sc[is + 2] * (float) ((int) ((l1 & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
                const float w2 = (float) sc[is + 4] * (float) ((int) ((l0 >> 4) | (((qh >> 4) & 3) << 4)) - 32);
                const float w3 = (float) sc[is + 6] * (float) ((int) ((l1 >> 4) | (((qh >> 6) & 3) << 4)) - 32);
                const int xn = x0 + nn * 128 + lane;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    a[t] += w0 * xval(xs[t], xn) + w1 * xval(xs[t], xn + 32) +
                            w2 * xval(xs[t], xn + 64) + w3 * xval(xs[t], xn + 96);
                }
            }
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                acc[t] += d * a[t];
            }
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* b = w + (size_t) bi * 176;
            const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
            const float mn = half_bits_to_float((uint16_t) (b[2] | (b[3] << 8)));
            const uint8_t* q = b + 4;
            const uint8_t qh = b[16 + lane];
            const int x0 = bi * 256 + lane;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t lo = b[48 + 32 * j64 + lane];
                const int is0 = 2 * j64, is1 = is0 + 1;
                int sc0, m0, sc1, m1;
                if (is0 < 4) {
                    sc0 = q[is0] & 63, m0 = q[is0 + 4] & 63;
                    sc1 = q[is1] & 63, m1 = q[is1 + 4] & 63;
                } else {
                    sc0 = (q[is0 + 4] & 0xF) | ((q[is0 - 4] >> 6) << 4), m0 = (q[is0 + 4] >> 4) | ((q[is0] >> 6) << 4);
                    sc1 = (q[is1 + 4] & 0xF) | ((q[is1 - 4] >> 6) << 4), m1 = (q[is1 + 4] >> 4) | ((q[is1] >> 6) << 4);
                }
                const float w0 = d * (float) sc0 * (float) ((lo & 0xF) + (((qh >> is0) & 1) ? 16 : 0)) - mn * (float) m0;
                const float w1 = d * (float) sc1 * (float) ((lo >> 4) + (((qh >> is1) & 1) ? 16 : 0)) - mn * (float) m1;
                const int xp0 = x0 + 64 * j64;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    acc[t] += w0 * xval(xs[t], xp0) + w1 * xval(xs[t], xp0 + 32);
                }
            }
        }
    } else {
        for (int i = lane; i < K; i += 32) {
            const float wv = decode_value(type, w, (size_t) i);
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                acc[t] += wv * xval(xs[t], i);
            }
        }
    }
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) {
        float a = acc[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        out[t] = a;
    }
}

// Single-pass grouped (gate, up) dot of the same quant type against xs[0..n-1] (reading each xs[t] once)
template <int N_CAP>
__device__ void row_dot_pair_multi(int type, const uint8_t* wg, const uint8_t* wu, const float* const* xs, int n,
                                   int K, float* og, float* ou) {
    const int lane = threadIdx.x & 31;
    float ag[N_CAP], au[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) ag[t] = 0.0f, au[t] = 0.0f;
    if (type == kNVFP4) {
        const int nsub = K / 16;
        for (int sb = lane; sb < nsub; sb += 32) {
            const size_t blk_off = (size_t) (sb >> 2) * 36;
            const int sub = sb & 3;
            const uint8_t* bg = wg + blk_off;
            const uint8_t* bu = wu + blk_off;
            const float dg = ue4m3((uint8_t) (*(const uint32_t*) bg >> (8 * sub)));
            const float du = ue4m3((uint8_t) (*(const uint32_t*) bu >> (8 * sub)));
            float wvg[16], wvu[16];
            decode_nvfp4_wv(*(const uint32_t*) (bg + 4 + sub * 8), *(const uint32_t*) (bg + 8 + sub * 8), wvg);
            decode_nvfp4_wv(*(const uint32_t*) (bu + 4 + sub * 8), *(const uint32_t*) (bu + 8 + sub * 8), wvu);
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                const float* xp_t = xs[t] + sb * 16;
                if (((uintptr_t) xp_t & 15) == 0) {
                    const float4* p4 = (const float4*) xp_t;
                    const float4 v0 = __ldg(p4 + 0), v1 = __ldg(p4 + 1), v2 = __ldg(p4 + 2), v3 = __ldg(p4 + 3);
                    ag[t] += dg * (wvg[0] * v0.x + wvg[1] * v0.y + wvg[2] * v0.z + wvg[3] * v0.w +
                                   wvg[4] * v1.x + wvg[5] * v1.y + wvg[6] * v1.z + wvg[7] * v1.w +
                                   wvg[8] * v2.x + wvg[9] * v2.y + wvg[10] * v2.z + wvg[11] * v2.w +
                                   wvg[12] * v3.x + wvg[13] * v3.y + wvg[14] * v3.z + wvg[15] * v3.w);
                    au[t] += du * (wvu[0] * v0.x + wvu[1] * v0.y + wvu[2] * v0.z + wvu[3] * v0.w +
                                   wvu[4] * v1.x + wvu[5] * v1.y + wvu[6] * v1.z + wvu[7] * v1.w +
                                   wvu[8] * v2.x + wvu[9] * v2.y + wvu[10] * v2.z + wvu[11] * v2.w +
                                   wvu[12] * v3.x + wvu[13] * v3.y + wvu[14] * v3.z + wvu[15] * v3.w);
                } else {
                    float sg = 0.0f, su = 0.0f;
#pragma unroll
                    for (int i = 0; i < 16; ++i) {
                        const float xv = __ldg(xp_t + i);
                        sg += wvg[i] * xv;
                        su += wvu[i] * xv;
                    }
                    ag[t] += dg * sg;
                    au[t] += du * su;
                }
            }
        }
    } else if (type == kQ6_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 210;
            const uint8_t* bu = wu + (size_t) bi * 210;
            const float dg = half_bits_to_float((uint16_t) (bg[208] | (bg[209] << 8)));
            const float du = half_bits_to_float((uint16_t) (bu[208] | (bu[209] << 8)));
            const int8_t* scg = (const int8_t*) (bg + 192);
            const int8_t* scu = (const int8_t*) (bu + 192);
            const int x0 = bi * 256;
            float sg[N_CAP], su[N_CAP];
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) sg[t] = 0.0f, su[t] = 0.0f;
#pragma unroll
            for (int nn = 0; nn < 2; ++nn) {
                const int is = nn * 8 + (lane >> 4);
                const uint8_t* qlg = bg + nn * 64;
                const uint8_t qhg = bg[128 + nn * 32 + lane];
                const uint8_t lg0 = qlg[lane], lg1 = qlg[lane + 32];
                const float wg0 = (float) scg[is + 0] * (float) ((int) ((lg0 & 0xF) | (((qhg >> 0) & 3) << 4)) - 32);
                const float wg1 = (float) scg[is + 2] * (float) ((int) ((lg1 & 0xF) | (((qhg >> 2) & 3) << 4)) - 32);
                const float wg2 = (float) scg[is + 4] * (float) ((int) ((lg0 >> 4) | (((qhg >> 4) & 3) << 4)) - 32);
                const float wg3 = (float) scg[is + 6] * (float) ((int) ((lg1 >> 4) | (((qhg >> 6) & 3) << 4)) - 32);

                const uint8_t* qlu = bu + nn * 64;
                const uint8_t qhu = bu[128 + nn * 32 + lane];
                const uint8_t lu0 = qlu[lane], lu1 = qlu[lane + 32];
                const float wu0 = (float) scu[is + 0] * (float) ((int) ((lu0 & 0xF) | (((qhu >> 0) & 3) << 4)) - 32);
                const float wu1 = (float) scu[is + 2] * (float) ((int) ((lu1 & 0xF) | (((qhu >> 2) & 3) << 4)) - 32);
                const float wu2 = (float) scu[is + 4] * (float) ((int) ((lu0 >> 4) | (((qhu >> 4) & 3) << 4)) - 32);
                const float wu3 = (float) scu[is + 6] * (float) ((int) ((lu1 >> 4) | (((qhu >> 6) & 3) << 4)) - 32);

                const int xn = x0 + nn * 128 + lane;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    const float xv0 = __ldg(xs[t] + xn), xv1 = __ldg(xs[t] + xn + 32);
                    const float xv2 = __ldg(xs[t] + xn + 64), xv3 = __ldg(xs[t] + xn + 96);
                    sg[t] += wg0 * xv0 + wg1 * xv1 + wg2 * xv2 + wg3 * xv3;
                    su[t] += wu0 * xv0 + wu1 * xv1 + wu2 * xv2 + wu3 * xv3;
                }
            }
#pragma unroll
            for (int t = 0; t < N_CAP; ++t) {
                if (t >= n) break;
                ag[t] += dg * sg[t];
                au[t] += du * su[t];
            }
        }
    } else if (type == kQ5_K) {
        const int nb = K / 256;
        for (int bi = 0; bi < nb; ++bi) {
            const uint8_t* bg = wg + (size_t) bi * 176;
            const uint8_t* bu = wu + (size_t) bi * 176;
            const float dg = half_bits_to_float((uint16_t) (bg[0] | (bg[1] << 8)));
            const float mng = half_bits_to_float((uint16_t) (bg[2] | (bg[3] << 8)));
            const float du = half_bits_to_float((uint16_t) (bu[0] | (bu[1] << 8)));
            const float mnu = half_bits_to_float((uint16_t) (bu[2] | (bu[3] << 8)));
            const uint8_t* qg = bg + 4;
            const uint8_t* qu = bu + 4;
            const uint8_t qhg = bg[16 + lane], qhu = bu[16 + lane];
            const int x0 = bi * 256 + lane;
#pragma unroll
            for (int j64 = 0; j64 < 4; ++j64) {
                const uint8_t log_ = bg[48 + 32 * j64 + lane], lou = bu[48 + 32 * j64 + lane];
                const int is0 = 2 * j64, is1 = is0 + 1;
                int scg0, mg0, scg1, mg1, scu0, mu0, scu1, mu1;
                if (is0 < 4) {
                    scg0 = qg[is0] & 63, mg0 = qg[is0 + 4] & 63, scg1 = qg[is1] & 63, mg1 = qg[is1 + 4] & 63;
                    scu0 = qu[is0] & 63, mu0 = qu[is0 + 4] & 63, scu1 = qu[is1] & 63, mu1 = qu[is1 + 4] & 63;
                } else {
                    scg0 = (qg[is0 + 4] & 0xF) | ((qg[is0 - 4] >> 6) << 4), mg0 = (qg[is0 + 4] >> 4) | ((qg[is0] >> 6) << 4);
                    scg1 = (qg[is1 + 4] & 0xF) | ((qg[is1 - 4] >> 6) << 4), mg1 = (qg[is1 + 4] >> 4) | ((qg[is1] >> 6) << 4);
                    scu0 = (qu[is0 + 4] & 0xF) | ((qu[is0 - 4] >> 6) << 4), mu0 = (qu[is0 + 4] >> 4) | ((qu[is0] >> 6) << 4);
                    scu1 = (qu[is1 + 4] & 0xF) | ((qu[is1 - 4] >> 6) << 4), mu1 = (qu[is1 + 4] >> 4) | ((qu[is1] >> 6) << 4);
                }
                const float wg0 = dg * (float) scg0 * (float) ((log_ & 0xF) + (((qhg >> is0) & 1) ? 16 : 0)) - mng * (float) mg0;
                const float wg1 = dg * (float) scg1 * (float) ((log_ >> 4) + (((qhg >> is1) & 1) ? 16 : 0)) - mng * (float) mg1;
                const float wu0 = du * (float) scu0 * (float) ((lou & 0xF) + (((qhu >> is0) & 1) ? 16 : 0)) - mnu * (float) mu0;
                const float wu1 = du * (float) scu1 * (float) ((lou >> 4) + (((qhu >> is1) & 1) ? 16 : 0)) - mnu * (float) mu1;
                const int xp0 = x0 + 64 * j64;
#pragma unroll
                for (int t = 0; t < N_CAP; ++t) {
                    if (t >= n) break;
                    const float xv0 = __ldg(xs[t] + xp0), xv1 = __ldg(xs[t] + xp0 + 32);
                    ag[t] += wg0 * xv0 + wg1 * xv1;
                    au[t] += wu0 * xv0 + wu1 * xv1;
                }
            }
        }
    } else {
        row_dot_multi<float, N_CAP>(type, wg, xs, n, K, og);
        row_dot_multi<float, N_CAP>(type, wu, xs, n, K, ou);
        return;
    }
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) {
        float g_a = ag[t], u_a = au[t];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            g_a += __shfl_xor_sync(0xffffffffu, g_a, o);
            u_a += __shfl_xor_sync(0xffffffffu, u_a, o);
        }
        og[t] = g_a;
        ou[t] = u_a;
    }
}

constexpr int kGroupWarps = 4, kGroupRows = 4, kGroupSwigluRows = 2;

template <int N_CAP>
__device__ __forceinline__ void experts_gate_up_g_run(const uint8_t* bl, const int* tok_g, const int* orow_g,
                                                      int ng, int ty_gate, int ty_up, size_t eb0, const float* x,
                                                      int C, int F, size_t rb_gate, size_t rb_up, float* gu) {
    const float* xs[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) xs[t] = x + (size_t) tok_g[t < ng ? t : 0] * C;
    const int warp = threadIdx.x >> 5;
    float out[N_CAP];
    for (int j = 0; j < kGroupRows; ++j) {
        const int row = (blockIdx.x * kGroupWarps + warp) * kGroupRows + j;
        if (row >= 2 * F) break;
        const bool up = row >= F;
        const int r = up ? row - F : row;
        row_dot_multi<float, N_CAP>(up ? ty_up : ty_gate, bl + (up ? eb0 + r * rb_up : r * rb_gate), xs, ng, C, out);
        if ((threadIdx.x & 31) == 0)
            for (int t = 0; t < ng; ++t) gu[(size_t) orow_g[t] * 2 * F + row] = out[t];
    }
}

template <int N_CAP>
__device__ __forceinline__ void experts_gate_up_swiglu_g_run(const uint8_t* bl, const int* tok_g, const int* orow_g,
                                                             int ng, int ty_gate, int ty_up, size_t eb0,
                                                             const float* x, int C, int F, size_t rb_gate,
                                                             size_t rb_up, const float* sg, const float* su,
                                                             float gcap, float ucap, uint16_t* h) {
    const float* xs[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) xs[t] = x + (size_t) tok_g[t < ng ? t : 0] * C;
    const int warp = threadIdx.x >> 5;
    float og[N_CAP], ou[N_CAP];
    for (int j = 0; j < kGroupSwigluRows; ++j) {
        const int r = (blockIdx.x * kGroupWarps + warp) * kGroupSwigluRows + j;
        if (r >= F) break;
        if (ty_gate == ty_up) {
            row_dot_pair_multi<N_CAP>(ty_gate, bl + (size_t) r * rb_gate, bl + eb0 + (size_t) r * rb_up, xs, ng, C,
                                      og, ou);
        } else {
            row_dot_multi<float, N_CAP>(ty_gate, bl + (size_t) r * rb_gate, xs, ng, C, og);
            row_dot_multi<float, N_CAP>(ty_up, bl + eb0 + (size_t) r * rb_up, xs, ng, C, ou);
        }
        if ((threadIdx.x & 31) == 0) {
            for (int t = 0; t < ng; ++t) {
                const int ro = orow_g[t];
                const float g = og[t] * (sg ? sg[ro] : 1.0f);
                const float u = ou[t] * (su ? su[ro] : 1.0f);
                h[(size_t) ro * F + r] = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
            }
        }
    }
}

template <int N_CAP>
__device__ __forceinline__ void experts_down_g_run(const uint8_t* w, const int* orow_g, int ng, int ty_down,
                                                   const uint16_t* h, int C, int F, size_t rb_down, float* y) {
    const uint16_t* xs[N_CAP];
#pragma unroll
    for (int t = 0; t < N_CAP; ++t) xs[t] = h + (size_t) orow_g[t < ng ? t : 0] * F;
    const int warp = threadIdx.x >> 5;
    float out[N_CAP];
    for (int j = 0; j < kGroupRows; ++j) {
        const int row = (blockIdx.x * kGroupWarps + warp) * kGroupRows + j;
        if (row >= C) break;
        row_dot_multi<uint16_t, N_CAP>(ty_down, w + row * rb_down, xs, ng, F, out);
        if ((threadIdx.x & 31) == 0)
            for (int t = 0; t < ng; ++t) y[(size_t) orow_g[t] * C + row] = out[t];
    }
}

// group g: blob[g], n[g] tokens, their rows of x tok[g][j] and their output rows row[g][j]
__global__ void experts_gate_up_g_kernel(const uint8_t* const* blob, const int* n, const int* tok, const int* orow,
                                         int ty_gate, int ty_up, size_t eb0, const float* x, int C, int F,
                                         size_t rb_gate, size_t rb_up, float* gu) {
    const int g = blockIdx.y, ng = n[g];
    if (ng <= 0) return;
    const uint8_t* bl = blob[g];
    const int* tok_g = tok + g * kGroupMax;
    const int* orow_g = orow + g * kGroupMax;
    if (ng == 1)
        experts_gate_up_g_run<1>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, gu);
    else if (ng == 2)
        experts_gate_up_g_run<2>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, gu);
    else if (ng <= 4)
        experts_gate_up_g_run<4>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, gu);
    else
        experts_gate_up_g_run<kGroupMax>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, gu);
}

__global__ void experts_gate_up_swiglu_g_kernel(const uint8_t* const* blob, const int* n, const int* tok,
                                                const int* orow, int ty_gate, int ty_up, size_t eb0, const float* x,
                                                int C, int F, size_t rb_gate, size_t rb_up, const float* sg,
                                                const float* su, float gcap, float ucap, uint16_t* h) {
    const int g = blockIdx.y, ng = n[g];
    if (ng <= 0) return;
    const uint8_t* bl = blob[g];
    const int* tok_g = tok + g * kGroupMax;
    const int* orow_g = orow + g * kGroupMax;
    if (ng == 1)
        experts_gate_up_swiglu_g_run<1>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, sg, su,
                                        gcap, ucap, h);
    else if (ng == 2)
        experts_gate_up_swiglu_g_run<2>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, sg, su,
                                        gcap, ucap, h);
    else if (ng <= 4)
        experts_gate_up_swiglu_g_run<4>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up, sg, su,
                                        gcap, ucap, h);
    else
        experts_gate_up_swiglu_g_run<kGroupMax>(bl, tok_g, orow_g, ng, ty_gate, ty_up, eb0, x, C, F, rb_gate, rb_up,
                                                sg, su, gcap, ucap, h);
}

__global__ void experts_down_g_kernel(const uint8_t* const* blob, const int* n, const int* orow, int ty_down,
                                      size_t off_down, const uint16_t* h, int C, int F, size_t rb_down, float* y) {
    const int g = blockIdx.y, ng = n[g];
    if (ng <= 0) return;
    const uint8_t* w = blob[g] + off_down;
    const int* orow_g = orow + g * kGroupMax;
    if (ng == 1)
        experts_down_g_run<1>(w, orow_g, ng, ty_down, h, C, F, rb_down, y);
    else if (ng == 2)
        experts_down_g_run<2>(w, orow_g, ng, ty_down, h, C, F, rb_down, y);
    else if (ng <= 4)
        experts_down_g_run<4>(w, orow_g, ng, ty_down, h, C, F, rb_down, y);
    else
        experts_down_g_run<kGroupMax>(w, orow_g, ng, ty_down, h, C, F, rb_down, y);
}

// ---- attention decode & multi-query decode (F32, FP16, or FP8 KV cache, optional kv_cap ring buffer)

template <class KV>
__device__ __forceinline__ float4 v_load4(const KV* vr, int lane);
template <>
__device__ __forceinline__ float4 v_load4<float>(const float* vr, int lane) {
    const int o = lane * 4;
    if (((uintptr_t) vr & 15) == 0) return __ldg((const float4*) (vr + o));
    return make_float4(vr[o], vr[o + 1], vr[o + 2], vr[o + 3]);
}
template <>
__device__ __forceinline__ float4 v_load4<uint16_t>(const uint16_t* vr, int lane) {
    const int o = lane * 4;
    if (((uintptr_t) vr & 7) == 0) {
        const uint2 v4 = __ldg((const uint2*) (vr + o));
        const float2 f0 = u32_to_f2(v4.x), f1 = u32_to_f2(v4.y);
        return make_float4(f0.x, f0.y, f1.x, f1.y);
    }
    return make_float4(h2f(vr[o]), h2f(vr[o + 1]), h2f(vr[o + 2]), h2f(vr[o + 3]));
}
template <>
__device__ __forceinline__ float4 v_load4<uint8_t>(const uint8_t* vr, int lane) {
    const int o = lane * 4;
    if (((uintptr_t) vr & 3) == 0) {
        const uint32_t v4 = __ldg((const uint32_t*) (vr + o));
        const float2 f0 = e4m3fnx2_to_f2((uint16_t) (v4 & 0xFFFFu));
        const float2 f1 = e4m3fnx2_to_f2((uint16_t) (v4 >> 16));
        return make_float4(f0.x, f0.y, f1.x, f1.y);
    }
    return make_float4(e4m3fn_to_f32(vr[o]), e4m3fn_to_f32(vr[o + 1]), e4m3fn_to_f32(vr[o + 2]),
                       e4m3fn_to_f32(vr[o + 3]));
}

template <class KV>
__device__ __forceinline__ void attn_flash_range(const float* qh, const float* rh, const KV* kc, const KV* vc,
                                                 int KD, int g, int kv_cap, int k0, int j0, int j1, int qpos,
                                                 float qk_scale, const float* proj, int d_rel, int extent, int window,
                                                 float* sm_m, float* sm_s, float sm_o[][128]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int o = lane * 4;
    const float4 q4 = (((uintptr_t) qh & 15) == 0) ? __ldg((const float4*) (qh + o))
                                                    : make_float4(qh[o], qh[o + 1], qh[o + 2], qh[o + 3]);
    const float r_lane = (lane < d_rel) ? __ldg(rh + lane) : 0.0f;
    float m_w = -INFINITY, s_w = 0.0f;
    float4 o_w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = j0 + warp; j < j1; j += 4) {
        const int pos = k0 + j;
        const int dist = qpos - pos;
        if (dist < 0 || dist >= window) continue;
        const int slot = kv_cap > 0 ? pos % kv_cap : pos;
        const size_t head_off = (size_t) slot * KD + (size_t) g * 128;
        const float4 k4 = v_load4(kc + head_off, lane);
        float a = q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
        float l = wsum(a) * qk_scale;
        if (dist < extent) {
            const float b = (lane < d_rel) ? r_lane * __ldg(proj + (size_t) dist * d_rel + lane) : 0.0f;
            l += wsum(b);
        }
        if (l > -INFINITY) {
            const float4 v4 = v_load4(vc + head_off, lane);
            const float m_new = fmaxf(m_w, l);
            const float alpha = (m_w == -INFINITY) ? 0.0f : expf(m_w - m_new);
            const float beta = expf(l - m_new);
            m_w = m_new;
            s_w = s_w * alpha + beta;
            o_w.x = o_w.x * alpha + v4.x * beta;
            o_w.y = o_w.y * alpha + v4.y * beta;
            o_w.z = o_w.z * alpha + v4.z * beta;
            o_w.w = o_w.w * alpha + v4.w * beta;
        }
    }
    if (lane == 0) {
        sm_m[warp] = m_w;
        sm_s[warp] = s_w;
    }
    *(float4*) (&sm_o[warp][o]) = o_w;
    __syncthreads();
}

// Unified split kernel for both attn_decode (gridDim.z = 1) and attn_multi (gridDim.z = nq).
template <class KV>
__global__ void attn_split_kernel(const float* q, const float* r, const KV* kc, const KV* vc, int kv_cap, int H,
                                  int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj, int d_rel,
                                  int extent, int window, float* part, const int* pos_d) {
    if (pos_d) {   // (tokens of a replayed graph: the first's position read here, the keys they may see from it)
        qpos0 = *pos_d;
        k0 = window < (1 << 30) ? max(0, qpos0 - window + 1) : 0;
        nk = qpos0 + (int) gridDim.z - k0;
    }
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, p = blockIdx.y, P = gridDim.y, qi = blockIdx.z, g = h / (H / Hkv), tid = threadIdx.x;
    const int KD = Hkv * 128, qpos = qpos0 + qi;
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    const float* qh = q + ((size_t) qi * H + h) * 128;
    const float* rh = r + ((size_t) qi * H + h) * d_rel;
    attn_flash_range(qh, rh, kc, vc, KD, g, kv_cap, k0, j0, j1, qpos, qk_scale, proj, d_rel, extent, window, sm_m,
                     sm_s, sm_o);
    const float M = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
    float sum = 0.0f, o = 0.0f;
    if (M > -INFINITY) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (sm_m[w] == -INFINITY) continue;
            const float f = expf(sm_m[w] - M);
            sum += sm_s[w] * f;
            o += sm_o[w][tid] * f;
        }
    }
    float* pp = part + (((size_t) qi * H + h) * P + p) * 130;
    if (tid == 0) pp[0] = M, pp[1] = sum;
    pp[2 + tid] = o;
}

__global__ void attn_combine_kernel(const float* sink, const float* part, int H, int P, float v_inv_scale, float* out) {
    const int h = blockIdx.x, qi = blockIdx.y, tid = threadIdx.x;
    const float* pp = part + ((size_t) qi * H + h) * P * 130;
    const float sk = sink[(size_t) qi * H + h];
    float M = sk;
    for (int p = 0; p < P; ++p) M = fmaxf(M, pp[(size_t) p * 130]);
    float den = expf(sk - M), o = 0.0f;
    for (int p = 0; p < P; ++p) {
        const float mp = pp[(size_t) p * 130];
        if (mp == -INFINITY) continue;
        const float f = expf(mp - M);
        den += pp[(size_t) p * 130 + 1] * f;
        o += pp[(size_t) p * 130 + 2 + tid] * f;
    }
    out[((size_t) qi * H + h) * 128 + tid] = (o / den) * v_inv_scale;
}

// the scalar decode combine with the output gate riding its epilogue (a launch less per visit: the tensor-core
// merge already has it; this is the scalar path's equivalent). gate==nullptr keeps the plain combine.
__global__ void attn_combine_gate_kernel(const float* sink, const float* part, int H, int P, float v_inv_scale,
                                         const float* gate, float* out) {
    const int h = blockIdx.x, qi = blockIdx.y, tid = threadIdx.x;
    const float* pp = part + ((size_t) qi * H + h) * P * 130;
    const float sk = sink[(size_t) qi * H + h];
    float M = sk;
    for (int p = 0; p < P; ++p) M = fmaxf(M, pp[(size_t) p * 130]);
    float den = expf(sk - M), o = 0.0f;
    for (int p = 0; p < P; ++p) {
        const float mp = pp[(size_t) p * 130];
        if (mp == -INFINITY) continue;
        const float f = expf(mp - M);
        den += pp[(size_t) p * 130 + 1] * f;
        o += pp[(size_t) p * 130 + 2 + tid] * f;
    }
    float v = (o / den) * v_inv_scale;
    if (gate) v *= 1.0f / (1.0f + expf(-gate[((size_t) qi * H + h) * 128 + tid]));
    out[((size_t) qi * H + h) * 128 + tid] = v;
}

// IK2_VERIFY_MSHARED: M-row-shared split-KV verify attention. attn_split_kernel reads the whole KV range once
// per query row (grid.z = nq -> nq x KV traffic); this kernel keeps grid = (H, P) and walks the key range ONCE
// for ALL nq rows: each wave loads a K row (and, if any row needs it, the V row) a single time, then dots it
// against every query's register-resident q4 and runs each row's own online softmax. Per-wave key order
// (j = j0 + warp, stride 4), the dot / rel-bias expressions, the softmax update order and the 4-wave merge are
// copied verbatim from attn_flash_range / attn_split_kernel, so the partials are bit-identical and
// attn_combine_kernel is reused unchanged (same [nq, H, P, 130] layout, sink folded there).
// NQ is the compile-time row count (dispatched 1..8); nq > 8 stays on the old path.
template <class KV, int NQ>
__global__ void attn_multi_shared_kernel(const float* q, const float* r, const KV* kc, const KV* vc, int kv_cap,
                                         int H, int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj,
                                         int d_rel, int extent, int window, float* part, const int* pos_d, int nq) {
    if (pos_d) {   // replayed graph: derive the base position / key range on device (nq replaces gridDim.z)
        qpos0 = *pos_d;
        k0 = window < (1 << 30) ? max(0, qpos0 - window + 1) : 0;
        nk = qpos0 + nq - k0;
    }
    __shared__ float sm_m[4][NQ], sm_s[4][NQ];
    __shared__ __align__(16) float sm_o[4][NQ][128];
    const int h = blockIdx.x, p = blockIdx.y, P = gridDim.y, g = h / (H / Hkv), tid = threadIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, o = lane * 4;
    const int KD = Hkv * 128;
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    float4 q4[NQ];
    float r_lane[NQ], m_w[NQ], s_w[NQ];
    float4 o_w[NQ];
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {
        const float* qh = q + ((size_t) qi * H + h) * 128;
        q4[qi] = (((uintptr_t) qh & 15) == 0) ? __ldg((const float4*) (qh + o))
                                              : make_float4(qh[o], qh[o + 1], qh[o + 2], qh[o + 3]);
        const float* rh = r + ((size_t) qi * H + h) * d_rel;
        r_lane[qi] = (lane < d_rel) ? __ldg(rh + lane) : 0.0f;
        m_w[qi] = -INFINITY;
        s_w[qi] = 0.0f;
        o_w[qi] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int j = j0 + warp; j < j1; j += 4) {
        const int pos = k0 + j;
        const int dist0 = qpos0 - pos;
        if (dist0 <= -NQ || dist0 >= window) continue;   // no query row can see this key: skip the K load too
        const int slot = kv_cap > 0 ? pos % kv_cap : pos;
        const size_t head_off = (size_t) slot * KD + (size_t) g * 128;
        const float4 k4 = v_load4(kc + head_off, lane);   // ONE K load shared by all NQ rows
        float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        bool v_loaded = false;
#pragma unroll
        for (int qi = 0; qi < NQ; ++qi) {
            const int dist = dist0 + qi;   // qpos = qpos0 + qi, per-row causal / sliding mask
            if (dist < 0 || dist >= window) continue;
            float a = q4[qi].x * k4.x + q4[qi].y * k4.y + q4[qi].z * k4.z + q4[qi].w * k4.w;
            float l = wsum(a) * qk_scale;
            if (dist < extent) {
                const float b = (lane < d_rel) ? r_lane[qi] * __ldg(proj + (size_t) dist * d_rel + lane) : 0.0f;
                l += wsum(b);
            }
            if (l > -INFINITY) {
                if (!v_loaded) {   // ONE V load, only if some row survives (l is wave-uniform: no divergence)
                    v4 = v_load4(vc + head_off, lane);
                    v_loaded = true;
                }
                const float m_new = fmaxf(m_w[qi], l);
                const float alpha = (m_w[qi] == -INFINITY) ? 0.0f : expf(m_w[qi] - m_new);
                const float beta = expf(l - m_new);
                m_w[qi] = m_new;
                s_w[qi] = s_w[qi] * alpha + beta;
                o_w[qi].x = o_w[qi].x * alpha + v4.x * beta;
                o_w[qi].y = o_w[qi].y * alpha + v4.y * beta;
                o_w[qi].z = o_w[qi].z * alpha + v4.z * beta;
                o_w[qi].w = o_w[qi].w * alpha + v4.w * beta;
            }
        }
    }
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {
        if (lane == 0) {
            sm_m[warp][qi] = m_w[qi];
            sm_s[warp][qi] = s_w[qi];
        }
        *(float4*) (&sm_o[warp][qi][o]) = o_w[qi];
    }
    __syncthreads();
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {   // same 4-wave merge as attn_split_kernel, once per row
        const float M = fmaxf(fmaxf(sm_m[0][qi], sm_m[1][qi]), fmaxf(sm_m[2][qi], sm_m[3][qi]));
        float sum = 0.0f, ov = 0.0f;
        if (M > -INFINITY) {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (sm_m[w][qi] == -INFINITY) continue;
                const float f = expf(sm_m[w][qi] - M);
                sum += sm_s[w][qi] * f;
                ov += sm_o[w][qi][tid] * f;
            }
        }
        float* pp = part + (((size_t) qi * H + h) * P + p) * 130;
        if (tid == 0) pp[0] = M, pp[1] = sum;
        pp[2 + tid] = ov;
    }
}

template <class KV, int NQ>
void attn_multi_shared_launch(const float* q, const float* r, const KV* kc, const KV* vc, int kv_cap, int H,
                              int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj, int d_rel,
                              int extent, int window, float* part, const int* pos_d, int P, cudaStream_t s) {
    attn_multi_shared_kernel<KV, NQ><<<dim3(H, (unsigned) P), 128, 0, s>>>(
        q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos0, qk_scale, proj, d_rel, extent, window, part, pos_d, NQ);
}

// nq 1..8 -> the matching compile-time row count (caller gates the range).
template <class KV>
void attn_multi_shared_dispatch(const float* q, const float* r, const KV* kc, const KV* vc, int kv_cap, int nq,
                                int H, int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj,
                                int d_rel, int extent, int window, float* part, const int* pos_d, int P,
                                cudaStream_t s) {
#define IK2_MSH_CASE(N)                                                                                             \
    case N:                                                                                                         \
        attn_multi_shared_launch<KV, N>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos0, qk_scale, proj, d_rel, extent, \
                                        window, part, pos_d, P, s);                                                 \
        break
    switch (nq) {
        IK2_MSH_CASE(1);
        IK2_MSH_CASE(2);
        IK2_MSH_CASE(3);
        IK2_MSH_CASE(4);
        IK2_MSH_CASE(5);
        IK2_MSH_CASE(6);
        IK2_MSH_CASE(7);
        IK2_MSH_CASE(8);
        default: std::abort();
    }
#undef IK2_MSH_CASE
}

// IK2_VERIFY_MSHARED=1 opts the multi-query verify attention into the shared-KV kernel above; default OFF
// keeps the per-query split path byte-identical. Read once per process, like the engine's other env gates.
bool mshared_on() {
    static const bool on = [] {
        const char* e = std::getenv("IK2_VERIFY_MSHARED");
        return e && std::atoi(e) != 0;
    }();
    return on;
}

// IK2_FA_SPLIT=1 opts the int4 decode split kernels into the coalesced 8-byte K/V loads (v_load4_int_fa).
// Default OFF keeps the original 2-byte/lane path byte-identical.  Read once per process, like the other
// env gates.  An explicit fa_split argument overrides it: >0 forces FA, <0 forces the legacy path
// (the unit test uses both to compare the two paths without touching the environment).
static int fa_split_on() {
    static const int v = [] {
        const char* e = std::getenv("IK2_FA_SPLIT");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}

// IK2_FA_RDNA3 is an integer bitmask of opt-in attention-decode levers.  It is read once per process,
// like IK2_FA_SPLIT.  Default 0 keeps the original split kernels byte-identical.  An explicit
// rdna3_mask argument overrides the env: -1 follows the env, >=0 is used verbatim (the unit test
// passes 0..15 so it need not mutate a process-global env).
//   bit 0 (1): wave-balanced P - align H*P (plain) / Hkv*P (GQA) CTAs to the device wave (48 WGP on gfx1100).
//   bit 1 (2): PRE_LOAD_V - GQA loads V before the R-head loop instead of inside the first l>-inf branch.
//   bit 2 (4): wideload - v_load4_int_fa's preserve-order 8-byte K/V loads, independent of IK2_FA_SPLIT.
//   bit 3 (8): exp2/merge relax - exp2f(x*log2(e)) replaces expf(x) in the online softmax and warp merge.
enum {
    IK2_FA_RDNA3_WAVE_P = 1,
    IK2_FA_RDNA3_PRE_LOAD_V = 2,
    IK2_FA_RDNA3_WIDELOAD = 4,
    IK2_FA_RDNA3_EXP2 = 8,
};
static int fa_rdna3_mask() {
    static const int v = [] {
        const char* e = std::getenv("IK2_FA_RDNA3");
        return e ? std::atoi(e) : 0;
    }();
    return v;
}
static int fa_rdna3_effective(int rdna3_mask) { return rdna3_mask < 0 ? fa_rdna3_mask() : rdna3_mask; }
static int fa_rdna3_sms() {
    static const int n = [] {
        int dev = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 48;
        cudaDeviceProp p{};
        return cudaGetDeviceProperties(&p, dev) == cudaSuccess && p.multiProcessorCount > 0
                   ? p.multiProcessorCount : 48;
    }();
    return n;
}
// Round P to the nearest wave-up multiple for grid.x CTAs per wave, capped at the engine's 128 splits.
// Never increases past 128 and never returns < 1, so the [H,P,130] scratch stays within the engine's cap.
static int fa_rdna3_wave_p(int P, int gridx) {
    if (P <= 0 || gridx <= 0) return P;
    const int sm = fa_rdna3_sms();
    int a = sm, b = gridx;
    while (b) { const int t = a % b; a = b; b = t; }
    const int wave_p = a > 0 ? sm / a : 1;          // CTAs per wave = gridx * wave_p
    if (wave_p <= 1) return P;
    int q = (P + wave_p - 1) / wave_p;
    if (q > 128 / wave_p) q = 128 / wave_p;        // stay within the existing split cap
    return std::max(1, q * wave_p);
}

template <int WARPS, int ROWS>
__global__ void mv_blocks_kernel(int type, const uint8_t* W, int N, int K, size_t rb, const float* x, float* y) {
    extern __shared__ float xs[];
    load_xs_f32(x, K, xs);
    __syncthreads();
    const int warp = threadIdx.x >> 5;
    for (int j = 0; j < ROWS; ++j) {
        const int row = (blockIdx.x * WARPS + warp) * ROWS + j;
        if (row >= N) break;
        const float v = row_dot(type, W + (size_t) row * rb, xs, K);
        if ((threadIdx.x & 31) == 0) y[row] = v;
    }
}

// one block a head, 4 warps: register Flash-Decode online softmax per warp, 1 SMEM combine
template <class KV>
__global__ void attn_decode_kernel(const float* q, const float* r, const float* sink, const KV* kc, const KV* vc,
                                   int kv_cap, int H, int Hkv, int k0, int nk, int qpos, float qk_scale,
                                   float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* out) {
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, g = h / (H / Hkv), tid = threadIdx.x;
    const int KD = Hkv * 128;
    const float* qh = q + (size_t) h * 128;
    const float* rh = r + (size_t) h * d_rel;
    attn_flash_range(qh, rh, kc, vc, KD, g, kv_cap, k0, 0, nk, qpos, qk_scale, proj, d_rel, extent, window, sm_m, sm_s,
                     sm_o);
    const float sk = sink[h];
    const float M = fmaxf(sk, fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3])));
    float den = expf(sk - M), o = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (sm_m[w] == -INFINITY) continue;
        const float f = expf(sm_m[w] - M);
        den += sm_s[w] * f;
        o += sm_o[w][tid] * f;
    }
    out[(size_t) h * 128 + tid] = (o / den) * v_inv_scale;
}

__global__ void unpack_kv_fp8_to_f16_vec4_kernel(const uint8_t* kc, const uint8_t* vc, int kv_cap, int k0, int nk,
                                                 int KD, float k_inv_scale, float v_inv_scale, uint16_t* k_out16,
                                                 uint16_t* v_out16) {
    const int kd4 = KD >> 2;
    const int d4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (d4 >= kd4) return;
    for (int j = blockIdx.y; j < nk; j += gridDim.y) {
        const int slot = kv_cap > 0 ? (k0 + j) % kv_cap : (k0 + j);
        const size_t src_off = (size_t) slot * KD;
        const size_t dst_off = (size_t) j * KD;
        if (k_out16) {
            const uint32_t k4 = __ldg((const uint32_t*) (kc + src_off) + d4);
            const float2 f0 = e4m3fnx2_to_f2((uint16_t) (k4 & 0xFFFFu));
            const float2 f1 = e4m3fnx2_to_f2((uint16_t) (k4 >> 16));
            ((uint2*) (k_out16 + dst_off))[d4] = make_uint2(f2_to_u32(f0.x * k_inv_scale, f0.y * k_inv_scale),
                                                            f2_to_u32(f1.x * k_inv_scale, f1.y * k_inv_scale));
        }
        if (v_out16) {
            const uint32_t v4 = __ldg((const uint32_t*) (vc + src_off) + d4);
            const float2 f0 = e4m3fnx2_to_f2((uint16_t) (v4 & 0xFFFFu));
            const float2 f1 = e4m3fnx2_to_f2((uint16_t) (v4 >> 16));
            ((uint2*) (v_out16 + dst_off))[d4] = make_uint2(f2_to_u32(f0.x * v_inv_scale, f0.y * v_inv_scale),
                                                            f2_to_u32(f1.x * v_inv_scale, f1.y * v_inv_scale));
        }
    }
}

__global__ void unpack_kv_fp8_to_f16_kernel(const uint8_t* kc, const uint8_t* vc, int kv_cap, int k0, int nk, int KD,
                                            float k_inv_scale, float v_inv_scale, uint16_t* k_out16,
                                            uint16_t* v_out16) {
    const size_t total = (size_t) nk * KD;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t) gridDim.x * blockDim.x) {
        const int j = (int) (idx / (size_t) KD);
        const int d = (int) (idx % (size_t) KD);
        const int slot = kv_cap > 0 ? (k0 + j) % kv_cap : (k0 + j);
        const size_t src = (size_t) slot * KD + d;
        if (k_out16) k_out16[idx] = f2h(e4m3fn_to_f32(kc[src]) * k_inv_scale);
        if (v_out16) v_out16[idx] = f2h(e4m3fn_to_f32(vc[src]) * v_inv_scale);
    }
}

// ---- the resident token path: every expert in VRAM and nothing of a token on the host

// y [N] = W [N][K] FP16 . x [K]: a warp a row (the small dense matrices kept in FP16: sinks, dynamic taps, the router)
template <int WARPS>
__global__ void mv_f16_kernel(const uint16_t* __restrict__ W, int N, int K, const float* __restrict__ x,
                              float* __restrict__ y) {
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint16_t* w = W + (size_t) row * K;
    float a = 0.0f;
    for (int i = lane; i < K; i += 32) a += h2f(w[i]) * x[i];
    a = wsum(a);
    if (lane == 0) y[row] = a;
}
// mv_f16 with the input RMSNorm'd in the kernel's prologue (x raw, nw the norm weights): saves the separate rms
// launch before a full layer's dynamic-tap products (which take the normed k/v input). K<=2304.
template <int WARPS>
__global__ void mv_f16_rms_kernel(const uint16_t* __restrict__ W, int N, int K, const float* __restrict__ x,
                                  const float* __restrict__ nw, float eps, float* __restrict__ y) {
    __shared__ __align__(16) float xs[2304];
    __shared__ float warp_ss[WARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < K; i += WARPS * 32) { const float v = x[i]; ss += v * v; }
    ss = wsum(ss);
    if (lane == 0) warp_ss[warp] = ss;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) total += warp_ss[w];
    const float inv = rsqrtf(total / (float) K + eps);
    for (int i = tid; i < K; i += WARPS * 32) xs[i] = x[i] * inv * nw[i];
    __syncthreads();
    const int row = blockIdx.x * WARPS + warp;
    if (row >= N) return;
    const uint16_t* w = W + (size_t) row * K;
    float a = 0.0f;
    for (int i = lane; i < K; i += 32) a += h2f(w[i]) * xs[i];
    a = wsum(a);
    if (lane == 0) y[row] = a;
}

// a product's row through its short conv (conv4_fused_kernel's arithmetic for one token): false if the row has none
__device__ __forceinline__ bool mv_conv_row(const MvConv& cv, int row, float x_new) {
    const int c = row - cv.r0;
    if (c < 0 || c >= cv.n) return false;
    const float h0 = cv.hist[c], h1 = cv.hist[(size_t) cv.n + c], h2 = cv.hist[(size_t) 2 * cv.n + c];
    const float* wc = cv.w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float a = w0 * h0 + w1 * h1 + w2 * h2 + w3 * x_new;
    if (cv.dyn) {
        const float* dc = cv.dyn + (size_t) (c / (cv.n / cv.heads)) * 4;
        a += dc[0] * x_new + dc[1] * h2 + dc[2] * h1 + dc[3] * h0;
    }
    if (cv.res) a += cv.res[c];
    cv.y[c] = a;
    cv.hist[c] = h1;
    cv.hist[(size_t) cv.n + c] = h2;
    cv.hist[(size_t) 2 * cv.n + c] = x_new;
    return true;
}

// y [N] = W . x for a Q8_0 matrix kept planar - its values q [N][K] int8 and its blocks' scales d [N][K / 32] FP16 -
// so that a warp reads 128 values an instruction; x where it is.  A warp a row; a row of c0's or c1's goes through
// its short conv instead of to y (a launch less a conv: a token's launches each cost 2 us and more).
template <int WARPS>
__global__ void mv_q8p_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d, int N, int K,
                              const float* __restrict__ x, float* y, MvConv c0, MvConv c1, MvAlt a0, MvAlt a1) {
    const int row = blockIdx.x * WARPS + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= N) return;
    const uint32_t* qr = (const uint32_t*) (q + (size_t) row * K);
    const uint16_t* dr = d + (size_t) row * (K >> 5);
    // (a0's and a1's rows have an input of their own: several products of a layer's fused matrix in one launch)
    const float4* x4 = (const float4*) ((unsigned) (row - a0.r0) < (unsigned) a0.n   ? a0.x
                                        : (unsigned) (row - a1.r0) < (unsigned) a1.n ? a1.x
                                                                                     : x);
    float acc = 0.0f;
    for (int it = 0, n_it = K >> 7; it < n_it; ++it) {
        const uint32_t w = __ldg(qr + it * 32 + lane);
        const float4 xv = __ldg(x4 + it * 32 + lane);
        const float dv = h2f(__ldg(dr + it * 4 + (lane >> 3)));
        acc += dv * ((float) (int8_t) (w & 0xFFu) * xv.x + (float) (int8_t) ((w >> 8) & 0xFFu) * xv.y +
                     (float) (int8_t) ((w >> 16) & 0xFFu) * xv.z + (float) (int8_t) (w >> 24) * xv.w);
    }
    acc = wsum(acc);
    if (lane == 0 && !mv_conv_row(c0, row, acc) && !mv_conv_row(c1, row, acc)) y[row] = acc;
}
// mv_q8p whose input is RMSNorm'd in the kernel's prologue (x raw, nw weights, eps), with the same MvAlt per-row
// inputs as mv_q8p: the norm rides the product launch (a launch less a visit). The prologue norms the default input
// x into LDS once; a0/a1's rows read their (already-normed) inputs. K<=2304, K%128==0, q 4B / x 16B aligned.
template <int WARPS>
__global__ void mv_q8p_rms_alt_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d, int N, int K,
                                      const float* __restrict__ x, const float* __restrict__ nw, float eps, float* y,
                                      MvConv c0, MvConv c1, MvAlt a0, MvAlt a1) {
    __shared__ __align__(16) float xs[2304];
    __shared__ float warp_ss[WARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    float ss = 0.0f;
    for (int i = tid; i < K; i += WARPS * 32) { const float v = x[i]; ss += v * v; }
    ss = wsum(ss);
    if (lane == 0) warp_ss[warp] = ss;
    __syncthreads();
    float total = 0.0f;
#pragma unroll
    for (int w = 0; w < WARPS; ++w) total += warp_ss[w];
    const float inv = rsqrtf(total / (float) K + eps);
    for (int i = tid; i < K; i += WARPS * 32) xs[i] = x[i] * inv * nw[i];
    __syncthreads();
    const int row = blockIdx.x * WARPS + warp;
    if (row >= N) return;
    const uint32_t* qr = (const uint32_t*) (q + (size_t) row * K);
    const uint16_t* dr = d + (size_t) row * (K >> 5);
    const float4* x4 = (const float4*) ((unsigned) (row - a0.r0) < (unsigned) a0.n   ? a0.x
                                        : (unsigned) (row - a1.r0) < (unsigned) a1.n ? a1.x
                                                                                     : (const float*) xs);
    float acc = 0.0f;
    for (int it = 0, n_it = K >> 7; it < n_it; ++it) {
        const uint32_t w = __ldg(qr + it * 32 + lane);
        const float4 xv = x4[it * 32 + lane];
        const float dv = h2f(__ldg(dr + it * 4 + (lane >> 3)));
        acc += dv * ((float) (int8_t) (w & 0xFFu) * xv.x + (float) (int8_t) ((w >> 8) & 0xFFu) * xv.y +
                     (float) (int8_t) ((w >> 16) & 0xFFu) * xv.z + (float) (int8_t) (w >> 24) * xv.w);
    }
    acc = wsum(acc);
    if (lane == 0 && !mv_conv_row(c0, row, acc) && !mv_conv_row(c1, row, acc)) y[row] = acc;
}
// mv_q8p with the input RMSNorm riding its prologue: the block norms its rows' input once into LDS (a launch less a
// visit for the full layer's wk/wv products). Only the plain single-input case (no MvAlt); RMS_PROLOGUE picks the
// input to norm, else the kernel is mv_q8p.
template <int WARPS, bool RMS_PROLOGUE>
__global__ void mv_q8p_rms_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d, int N, int K,
                                  const float* __restrict__ x, const float* __restrict__ nw, float eps, float* y,
                                  MvConv c0, MvConv c1) {
    __shared__ __align__(16) float xs[2304];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
    if (RMS_PROLOGUE) {
        float ss = 0.0f;
        for (int i = tid; i < K; i += WARPS * 32) { const float v = x[i]; ss += v * v; }
        ss = wsum(ss);
        __shared__ float warp_ss[WARPS];
        if (lane == 0) warp_ss[warp] = ss;
        __syncthreads();
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) total += warp_ss[w];
        const float inv = rsqrtf(total / (float) K + eps);
        for (int i = tid; i < K; i += WARPS * 32) xs[i] = x[i] * inv * nw[i];
        __syncthreads();
    }
    const int row = blockIdx.x * WARPS + warp;
    if (row >= N) return;
    const uint32_t* qr = (const uint32_t*) (q + (size_t) row * K);
    const uint16_t* dr = d + (size_t) row * (K >> 5);
    const float4* x4 = RMS_PROLOGUE ? (const float4*) xs : (const float4*) x;
    float acc = 0.0f;
    for (int it = 0, n_it = K >> 7; it < n_it; ++it) {
        const uint32_t w = __ldg(qr + it * 32 + lane);
        const float4 xv = RMS_PROLOGUE ? x4[it * 32 + lane] : __ldg(x4 + it * 32 + lane);
        const float dv = h2f(__ldg(dr + it * 4 + (lane >> 3)));
        acc += dv * ((float) (int8_t) (w & 0xFFu) * xv.x + (float) (int8_t) ((w >> 8) & 0xFFu) * xv.y +
                     (float) (int8_t) ((w >> 16) & 0xFFu) * xv.z + (float) (int8_t) (w >> 24) * xv.w);
    }
    acc = wsum(acc);
    if (lane == 0 && !mv_conv_row(c0, row, acc) && !mv_conv_row(c1, row, acc)) y[row] = acc;
}
// mv_q8p with TWO inputs RMSNorm'd in the kernel's prologue (x raw, nw[2] the norm weights, eps): the block norms
// both into LDS once (xs[2][K]), saving the separate rms/rms_multi launch before a full layer's wk/wv products.
// Each row's input picked by which MvConv range it falls in (c0.n -> input 0, c1.n -> input 1, else input 0).
template <int WARPS>
__global__ void mv_q8p_rms2_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d, int N, int K,
                                   const float* __restrict__ x, const float* __restrict__ nw0,
                                   const float* __restrict__ nw1, float eps, float* y, MvConv c0, MvConv c1) {
    __shared__ __align__(16) float xs[2][2304];
    __shared__ float warp_ss[2][WARPS];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, tid = threadIdx.x;
#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const float* nw = b == 0 ? nw0 : nw1;
        float ss = 0.0f;
        for (int i = tid; i < K; i += WARPS * 32) { const float v = x[i]; ss += v * v; }
        ss = wsum(ss);
        if (lane == 0) warp_ss[b][warp] = ss;
        __syncthreads();
        float total = 0.0f;
#pragma unroll
        for (int w = 0; w < WARPS; ++w) total += warp_ss[b][w];
        const float inv = rsqrtf(total / (float) K + eps);
        for (int i = tid; i < K; i += WARPS * 32) xs[b][i] = x[i] * inv * nw[i];
        __syncthreads();
    }
    const int row = blockIdx.x * WARPS + warp;
    if (row >= N) return;
    const int which = ((unsigned) (row - c0.r0) < (unsigned) c0.n) ? 0 : 1;
    const uint32_t* qr = (const uint32_t*) (q + (size_t) row * K);
    const uint16_t* dr = d + (size_t) row * (K >> 5);
    const float4* x4 = (const float4*) xs[which];
    float acc = 0.0f;
    for (int it = 0, n_it = K >> 7; it < n_it; ++it) {
        const uint32_t w = __ldg(qr + it * 32 + lane);
        const float4 xv = x4[it * 32 + lane];
        const float dv = h2f(__ldg(dr + it * 4 + (lane >> 3)));
        acc += dv * ((float) (int8_t) (w & 0xFFu) * xv.x + (float) (int8_t) ((w >> 8) & 0xFFu) * xv.y +
                     (float) (int8_t) ((w >> 16) & 0xFFu) * xv.z + (float) (int8_t) (w >> 24) * xv.w);
    }
    acc = wsum(acc);
    if (lane == 0 && !mv_conv_row(c0, row, acc) && !mv_conv_row(c1, row, acc)) y[row] = acc;
}
__global__ void decode_q8p_f16_kernel(const int8_t* __restrict__ q, const uint16_t* __restrict__ d, size_t n,
                                      uint16_t* __restrict__ out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2h(h2f(d[i >> 5]) * (float) q[i]);
}

// the routed slots' tables from the picks, on the device: slot i's expert e = top[i] - its blocks, its token, its
// scales; wt = the route weight times down's scale
__global__ void route_gather_kernel(const int* __restrict__ top, const float* __restrict__ w,
                                    const uint8_t* const* __restrict__ tab, const float* __restrict__ sg_t,
                                    const float* __restrict__ su_t, const float* __restrict__ sd_t, int S, int K,
                                    const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt) {
    const int i = threadIdx.x;
    if (i >= S) return;
    const int e = top[i];
    blob[i] = tab[e];
    tok[i] = i / K;
    inv[i] = i;
    sg[i] = sg_t ? sg_t[e] : 1.0f;
    su[i] = su_t ? su_t[e] : 1.0f;
    wt[i] = w[i] * (sd_t ? sd_t[e] : 1.0f);
}
// One token's routing in one launch: route's picks and weights and route_gather's tables, the same values.  A
// thread an expert: its rank is the count of the experts ahead of it (larger sigmoid + bias; a tie: the lower
// expert), the K of rank under K are the picks, and a pick's place among them - ascending experts - is the count of
// picks under it.  (route_warp_kernel finds them by K rounds of one warp's argmax: 10 us of a lone warp's latency.)
__global__ void route_token_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int E, int K,
                                   float scale, const uint8_t* const* __restrict__ tab, const float* __restrict__ sg_t,
                                   const float* __restrict__ su_t, const float* __restrict__ sd_t, int* top, float* w,
                                   const uint8_t** blob, int* tok, int* inv, float* sg, float* su, float* wt,
                                   const float* x, int width, uint16_t* x_limbs,
                                   const uint8_t* const* __restrict__ i4_tabWp, const uint8_t* const* __restrict__ i4_tabWh,
                                   const float* const* __restrict__ i4_tabWS,
                                   const uint8_t** __restrict__ i4_Wp, const uint8_t** __restrict__ i4_Wh, const float** __restrict__ i4_WS) {
    if (blockIdx.x) {   // (the blocks after the first: x = hi + lo in FP16, [2][width] - experts_gate_up_swiglu_mma's input)
        const int i = (blockIdx.x - 1) * blockDim.x + threadIdx.x;
        if (i < width) {
            const uint16_t hi = f2h(x[i]);
            x_limbs[i] = hi, x_limbs[width + i] = f2h(x[i] - h2f(hi));
        }
        return;
    }
    extern __shared__ float rt_sm[];
    float* val = rt_sm;                                       // [E] sigmoid + bias
    uint32_t* picked = (uint32_t*) (rt_sm + E);               // [warps] a warp's picks, a bit a lane
    float* sig = (float*) (picked + (blockDim.x >> 5));       // [K] the picks' sigmoids in order, [K]: their sum
    const int e = threadIdx.x, lane = e & 31, warp = e >> 5;
    const float sl = e < E ? 1.0f / (1.0f + expf(-__ldg(logits + e))) : 0.0f;
    const float v = e < E ? sl + __ldg(bias + e) : -INFINITY;
    if (e < E) val[e] = v;
    __syncthreads();
    int rank = 0;
    for (int j = 0; j < E; ++j) {
        const float o = val[j];
        rank += (o > v || (o == v && j < e)) ? 1 : 0;
    }
    const bool pick = e < E && rank < K;
    const uint32_t mine = __ballot_sync(0xffffffffu, pick);
    if (lane == 0) picked[warp] = mine;
    __syncthreads();
    int pos = __popc(mine & ((1u << lane) - 1u));
    for (int ww = 0; ww < warp; ++ww) pos += __popc(picked[ww]);
    if (pick) sig[pos] = sl;
    __syncthreads();
    if (e == 0) {   // (the sum in route's order: the same weights to the bit)
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) sum += sig[k];
        sig[K] = sum;
    }
    __syncthreads();
    if (pick) {
        const float wk = sl / (sig[K] + 1e-20f) * scale;
        top[pos] = e;
        w[pos] = wk;
        blob[pos] = tab[e];
        tok[pos] = 0;
        inv[pos] = pos;
        sg[pos] = sg_t ? sg_t[e] : 1.0f;
        su[pos] = su_t ? su_t[e] : 1.0f;
        wt[pos] = wk * (sd_t ? sd_t[e] : 1.0f);
        if (i4_tabWp) { i4_Wp[pos] = i4_tabWp[e]; i4_Wh[pos] = i4_tabWh[e]; i4_WS[pos] = i4_tabWS[e]; }
    }
}

__global__ void ep_select_token_kernel(const int* __restrict__ top, const float* __restrict__ w,
                                       const uint8_t* const* __restrict__ tab,
                                       const float* __restrict__ sg_t,
                                       const float* __restrict__ su_t,
                                       const float* __restrict__ sd_t,
                                       int S, int K, const uint8_t** blob, int* tok, int* inv,
                                       float* sg, float* su, float* wt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S) return;
    const int e = top[i];
    const uint8_t* ptr = (tab && e >= 0) ? tab[e] : nullptr;
    blob[i] = ptr;
    if (tok) tok[i] = K > 0 ? (i / K) : 0;
    if (inv) inv[i] = i;
    if (sg) sg[i] = (sg_t && e >= 0) ? sg_t[e] : 1.0f;
    if (su) su[i] = (su_t && e >= 0) ? su_t[e] : 1.0f;
    if (wt) wt[i] = (w ? w[i] : 1.0f) * ((sd_t && e >= 0) ? sd_t[e] : 1.0f);
}

__global__ void ep_combine_token_kernel(const float* __restrict__ y0,
                                        const float* __restrict__ y1,
                                        const uint8_t* const* __restrict__ blob0,
                                        const float* __restrict__ wt,
                                        int S, int C, float* __restrict__ out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float a = 0.0f;
    for (int k = 0; k < S; ++k) {
        const float* src = blob0[k] ? y0 : y1;
        a += wt[k] * src[(size_t) k * C + c];
    }
    out[c] = a;
}

__global__ void ep_combine_block_kernel(const float* __restrict__ y0,
                                        const float* __restrict__ y1,
                                        const uint8_t* const* __restrict__ blob0,
                                        const float* __restrict__ wt,
                                        int T, int K, int C, float* __restrict__ out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * C) return;
    const size_t t = i / C, c = i % C;
    float a = 0.0f;
    for (int k = 0; k < K; ++k) {
        const int s = (int) (t * K + k);
        const float* src = blob0[s] ? y0 : y1;
        a += wt[s] * src[(size_t) s * C + c];
    }
    out[i] = a;
}

__global__ void ep_combine_conv_kernel(const float* __restrict__ y0,
                                       const float* __restrict__ y1,
                                       const uint8_t* const* __restrict__ blob0,
                                       const float* __restrict__ wt,
                                       int S, int C, float* hist,
                                       const float* __restrict__ w,
                                       const float* res, float* out) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float a = 0.0f;
    for (int k = 0; k < S; ++k) {
        const float* src = blob0[k] ? y0 : y1;
        a += wt[k] * src[(size_t) k * C + c];
    }
    const float h0 = hist[c], h1 = hist[(size_t) C + c], h2 = hist[(size_t) 2 * C + c];
    const float* wc = w + (size_t) c * 4;
    const float w0 = wc[0], w1 = wc[1], w2 = wc[2], w3 = wc[3];
    float b = w0 * h0 + w1 * h1 + w2 * h2 + w3 * a;
    if (res) b += res[c];
    out[c] = b;
    hist[c] = h1;
    hist[(size_t) C + c] = h2;
    hist[(size_t) 2 * C + c] = a;
}

// ---- speculative decoding

__global__ void conv_hist_after_kernel(const float* old, const float* x, int T, int C, float* hist) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    float v[3];
    for (int j = 0; j < 3; ++j) {
        const int r = T + j;   // row r of [old; x]
        v[j] = r < 3 ? old[(size_t) r * C + c] : x[(size_t) (r - 3) * C + c];
    }
    for (int j = 0; j < 3; ++j) hist[(size_t) j * C + c] = v[j];
}

__device__ __forceinline__ uint16_t f2bf_rn(float f) {
    uint32_t u = __float_as_uint(f);
    if ((u & 0x7fffffffu) > 0x7f800000u) return 0x7fc0;   // NaN
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
__device__ __forceinline__ float bf2f(uint16_t b) { return __uint_as_float((uint32_t) b << 16); }

__global__ void to_bf16_kernel(const float* x, size_t n, uint16_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f2bf_rn(x[i]);
}

// F32 -> FP8 E4M3 pack (saturating at +-448, f32_to_e4m3fn): the drafter's kv_fp8 context caches.
__global__ void pack_e4m3_kernel(const float* x, size_t n, uint8_t* y) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = f32_to_e4m3fn(x[i]);
}

__global__ void mix_rms_kernel(const float* x, int n_tap, size_t tap_stride, int C, const float* alpha, const float* w,
                               float eps, float* out) {
    __shared__ float sh[kBlk];
    const int row = blockIdx.x;
    float a = 0.0f;
    for (int c = threadIdx.x; c < C; c += kBlk) {
        float m = 0.0f;
        for (int t = 0; t < n_tap; ++t) m += alpha[t] * x[(size_t) t * tap_stride + (size_t) row * C + c];
        out[(size_t) row * C + c] = m;
        a += m * m;
    }
    const float r = rsqrtf(bsum(a, sh) / (float) C + eps);
    for (int c = threadIdx.x; c < C; c += kBlk) out[(size_t) row * C + c] *= r * w[c];
}

__global__ void rope_neox_kernel(float* x, int heads, const int* pos, float theta) {
    const int row = blockIdx.x, h = blockIdx.y, i = threadIdx.x;   // i < 64
    const float inv = 1.0f / powf(theta, (float) (2 * i) / 128.0f);
    const float ang = (float) pos[row] * inv;
    float sn, cs;
    sincosf(ang, &sn, &cs);
    float* p = x + ((size_t) row * heads + h) * 128;
    const float a = p[i], b = p[i + 64];
    p[i] = a * cs - b * sn;
    p[i + 64] = b * cs + a * sn;
}

// a block a (head, query): the logits a key a warp, softmax, the values
__global__ void draft_attn_kernel(const float* q, const float* kc, const float* vc, int k_lo, int k_hi, const float* kb,
                                  const float* vb, int nq, int H, int Hkv, float scale, float* scratch, float* out,
                                  int ring_mask) {
    __shared__ float sh[128];
    __shared__ float qs[128];
    const int h = blockIdx.x, qi = blockIdx.y, g = h / (H / Hkv), tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int KD = Hkv * 128, nc = k_hi - k_lo, nk = nc + nq;
    // ring_mask == 0 means no ring (full-ctx linear): all-ones so (& ctx_mask) is identity. A zero mask
    // here would alias every context position to row 0 and silently destroy drafter accuracy.
    const int ctx_mask = ring_mask ? ring_mask : 0x7fffffff;
    qs[tid] = q[((size_t) qi * H + h) * 128 + tid];
    __syncthreads();
    float* lg = scratch + ((size_t) h * nq + qi) * nk;
    float mx = -INFINITY;
    for (int j = warp; j < nk; j += 4) {
        const float* kr = j < nc ? kc + (size_t) ((k_lo + j) & ctx_mask) * KD + (size_t) g * 128
                                 : kb + (size_t) (j - nc) * KD + (size_t) g * 128;
        float a = qs[lane] * kr[lane] + qs[lane + 32] * kr[lane + 32] + qs[lane + 64] * kr[lane + 64] +
                  qs[lane + 96] * kr[lane + 96];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        a *= scale;
        if (lane == 0) lg[j] = a;
        mx = fmaxf(mx, a);
    }
    __syncthreads();
    const float m = bmax(mx, sh);
    float a = 0.0f;
    for (int j = tid; j < nk; j += 128) {
        const float e = expf(lg[j] - m);
        lg[j] = e;
        a += e;
    }
    const float inv = 1.0f / bsum(a, sh);
    __syncthreads();
    float o = 0.0f;
    for (int j = 0; j < nk; ++j) {
        const float* vr = j < nc ? vc + (size_t) ((k_lo + j) & ctx_mask) * KD : vb + (size_t) (j - nc) * KD;
        o += lg[j] * vr[(size_t) g * 128 + tid];
    }
    out[((size_t) qi * H + h) * 128 + tid] = o * inv;
}

// IK2_DRAFT_KV_FP8: draft_attn over FP8 E4M3 context caches (kc/vc, 1 byte per element); the block's
// own kb/vb stay F32. Same grid (H, nq) and same math as draft_attn_kernel -- each context element is
// unpacked e4m3 -> F32 (e4m3fn_to_f32) before the dot / PV, so context K/V bytes drop 4x at the cost
// of a little ALU. NOT bit-exact vs the F32 cache: quantization happened at write_context's pack.
__global__ void draft_attn_fp8_kernel(const float* q, const uint8_t* kc, const uint8_t* vc, int k_lo, int k_hi,
                                      const float* kb, const float* vb, int nq, int H, int Hkv, int ring_mask,
                                      float scale, float* scratch, float* out) {
    __shared__ float sh[128];
    __shared__ float qs[128];
    const int h = blockIdx.x, qi = blockIdx.y, g = h / (H / Hkv), tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int KD = Hkv * 128, nc = k_hi - k_lo, nk = nc + nq;
    qs[tid] = q[((size_t) qi * H + h) * 128 + tid];
    __syncthreads();
    float* lg = scratch + ((size_t) h * nq + qi) * nk;
    float mx = -INFINITY;
    for (int j = warp; j < nk; j += 4) {
        float k0, k1, k2, k3;
        if (j < nc) {   // context row: E4M3 bytes, unpack to F32
            // ring_mask > 0: the local layers keep a ring cache, so position p lives at (p & ring_mask)
            const int kp = ring_mask > 0 ? ((k_lo + j) & ring_mask) : (k_lo + j);
            const uint8_t* kr = kc + (size_t) kp * KD + (size_t) g * 128;
            k0 = e4m3fn_to_f32(kr[lane]);
            k1 = e4m3fn_to_f32(kr[lane + 32]);
            k2 = e4m3fn_to_f32(kr[lane + 64]);
            k3 = e4m3fn_to_f32(kr[lane + 96]);
        } else {        // the block's own row: still F32
            const float* kr = kb + (size_t) (j - nc) * KD + (size_t) g * 128;
            k0 = kr[lane];
            k1 = kr[lane + 32];
            k2 = kr[lane + 64];
            k3 = kr[lane + 96];
        }
        float a = qs[lane] * k0 + qs[lane + 32] * k1 + qs[lane + 64] * k2 + qs[lane + 96] * k3;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
        a *= scale;
        if (lane == 0) lg[j] = a;
        mx = fmaxf(mx, a);
    }
    __syncthreads();
    const float m = bmax(mx, sh);
    float a = 0.0f;
    for (int j = tid; j < nk; j += 128) {
        const float e = expf(lg[j] - m);
        lg[j] = e;
        a += e;
    }
    const float inv = 1.0f / bsum(a, sh);
    __syncthreads();
    float o = 0.0f;
    for (int j = 0; j < nk; ++j) {
        const int kp = ring_mask > 0 ? ((k_lo + j) & ring_mask) : (k_lo + j);
        const float v = j < nc ? e4m3fn_to_f32(vc[(size_t) kp * KD + (size_t) g * 128 + tid])
                               : vb[(size_t) (j - nc) * KD + (size_t) g * 128 + tid];
        o += lg[j] * v;
    }
    out[((size_t) qi * H + h) * 128 + tid] = o * inv;
}

// GQA-shared draft attention (IK2_DRAFT_ATTN_GQA, opt-in): one block per (KV head g, query qi) serves
// the R query heads h = g*R + r, so each K/V row of head g is read once instead of R times (R = 3 for
// the drafter's H = 24, Hkv = 8: 3x less K/V traffic, the bottleneck at long context). The per-head
// arithmetic is identical to draft_attn_kernel: same 4-term lane dot expression, same shfl_xor
// butterfly, same bmax/bsum reduction trees, same ascending-j PV accumulation, and each head keeps
// its own independent softmax (m, sum) -> bit-exact outputs; only the reads are shared. R is a
// template parameter so the per-head register arrays stay in registers (unrolled, compile-time).
template <int R>
__global__ void draft_attn_gqa_kernel(const float* q, const float* kc, const float* vc, int k_lo, int k_hi,
                                      const float* kb, const float* vb, int nq, int H, int Hkv, float scale,
                                      float* scratch, float* out) {
    __shared__ float sh[128];
    __shared__ float qs[R][128];
    const int g = blockIdx.x, qi = blockIdx.y, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int KD = Hkv * 128, nc = k_hi - k_lo, nk = nc + nq;
#pragma unroll
    for (int r = 0; r < R; ++r) qs[r][tid] = q[((size_t) qi * H + g * R + r) * 128 + tid];
    __syncthreads();
    // same per-(h, qi) scratch layout as draft_attn_kernel: lg[r] = scratch[(h * nq + qi) * nk]
    float* lg[R];
    float mx[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        lg[r] = scratch + ((size_t) (g * R + r) * nq + qi) * nk;
        mx[r] = -INFINITY;
    }
    for (int j = warp; j < nk; j += 4) {
        const float* kr = j < nc ? kc + (size_t) (k_lo + j) * KD + (size_t) g * 128
                                 : kb + (size_t) (j - nc) * KD + (size_t) g * 128;
        // the K row is read once, then shared by the R dot products
        const float k0 = kr[lane], k1 = kr[lane + 32], k2 = kr[lane + 64], k3 = kr[lane + 96];
#pragma unroll
        for (int r = 0; r < R; ++r) {
            float a = qs[r][lane] * k0 + qs[r][lane + 32] * k1 + qs[r][lane + 64] * k2 + qs[r][lane + 96] * k3;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) a += __shfl_xor_sync(0xffffffffu, a, o);
            a *= scale;
            if (lane == 0) lg[r][j] = a;
            mx[r] = fmaxf(mx[r], a);
        }
    }
    __syncthreads();
    float inv[R];
#pragma unroll
    for (int r = 0; r < R; ++r) {
        // bmax/bsum each end with __syncthreads, so sequential calls reuse sh safely and every
        // head's softmax stays independent (its own m and sum, identical trees to the original).
        const float m = bmax(mx[r], sh);
        float a = 0.0f;
        for (int j = tid; j < nk; j += 128) {
            const float e = expf(lg[r][j] - m);
            lg[r][j] = e;
            a += e;
        }
        inv[r] = 1.0f / bsum(a, sh);
    }
    __syncthreads();
    float o[R];
#pragma unroll
    for (int r = 0; r < R; ++r) o[r] = 0.0f;
    for (int j = 0; j < nk; ++j) {
        const float* vr = j < nc ? vc + (size_t) (k_lo + j) * KD : vb + (size_t) (j - nc) * KD;
        // the V element is read once, then shared by the R accumulators
        const float v = vr[(size_t) g * 128 + tid];
#pragma unroll
        for (int r = 0; r < R; ++r) o[r] += lg[r][j] * v;
    }
#pragma unroll
    for (int r = 0; r < R; ++r) out[((size_t) qi * H + g * R + r) * 128 + tid] = o[r] * inv[r];
}

__global__ void silu_mul_bf16_kernel(const float* gu, int T, int F, uint16_t* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t) T * F) return;
    const size_t t = i / F, f = i % F;
    const float g = gu[t * 2 * F + f], u = gu[t * 2 * F + F + f];
    out[i] = f2bf_rn(g / (1.0f + expf(-g)) * u);
}

__global__ void rnn_inputs_kernel(const float* state, const uint16_t* emb, const int* prev, const float* hidden,
                                  const float* f, int R, int C, uint16_t* out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int W = 2 * R + 2 * C;
    if (i >= W) return;
    float v;
    if (i < R) v = state[i];
    else if (i < 2 * R) v = bf2f(emb[(size_t) (*prev) * R + (i - R)]);
    else if (i < 2 * R + C) v = hidden[i - 2 * R];
    else v = f[i - 2 * R - C];
    out[i] = f2bf_rn(v);
}

__global__ void rnn_gates_kernel(const float* proj, const float* bias, int R, float* state, float* latent) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= R) return;
    const float gr = proj[i] + bias[i], cr = proj[R + i] + bias[R + i], orw = proj[2 * R + i] + bias[2 * R + i];
    const float gate = 1.0f / (1.0f + expf(-gr));
    state[i] = gate * state[i] + (1.0f - gate) * tanhf(cr);
    latent[i] = tanhf(orw);
}

constexpr int kArgmaxMaxPartials = 1024;
__device__ float g_argmax_v[kArgmaxMaxPartials];
__device__ int g_argmax_i[kArgmaxMaxPartials];

__device__ __forceinline__ void argmax_better(float v, int i, float& best, int& idx) {
    if (v > best || (v == best && i < idx)) best = v, idx = i;
}

__device__ __forceinline__ void argmax_block_reduce(float& best, int& idx) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float ob = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
        argmax_better(ob, oi, best, idx);
    }
    __shared__ float bv[32];
    __shared__ int bi[32];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    if (lane == 0) bv[warp] = best, bi[warp] = idx;
    __syncthreads();
    if (warp == 0) {
        best = (lane < nw) ? bv[lane] : -INFINITY;
        idx = (lane < nw) ? bi[lane] : 0x7fffffff;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) {
            const float ob = __shfl_xor_sync(0xffffffffu, best, o);
            const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
            argmax_better(ob, oi, best, idx);
        }
    }
}

template <bool SINGLE_BLOCK>
__global__ void argmax_sum_part_kernel(const float* a, const float* b, const float* c, int n, int* out) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    const bool vec4 = (((uintptr_t) a & 15) == 0) && (!b || ((uintptr_t) b & 15) == 0) && (!c || ((uintptr_t) c & 15) == 0);
    if (vec4) {
        const int n4 = n >> 2;
        const float4* a4 = (const float4*) a;
        const float4* b4 = (const float4*) b;
        const float4* c4 = (const float4*) c;
        for (int v4 = gtid; v4 < n4; v4 += gstride) {
            const float4 va = __ldg(a4 + v4);
            const float4 vb = b4 ? __ldg(b4 + v4) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 vc = c4 ? __ldg(c4 + v4) : make_float4(0.f, 0.f, 0.f, 0.f);
            const int base = v4 << 2;
            argmax_better(va.x + vb.x + vc.x, base + 0, best, idx);
            argmax_better(va.y + vb.y + vc.y, base + 1, best, idx);
            argmax_better(va.z + vb.z + vc.z, base + 2, best, idx);
            argmax_better(va.w + vb.w + vc.w, base + 3, best, idx);
        }
        for (int v = (n4 << 2) + gtid; v < n; v += gstride)
            argmax_better(__ldg(a + v) + (b ? __ldg(b + v) : 0.0f) + (c ? __ldg(c + v) : 0.0f), v, best, idx);
    } else {
        for (int v = gtid; v < n; v += gstride)
            argmax_better(__ldg(a + v) + (b ? __ldg(b + v) : 0.0f) + (c ? __ldg(c + v) : 0.0f), v, best, idx);
    }
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) {
        if (SINGLE_BLOCK) *out = idx;
        else g_argmax_v[blockIdx.x] = best, g_argmax_i[blockIdx.x] = idx;
    }
}

// IK2_AR_STREAM2: the multi-block partials above are process-wide globals. Two AR slot graphs replaying on two
// streams would mix their 197 padded-vocab partials. This scratch variant keeps the exact same reduction math and
// launch geometry but writes the partials into caller-owned memory (one buffer per Work). Only the fast T=1 graph
// uses it; every other caller keeps the legacy globals and remains byte-identical.
template <bool SINGLE_BLOCK>
__global__ void argmax_sum_part_scratch_kernel(const float* a, const float* b, const float* c, int n, int* out,
                                               float* part_v, int* part_i) {
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    const bool vec4 = (((uintptr_t) a & 15) == 0) && (!b || ((uintptr_t) b & 15) == 0) && (!c || ((uintptr_t) c & 15) == 0);
    if (vec4) {
        const int n4 = n >> 2;
        const float4* a4 = (const float4*) a;
        const float4* b4 = (const float4*) b;
        const float4* c4 = (const float4*) c;
        for (int v4 = gtid; v4 < n4; v4 += gstride) {
            const float4 va = __ldg(a4 + v4);
            const float4 vb = b4 ? __ldg(b4 + v4) : make_float4(0.f, 0.f, 0.f, 0.f);
            const float4 vc = c4 ? __ldg(c4 + v4) : make_float4(0.f, 0.f, 0.f, 0.f);
            const int base = v4 << 2;
            argmax_better(va.x + vb.x + vc.x, base + 0, best, idx);
            argmax_better(va.y + vb.y + vc.y, base + 1, best, idx);
            argmax_better(va.z + vb.z + vc.z, base + 2, best, idx);
            argmax_better(va.w + vb.w + vc.w, base + 3, best, idx);
        }
        for (int v = (n4 << 2) + gtid; v < n; v += gstride)
            argmax_better(__ldg(a + v) + (b ? __ldg(b + v) : 0.0f) + (c ? __ldg(c + v) : 0.0f), v, best, idx);
    } else {
        for (int v = gtid; v < n; v += gstride)
            argmax_better(__ldg(a + v) + (b ? __ldg(b + v) : 0.0f) + (c ? __ldg(c + v) : 0.0f), v, best, idx);
    }
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) {
        if (SINGLE_BLOCK) *out = idx;
        else part_v[blockIdx.x] = best, part_i[blockIdx.x] = idx;
    }
}

template <bool SINGLE_BLOCK>
__global__ void argmax_rows_part_kernel(const float* x, int ld, int n_cols, int* out) {
    const int r = blockIdx.y;
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;
    const float* row = x + (size_t) r * ld;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    if (((uintptr_t) row & 15) == 0) {
        const int n4 = n_cols >> 2;
        const float4* r4 = (const float4*) row;
        for (int v4 = gtid; v4 < n4; v4 += gstride) {
            const float4 f = __ldg(r4 + v4);
            const int base = v4 << 2;
            argmax_better(f.x, base + 0, best, idx);
            argmax_better(f.y, base + 1, best, idx);
            argmax_better(f.z, base + 2, best, idx);
            argmax_better(f.w, base + 3, best, idx);
        }
        for (int v = (n4 << 2) + gtid; v < n_cols; v += gstride) argmax_better(__ldg(row + v), v, best, idx);
    } else {
        for (int v = gtid; v < n_cols; v += gstride) argmax_better(__ldg(row + v), v, best, idx);
    }
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) {
        if (SINGLE_BLOCK) out[r] = idx;
        else g_argmax_v[r * gridDim.x + blockIdx.x] = best, g_argmax_i[r * gridDim.x + blockIdx.x] = idx;
    }
}

template <bool SINGLE_BLOCK>
__global__ void argmax_gemv_f16_part_kernel(const uint16_t* W, int ld, const float* x, const float* base,
                                            const uint16_t* bias_f16, int R, int n, bool use_smem, int* out) {
    extern __shared__ float sx[];
    if (use_smem) {
        for (int r = threadIdx.x; r < R; r += blockDim.x) sx[r] = x[r];
        __syncthreads();
    }
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gstride = gridDim.x * blockDim.x;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    const bool vec4 = (((uintptr_t) W & 7) == 0) && ((ld & 3) == 0) &&
                      (!base || ((uintptr_t) base & 15) == 0) && (!bias_f16 || ((uintptr_t) bias_f16 & 7) == 0);
    if (vec4) {
        const int n4 = n >> 2;
        for (int v4 = gtid; v4 < n4; v4 += gstride) {
            const int v = v4 << 2;
            float d0 = 0.0f, d1 = 0.0f, d2 = 0.0f, d3 = 0.0f;
            const uint16_t* wr = W + v;
            for (int r = 0; r < R; ++r, wr += ld) {
                const float xr = use_smem ? sx[r] : __ldg(x + r);
                const uint2 w4 = __ldg((const uint2*) wr);
                const float2 f0 = u32_to_f2(w4.x), f1 = u32_to_f2(w4.y);
                d0 = fmaf(f0.x, xr, d0);
                d1 = fmaf(f0.y, xr, d1);
                d2 = fmaf(f1.x, xr, d2);
                d3 = fmaf(f1.y, xr, d3);
            }
            if (base) {
                const float4 bs = __ldg((const float4*) (base + v));
                d0 += bs.x, d1 += bs.y, d2 += bs.z, d3 += bs.w;
            }
            if (bias_f16) {
                const uint2 b4 = __ldg((const uint2*) (bias_f16 + v));
                const float2 f0 = u32_to_f2(b4.x), f1 = u32_to_f2(b4.y);
                d0 += f0.x, d1 += f0.y, d2 += f1.x, d3 += f1.y;
            }
            argmax_better(d0, v + 0, best, idx);
            argmax_better(d1, v + 1, best, idx);
            argmax_better(d2, v + 2, best, idx);
            argmax_better(d3, v + 3, best, idx);
        }
        for (int v = (n4 << 2) + gtid; v < n; v += gstride) {
            float d = 0.0f;
            const uint16_t* wr = W + v;
            for (int r = 0; r < R; ++r, wr += ld)
                d = fmaf(h2f(__ldg(wr)), use_smem ? sx[r] : __ldg(x + r), d);
            if (base) d += __ldg(base + v);
            if (bias_f16) d += h2f(__ldg(bias_f16 + v));
            argmax_better(d, v, best, idx);
        }
    } else {
        for (int v = gtid; v < n; v += gstride) {
            float d = 0.0f;
            const uint16_t* wr = W + v;
            for (int r = 0; r < R; ++r, wr += ld)
                d = fmaf(h2f(__ldg(wr)), use_smem ? sx[r] : __ldg(x + r), d);
            if (base) d += __ldg(base + v);
            if (bias_f16) d += h2f(__ldg(bias_f16 + v));
            argmax_better(d, v, best, idx);
        }
    }
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) {
        if (SINGLE_BLOCK) *out = idx;
        else g_argmax_v[blockIdx.x] = best, g_argmax_i[blockIdx.x] = idx;
    }
}

__global__ void argmax_final_kernel(int n_partials, int stride_partials, int* out) {
    const int r = blockIdx.x;
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int j = threadIdx.x; j < n_partials; j += blockDim.x)
        argmax_better(g_argmax_v[r * stride_partials + j], g_argmax_i[r * stride_partials + j], best, idx);
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) out[r] = idx;
}

__global__ void argmax_sum_final_scratch_kernel(int n_partials, int* out, const float* part_v, const int* part_i) {
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int j = threadIdx.x; j < n_partials; j += blockDim.x)
        argmax_better(part_v[j], part_i[j], best, idx);
    argmax_block_reduce(best, idx);
    if (threadIdx.x == 0) *out = idx;
}

// fast FP16 decodes: NVFP4 a thread a 16-value sub-block (__byte_perm + uint4 stores); Q8_0 8 threads a 32-value block
// (uint2 stores); Q5_K / Q6_K a warp a 256-value block (4 values/lane with coalesced uint2 stores)
__global__ void decode_f16_nvfp4_kernel(const uint8_t* blocks, size_t nsub, uint16_t* out) {
    const size_t sb = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= nsub) return;
    const uint8_t* b = blocks + (sb >> 2) * 36;
    const int sub = (int) (sb & 3);
    const float d = ue4m3((uint8_t) (*(const uint32_t*) b >> (8 * sub)));
    const uint32_t q0 = *(const uint32_t*) (b + 4 + sub * 8), q1 = *(const uint32_t*) (b + 8 + sub * 8);
    const float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
    const float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
    const float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
    const float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
    const uint4 o0 = make_uint4(f2_to_u32(w01.x * d, w01.y * d), f2_to_u32(w23.x * d, w23.y * d),
                                f2_to_u32(w45.x * d, w45.y * d), f2_to_u32(w67.x * d, w67.y * d));
    const uint4 o1 = make_uint4(f2_to_u32(w89.x * d, w89.y * d), f2_to_u32(wAB.x * d, wAB.y * d),
                                f2_to_u32(wCD.x * d, wCD.y * d), f2_to_u32(wEF.x * d, wEF.y * d));
    uint4* o = (uint4*) (out + sb * 16);
    o[0] = o0;
    o[1] = o1;
}

// NVFP4 -> FP16 expand, one thread per 64-value group (a 36-byte block: 4 E4M3 scales + 32 packed codes).
// Four contiguous sub-blocks share the block read; fewer threads and wider stores than the per-16 kernel,
// which the prefill's per-expert weight expand launches at ~1.7 TB/s. Same values, same order.
__global__ void decode_f16_nvfp4_x4_kernel(const uint8_t* blocks, size_t ngrp, uint16_t* out) {
    const size_t g = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= ngrp) return;
    const uint8_t* b = blocks + g * 36;
    const uint32_t sc4 = *(const uint32_t*) b;
    uint16_t* o = out + g * 64;
#pragma unroll
    for (int sub = 0; sub < 4; ++sub) {
        const float d = ue4m3((uint8_t) (sc4 >> (8 * sub)));
        const uint32_t q0 = *(const uint32_t*) (b + 4 + sub * 8), q1 = *(const uint32_t*) (b + 8 + sub * 8);
        const float2 w01 = e2m1_pair_lo(q0), w23 = e2m1_pair_lo(q0 >> 16);
        const float2 w45 = e2m1_pair_lo(q1), w67 = e2m1_pair_lo(q1 >> 16);
        const float2 w89 = e2m1_pair_hi(q0), wAB = e2m1_pair_hi(q0 >> 16);
        const float2 wCD = e2m1_pair_hi(q1), wEF = e2m1_pair_hi(q1 >> 16);
        uint4* oo = (uint4*) (o + sub * 16);
        oo[0] = make_uint4(f2_to_u32(w01.x * d, w01.y * d), f2_to_u32(w23.x * d, w23.y * d),
                           f2_to_u32(w45.x * d, w45.y * d), f2_to_u32(w67.x * d, w67.y * d));
        oo[1] = make_uint4(f2_to_u32(w89.x * d, w89.y * d), f2_to_u32(wAB.x * d, wAB.y * d),
                           f2_to_u32(wCD.x * d, wCD.y * d), f2_to_u32(wEF.x * d, wEF.y * d));
    }
}

__global__ void decode_f16_q8_kernel(const uint8_t* blocks, size_t nblk, uint16_t* out) {
    const size_t tid_g = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    const size_t bi = tid_g >> 3;
    if (bi >= nblk) return;
    const int sub = (int) (tid_g & 7);
    const uint8_t* b = blocks + bi * 34;
    const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
    const uint8_t* qp = b + 2 + sub * 4;
    int8_t q0, q1, q2, q3;
    if (((uintptr_t) qp & 3) == 0) {
        const uint32_t q4 = *(const uint32_t*) qp;
        q0 = (int8_t) (q4 & 0xFFu);
        q1 = (int8_t) ((q4 >> 8) & 0xFFu);
        q2 = (int8_t) ((q4 >> 16) & 0xFFu);
        q3 = (int8_t) (q4 >> 24);
    } else {
        q0 = (int8_t) qp[0], q1 = (int8_t) qp[1], q2 = (int8_t) qp[2], q3 = (int8_t) qp[3];
    }
    uint16_t* dst = out + bi * 32 + sub * 4;
    if (((uintptr_t) dst & 7) == 0) {
        *(uint2*) dst = make_uint2(f2_to_u32(d * (float) q0, d * (float) q1),
                                   f2_to_u32(d * (float) q2, d * (float) q3));
    } else {
        dst[0] = f2h(d * (float) q0), dst[1] = f2h(d * (float) q1);
        dst[2] = f2h(d * (float) q2), dst[3] = f2h(d * (float) q3);
    }
}

__device__ __forceinline__ void store_f16x4(uint16_t* dst, float v0, float v1, float v2, float v3) {
    if (((uintptr_t) dst & 7) == 0) {
        *(uint2*) dst = make_uint2(f2_to_u32(v0, v1), f2_to_u32(v2, v3));
    } else {
        dst[0] = f2h(v0), dst[1] = f2h(v1), dst[2] = f2h(v2), dst[3] = f2h(v3);
    }
}

// one thread per 32-value sub-block of IQ3_XXS / IQ4_XS -> 32 FP16 (ggml's dequantize_row_iq3_xxs / iq4_xs values)
__global__ void decode_f16_iq_kernel(int type, const uint8_t* blocks, size_t nsub, uint16_t* out) {
    const size_t sb = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= nsub) return;
    const int ib = (int) (sb & 7);
    float y[32];
    if (type == kIQ3_XXS) {
        const uint8_t* b = blocks + (sb >> 3) * 98;
        const float d = half_bits_to_float(__ldg((const uint16_t*) b));
        const uint16_t* qs = (const uint16_t*) (b + 2 + 8 * ib);
        const uint16_t* sp = (const uint16_t*) (b + 66 + 4 * ib);
        const uint32_t aux = (uint32_t) __ldg(sp) | ((uint32_t) __ldg(sp + 1) << 16);
        const float db = d * (0.5f + (float) (aux >> 28)) * 0.5f;
#pragma unroll
        for (int l = 0; l < 4; ++l) {
            const uint32_t qq = __ldg(qs + l);
            const uint32_t g1 = __ldg(d_iq3xxs_grid + (qq & 0xFFu)), g2 = __ldg(d_iq3xxs_grid + (qq >> 8));
            const uint32_t sg = iq_signs8((aux >> (7 * l)) & 127u);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                y[8 * l + j] = db * (float) ((g1 >> (8 * j)) & 0xFFu) * ((sg >> j) & 1u ? -1.0f : 1.0f);
                y[8 * l + 4 + j] = db * (float) ((g2 >> (8 * j)) & 0xFFu) * ((sg >> (4 + j)) & 1u ? -1.0f : 1.0f);
            }
        }
    } else {
        const uint8_t* b = blocks + (sb >> 3) * 136;
        const uint32_t h = __ldg((const uint32_t*) b), sl = __ldg((const uint32_t*) (b + 4));
        const float d = half_bits_to_float((uint16_t) (h & 0xFFFFu));
        const int ls = (int) ((sl >> (4 * ib)) & 0xFu) | (int) (((h >> (16 + 2 * ib)) & 3u) << 4);
        const float dl = d * (float) (ls - 32);
        const uint8_t* q = b + 8 + 16 * ib;
#pragma unroll
        for (int j = 0; j < 16; ++j) {
            const uint8_t v = __ldg(q + j);
            y[j] = dl * (float) d_iq4nl[v & 0xF];
            y[16 + j] = dl * (float) d_iq4nl[v >> 4];
        }
    }
    uint32_t* o = (uint32_t*) (out + sb * 32);
#pragma unroll
    for (int j = 0; j < 16; ++j) o[j] = (uint32_t) f2h(y[2 * j]) | ((uint32_t) f2h(y[2 * j + 1]) << 16);
}

__global__ void decode_f16_k_kernel(int type, const uint8_t* blocks, size_t nblk, uint16_t* out) {
    const size_t bi = (size_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (bi >= nblk) return;
    const int lane = threadIdx.x & 31;
    const int gr = lane >> 3, l0 = (lane & 7) * 4;
    uint16_t* y = out + bi * 256;
    if (type == kQ6_K) {
        const uint8_t* b = blocks + bi * 210;
        const float d = half_bits_to_float((uint16_t) (b[208] | (b[209] << 8)));
        const int8_t* sc = (const int8_t*) (b + 192);
        const int qd = gr;
        const int sh = 2 * qd;
#pragma unroll
        for (int n = 0; n < 2; ++n) {
            const float d_sc = d * (float) sc[n * 8 + (l0 >> 4) + 2 * qd];
            const uint8_t* qlp = b + n * 64 + ((qd & 1) ? 32 : 0) + l0;
            const uint8_t* qhp = b + 128 + n * 32 + l0;
            uint8_t lo[4], qh[4];
            if (((uintptr_t) qlp & 1) == 0) {
                const uint16_t lo0 = *(const uint16_t*) qlp, lo1 = *(const uint16_t*) (qlp + 2);
                const uint16_t qh0 = *(const uint16_t*) qhp, qh1 = *(const uint16_t*) (qhp + 2);
                lo[0] = (uint8_t) lo0, lo[1] = (uint8_t) (lo0 >> 8);
                lo[2] = (uint8_t) lo1, lo[3] = (uint8_t) (lo1 >> 8);
                qh[0] = (uint8_t) qh0, qh[1] = (uint8_t) (qh0 >> 8);
                qh[2] = (uint8_t) qh1, qh[3] = (uint8_t) (qh1 >> 8);
            } else {
                for (int k = 0; k < 4; ++k) lo[k] = qlp[k], qh[k] = qhp[k];
            }
            float f[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int nib = qd >= 2 ? (lo[k] >> 4) : (lo[k] & 0xF);
                const int q = (int) (nib | (((qh[k] >> sh) & 3) << 4)) - 32;
                f[k] = d_sc * (float) q;
            }
            store_f16x4(y + n * 128 + qd * 32 + l0, f[0], f[1], f[2], f[3]);
        }
    } else {   // Q5_K
        const uint8_t* b = blocks + bi * 176;
        const float d = half_bits_to_float((uint16_t) (b[0] | (b[1] << 8)));
        const float mn = half_bits_to_float((uint16_t) (b[2] | (b[3] << 8)));
        const uint8_t* q = b + 4;
        const uint8_t* qhp = b + 16 + l0;
        const uint32_t qh4 = (((uintptr_t) qhp & 3) == 0)
                                 ? *(const uint32_t*) qhp
                                 : ((uint32_t) qhp[0] | ((uint32_t) qhp[1] << 8) | ((uint32_t) qhp[2] << 16) |
                                    ((uint32_t) qhp[3] << 24));
#pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int is = pass * 4 + gr, j64 = is >> 1, half = is & 1;
            int sc, m;
            if (is < 4) {
                sc = q[is] & 63, m = q[is + 4] & 63;
            } else {
                sc = (q[is + 4] & 0xF) | ((q[is - 4] >> 6) << 4), m = (q[is + 4] >> 4) | ((q[is] >> 6) << 4);
            }
            const float dsc = d * (float) sc, mnm = mn * (float) m;
            const uint8_t* lop = b + 48 + 32 * j64 + l0;
            const uint32_t lo4 = (((uintptr_t) lop & 3) == 0)
                                     ? *(const uint32_t*) lop
                                     : ((uint32_t) lop[0] | ((uint32_t) lop[1] << 8) | ((uint32_t) lop[2] << 16) |
                                        ((uint32_t) lop[3] << 24));
            float f[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const uint32_t lo_k = (lo4 >> (8 * k)) & 0xFFu;
                const uint32_t qh_k = (qh4 >> (8 * k)) & 0xFFu;
                const int val = (half ? (lo_k >> 4) : (lo_k & 0xF)) + (((qh_k >> is) & 1) ? 16 : 0);
                f[k] = dsc * (float) val - mnm;
            }
            store_f16x4(y + is * 32 + l0, f[0], f[1], f[2], f[3]);
        }
    }
}

__global__ void decode_f16_kernel(int type, const uint8_t* blocks, size_t n, uint16_t* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = f2h(decode_value(type, blocks, i));
}
__global__ void decode_f32_kernel(int type, const uint8_t* blocks, size_t n, float* out) {
    const size_t i = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = decode_value(type, blocks, i);
}

// IK2_AR_KSTEP: gather one or more embedding rows straight from the device-resident Q8_0 table.  One thread per
// output float; the guard keeps out-of-range token ids exactly at the host decode_host() fallback (0.0f).  This calls
// the SAME decode_value() as the host path, so the values are bit-identical (no FP16 round trip).
__global__ void embed_gather_f32_kernel(int type, const uint8_t* table, int n_rows, size_t row_bytes, int C,
                                        const int* tok, int n_tokens, float* out) {
    const int t = blockIdx.y;
    const int c = (int) (blockIdx.x * blockDim.x + threadIdx.x);
    if (t >= n_tokens || c >= C) return;
    const int id = tok[t];
    if (id < 0 || id >= n_rows) { out[(size_t) t * C + c] = 0.0f; return; }
    out[(size_t) t * C + c] = decode_value(type, table + (size_t) id * row_bytes, (size_t) c);
}

// IK2_AR_KSTEP: advance the graph's staged base position by one.  One thread, one int; queued on the token stream
// between K-step argmax writes, so the next step reads base+1 without a host round trip.
__global__ void pos_inc_kernel(int* pos) { if (threadIdx.x == 0) ++*pos; }

// A looped core's pass > 0 bank copies in one launch (the per-copy cudaMemcpyAsync's of the engine's two token
// paths): bank i < n_bank - 1 is block nb_pre + 1 + i (blocks + i * blk_stride), bank n_bank - 1 is h; then
// h_out = lp_pre (h_out null: no restore).  A thread copies one float of a row of `row` floats.  The h bank copy
// and the restore of the same element run on the SAME thread in that order, so h is read before it is overwritten
// without a grid barrier; the block copies touch neither h nor lp_pre.
__global__ void loop_bank_copies_kernel(const float* lp_pre, const float* h, const float* blocks, size_t blk_stride,
                                        int nb_pre, int n_bank, float* lp_bank, float* h_out, size_t row) {
    const size_t c = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= row) return;
    const int bank = blockIdx.y;
    if (bank < n_bank - 1) {
        lp_bank[(size_t) bank * row + c] = blocks[(size_t) (nb_pre + 1 + bank) * blk_stride + c];
    } else {
        lp_bank[(size_t) bank * row + c] = h[c];
        if (h_out) h_out[c] = lp_pre[c];
    }
}

unsigned grid(size_t n, int b = kBlk) { return (unsigned) ((n + b - 1) / b); }

}  // namespace

void rms(const float* x, const float* w, int rows, int D, float eps, float* y, cudaStream_t s) {
    if (rows <= 0) return;
    rms_impl_kernel<false, true><<<rows, kBlk, 0, s>>>(x, w, D, eps, nullptr, y);
    check("rms");
}
void rms_multi(const RmsJobs& jobs, int n, int D, float eps, cudaStream_t s) {
    if (n <= 0) return;
    if (n > 3) {
        std::fprintf(stderr, "ik2::k::rms_multi: %d vectors\n", n);
        std::abort();
    }
    rms_multi_kernel<<<(unsigned) n, kBlk, 0, s>>>(jobs, D, eps);
    check("rms_multi");
}
void rms_f16(const float* x, const float* w, int rows, int D, float eps, uint16_t* y16, float* y32, cudaStream_t s) {
    if (rows <= 0) return;
    if (y16 && y32) rms_impl_kernel<true, true><<<rows, kBlk, 0, s>>>(x, w, D, eps, y16, y32);
    else if (y16) rms_impl_kernel<true, false><<<rows, kBlk, 0, s>>>(x, w, D, eps, y16, nullptr);
    else if (y32) rms_impl_kernel<false, true><<<rows, kBlk, 0, s>>>(x, w, D, eps, nullptr, y32);
    check("rms_f16");
}

void l2clip(const float* x, const float* scale, int rows, int G, int D, float th, const int* pos, float alpha,
            float n_floor, float* y, cudaStream_t s) {
    if (rows <= 0) return;
    l2clip_kernel<<<rows * G, 128, 0, s>>>(x, scale, G, D, th, pos, alpha, n_floor, y);
    check("l2clip");
}

void conv4_res(const float* x, float* hist, const float* w, const float* dyn, int heads, int T, int C,
               const float* residual, float* y, cudaStream_t s) {
    if (T <= 0) return;
    if (T <= kConvFusedMaxT) {
        conv4_fused_kernel<<<(unsigned) ((C + 127) / 128), 128, 0, s>>>(x, hist, w, dyn, heads, T, C, residual, y);
        check("conv4_res");
        return;
    }
    const dim3 g((unsigned) ((C + 127) / 128), (unsigned) ((T + kConvTileT - 1) / kConvTileT));
    conv4_tiled_kernel<<<g, 128, 0, s>>>(x, hist, w, dyn, heads, T, C, residual, y);
    check("conv4_res");
    conv4_hist_kernel<<<grid(C), kBlk, 0, s>>>(x, hist, T, C);
    check("conv4_hist");
}

void conv4(const float* x, float* hist, const float* w, const float* dyn, int heads, int T, int C, float* y,
           cudaStream_t s) {
    conv4_res(x, hist, w, dyn, heads, T, C, nullptr, y, s);
}

// Batched hosts: one launch per T regime (two for T > kConvFusedMaxT, where the scalar path also needs the
// separate history-fixup launch).  hist_tab is a DEVICE array of n_rows history pointers, each [3][C] floats.
void conv4_res_rows(const float* x, float* const* hist_tab, const float* w, const float* dyn, int heads, int T,
                    int C, const float* residual, float* y, int n_rows, cudaStream_t s) {
    if (T <= 0 || n_rows <= 0) return;
    if (T <= kConvFusedMaxT) {
        const dim3 g((unsigned) ((C + 127) / 128), 1, (unsigned) n_rows);
        conv4_fused_rows_kernel<<<g, 128, 0, s>>>(x, hist_tab, w, dyn, heads, T, C, residual, y);
        check("conv4_res_rows");
        return;
    }
    const dim3 g((unsigned) ((C + 127) / 128), (unsigned) ((T + kConvTileT - 1) / kConvTileT), (unsigned) n_rows);
    conv4_tiled_rows_kernel<<<g, 128, 0, s>>>(x, hist_tab, w, dyn, heads, T, C, residual, y);
    check("conv4_res_rows");
    const dim3 gh(grid(C), 1, (unsigned) n_rows);
    conv4_hist_rows_kernel<<<gh, kBlk, 0, s>>>(x, hist_tab, T, C);
    check("conv4_hist_rows");
}

void conv4_rows(const float* x, float* const* hist_tab, const float* w, const float* dyn, int heads, int T, int C,
                float* y, int n_rows, cudaStream_t s) {
    conv4_res_rows(x, hist_tab, w, dyn, heads, T, C, nullptr, y, n_rows, s);
}

template <bool FUSED_RMS>
void launch_attn_res(const ResStates& states, int n, const float* q, int nq, int T, int C, float res_eps, float rms_eps,
                     float* out, const RmsOuts& outs, cudaStream_t s, int jobs = 1) {
    if (T <= 0) return;
    const size_t job_step = (size_t) nq * T * C;
    const dim3 g((unsigned) T, (unsigned) jobs);
    if (n < 1 || n > kResMax || nq < 1 || nq > 3) {
        std::fprintf(stderr, "ik2::k::attn_res: n %d, nq %d\n", n, nq);
        std::abort();
    }
    bool vec4 = ((C & 3) == 0) && (((uintptr_t) q & 15) == 0);
    for (int i = 0; i < n && vec4; ++i) vec4 &= (((uintptr_t) states.st[i] & 15) == 0);
    if constexpr (!FUSED_RMS) {
        vec4 &= (((uintptr_t) out & 15) == 0);
    } else {
        for (int qi = 0; qi < nq && vec4; ++qi) {
            vec4 &= (!outs.w[qi] || (((uintptr_t) outs.w[qi] & 15) == 0)) &&
                    (!outs.y16[qi] || (((uintptr_t) outs.y16[qi] & 7) == 0)) &&
                    (!outs.y32[qi] || (((uintptr_t) outs.y32[qi] & 15) == 0));
        }
    }
    if (vec4 && C == 4608) {
        attn_res_vec4_kernel<FUSED_RMS, 576><<<g, 576, 0, s>>>(states, n, q, nq, T, C, res_eps, rms_eps, out, outs,
                                                               job_step);
    } else if (vec4 && C <= 4096) {
        attn_res_vec4_kernel<FUSED_RMS, 512><<<g, 512, 0, s>>>(states, n, q, nq, T, C, res_eps, rms_eps, out, outs,
                                                               job_step);
    } else {
        attn_res_scalar_kernel<FUSED_RMS><<<g, kBlk, 0, s>>>(states, n, q, nq, T, C, res_eps, rms_eps, out, outs,
                                                             job_step);
    }
    check(FUSED_RMS ? "attn_res_rms_f16" : "attn_res");
}

void attn_res(const ResStates& states, int n, const float* q, int nq, int T, int C, float eps, float* out,
              cudaStream_t s) {
    launch_attn_res<false>(states, n, q, nq, T, C, eps, 0.0f, out, RmsOuts{}, s);
}
void attn_res_jobs(const ResStates& states, int n, const float* q, int nq, int jobs, int C, float eps, float* out,
                   cudaStream_t s) {
    launch_attn_res<false>(states, n, q, nq, 1, C, eps, 0.0f, out, RmsOuts{}, s, jobs);
}

void loop_bank_copies(const float* lp_pre, const float* h, const float* blocks, size_t blk_stride, int nb_pre,
                      int nb, float* lp_bank, float* h_out, size_t row, cudaStream_t s) {
    const int n_bank = nb - nb_pre;
    if (n_bank < 1 || row == 0) return;
    const dim3 g(grid(row), (unsigned) n_bank);
    loop_bank_copies_kernel<<<g, kBlk, 0, s>>>(lp_pre, h, blocks, blk_stride, nb_pre, n_bank, lp_bank, h_out, row);
    check("loop_bank_copies");
}

void attn_res_rms_f16(const ResStates& states, int n, const float* q, int nq, int T, int C, float res_eps,
                      float rms_eps, const RmsOuts& outs, cudaStream_t s) {
    launch_attn_res<true>(states, n, q, nq, T, C, res_eps, rms_eps, nullptr, outs, s);
}

void attn_probs(float* lg, int Tq, int nk, int qpos0, int kpos0, float qk_scale, const float* r, int r_ld, int d_rel,
                const float* proj, int extent, int window, const float* sink, int sink_ld, cudaStream_t s) {
    if (Tq <= 0 || nk <= 0) return;
    attn_probs_kernel<<<Tq, kBlk, 0, s>>>(lg, nk, qpos0, kpos0, qk_scale, r, r_ld, d_rel, proj, extent, window, sink,
                                           sink_ld);
    check("attn_probs");
}
void attn_probs_bias_f16(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0, int kpos0,
                         float qk_scale, const float* bias, int extent, int window, const float* sink, cudaStream_t s) {
    if (H <= 0 || Tq <= 0 || nk <= 0) return;
    attn_probs_bias_f16_kernel<<<dim3((unsigned) Tq, (unsigned) H), kBlk, 0, s>>>(lg, out16, H, Tq, nk, qpos0, kpos0,
                                                                                   qk_scale, bias, extent, window, sink);
    check("attn_probs_bias_f16");
}
// the Hopper kernel (wgmma) on a 9.x GPU of a build with sm_90a; IK2_WGMMA=0: the mma.sync one everywhere
bool fa_wgmma() {
    static const int on = [] {
        if (getenv("IK2_WGMMA") && !std::atoi(getenv("IK2_WGMMA"))) return 0;
        int dev = 0, major = 0;
        cudaGetDevice(&dev);
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major == 9 ? 1 : 0;
    }();
    return on != 0;
}
template <int NW, bool WG, class KVT>
void attn_flash_launch(const uint16_t* q16, int q_ld, const KVT* kc, const KVT* vc, int kv_cap, int H, int KV,
                       int Tq, int qpos0, float qk_scale, const float* bias, int extent, int window, int n_split,
                       float* o_part, float2* ml_part, cudaStream_t s) {
    static bool attr[16] = {};   // (per device: the attribute is a device's, and the layer split runs on several)
    int dev = 0;
    cudaGetDevice(&dev);
    if (dev < 16 && !attr[dev]) {
        cudaFuncSetAttribute(attn_flash_f16_kernel<NW, WG, KVT>, cudaFuncAttributeMaxDynamicSharedMemorySize, kFaSmem);
        check("attn_flash (shared memory)");
        attr[dev] = true;
    }
    const dim3 grid((unsigned) ((Tq + 16 * NW - 1) / (16 * NW)), (unsigned) H, (unsigned) n_split);
    attn_flash_f16_kernel<NW, WG, KVT><<<grid, NW * 32, kFaSmem, s>>>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0,
                                                                     qk_scale, bias, extent, window, n_split, o_part,
                                                                     ml_part);
    check("attn_flash");
}
void attn_flash_f16(const uint16_t* q16, int q_ld, const uint16_t* kc, const uint16_t* vc, int kv_cap, int H, int KV,
                    int D, int Tq, int qpos0, float qk_scale, const float* bias, int extent, int window,
                    const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats, int n_split,
                    cudaStream_t s) {
    if (Tq <= 0) return;
    if (D != kFaD) {
        std::fprintf(stderr, "ik2::k::attn_flash_f16: D = %d, not 128\n", D);
        std::abort();
    }
    const size_t per_split = (size_t) Tq * H * (kFaD + 2);
    n_split = std::max(1, std::min<int>(n_split, (int) (scratch_floats / per_split)));
    float* o_part = scratch;
    float2* ml_part = (float2*) (scratch + (size_t) n_split * Tq * H * kFaD);
    if (fa_wgmma())
        attn_flash_launch<kFaNW, true, uint16_t>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent,
                                                 window, n_split, o_part, ml_part, s);
    else
        attn_flash_launch<kFaNW, false, uint16_t>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent,
                                                  window, n_split, o_part, ml_part, s);
    attn_flash_combine_kernel<<<dim3((unsigned) Tq, (unsigned) H), kFaD, 0, s>>>(o_part, ml_part, n_split, Tq, H, sink,
                                                                                 out, o_ld, 1.0f);
    check("attn_flash_combine");
}
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
// gfx1100 WMMA prefill attention (FP16 KV): the fused kernel above, one wave a head's 16 queries,
// the block's waves sharing each K/V tile. Same split/combine contract as attn_flash_f16.
void attn_flash_wmma_f16(const uint16_t* q16, int q_ld, const uint16_t* kc, const uint16_t* vc, int kv_cap, int H,
                         int KV, int D, int Tq, int qpos0, float qk_scale, const float* bias, int extent, int window,
                         const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats, int n_split,
                         cudaStream_t s) {
    if (Tq <= 0) return;
    if (D != kFwD) {
        std::fprintf(stderr, "ik2::k::attn_flash_wmma_f16: D = %d, not 128\n", D);
        std::abort();
    }
    static bool attr[16] = {};
    int dev = 0;
    cudaGetDevice(&dev);
    // IK2_FLASH_BN: keys per tile (32 default). IK2_FLASH_NW: waves a block (8 default = 128 queries share a
    // K/V tile, more work and latency-hiding per wave; 4/2 selectable). smem = (BN*kFwLd + kFwD*BN + NW*16*BN)*2.
    static const int bn = getenv("IK2_FLASH_BN") ? std::atoi(getenv("IK2_FLASH_BN")) : 32;
    static const int nw = getenv("IK2_FLASH_NW") ? std::atoi(getenv("IK2_FLASH_NW")) : 8;
    const int BN = (bn == 64) ? 64 : 32;
    const int NW = (nw == 8) ? 8 : (nw == 2 ? 2 : 4);
    const int BM = 16 * NW;
    const int smem = (BN * kFwLd + kFwD * BN + NW * 16 * BN) * (int) sizeof(uint16_t);
    if (dev < 16 && !attr[dev]) {
#define FW_ATTR(B, W) cudaFuncSetAttribute(attn_flash_wmma_kernel<B, W, uint16_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                           (B * kFwLd + kFwD * B + W * 16 * B) * (int) sizeof(uint16_t)); \
                      cudaFuncSetAttribute(attn_flash_wmma_kernel<B, W, uint8_t>, cudaFuncAttributeMaxDynamicSharedMemorySize, \
                                           (B * kFwLd + kFwD * B + W * 16 * B) * (int) sizeof(uint16_t))
        FW_ATTR(32, 2); FW_ATTR(32, 4); FW_ATTR(32, 8); FW_ATTR(64, 2); FW_ATTR(64, 4); FW_ATTR(64, 8);
        check("attn_flash_wmma (shared memory)");
        attr[dev] = true;
#undef FW_ATTR
    }
    const size_t per_split = (size_t) Tq * H * (kFwD + 2);
    n_split = std::max(1, std::min<int>(n_split, (int) (scratch_floats / per_split)));
    float* o_part = scratch;
    float2* ml_part = (float2*) (scratch + (size_t) n_split * Tq * H * kFwD);
    const dim3 grid((unsigned) ((Tq + BM - 1) / BM), (unsigned) H, (unsigned) n_split);
#define FW_LAUNCH(B, W) attn_flash_wmma_kernel<B, W, uint16_t><<<grid, W * 32, smem, s>>>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, \
                          qpos0, qk_scale, bias, extent, window, n_split, o_part, ml_part)
    if (BN == 32) { if (NW == 2) FW_LAUNCH(32, 2); else if (NW == 8) FW_LAUNCH(32, 8); else FW_LAUNCH(32, 4); }
    else { if (NW == 2) FW_LAUNCH(64, 2); else if (NW == 8) FW_LAUNCH(64, 8); else FW_LAUNCH(64, 4); }
#undef FW_LAUNCH
    check("attn_flash_wmma");
    attn_flash_combine_kernel<<<dim3((unsigned) Tq, (unsigned) H), kFwD, 0, s>>>(o_part, ml_part, n_split, Tq, H, sink,
                                                                                 out, o_ld, 1.0f);
    check("attn_flash_wmma_combine");
}
// attn_flash_wmma over an FP8 (E4M3) cache: tiles dequantized to FP16 as staged (e4m3x8_h8); qk_scale with K's scale
// folded in, output times v_inv_scale. Same WMMA body as the f16 variant, half the KV bytes.
void attn_flash_wmma_fp8(const uint16_t* q16, int q_ld, const uint8_t* kc, const uint8_t* vc, int kv_cap, int H,
                         int KV, int D, int Tq, int qpos0, float qk_scale, float v_inv_scale, const float* bias, int extent,
                         int window, const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats,
                         int n_split, cudaStream_t s) {
    if (Tq <= 0) return;
    if (D != kFwD) {
        std::fprintf(stderr, "ik2::k::attn_flash_wmma_fp8: D = %d, not 128\n", D);
        std::abort();
    }
    static const int bn = getenv("IK2_FLASH_BN") ? std::atoi(getenv("IK2_FLASH_BN")) : 32;
    static const int nw = getenv("IK2_FLASH_NW") ? std::atoi(getenv("IK2_FLASH_NW")) : 8;
    const int BN = (bn == 64) ? 64 : 32;
    const int NW = (nw == 8) ? 8 : (nw == 2 ? 2 : 4);
    const int BM = 16 * NW;
    const int smem = (BN * kFwLd + kFwD * BN + NW * 16 * BN) * (int) sizeof(uint16_t);
    const size_t per_split = (size_t) Tq * H * (kFwD + 2);
    n_split = std::max(1, std::min<int>(n_split, (int) (scratch_floats / per_split)));
    float* o_part = scratch;
    float2* ml_part = (float2*) (scratch + (size_t) n_split * Tq * H * kFwD);
    const dim3 grid((unsigned) ((Tq + BM - 1) / BM), (unsigned) H, (unsigned) n_split);
#define FW_LAUNCH8(B, W) attn_flash_wmma_kernel<B, W, uint8_t><<<grid, W * 32, smem, s>>>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, \
                           qpos0, qk_scale, bias, extent, window, n_split, o_part, ml_part)
    if (BN == 32) { if (NW == 2) FW_LAUNCH8(32, 2); else if (NW == 8) FW_LAUNCH8(32, 8); else FW_LAUNCH8(32, 4); }
    else { if (NW == 2) FW_LAUNCH8(64, 2); else if (NW == 8) FW_LAUNCH8(64, 8); else FW_LAUNCH8(64, 4); }
#undef FW_LAUNCH8
    check("attn_flash_wmma_fp8");
    attn_flash_combine_kernel<<<dim3((unsigned) Tq, (unsigned) H), kFwD, 0, s>>>(o_part, ml_part, n_split, Tq, H, sink,
                                                                                 out, o_ld, v_inv_scale);
    check("attn_flash_wmma_fp8_combine");
}
// attn_flash_wmma over an INT8 / INT4 cache (bits = 8 or 4): tiles dequantized to FP16 as staged with the
// per-(token,head) FP16 scale (kv_intx8_h8); qk_scale is the plain 1/sqrt(D) (the row scale carries magnitude),
// output v_inv_scale = 1. Same WMMA body as f16/fp8, so int KV keeps the fast prefill flash path.
void attn_flash_wmma_int(int bits, const uint16_t* q16, int q_ld, const uint8_t* kci, const uint8_t* vci,
                         const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int KV, int D, int Tq, int qpos0,
                         float qk_scale, const float* bias, int extent, int window, const float* sink, float* out,
                         int o_ld, float* scratch, size_t scratch_floats, int n_split, cudaStream_t s) {
    if (Tq <= 0) return;
    if (D != kFwD) { std::fprintf(stderr, "ik2::k::attn_flash_wmma_int: D = %d, not 128\n", D); std::abort(); }
    if (bits != 8 && bits != 4) { std::fprintf(stderr, "attn_flash_wmma_int: bits %d\n", bits); std::abort(); }
    static const int bn = getenv("IK2_FLASH_BN") ? std::atoi(getenv("IK2_FLASH_BN")) : 32;
    static const int nw = getenv("IK2_FLASH_NW") ? std::atoi(getenv("IK2_FLASH_NW")) : 8;
    const int BN = (bn == 64) ? 64 : 32;
    const int NW = (nw == 8) ? 8 : (nw == 2 ? 2 : 4);
    const int BM = 16 * NW;
    const int smem = (BN * kFwLd + kFwD * BN + NW * 16 * BN) * (int) sizeof(uint16_t);
    const size_t per_split = (size_t) Tq * H * (kFwD + 2);
    n_split = std::max(1, std::min<int>(n_split, (int) (scratch_floats / per_split)));
    float* o_part = scratch;
    float2* ml_part = (float2*) (scratch + (size_t) n_split * Tq * H * kFwD);
    const dim3 grid((unsigned) ((Tq + BM - 1) / BM), (unsigned) H, (unsigned) n_split);
#define FW_LAUNCHI(B, W, BB) attn_flash_wmma_kernel<B, W, uint8_t, BB><<<grid, W * 32, smem, s>>>(q16, q_ld, kci, vci, \
                           kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent, window, n_split, o_part, ml_part, ks, vs)
#define FW_PICKI(BB) if (BN == 32) { if (NW == 2) FW_LAUNCHI(32, 2, BB); else if (NW == 8) FW_LAUNCHI(32, 8, BB); \
                     else FW_LAUNCHI(32, 4, BB); } else { if (NW == 2) FW_LAUNCHI(64, 2, BB); \
                     else if (NW == 8) FW_LAUNCHI(64, 8, BB); else FW_LAUNCHI(64, 4, BB); }
    if (bits == 8) FW_PICKI(8) else FW_PICKI(4)
#undef FW_PICKI
#undef FW_LAUNCHI
    check("attn_flash_wmma_int");
    attn_flash_combine_kernel<<<dim3((unsigned) Tq, (unsigned) H), kFwD, 0, s>>>(o_part, ml_part, n_split, Tq, H, sink,
                                                                                 out, o_ld, 1.0f);
    check("attn_flash_wmma_int_combine");
}
#endif   // STRATA_USE_HIP && STRATA_WMMA_GFX11
void attn_flash_fp8(const uint16_t* q16, int q_ld, const uint8_t* kc, const uint8_t* vc, int kv_cap, int H, int KV,
                    int D, int Tq, int qpos0, float qk_scale, float v_inv_scale, const float* bias, int extent,
                    int window, const float* sink, float* out, int o_ld, float* scratch, size_t scratch_floats,
                    int n_split, cudaStream_t s) {
    if (Tq <= 0) return;
    if (D != kFaD) {
        std::fprintf(stderr, "ik2::k::attn_flash_fp8: D = %d, not 128\n", D);
        std::abort();
    }
    const size_t per_split = (size_t) Tq * H * (kFaD + 2);
    n_split = std::max(1, std::min<int>(n_split, (int) (scratch_floats / per_split)));
    float* o_part = scratch;
    float2* ml_part = (float2*) (scratch + (size_t) n_split * Tq * H * kFaD);
    attn_flash_launch<kFaNW, false, uint8_t>(q16, q_ld, kc, vc, kv_cap, H, KV, Tq, qpos0, qk_scale, bias, extent, window,
                                             n_split, o_part, ml_part, s);
    attn_flash_combine_kernel<<<dim3((unsigned) Tq, (unsigned) H), kFaD, 0, s>>>(o_part, ml_part, n_split, Tq, H, sink,
                                                                                 out, o_ld, v_inv_scale);
    check("attn_flash_combine");
}
int attn_flash_splits(int Tq, int H, int nk, int n_sm) {
    const int blocks = (Tq + 16 * kFaNW - 1) / (16 * kFaNW) * H;
    const int want = (6 * n_sm + blocks - 1) / blocks;   // (about three waves of two blocks an SM)
    const int most = std::max(1, (nk + 16 * kFaBN - 1) / (16 * kFaBN));   // (>= 1024 keys a split)
    return std::max(1, std::min({want, most, 16}));
}
void attn_probs_f16(const float* lg, uint16_t* out16, int H, int Tq, int nk, int qpos0, int kpos0, float qk_scale,
                    const float* r, int d_rel, const float* proj, int extent, int window, const float* sink,
                    cudaStream_t s) {
    if (H <= 0 || Tq <= 0 || nk <= 0) return;
    attn_probs_f16_kernel<<<dim3((unsigned) Tq, (unsigned) H), kBlk, 0, s>>>(lg, out16, H, Tq, nk, qpos0, kpos0,
                                                                              qk_scale, r, d_rel, proj, extent, window,
                                                                              sink);
    check("attn_probs_f16");
}

void gate_mul(float* o, const float* g, size_t n, cudaStream_t s) {
    if (!n) return;
    const bool vec4 = (((uintptr_t) o & 15) == 0) && (((uintptr_t) g & 15) == 0);
    gate_mul_impl_kernel<false, true><<<vec4 ? grid((n + 3) >> 2) : grid(n), kBlk, 0, s>>>(o, g, n, nullptr);
    check("gate_mul");
}
void gate_mul_f16(float* o, const float* g, size_t n, uint16_t* o16, bool write_f32, cudaStream_t s) {
    if (!n) return;
    const bool vec4 = (((uintptr_t) o & 15) == 0) && (((uintptr_t) g & 15) == 0) &&
                      (!o16 || ((uintptr_t) o16 & 7) == 0);
    const unsigned nb = vec4 ? grid((n + 3) >> 2) : grid(n);
    if (o16 && write_f32) gate_mul_impl_kernel<true, true><<<nb, kBlk, 0, s>>>(o, g, n, o16);
    else if (o16) gate_mul_impl_kernel<true, false><<<nb, kBlk, 0, s>>>(o, g, n, o16);
    else if (write_f32) gate_mul_impl_kernel<false, true><<<nb, kBlk, 0, s>>>(o, g, n, nullptr);
    check("gate_mul_f16");
}

void route(const float* logits, const float* bias, int T, int E, int K, float scale, int* top, float* w,
           cudaStream_t s) {
    if (T <= 0) return;
    if (K < 1 || K > kRouteMax) {
        std::fprintf(stderr, "ik2::k::route: K %d\n", K);
        std::abort();
    }
    if (E <= 512) {
        route_warp_kernel<<<T, 32, 0, s>>>(logits, bias, T, E, K, scale, top, w);
        check("route_warp");
        return;
    }
    route_kernel<<<T, 128, (size_t) E * 4, s>>>(logits, bias, T, E, K, scale, top, w);
    check("route");
}

void gather_f16(const float* x, const int* idx, int n, int C, uint16_t* y, cudaStream_t s) {
    if (n <= 0) return;
    gather_f16_kernel<<<grid((size_t) n * C), kBlk, 0, s>>>(x, idx, n, C, y);
    check("gather_f16");
}

void swiglu_cap(const uint16_t* gu, int n, int F, const float* sg, const float* su, float gcap, float ucap,
                uint16_t* h, cudaStream_t s) {
    if (n <= 0) return;
    swiglu_cap_kernel<<<grid((size_t) n * F), kBlk, 0, s>>>(gu, n, F, sg, su, gcap, ucap, h);
    check("swiglu_cap");
}

void moe_combine(const uint16_t* y, const int* inv, const float* wt, int T, int K, int C, float* out, bool add,
                 cudaStream_t s) {
    if (T <= 0) return;
    moe_combine_kernel<<<grid((size_t) T * C), kBlk, 0, s>>>(y, inv, wt, T, K, C, out, add);
    check("moe_combine");
}

void swiglu_cap_f32(const float* gu, int n, int F, const float* sg, const float* su, float gcap, float ucap,
                    uint16_t* h, cudaStream_t s) {
    if (n <= 0) return;
    swiglu_cap_f32_kernel<<<grid((size_t) n * F), kBlk, 0, s>>>(gu, n, F, sg, su, gcap, ucap, h);
    check("swiglu_cap_f32");
}
void moe_combine_f32(const float* y, const int* inv, const float* wt, int T, int K, int C, float* out, bool add,
                     cudaStream_t s) {
    if (T <= 0) return;
    moe_combine_f32_kernel<<<grid((size_t) T * C), kBlk, 0, s>>>(y, inv, wt, T, K, C, out, add);
    check("moe_combine_f32");
}
void moe_combine_conv(const float* y, const int* inv, const float* wt, int K, int C, float* hist, const float* w,
                      const float* res, float* out, cudaStream_t s) {
    moe_combine_conv_kernel<<<(unsigned) ((C + 127) / 128), 128, 0, s>>>(y, inv, wt, K, C, hist, w, res, out);
    check("moe_combine_conv");
}
void add(const float* a, const float* b, size_t n, float* y, cudaStream_t s) {
    if (!n) return;
    add_kernel<<<grid(n), kBlk, 0, s>>>(a, b, n, y);
    check("add");
}
void to_f16(const float* x, size_t n, uint16_t* y, cudaStream_t s) {
    if (!n) return;
    to_f16_kernel<<<grid(n), kBlk, 0, s>>>(x, n, y);
    check("to_f16");
}

void to_f16_limbs(const float* x, size_t n, uint16_t* y, cudaStream_t s) {
    if (!n) return;
    to_f16_limbs_kernel<<<grid(n), kBlk, 0, s>>>(x, n, y);
    check("to_f16_limbs");
}
void to_f16_limbs_tokens(const float* x, int T, int C, uint16_t* y, cudaStream_t s) {
    const size_t total = (size_t) T * C;
    if (!total) return;
    to_f16_limbs_tokens_kernel<<<grid(total), kBlk, 0, s>>>(x, T, C, y);
    check("to_f16_limbs_tokens");
}
void softcap(float* x, size_t n, float cap, cudaStream_t s) {
    if (!n) return;
    softcap_kernel<<<grid(n), kBlk, 0, s>>>(x, n, cap);
    check("softcap");
}

void loop_gates(const LoopGate* gt, int n, int T, int C, int R, float eps, float* sc, cudaStream_t s) {
    if (T <= 0 || n <= 0) return;
    if (n > 3 || R > kLoopRankMax || T > 65535) {
        std::fprintf(stderr, "ik2::k::loop_gates: %d gates, rank %d, %d tokens\n", n, R, T);
        std::abort();
    }
    LoopGateArgs a{};
    for (int j = 0; j < n; ++j)
        a.x[j] = gt[j].x, a.m[j] = gt[j].m, a.D[j] = gt[j].D, a.F[j] = gt[j].F, a.G[j] = gt[j].G, a.y[j] = gt[j].y;
    loop_gate_dots_kernel<<<dim3((unsigned) R, (unsigned) n, (unsigned) T), 128, 0, s>>>(a, T, C, R, sc);
    check("loop_gates (dots)");
    loop_gate_apply_kernel<<<dim3((unsigned) ((C + kBlk - 1) / kBlk), (unsigned) n, (unsigned) T), kBlk, 0, s>>>(
        a, T, C, R, eps, sc);
    check("loop_gates (apply)");
}
void loop_gate(const float* x, const float* m, const float* D, const float* F, const float* G, int T, int C, int R,
               float eps, float* sc, float* y, cudaStream_t s) {
    const LoopGate g{x, m, D, F, G, y};
    loop_gates(&g, 1, T, C, R, eps, sc, s);
}
constexpr int kCandPer = 8192;   // values a block of the histogram
size_t cand_scratch(int n) { return (size_t) ((n + kCandPer - 1) / kCandPer) * kCandBins; }
void cand_topk(const float* x, int n, float lo, float hi, int K, int cap, int* scratch, int* out, cudaStream_t s) {
    if (n <= 0) return;
    const float inv_w = (float) kCandBins / (hi - lo);
    const int n_part = (n + kCandPer - 1) / kCandPer;
    cand_hist_kernel<<<(unsigned) n_part, kBlk, 0, s>>>(x, n, lo, inv_w, kCandPer, scratch);
    check("cand_topk (histogram)");
    cand_threshold_kernel<<<1, kCandBins, 0, s>>>(scratch, n_part, K, out);
    check("cand_topk (threshold)");
    cand_gather_kernel<<<grid((size_t) n), kBlk, 0, s>>>(x, n, lo, inv_w, cap, out, out + 2, (float*) (out + 2 + cap));
    check("cand_topk (gather)");
}
void loop_gate_mix(const float* x, const float* m, const float* a1, const float* a2, int T, int C, int R, float eps,
                   float* g, cudaStream_t s) {
    if (T <= 0) return;
    if (R > kBlk) {
        std::fprintf(stderr, "ik2::k::loop_gate_mix: rank %d\n", R);
        std::abort();
    }
    loop_gate_mix_kernel<<<(unsigned) T, kBlk, 0, s>>>(x, m, a1, a2, C, R, eps, g);
    check("loop_gate_mix");
}
void loop_gate_blend(const float* x, const float* m, const float* f, const float* gn, size_t n, float* y,
                     cudaStream_t s) {
    if (!n) return;
    loop_gate_blend_kernel<<<grid(n), kBlk, 0, s>>>(x, m, f, gn, n, y);
    check("loop_gate_blend");
}

void conv_hist_after(const float* old, const float* x, int T, int C, float* hist, cudaStream_t s) {
    conv_hist_after_kernel<<<grid(C), kBlk, 0, s>>>(old, x, T, C, hist);
    check("conv_hist_after");
}
void to_bf16(const float* x, size_t n, uint16_t* y, cudaStream_t s) {
    if (!n) return;
    to_bf16_kernel<<<grid(n), kBlk, 0, s>>>(x, n, y);
    check("to_bf16");
}
void pack_e4m3(const float* x, size_t n, uint8_t* y, cudaStream_t s) {
    if (!n) return;
    pack_e4m3_kernel<<<grid(n), kBlk, 0, s>>>(x, n, y);
    check("pack_e4m3");
}
// pack_e4m3 into a ring cache: row r lands in slot ((pos0 + r) & ring_mask). One block per row, so a block
// never straddles the wrap point. The plain pack_e4m3 writes its rows contiguously, which is what made
// IK2_DRAFT_CTX_CAP + IK2_DRAFT_KV_FP8 write past the 2048-row local ring once a prompt passed 2048
// positions (the launch then failed with "unspecified launch failure" at the first decode token).
__global__ void pack_e4m3_ring_kernel(const float* x, int cols, uint8_t* y, int ring_mask, int pos0) {
    const int r = blockIdx.y, wp = (pos0 + r) & ring_mask;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < cols; i += blockDim.x * gridDim.x)
        y[(size_t) wp * cols + i] = f32_to_e4m3fn(x[(size_t) r * cols + i]);
}
void pack_e4m3_ring(const float* x, int rows, int cols, uint8_t* y, int ring_mask, int pos0, cudaStream_t s) {
    if (rows <= 0 || cols <= 0) return;
    pack_e4m3_ring_kernel<<<dim3((cols + kBlk - 1) / kBlk, rows), kBlk, 0, s>>>(x, cols, y, ring_mask, pos0);
    check("pack_e4m3_ring");
}
void mix_rms(const float* x, int n_tap, size_t tap_stride, int rows, int C, const float* alpha, const float* w,
             float eps, float* out, cudaStream_t s) {
    if (rows <= 0) return;
    mix_rms_kernel<<<rows, kBlk, 0, s>>>(x, n_tap, tap_stride, C, alpha, w, eps, out);
    check("mix_rms");
}
void rope_neox(float* x, int rows, int heads, const int* pos, float theta, cudaStream_t s) {
    if (rows <= 0) return;
    rope_neox_kernel<<<dim3(rows, heads), 64, 0, s>>>(x, heads, pos, theta);
    check("rope_neox");
}
void draft_attn(const float* q, const float* kc, const float* vc, int k_lo, int k_hi, const float* kb, const float* vb,
                int nq, int H, int Hkv, float scale, float* scratch, float* out, cudaStream_t s, int ring_mask) {
    draft_attn_kernel<<<dim3(H, nq), 128, 0, s>>>(q, kc, vc, k_lo, k_hi, kb, vb, nq, H, Hkv, scale, scratch, out,
                                                  ring_mask);
    check("draft_attn");
}
// FP8 E4M3 context caches (IK2_DRAFT_KV_FP8): same grid (H, nq) and math as draft_attn, context K/V
// read as e4m3 bytes (4x fewer bytes); the block's kb/vb stay F32. Not bit-exact (cache quantization).
void draft_attn_fp8(const float* q, const uint8_t* kc, const uint8_t* vc, int k_lo, int k_hi, const float* kb,
                    const float* vb, int nq, int H, int Hkv, float scale, float* scratch, float* out,
                    int ring_mask, cudaStream_t s) {
    draft_attn_fp8_kernel<<<dim3(H, nq), 128, 0, s>>>(q, kc, vc, k_lo, k_hi, kb, vb, nq, H, Hkv, ring_mask, scale,
                                                      scratch, out);
    check("draft_attn_fp8");
}
// GQA-shared draft attention: grid (Hkv, nq), one block reads each K/V row of its KV head once for
// the R = H / Hkv query heads. Instantiated for R = 2 and R = 3 (the drafter is 24 / 8 = 3); any other
// ratio falls back to the original per-head kernel so behaviour is unchanged.
void draft_attn_gqa(const float* q, const float* kc, const float* vc, int k_lo, int k_hi, const float* kb,
                    const float* vb, int nq, int H, int Hkv, float scale, float* scratch, float* out,
                    cudaStream_t s) {
    const int R = Hkv > 0 ? H / Hkv : 0;
    if (Hkv <= 0 || H % Hkv != 0 || (R != 2 && R != 3)) {
        draft_attn(q, kc, vc, k_lo, k_hi, kb, vb, nq, H, Hkv, scale, scratch, out, s);
        return;
    }
    if (R == 3)
        draft_attn_gqa_kernel<3><<<dim3(Hkv, nq), 128, 0, s>>>(q, kc, vc, k_lo, k_hi, kb, vb, nq, H, Hkv, scale,
                                                               scratch, out);
    else
        draft_attn_gqa_kernel<2><<<dim3(Hkv, nq), 128, 0, s>>>(q, kc, vc, k_lo, k_hi, kb, vb, nq, H, Hkv, scale,
                                                               scratch, out);
    check("draft_attn_gqa");
}
void silu_mul_bf16(const float* gu, int T, int F, uint16_t* out, cudaStream_t s) {
    silu_mul_bf16_kernel<<<grid((size_t) T * F), kBlk, 0, s>>>(gu, T, F, out);
    check("silu_mul_bf16");
}
void rnn_inputs(const float* state, const uint16_t* emb, const int* prev, const float* hidden, const float* f, int R,
                int C, uint16_t* out, cudaStream_t s) {
    rnn_inputs_kernel<<<grid((size_t) 2 * R + 2 * C), kBlk, 0, s>>>(state, emb, prev, hidden, f, R, C, out);
    check("rnn_inputs");
}
void rnn_gates(const float* proj, const float* bias, int R, float* state, float* latent, cudaStream_t s) {
    rnn_gates_kernel<<<grid(R), kBlk, 0, s>>>(proj, bias, R, state, latent);
    check("rnn_gates");
}
void argmax_sum(const float* a, const float* b, const float* c, int n, int* out, cudaStream_t s) {
    if (n <= 0) return;
    const int B = std::min(256, std::max(1, (n + 1023) / 1024));
    if (B == 1) {
        argmax_sum_part_kernel<true><<<1, 256, 0, s>>>(a, b, c, n, out);
        check("argmax_sum");
        return;
    }
    argmax_sum_part_kernel<false><<<B, 256, 0, s>>>(a, b, c, n, out);
    check("argmax_sum_part");
    argmax_final_kernel<<<1, 256, 0, s>>>(B, 1, out);
    check("argmax_sum_final");
}
void argmax_sum(const float* a, const float* b, const float* c, int n, int* out, cudaStream_t s, float* part_v,
                int* part_i) {
    if (n <= 0) return;
    const int B = std::min(256, std::max(1, (n + 1023) / 1024));
    if (B == 1) {
        argmax_sum_part_scratch_kernel<true><<<1, 256, 0, s>>>(a, b, c, n, out, part_v, part_i);
        check("argmax_sum");
        return;
    }
    argmax_sum_part_scratch_kernel<false><<<B, 256, 0, s>>>(a, b, c, n, out, part_v, part_i);
    check("argmax_sum_part");
    argmax_sum_final_scratch_kernel<<<1, 256, 0, s>>>(B, out, part_v, part_i);
    check("argmax_sum_final");
}
void argmax_rows(const float* x, int ld, int n_rows, int n_cols, int* out, cudaStream_t s) {
    if (n_rows <= 0 || n_cols <= 0) return;
    const int max_b = std::max(1, kArgmaxMaxPartials / n_rows);
    const int B = std::min(max_b, std::min(256, std::max(1, (n_cols + 1023) / 1024)));
    if (B == 1) {
        argmax_rows_part_kernel<true><<<dim3(1, (unsigned) n_rows), 256, 0, s>>>(x, ld, n_cols, out);
        check("argmax_rows");
        return;
    }
    argmax_rows_part_kernel<false><<<dim3((unsigned) B, (unsigned) n_rows), 256, 0, s>>>(x, ld, n_cols, out);
    check("argmax_rows_part");
    argmax_final_kernel<<<(unsigned) n_rows, 256, 0, s>>>(B, B, out);
    check("argmax_rows_final");
}
void argmax_gemv_f16(const uint16_t* W, int ld, const float* x, const float* base, const uint16_t* bias_f16,
                     int R, int n, int* out, cudaStream_t s) {
    if (n <= 0) return;
    const int n4 = std::max(1, n >> 2);
    const int B = std::min(512, std::max(1, (n4 + 127) / 128));
    const bool use_smem = R <= 4096;
    const size_t smem = use_smem ? (size_t) R * sizeof(float) : 0;
    if (B == 1) {
        argmax_gemv_f16_part_kernel<true><<<1, 128, smem, s>>>(W, ld, x, base, bias_f16, R, n, use_smem, out);
        check("argmax_gemv_f16");
        return;
    }
    argmax_gemv_f16_part_kernel<false><<<B, 128, smem, s>>>(W, ld, x, base, bias_f16, R, n, use_smem, out);
    check("argmax_gemv_f16_part");
    argmax_final_kernel<<<1, 256, 0, s>>>(B, 1, out);
    check("argmax_gemv_f16_final");
}

// IK2_DRAFT_CAND (opt-in, default OFF): candidate-restricted Markov argmax. The score is EXACTLY
// argmax_gemv_f16's (sum over r of fmaf(W[r*ld+v], x[r]), then +base[v], then +FP16 bias[v]; ties to
// the lower id) -- only the set of v is restricted. No softcap here: the existing argmax has none
// (final_logit_softcapping is a monotonic transform applied to the verify logits, order-preserving).
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
// Stage 1: grid (n_blocks, 2); each block scans 1024 logits, thread t four of them, keeping its
// best id for set 0 = base[v], set 1 = base[v] + fold_b[v]. Eight successive block argmaxes emit
// this block's top 8 for the active set. The caller passes one step's base row (stride ld).
__global__ void __launch_bounds__(256) k_draft_base_topk_part(const float* __restrict__ base, int ld,
                                                              const uint16_t* __restrict__ bias_f16, int n,
                                                              float* __restrict__ part_val, int* __restrict__ part_id) {
    const int set = (int) blockIdx.y;
    const int i0 = (int) blockIdx.x * 1024 + (int) threadIdx.x * 4;
    float b0 = -INFINITY, b1 = -INFINITY;
    int i0best = 0x7fffffff, i1best = 0x7fffffff;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int v = i0 + j;
        if (v < n) {
            const float s = __ldg(base + v);
            argmax_better(s, v, b0, i0best);
            argmax_better(s + h2f(__ldg(bias_f16 + v)), v, b1, i1best);
        }
    }
    float* my_best = set ? &b1 : &b0;
    int* my_idx = set ? &i1best : &i0best;
    float* pv = part_val + (size_t) set * (size_t) gridDim.x * kDraftCandPerBlk +
                (size_t) blockIdx.x * kDraftCandPerBlk;
    int* pi = part_id + (size_t) set * (size_t) gridDim.x * kDraftCandPerBlk +
              (size_t) blockIdx.x * kDraftCandPerBlk;
    __shared__ int win;
#pragma unroll
    for (int k = 0; k < kDraftCandPerBlk; ++k) {
        float rb = *my_best;
        int ri = *my_idx;
        argmax_block_reduce(rb, ri);   // final (rb, ri) is held by warp 0
        if (threadIdx.x == 0) {
            pv[k] = rb;
            pi[k] = (ri == 0x7fffffff) ? -1 : ri;
            win = ri;                  // broadcast the winner so its owner can retire it
        }
        __syncthreads();
        if (*my_idx == win) { *my_best = -INFINITY; *my_idx = 0x7fffffff; }
        __syncthreads();               // argmax_block_reduce reuses its bv/bi shared arrays next round
    }
}

// Stage 2: grid (1, 2), one block per score set. Each thread owns up to 32 partials in registers,
// then K block argmaxes extract the global top K (ties: lower id). Slots past the partials write -1.
__global__ void __launch_bounds__(256) k_draft_base_topk_reduce(const float* __restrict__ part_val,
                                                                const int* __restrict__ part_id, int n_parts, int K,
                                                                float* __restrict__ cand_val, int* __restrict__ cand_id) {
    const int set = (int) blockIdx.y;
    const float* sv = part_val + (size_t) set * n_parts;
    const int* si = part_id + (size_t) set * n_parts;
    float vals[32];
    int ids[32];
    int n_local = 0;
    for (int j = (int) threadIdx.x; j < n_parts && n_local < 32; j += blockDim.x) {
        vals[n_local] = __ldg(sv + j);
        ids[n_local] = __ldg(si + j);
        ++n_local;
    }
    __shared__ int win;
    for (int k = 0; k < K; ++k) {
        float lb = -INFINITY;
        int li = 0x7fffffff;
        for (int q = 0; q < n_local; ++q)
            if (ids[q] >= 0) argmax_better(vals[q], ids[q], lb, li);
        float rb = lb;
        int ri = li;
        argmax_block_reduce(rb, ri);
        if (threadIdx.x == 0) {
            cand_id[(size_t) set * K + k] = (ri == 0x7fffffff) ? -1 : ri;
            if (cand_val) cand_val[(size_t) set * K + k] = rb;
            win = ri;
        }
        __syncthreads();
        for (int q = 0; q < n_local; ++q)
            if (ids[q] == win) { vals[q] = -INFINITY; ids[q] = 0x7fffffff; }
        __syncthreads();
    }
}

// Final step: one block, one warp per group of four candidates, 32 lanes per candidate (8 fold rows
// each). Exactly argmax_gemv_f16's score expression and tie rule; *out written by thread 0.
__global__ void __launch_bounds__(256) k_draft_cand_argmax(const uint16_t* __restrict__ W, int ld,
                                                           const float* __restrict__ x, const float* __restrict__ base,
                                                           const uint16_t* __restrict__ bias_f16, int R, int n,
                                                           const int* __restrict__ cand_id, int ncand, int* __restrict__ out) {
    const int lane = (int) threadIdx.x & 31;
    const int warp = (int) threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    float best = -INFINITY;
    int bidx = 0x7fffffff;
    for (int ci = warp * 4; ci < ncand; ci += nwarps * 4) {
        const int v = __ldg(cand_id + ci);
        float acc = 0.0f;
        if (v >= 0 && v < n) {
            for (int r = lane; r < R; r += 32)
                acc = fmaf(h2f(__ldg(W + (size_t) r * ld + v)), __ldg(x + r), acc);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor_sync(0xffffffffu, acc, o);
            if (lane == 0) {
                const float s = acc + __ldg(base + v) + h2f(__ldg(bias_f16 + v));
                argmax_better(s, v, best, bidx);
            }
        }
    }
    argmax_block_reduce(best, bidx);
    if (threadIdx.x == 0) *out = bidx;
}

void draft_base_topk_part(const float* base, int ld, const uint16_t* bias_f16, int n, int n_blocks,
                          float* part_val, int* part_id, cudaStream_t s) {
    if (n <= 0 || n_blocks <= 0) return;
    k_draft_base_topk_part<<<dim3((unsigned) n_blocks, 2), 256, 0, s>>>(base, ld, bias_f16, n, part_val, part_id);
    check("draft_base_topk_part");
}
void draft_base_topk_reduce(const float* part_val, const int* part_id, int n_parts, int K, int* cand_id,
                            float* cand_val, cudaStream_t s) {
    if (n_parts <= 0 || K <= 0) return;
    k_draft_base_topk_reduce<<<dim3(1, 2), 256, 0, s>>>(part_val, part_id, n_parts, K, cand_val, cand_id);
    check("draft_base_topk_reduce");
}
void draft_cand_argmax(const uint16_t* W, int ld, const float* x, const float* base, const uint16_t* bias_f16,
                       int R, int n, const int* cand_id, int ncand, int* out, cudaStream_t s) {
    if (ncand <= 0) return;
    k_draft_cand_argmax<<<1, 256, 0, s>>>(W, ld, x, base, bias_f16, R, n, cand_id, ncand, out);
    check("draft_cand_argmax");
}
#endif   // STRATA_USE_HIP && STRATA_WMMA_GFX11

int block_values(int type) {
    return type == kNVFP4 ? 64 : type == kQ8_0 ? 32
         : type == kQ5_K || type == kQ6_K || type == kIQ3_XXS || type == kIQ4_XS ? 256 : 0;
}
size_t block_bytes(int type) {
    return type == kNVFP4 ? 36 : type == kQ8_0 ? 34 : type == kQ5_K ? 176 : type == kQ6_K ? 210
         : type == kIQ3_XXS ? 98 : type == kIQ4_XS ? 136 : 0;
}

template <int WARPS, int ROWS>
void launch_experts_gate_up(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                            const float* x, int C, int F, float* gu, cudaStream_t s) {
    const dim3 g((unsigned) ((2 * F + WARPS * ROWS - 1) / (WARPS * ROWS)), (unsigned) S);
    experts_gate_up_kernel<WARPS, ROWS><<<g, WARPS * 32, xp_floats(C) * 4, s>>>(
        blob, tok, ty_gate, ty_up, eb0, x, C, F, mv_row_bytes(ty_gate, C), mv_row_bytes(ty_up, C), gu);
}
template <int WARPS, int ROWS>
void launch_experts_down(const uint8_t* const* blob, int S, int ty_down, size_t off_down, const uint16_t* h, int C,
                         int F, float* y, cudaStream_t s) {
    const dim3 g((unsigned) ((C + WARPS * ROWS - 1) / (WARPS * ROWS)), (unsigned) S);
    experts_down_kernel<WARPS, ROWS><<<g, WARPS * 32, xp_floats(F) * 4, s>>>(
        blob, ty_down, off_down, h, C, F, mv_row_bytes(ty_down, F), y);
}

bool experts_mma_fits(int ty_gate, int ty_up, int ty_down, int C, int F) {
    return ty_gate == kNVFP4 && ty_up == kNVFP4 && ty_down == kNVFP4 && C % 64 == 0 && F % 64 == 0;
}
static int experts_mma_split(bool gate) {
    static const int wg = getenv("IK2_EXPERT_MMA_SPLIT") ? atoi(getenv("IK2_EXPERT_MMA_SPLIT")) : 8;
    static const int wd = getenv("IK2_EXPERT_MMA_SPLIT_DOWN") ? atoi(getenv("IK2_EXPERT_MMA_SPLIT_DOWN")) : 4;
    return gate ? wg : wd;
}
#if defined(__HIPCC__)
// gfx1100 WMMA GEMV for the NVFP4 experts: a wave is 16 output rows (m) over the whole K, one token (n=0).
// A lane lane_lo holds row lane_lo of A (16 dequantized weights along K) and the input x[k0..k0+15] of B;
// lanes 16..31 duplicate. The accumulator's lane lane_lo holds the 8 even/odd rows 2*i+lane_hi of column 0.
// The NVFP4 sub-block scale is applied to the dequantized FP16 weight (exact: code<=12, ue4m3 in FP16 range).
using ik2_v8fp32 = float __attribute__((ext_vector_type(8)));
using ik2_v16f16 = _Float16 __attribute__((ext_vector_type(16)));
__device__ __forceinline__ ik2_v8fp32 ik2_wmma_f16(ik2_v16f16 a, ik2_v16f16 b, ik2_v8fp32 c) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || defined(__gfx1150__) || defined(__gfx1151__)
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
#else
    (void) a; (void) b; (void) c;
    __builtin_trap();
#endif
}
// A lane's 16 dequantized FP16 weights of one NVFP4 sub-block (row r, sub-block sb), scaled by ue4m3.
__device__ __forceinline__ ik2_v16f16 ik2_nvfp4_frag(const uint8_t* row, int sb, int K) {
    const uint8_t* b = row + (size_t) (sb >> 2) * 36;
    const int sub = sb & 3;
    const uint32_t scale_w = __ldg((const uint32_t*) b);
    const uint32_t sc = ue4m3_h2((uint8_t) (scale_w >> (8 * sub)));   // FP16 bits of the scale in both halves
    const uint32_t q0 = __ldg((const uint32_t*) (b + 4 + sub * 8));
    const uint32_t q1 = __ldg((const uint32_t*) (b + 8 + sub * 8));
    // low nibbles -> values 0..7, high nibbles -> values 8..15; each pair times the sub-block scale (round-nearest)
    const uint32_t w[8] = {nv_pair_h2(q0, sc), nv_pair_h2(q0 >> 16, sc), nv_pair_h2(q1, sc), nv_pair_h2(q1 >> 16, sc),
                           nv_pair_h2(q0 >> 4, sc), nv_pair_h2(q0 >> 20, sc), nv_pair_h2(q1 >> 4, sc), nv_pair_h2(q1 >> 20, sc)};
    ik2_v16f16 a;
    __builtin_memcpy(&a, w, sizeof(a));
    return a;
}
template <bool Gate, int SPLIT>
__global__ void __launch_bounds__(SPLIT * 32)
experts_mma_wmma_kernel(const uint8_t* const* __restrict__ blob, size_t off, int C, int F, const uint16_t* __restrict__ x16,
                        int x_ld, uint16_t* hidden, float* y, const float* sg, const float* su, float gcap, float ucap,
                        const int* __restrict__ tok) {
    const int slot = blockIdx.y;
    const uint8_t* bslot = blob[slot];
    if (!bslot) return;
    __shared__ float2 part[SPLIT][16];
    const int lane = threadIdx.x & 31, split = threadIdx.x >> 5;
    const int lane_lo = lane & 15, lane_hi = lane >> 4;
    const int K = Gate ? C : F;
    const int rows = Gate ? 16 : 16;   // gate: 16 gate rows + the same 16 of up; down: 16 rows
    const int r0 = blockIdx.x * rows;
    const size_t rb = (size_t) (K / 64) * 36;
    const uint8_t* a = bslot + (Gate ? 0 : off) + (size_t) (r0 + lane_lo) * rb;
    const uint8_t* b = Gate ? a + off : a;   // gate: up rows at off; down: same rows
    const int tok_idx = (Gate && tok) ? tok[slot] : 0;
    const uint16_t* xp = Gate ? x16 + (size_t) tok_idx * 2 * C : x16 + (size_t) slot * x_ld;
    const int nsub = K / 16;
    ik2_v8fp32 vg = {0, 0, 0, 0, 0, 0, 0, 0}, vu = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int sb = split; sb < nsub; sb += SPLIT) {
        ik2_v16f16 bf;
        if (Gate) {
            // two FP16 limbs: x = hi + lo, [2][C]; the product with the FP32 input is hi-dot + lo-dot
            const uint16_t* hi = xp + sb * 16;
            const uint16_t* lo = xp + C + sb * 16;
            ik2_v16f16 bh, bl;
            __builtin_memcpy(&bh, hi, sizeof(bh));
            __builtin_memcpy(&bl, lo, sizeof(bl));
            ik2_v16f16 ag = ik2_nvfp4_frag(a, sb, K), au = ik2_nvfp4_frag(b, sb, K);
            vg = ik2_wmma_f16(ag, bh, vg);
            vu = ik2_wmma_f16(au, bh, vu);
            vg = ik2_wmma_f16(ag, bl, vg);
            vu = ik2_wmma_f16(au, bl, vu);
        } else {
            __builtin_memcpy(&bf, xp + sb * 16, sizeof(bf));
            vg = ik2_wmma_f16(ik2_nvfp4_frag(a, sb, K), bf, vg);
        }
    }
    // lane lane_lo holds rows 2*i+lane_hi of column 0; reduce the SPLIT warps' partials per row
    if (lane_hi == 0) {
        for (int i = 0; i < 8; ++i) part[split][2 * i] = make_float2(vg[i], vu[i]);
    } else {
        for (int i = 0; i < 8; ++i) part[split][2 * i + 1] = make_float2(vg[i], vu[i]);
    }
    __syncthreads();
    if (split) return;
    const int row = r0 + lane;   // lane 0..15 -> row r0+lane (both halves wrote distinct rows)
    if (lane >= 16) return;
    float2 tot = part[0][lane];
    for (int i = 1; i < SPLIT; ++i) tot.x += part[i][lane].x, tot.y += part[i][lane].y;
    if (Gate) {
        const float gg = tot.x * (sg ? sg[slot] : 1.0f), u = tot.y * (su ? su[slot] : 1.0f);
        hidden[(size_t) slot * F + row] = f2h(gcap * tanhf(gg / gcap) * (1.0f / (1.0f + expf(-gg))) * ucap * tanhf(u / ucap));
    } else {
        y[(size_t) slot * C + row] = tot.x;
    }
}
template <bool Gate>
static void launch_experts_mma_wmma(const uint8_t* const* blob, int S, size_t off, const uint16_t* x16, int x_ld, int C,
                                    int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h, float* y,
                                    cudaStream_t s, const int* tok) {
    const int w = experts_mma_split(Gate);
    const dim3 g((unsigned) ((Gate ? F : C) / 16), (unsigned) S);
#define IK2_WMMA_LAUNCH(W) experts_mma_wmma_kernel<Gate, W><<<g, W * 32, 0, s>>>(blob, off, C, F, x16, x_ld, h, y, sg, su, gcap, ucap, tok)
    if (w == 1) IK2_WMMA_LAUNCH(1);
    else if (w == 2) IK2_WMMA_LAUNCH(2);
    else if (w == 3) IK2_WMMA_LAUNCH(3);
    else if (w == 6) IK2_WMMA_LAUNCH(6);
    else if (w == 8) IK2_WMMA_LAUNCH(8);
    else IK2_WMMA_LAUNCH(4);
#undef IK2_WMMA_LAUNCH
}
#endif   // __HIPCC__

template <bool Gate>
static void launch_experts_mma(const uint8_t* const* blob, int S, size_t off, const uint16_t* x16, int x_ld, int C,
                               int F, const float* sg, const float* su, float gcap, float ucap, uint16_t* h, float* y,
                               cudaStream_t s, const int* tok = nullptr) {
#if defined(__HIPCC__)
    launch_experts_mma_wmma<Gate>(blob, S, off, x16, x_ld, C, F, sg, su, gcap, ucap, h, y, s, tok);
    return;
#endif
    const int w = experts_mma_split(Gate);
    const dim3 g((unsigned) (Gate ? F / 8 : C / 16), (unsigned) S);
#define IK2_MMA_LAUNCH(W) experts_mma_kernel<Gate, W><<<g, W * 32, 0, s>>>(blob, off, C, F, x16, x_ld, h, y, sg, su, gcap, ucap, tok)
    if (w == 1) IK2_MMA_LAUNCH(1);
    else if (w == 2) IK2_MMA_LAUNCH(2);
    else if (w == 3) IK2_MMA_LAUNCH(3);
    else if (w == 6) IK2_MMA_LAUNCH(6);
    else if (w == 8) IK2_MMA_LAUNCH(8);
    else IK2_MMA_LAUNCH(4);
#undef IK2_MMA_LAUNCH
}
void experts_gate_up_swiglu_mma(const uint8_t* const* blob, int S, size_t eb0, const uint16_t* x16, int C, int F,
                                const float* sg, const float* su, float gcap, float ucap, uint16_t* h,
                                cudaStream_t s, const int* tok) {
    if (S <= 0) return;
    launch_experts_mma<true>(blob, S, eb0, x16, 0, C, F, sg, su, gcap, ucap, h, nullptr, s, tok);
    check("experts_gate_up_swiglu_mma");
}
void experts_down_mma(const uint8_t* const* blob, int S, size_t off_down, const uint16_t* h, int C, int F, float* y,
                      cudaStream_t s) {
    if (S <= 0) return;
    launch_experts_mma<false>(blob, S, off_down, h, F, C, F, nullptr, nullptr, 0.0f, 0.0f, nullptr, y, s);
    check("experts_down_mma");
}
#if defined(__HIPCC__) && defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
// Multi-token NVFP4-direct WMMA GEMM for ONE expert's gathered prefill tokens (n of them).
// Reads the NVFP4 fragments straight from the blocks (no decode_f16 whole-weight expand).
// A lane lane_lo holds output-row r0+lane_lo (16 weights along K); B lane lane_lo holds token n0+lane_lo;
// the accumulator's lane lane_lo holds token-column lane_lo, rows 2*i+lane_hi of output. Lanes 16..31 duplicate.
// gate_up writes gu[n][2F] F32 (gate rows then up rows, raw sums; swiglu_cap_f32 applies the scales).
// down writes y[n][C] F32. C and F multiples of 64; n any positive. One wave a (row-tile, token-tile).
__global__ void __launch_bounds__(32)
experts_prefill_gate_up_wmma_kernel(const uint8_t* __restrict__ sl, size_t eb_up, int C, int F, int n,
                                    const uint16_t* __restrict__ x16, float* __restrict__ gu) {
    const int lane = threadIdx.x & 31, lane_lo = lane & 15, lane_hi = lane >> 4;
    const int r0 = blockIdx.x * 16;          // gate row tile (0..F-16)
    const int n0 = blockIdx.y * 16;          // token tile
    const size_t rb = (size_t) (C / 64) * 36;
    const uint8_t* ag = sl + (size_t) (r0 + lane_lo) * rb;          // gate rows at offset 0
    const uint8_t* au = sl + eb_up + (size_t) (r0 + lane_lo) * rb;  // up rows at eb_up
    const int tok = n0 + lane_lo;
    const bool tok_ok = tok < n;
    const uint16_t* xp = x16 + (size_t) (tok_ok ? tok : 0) * C;
    const int nsub = C / 16;
    ik2_v8fp32 vg = {0, 0, 0, 0, 0, 0, 0, 0}, vu = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int sb = 0; sb < nsub; ++sb) {
        ik2_v16f16 bf;
        __builtin_memcpy(&bf, xp + sb * 16, sizeof(bf));   // token lane_lo's 16 inputs (zero row if out of range)
        if (!tok_ok) { for (int q = 0; q < 16; ++q) bf[q] = (_Float16) 0.0f; }
        vg = ik2_wmma_f16(ik2_nvfp4_frag(ag, sb, C), bf, vg);
        vu = ik2_wmma_f16(ik2_nvfp4_frag(au, sb, C), bf, vu);
    }
    // lane lane_lo = token column; rows 2*i+lane_hi = gate/up output row r0+(2*i+lane_hi)
    const int t = n0 + lane_lo;
    if (t < n) {
        for (int i = 0; i < 8; ++i) {
            const int row = r0 + 2 * i + lane_hi;
            gu[(size_t) t * 2 * F + row] = vg[i];
            gu[(size_t) t * 2 * F + F + row] = vu[i];
        }
    }
}
__global__ void __launch_bounds__(32)
experts_prefill_down_wmma_kernel(const uint8_t* __restrict__ sl, size_t off_down, int C, int F, int n,
                                 const uint16_t* __restrict__ h16, float* __restrict__ y) {
    const int lane = threadIdx.x & 31, lane_lo = lane & 15, lane_hi = lane >> 4;
    const int r0 = blockIdx.x * 16;          // down output-row tile (0..C-16)
    const int n0 = blockIdx.y * 16;
    const size_t rb = (size_t) (F / 64) * 36;
    const uint8_t* a = sl + off_down + (size_t) (r0 + lane_lo) * rb;
    const int tok = n0 + lane_lo;
    const bool tok_ok = tok < n;
    const uint16_t* hp = h16 + (size_t) (tok_ok ? tok : 0) * F;
    const int nsub = F / 16;
    ik2_v8fp32 v = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int sb = 0; sb < nsub; ++sb) {
        ik2_v16f16 bf;
        __builtin_memcpy(&bf, hp + sb * 16, sizeof(bf));
        if (!tok_ok) { for (int q = 0; q < 16; ++q) bf[q] = (_Float16) 0.0f; }
        v = ik2_wmma_f16(ik2_nvfp4_frag(a, sb, F), bf, v);
    }
    const int t = n0 + lane_lo;
    if (t < n) {
        for (int i = 0; i < 8; ++i) {
            const int row = r0 + 2 * i + lane_hi;
            y[(size_t) t * C + row] = v[i];
        }
    }
}
void experts_prefill_gate_up_wmma(const uint8_t* sl, size_t eb_up, int C, int F, int n, const uint16_t* x16,
                                  float* gu, cudaStream_t s) {
    if (n <= 0) return;
    const dim3 g((unsigned) (F / 16), (unsigned) ((n + 15) / 16));
    experts_prefill_gate_up_wmma_kernel<<<g, 32, 0, s>>>(sl, eb_up, C, F, n, x16, gu);
    check("experts_prefill_gate_up_wmma");
}
void experts_prefill_down_wmma(const uint8_t* sl, size_t off_down, int C, int F, int n, const uint16_t* h16,
                               float* y, cudaStream_t s) {
    if (n <= 0) return;
    const dim3 g((unsigned) (C / 16), (unsigned) ((n + 15) / 16));
    experts_prefill_down_wmma_kernel<<<g, 32, 0, s>>>(sl, off_down, C, F, n, h16, y);
    check("experts_prefill_down_wmma");
}
#endif   // __HIPCC__ && STRATA_USE_HIP && STRATA_WMMA_GFX11
void experts_gate_up(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                     const float* x, int C, int F, float* gu, cudaStream_t s) {
    if (S <= 0) return;
    if (S <= 2) launch_experts_gate_up<4, 2>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, gu, s);
    else launch_experts_gate_up<4, 4>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, gu, s);
    check("experts_gate_up");
}
void experts_down(const uint8_t* const* blob, int S, int ty_down, size_t off_down, const uint16_t* h, int C, int F,
                  float* y, cudaStream_t s) {
    if (S <= 0) return;
    if (S <= 4) launch_experts_down<4, 2>(blob, S, ty_down, off_down, h, C, F, y, s);
    else launch_experts_down<4, 4>(blob, S, ty_down, off_down, h, C, F, y, s);
    check("experts_down");
}
void experts_gate_up_g(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G, int ty_gate,
                       int ty_up, size_t eb0, const float* x, int C, int F, float* gu, cudaStream_t s) {
    if (G <= 0) return;
    const dim3 g((unsigned) ((2 * F + kGroupWarps * kGroupRows - 1) / (kGroupWarps * kGroupRows)), (unsigned) G);
    experts_gate_up_g_kernel<<<g, kGroupWarps * 32, 0, s>>>(blob, n, tok, row, ty_gate, ty_up, eb0, x, C, F,
                                                            mv_row_bytes(ty_gate, C), mv_row_bytes(ty_up, C), gu);
    check("experts_gate_up_g");
}
void experts_down_g(const uint8_t* const* blob, const int* n, const int* row, int G, int ty_down, size_t off_down,
                    const uint16_t* h, int C, int F, float* y, cudaStream_t s) {
    if (G <= 0) return;
    const dim3 g((unsigned) ((C + kGroupWarps * kGroupRows - 1) / (kGroupWarps * kGroupRows)), (unsigned) G);
    experts_down_g_kernel<<<g, kGroupWarps * 32, 0, s>>>(blob, n, row, ty_down, off_down, h, C, F,
                                                         mv_row_bytes(ty_down, F), y);
    check("experts_down_g");
}

template <int WARPS, int ROWS>
void launch_experts_gate_up_swiglu(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up,
                                   size_t eb0, const float* x, int C, int F, const float* sg, const float* su,
                                   float gcap, float ucap, uint16_t* h, cudaStream_t s, const int* orow = nullptr,
                                   int h_ld = 0) {
    const dim3 g((unsigned) ((F + WARPS * ROWS - 1) / (WARPS * ROWS)), (unsigned) S);
    experts_gate_up_swiglu_kernel<WARPS, ROWS><<<g, WARPS * 32, xp_floats(C) * 4, s>>>(
        blob, tok, ty_gate, ty_up, eb0, x, C, F, mv_row_bytes(ty_gate, C), mv_row_bytes(ty_up, C), sg, su, gcap,
        ucap, h, orow, h_ld > 0 ? h_ld : F);
}

void experts_gate_up_swiglu_rows(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up,
                                 size_t eb0, const float* x, int C, int F, const float* sg, const float* su, float gcap,
                                 float ucap, uint16_t* h, const int* orow, int h_ld, cudaStream_t s) {
    if (S <= 0) return;
    if (S <= 8) launch_experts_gate_up_swiglu<4, 2>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld);
    else launch_experts_gate_up_swiglu<4, 4>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, h, s, orow, h_ld);
    check("experts_gate_up_swiglu_rows");
}
void experts_gate_up_swiglu(const uint8_t* const* blob, const int* tok, int S, int ty_gate, int ty_up, size_t eb0,
                            const float* x, int C, int F, const float* sg, const float* su, float gcap, float ucap,
                            uint16_t* h, cudaStream_t s) {
    if (S <= 0) return;
    if (S <= 8) launch_experts_gate_up_swiglu<4, 2>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, h, s);
    else launch_experts_gate_up_swiglu<4, 4>(blob, tok, S, ty_gate, ty_up, eb0, x, C, F, sg, su, gcap, ucap, h, s);
    check("experts_gate_up_swiglu");
}

void experts_gate_up_swiglu_g(const uint8_t* const* blob, const int* n, const int* tok, const int* row, int G,
                              int ty_gate, int ty_up, size_t eb0, const float* x, int C, int F, const float* sg,
                              const float* su, float gcap, float ucap, uint16_t* h, cudaStream_t s) {
    if (G <= 0) return;
    const dim3 g((unsigned) ((F + kGroupWarps * kGroupSwigluRows - 1) / (kGroupWarps * kGroupSwigluRows)), (unsigned) G);
    experts_gate_up_swiglu_g_kernel<<<g, kGroupWarps * 32, 0, s>>>(blob, n, tok, row, ty_gate, ty_up, eb0, x, C, F,
                                                                   mv_row_bytes(ty_gate, C), mv_row_bytes(ty_up, C),
                                                                   sg, su, gcap, ucap, h);
    check("experts_gate_up_swiglu_g");
}

void prep_qkv_r_f16(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in, uint16_t* vc,
                    float* r, const float* r_clip, int T, int H, int KV, int D, int d_rel, float qk_clip,
                    float rel_clip, float eps, const int* pos, float alpha, float n_floor, int pos0, int kv_cap,
                    cudaStream_t s) {
    if (T <= 0) return;
    const int total = T * (2 * H + 2 * KV);
    prep_qkv_r_impl_kernel<uint16_t><<<(unsigned) total, 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D, d_rel, qk_clip, rel_clip, eps, pos, alpha, n_floor,
        pos0, kv_cap, 1.0f, 1.0f, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    check("prep_qkv_r_f16");
}

void prep_qkv_r_fp8(float* q, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in, uint8_t* vc,
                    float* r, const float* r_clip, int T, int H, int KV, int D, int d_rel, float qk_clip,
                    float rel_clip, float eps, const int* pos, float alpha, float n_floor, int pos0, int kv_cap,
                    float k_scale, float v_scale, cudaStream_t s) {
    if (T <= 0) return;
    const int total = T * (2 * H + 2 * KV);
    prep_qkv_r_impl_kernel<uint8_t><<<(unsigned) total, 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D, d_rel, qk_clip, rel_clip, eps, pos, alpha, n_floor,
        pos0, kv_cap, k_scale, v_scale, nullptr, nullptr, nullptr, nullptr, 0, nullptr);
    check("prep_qkv_r_fp8");
}

template <class KV>
void attn_multi_impl(const float* q, const float* r, const float* sink, const KV* kc, const KV* vc, int kv_cap,
                     int nq, int H, int Hkv, int k0, int nk, int qpos0, float qk_scale, float v_inv_scale,
                     const float* proj, int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    if (d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_multi: d_rel %d\n", d_rel);
        std::abort();
    }
    const int P = std::min(128, std::max(1, (nk + 63) / 64));
    float* part = scratch;
    if (mshared_on() && nq >= 1 && nq <= 8) {   // IK2_VERIFY_MSHARED: one KV walk for all nq rows (see kernel)
        attn_multi_shared_dispatch(q, r, kc, vc, kv_cap, nq, H, Hkv, k0, nk, qpos0, qk_scale, proj, d_rel, extent,
                                   window, part, nullptr, P, s);
        check("attn_multi_mshared_split");
    } else {
        attn_split_kernel<KV><<<dim3(H, P, nq), 128, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos0, qk_scale,
                                                              proj, d_rel, extent, window, part, nullptr);
        check("attn_multi_split");
    }
    attn_combine_kernel<<<dim3(H, nq), 128, 0, s>>>(sink, part, H, P, v_inv_scale, out);
    check("attn_multi_combine");
}

void attn_multi(const float* q, const float* r, const float* sink, const float* kc, const float* vc, int nq, int H,
                int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj, int d_rel, int extent, int window,
                float* scratch, float* out, cudaStream_t s) {
    attn_multi_impl(q, r, sink, kc, vc, 0, nq, H, Hkv, k0, nk, qpos0, qk_scale, 1.0f, proj, d_rel, extent, window,
                    scratch, out, s);
}

void attn_multi_f16(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                    int kv_cap, int nq, int H, int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj,
                    int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    attn_multi_impl(q, r, sink, kc, vc, kv_cap, nq, H, Hkv, k0, nk, qpos0, qk_scale, 1.0f, proj, d_rel, extent, window,
                    scratch, out, s);
}

void attn_multi_fp8(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                    int kv_cap, int nq, int H, int Hkv, int k0, int nk, int qpos0, float qk_scale,
                    float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                    float* out, cudaStream_t s) {
    attn_multi_impl(q, r, sink, kc, vc, kv_cap, nq, H, Hkv, k0, nk, qpos0, qk_scale, v_inv_scale, proj, d_rel, extent,
                    window, scratch, out, s);
}

// ---- IK2_VERIFY_GRAPH: T>1 prepare / attention with a device-side base position ------------------------------
// The kernels themselves already support this (prep_qkv_r_impl_kernel's pos0_d and attn_split_kernel's pos_d);
// these wrappers expose it for a graph that is captured once and replayed at a new pos0 each speculative step.
// The split count P is fixed at capture so the launch grid does not change; W.lg is sized for P = 128, nq <= 8.
void prep_qkv_r_f16_g(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in, uint16_t* vc,
                      float* r, const float* r_clip, int T, int H, int KV, int D, int d_rel, float qk_clip,
                      float rel_clip, float eps, const int* pos, float alpha, float n_floor, const int* pos0_d,
                      int kv_cap, cudaStream_t s) {
    if (T <= 0) return;
    const int total = T * (2 * H + 2 * KV);
    prep_qkv_r_impl_kernel<uint16_t><<<(unsigned) total, 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D, d_rel, qk_clip, rel_clip, eps, pos, alpha, n_floor,
        0, kv_cap, 1.0f, 1.0f, pos0_d, nullptr, nullptr, nullptr, 0, nullptr);
    check("prep_qkv_r_f16_g");
}
void prep_qkv_r_fp8_g(float* q, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in, uint8_t* vc,
                      float* r, const float* r_clip, int T, int H, int KV, int D, int d_rel, float qk_clip,
                      float rel_clip, float eps, const int* pos, float alpha, float n_floor, const int* pos0_d,
                      int kv_cap, float k_scale, float v_scale, cudaStream_t s) {
    if (T <= 0) return;
    const int total = T * (2 * H + 2 * KV);
    prep_qkv_r_impl_kernel<uint8_t><<<(unsigned) total, 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, T, H, KV, D, d_rel, qk_clip, rel_clip, eps, pos, alpha, n_floor,
        0, kv_cap, k_scale, v_scale, pos0_d, nullptr, nullptr, nullptr, 0, nullptr);
    check("prep_qkv_r_fp8_g");
}
template <class KV>
void attn_multi_g_impl(const float* q, const float* r, const float* sink, const KV* kc, const KV* vc, int kv_cap,
                       int nq, int H, int Hkv, const int* pos_d, int P, float qk_scale, float v_inv_scale,
                       const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                       cudaStream_t s) {
    if (d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_multi_g: d_rel %d\n", d_rel);
        std::abort();
    }
    const int splits = P > 0 ? P : 128;
    float* part = scratch;
    if (mshared_on() && nq >= 1 && nq <= 8) {   // IK2_VERIFY_MSHARED: shared-KV path (pos_d handled in kernel)
        attn_multi_shared_dispatch(q, r, kc, vc, kv_cap, nq, H, Hkv, 0, 0, 0, qk_scale, proj, d_rel, extent, window,
                                   part, pos_d, splits, s);
        check("attn_multi_g_mshared_split");
    } else {
        // With pos_d set, attn_split_kernel derives qpos0 = *pos_d, nk = qpos0 + gridDim.z - k0 and k0 from the window.
        attn_split_kernel<KV><<<dim3(H, (unsigned) splits, (unsigned) nq), 128, 0, s>>>(
            q, r, kc, vc, kv_cap, H, Hkv, 0, 0, 0, qk_scale, proj, d_rel, extent, window, part, pos_d);
        check("attn_multi_g_split");
    }
    attn_combine_kernel<<<dim3(H, (unsigned) nq), 128, 0, s>>>(sink, part, H, splits, v_inv_scale, out);
    check("attn_multi_g_combine");
}
void attn_multi_f16_g(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                      int kv_cap, int nq, int H, int Hkv, const int* pos_d, int P, float qk_scale, const float* proj,
                      int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    attn_multi_g_impl(q, r, sink, kc, vc, kv_cap, nq, H, Hkv, pos_d, P, qk_scale, 1.0f, proj, d_rel, extent, window,
                      scratch, out, s);
}
void attn_multi_fp8_g(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                      int kv_cap, int nq, int H, int Hkv, const int* pos_d, int P, float qk_scale,
                      float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                      float* out, cudaStream_t s) {
    attn_multi_g_impl(q, r, sink, kc, vc, kv_cap, nq, H, Hkv, pos_d, P, qk_scale, v_inv_scale, proj, d_rel, extent,
                      window, scratch, out, s);
}

template <int WARPS, int ROWS>
void launch_mv_blocks(int type, const uint8_t* W, int N, int K, const float* x, float* y, cudaStream_t s) {
    mv_blocks_kernel<WARPS, ROWS><<<(unsigned) ((N + WARPS * ROWS - 1) / (WARPS * ROWS)), WARPS * 32,
                                    xp_floats(K) * 4, s>>>(type, W, N, K, mv_row_bytes(type, K), x, y);
}

void mv_blocks(int type, const uint8_t* W, int N, int K, const float* x, float* y, cudaStream_t s) {
    if (N <= 0) return;
    if (K > 8192 || !block_values(type) || K % block_values(type)) {
        std::fprintf(stderr, "ik2::k::mv_blocks: type %d, K %d\n", type, K);
        std::abort();
    }
    if (N <= 512) launch_mv_blocks<2, 1>(type, W, N, K, x, y, s);
    else if (N <= 2048) launch_mv_blocks<4, 1>(type, W, N, K, x, y, s);
    else launch_mv_blocks<4, 2>(type, W, N, K, x, y, s);
    check("mv_blocks");
}

template <class KV>
void attn_decode_impl(const float* q, const float* r, const float* sink, const KV* kc, const KV* vc, int kv_cap, int H,
                      int Hkv, int D, int k0, int nk, int qpos, float qk_scale, float v_inv_scale, const float* proj,
                      int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    if (D != 128 || d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_decode: D %d, d_rel %d\n", D, d_rel);
        std::abort();
    }
    // Flash-Decode splits over chunks of ~64 keys, up to 128 blocks per head (partials written directly to scratch)
    const int P = std::min(128, std::max(1, (nk + 63) / 64));
    if (P == 1) {
        attn_decode_kernel<KV><<<H, 128, 0, s>>>(q, r, sink, kc, vc, kv_cap, H, Hkv, k0, nk, qpos, qk_scale,
                                                   v_inv_scale, proj, d_rel, extent, window, out);
        check("attn_decode");
        return;
    }
    float* part = scratch;
    attn_split_kernel<KV><<<dim3(H, P), 128, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, k0, nk, qpos, qk_scale, proj, d_rel,
                                                      extent, window, part, nullptr);
    check("attn_decode_split");
    attn_combine_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, part, H, P, v_inv_scale, out);
    check("attn_decode_combine");
}

void attn_decode(const float* q, const float* r, const float* sink, const float* kc, const float* vc, int H, int Hkv,
                 int D, int k0, int nk, int qpos, float qk_scale, const float* proj, int d_rel, int extent, int window,
                 float* scratch, float* out, cudaStream_t s) {
    attn_decode_impl(q, r, sink, kc, vc, 0, H, Hkv, D, k0, nk, qpos, qk_scale, 1.0f, proj, d_rel, extent, window,
                     scratch, out, s);
}

void attn_decode_f16(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                     int kv_cap, int H, int Hkv, int D, int k0, int nk, int qpos, float qk_scale, const float* proj,
                     int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    attn_decode_impl(q, r, sink, kc, vc, kv_cap, H, Hkv, D, k0, nk, qpos, qk_scale, 1.0f, proj, d_rel, extent, window,
                     scratch, out, s);
}

void attn_decode_fp8(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                     int kv_cap, int H, int Hkv, int D, int k0, int nk, int qpos, float qk_scale,
                     float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                     float* out, cudaStream_t s) {
    attn_decode_impl(q, r, sink, kc, vc, kv_cap, H, Hkv, D, k0, nk, qpos, qk_scale, v_inv_scale, proj, d_rel, extent,
                     window, scratch, out, s);
}

void mv_q8p_conv(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, const MvConv& c0,
                 const MvConv& c1, cudaStream_t s, const MvAlt& a0, const MvAlt& a1) {
    if (N <= 0) return;
    if (K % 128 || ((uintptr_t) q & 3) || (((uintptr_t) x | (uintptr_t) a0.x | (uintptr_t) a1.x) & 15)) {
        std::fprintf(stderr, "ik2::k::mv_q8p: K %d, or q / x not aligned\n", K);
        std::abort();
    }
    mv_q8p_kernel<4><<<(unsigned) ((N + 3) / 4), 128, 0, s>>>(q, d, N, K, x, y, c0, c1, a0, a1);
    check("mv_q8p");
}
void mv_q8p(const int8_t* q, const uint16_t* d, int N, int K, const float* x, float* y, cudaStream_t s) {
    mv_q8p_conv(q, d, N, K, x, y, MvConv{}, MvConv{}, s);
}
// mv_q8p_conv whose default input is RMSNorm'd in the kernel's prologue (x raw, nw weights, eps); a0/a1's rows keep
// their own (already-normed) inputs. Saves the separate rms launch before a sliding layer's fused q|k|v|r|gate
// product. K<=2304, K%128==0, q 4B / x 16B aligned.
void mv_q8p_rms_alt(const int8_t* q, const uint16_t* d, int N, int K, const float* x, const float* nw, float eps,
                    float* y, const MvConv& c0, const MvConv& c1, cudaStream_t s, const MvAlt& a0, const MvAlt& a1) {
    if (N <= 0) return;
    if (K % 128 || K > 2304 || ((uintptr_t) q & 3) || (((uintptr_t) x | (uintptr_t) a0.x | (uintptr_t) a1.x) & 15)) {
        std::fprintf(stderr, "ik2::k::mv_q8p_rms_alt: K %d, or q / x not aligned\n", K);
        std::abort();
    }
    mv_q8p_rms_alt_kernel<4><<<(unsigned) ((N + 3) / 4), 128, 0, s>>>(q, d, N, K, x, nw, eps, y, c0, c1, a0, a1);
    check("mv_q8p_rms_alt");
}
// mv_q8p whose input is RMSNorm'd in the kernel's prologue (x raw, nw the norm weights): saves the separate rms
// launch before a full layer's wk/wv products. K<=2304, 16B-aligned q/x.
void mv_q8p_rms(const int8_t* q, const uint16_t* d, int N, int K, const float* x, const float* nw, float eps, float* y,
                const MvConv& c0, const MvConv& c1, cudaStream_t s) {
    if (N <= 0) return;
    if (K % 128 || K > 2304 || ((uintptr_t) q & 3) || ((uintptr_t) x & 15)) {
        std::fprintf(stderr, "ik2::k::mv_q8p_rms: K %d, or q / x not aligned\n", K);
        std::abort();
    }
    mv_q8p_rms_kernel<4, true><<<(unsigned) ((N + 3) / 4), 128, 0, s>>>(q, d, N, K, x, nw, eps, y, c0, c1);
    check("mv_q8p_rms");
}
// two inputs normed in the prologue (a full layer's wk/wv fused into one launch, each row's input picked by its conv)
void mv_q8p_rms2(const int8_t* q, const uint16_t* d, int N, int K, const float* x, const float* nw0, const float* nw1,
                 float eps, float* y, const MvConv& c0, const MvConv& c1, cudaStream_t s) {
    if (N <= 0) return;
    if (K % 128 || K > 2304 || ((uintptr_t) q & 3) || ((uintptr_t) x & 15)) {
        std::fprintf(stderr, "ik2::k::mv_q8p_rms2: K %d, or q / x not aligned\n", K);
        std::abort();
    }
    mv_q8p_rms2_kernel<4><<<(unsigned) ((N + 3) / 4), 128, 0, s>>>(q, d, N, K, x, nw0, nw1, eps, y, c0, c1);
    check("mv_q8p_rms2");
}
void decode_q8p_f16(const int8_t* q, const uint16_t* d, size_t n, uint16_t* out, cudaStream_t s) {
    if (!n) return;
    decode_q8p_f16_kernel<<<grid(n), kBlk, 0, s>>>(q, d, n, out);
    check("decode_q8p_f16");
}

void mv_f16(const uint16_t* W, int N, int K, const float* x, float* y, cudaStream_t s) {
    if (N <= 0) return;
    mv_f16_kernel<4><<<(unsigned) ((N + 3) / 4), 128, 0, s>>>(W, N, K, x, y);
    check("mv_f16");
}
// mv_f16 whose input is RMSNorm'd in the prologue (x raw, nw weights, eps): saves the rms launch before the dynamic
// taps. K<=2304.
void mv_f16_rms(const uint16_t* W, int N, int K, const float* x, const float* nw, float eps, float* y, cudaStream_t s) {
    if (N <= 0) return;
    if (K > 2304) {
        std::fprintf(stderr, "ik2::k::mv_f16_rms: K %d\n", K);
        std::abort();
    }
    mv_f16_rms_kernel<4><<<(unsigned) ((N + 3) / 4), 128, 0, s>>>(W, N, K, x, nw, eps, y);
    check("mv_f16_rms");
}

void route_gather(const int* top, const float* w, const uint8_t* const* tab, const float* sg_t, const float* su_t,
                  const float* sd_t, int S, int K, const uint8_t** blob, int* tok, int* inv, float* sg, float* su,
                  float* wt, cudaStream_t s) {
    if (S <= 0) return;
    if (S > 1024) {
        std::fprintf(stderr, "ik2::k::route_gather: %d slots\n", S);
        std::abort();
    }
    route_gather_kernel<<<1, (unsigned) ((S + 31) / 32 * 32), 0, s>>>(top, w, tab, sg_t, su_t, sd_t, S, K, blob, tok, inv,
                                                                       sg, su, wt);
    check("route_gather");
}
void route_token(const float* logits, const float* bias, int E, int K, float scale, const uint8_t* const* tab,
                 const float* sg_t, const float* su_t, const float* sd_t, int* top, float* w, const uint8_t** blob,
                 int* tok, int* inv, float* sg, float* su, float* wt, cudaStream_t s, const float* x, int width,
                 uint16_t* x_limbs, const uint8_t* const* i4_tabWp, const uint8_t* const* i4_tabWh,
                 const float* const* i4_tabWS, const uint8_t** i4_Wp, const uint8_t** i4_Wh, const float** i4_WS) {
    if (E > 1024 || K > E) {   // (more experts than a block's threads: the two kernels)
        if (x_limbs) to_f16_limbs(x, (size_t) width, x_limbs, s);
        route(logits, bias, 1, E, K, scale, top, w, s);
        route_gather(top, w, tab, sg_t, su_t, sd_t, K, K, blob, tok, inv, sg, su, wt, s);
        if (i4_tabWp) int4_gather_slots(top, i4_tabWp, i4_tabWh, i4_tabWS, K, i4_Wp, i4_Wh, i4_WS, s);
        return;
    }
    const unsigned nt = (unsigned) ((E + 31) / 32 * 32);
    const unsigned blocks = 1 + (x_limbs ? (width + nt - 1) / nt : 0);
    route_token_kernel<<<blocks, nt, (size_t) (E + nt / 32 + K + 1) * sizeof(float), s>>>(
        logits, bias, E, K, scale, tab, sg_t, su_t, sd_t, top, w, blob, tok, inv, sg, su, wt, x, width, x_limbs,
        i4_tabWp, i4_tabWh, i4_tabWS, i4_Wp, i4_Wh, i4_WS);
    check("route_token");
}

void ep_select_token(const int* top, const float* w, const uint8_t* const* tab,
                     const float* sg_t, const float* su_t, const float* sd_t,
                     int S, const uint8_t** blob, int* tok, int* inv,
                     float* sg, float* su, float* wt, cudaStream_t s, int K) {
    if (S <= 0) return;
    ep_select_token_kernel<<<1, (unsigned) ((S + 31) / 32 * 32), 0, s>>>(
        top, w, tab, sg_t, su_t, sd_t, S, K, blob, tok, inv, sg, su, wt);
    check("ep_select_token");
}

void ep_combine_token(const float* y0, const float* y1, const uint8_t* const* blob0,
                      const float* wt, int S, int C, float* out, cudaStream_t s) {
    if (S <= 0 || C <= 0) return;
    ep_combine_token_kernel<<<(unsigned) ((C + 127) / 128), 128, 0, s>>>(
        y0, y1, blob0, wt, S, C, out);
    check("ep_combine_token");
}

void ep_combine_block(const float* y0, const float* y1, const uint8_t* const* blob0,
                      const float* wt, int T, int K, int C, float* out, cudaStream_t s) {
    if (T <= 0 || K <= 0 || C <= 0) return;
    const size_t total = (size_t) T * C;
    ep_combine_block_kernel<<<(unsigned) ((total + 255) / 256), 256, 0, s>>>(
        y0, y1, blob0, wt, T, K, C, out);
    check("ep_combine_block");
}

void ep_combine_conv(const float* y0, const float* y1, const uint8_t* const* blob0,
                     const float* wt, int S, int C, float* hist, const float* w,
                     const float* res, float* out, cudaStream_t s) {
    if (S <= 0 || C <= 0) return;
    ep_combine_conv_kernel<<<(unsigned) ((C + 127) / 128), 128, 0, s>>>(
        y0, y1, blob0, wt, S, C, hist, w, res, out);
    check("ep_combine_conv");
}

void prep_qkv_r_f16_at(float* q, const float* q_clip, const float* k_in, uint16_t* kc, const float* v_in, uint16_t* vc,
                       float* r, const float* r_clip, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip,
                       float eps, const int* pos, bool log_scale, float alpha, float n_floor, int kv_cap,
                       uint16_t* q16, cudaStream_t s, const uint16_t* sink_w, const float* sink_x, int sink_k,
                       float* sink_y) {
    prep_qkv_r_impl_kernel<uint16_t><<<(unsigned) (2 * H + 2 * KV + (sink_w ? H : 0)), 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, 1, H, KV, D, d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
        alpha, n_floor, 0, kv_cap, 1.0f, 1.0f, pos, q16, sink_w, sink_x, sink_k, sink_y);
    check("prep_qkv_r_f16_at");
}
void prep_qkv_r_fp8_at(float* q, const float* q_clip, const float* k_in, uint8_t* kc, const float* v_in, uint8_t* vc,
                       float* r, const float* r_clip, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip,
                       float eps, const int* pos, bool log_scale, float alpha, float n_floor, int kv_cap,
                       float k_scale, float v_scale, uint16_t* q16, cudaStream_t s, const uint16_t* sink_w,
                       const float* sink_x, int sink_k, float* sink_y) {
    prep_qkv_r_impl_kernel<uint8_t><<<(unsigned) (2 * H + 2 * KV + (sink_w ? H : 0)), 32, 0, s>>>(
        q, q_clip, k_in, kc, v_in, vc, r, r_clip, 1, H, KV, D, d_rel, qk_clip, rel_clip, eps, log_scale ? pos : nullptr,
        alpha, n_floor, 0, kv_cap, k_scale, v_scale, pos, q16, sink_w, sink_x, sink_k, sink_y);
    check("prep_qkv_r_fp8_at");
}

// One block per head, no split and no separate combine: the sliding window's keys all fit one 4-warp pass.  The
// position is read from *pos_d so a captured graph replay still sees each token's position.  Sink and the output
// gate fold into the same block.  Only sound for a bounded window (a full-context layer has too many keys).
template <class KV>
__global__ void attn_decode_onepass_at_kernel(const float* q, const float* r, const float* sink, const KV* kc,
                                              const KV* vc, int kv_cap, int H, int Hkv, const int* pos_d,
                                              float qk_scale, float v_inv_scale, const float* proj, int d_rel,
                                              int extent, int window, const float* gate, float* out) {
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, g = h / (H / Hkv), tid = threadIdx.x;
    const int qpos = *pos_d;
    const int k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0;
    const int nk = qpos + 1 - k0;
    const int KD = Hkv * 128;
    const float* qh = q + (size_t) h * 128;
    const float* rh = r + (size_t) h * d_rel;
    attn_flash_range(qh, rh, kc, vc, KD, g, kv_cap, k0, 0, nk, qpos, qk_scale, proj, d_rel, extent, window, sm_m, sm_s,
                     sm_o);
    const float sk = sink[h];
    const float M = fmaxf(sk, fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3])));
    float den = expf(sk - M), o = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (sm_m[w] == -INFINITY) continue;
        const float f = expf(sm_m[w] - M);
        den += sm_s[w] * f;
        o += sm_o[w][tid] * f;
    }
    float v = (o / den) * v_inv_scale;
    if (gate) v *= 1.0f / (1.0f + expf(-gate[(size_t) h * 128 + tid]));
    out[(size_t) h * 128 + tid] = v;
}

void attn_decode_f16_onepass(const float* q, const float* r, const float* sink, const uint16_t* kc,
                             const uint16_t* vc, int kv_cap, int H, int Hkv, const int* pos, float qk_scale,
                             const float* proj, int d_rel, int extent, int window, float* out, cudaStream_t s,
                             const float* gate) {
    attn_decode_onepass_at_kernel<uint16_t><<<H, 128, 0, s>>>(q, r, sink, kc, vc, kv_cap, H, Hkv, pos, qk_scale, 1.0f,
                                                              proj, d_rel, extent, window, gate, out);
    check("attn_decode_f16_onepass");
}
void attn_decode_fp8_onepass(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                             int kv_cap, int H, int Hkv, const int* pos, float qk_scale, float v_inv_scale,
                             const float* proj, int d_rel, int extent, int window, float* out, cudaStream_t s,
                             const float* gate) {
    attn_decode_onepass_at_kernel<uint8_t><<<H, 128, 0, s>>>(q, r, sink, kc, vc, kv_cap, H, Hkv, pos, qk_scale,
                                                             v_inv_scale, proj, d_rel, extent, window, gate, out);
    check("attn_decode_fp8_onepass");
}

void attn_decode_f16_at(const float* q, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                        int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale, const float* proj,
                        int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s,
                        const float* gate) {
    attn_split_kernel<uint16_t><<<dim3(H, P), 128, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, 0, 0, 0, qk_scale, proj,
                                                            d_rel, extent, window, scratch, pos);
    check("attn_decode_f16_at (split)");
    attn_combine_gate_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, scratch, H, P, 1.0f, gate, out);
    check("attn_decode_f16_at (combine)");
}
void attn_decode_fp8_at(const float* q, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                        int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale, float v_inv_scale,
                        const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                        cudaStream_t s, const float* gate) {
    attn_split_kernel<uint8_t><<<dim3(H, P), 128, 0, s>>>(q, r, kc, vc, kv_cap, H, Hkv, 0, 0, 0, qk_scale, proj, d_rel,
                                                           extent, window, scratch, pos);
    check("attn_decode_fp8_at (split)");
    attn_combine_gate_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, scratch, H, P, v_inv_scale, gate, out);
    check("attn_decode_fp8_at (combine)");
}

int attn_decode_mma_blocks(int Hkv) {
    int dev = 0, n_sm = 0, occ16 = 0, occ8 = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&n_sm, cudaDevAttrMultiProcessorCount, dev);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ16, attn_decode_mma_kernel<uint16_t>, kWaNW * 32, 0);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ8, attn_decode_mma_kernel<uint8_t>, kWaNW * 32, 0);
    check("attn_decode_mma (occupancy)");
    return std::max(1, n_sm * std::min(occ16, occ8) / std::max(1, Hkv));
}
size_t attn_decode_mma_scratch(int H, int n_split) { return (size_t) n_split * kWaNW * H * (kFaD + 2); }
template <class KV>
void attn_decode_mma_impl(const uint16_t* q16, const float* r, const float* sink, const KV* kc, const KV* vc, int kv_cap,
                          int H, int Hkv, const int* pos, int qpos, int n_split, float qk_scale, float v_inv,
                          const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                          const float* gate, uint16_t* out16, cudaStream_t s) {
    if (H % Hkv || H / Hkv > 8 || n_split < 1 || (kv_cap > 0 && kv_cap < 16)) {
        std::fprintf(stderr, "ik2::k::attn_decode_mma: %d heads on %d, %d splits, a cache of %d\n", H, Hkv, n_split,
                     kv_cap);
        std::abort();
    }
    float* o_part = scratch;
    float2* ml_part = (float2*) (o_part + (size_t) n_split * kWaNW * H * kFaD);
    const bool rel4 = d_rel % 4 == 0 && (((uintptr_t) r | (uintptr_t) proj) & 15) == 0;
    attn_decode_mma_kernel<KV><<<dim3((unsigned) Hkv, (unsigned) n_split), kWaNW * 32, 0, s>>>(
        q16, kc, vc, kv_cap, H, Hkv, pos, qpos, qk_scale, r, proj, d_rel, rel4, extent, window, n_split, o_part,
        ml_part);
    check("attn_decode_mma");
    attn_decode_mma_combine_kernel<<<(unsigned) H, kFaD, (size_t) (kFaD + n_split * kWaNW) * sizeof(float), s>>>(
        o_part, ml_part, n_split, H, sink, v_inv, pos, qpos, window, gate, out, out16);
    check("attn_decode_mma (combine)");
}
void attn_decode_mma_f16(const uint16_t* q16, const float* r, const float* sink, const uint16_t* kc, const uint16_t* vc,
                         int kv_cap, int H, int Hkv, const int* pos, int qpos, int n_split, float qk_scale,
                         const float* proj, int d_rel, int extent, int window, float* scratch, float* out,
                         const float* gate, uint16_t* out16, cudaStream_t s) {
    attn_decode_mma_impl(q16, r, sink, kc, vc, kv_cap, H, Hkv, pos, qpos, n_split, qk_scale, 1.0f, proj, d_rel, extent,
                         window, scratch, out, gate, out16, s);
}
void attn_decode_mma_fp8(const uint16_t* q16, const float* r, const float* sink, const uint8_t* kc, const uint8_t* vc,
                         int kv_cap, int H, int Hkv, const int* pos, int qpos, int n_split, float qk_scale,
                         float v_inv_scale, const float* proj, int d_rel, int extent, int window, float* scratch,
                         float* out, const float* gate, uint16_t* out16, cudaStream_t s) {
    attn_decode_mma_impl(q16, r, sink, kc, vc, kv_cap, H, Hkv, pos, qpos, n_split, qk_scale, v_inv_scale, proj, d_rel,
                         extent, window, scratch, out, gate, out16, s);
}

void unpack_kv_fp8_to_f16(const uint8_t* kc, const uint8_t* vc, int kv_cap, int k0, int nk, int KD,
                          float k_inv_scale, float v_inv_scale, uint16_t* k_out16, uint16_t* v_out16,
                          cudaStream_t s) {
    if (nk <= 0 || KD <= 0) return;
    if ((KD & 3) == 0 && (!k_out16 || ((((uintptr_t) kc & 3) == 0) && (((uintptr_t) k_out16 & 7) == 0))) &&
        (!v_out16 || ((((uintptr_t) vc & 3) == 0) && (((uintptr_t) v_out16 & 7) == 0)))) {
        const int kd4 = KD >> 2;
        const int bx = std::min(256, kd4);
        const dim3 g((unsigned) ((kd4 + bx - 1) / bx), (unsigned) std::min(nk, 65535));
        unpack_kv_fp8_to_f16_vec4_kernel<<<g, (unsigned) bx, 0, s>>>(kc, vc, kv_cap, k0, nk, KD, k_inv_scale,
                                                                     v_inv_scale, k_out16, v_out16);
        check("unpack_kv_fp8_to_f16");
        return;
    }
    const size_t total = (size_t) nk * KD;
    const unsigned blocks = (unsigned) std::min<size_t>((total + kBlk - 1) / kBlk, 65535u);
    unpack_kv_fp8_to_f16_kernel<<<blocks, kBlk, 0, s>>>(kc, vc, kv_cap, k0, nk, KD, k_inv_scale, v_inv_scale, k_out16,
                                                        v_out16);
    check("unpack_kv_fp8_to_f16");
}

// ---- INT8 / INT4 K / V cache (IK2_KV=int8 / int4) ---------------------------------------------------------
// Per-(token, KV-head) symmetric quantization: one FP16 scale per 128-dim head row, codes int8 (1 byte per
// element) or int4 (2 nibbles per byte, low nibble = even element).  K is L2-clipped and V rms-normalized
// exactly as the FP8 path, but WITHOUT the k_scale / v_scale unit scaling: the per-row scale replaces it.
// Dequant is software (code * scale at load), so the KV read needs no tensor-core support.  The f16 / fp8
// kernels above are untouched; these are separate instantiations so the default path stays byte-identical.

__device__ __forceinline__ int kv_int_code(float x, int qmax) {
    const int q = (int) rintf(x);
    return max(-qmax, min(qmax, q));
}

// The FP16-rounded per-row scale is shared by encode and decode: encode divides by the ROUNDED scale so
// stored code * stored scale round-trips.  An all-zero row gets the smallest positive subnormal instead of
// 0 so the reciprocal stays finite (its codes are all 0 either way).
__device__ __forceinline__ float kv_int_scale(float amax, int qmax, uint16_t* s16_out) {
    uint16_t s16 = f2h(amax / (float) qmax);
    float sc = h2f(s16);
    if (sc == 0.0f) {
        s16 = 1;
        sc = h2f((uint16_t) 1);
    }
    *s16_out = s16;
    return sc;
}

// One warp stores one 128-elem row: lane owns elements 4*lane .. 4*lane+3 (a float4) and packs them into
// one uint32 (int8) or uint16 (int4) at index lane; lane 0 writes the row's FP16 scale.
template <int BITS>
__device__ __forceinline__ void kv_int_store4(uint8_t* dst, uint16_t* scale_dst, int lane, float4 y) {
    const int qmax = (BITS == 8) ? 127 : 7;
    float am = fmaxf(fmaxf(fabsf(y.x), fabsf(y.y)), fmaxf(fabsf(y.z), fabsf(y.w)));
    am = wmax(am);
    uint16_t s16;
    const float sc = kv_int_scale(am, qmax, &s16);
    if (lane == 0) *scale_dst = s16;
    const float inv = 1.0f / sc;
    const int q0 = kv_int_code(y.x * inv, qmax), q1 = kv_int_code(y.y * inv, qmax);
    const int q2 = kv_int_code(y.z * inv, qmax), q3 = kv_int_code(y.w * inv, qmax);
    if (BITS == 8) {
        ((uint32_t*) dst)[lane] = (uint32_t) (q0 & 0xFF) | ((uint32_t) (q1 & 0xFF) << 8) |
                                  ((uint32_t) (q2 & 0xFF) << 16) | ((uint32_t) (q3 & 0xFF) << 24);
    } else {
        ((uint16_t*) dst)[lane] =
            (uint16_t) ((q0 & 0xF) | ((q1 & 0xF) << 4) | ((q2 & 0xF) << 8) | ((q3 & 0xF) << 12));
    }
}

template <int BITS>
__device__ __forceinline__ void store_k_int_warp(const float* src, uint8_t* dst, uint16_t* scale_dst, int D,
                                                 float th) {
    const int lane = threadIdx.x & 31;
    const int qmax = (BITS == 8) ? 127 : 7;
    const uintptr_t al = (BITS == 8) ? (uintptr_t) 3 : (uintptr_t) 1;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (((uintptr_t) dst & al) == 0)) {
        const float4 xv = ((const float4*) src)[lane];
        const float nrm = sqrtf(wsum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w));
        const float f = fminf(1.0f, th / (nrm + 1e-6f));
        kv_int_store4<BITS>(dst, scale_dst, lane, make_float4(xv.x * f, xv.y * f, xv.z * f, xv.w * f));
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float nrm = sqrtf(wsum(a));
    const float f = fminf(1.0f, th / (nrm + 1e-6f));
    float am = 0.0f;
    for (int d = lane; d < D; d += 32) am = fmaxf(am, fabsf(src[d] * f));
    am = wmax(am);
    uint16_t s16;
    const float sc = kv_int_scale(am, qmax, &s16);
    if (lane == 0) *scale_dst = s16;
    const float inv = 1.0f / sc;
    if (BITS == 8) {
        for (int d = lane; d < D; d += 32) dst[d] = (uint8_t) (kv_int_code(src[d] * f * inv, qmax) & 0xFF);
    } else {
        for (int b = lane; b < D / 2; b += 32) {
            const int q0 = kv_int_code(src[2 * b] * f * inv, qmax);
            const int q1 = kv_int_code(src[2 * b + 1] * f * inv, qmax);
            dst[b] = (uint8_t) ((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
}

template <int BITS>
__device__ __forceinline__ void store_v_int_warp(const float* src, uint8_t* dst, uint16_t* scale_dst, int D,
                                                 float eps) {
    const int lane = threadIdx.x & 31;
    const int qmax = (BITS == 8) ? 127 : 7;
    const uintptr_t al = (BITS == 8) ? (uintptr_t) 3 : (uintptr_t) 1;
    if (D == 128 && (((uintptr_t) src & 15) == 0) && (((uintptr_t) dst & al) == 0)) {
        const float4 xv = ((const float4*) src)[lane];
        const float rs = rsqrtf(wsum(xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w) / (float) D + eps);
        kv_int_store4<BITS>(dst, scale_dst, lane, make_float4(xv.x * rs, xv.y * rs, xv.z * rs, xv.w * rs));
        return;
    }
    float a = 0.0f;
    for (int d = lane; d < D; d += 32) {
        const float v = src[d];
        a += v * v;
    }
    const float rs = rsqrtf(wsum(a) / (float) D + eps);
    float am = 0.0f;
    for (int d = lane; d < D; d += 32) am = fmaxf(am, fabsf(src[d] * rs));
    am = wmax(am);
    uint16_t s16;
    const float sc = kv_int_scale(am, qmax, &s16);
    if (lane == 0) *scale_dst = s16;
    const float inv = 1.0f / sc;
    if (BITS == 8) {
        for (int d = lane; d < D; d += 32) dst[d] = (uint8_t) (kv_int_code(src[d] * rs * inv, qmax) & 0xFF);
    } else {
        for (int b = lane; b < D / 2; b += 32) {
            const int q0 = kv_int_code(src[2 * b] * rs * inv, qmax);
            const int q1 = kv_int_code(src[2 * b + 1] * rs * inv, qmax);
            dst[b] = (uint8_t) ((q0 & 0xF) | ((q1 & 0xF) << 4));
        }
    }
}

// prep_qkv_r_impl_kernel's int counterpart: q / r / sinks identical, K / V go through the int warps above.
template <int BITS>
__global__ void prep_qkv_r_int_kernel(float* q, const float* q_clip, const float* k_in, uint8_t* kci, uint16_t* ks,
                                      const float* v_in, uint8_t* vci, uint16_t* vs, float* r, const float* r_clip,
                                      int T, int H, int KVh, int D, int d_rel, float qk_clip, float rel_clip,
                                      float eps, const int* pos, float alpha, float n_floor, int pos0, int kv_cap,
                                      const int* pos0_d, uint16_t* q16, const uint16_t* sink_w, const float* sink_x,
                                      int sink_k, float* sink_y) {
    if (pos0_d) pos0 = *pos0_d;
    const int gid = blockIdx.x;
    const int per_tok = 2 * H + 2 * KVh;
    if (gid >= T * per_tok) {
        const int h = gid - T * per_tok, lane = threadIdx.x;
        if (!sink_w || h >= H) return;
        const uint16_t* w = sink_w + (size_t) h * sink_k;
        float a = 0.0f;
        for (int i = lane; i < sink_k; i += 32) a += h2f(w[i]) * sink_x[i];
        a = wsum(a);
        if (lane == 0) sink_y[h] = a;
        return;
    }
    const int t = gid / per_tok, rem = gid % per_tok;
    const float tau = pos ? (1.0f + alpha * logf(fmaxf(1.0f, ((float) pos[t] + 1.0f) / n_floor))) : 1.0f;
    const int cb = (BITS == 8) ? D : D / 2;   // code bytes per head row
    if (rem < H) {
        const int h = rem;
        float* qp = q + ((size_t) t * H + h) * D;
        l2clip_warp(qp, q_clip ? q_clip + (size_t) h * D : nullptr, qp,
                    q16 ? q16 + ((size_t) t * H + h) * D : nullptr, D, qk_clip, tau);
    } else if (rem < H + KVh) {
        const int kv = rem - H;
        const int slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t);
        const float* kp = k_in + ((size_t) t * KVh + kv) * D;
        store_k_int_warp<BITS>(kp, kci + ((size_t) slot * KVh + kv) * cb, ks + (size_t) slot * KVh + kv, D, qk_clip);
    } else if (rem < H + 2 * KVh) {
        const int kv = rem - H - KVh;
        const int slot = kv_cap > 0 ? (pos0 + t) % kv_cap : (pos0 + t);
        const float* vp = v_in + ((size_t) t * KVh + kv) * D;
        store_v_int_warp<BITS>(vp, vci + ((size_t) slot * KVh + kv) * cb, vs + (size_t) slot * KVh + kv, D, eps);
    } else {
        const int h = rem - H - 2 * KVh;
        float* rp = r + ((size_t) t * H + h) * d_rel;
        l2clip_warp(rp, r_clip ? r_clip + (size_t) h * d_rel : nullptr, rp, nullptr, d_rel, rel_clip, tau);
    }
}

// True batched write counterpart: the t-th row writes its K/V into the t-th cache through the device pointer
// tables, at slot pos[t] (mod kv_cap when given).  q / r / the sink tail keep prep_qkv_r_int_kernel's math, so
// a batch of one row with pos[0] == pos0 is bit-identical to the scalar kernel.  K/V caches are mutable through
// the table, hence the const_cast on the table's const pointee type.
template <int BITS>
__global__ void prep_qkv_r_int_rows_kernel(float* q, const float* q_clip, const float* k_in,
                                           const uint8_t* const* kci_tab, const uint16_t* const* ks_tab,
                                           const float* v_in, const uint8_t* const* vci_tab,
                                           const uint16_t* const* vs_tab, float* r, const float* r_clip, int T,
                                           int H, int KVh, int D, int d_rel, float qk_clip, float rel_clip, float eps,
                                           const int* pos, float alpha, float n_floor, int kv_cap, uint16_t* q16,
                                           const uint16_t* sink_w, const float* sink_x, int sink_k, float* sink_y,
                                           const int* tau_pos) {
    const int gid = blockIdx.x;
    const int per_tok = 2 * H + 2 * KVh;
    if (gid >= T * per_tok) {
        const int h = gid - T * per_tok, lane = threadIdx.x;
        if (!sink_w || h >= H) return;
        const uint16_t* w = sink_w + (size_t) h * sink_k;
        float a = 0.0f;
        for (int i = lane; i < sink_k; i += 32) a += h2f(w[i]) * sink_x[i];
        a = wsum(a);
        if (lane == 0) sink_y[h] = a;
        return;
    }
    const int t = gid / per_tok, rem = gid % per_tok;
    const int pt = pos[t];
    const float tau = tau_pos ? (1.0f + alpha * logf(fmaxf(1.0f, ((float) tau_pos[t] + 1.0f) / n_floor))) : 1.0f;
    const int cb = (BITS == 8) ? D : D / 2;
    if (rem < H) {
        const int h = rem;
        float* qp = q + ((size_t) t * H + h) * D;
        l2clip_warp(qp, q_clip ? q_clip + (size_t) h * D : nullptr, qp,
                    q16 ? q16 + ((size_t) t * H + h) * D : nullptr, D, qk_clip, tau);
    } else if (rem < H + KVh) {
        const int kv = rem - H;
        const int slot = kv_cap > 0 ? pt % kv_cap : pt;
        uint8_t* kci = const_cast<uint8_t*>(kci_tab[t]);
        uint16_t* ks = const_cast<uint16_t*>(ks_tab[t]);
        const float* kp = k_in + ((size_t) t * KVh + kv) * D;
        store_k_int_warp<BITS>(kp, kci + ((size_t) slot * KVh + kv) * cb, ks + (size_t) slot * KVh + kv, D, qk_clip);
    } else if (rem < H + 2 * KVh) {
        const int kv = rem - H - KVh;
        const int slot = kv_cap > 0 ? pt % kv_cap : pt;
        uint8_t* vci = const_cast<uint8_t*>(vci_tab[t]);
        uint16_t* vs = const_cast<uint16_t*>(vs_tab[t]);
        const float* vp = v_in + ((size_t) t * KVh + kv) * D;
        store_v_int_warp<BITS>(vp, vci + ((size_t) slot * KVh + kv) * cb, vs + (size_t) slot * KVh + kv, D, eps);
    } else {
        const int h = rem - H - 2 * KVh;
        float* rp = r + ((size_t) t * H + h) * d_rel;
        l2clip_warp(rp, r_clip ? r_clip + (size_t) h * d_rel : nullptr, rp, nullptr, d_rel, rel_clip, tau);
    }
}

// int dequant load: lane's 4 codes * the row's FP16 scale -> float4 (the int counterpart of v_load4).
template <int BITS>
__device__ __forceinline__ float4 v_load4_int(const uint8_t* vr, int lane, float scale) {
    if (BITS == 8) {
        const int o = lane * 4;
        if (((uintptr_t) vr & 3) == 0) {
            const uint32_t v4 = __ldg((const uint32_t*) (vr + o));
            return make_float4((float) (int8_t) (v4 & 0xFFu) * scale, (float) (int8_t) ((v4 >> 8) & 0xFFu) * scale,
                               (float) (int8_t) ((v4 >> 16) & 0xFFu) * scale, (float) (int8_t) (v4 >> 24) * scale);
        }
        return make_float4((float) (int8_t) vr[o] * scale, (float) (int8_t) vr[o + 1] * scale,
                           (float) (int8_t) vr[o + 2] * scale, (float) (int8_t) vr[o + 3] * scale);
    }
    const int b = lane * 2;
    if (((uintptr_t) vr & 1) == 0) {
        const uint32_t v4 = __ldg((const uint16_t*) (vr + b));
        return make_float4((float) kv_int_nib(v4 & 0xFu) * scale, (float) kv_int_nib((v4 >> 4) & 0xFu) * scale,
                           (float) kv_int_nib((v4 >> 8) & 0xFu) * scale, (float) kv_int_nib((v4 >> 12) & 0xFu) * scale);
    }
    const uint32_t b0 = vr[b], b1 = vr[b + 1];
    return make_float4((float) kv_int_nib(b0 & 0xFu) * scale, (float) kv_int_nib((b0 >> 4) & 0xFu) * scale,
                       (float) kv_int_nib(b1 & 0xFu) * scale, (float) kv_int_nib((b1 >> 4) & 0xFu) * scale);
}

// IK2_FA_SPLIT: coalesced 8-byte int4 row load with a preserve-order register unpack.  Lanes 0..7 each
// fetch one uint2 (16 nibble codes = 16 consecutive dims); two warp shuffles hand every lane its own
// 4-code (2-byte) slice.  Order and dequant are exactly v_load4_int<4>'s, so the 32-lane wsum/DPP
// reduction, the 4-warp shared-memory merge and the [H,P,130] partials all stay unchanged.  The int4
// row is 64B and 8B-aligned (slot * Hkv * 64 + g * 64); the byte path is kept as a safety guard.
// No LDS staging for V: all lanes reading a given 8B chunk want the same address (a warp broadcast),
// so an LDS XOR swizzle would only add a barrier per key without removing any bank conflict.
template <int BITS>
__device__ __forceinline__ float4 v_load4_int_fa(const uint8_t* vr, int lane, float scale) {
    if constexpr (BITS != 4) {
        return v_load4_int<BITS>(vr, lane, scale);
    } else if (((uintptr_t) vr & 7u) != 0) {
        return v_load4_int<4>(vr, lane, scale);
    } else {
        const int g = lane >> 2, s = lane & 3;      // owner lane g (0..7) holds 8B chunk g; slice s (0..3)
        const uint2 c = (lane < 8) ? __ldg((const uint2*) (vr + (size_t) lane * 8)) : make_uint2(0u, 0u);
        const uint32_t lo = __shfl_sync(0xffffffffu, c.x, g, 32);
        const uint32_t hi = __shfl_sync(0xffffffffu, c.y, g, 32);
        const uint32_t h = (s == 0) ? (lo & 0xFFFFu) : (s == 1) ? (lo >> 16)
                        : (s == 2) ? (hi & 0xFFFFu) : (hi >> 16);
        return make_float4((float) kv_int_nib(h & 0xFu) * scale,
                           (float) kv_int_nib((h >> 4) & 0xFu) * scale,
                           (float) kv_int_nib((h >> 8) & 0xFu) * scale,
                           (float) kv_int_nib((h >> 12) & 0xFu) * scale);
    }
}

// Compile-time selector used by both the plain and GQA split kernels; FA is folded away at 0.
template <int BITS, int FA>
__device__ __forceinline__ float4 v_load4_int_sel(const uint8_t* vr, int lane, float scale) {
    return FA ? v_load4_int_fa<BITS>(vr, lane, scale) : v_load4_int<BITS>(vr, lane, scale);
}

// IK2_FA_RDNA3 bit3: exp2f(x*log2(e)).  EX2=0 keeps the original expf call and its exact rounding.
template <int EX2>
__device__ __forceinline__ float fa_softmax_exp(float x) {
    if constexpr (EX2)
        return exp2f(x * 1.4426950408889634f);
    else
        return expf(x);
}

// IK2_ATTN_WIDELOAD: one uint2 (8 bytes = 16 int4 codes) per lane instead of the old uint16 (4 codes) per lane.
// Four eight-lane groups share a wave32 warp, so each warp carries four query heads at once: 16 codes/lane and
// a 3-step butterfly (4,2,1) inside each group replace the old 4 codes/lane plus a 5-step, 32-lane wsum.
__device__ __forceinline__ float4 wideload_int4_four(uint32_t c, float scale) {
    return make_float4((float) kv_int_nib(c & 0xFu) * scale, (float) kv_int_nib((c >> 4) & 0xFu) * scale,
                       (float) kv_int_nib((c >> 8) & 0xFu) * scale, (float) kv_int_nib((c >> 12) & 0xFu) * scale);
}

__device__ __forceinline__ float wideload_sum8(float v) {
#pragma unroll
    for (int o = 4; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o, 8);
    return v;
}

// Split-K int4 decode attention with coalesced 16-code loads and zero shared memory / zero barriers.  The layout
// is the donor k3_flash_swa_gqa's: a wave32 warp owns NQ query heads, an eight-lane group owns one head's D=128
// row.  Each thread accumulates 16 of the 128 output channels and writes them straight to the split partials,
// so the existing attn_combine_gate_kernel consumes the same [H,P,130] layout unchanged.  The dequant formula
// stays code (signed nibble) * h2f(per-(token,KV-head) scale); the scale is never folded into q.
template <int BITS, int NQ>
__global__ void attn_wideload_int_kernel(const float* q, const float* r, const uint8_t* kci, const uint8_t* vci,
                                        const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv, int k0,
                                        int nk, int qpos, float qk_scale, const float* proj, int d_rel, int extent,
                                        int window, float* part, const int* pos_d) {
    static_assert(BITS == 4 && NQ == 4, "attn_wideload_int_kernel supports int4, four heads per wave32 warp");
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int qg = lane >> 3, l = lane & 7;
    if (pos_d) {
        qpos = *pos_d;
        k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0;
        nk = qpos + 1 - k0;
    }
    const int h_want = blockIdx.x * (NQ * 4) + warp + qg * NQ;   // strided heads: four distinct KV heads / wave
    const bool keep = h_want < H;
    const int h = keep ? h_want : H - 1;                          // dummy head keeps every warp lane in the shuffles
    const int g = h / (H / Hkv), p = blockIdx.y, P = gridDim.y;
    const int cb = 64;                                            // BITS==4: 64 code bytes per head row
    const size_t rowb = (size_t) Hkv * cb;
    const size_t sidx_base = (size_t) Hkv;
    float* pp = part + (((size_t) h * P + p) * 130);

    const float* qh = q + (size_t) h * 128;
    const float* rh = r + (size_t) h * d_rel;
    float4 q4[4];
    const bool q_aligned = (((uintptr_t) qh & 15) == 0);
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d0 = l * 16 + k * 4;
        q4[k] = q_aligned ? __ldg((const float4*) (qh + d0))
                          : make_float4(qh[d0], qh[d0 + 1], qh[d0 + 2], qh[d0 + 3]);
    }
    float rr[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const int d = l * 4 + k;
        rr[k] = d < d_rel ? __ldg(rh + d) : 0.0f;
    }

    float m_w = -INFINITY, s_w = 0.0f;
    float4 o_w[4] = {make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 0.0f, 0.0f),
                     make_float4(0.0f, 0.0f, 0.0f, 0.0f), make_float4(0.0f, 0.0f, 0.0f, 0.0f)};
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    for (int j = j0; j < j1; ++j) {
        const int pos = k0 + j;
        const int dist = qpos - pos;
        if (dist < 0 || dist >= window) continue;
        const int slot = kv_cap > 0 ? pos % kv_cap : pos;
        const size_t head_off = (size_t) slot * rowb + (size_t) g * cb;
        const size_t sidx = (size_t) slot * sidx_base + g;
        const float ksc = h2f(ks[sidx]);
        const uint2 kc = __ldg((const uint2*) (kci + head_off + l * 8));
        float4 kk[4];
        kk[0] = wideload_int4_four(kc.x, ksc);
        kk[1] = wideload_int4_four(kc.x >> 16, ksc);
        kk[2] = wideload_int4_four(kc.y, ksc);
        kk[3] = wideload_int4_four(kc.y >> 16, ksc);
        float a = 0.0f;
#pragma unroll
        for (int k = 0; k < 4; ++k)
            a += q4[k].x * kk[k].x + q4[k].y * kk[k].y + q4[k].z * kk[k].z + q4[k].w * kk[k].w;
        float logit = wideload_sum8(a) * qk_scale;
        if (dist < extent) {
            const float* pr = proj + (size_t) dist * d_rel;
            float b = 0.0f;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const int d = l * 4 + k;
                if (d < d_rel) b += rr[k] * __ldg(pr + d);
            }
            logit += wideload_sum8(b);
        }
        if (logit > -INFINITY) {
            const float vsc = h2f(vs[sidx]);
            const uint2 vc = __ldg((const uint2*) (vci + head_off + l * 8));
            float4 vv[4];
            vv[0] = wideload_int4_four(vc.x, vsc);
            vv[1] = wideload_int4_four(vc.x >> 16, vsc);
            vv[2] = wideload_int4_four(vc.y, vsc);
            vv[3] = wideload_int4_four(vc.y >> 16, vsc);
            const float m_new = fmaxf(m_w, logit);
            const float alpha = (m_w == -INFINITY) ? 0.0f : expf(m_w - m_new);
            const float beta = expf(logit - m_new);
            m_w = m_new;
            s_w = s_w * alpha + beta;
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                o_w[k].x = o_w[k].x * alpha + vv[k].x * beta;
                o_w[k].y = o_w[k].y * alpha + vv[k].y * beta;
                o_w[k].z = o_w[k].z * alpha + vv[k].z * beta;
                o_w[k].w = o_w[k].w * alpha + vv[k].w * beta;
            }
        }
    }
    if (keep) {
        if (l == 0) pp[0] = m_w, pp[1] = s_w;
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int d = 2 + l * 16 + k * 4;
            pp[d] = o_w[k].x;
            pp[d + 1] = o_w[k].y;
            pp[d + 2] = o_w[k].z;
            pp[d + 3] = o_w[k].w;
        }
    }
}

// attn_flash_range's int counterpart: same online softmax, but each key / value row is dequantized with its
// own per-(token, head) FP16 scale (ks / vs indexed slot * Hkv + g).  qk_scale is the plain 1/sqrt(D): no
// unit-scale folding, the per-row scale already carries the magnitude.
template <int BITS, int DPP = 0, int FA = 0, int EX2 = 0>
__device__ __forceinline__ void attn_flash_range_int(const float* qh, const float* rh, const uint8_t* kci,
                                                     const uint8_t* vci, const uint16_t* ks, const uint16_t* vs,
                                                     int Hkv, int g, int kv_cap, int k0, int j0, int j1, int qpos,
                                                     float qk_scale, const float* proj, int d_rel, int extent,
                                                     int window, float* sm_m, float* sm_s, float sm_o[][128]) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int o = lane * 4;
    const int cb = (BITS == 8) ? 128 : 64;        // code bytes per head
    const size_t rowb = (size_t) Hkv * cb;        // code bytes per token row
    const float4 q4 = (((uintptr_t) qh & 15) == 0) ? __ldg((const float4*) (qh + o))
                                                    : make_float4(qh[o], qh[o + 1], qh[o + 2], qh[o + 3]);
    const float r_lane = (lane < d_rel) ? __ldg(rh + lane) : 0.0f;
    float m_w = -INFINITY, s_w = 0.0f;
    float4 o_w = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    for (int j = j0 + warp; j < j1; j += 4) {
        const int pos = k0 + j;
        const int dist = qpos - pos;
        if (dist < 0 || dist >= window) continue;
        const int slot = kv_cap > 0 ? pos % kv_cap : pos;
        const size_t head_off = (size_t) slot * rowb + (size_t) g * cb;
        const size_t sidx = (size_t) slot * Hkv + g;
        const float ksc = h2f(ks[sidx]);
        const float4 k4 = v_load4_int_sel<BITS, FA>(kci + head_off, lane, ksc);
        float a = q4.x * k4.x + q4.y * k4.y + q4.z * k4.z + q4.w * k4.w;
        float l = (DPP ? wsum_dpp(a) : wsum(a)) * qk_scale;
        if (dist < extent) {
            const float b = (lane < d_rel) ? r_lane * __ldg(proj + (size_t) dist * d_rel + lane) : 0.0f;
            l += DPP ? wsum_dpp(b) : wsum(b);
        }
        if (l > -INFINITY) {
            const float vsc = h2f(vs[sidx]);
            const float4 v4 = v_load4_int_sel<BITS, FA>(vci + head_off, lane, vsc);
            const float m_new = fmaxf(m_w, l);
            const float alpha = (m_w == -INFINITY) ? 0.0f : fa_softmax_exp<EX2>(m_w - m_new);
            const float beta = fa_softmax_exp<EX2>(l - m_new);
            m_w = m_new;
            s_w = s_w * alpha + beta;
            o_w.x = o_w.x * alpha + v4.x * beta;
            o_w.y = o_w.y * alpha + v4.y * beta;
            o_w.z = o_w.z * alpha + v4.z * beta;
            o_w.w = o_w.w * alpha + v4.w * beta;
        }
    }
    if (lane == 0) {
        sm_m[warp] = m_w;
        sm_s[warp] = s_w;
    }
    *(float4*) (&sm_o[warp][o]) = o_w;
    __syncthreads();
}

// attn_split_kernel's int counterpart (partials identical, so attn_combine_kernel / _gate_kernel are reused
// with v_inv_scale = 1.0f: the per-row scale was applied at load).
template <int BITS, int DPP = 0, int FA = 0, int EX2 = 0>
__global__ void attn_split_int_kernel(const float* q, const float* r, const uint8_t* kci, const uint8_t* vci,
                                      const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv, int k0,
                                      int nk, int qpos0, float qk_scale, const float* proj, int d_rel, int extent,
                                      int window, float* part, const int* pos_d) {
    if (pos_d) {
        qpos0 = *pos_d;
        k0 = window < (1 << 30) ? max(0, qpos0 - window + 1) : 0;
        nk = qpos0 + (int) gridDim.z - k0;
    }
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, p = blockIdx.y, P = gridDim.y, qi = blockIdx.z, g = h / (H / Hkv), tid = threadIdx.x;
    const int qpos = qpos0 + qi;
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    const float* qh = q + ((size_t) qi * H + h) * 128;
    const float* rh = r + ((size_t) qi * H + h) * d_rel;
    attn_flash_range_int<BITS, DPP, FA, EX2>(qh, rh, kci, vci, ks, vs, Hkv, g, kv_cap, k0, j0, j1, qpos, qk_scale,
                                             proj, d_rel, extent, window, sm_m, sm_s, sm_o);
    const float M = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
    float sum = 0.0f, o = 0.0f;
    if (M > -INFINITY) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (sm_m[w] == -INFINITY) continue;
            const float f = fa_softmax_exp<EX2>(sm_m[w] - M);
            sum += sm_s[w] * f;
            o += sm_o[w][tid] * f;
        }
    }
    float* pp = part + (((size_t) qi * H + h) * P + p) * 130;
    if (tid == 0) pp[0] = M, pp[1] = sum;
    pp[2 + tid] = o;
}

// ---- True batched (vLLM-style) decode attention.  One kernel advances n_rows queries that may belong to
// DIFFERENT slots; each row qi reads its own K/V cache through a device pointer table (kci_tab[qi] etc.) and
// its own position pos[qi].  The math inside attn_flash_range_int is byte-for-byte the single-token path, so a
// batch of one row is bit-identical to attn_split_int_kernel.  Partials use the same [qi,H,P,130] layout, so
// attn_combine_gate_kernel is reused unchanged (grid.y = n_rows).  Gated by IK2_BATCH_ATTN, default off.
template <int BITS, int DPP = 0>
__global__ void attn_split_int_rows_kernel(const float* q, const float* r, const uint8_t* const* kci_tab,
                                           const uint8_t* const* vci_tab, const uint16_t* const* ks_tab,
                                           const uint16_t* const* vs_tab, int kv_cap, int H, int Hkv,
                                           float qk_scale, const float* proj, int d_rel, int extent, int window,
                                           float* part, const int* pos) {
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, p = blockIdx.y, P = gridDim.y, qi = blockIdx.z, g = h / (H / Hkv), tid = threadIdx.x;
    const int qpos = pos[qi];
    const int k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0;
    const int nk = qpos + 1 - k0;                 // one next-token query per row
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    const float* qh = q + ((size_t) qi * H + h) * 128;
    const float* rh = r + ((size_t) qi * H + h) * d_rel;
    attn_flash_range_int<BITS, DPP>(qh, rh, kci_tab[qi], vci_tab[qi], ks_tab[qi], vs_tab[qi], Hkv, g, kv_cap, k0,
                                    j0, j1, qpos, qk_scale, proj, d_rel, extent, window, sm_m, sm_s, sm_o);
    const float M = fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3]));
    float sum = 0.0f, o = 0.0f;
    if (M > -INFINITY) {
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            if (sm_m[w] == -INFINITY) continue;
            const float f = expf(sm_m[w] - M);
            sum += sm_s[w] * f;
            o += sm_o[w][tid] * f;
        }
    }
    float* pp = part + (((size_t) qi * H + h) * P + p) * 130;
    if (tid == 0) pp[0] = M, pp[1] = sum;
    pp[2 + tid] = o;
}

// Host launcher for the batched rows kernel.  kci_tab/vci_tab/ks_tab/vs_tab are DEVICE arrays of n_rows pointers
// (each pointing at that row's slot KV buffer); pos is a DEVICE int array of n_rows positions.  out/gate/scratch
// are [n_rows,H,*].  When n_rows == 1 this is bit-identical to attn_decode_int_at.
void attn_decode_int_rows(int bits, const float* q, const float* r, const float* sink,
                          const uint8_t* const* kci_tab, const uint8_t* const* vci_tab,
                          const uint16_t* const* ks_tab, const uint16_t* const* vs_tab,
                          int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale, const float* proj,
                          int d_rel, int extent, int window, int n_rows, float* scratch, float* out,
                          cudaStream_t s, const float* gate, int dpp) {
    if (bits != 8 && bits != 4) {
        std::fprintf(stderr, "ik2::k::attn_decode_int_rows: bits %d (4 or 8)\n", bits);
        std::abort();
    }
    if (H <= 0 || Hkv <= 0 || H % Hkv || n_rows <= 0) {
        std::fprintf(stderr, "ik2::k::attn_decode_int_rows: H %d Hkv %d n_rows %d\n", H, Hkv, n_rows);
        std::abort();
    }
    if (bits == 8)
        attn_split_int_rows_kernel<8, 0><<<dim3(H, (unsigned) P, (unsigned) n_rows), 128, 0, s>>>(
            q, r, kci_tab, vci_tab, ks_tab, vs_tab, kv_cap, H, Hkv, qk_scale, proj, d_rel, extent, window, scratch, pos);
    else if (dpp)
        attn_split_int_rows_kernel<4, 1><<<dim3(H, (unsigned) P, (unsigned) n_rows), 128, 0, s>>>(
            q, r, kci_tab, vci_tab, ks_tab, vs_tab, kv_cap, H, Hkv, qk_scale, proj, d_rel, extent, window, scratch, pos);
    else
        attn_split_int_rows_kernel<4, 0><<<dim3(H, (unsigned) P, (unsigned) n_rows), 128, 0, s>>>(
            q, r, kci_tab, vci_tab, ks_tab, vs_tab, kv_cap, H, Hkv, qk_scale, proj, d_rel, extent, window, scratch, pos);
    check("attn_decode_int_rows (split)");
    attn_combine_gate_kernel<<<dim3(H, (unsigned) n_rows), 128, 0, s>>>(sink, scratch, H, P, 1.0f, gate, out);
    check("attn_decode_int_rows (combine)");
}

// Forward decl: the GQA rows kernel is defined just below the plain rows launcher.
template <int BITS, int DPP, int R>
__global__ void attn_split_int_rows_gqa_kernel(const float* q, const float* r, const uint8_t* const* kci_tab,
                                               const uint8_t* const* vci_tab, const uint16_t* const* ks_tab,
                                               const uint16_t* const* vs_tab, int kv_cap, int H, int Hkv,
                                               float qk_scale, const float* proj, int d_rel, int extent, int window,
                                               float* part, const int* pos);

// GQA-shared batched launcher: grid (Hkv, P, n_rows); only H/Hkv == 3 is supported (R = 3).
void attn_decode_int_rows_gqa(int bits, const float* q, const float* r, const float* sink,
                              const uint8_t* const* kci_tab, const uint8_t* const* vci_tab,
                              const uint16_t* const* ks_tab, const uint16_t* const* vs_tab,
                              int kv_cap, int H, int Hkv, const int* pos, int P, float qk_scale, const float* proj,
                              int d_rel, int extent, int window, int n_rows, float* scratch, float* out,
                              cudaStream_t s, const float* gate, int dpp) {
    if (bits != 8 && bits != 4) std::abort();
    if (H <= 0 || Hkv <= 0 || H % Hkv || H / Hkv != 3 || n_rows <= 0) {
        std::fprintf(stderr, "ik2::k::attn_decode_int_rows_gqa: H %d Hkv %d n_rows %d\n", H, Hkv, n_rows);
        std::abort();
    }
    if (bits == 8)
        attn_split_int_rows_gqa_kernel<8, 0, 3><<<dim3(Hkv, (unsigned) P, (unsigned) n_rows), 128, 0, s>>>(
            q, r, kci_tab, vci_tab, ks_tab, vs_tab, kv_cap, H, Hkv, qk_scale, proj, d_rel, extent, window, scratch, pos);
    else if (dpp)
        attn_split_int_rows_gqa_kernel<4, 1, 3><<<dim3(Hkv, (unsigned) P, (unsigned) n_rows), 128, 0, s>>>(
            q, r, kci_tab, vci_tab, ks_tab, vs_tab, kv_cap, H, Hkv, qk_scale, proj, d_rel, extent, window, scratch, pos);
    else
        attn_split_int_rows_gqa_kernel<4, 0, 3><<<dim3(Hkv, (unsigned) P, (unsigned) n_rows), 128, 0, s>>>(
            q, r, kci_tab, vci_tab, ks_tab, vs_tab, kv_cap, H, Hkv, qk_scale, proj, d_rel, extent, window, scratch, pos);
    check("attn_decode_int_rows_gqa (split)");
    attn_combine_gate_kernel<<<dim3(H, (unsigned) n_rows), 128, 0, s>>>(sink, scratch, H, P, 1.0f, gate, out);
    check("attn_decode_int_rows_gqa (combine)");
}

// the R query heads h = g*R + r of KV head g, so each K row (and, when needed, V row) is dequantized once and
// reused by R *independent* online softmaxes.  The key-walk order (j = j0 + warp, stride 4), the four-term
// lane dot, the rel-bias projection, the wsum_dpp/wsum choice, the fmaxf/expf alpha/beta update and the
// ascending-w four-warp shared-memory merge are copied verbatim from attn_flash_range_int /
// attn_split_int_kernel; only the K/V reads are shared.  The partials therefore stay bit-identical and the
// [H,P,130] layout lets attn_combine_kernel / attn_combine_gate_kernel run unchanged.  R is a compile-time
// parameter so the per-head q4 / rel / accumulator arrays stay in registers; R = 3 is the target (H=24, Hkv=8).
template <int BITS, int DPP = 0, int R = 3, int FA = 0, int PV = 0, int EX2 = 0>
__global__ void attn_split_int_gqa_kernel(const float* q, const float* r, const uint8_t* kci, const uint8_t* vci,
                                          const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv, int k0,
                                          int nk, int qpos0, float qk_scale, const float* proj, int d_rel, int extent,
                                          int window, float* part, const int* pos_d) {
    if (pos_d) {
        qpos0 = *pos_d;
        k0 = window < (1 << 30) ? max(0, qpos0 - window + 1) : 0;
        nk = qpos0 + (int) gridDim.z - k0;
    }
    __shared__ float sm_m[4][R], sm_s[4][R];
    __shared__ __align__(16) float sm_o[4][R][128];
    const int g = blockIdx.x, p = blockIdx.y, P = gridDim.y, qi = blockIdx.z, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, o = lane * 4;
    const int cb = (BITS == 8) ? 128 : 64;        // code bytes per head
    const size_t rowb = (size_t) Hkv * cb;        // code bytes per token row
    const int qpos = qpos0 + qi;
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    float4 q4[R];
    float r_lane[R], m_w[R], s_w[R];
    float4 o_w[R];
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
        const int h = g * R + rr;
        const float* qh = q + ((size_t) qi * H + h) * 128;
        q4[rr] = (((uintptr_t) qh & 15) == 0) ? __ldg((const float4*) (qh + o))
                                               : make_float4(qh[o], qh[o + 1], qh[o + 2], qh[o + 3]);
        const float* rh = r + ((size_t) qi * H + h) * d_rel;   // rel is per Q head, not per KV head
        r_lane[rr] = (lane < d_rel) ? __ldg(rh + lane) : 0.0f;
        m_w[rr] = -INFINITY;
        s_w[rr] = 0.0f;
        o_w[rr] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int j = j0 + warp; j < j1; j += 4) {
        const int pos = k0 + j;
        const int dist = qpos - pos;
        if (dist < 0 || dist >= window) continue;
        const int slot = kv_cap > 0 ? pos % kv_cap : pos;
        const size_t head_off = (size_t) slot * rowb + (size_t) g * cb;
        const size_t sidx = (size_t) slot * Hkv + g;
        const float ksc = h2f(ks[sidx]);
        const float4 k4 = v_load4_int_sel<BITS, FA>(kci + head_off, lane, ksc);   // ONE K load shared by all R heads
        const bool rel = dist < extent;
        const float pb = (rel && lane < d_rel) ? __ldg(proj + (size_t) dist * d_rel + lane) : 0.0f;
        float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        bool v_ready = false; (void) v_ready;
        if constexpr (PV) {
            // IK2_FA_RDNA3 bit1: same address, same scale, same values - just hoisted before the R loop.
            const float vsc = h2f(vs[sidx]);
            v4 = v_load4_int_sel<BITS, FA>(vci + head_off, lane, vsc);
        }
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            float a = q4[rr].x * k4.x + q4[rr].y * k4.y + q4[rr].z * k4.z + q4[rr].w * k4.w;
            float l = (DPP ? wsum_dpp(a) : wsum(a)) * qk_scale;
            if (rel) {
                const float b = (lane < d_rel) ? r_lane[rr] * pb : 0.0f;
                l += DPP ? wsum_dpp(b) : wsum(b);
            }
            if (l > -INFINITY) {
                if constexpr (!PV) {
                    if (!v_ready) {
                        const float vsc = h2f(vs[sidx]);
                        v4 = v_load4_int_sel<BITS, FA>(vci + head_off, lane, vsc);    // ONE V load shared by all R heads
                        v_ready = true;
                    }
                }
                const float m_new = fmaxf(m_w[rr], l);
                const float alpha = (m_w[rr] == -INFINITY) ? 0.0f : fa_softmax_exp<EX2>(m_w[rr] - m_new);
                const float beta = fa_softmax_exp<EX2>(l - m_new);
                m_w[rr] = m_new;
                s_w[rr] = s_w[rr] * alpha + beta;
                o_w[rr].x = o_w[rr].x * alpha + v4.x * beta;
                o_w[rr].y = o_w[rr].y * alpha + v4.y * beta;
                o_w[rr].z = o_w[rr].z * alpha + v4.z * beta;
                o_w[rr].w = o_w[rr].w * alpha + v4.w * beta;
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int rr = 0; rr < R; ++rr) sm_m[warp][rr] = m_w[rr], sm_s[warp][rr] = s_w[rr];
    }
#pragma unroll
    for (int rr = 0; rr < R; ++rr) *(float4*) (&sm_o[warp][rr][o]) = o_w[rr];
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
        const float M = fmaxf(fmaxf(sm_m[0][rr], sm_m[1][rr]), fmaxf(sm_m[2][rr], sm_m[3][rr]));
        float sum = 0.0f, out = 0.0f;
        if (M > -INFINITY) {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (sm_m[w][rr] == -INFINITY) continue;
                const float f = fa_softmax_exp<EX2>(sm_m[w][rr] - M);
                sum += sm_s[w][rr] * f;
                out += sm_o[w][rr][tid] * f;
            }
        }
        const int h = g * R + rr;
        float* pp = part + (((size_t) qi * H + h) * P + p) * 130;
        if (tid == 0) pp[0] = M, pp[1] = sum;
        pp[2 + tid] = out;
    }
}

// True-batched GQA rows: like attn_split_int_rows_kernel but grid.x is the KV head (Hkv) and R=3 query heads
// share each K/V row.  kci_tab/vci_tab/ks_tab/vs_tab are DEVICE arrays of n_rows slot pointers; pos is DEVICE
// int[n_rows].  Partials stay [qi,H,P,130], so the combine is unchanged.  Bit-exact to attn_decode_int_at_gqa.
template <int BITS, int DPP = 0, int R = 3>
__global__ void attn_split_int_rows_gqa_kernel(const float* q, const float* r,
                                               const uint8_t* const* kci_tab, const uint8_t* const* vci_tab,
                                               const uint16_t* const* ks_tab, const uint16_t* const* vs_tab,
                                               int kv_cap, int H, int Hkv, float qk_scale, const float* proj,
                                               int d_rel, int extent, int window, float* part, const int* pos) {
    const uint8_t* kci = kci_tab[blockIdx.z];
    const uint8_t* vci = vci_tab[blockIdx.z];
    const uint16_t* ks = ks_tab[blockIdx.z];
    const uint16_t* vs = vs_tab[blockIdx.z];
    __shared__ float sm_m[4][R], sm_s[4][R];
    __shared__ __align__(16) float sm_o[4][R][128];
    const int g = blockIdx.x, p = blockIdx.y, P = gridDim.y, qi = blockIdx.z, tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, o = lane * 4;
    const int cb = (BITS == 8) ? 128 : 64;
    const size_t rowb = (size_t) Hkv * cb;
    const int qpos = pos[qi];
    const int k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0;
    const int nk = qpos + 1 - k0;
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    float4 q4[R]; float r_lane[R], m_w[R], s_w[R]; float4 o_w[R];
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
        const int h = g * R + rr;
        const float* qh = q + ((size_t) qi * H + h) * 128;
        q4[rr] = (((uintptr_t) qh & 15) == 0) ? __ldg((const float4*) (qh + o))
                                              : make_float4(qh[o], qh[o + 1], qh[o + 2], qh[o + 3]);
        const float* rh = r + ((size_t) qi * H + h) * d_rel;
        r_lane[rr] = (lane < d_rel) ? __ldg(rh + lane) : 0.0f;
        m_w[rr] = -INFINITY; s_w[rr] = 0.0f; o_w[rr] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int j = j0 + warp; j < j1; j += 4) {
        const int pos_j = k0 + j;
        const int dist = qpos - pos_j;
        if (dist < 0 || dist >= window) continue;
        const int slot = kv_cap > 0 ? pos_j % kv_cap : pos_j;
        const size_t head_off = (size_t) slot * rowb + (size_t) g * cb;
        const size_t sidx = (size_t) slot * Hkv + g;
        const float ksc = h2f(ks[sidx]);
        const float4 k4 = v_load4_int<BITS>(kci + head_off, lane, ksc);
        const bool rel = dist < extent;
        const float pb = (rel && lane < d_rel) ? __ldg(proj + (size_t) dist * d_rel + lane) : 0.0f;
        float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        bool v_ready = false;
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            float a = q4[rr].x * k4.x + q4[rr].y * k4.y + q4[rr].z * k4.z + q4[rr].w * k4.w;
            float l = (DPP ? wsum_dpp(a) : wsum(a)) * qk_scale;
            if (rel) { const float b = (lane < d_rel) ? r_lane[rr] * pb : 0.0f; l += DPP ? wsum_dpp(b) : wsum(b); }
            if (l > -INFINITY) {
                if (!v_ready) { const float vsc = h2f(vs[sidx]); v4 = v_load4_int<BITS>(vci + head_off, lane, vsc); v_ready = true; }
                const float m_new = fmaxf(m_w[rr], l), alpha = (m_w[rr] == -INFINITY) ? 0.0f : expf(m_w[rr] - m_new),
                            beta = expf(l - m_new);
                m_w[rr] = m_new; s_w[rr] = s_w[rr] * alpha + beta;
                o_w[rr].x = o_w[rr].x * alpha + v4.x * beta; o_w[rr].y = o_w[rr].y * alpha + v4.y * beta;
                o_w[rr].z = o_w[rr].z * alpha + v4.z * beta; o_w[rr].w = o_w[rr].w * alpha + v4.w * beta;
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int rr = 0; rr < R; ++rr) sm_m[warp][rr] = m_w[rr], sm_s[warp][rr] = s_w[rr];
    }
#pragma unroll
    for (int rr = 0; rr < R; ++rr) *(float4*) (&sm_o[warp][rr][o]) = o_w[rr];
    __syncthreads();
#pragma unroll
    for (int rr = 0; rr < R; ++rr) {
        const float M = fmaxf(fmaxf(sm_m[0][rr], sm_m[1][rr]), fmaxf(sm_m[2][rr], sm_m[3][rr]));
        float sum = 0.0f, out = 0.0f;
        if (M > -INFINITY) {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (sm_m[w][rr] == -INFINITY) continue;
                const float f = expf(sm_m[w][rr] - M);
                sum += sm_s[w][rr] * f; out += sm_o[w][rr][tid] * f;
            }
        }
        const int h = g * R + rr;
        float* pp = part + (((size_t) qi * H + h) * P + p) * 130;
        if (tid == 0) pp[0] = M, pp[1] = sum;
        pp[2 + tid] = out;
    }
}

// int8 / int4 counterpart of attn_multi_shared_kernel: same one-K-load / one-V-load-per-key sharing, with the
// per-(slot, kv head) fp16 scales applied at load exactly as in attn_flash_range_int (v_inv_scale = 1 in combine).
template <int BITS, int NQ>
__global__ void attn_multi_shared_int_kernel(const float* q, const float* r, const uint8_t* kci,
                                             const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap,
                                             int H, int Hkv, int k0, int nk, int qpos0, float qk_scale,
                                             const float* proj, int d_rel, int extent, int window, float* part,
                                             const int* pos_d, int nq) {
    if (pos_d) {
        qpos0 = *pos_d;
        k0 = window < (1 << 30) ? max(0, qpos0 - window + 1) : 0;
        nk = qpos0 + nq - k0;
    }
    __shared__ float sm_m[4][NQ], sm_s[4][NQ];
    __shared__ __align__(16) float sm_o[4][NQ][128];
    const int h = blockIdx.x, p = blockIdx.y, P = gridDim.y, g = h / (H / Hkv), tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, o = lane * 4;
    const int cb = (BITS == 8) ? 128 : 64;
    const size_t rowb = (size_t) Hkv * cb;
    const int len = (nk + P - 1) / P, j0 = p * len, j1 = min(nk, j0 + len);
    float4 q4[NQ];
    float r_lane[NQ], m_w[NQ], s_w[NQ];
    float4 o_w[NQ];
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {
        const float* qh = q + ((size_t) qi * H + h) * 128;
        q4[qi] = (((uintptr_t) qh & 15) == 0) ? __ldg((const float4*) (qh + o))
                                              : make_float4(qh[o], qh[o + 1], qh[o + 2], qh[o + 3]);
        const float* rh = r + ((size_t) qi * H + h) * d_rel;
        r_lane[qi] = (lane < d_rel) ? __ldg(rh + lane) : 0.0f;
        m_w[qi] = -INFINITY;
        s_w[qi] = 0.0f;
        o_w[qi] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    for (int j = j0 + warp; j < j1; j += 4) {
        const int pos = k0 + j;
        const int dist0 = qpos0 - pos;
        if (dist0 <= -NQ || dist0 >= window) continue;
        const int slot = kv_cap > 0 ? pos % kv_cap : pos;
        const size_t head_off = (size_t) slot * rowb + (size_t) g * cb;
        const size_t sidx = (size_t) slot * Hkv + g;
        const float ksc = h2f(ks[sidx]);
        const float4 k4 = v_load4_int<BITS>(kci + head_off, lane, ksc);   // ONE K load shared by all rows
        float4 v4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        bool v_loaded = false;
#pragma unroll
        for (int qi = 0; qi < NQ; ++qi) {
            const int dist = dist0 + qi;
            if (dist < 0 || dist >= window) continue;
            float a = q4[qi].x * k4.x + q4[qi].y * k4.y + q4[qi].z * k4.z + q4[qi].w * k4.w;
            float l = wsum(a) * qk_scale;
            if (dist < extent) {
                const float b = (lane < d_rel) ? r_lane[qi] * __ldg(proj + (size_t) dist * d_rel + lane) : 0.0f;
                l += wsum(b);
            }
            if (l > -INFINITY) {
                if (!v_loaded) {
                    const float vsc = h2f(vs[sidx]);
                    v4 = v_load4_int<BITS>(vci + head_off, lane, vsc);    // ONE V load shared by all rows
                    v_loaded = true;
                }
                const float m_new = fmaxf(m_w[qi], l);
                const float alpha = (m_w[qi] == -INFINITY) ? 0.0f : expf(m_w[qi] - m_new);
                const float beta = expf(l - m_new);
                m_w[qi] = m_new;
                s_w[qi] = s_w[qi] * alpha + beta;
                o_w[qi].x = o_w[qi].x * alpha + v4.x * beta;
                o_w[qi].y = o_w[qi].y * alpha + v4.y * beta;
                o_w[qi].z = o_w[qi].z * alpha + v4.z * beta;
                o_w[qi].w = o_w[qi].w * alpha + v4.w * beta;
            }
        }
    }
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {
        if (lane == 0) {
            sm_m[warp][qi] = m_w[qi];
            sm_s[warp][qi] = s_w[qi];
        }
        *(float4*) (&sm_o[warp][qi][o]) = o_w[qi];
    }
    __syncthreads();
#pragma unroll
    for (int qi = 0; qi < NQ; ++qi) {
        const float M = fmaxf(fmaxf(sm_m[0][qi], sm_m[1][qi]), fmaxf(sm_m[2][qi], sm_m[3][qi]));
        float sum = 0.0f, ov = 0.0f;
        if (M > -INFINITY) {
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                if (sm_m[w][qi] == -INFINITY) continue;
                const float f = expf(sm_m[w][qi] - M);
                sum += sm_s[w][qi] * f;
                ov += sm_o[w][qi][tid] * f;
            }
        }
        float* pp = part + (((size_t) qi * H + h) * P + p) * 130;
        if (tid == 0) pp[0] = M, pp[1] = sum;
        pp[2 + tid] = ov;
    }
}

template <int BITS, int NQ>
void attn_multi_shared_int_launch(const float* q, const float* r, const uint8_t* kci, const uint8_t* vci,
                                  const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv, int k0, int nk,
                                  int qpos0, float qk_scale, const float* proj, int d_rel, int extent, int window,
                                  float* part, const int* pos_d, int P, cudaStream_t s) {
    attn_multi_shared_int_kernel<BITS, NQ><<<dim3(H, (unsigned) P), 128, 0, s>>>(
        q, r, kci, vci, ks, vs, kv_cap, H, Hkv, k0, nk, qpos0, qk_scale, proj, d_rel, extent, window, part, pos_d, NQ);
}

template <int BITS>
void attn_multi_shared_int_dispatch(const float* q, const float* r, const uint8_t* kci, const uint8_t* vci,
                                    const uint16_t* ks, const uint16_t* vs, int kv_cap, int nq, int H, int Hkv,
                                    int k0, int nk, int qpos0, float qk_scale, const float* proj, int d_rel,
                                    int extent, int window, float* part, const int* pos_d, int P, cudaStream_t s) {
#define IK2_MSH_INT_CASE(N)                                                                                         \
    case N:                                                                                                         \
        attn_multi_shared_int_launch<BITS, N>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, k0, nk, qpos0, qk_scale, proj, \
                                              d_rel, extent, window, part, pos_d, P, s);                             \
        break
    switch (nq) {
        IK2_MSH_INT_CASE(1);
        IK2_MSH_INT_CASE(2);
        IK2_MSH_INT_CASE(3);
        IK2_MSH_INT_CASE(4);
        IK2_MSH_INT_CASE(5);
        IK2_MSH_INT_CASE(6);
        IK2_MSH_INT_CASE(7);
        IK2_MSH_INT_CASE(8);
        default: std::abort();
    }
#undef IK2_MSH_INT_CASE
}

template <int BITS>
__global__ void attn_decode_int_kernel(const float* q, const float* r, const float* sink, const uint8_t* kci,
                                       const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int H,
                                       int Hkv, int k0, int nk, int qpos, float qk_scale, const float* proj,
                                       int d_rel, int extent, int window, float* out) {
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, g = h / (H / Hkv), tid = threadIdx.x;
    const float* qh = q + (size_t) h * 128;
    const float* rh = r + (size_t) h * d_rel;
    attn_flash_range_int<BITS>(qh, rh, kci, vci, ks, vs, Hkv, g, kv_cap, k0, 0, nk, qpos, qk_scale, proj, d_rel,
                               extent, window, sm_m, sm_s, sm_o);
    const float sk = sink[h];
    const float M = fmaxf(sk, fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3])));
    float den = expf(sk - M), o = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (sm_m[w] == -INFINITY) continue;
        const float f = expf(sm_m[w] - M);
        den += sm_s[w] * f;
        o += sm_o[w][tid] * f;
    }
    out[(size_t) h * 128 + tid] = o / den;
}

template <int BITS>
__global__ void attn_decode_onepass_int_kernel(const float* q, const float* r, const float* sink, const uint8_t* kci,
                                               const uint8_t* vci, const uint16_t* ks, const uint16_t* vs,
                                               int kv_cap, int H, int Hkv, const int* pos_d, float qk_scale,
                                               const float* proj, int d_rel, int extent, int window,
                                               const float* gate, float* out) {
    __shared__ float sm_m[4], sm_s[4];
    __shared__ __align__(16) float sm_o[4][128];
    const int h = blockIdx.x, g = h / (H / Hkv), tid = threadIdx.x;
    const int qpos = *pos_d;
    const int k0 = window < (1 << 30) ? max(0, qpos - window + 1) : 0;
    const int nk = qpos + 1 - k0;
    const float* qh = q + (size_t) h * 128;
    const float* rh = r + (size_t) h * d_rel;
    attn_flash_range_int<BITS>(qh, rh, kci, vci, ks, vs, Hkv, g, kv_cap, k0, 0, nk, qpos, qk_scale, proj, d_rel,
                               extent, window, sm_m, sm_s, sm_o);
    const float sk = sink[h];
    const float M = fmaxf(sk, fmaxf(fmaxf(sm_m[0], sm_m[1]), fmaxf(sm_m[2], sm_m[3])));
    float den = expf(sk - M), o = 0.0f;
#pragma unroll
    for (int w = 0; w < 4; ++w) {
        if (sm_m[w] == -INFINITY) continue;
        const float f = expf(sm_m[w] - M);
        den += sm_s[w] * f;
        o += sm_o[w][tid] * f;
    }
    float v = o / den;
    if (gate) v *= 1.0f / (1.0f + expf(-gate[(size_t) h * 128 + tid]));
    out[(size_t) h * 128 + tid] = v;
}

// unpack int ring rows into contiguous FP16 [nk, KD] for the cuBLAS prefill GEMMs (the int counterpart of
// unpack_kv_fp8_to_f16; the per-row scale is applied here, so no k_inv / v_inv afterwards).
template <int BITS>
__global__ void unpack_kv_int_to_f16_vec4_kernel(const uint8_t* kci, const uint8_t* vci, const uint16_t* ks,
                                                 const uint16_t* vs, int kv_cap, int k0, int nk, int KD,
                                                 uint16_t* k_out16, uint16_t* v_out16) {
    const int kd4 = KD >> 2;
    const int d4 = blockIdx.x * blockDim.x + threadIdx.x;
    if (d4 >= kd4) return;
    const int Hkv = KD >> 7;                 // KD / 128
    const int g = d4 >> 5;                   // 32 float4 per 128-elem head
    const int cb = (BITS == 8) ? 128 : 64;
    const size_t rowb = (size_t) Hkv * cb;
    for (int j = blockIdx.y; j < nk; j += gridDim.y) {
        const int slot = kv_cap > 0 ? (k0 + j) % kv_cap : (k0 + j);
        const size_t head_off = (size_t) slot * rowb + (size_t) g * cb;
        const size_t sidx = (size_t) slot * Hkv + g;
        const size_t dst_off = (size_t) j * KD;
        if (k_out16) {
            const float4 f = v_load4_int<BITS>(kci + head_off, d4 & 31, h2f(ks[sidx]));
            ((uint2*) (k_out16 + dst_off))[d4] = make_uint2(f2_to_u32(f.x, f.y), f2_to_u32(f.z, f.w));
        }
        if (v_out16) {
            const float4 f = v_load4_int<BITS>(vci + head_off, d4 & 31, h2f(vs[sidx]));
            ((uint2*) (v_out16 + dst_off))[d4] = make_uint2(f2_to_u32(f.x, f.y), f2_to_u32(f.z, f.w));
        }
    }
}

template <int BITS>
__global__ void unpack_kv_int_to_f16_kernel(const uint8_t* kci, const uint8_t* vci, const uint16_t* ks,
                                            const uint16_t* vs, int kv_cap, int k0, int nk, int KD,
                                            uint16_t* k_out16, uint16_t* v_out16) {
    const int Hkv = KD >> 7;
    const size_t rowb = (size_t) Hkv * ((BITS == 8) ? 128 : 64);
    const size_t total = (size_t) nk * KD;
    for (size_t idx = (size_t) blockIdx.x * blockDim.x + threadIdx.x; idx < total;
         idx += (size_t) gridDim.x * blockDim.x) {
        const int j = (int) (idx / (size_t) KD);
        const int d = (int) (idx % (size_t) KD);
        const int slot = kv_cap > 0 ? (k0 + j) % kv_cap : (k0 + j);
        const size_t sidx = (size_t) slot * Hkv + (d >> 7);
        const size_t src = (size_t) slot * rowb;
        if (k_out16) {
            const float sc = h2f(ks[sidx]);
            const float x = (BITS == 8) ? (float) (int8_t) kci[src + d]
                                        : (float) kv_int_nib((kci[src + (d >> 1)] >> ((d & 1) ? 4 : 0)) & 0xFu);
            k_out16[idx] = f2h(x * sc);
        }
        if (v_out16) {
            const float sc = h2f(vs[sidx]);
            const float x = (BITS == 8) ? (float) (int8_t) vci[src + d]
                                        : (float) kv_int_nib((vci[src + (d >> 1)] >> ((d & 1) ? 4 : 0)) & 0xFu);
            v_out16[idx] = f2h(x * sc);
        }
    }
}

// ---- int host wrappers (bits = 8 or 4 selects the instantiation; everything else mirrors the fp8 hosts) ----

static void kv_int_bits_check(int bits, const char* who) {
    if (bits != 8 && bits != 4) {
        std::fprintf(stderr, "ik2::k::%s: bits %d (expected 8 or 4)\n", who, bits);
        std::abort();
    }
}

template <int BITS>
static void prep_qkv_r_int_launch(float* q, const float* q_clip, const float* k_in, uint8_t* kci, uint16_t* ks,
                                  const float* v_in, uint8_t* vci, uint16_t* vs, float* r, const float* r_clip,
                                  int T, int H, int KV, int D, int d_rel, float qk_clip, float rel_clip, float eps,
                                  const int* pos, float alpha, float n_floor, int pos0, int kv_cap,
                                  const int* pos0_d, uint16_t* q16, const uint16_t* sink_w, const float* sink_x,
                                  int sink_k, float* sink_y, cudaStream_t s) {
    if (T <= 0) return;
    if (BITS == 4 && (D & 1)) {
        std::fprintf(stderr, "ik2::k::prep_qkv_r_int: int4 needs even D, got %d\n", D);
        std::abort();
    }
    const int total = T * (2 * H + 2 * KV) + (sink_w ? H : 0);
    prep_qkv_r_int_kernel<BITS><<<(unsigned) total, 32, 0, s>>>(
        q, q_clip, k_in, kci, ks, v_in, vci, vs, r, r_clip, T, H, KV, D, d_rel, qk_clip, rel_clip, eps, pos, alpha,
        n_floor, pos0, kv_cap, pos0_d, q16, sink_w, sink_x, sink_k, sink_y);
    check("prep_qkv_r_int");
}

void prep_qkv_r_int(int bits, float* q, const float* q_clip, const float* k_in, uint8_t* kci, uint16_t* ks,
                    const float* v_in, uint8_t* vci, uint16_t* vs, float* r, const float* r_clip, int T, int H,
                    int KV, int D, int d_rel, float qk_clip, float rel_clip, float eps, const int* pos, float alpha,
                    float n_floor, int pos0, int kv_cap, cudaStream_t s) {
    kv_int_bits_check(bits, "prep_qkv_r_int");
    if (bits == 8)
        prep_qkv_r_int_launch<8>(q, q_clip, k_in, kci, ks, v_in, vci, vs, r, r_clip, T, H, KV, D, d_rel, qk_clip,
                                 rel_clip, eps, pos, alpha, n_floor, pos0, kv_cap, nullptr, nullptr, nullptr,
                                 nullptr, 0, nullptr, s);
    else
        prep_qkv_r_int_launch<4>(q, q_clip, k_in, kci, ks, v_in, vci, vs, r, r_clip, T, H, KV, D, d_rel, qk_clip,
                                 rel_clip, eps, pos, alpha, n_floor, pos0, kv_cap, nullptr, nullptr, nullptr,
                                 nullptr, 0, nullptr, s);
}

void prep_qkv_r_int_at(int bits, float* q, const float* q_clip, const float* k_in, uint8_t* kci, uint16_t* ks,
                       const float* v_in, uint8_t* vci, uint16_t* vs, float* r, const float* r_clip, int H, int KV,
                       int D, int d_rel, float qk_clip, float rel_clip, float eps, const int* pos, bool log_scale,
                       float alpha, float n_floor, int kv_cap, uint16_t* q16, cudaStream_t s,
                       const uint16_t* sink_w, const float* sink_x, int sink_k, float* sink_y) {
    kv_int_bits_check(bits, "prep_qkv_r_int_at");
    if (bits == 8)
        prep_qkv_r_int_launch<8>(q, q_clip, k_in, kci, ks, v_in, vci, vs, r, r_clip, 1, H, KV, D, d_rel, qk_clip,
                                 rel_clip, eps, log_scale ? pos : nullptr, alpha, n_floor, 0, kv_cap, pos, q16,
                                 sink_w, sink_x, sink_k, sink_y, s);
    else
        prep_qkv_r_int_launch<4>(q, q_clip, k_in, kci, ks, v_in, vci, vs, r, r_clip, 1, H, KV, D, d_rel, qk_clip,
                                 rel_clip, eps, log_scale ? pos : nullptr, alpha, n_floor, 0, kv_cap, pos, q16,
                                 sink_w, sink_x, sink_k, sink_y, s);
}

void prep_qkv_r_int_g(int bits, float* q, const float* q_clip, const float* k_in, uint8_t* kci, uint16_t* ks,
                      const float* v_in, uint8_t* vci, uint16_t* vs, float* r, const float* r_clip, int T, int H,
                      int KV, int D, int d_rel, float qk_clip, float rel_clip, float eps, const int* pos,
                      float alpha, float n_floor, const int* pos0_d, int kv_cap, cudaStream_t s) {
    kv_int_bits_check(bits, "prep_qkv_r_int_g");
    if (bits == 8)
        prep_qkv_r_int_launch<8>(q, q_clip, k_in, kci, ks, v_in, vci, vs, r, r_clip, T, H, KV, D, d_rel, qk_clip,
                                 rel_clip, eps, pos, alpha, n_floor, 0, kv_cap, pos0_d, nullptr, nullptr, nullptr,
                                 0, nullptr, s);
    else
        prep_qkv_r_int_launch<4>(q, q_clip, k_in, kci, ks, v_in, vci, vs, r, r_clip, T, H, KV, D, d_rel, qk_clip,
                                 rel_clip, eps, pos, alpha, n_floor, 0, kv_cap, pos0_d, nullptr, nullptr, nullptr,
                                 0, nullptr, s);
}

// IK2_BATCH_GRAPH_EXACT: T-row f32 sink dot. The loop, h2f conversion and wsum reduction are the sink tail
// of prep_qkv_r_int_kernel verbatim; only the activation pointer is advanced per row. sink_y is [T][H].
__global__ void sink_dot_rows_kernel(const uint16_t* __restrict__ sink_w, const float* __restrict__ sink_x,
                                     int H, int K, int sink_ld, float* __restrict__ sink_y) {
    const int h = blockIdx.x, t = blockIdx.y, lane = threadIdx.x;
    const uint16_t* w = sink_w + (size_t) h * K;
    const float* x = sink_x + (size_t) t * sink_ld;
    float a = 0.0f;
    for (int i = lane; i < K; i += 32) a += h2f(w[i]) * x[i];
    a = wsum(a);
    if (lane == 0) sink_y[(size_t) t * H + h] = a;
}

void sink_dot_rows(const uint16_t* sink_w, const float* sink_x, int T, int H, int K, int sink_ld,
                   float* sink_y, cudaStream_t s) {
    if (!sink_w || T <= 0 || H <= 0 || K <= 0) return;
    sink_dot_rows_kernel<<<dim3((unsigned) H, (unsigned) T), 32, 0, s>>>(sink_w, sink_x, H, K, sink_ld, sink_y);
    check("sink_dot_rows");
}

// True batched write counterpart of prep_qkv_r_int: every row t writes its K/V through its own cache pointers
// (kci_tab[t] etc.) into slot pos[t] (mod kv_cap when given) and uses pos[t] for q/r's tau.  The tables are DEVICE
// arrays of n_rows pointers; pos is a DEVICE int array used for KV slots, and optional tau_pos is a DEVICE int
// array used for q/r tau (null -> tau 1, matching prep_qkv_r_int with a null pos). The K/V stores, q/r clips and
// sink tail are the scalar kernel's calls unchanged, so one row is bit-identical to prep_qkv_r_int with
// pos0 == pos[t] and tau pos == tau_pos[t].
void prep_qkv_r_int_rows(int bits, float* q, const float* q_clip, const float* k_in, uint8_t* const* kci_tab,
                         uint16_t* const* ks_tab, const float* v_in, uint8_t* const* vci_tab,
                         uint16_t* const* vs_tab, float* r, const float* r_clip, int T, int H, int KVh, int D,
                         int d_rel, float qk_clip, float rel_clip, float eps, const int* pos, float alpha,
                         float n_floor, int kv_cap, uint16_t* q16, const uint16_t* sink_w, const float* sink_x,
                         int sink_k, float* sink_y, cudaStream_t s, const int* tau_pos) {
    kv_int_bits_check(bits, "prep_qkv_r_int_rows");
    if (T <= 0) return;
    if (H <= 0 || KVh <= 0 || D <= 0 || d_rel <= 0 || !pos) {
        std::fprintf(stderr, "ik2::k::prep_qkv_r_int_rows: H %d KVh %d D %d d_rel %d pos %p\n", H, KVh, D, d_rel,
                     (const void*) pos);
        std::abort();
    }
    if (bits == 4 && (D & 1)) {
        std::fprintf(stderr, "ik2::k::prep_qkv_r_int_rows: int4 needs even D, got %d\n", D);
        std::abort();
    }
    const int total = T * (2 * H + 2 * KVh) + (sink_w ? H : 0);
    if (bits == 8)
        prep_qkv_r_int_rows_kernel<8><<<(unsigned) total, 32, 0, s>>>(
            q, q_clip, k_in, kci_tab, ks_tab, v_in, vci_tab, vs_tab, r, r_clip, T, H, KVh, D, d_rel, qk_clip,
            rel_clip, eps, pos, alpha, n_floor, kv_cap, q16, sink_w, sink_x, sink_k, sink_y, tau_pos);
    else
        prep_qkv_r_int_rows_kernel<4><<<(unsigned) total, 32, 0, s>>>(
            q, q_clip, k_in, kci_tab, ks_tab, v_in, vci_tab, vs_tab, r, r_clip, T, H, KVh, D, d_rel, qk_clip,
            rel_clip, eps, pos, alpha, n_floor, kv_cap, q16, sink_w, sink_x, sink_k, sink_y, tau_pos);
    check("prep_qkv_r_int_rows");
}

template <int BITS>
static void attn_multi_int_impl(const float* q, const float* r, const float* sink, const uint8_t* kci,
                                const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int nq,
                                int H, int Hkv, int k0, int nk, int qpos0, float qk_scale, const float* proj,
                                int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    if (d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_multi_int: d_rel %d\n", d_rel);
        std::abort();
    }
    const int P = std::min(128, std::max(1, (nk + 63) / 64));
    float* part = scratch;
    if (mshared_on() && nq >= 1 && nq <= 8) {
        attn_multi_shared_int_dispatch<BITS>(q, r, kci, vci, ks, vs, kv_cap, nq, H, Hkv, k0, nk, qpos0, qk_scale,
                                             proj, d_rel, extent, window, part, nullptr, P, s);
        check("attn_multi_int_mshared_split");
    } else {
        attn_split_int_kernel<BITS><<<dim3(H, P, nq), 128, 0, s>>>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, k0, nk,
                                                                    qpos0, qk_scale, proj, d_rel, extent, window,
                                                                    part, nullptr);
        check("attn_multi_int_split");
    }
    attn_combine_kernel<<<dim3(H, nq), 128, 0, s>>>(sink, part, H, P, 1.0f, out);
    check("attn_multi_int_combine");
}

void attn_multi_int(int bits, const float* q, const float* r, const float* sink, const uint8_t* kci,
                    const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int nq, int H, int Hkv,
                    int k0, int nk, int qpos0, float qk_scale, const float* proj, int d_rel, int extent, int window,
                    float* scratch, float* out, cudaStream_t s) {
    kv_int_bits_check(bits, "attn_multi_int");
    if (bits == 8)
        attn_multi_int_impl<8>(q, r, sink, kci, vci, ks, vs, kv_cap, nq, H, Hkv, k0, nk, qpos0, qk_scale, proj,
                               d_rel, extent, window, scratch, out, s);
    else
        attn_multi_int_impl<4>(q, r, sink, kci, vci, ks, vs, kv_cap, nq, H, Hkv, k0, nk, qpos0, qk_scale, proj,
                               d_rel, extent, window, scratch, out, s);
}

template <int BITS>
static void attn_multi_int_g_impl(const float* q, const float* r, const float* sink, const uint8_t* kci,
                                  const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int nq,
                                  int H, int Hkv, const int* pos_d, int P, float qk_scale, const float* proj,
                                  int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    if (d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_multi_int_g: d_rel %d\n", d_rel);
        std::abort();
    }
    const int splits = P > 0 ? P : 128;
    float* part = scratch;
    if (mshared_on() && nq >= 1 && nq <= 8) {
        attn_multi_shared_int_dispatch<BITS>(q, r, kci, vci, ks, vs, kv_cap, nq, H, Hkv, 0, 0, 0, qk_scale, proj,
                                             d_rel, extent, window, part, pos_d, splits, s);
        check("attn_multi_int_g_mshared_split");
    } else {
        attn_split_int_kernel<BITS><<<dim3(H, (unsigned) splits, (unsigned) nq), 128, 0, s>>>(
            q, r, kci, vci, ks, vs, kv_cap, H, Hkv, 0, 0, 0, qk_scale, proj, d_rel, extent, window, part, pos_d);
        check("attn_multi_int_g_split");
    }
    attn_combine_kernel<<<dim3(H, (unsigned) nq), 128, 0, s>>>(sink, part, H, splits, 1.0f, out);
    check("attn_multi_int_g_combine");
}

void attn_multi_int_g(int bits, const float* q, const float* r, const float* sink, const uint8_t* kci,
                      const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int nq, int H,
                      int Hkv, const int* pos_d, int P, float qk_scale, const float* proj, int d_rel, int extent,
                      int window, float* scratch, float* out, cudaStream_t s) {
    kv_int_bits_check(bits, "attn_multi_int_g");
    if (bits == 8)
        attn_multi_int_g_impl<8>(q, r, sink, kci, vci, ks, vs, kv_cap, nq, H, Hkv, pos_d, P, qk_scale, proj, d_rel,
                                 extent, window, scratch, out, s);
    else
        attn_multi_int_g_impl<4>(q, r, sink, kci, vci, ks, vs, kv_cap, nq, H, Hkv, pos_d, P, qk_scale, proj, d_rel,
                                 extent, window, scratch, out, s);
}

template <int BITS>
static void attn_decode_int_impl(const float* q, const float* r, const float* sink, const uint8_t* kci,
                                 const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int H,
                                 int Hkv, int D, int k0, int nk, int qpos, float qk_scale, const float* proj,
                                 int d_rel, int extent, int window, float* scratch, float* out, cudaStream_t s) {
    if (D != 128 || d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_decode_int: D %d, d_rel %d\n", D, d_rel);
        std::abort();
    }
    const int P = std::min(128, std::max(1, (nk + 63) / 64));
    if (P == 1) {
        attn_decode_int_kernel<BITS><<<H, 128, 0, s>>>(q, r, sink, kci, vci, ks, vs, kv_cap, H, Hkv, k0, nk, qpos,
                                                      qk_scale, proj, d_rel, extent, window, out);
        check("attn_decode_int");
        return;
    }
    float* part = scratch;
    attn_split_int_kernel<BITS><<<dim3(H, P), 128, 0, s>>>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, k0, nk, qpos,
                                                          qk_scale, proj, d_rel, extent, window, part, nullptr);
    check("attn_decode_int_split");
    attn_combine_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, part, H, P, 1.0f, out);
    check("attn_decode_int_combine");
}

void attn_decode_int(int bits, const float* q, const float* r, const float* sink, const uint8_t* kci,
                     const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv, int D,
                     int k0, int nk, int qpos, float qk_scale, const float* proj, int d_rel, int extent, int window,
                     float* scratch, float* out, cudaStream_t s) {
    kv_int_bits_check(bits, "attn_decode_int");
    if (bits == 8)
        attn_decode_int_impl<8>(q, r, sink, kci, vci, ks, vs, kv_cap, H, Hkv, D, k0, nk, qpos, qk_scale, proj,
                                d_rel, extent, window, scratch, out, s);
    else
        attn_decode_int_impl<4>(q, r, sink, kci, vci, ks, vs, kv_cap, H, Hkv, D, k0, nk, qpos, qk_scale, proj,
                                d_rel, extent, window, scratch, out, s);
}

void attn_decode_int_wideload(int bits, const float* q, const float* r, const float* sink, const uint8_t* kci,
                              const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int H,
                              int Hkv, const int* pos, int P, float qk_scale, const float* proj, int d_rel,
                              int extent, int window, float* scratch, float* out, cudaStream_t s, const float* gate) {
    kv_int_bits_check(bits, "attn_decode_int_wideload");
    if (bits != 4 || H <= 0 || Hkv <= 0 || H % Hkv || d_rel > 32) {
        std::fprintf(stderr, "ik2::k::attn_decode_int_wideload: bits %d, H %d, Hkv %d, d_rel %d\n", bits, H, Hkv,
                     d_rel);
        std::abort();
    }
    constexpr int NQ = 4;                  // four query heads per wave32 warp, 16 heads per 128-thread block
    const unsigned gb = (unsigned) ((H + NQ * 4 - 1) / (NQ * 4));
    attn_wideload_int_kernel<4, NQ><<<dim3(gb, (unsigned) P), 128, 0, s>>>(
        q, r, kci, vci, ks, vs, kv_cap, H, Hkv, 0, 0, 0, qk_scale, proj, d_rel, extent, window, scratch, pos);
    check("attn_decode_int_wideload (split)");
    attn_combine_gate_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, scratch, H, P, 1.0f, gate, out);
    check("attn_decode_int_wideload (combine)");
}

template <int BITS, int DPP, int FA, int EX2>
static void attn_decode_int_at_launch(const float* q, const float* r, const uint8_t* kci, const uint8_t* vci,
                                      const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv,
                                      const int* pos, int P, float qk_scale, const float* proj, int d_rel, int extent,
                                      int window, float* scratch, cudaStream_t s) {
    attn_split_int_kernel<BITS, DPP, FA, EX2><<<dim3(H, (unsigned) P), 128, 0, s>>>(
        q, r, kci, vci, ks, vs, kv_cap, H, Hkv, 0, 0, 0, qk_scale, proj, d_rel, extent, window, scratch, pos);
}

void attn_decode_int_at(int bits, const float* q, const float* r, const float* sink, const uint8_t* kci,
                        const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv,
                        const int* pos, int P, float qk_scale, const float* proj, int d_rel, int extent, int window,
                        float* scratch, float* out, cudaStream_t s, const float* gate, int fa_split,
                        int rdna3_mask) {
    kv_int_bits_check(bits, "attn_decode_int_at");
    const int fa = fa_split > 0 ? 1 : (fa_split < 0 ? 0 : fa_split_on());
    const int m = fa_rdna3_effective(rdna3_mask);
    const int fa_eff = fa || (m & IK2_FA_RDNA3_WIDELOAD) ? 1 : 0;
    const int ex2 = (m & IK2_FA_RDNA3_EXP2) ? 1 : 0;
    if (m & IK2_FA_RDNA3_WAVE_P) P = fa_rdna3_wave_p(P, H);
    if (bits == 8 && ex2)
        attn_decode_int_at_launch<8, 0, 0, 1>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel,
                                              extent, window, scratch, s);
    else if (bits == 8)
        attn_decode_int_at_launch<8, 0, 0, 0>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel,
                                              extent, window, scratch, s);
    else if (fa_eff && ex2)
        attn_decode_int_at_launch<4, 0, 1, 1>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel,
                                              extent, window, scratch, s);
    else if (fa_eff)
        attn_decode_int_at_launch<4, 0, 1, 0>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel,
                                              extent, window, scratch, s);
    else if (ex2)
        attn_decode_int_at_launch<4, 0, 0, 1>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel,
                                              extent, window, scratch, s);
    else
        attn_decode_int_at_launch<4, 0, 0, 0>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel,
                                              extent, window, scratch, s);
    check("attn_decode_int_at (split)");
    attn_combine_gate_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, scratch, H, P, 1.0f, gate, out);
    check("attn_decode_int_at (combine)");
}

// IK2_ATTN_DPP opt-in wrapper: identical grid/CTA count, 2-byte/lane loads and online-softmax order as
// attn_decode_int_at; only the per-warp wsum becomes RDNA3 DPP/permlane (attn_flash_range_int<4,1>).
void attn_decode_int_at_dpp(int bits, const float* q, const float* r, const float* sink, const uint8_t* kci,
                            const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv,
                            const int* pos, int P, float qk_scale, const float* proj, int d_rel, int extent, int window,
                            float* scratch, float* out, cudaStream_t s, const float* gate, int rdna3_mask) {
    kv_int_bits_check(bits, "attn_decode_int_at_dpp");
    if (bits != 4) {
        std::fprintf(stderr, "ik2::k::attn_decode_int_at_dpp: bits %d (only 4 supported)\n", bits);
        std::abort();
    }
    const int m = fa_rdna3_effective(rdna3_mask);
    const int fa_eff = (m & IK2_FA_RDNA3_WIDELOAD) ? 1 : 0;
    const int ex2 = (m & IK2_FA_RDNA3_EXP2) ? 1 : 0;
    if (m & IK2_FA_RDNA3_WAVE_P) P = fa_rdna3_wave_p(P, H);
    if (fa_eff && ex2)
        attn_decode_int_at_launch<4, 1, 1, 1>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel,
                                              extent, window, scratch, s);
    else if (fa_eff)
        attn_decode_int_at_launch<4, 1, 1, 0>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel,
                                              extent, window, scratch, s);
    else if (ex2)
        attn_decode_int_at_launch<4, 1, 0, 1>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel,
                                              extent, window, scratch, s);
    else
        attn_decode_int_at_launch<4, 1, 0, 0>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, pos, P, qk_scale, proj, d_rel,
                                              extent, window, scratch, s);
    check("attn_decode_int_at_dpp (split)");
    attn_combine_gate_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, scratch, H, P, 1.0f, gate, out);
    check("attn_decode_int_at_dpp (combine)");
}

// IK2_ATTN_GQA opt-in wrappers: grid (Hkv, P, 1) so R = 3 query heads share one KV head's K/V rows.  Same
// [H,P,130] partials, hence the same combine kernels as attn_decode_int_at / attn_decode_int.  dpp selects
// the bit-exact DPP warp sum (int4 only, matching IK2_ATTN_DPP).
template <int BITS, int DPP, int FA = 0, int PV = 0, int EX2 = 0>
static void attn_decode_int_gqa_launch(const float* q, const float* r, const uint8_t* kci, const uint8_t* vci,
                                       const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv, int k0,
                                       int nk, int qpos, float qk_scale, const float* proj, int d_rel, int extent,
                                       int window, float* scratch, const int* pos_d, int P, cudaStream_t s) {
    constexpr int R = 3;
    attn_split_int_gqa_kernel<BITS, DPP, R, FA, PV, EX2><<<dim3(Hkv, (unsigned) P, 1), 128, 0, s>>>(
        q, r, kci, vci, ks, vs, kv_cap, H, Hkv, k0, nk, qpos, qk_scale, proj, d_rel, extent, window, scratch, pos_d);
}

#define IK2_GQA_LAUNCH_AT(B_, D_, F_, PV_, E_)                                                                      \
    attn_decode_int_gqa_launch<B_, D_, F_, PV_, E_>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, 0, 0, 0, qk_scale, proj, \
                                                    d_rel, extent, window, scratch, pos, P, s)
#define IK2_GQA_LAUNCH_HOST(B_, D_, F_, PV_, E_)                                                                   \
    attn_decode_int_gqa_launch<B_, D_, F_, PV_, E_>(q, r, kci, vci, ks, vs, kv_cap, H, Hkv, k0, nk, qpos, qk_scale, \
                                                    proj, d_rel, extent, window, scratch, nullptr, P, s)

void attn_decode_int_at_gqa(int bits, const float* q, const float* r, const float* sink, const uint8_t* kci,
                            const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv,
                            const int* pos, int P, float qk_scale, const float* proj, int d_rel, int extent, int window,
                            float* scratch, float* out, cudaStream_t s, const float* gate, int dpp, int fa_split,
                            int rdna3_mask) {
    kv_int_bits_check(bits, "attn_decode_int_at_gqa");
    if (H <= 0 || Hkv <= 0 || H % Hkv || H / Hkv != 3) {
        std::fprintf(stderr, "ik2::k::attn_decode_int_at_gqa: H %d, Hkv %d (only R = 3)\n", H, Hkv);
        std::abort();
    }
    const int fa = fa_split > 0 ? 1 : (fa_split < 0 ? 0 : fa_split_on());
    const int m = fa_rdna3_effective(rdna3_mask);
    const int fa_eff = fa || (m & IK2_FA_RDNA3_WIDELOAD) ? 1 : 0;
    const int pv = (m & IK2_FA_RDNA3_PRE_LOAD_V) ? 1 : 0;
    const int ex2 = (m & IK2_FA_RDNA3_EXP2) ? 1 : 0;
    if (m & IK2_FA_RDNA3_WAVE_P) P = fa_rdna3_wave_p(P, Hkv);
    if (bits == 8 && pv && ex2)
        IK2_GQA_LAUNCH_AT(8, 0, 0, 1, 1);
    else if (bits == 8 && pv)
        IK2_GQA_LAUNCH_AT(8, 0, 0, 1, 0);
    else if (bits == 8 && ex2)
        IK2_GQA_LAUNCH_AT(8, 0, 0, 0, 1);
    else if (bits == 8)
        IK2_GQA_LAUNCH_AT(8, 0, 0, 0, 0);
    else if (dpp && fa_eff && pv && ex2)
        IK2_GQA_LAUNCH_AT(4, 1, 1, 1, 1);
    else if (dpp && fa_eff && pv)
        IK2_GQA_LAUNCH_AT(4, 1, 1, 1, 0);
    else if (dpp && fa_eff && ex2)
        IK2_GQA_LAUNCH_AT(4, 1, 1, 0, 1);
    else if (dpp && fa_eff)
        IK2_GQA_LAUNCH_AT(4, 1, 1, 0, 0);
    else if (dpp && pv && ex2)
        IK2_GQA_LAUNCH_AT(4, 1, 0, 1, 1);
    else if (dpp && pv)
        IK2_GQA_LAUNCH_AT(4, 1, 0, 1, 0);
    else if (dpp && ex2)
        IK2_GQA_LAUNCH_AT(4, 1, 0, 0, 1);
    else if (dpp)
        IK2_GQA_LAUNCH_AT(4, 1, 0, 0, 0);
    else if (fa_eff && pv && ex2)
        IK2_GQA_LAUNCH_AT(4, 0, 1, 1, 1);
    else if (fa_eff && pv)
        IK2_GQA_LAUNCH_AT(4, 0, 1, 1, 0);
    else if (fa_eff && ex2)
        IK2_GQA_LAUNCH_AT(4, 0, 1, 0, 1);
    else if (fa_eff)
        IK2_GQA_LAUNCH_AT(4, 0, 1, 0, 0);
    else if (pv && ex2)
        IK2_GQA_LAUNCH_AT(4, 0, 0, 1, 1);
    else if (pv)
        IK2_GQA_LAUNCH_AT(4, 0, 0, 1, 0);
    else if (ex2)
        IK2_GQA_LAUNCH_AT(4, 0, 0, 0, 1);
    else
        IK2_GQA_LAUNCH_AT(4, 0, 0, 0, 0);
    check("attn_decode_int_at_gqa (split)");
    attn_combine_gate_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, scratch, H, P, 1.0f, gate, out);
    check("attn_decode_int_at_gqa (combine)");
}

void attn_decode_int_gqa(int bits, const float* q, const float* r, const float* sink, const uint8_t* kci,
                         const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv, int D,
                         int k0, int nk, int qpos, float qk_scale, const float* proj, int d_rel, int extent, int window,
                         float* scratch, float* out, cudaStream_t s, int dpp, int fa_split, int rdna3_mask) {
    kv_int_bits_check(bits, "attn_decode_int_gqa");
    if (D != 128 || H <= 0 || Hkv <= 0 || H % Hkv || H / Hkv != 3) {
        std::fprintf(stderr, "ik2::k::attn_decode_int_gqa: D %d, H %d, Hkv %d (only R = 3)\n", D, H, Hkv);
        std::abort();
    }
    const int fa = fa_split > 0 ? 1 : (fa_split < 0 ? 0 : fa_split_on());
    const int m = fa_rdna3_effective(rdna3_mask);
    const int fa_eff = fa || (m & IK2_FA_RDNA3_WIDELOAD) ? 1 : 0;
    const int pv = (m & IK2_FA_RDNA3_PRE_LOAD_V) ? 1 : 0;
    const int ex2 = (m & IK2_FA_RDNA3_EXP2) ? 1 : 0;
    int P = std::min(128, std::max(1, (nk + 63) / 64));
    if (m & IK2_FA_RDNA3_WAVE_P) P = fa_rdna3_wave_p(P, Hkv);
    if (bits == 8 && pv && ex2)
        IK2_GQA_LAUNCH_HOST(8, 0, 0, 1, 1);
    else if (bits == 8 && pv)
        IK2_GQA_LAUNCH_HOST(8, 0, 0, 1, 0);
    else if (bits == 8 && ex2)
        IK2_GQA_LAUNCH_HOST(8, 0, 0, 0, 1);
    else if (bits == 8)
        IK2_GQA_LAUNCH_HOST(8, 0, 0, 0, 0);
    else if (dpp && fa_eff && pv && ex2)
        IK2_GQA_LAUNCH_HOST(4, 1, 1, 1, 1);
    else if (dpp && fa_eff && pv)
        IK2_GQA_LAUNCH_HOST(4, 1, 1, 1, 0);
    else if (dpp && fa_eff && ex2)
        IK2_GQA_LAUNCH_HOST(4, 1, 1, 0, 1);
    else if (dpp && fa_eff)
        IK2_GQA_LAUNCH_HOST(4, 1, 1, 0, 0);
    else if (dpp && pv && ex2)
        IK2_GQA_LAUNCH_HOST(4, 1, 0, 1, 1);
    else if (dpp && pv)
        IK2_GQA_LAUNCH_HOST(4, 1, 0, 1, 0);
    else if (dpp && ex2)
        IK2_GQA_LAUNCH_HOST(4, 1, 0, 0, 1);
    else if (dpp)
        IK2_GQA_LAUNCH_HOST(4, 1, 0, 0, 0);
    else if (fa_eff && pv && ex2)
        IK2_GQA_LAUNCH_HOST(4, 0, 1, 1, 1);
    else if (fa_eff && pv)
        IK2_GQA_LAUNCH_HOST(4, 0, 1, 1, 0);
    else if (fa_eff && ex2)
        IK2_GQA_LAUNCH_HOST(4, 0, 1, 0, 1);
    else if (fa_eff)
        IK2_GQA_LAUNCH_HOST(4, 0, 1, 0, 0);
    else if (pv && ex2)
        IK2_GQA_LAUNCH_HOST(4, 0, 0, 1, 1);
    else if (pv)
        IK2_GQA_LAUNCH_HOST(4, 0, 0, 1, 0);
    else if (ex2)
        IK2_GQA_LAUNCH_HOST(4, 0, 0, 0, 1);
    else
        IK2_GQA_LAUNCH_HOST(4, 0, 0, 0, 0);
    check("attn_decode_int_gqa (split)");
    attn_combine_kernel<<<dim3(H, 1), 128, 0, s>>>(sink, scratch, H, P, 1.0f, out);
    check("attn_decode_int_gqa (combine)");
}

#undef IK2_GQA_LAUNCH_AT
#undef IK2_GQA_LAUNCH_HOST

void attn_decode_int_onepass(int bits, const float* q, const float* r, const float* sink, const uint8_t* kci,
                             const uint8_t* vci, const uint16_t* ks, const uint16_t* vs, int kv_cap, int H, int Hkv,
                             const int* pos, float qk_scale, const float* proj, int d_rel, int extent, int window,
                             float* out, cudaStream_t s, const float* gate) {
    kv_int_bits_check(bits, "attn_decode_int_onepass");
    if (bits == 8)
        attn_decode_onepass_int_kernel<8><<<H, 128, 0, s>>>(q, r, sink, kci, vci, ks, vs, kv_cap, H, Hkv, pos,
                                                           qk_scale, proj, d_rel, extent, window, gate, out);
    else
        attn_decode_onepass_int_kernel<4><<<H, 128, 0, s>>>(q, r, sink, kci, vci, ks, vs, kv_cap, H, Hkv, pos,
                                                           qk_scale, proj, d_rel, extent, window, gate, out);
    check("attn_decode_int_onepass");
}

void unpack_kv_int_to_f16(int bits, const uint8_t* kci, const uint8_t* vci, const uint16_t* ks, const uint16_t* vs,
                          int kv_cap, int k0, int nk, int KD, uint16_t* k_out16, uint16_t* v_out16,
                          cudaStream_t s) {
    kv_int_bits_check(bits, "unpack_kv_int_to_f16");
    if (nk <= 0 || KD <= 0) return;
    const uintptr_t cal = (bits == 8) ? (uintptr_t) 3 : (uintptr_t) 1;
    if ((KD & 3) == 0 && (!k_out16 || ((((uintptr_t) kci & cal) == 0) && (((uintptr_t) k_out16 & 7) == 0))) &&
        (!v_out16 || ((((uintptr_t) vci & cal) == 0) && (((uintptr_t) v_out16 & 7) == 0)))) {
        const int kd4 = KD >> 2;
        const int bx = std::min(256, kd4);
        const dim3 g((unsigned) ((kd4 + bx - 1) / bx), (unsigned) std::min(nk, 65535));
        if (bits == 8)
            unpack_kv_int_to_f16_vec4_kernel<8><<<g, (unsigned) bx, 0, s>>>(kci, vci, ks, vs, kv_cap, k0, nk, KD,
                                                                           k_out16, v_out16);
        else
            unpack_kv_int_to_f16_vec4_kernel<4><<<g, (unsigned) bx, 0, s>>>(kci, vci, ks, vs, kv_cap, k0, nk, KD,
                                                                           k_out16, v_out16);
        check("unpack_kv_int_to_f16");
        return;
    }
    const size_t total = (size_t) nk * KD;
    const unsigned blocks = (unsigned) std::min<size_t>((total + kBlk - 1) / kBlk, 65535u);
    if (bits == 8)
        unpack_kv_int_to_f16_kernel<8><<<blocks, kBlk, 0, s>>>(kci, vci, ks, vs, kv_cap, k0, nk, KD, k_out16,
                                                              v_out16);
    else
        unpack_kv_int_to_f16_kernel<4><<<blocks, kBlk, 0, s>>>(kci, vci, ks, vs, kv_cap, k0, nk, KD, k_out16,
                                                              v_out16);
    check("unpack_kv_int_to_f16");
}

void decode_f16(int type, const uint8_t* blocks, size_t n, uint16_t* out, cudaStream_t s) {
    if (!n) return;
    if (type == kNVFP4 && n % 16 == 0 && (((uintptr_t) blocks & 3) == 0) && (((uintptr_t) out & 15) == 0)) {
        if (n % 64 == 0 && (((uintptr_t) blocks & 3) == 0)) {
            decode_f16_nvfp4_x4_kernel<<<grid(n / 64), kBlk, 0, s>>>(blocks, n / 64, out);
            check("decode_f16_nvfp4_x4");
            return;
        }
        decode_f16_nvfp4_kernel<<<grid(n / 16), kBlk, 0, s>>>(blocks, n / 16, out);
        check("decode_f16_nvfp4");
        return;
    }
    if (type == kQ8_0 && n % 32 == 0) {
        decode_f16_q8_kernel<<<grid(n / 32, kBlk / 8), kBlk, 0, s>>>(blocks, n / 32, out);
        check("decode_f16_q8");
        return;
    }
    if ((type == kIQ3_XXS || type == kIQ4_XS) && n % 256 == 0 && (((uintptr_t) out & 3) == 0) &&
        (((uintptr_t) blocks & (type == kIQ4_XS ? 3 : 1)) == 0)) {
        decode_f16_iq_kernel<<<grid(n / 32), kBlk, 0, s>>>(type, blocks, n / 32, out);
        check("decode_f16_iq");
        return;
    }
    if ((type == kQ5_K || type == kQ6_K) && n % 256 == 0) {
        decode_f16_k_kernel<<<grid(n / 256, kBlk / 32), kBlk, 0, s>>>(type, blocks, n / 256, out);
        check("decode_f16_k");
        return;
    }
    decode_f16_kernel<<<grid(n), kBlk, 0, s>>>(type, blocks, n, out);
    check("decode_f16");
}
void decode_f32(int type, const uint8_t* blocks, size_t n, float* out, cudaStream_t s) {
    if (!n) return;
    decode_f32_kernel<<<grid(n), kBlk, 0, s>>>(type, blocks, n, out);
    check("decode_f32");
}
void embed_gather_f32(int type, const uint8_t* table, int n_rows, size_t row_bytes, int C, const int* tok,
                      int n_tokens, float* out, cudaStream_t s) {
    if (!n_tokens || C <= 0) return;
    dim3 g((unsigned) ((C + kBlk - 1) / kBlk), (unsigned) n_tokens);
    embed_gather_f32_kernel<<<g, kBlk, 0, s>>>(type, table, n_rows, row_bytes, C, tok, n_tokens, out);
    check("embed_gather_f32");
}
void pos_inc(int* pos, cudaStream_t s) {
    pos_inc_kernel<<<1, 1, 0, s>>>(pos);
    check("pos_inc");
}
void decode_host(int type, const uint8_t* blocks, size_t n, float* out) {
    for (size_t i = 0; i < n; ++i) out[i] = decode_value(type, blocks, i);
}

// ---- INT4/INT8 WMMA path, ported from llama.cpp-rdna3-int4 ggml/src/ggml-cuda/int4-wmma.cu ----
// Symmetric INT4 weights (Hadamard-rotated per 256-block along K, repacked to K-slices) x INT8
// activations on gfx1100 v_wmma_i32_16x16x16_iu4. The math (rotation, sub-scales, fragment order)
// is copied verbatim from that kernel; only the ggml_tensor glue is replaced by raw pointers.
#if defined(STRATA_USE_HIP) && defined(STRATA_WMMA_GFX11)
namespace int4w {

typedef int int8v __attribute__((ext_vector_type(8)));
typedef int int4v __attribute__((ext_vector_type(4)));
typedef int int2v __attribute__((ext_vector_type(2)));
typedef float float8v __attribute__((ext_vector_type(8)));
typedef _Float16 half16v __attribute__((ext_vector_type(16)));

constexpr int MI = 2, NJ = 2, WAVES_N = 8;
constexpr int TILE_M = MI * 16;
constexpr int TILE_N = WAVES_N * NJ * 16;

#define I4_RSRC(p) __builtin_amdgcn_make_buffer_rsrc((void *)(p), 0, 0x7fffffff, 0x31004000)
__device__ __forceinline__ int4v bl128(__amdgpu_buffer_rsrc_t r, int voff, int soff) { return __builtin_amdgcn_raw_buffer_load_b128(r, voff, soff, 0); }
__device__ __forceinline__ int2v bl64(__amdgpu_buffer_rsrc_t r, int voff, int soff) { return __builtin_amdgcn_raw_buffer_load_b64(r, voff, soff, 0); }
__device__ __forceinline__ uint4 nt128(const uint8_t * p, size_t offset) {
    const int4v v = __builtin_amdgcn_raw_buffer_load_b128(I4_RSRC(p), (int) offset, 0, 6);
    return make_uint4(v.x, v.y, v.z, v.w);
}
// Cached variant (aux=0): lets duplicate-expert weight lines survive in L2 (grouped verify).
__device__ __forceinline__ uint4 cl128(const uint8_t * p, size_t offset) {
    const int4v v = __builtin_amdgcn_raw_buffer_load_b128(I4_RSRC(p), (int) offset, 0, 0);
    return make_uint4(v.x, v.y, v.z, v.w);
}
#define I4_LO(v) __builtin_shufflevector((v), (v), 0, 1)
#define I4_HI(v) __builtin_shufflevector((v), (v), 2, 3)
__device__ __forceinline__ void mul24a(int & out, int a, int b) { asm volatile("v_mul_i32_i24 %0, %1, %2" : "=v"(out) : "v"(a), "v"(b)); }
__device__ __forceinline__ void mad24(int & acc, int a, int b) { asm volatile("v_mad_i32_i24 %0, %1, %2, %0" : "+v"(acc) : "v"(a), "v"(b)); }

__host__ __device__ constexpr int P(int s) {
    return (s < 8) ? ((s & 1) ? 4 + (s >> 1) : (s >> 1)) : 8 + ((s - 8) & 1 ? 4 + ((s - 8) >> 1) : ((s - 8) >> 1));
}
__device__ __forceinline__ float sign256(int j) { return (((uint32_t) j * 2654435761u) >> 31) ? -1.0f : 1.0f; }
__device__ __forceinline__ _Float16 h16(int r, int c) { return (__builtin_popcount(r & c) & 1) ? (_Float16) -1.0f : (_Float16) 1.0f; }

template <int CTRL> __device__ __forceinline__ float dppx(float x) { return __int_as_float(__builtin_amdgcn_update_dpp(0, __float_as_int(x), CTRL, 0xF, 0xF, false)); }
__device__ __forceinline__ uint32_t swap_rows_u(uint32_t x) { return (uint32_t) __builtin_amdgcn_permlanex16(0, (int) x, 0x76543210u, 0xFEDCBA98u, false, false); }
__device__ __forceinline__ float swap_rows(float x) { return __uint_as_float(swap_rows_u(__float_as_uint(x))); }
__device__ __forceinline__ float wave32_max(float x) {
    x = fmaxf(x, dppx<0xB1>(x)); x = fmaxf(x, dppx<0x4E>(x)); x = fmaxf(x, dppx<0x141>(x)); x = fmaxf(x, dppx<0x140>(x));
    return fmaxf(x, swap_rows(x));
}
__device__ __forceinline__ float wave32_sum(float x) {
    x += dppx<0xB1>(x); x += dppx<0x4E>(x); x += dppx<0x141>(x); x += dppx<0x140>(x);
    return x + swap_rows(x);
}
__device__ __forceinline__ float group8_sum(float x) {
    x += dppx<0xB1>(x); x += dppx<0x4E>(x); x += dppx<0x141>(x);
    return x;
}

// H256 = H16 (x) H16 on a 16x16 view of sign*x via two f16 WMMAs; lane (h,lc) elem i = y[16*(2i+h)+lc]*16
__device__ __forceinline__ float8v rotate_regs(const float4 (&v)[4], int lane, const half16v & b1, const half16v & a2) {
    const int h = lane >> 4, lc = lane & 15;
    half16v a1;
#pragma unroll
    for (int q = 0; q < 4; q++) {
        a1[4 * q + 0] = (_Float16) (v[q].x * sign256(lc * 16 + 4 * q + 0)); a1[4 * q + 1] = (_Float16) (v[q].y * sign256(lc * 16 + 4 * q + 1));
        a1[4 * q + 2] = (_Float16) (v[q].z * sign256(lc * 16 + 4 * q + 2)); a1[4 * q + 3] = (_Float16) (v[q].w * sign256(lc * 16 + 4 * q + 3));
    }
    float8v z = {0, 0, 0, 0, 0, 0, 0, 0};
    const float8v p = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a1, b1, z);
    uint32_t own[4], oth[4];
#pragma unroll
    for (int i = 0; i < 4; i++) {
        _Float16 lo = (_Float16) p[2 * i], hi = (_Float16) p[2 * i + 1];
        const uint32_t w = (uint32_t) __builtin_bit_cast(uint16_t, lo) | ((uint32_t) __builtin_bit_cast(uint16_t, hi) << 16);
        own[i] = w; oth[i] = swap_rows_u(w);
    }
    half16v b2;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t ev = h ? oth[i] : own[i], od = h ? own[i] : oth[i];
        b2[2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (ev & 0xFFFF)); b2[2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (ev >> 16));
        b2[8 + 2 * i] = __builtin_bit_cast(_Float16, (uint16_t) (od & 0xFFFF)); b2[8 + 2 * i + 1] = __builtin_bit_cast(_Float16, (uint16_t) (od >> 16));
    }
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a2, b2, z);
}
// plain rotation (no GLU): read the 16 values of this lane's row, rotate
__device__ __forceinline__ float8v rotate_block(const float * __restrict__ xrow, int lane, const half16v & b1, const half16v & a2) {
    const int lc = lane & 15;
    float4 v[4];
    const float4 * xp = (const float4 *) (xrow + lc * 16);
#pragma unroll
    for (int q = 0; q < 4; q++) { v[q] = xp[q]; }
    return rotate_regs(v, lane, b1, a2);
}

// 8-bit quantization of one rotated 256-block (this lane's 8 values)
__device__ __forceinline__ void quant_rot8(const float8v & w, int lane, int8_t * qdst, float2 * ddst) {
    const int h = lane >> 4, lc = lane & 15;
    float x[8], am[8], d[8], sq[8]; int q[8];
#pragma unroll
    for (int i = 0; i < 8; i++) { x[i] = w[i] * 0.0625f; am[i] = fabsf(x[i]); }
#pragma unroll
    for (int i = 0; i < 8; i++) am[i] = fmaxf(am[i], dppx<0xB1>(am[i]));
#pragma unroll
    for (int i = 0; i < 8; i++) am[i] = fmaxf(am[i], dppx<0x4E>(am[i]));
#pragma unroll
    for (int i = 0; i < 8; i++) am[i] = fmaxf(am[i], dppx<0x141>(am[i]));
#pragma unroll
    for (int i = 0; i < 8; i++) am[i] = fmaxf(am[i], dppx<0x140>(am[i]));
#pragma unroll
    for (int i = 0; i < 8; i++) am[i] = fmaxf(am[i], swap_rows(am[i]));
#pragma unroll
    for (int i = 0; i < 8; i++) { d[i] = am[i] / 127.0f; q[i] = am[i] == 0.0f ? 0 : (int) roundf(x[i] / d[i]); sq[i] = (float) q[i]; }
#pragma unroll
    for (int i = 0; i < 8; i++) sq[i] += dppx<0xB1>(sq[i]);
#pragma unroll
    for (int i = 0; i < 8; i++) sq[i] += dppx<0x4E>(sq[i]);
#pragma unroll
    for (int i = 0; i < 8; i++) sq[i] += dppx<0x141>(sq[i]);
#pragma unroll
    for (int i = 0; i < 8; i++) sq[i] += dppx<0x140>(sq[i]);
#pragma unroll
    for (int i = 0; i < 8; i++) sq[i] = sq[i] + swap_rows(sq[i]);
#pragma unroll
    for (int i = 0; i < 8; i++) qdst[i * 32 + 16 * h + lc] = (int8_t) q[i];
    if (lane == 0) {
#pragma unroll
        for (int i = 0; i < 8; i++) ddst[i] = make_float2(d[i], d[i] * sq[i]);
    }
}

// fused rotate + 8-bit quantize + pack: one 256-thread block per token
__global__ __launch_bounds__(256) void k_rot_pack8(const float * __restrict__ X, int K, uint8_t * __restrict__ Xhp, uint8_t * __restrict__ Xlp, uint8_t * __restrict__ XMp, float * __restrict__ XRp) {
    extern __shared__ __align__(16) unsigned char fsm[];
    __half * xr = (__half *) fsm;
    float * amx = (float *) (fsm + (size_t) K * 2);
    const int nsb = K / 256, n = blockIdx.x, tid = threadIdx.x, lane = tid & 31, wave = tid >> 5;
    const int h = lane >> 4, lc = lane & 15, tt = n >> 4, tl = n & 15;
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, lc); a2[k] = h16(lc, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    for (int sb = wave; sb < nsb; sb += 8) {
        const float8v w = rotate_block(X + (size_t) n * K + sb * 256, lane, b1, a2);
        float mx = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const __half hv = __float2half(w[i] * 0.0625f);
            xr[sb * 256 + 16 * (2 * i + h) + lc] = hv;
            mx = fmaxf(mx, fabsf(__half2float(hv)));
        }
        mx = wave32_max(mx);
        if (lane == 0) amx[sb] = mx;
    }
    __syncthreads();
    float dref = 0.f; for (int i = 0; i < nsb; i++) dref = fmaxf(dref, amx[i]); dref /= 127.f;
    for (int idx = tid; idx < nsb * 8; idx += 256) {
        const int sb = idx >> 3, g = idx & 7;
        int mm = 1;
        if (dref > 0.f) mm = max(1, min(16, (int) ceilf(16.f * (amx[sb] / 127.f) / dref - 1e-4f)));
        const float dsb = dref * mm / 16.f, inv = dsb > 0.f ? 1.f / dsb : 0.f;
        int q[32];
#pragma unroll
        for (int i = 0; i < 32; i++) q[i] = max(-127, min(127, (int) roundf(__half2float(xr[sb * 256 + g * 32 + i]) * inv)));
        uint32_t wh[4], wl[4];
#pragma unroll
        for (int sl = 0; sl < 2; sl++)
#pragma unroll
            for (int dw = 0; dw < 2; dw++) {
                uint32_t hw = 0, lw = 0;
#pragma unroll
                for (int s2 = 0; s2 < 8; s2++) {
                    const int v = q[16 * sl + P(dw * 8 + s2)];
                    hw |= (uint32_t) ((v >> 4) & 15) << (4 * s2);
                    lw |= (uint32_t) (v & 15) << (4 * s2);
                }
                wh[sl * 2 + dw] = hw; wl[sl * 2 + dw] = lw;
            }
        const size_t o = ((size_t) ((tt * nsb + sb) * 8 + g)) * 256 + tl * 16;
        *(uint4 *) (Xhp + o) = make_uint4(wh[0], wh[1], wh[2], wh[3]);
        *(uint4 *) (Xlp + o) = make_uint4(wl[0], wl[1], wl[2], wl[3]);
        if (g == 0) XMp[(size_t) (tt * nsb + sb) * 16 + (tl & 1) * 8 + (tl >> 1)] = (uint8_t) mm;
    }
    if (tid == 0) XRp[(size_t) tt * 16 + (tl & 1) * 8 + (tl >> 1)] = dref;
}

// 8-bit activation GEMM, GW scale groups chained on one accumulator
template <int GW, int WN>
__global__ __launch_bounds__(WN * 32) void k_gemm8w(const uint8_t * __restrict__ Wp, const uint8_t * __restrict__ Wh, const float * __restrict__ WS,
        const uint8_t * __restrict__ Xhp, const uint8_t * __restrict__ Xlp, const uint8_t * __restrict__ XMp, const float * __restrict__ XRp, float * __restrict__ out, int M, int N, int K, const float * res, int sh, int ldy) {
    const int lane = threadIdx.x & 31;
    const int wn = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int h = lane >> 4, lc = lane & 15;
    const int nsb = K / 256;
    const int ft0 = blockIdx.x * MI;
    const int tt0 = (blockIdx.y * WN + wn) * NJ;
    const __amdgpu_buffer_rsrc_t rWp = I4_RSRC(Wp), rWh = I4_RSRC(Wh), rXh = I4_RSRC(Xhp), rXl = I4_RSRC(Xlp), rXM = I4_RSRC(XMp);
    float tot[MI][NJ][8];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NJ; b++)
#pragma unroll
            for (int i = 0; i < 8; i++) tot[a][b][i] = 0.f;
    for (int sb = 0; sb < nsb; sb++) {
        uint32_t scw[MI][1];
#pragma unroll
        for (int mi = 0; mi < MI; mi++) scw[mi][0] = *(const uint32_t *) (Wh + (((ft0 + mi) * nsb + sb) * 16 + lc) * 4);
        int2v mmw[NJ];
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) mmw[nj] = bl64(rXM, h * 8, ((tt0 + nj) * nsb + sb) * 16);
        int isum[MI][NJ][8];
        for (int u0 = 0; u0 < 8; u0 += GW) {
            int8v ci[MI][NJ];
            const int8v z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                const __amdgpu_buffer_rsrc_t rX = pass ? rXl : rXh;
                int4v wc[MI], xc[NJ], wn_[MI], xn[NJ];
#pragma unroll
                for (int mi = 0; mi < MI; mi++) wc[mi] = bl128(rWp, lc * 16, (((ft0 + mi) * nsb + sb) * 8 + u0) * 256);
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) xc[nj] = bl128(rX, lc * 16, (((tt0 + nj) * nsb + sb) * 8 + u0) * 256);
#pragma unroll
                for (int q = 0; q < GW; q++) {
                    if (q + 1 < GW) {
#pragma unroll
                        for (int mi = 0; mi < MI; mi++) wn_[mi] = bl128(rWp, lc * 16, (((ft0 + mi) * nsb + sb) * 8 + u0 + q + 1) * 256);
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) xn[nj] = bl128(rX, lc * 16, (((tt0 + nj) * nsb + sb) * 8 + u0 + q + 1) * 256);
                    }
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for (int sl = 0; sl < 2; sl++)
#pragma unroll
                        for (int mi = 0; mi < MI; mi++)
#pragma unroll
                            for (int nj = 0; nj < NJ; nj++) {
                                const int2v xa = sl ? I4_HI(xc[nj]) : I4_LO(xc[nj]), wb = sl ? I4_HI(wc[mi]) : I4_LO(wc[mi]);
                                const int8v cin = (q == 0 && sl == 0 && pass == 0) ? z : ci[mi][nj];
                                if (pass == 0) ci[mi][nj] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, xa, true, wb, cin, false);
                                else           ci[mi][nj] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, xa, true, wb, cin, false);
                            }
                    if (q + 1 < GW) {
#pragma unroll
                        for (int mi = 0; mi < MI; mi++) wc[mi] = wn_[mi];
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) xc[nj] = xn[nj];
                    }
                }
                if (pass == 0) {
#pragma unroll
                    for (int mi = 0; mi < MI; mi++)
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) ci[mi][nj] = ci[mi][nj] << 4;
                }
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const int scg = (int) ((scw[mi][0] >> (4 * u0)) & 0xF);
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int i = 0; i < 8; i++) { if (u0 == 0) mul24a(isum[mi][nj][i], scg, ci[mi][nj][i]); else mad24(isum[mi][nj][i], scg, ci[mi][nj][i]); }
            }
        }
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) {
            float mv[8];
#pragma unroll
            for (int i = 0; i < 8; i++) mv[i] = (float) (int) (((uint32_t) mmw[nj][i >> 2] >> (8 * (i & 3))) & 0xFF);
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int i = 0; i < 8; i++) tot[mi][nj][i] = fmaf((float) isum[mi][nj][i], mv[i], tot[mi][nj][i]);
        }
    }
    const float inv = 1.f / (float) ((1 << sh) * 16);
#pragma unroll
    for (int nj = 0; nj < NJ; nj++) {
        float dr[8];
        { const float4 a = *(const float4 *) (XRp + (size_t) (tt0 + nj) * 16 + h * 8), b = *(const float4 *) (XRp + (size_t) (tt0 + nj) * 16 + h * 8 + 4);
          dr[0] = a.x; dr[1] = a.y; dr[2] = a.z; dr[3] = a.w; dr[4] = b.x; dr[5] = b.y; dr[6] = b.z; dr[7] = b.w; }
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            const int feat = (ft0 + mi) * 16 + lc;
            const float sf = WS[feat] * inv;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int tok = (tt0 + nj) * 16 + 2 * i + h;
                if (tok < N) { const size_t oi = (size_t) tok * ldy + feat; out[oi] = tot[mi][nj][i] * (sf * dr[i]) + (res ? res[oi] : 0.f); }
            }
        }
    }
}

// IK2_BATCH_MOE_DEVICE: k_gemm8w's exact arithmetic, but the expert is a BLOCK dimension and rows
// belonging to other experts are masked.  This lets one device launch replace the host's per-expert
// loop without a D2H sync: blockIdx.z selects the expert, row_expert[] masks each token row.  For
// every written row the packed activation, lane mapping and accumulator order are identical to
// calling int4_gemm8(W_e, ..., N=cnt[e]) from the general MoE path.
template <int GW, int WN>
__global__ __launch_bounds__(WN * 32) void k_gemm8w_moe_masked(const uint8_t* __restrict__ Wp,
        const uint8_t* __restrict__ Wh, const float* __restrict__ WS,
        size_t wp_stride, size_t wh_stride, int ws_stride, int E, const int* __restrict__ row_expert,
        const uint8_t* __restrict__ Xhp, const uint8_t* __restrict__ Xlp, const uint8_t* __restrict__ XMp,
        const float* __restrict__ XRp, float* __restrict__ out, int M, int N, int K, const float* res,
        int sh, int ldy, const int* __restrict__ active_expert, const int* __restrict__ active_count) {
    const int lane = threadIdx.x & 31;
    const int wn = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int h = lane >> 4, lc = lane & 15;
    const int nsb = K / 256;
    const int ft0 = blockIdx.x * MI;
    const int tt0 = (blockIdx.y * WN + wn) * NJ;
    if (active_expert && blockIdx.z >= *active_count) return;
    const int e = active_expert ? active_expert[blockIdx.z] : blockIdx.z;
    const uint8_t* Wp_e = Wp + (size_t) e * wp_stride;
    const uint8_t* Wh_e = Wh + (size_t) e * wh_stride;
    const float* WS_e = WS + (size_t) e * ws_stride;
    const __amdgpu_buffer_rsrc_t rWp = I4_RSRC(Wp_e), rXh = I4_RSRC(Xhp), rXl = I4_RSRC(Xlp), rXM = I4_RSRC(XMp);
    bool active[NJ];
#pragma unroll
    for (int nj = 0; nj < NJ; ++nj) {
        bool any = false;
        for (int t = 0; t < 16; ++t) {
            const int tok = (tt0 + nj) * 16 + t;
            if (tok < N && row_expert[tok] == e) any = true;
        }
        active[nj] = any;
    }
    if (!active[0] && !active[1]) return;
    float tot[MI][NJ][8];
#pragma unroll
    for (int a = 0; a < MI; a++)
#pragma unroll
        for (int b = 0; b < NJ; b++)
#pragma unroll
            for (int i = 0; i < 8; i++) tot[a][b][i] = 0.f;
    for (int sb = 0; sb < nsb; sb++) {
        uint32_t scw[MI][NJ];
#pragma unroll
        for (int nj = 0; nj < NJ; nj++)
#pragma unroll
            for (int mi = 0; mi < MI; mi++) scw[mi][nj] = *(const uint32_t*) (Wh_e + (((ft0 + mi) * nsb + sb) * 16 + lc) * 4);
        int2v mmw[NJ];
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) mmw[nj] = bl64(rXM, h * 8, ((tt0 + nj) * nsb + sb) * 16);
        int isum[MI][NJ][8];
        for (int u0 = 0; u0 < 8; u0 += GW) {
            int8v ci[MI][NJ];
            const int8v z = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int pass = 0; pass < 2; pass++) {
                const __amdgpu_buffer_rsrc_t rX = pass ? rXl : rXh;
                int4v wc[MI][NJ], xc[NJ], wn_[MI][NJ], xn[NJ];
#pragma unroll
                for (int nj = 0; nj < NJ; nj++) {
#pragma unroll
                    for (int mi = 0; mi < MI; mi++) wc[mi][nj] = bl128(rWp, lc * 16, (((ft0 + mi) * nsb + sb) * 8 + u0) * 256);
                    xc[nj] = bl128(rX, lc * 16, (((tt0 + nj) * nsb + sb) * 8 + u0) * 256);
                }
#pragma unroll
                for (int q = 0; q < GW; q++) {
                    if (q + 1 < GW) {
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) {
#pragma unroll
                            for (int mi = 0; mi < MI; mi++) wn_[mi][nj] = bl128(rWp, lc * 16, (((ft0 + mi) * nsb + sb) * 8 + u0 + q + 1) * 256);
                            xn[nj] = bl128(rX, lc * 16, (((tt0 + nj) * nsb + sb) * 8 + u0 + q + 1) * 256);
                        }
                    }
                    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
                    for (int sl = 0; sl < 2; sl++)
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                            for (int mi = 0; mi < MI; mi++) {
                                const int2v xa = sl ? I4_HI(xc[nj]) : I4_LO(xc[nj]), wb = sl ? I4_HI(wc[mi][nj]) : I4_LO(wc[mi][nj]);
                                const int8v cin = (q == 0 && sl == 0 && pass == 0) ? z : ci[mi][nj];
                                if (pass == 0) ci[mi][nj] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, xa, true, wb, cin, false);
                                else           ci[mi][nj] = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(false, xa, true, wb, cin, false);
                            }
                    if (q + 1 < GW) {
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) {
#pragma unroll
                            for (int mi = 0; mi < MI; mi++) wc[mi][nj] = wn_[mi][nj];
                            xc[nj] = xn[nj];
                        }
                    }
                }
                if (pass == 0) {
#pragma unroll
                    for (int mi = 0; mi < MI; mi++)
#pragma unroll
                        for (int nj = 0; nj < NJ; nj++) ci[mi][nj] = ci[mi][nj] << 4;
                }
            }
            __builtin_amdgcn_sched_barrier(0);
#pragma unroll
            for (int mi = 0; mi < MI; mi++) {
                const int scg = (int) ((scw[mi][0] >> (4 * u0)) & 0xF);
#pragma unroll
                for (int nj = 0; nj < NJ; nj++)
#pragma unroll
                    for (int i = 0; i < 8; i++) { if (u0 == 0) mul24a(isum[mi][nj][i], scg, ci[mi][nj][i]); else mad24(isum[mi][nj][i], scg, ci[mi][nj][i]); }
            }
        }
#pragma unroll
        for (int nj = 0; nj < NJ; nj++) {
            float mv[8];
#pragma unroll
            for (int i = 0; i < 8; i++) mv[i] = (float) (int) (((uint32_t) mmw[nj][i >> 2] >> (8 * (i & 3))) & 0xFF);
#pragma unroll
            for (int mi = 0; mi < MI; mi++)
#pragma unroll
                for (int i = 0; i < 8; i++) tot[mi][nj][i] = fmaf((float) isum[mi][nj][i], mv[i], tot[mi][nj][i]);
        }
    }
    const float inv = 1.f / (float) ((1 << sh) * 16);
#pragma unroll
    for (int nj = 0; nj < NJ; nj++) {
        float dr[8];
        { const float4 a = *(const float4*) (XRp + (size_t) (tt0 + nj) * 16 + h * 8), b = *(const float4*) (XRp + (size_t) (tt0 + nj) * 16 + h * 8 + 4);
          dr[0] = a.x; dr[1] = a.y; dr[2] = a.z; dr[3] = a.w; dr[4] = b.x; dr[5] = b.y; dr[6] = b.z; dr[7] = b.w; }
#pragma unroll
        for (int mi = 0; mi < MI; mi++) {
            const int feat = (ft0 + mi) * 16 + lc;
            const float sf = WS_e[feat] * inv;
#pragma unroll
            for (int i = 0; i < 8; i++) {
                const int tok = (tt0 + nj) * 16 + 2 * i + h;
                if (tok < N && row_expert[tok] == e) { const size_t oi = (size_t) tok * ldy + feat; out[oi] = tot[mi][nj][i] * (sf * dr[i]) + (res ? res[oi] : 0.f); }
            }
        }
    }
}

// rotate + 8-bit quantize for the decode matvec: int8 codes [n][K] + per-32-group (d, d*sum)
__global__ __launch_bounds__(32) void k_rot_q8x(const float * __restrict__ X, int8_t * __restrict__ Q, float2 * __restrict__ D, int K) {
    const int sb = blockIdx.x, n = blockIdx.y, lane = threadIdx.x, h = lane >> 4, lc = lane & 15;
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, lc); a2[k] = h16(lc, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    const float8v w = rotate_block(X + (size_t) n * K + sb * 256, lane, b1, a2);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float x = w[i] * 0.0625f;
        float amax = fabsf(x);
        amax = wave32_max(amax);
        const float d = amax / 127.0f;
        const int q = amax == 0.0f ? 0 : (int) roundf(x / d);
        float sq = (float) q;
        sq = wave32_sum(sq);
        Q[(size_t) n * K + sb * 256 + i * 32 + 16 * h + lc] = (int8_t) q;
        if (lane == 0) D[(size_t) n * (K / 32) + sb * 8 + i] = make_float2(d, d * sq);
    }
}

// Fused swiglu + prep_down + rot_q8x for the INT4 decode path: replaces three launches per visit.
// gu[S][2F] (F32, sg=su=1 in the INT4 path) -> swiglu -> f16 round-trip -> zero-pad to Kpad ->
// rotate the 256-block -> int8 codes Q[S][Kp] + per-32 (d,d*sum) D[S][Kp/32]. Bit-exact vs the
// three separate kernels (swiglu_cap_f32 -> prep_down -> rot_q8x).
__global__ __launch_bounds__(32) void k_int4_mid(const float* __restrict__ gu, int S, int F, int Kpad,
                                                  float gcap, float ucap, int8_t* __restrict__ Q,
                                                  float2* __restrict__ D) {
    const int sb = blockIdx.x, n = blockIdx.y, lane = threadIdx.x;
    const int h = lane >> 4, lc = lane & 15;
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, lc); a2[k] = h16(lc, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    // build this lane's 16 swiglu values at hf[n][sb*256 + lc*16 .. +15]
    float4 v[4];
#pragma unroll
    for (int q = 0; q < 4; q++) {
        float x0 = 0.f, x1 = 0.f, x2 = 0.f, x3 = 0.f;
        const int c0 = sb * 256 + lc * 16 + 4 * q;
        if (c0 < F) {
            const float g = gu[(size_t) n * 2 * F + c0];
            const float u = gu[(size_t) n * 2 * F + F + c0];
            const uint16_t hv = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
            x0 = h2f(hv);
        }
        if (c0 + 1 < F) {
            const float g = gu[(size_t) n * 2 * F + c0 + 1];
            const float u = gu[(size_t) n * 2 * F + F + c0 + 1];
            const uint16_t hv = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
            x1 = h2f(hv);
        }
        if (c0 + 2 < F) {
            const float g = gu[(size_t) n * 2 * F + c0 + 2];
            const float u = gu[(size_t) n * 2 * F + F + c0 + 2];
            const uint16_t hv = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
            x2 = h2f(hv);
        }
        if (c0 + 3 < F) {
            const float g = gu[(size_t) n * 2 * F + c0 + 3];
            const float u = gu[(size_t) n * 2 * F + F + c0 + 3];
            const uint16_t hv = f2h(gcap * tanhf(g / gcap) * (1.0f / (1.0f + expf(-g))) * ucap * tanhf(u / ucap));
            x3 = h2f(hv);
        }
        v[q] = make_float4(x0, x1, x2, x3);
    }
    const float8v w = rotate_regs(v, lane, b1, a2);
#pragma unroll
    for (int i = 0; i < 8; i++) {
        const float x = w[i] * 0.0625f;
        float amax = fabsf(x);
        amax = wave32_max(amax);
        const float d = amax / 127.0f;
        const int qi = amax == 0.0f ? 0 : (int) roundf(x / d);
        float sq = (float) qi;
        sq = wave32_sum(sq);
        Q[(size_t) n * Kpad + sb * 256 + i * 32 + 16 * h + lc] = (int8_t) qi;
        if (lane == 0) D[(size_t) n * (Kpad / 32) + sb * 8 + i] = make_float2(d, d * sq);
    }
}

// decode matvec on the repacked layout: 16 weight rows x MV_NW waves splitting K's 256-blocks
constexpr int MV_NW = 10;
__device__ __forceinline__ int sdot4(int a, int b, int c) { return strata::hip_compat::dp4a(a, b, c); }
template <int NC, int NW = MV_NW>
__global__ __launch_bounds__(32 * NW) void k_mv(const uint8_t * __restrict__ Wp, const uint8_t * __restrict__ Wh, const float * __restrict__ S, float inv,
        const int8_t * __restrict__ Xq, const float2 * __restrict__ Xd, float * __restrict__ dst, const float * __restrict__ xb, int M, int K, int stride_dst) {
    const int lane = threadIdx.x, r = lane & 15, h = lane >> 4, wv = threadIdx.y, ft = blockIdx.x, nsb = K / 256;
    const int prow = ft * 16 + r;
    float pS = S[prow], pxb = 0.f;
    if (xb && h == 0) pxb = xb[prow];
    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; j++) acc[j] = 0.f;
    for (int sb = wv; sb < nsb; sb += NW) {
        const size_t wo = ((size_t) (ft * nsb + sb) * 8 + h * 4) * 256 + r * 16, so = ((size_t) (ft * nsb + sb) * 16 + r) * 4 + h * 2;
        uint4 w[4];
#pragma unroll
        for (int g = 0; g < 4; g++) w[g] = nt128(Wp, wo + g * 256);
        const uint32_t sc = *(const uint16_t *) (Wh + so);
#pragma unroll
        for (int j = 0; j < NC; j++) {
            const uint4 * xq = (const uint4 *) (Xq + (size_t) j * K + sb * 256 + h * 128);
            const float4 * xd = (const float4 *) (Xd + (size_t) j * (K / 32) + sb * 8 + h * 4);
            uint4 xv[8]; float4 dv[2];
#pragma unroll
            for (int i = 0; i < 8; i++) xv[i] = xq[i];
            dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const float2 ds = g < 2 ? (g == 0 ? make_float2(dv[0].x, dv[0].y) : make_float2(dv[0].z, dv[0].w)) : (g == 2 ? make_float2(dv[1].x, dv[1].y) : make_float2(dv[1].z, dv[1].w));
                const uint32_t xw[8] = {xv[2 * g].x, xv[2 * g].y, xv[2 * g].z, xv[2 * g].w, xv[2 * g + 1].x, xv[2 * g + 1].y, xv[2 * g + 1].z, xv[2 * g + 1].w};
                const uint32_t ww[4] = {w[g].x, w[g].y, w[g].z, w[g].w};
                int si = 0;
#pragma unroll
                for (int wi = 0; wi < 4; wi++) {
                    si = sdot4((int) ((ww[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi], si);
                    si = sdot4((int) (((ww[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi + 1], si);
                }
                acc[j] += (float) ((sc >> (4 * g)) & 0xF) * (ds.x * (float) si - 8.f * ds.y);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NC; j++) acc[j] += swap_rows(acc[j]);
    __shared__ float red[NW][NC][16];
    if (h == 0) {
#pragma unroll
        for (int j = 0; j < NC; j++) red[wv][j][r] = acc[j];
    }
    __syncthreads();
    if (wv == 0 && h == 0) {
        const int row = ft * 16 + r;
        const float sf = pS * inv;
#pragma unroll
        for (int j = 0; j < NC; j++) {
            float a = 0.f;
#pragma unroll
            for (int q = 0; q < NW; q++) a += red[q][j][r];
            float res = a * sf + pxb;
            dst[(size_t) j * stride_dst + row] = res;
        }
    }
}

// IK2_MV_LDS (default OFF): k_mv with the whole NC-token activation staged into dynamic LDS once per
// CTA (host sizes smem = NC*K + NC*(K/32)*8 bytes; K=2304,NC=8 -> 23040B). The sb/j/g loop then reads
// each 64-code group's slice from LDS right before use instead of staging xv[8]+dv[2] (40 VGPRs) from
// global, cutting live activation registers so more waves fit per SIMD. Bit-exact vs k_mv: identical
// weight loads (nt128, registers), dp4a chain, op order, swap_rows + red[NW][NC][16] reduction and dst
// write; the ONLY change is the activation source (global -> LDS).
template <int NC, int NW = MV_NW>
__global__ __launch_bounds__(32 * NW) void k_mv_lds(const uint8_t * __restrict__ Wp, const uint8_t * __restrict__ Wh, const float * __restrict__ S, float inv,
        const int8_t * __restrict__ Xq, const float2 * __restrict__ Xd, float * __restrict__ dst, const float * __restrict__ xb, int M, int K, int stride_dst) {
    extern __shared__ __align__(16) char smem[];
    int8_t * sxq = (int8_t *) smem;                       // [NC][K] int8 codes
    float2 * sxd = (float2 *) (smem + (size_t) NC * K);   // [NC][K/32] (d, d*sum); NC*K is 256-aligned
    // Cooperative stage: vectorized 16B copies, CTA-strided. Xq rows are K bytes (K%256==0) and Xd rows
    // K/32 float2, both contiguous from a 256B-aligned base, so every chunk below is 16B-aligned.
    const int tid = threadIdx.y * 32 + threadIdx.x, nt = 32 * NW;
    const int nq = NC * (K / 16);                         // uint4 chunks of Xq
    for (int i = tid; i < nq; i += nt) ((uint4 *) sxq)[i] = ((const uint4 *) Xq)[i];
    const int nd = NC * (K / 64);                         // float4 chunks of Xd (2 float2 each)
    for (int i = tid; i < nd; i += nt) ((float4 *) sxd)[i] = ((const float4 *) Xd)[i];
    __syncthreads();
    const int lane = threadIdx.x, r = lane & 15, h = lane >> 4, wv = threadIdx.y, ft = blockIdx.x, nsb = K / 256;
    const int prow = ft * 16 + r;
    float pS = S[prow], pxb = 0.f;
    if (xb && h == 0) pxb = xb[prow];
    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; j++) acc[j] = 0.f;
    for (int sb = wv; sb < nsb; sb += NW) {
        const size_t wo = ((size_t) (ft * nsb + sb) * 8 + h * 4) * 256 + r * 16, so = ((size_t) (ft * nsb + sb) * 16 + r) * 4 + h * 2;
        uint4 w[4];
#pragma unroll
        for (int g = 0; g < 4; g++) w[g] = nt128(Wp, wo + g * 256);
        const uint32_t sc = *(const uint16_t *) (Wh + so);
#pragma unroll
        for (int j = 0; j < NC; j++) {
            const uint4 * xq = (const uint4 *) (sxq + (size_t) j * K + sb * 256 + h * 128);
            const float2 * xd = (const float2 *) (sxd + (size_t) j * (K / 32) + sb * 8 + h * 4);
#pragma unroll
            for (int g = 0; g < 4; g++) {
                // Per-group LDS reads: only 2 uint4 + 1 float2 of activation live at a time. Same values
                // and order as k_mv's xv[2g]/xv[2g+1] and the dv-derived ds (xd[g] == dv[g/2].{x,y|z,w}).
                const uint4 x0 = xq[2 * g], x1 = xq[2 * g + 1];
                const float2 ds = xd[g];
                const uint32_t xw[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
                const uint32_t ww[4] = {w[g].x, w[g].y, w[g].z, w[g].w};
                int si = 0;
#pragma unroll
                for (int wi = 0; wi < 4; wi++) {
                    si = sdot4((int) ((ww[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi], si);
                    si = sdot4((int) (((ww[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi + 1], si);
                }
                acc[j] += (float) ((sc >> (4 * g)) & 0xF) * (ds.x * (float) si - 8.f * ds.y);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NC; j++) acc[j] += swap_rows(acc[j]);
    __shared__ float red[NW][NC][16];
    if (h == 0) {
#pragma unroll
        for (int j = 0; j < NC; j++) red[wv][j][r] = acc[j];
    }
    __syncthreads();
    if (wv == 0 && h == 0) {
        const int row = ft * 16 + r;
        const float sf = pS * inv;
#pragma unroll
        for (int j = 0; j < NC; j++) {
            float a = 0.f;
#pragma unroll
            for (int q = 0; q < NW; q++) a += red[q][j][r];
            float res = a * sf + pxb;
            dst[(size_t) j * stride_dst + row] = res;
        }
    }
}

// IK2_INT4_MV_ALT: k_mv over a fused projection whose row ranges take different activations.
// One full-grid launch replaces the msg branch's four small-grid k_mv launches on slices of the
// SAME sidecar. Tile ft picks its activation set by row range: [0,b0) -> (Xq0,Xd0), [b0,b1) ->
// (Xq1,Xd1), [b1,b2) -> (Xq2,Xd2), [b2,M) -> (Xq0,Xd0) again (r+gate reuses q's input). Boundaries
// are 16-aligned so the select is uniform per 16-row tile. Everything below the select is the
// k_mv body verbatim: same weight addresses (repack is 16-row-block local), same dp4a chain,
// same sb/wave partition and reduction order -> bit-identical outputs vs the sliced calls.
template <int NC, int NW = MV_NW>
__global__ __launch_bounds__(32 * NW) void k_mv_alt(const uint8_t * __restrict__ Wp, const uint8_t * __restrict__ Wh, const float * __restrict__ S, float inv,
        const int8_t * __restrict__ Xq0, const float2 * __restrict__ Xd0, const int8_t * __restrict__ Xq1, const float2 * __restrict__ Xd1,
        const int8_t * __restrict__ Xq2, const float2 * __restrict__ Xd2, int b0, int b1, int b2,
        float * __restrict__ dst, const float * __restrict__ xb, int M, int K, int stride_dst) {
    const int lane = threadIdx.x, r = lane & 15, h = lane >> 4, wv = threadIdx.y, ft = blockIdx.x, nsb = K / 256;
    const int row0 = ft * 16;
    const int8_t * Xq = row0 < b0 ? Xq0 : (row0 < b1 ? Xq1 : (row0 < b2 ? Xq2 : Xq0));
    const float2 * Xd = row0 < b0 ? Xd0 : (row0 < b1 ? Xd1 : (row0 < b2 ? Xd2 : Xd0));
    const int prow = ft * 16 + r;
    float pS = S[prow], pxb = 0.f;
    if (xb && h == 0) pxb = xb[prow];
    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; j++) acc[j] = 0.f;
    for (int sb = wv; sb < nsb; sb += NW) {
        const size_t wo = ((size_t) (ft * nsb + sb) * 8 + h * 4) * 256 + r * 16, so = ((size_t) (ft * nsb + sb) * 16 + r) * 4 + h * 2;
        uint4 w[4];
#pragma unroll
        for (int g = 0; g < 4; g++) w[g] = nt128(Wp, wo + g * 256);
        const uint32_t sc = *(const uint16_t *) (Wh + so);
#pragma unroll
        for (int j = 0; j < NC; j++) {
            const uint4 * xq = (const uint4 *) (Xq + (size_t) j * K + sb * 256 + h * 128);
            const float4 * xd = (const float4 *) (Xd + (size_t) j * (K / 32) + sb * 8 + h * 4);
            uint4 xv[8]; float4 dv[2];
#pragma unroll
            for (int i = 0; i < 8; i++) xv[i] = xq[i];
            dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const float2 ds = g < 2 ? (g == 0 ? make_float2(dv[0].x, dv[0].y) : make_float2(dv[0].z, dv[0].w)) : (g == 2 ? make_float2(dv[1].x, dv[1].y) : make_float2(dv[1].z, dv[1].w));
                const uint32_t xw[8] = {xv[2 * g].x, xv[2 * g].y, xv[2 * g].z, xv[2 * g].w, xv[2 * g + 1].x, xv[2 * g + 1].y, xv[2 * g + 1].z, xv[2 * g + 1].w};
                const uint32_t ww[4] = {w[g].x, w[g].y, w[g].z, w[g].w};
                int si = 0;
#pragma unroll
                for (int wi = 0; wi < 4; wi++) {
                    si = sdot4((int) ((ww[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi], si);
                    si = sdot4((int) (((ww[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi + 1], si);
                }
                acc[j] += (float) ((sc >> (4 * g)) & 0xF) * (ds.x * (float) si - 8.f * ds.y);
            }
        }
    }
#pragma unroll
    for (int j = 0; j < NC; j++) acc[j] += swap_rows(acc[j]);
    __shared__ float red[NW][NC][16];
    if (h == 0) {
#pragma unroll
        for (int j = 0; j < NC; j++) red[wv][j][r] = acc[j];
    }
    __syncthreads();
    if (wv == 0 && h == 0) {
        const int row = ft * 16 + r;
        const float sf = pS * inv;
#pragma unroll
        for (int j = 0; j < NC; j++) {
            float a = 0.f;
#pragma unroll
            for (int q = 0; q < NW; q++) a += red[q][j][r];
            float res = a * sf + pxb;
            dst[(size_t) j * stride_dst + row] = res;
        }
    }
}

// Grouped decode matvec: S slots, each a different expert but the SAME activation token.
// One launch grid(M/16, S); per-slot Wp/Wh/WS bases (already offset by that expert).
// Xq/Xd is the single rotated+quantized activation, shared by every slot.
template <int NC, int NW = MV_NW, bool SWAP = false, bool CACHE = false>
__global__ __launch_bounds__(32 * NW) void k_mv_grouped(
        const uint8_t* const* Wp, const uint8_t* const* Wh, const float* const* Sg, float inv,
        const int8_t* __restrict__ Xq, const float2* __restrict__ Xd, float* __restrict__ dst,
        int M, int K, int stride_dst, int xq_stride, int xd_stride) {
    const int lane = threadIdx.x, r = lane & 15, h = lane >> 4, wv = threadIdx.y;
    // SWAP: slot-major dispatch (IK2_MV_SLOT_MAJOR=1). Blocks of one ft-slab run back-to-back, so slots that
    // route to the same expert read its weights close together (L2 hits) instead of ~20 slots apart. Bit-exact.
    const int ft = SWAP ? blockIdx.y : blockIdx.x, s = SWAP ? blockIdx.x : blockIdx.y, nsb = K / 256;
    const uint8_t* Wp_s = Wp[s]; const uint8_t* Wh_s = Wh[s]; const float* S_s = Sg[s];
    Xq += (size_t) s * xq_stride; Xd += (size_t) s * xd_stride;
    const int prow = ft * 16 + r;
    const float pS = S_s[prow];
    float acc[NC];
#pragma unroll
    for (int j = 0; j < NC; j++) acc[j] = 0.f;
    for (int sb = wv; sb < nsb; sb += NW) {
        const size_t wo = ((size_t) (ft * nsb + sb) * 8 + h * 4) * 256 + r * 16, so = ((size_t) (ft * nsb + sb) * 16 + r) * 4 + h * 2;
        uint4 w[4];
#pragma unroll
        for (int g = 0; g < 4; g++) w[g] = CACHE ? cl128(Wp_s, wo + g * 256) : nt128(Wp_s, wo + g * 256);
        const uint32_t sc = *(const uint16_t *) (Wh_s + so);
        const uint4 * xq = (const uint4 *) (Xq + sb * 256 + h * 128);
        const float4 * xd = (const float4 *) (Xd + sb * 8 + h * 4);
        uint4 xv[8]; float4 dv[2];
#pragma unroll
        for (int i = 0; i < 8; i++) xv[i] = xq[i];
        dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const float2 ds = g < 2 ? (g == 0 ? make_float2(dv[0].x, dv[0].y) : make_float2(dv[0].z, dv[0].w)) : (g == 2 ? make_float2(dv[1].x, dv[1].y) : make_float2(dv[1].z, dv[1].w));
            const uint32_t xw[8] = {xv[2 * g].x, xv[2 * g].y, xv[2 * g].z, xv[2 * g].w, xv[2 * g + 1].x, xv[2 * g + 1].y, xv[2 * g + 1].z, xv[2 * g + 1].w};
            const uint32_t ww[4] = {w[g].x, w[g].y, w[g].z, w[g].w};
            int si = 0;
#pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                si = sdot4((int) ((ww[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi], si);
                si = sdot4((int) (((ww[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi + 1], si);
            }
            for (int j = 0; j < NC; j++) acc[j] += (float) ((sc >> (4 * g)) & 0xF) * (ds.x * (float) si - 8.f * ds.y);
        }
    }
#pragma unroll
    for (int j = 0; j < NC; j++) acc[j] += swap_rows(acc[j]);
    __shared__ float red[NW][16];
    if (h == 0) red[wv][r] = acc[0];
    __syncthreads();
    if (wv == 0 && h == 0) {
        float a = 0.f;
#pragma unroll
        for (int q = 0; q < NW; q++) a += red[q][r];
        dst[(size_t) s * stride_dst + prow] = a * (pS * inv);
    }
}

// ---- IK2_MOE_GROUP_EXPERT (default OFF): grouped-by-expert verify MoE ----
// One block per (expert group, 16-row tile). The group's expert weight strip is read ONCE and
// dotted against ALL its slots (MAXS), instead of once per slot as in k_mv_grouped. Deterministic
// grouping from the device-side routing (k_moe_group_build): one block, one thread, S<=64 slots in
// ascending slot order, experts in first-appearance order, at most 8 slots per group (route picks
// K distinct experts per token, so an expert appears at most T<=8 times); overflow starts a new
// group. Fixed launch grid (M/16, S): blocks with g >= ngrp early-exit, so IK2_VERIFY_GRAPH stays
// capturable. Bit-exact vs k_mv_grouped per slot: same sb/wave partition, same dp4a chain and g
// order into acc[j], same swap_rows, same ascending red[q] sum, same final (pS*inv) multiply --
// only the number of weight loads changes.

constexpr int MOE_GMAXS = 8;

__global__ void k_moe_group_build(const int* __restrict__ top, int S, int* __restrict__ gcnt,
                                  int* __restrict__ gslots, int* __restrict__ gexp,
                                  int* __restrict__ ngrp) {
    // LDS staging: the single grouping thread scans group metadata thousands of times; global
    // loads would serialize ~50us of latency per visit. Cooperative preload, then LDS-only scans.
    __shared__ int st[64], se[64], sc[64];
    const int tid = threadIdx.x;
    for (int i = tid; i < S; i += 32) st[i] = top[i];
    __syncthreads();
    if (tid != 0) return;
    int ng = 0;
    for (int s = 0; s < S; s++) {
        const int e = st[s];
        int g = -1;
        for (int i = ng - 1; i >= 0; i--)   // last group of e first: fills it before opening a new one
            if (se[i] == e && sc[i] < MOE_GMAXS) { g = i; break; }
        if (g < 0) { g = ng++; se[g] = e; sc[g] = 0; }
        gslots[g * MOE_GMAXS + sc[g]++] = s;
    }
    for (int i = 0; i < ng; i++) { gcnt[i] = sc[i]; gexp[i] = se[i]; }
    *ngrp = ng;
}

template <int NW>
__global__ __launch_bounds__(32 * NW) void k_mv_grouped_exp(
        const uint8_t* const* Wp, const uint8_t* const* Wh, const float* const* Sg, float inv,
        const int8_t* __restrict__ Xq, const float2* __restrict__ Xd, float* __restrict__ dst,
        int M, int K, int stride_dst, int xq_stride, int xd_stride,
        const int* __restrict__ gcnt, const int* __restrict__ gslots, const int* __restrict__ gexp,
        const int* __restrict__ ngrp) {
    const int g = blockIdx.y;
    if (g >= *ngrp) return;   // block-uniform: no barrier below this point is skipped by part of a block
    const int ns = gcnt[g], e = gexp[g];
    const int lane = threadIdx.x, r = lane & 15, h = lane >> 4, wv = threadIdx.y;
    const int ft = blockIdx.x, nsb = K / 256;
    const uint8_t* Wp_s = Wp[e]; const uint8_t* Wh_s = Wh[e];
    const int prow = ft * 16 + r;
    const float pS = Sg[e][prow];   // per-row scale: one WS array per expert, shared by all its slots
    float acc[MOE_GMAXS];
#pragma unroll
    for (int j = 0; j < MOE_GMAXS; j++) acc[j] = 0.f;
    for (int sb = wv; sb < nsb; sb += NW) {
        const size_t wo = ((size_t) (ft * nsb + sb) * 8 + h * 4) * 256 + r * 16, so = ((size_t) (ft * nsb + sb) * 16 + r) * 4 + h * 2;
        uint4 w[4];
#pragma unroll
        for (int gg = 0; gg < 4; gg++) w[gg] = nt128(Wp_s, wo + gg * 256);
        const uint32_t sc = *(const uint16_t *) (Wh_s + so);
#pragma unroll
        for (int j = 0; j < MOE_GMAXS; j++) {
            if (j >= ns) break;   // wave-uniform: ns is block-uniform, no divergence
            // s is wave-uniform (scalar load): the activation base stays in SGPRs, no VGPR pointer array.
            const int s = gslots[g * MOE_GMAXS + j];
            const uint4* xq = (const uint4*) (Xq + (size_t) s * xq_stride + sb * 256 + h * 128);
            const float4* xd = (const float4*) (Xd + (size_t) s * xd_stride + sb * 8 + h * 4);
            uint4 xv[8]; float4 dv[2];
#pragma unroll
            for (int i = 0; i < 8; i++) xv[i] = xq[i];
            dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
            for (int gg = 0; gg < 4; gg++) {
                const float2 ds = gg < 2 ? (gg == 0 ? make_float2(dv[0].x, dv[0].y) : make_float2(dv[0].z, dv[0].w)) : (gg == 2 ? make_float2(dv[1].x, dv[1].y) : make_float2(dv[1].z, dv[1].w));
                const uint32_t xw[8] = {xv[2 * gg].x, xv[2 * gg].y, xv[2 * gg].z, xv[2 * gg].w, xv[2 * gg + 1].x, xv[2 * gg + 1].y, xv[2 * gg + 1].z, xv[2 * gg + 1].w};
                const uint32_t ww[4] = {w[gg].x, w[gg].y, w[gg].z, w[gg].w};
                int si = 0;
#pragma unroll
                for (int wi = 0; wi < 4; wi++) {
                    si = sdot4((int) ((ww[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi], si);
                    si = sdot4((int) (((ww[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi + 1], si);
                }
                acc[j] += (float) ((sc >> (4 * gg)) & 0xF) * (ds.x * (float) si - 8.f * ds.y);
            }
        }
    }
    __shared__ float red[NW][MOE_GMAXS][16];
#pragma unroll
    for (int j = 0; j < MOE_GMAXS; j++) {
        if (j >= ns) break;
        acc[j] += swap_rows(acc[j]);
        if (h == 0) red[wv][j][r] = acc[j];
    }
    __syncthreads();
    if (wv == 0 && h == 0) {
#pragma unroll
        for (int j = 0; j < MOE_GMAXS; j++) {
            if (j >= ns) break;
            float a = 0.f;
#pragma unroll
            for (int q = 0; q < NW; q++) a += red[q][j][r];
            dst[(size_t) gslots[g * MOE_GMAXS + j] * stride_dst + prow] = a * (pS * inv);
        }
    }
}

// Phase-D fusion: k_mv_grouped with the rot_q8x prologue folded in. Wave wv rotates+quants its
// own 256-block of the activation into LDS (one block per wave: host guarantees K/256 == NW),
// then the matvec reads codes/scales from LDS instead of global memory. SRC=0: shared f32 X[K]
// (T=1 gate_up). SRC=1: per-slot f16 H16[s][F] zero-padded to K (down: replaces prep_down+rot).
// Bit-exact vs the separate kernels: same rotate_block/rotate_regs + quant_rot8 math, same
// accumulation order. Matvec grid/threads unchanged (no parallelism lost, unlike the mid fusion).
template <int NW, int SRC>
__global__ __launch_bounds__(32 * NW) void k_mv_grouped_rot(
        const uint8_t* const* Wp, const uint8_t* const* Wh, const float* const* Sg, float inv,
        const float* __restrict__ X, const uint16_t* __restrict__ H16, int F,
        float* __restrict__ dst, int M, int K, int stride_dst) {
    const int lane = threadIdx.x, r = lane & 15, h = lane >> 4, wv = threadIdx.y;
    const int ft = blockIdx.x, s = blockIdx.y;
    const uint8_t* Wp_s = Wp[s]; const uint8_t* Wh_s = Wh[s]; const float* S_s = Sg[s];
    const int prow = ft * 16 + r;
    const float pS = S_s[prow];
    __shared__ __align__(16) int8_t lq[NW * 256];
    __shared__ __align__(16) float2 ld[NW * 8];
    // prologue: wave wv owns block sb=wv (K/256 == NW)
    const int sb = wv;
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, r); a2[k] = h16(r, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    float8v rot;
    if (SRC == 0) {
        rot = rotate_block(X + sb * 256, lane, b1, a2);
    } else {
        const uint16_t* hs = H16 + (size_t) s * F;
        float4 v[4];
#pragma unroll
        for (int q = 0; q < 4; q++) {
            const int c0 = sb * 256 + r * 16 + 4 * q;
            float x0 = 0.f, x1 = 0.f, x2 = 0.f, x3 = 0.f;
            if (c0 < F) x0 = h2f(hs[c0]);
            if (c0 + 1 < F) x1 = h2f(hs[c0 + 1]);
            if (c0 + 2 < F) x2 = h2f(hs[c0 + 2]);
            if (c0 + 3 < F) x3 = h2f(hs[c0 + 3]);
            v[q] = make_float4(x0, x1, x2, x3);
        }
        rot = rotate_regs(v, lane, b1, a2);
    }
    quant_rot8(rot, lane, lq + wv * 256, ld + wv * 8);
    __syncthreads();
    float acc = 0.f;
    {
        const size_t wo = ((size_t) (ft * NW + sb) * 8 + h * 4) * 256 + r * 16, so = ((size_t) (ft * NW + sb) * 16 + r) * 4 + h * 2;
        uint4 w[4];
#pragma unroll
        for (int g = 0; g < 4; g++) w[g] = nt128(Wp_s, wo + g * 256);
        const uint32_t sc = *(const uint16_t *) (Wh_s + so);
        const uint4* xq = (const uint4*) (lq + wv * 256 + h * 128);
        const float4* xd = (const float4*) (ld + wv * 8 + h * 4);
        uint4 xv[8]; float4 dv[2];
#pragma unroll
        for (int i = 0; i < 8; i++) xv[i] = xq[i];
        dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const float2 ds = g < 2 ? (g == 0 ? make_float2(dv[0].x, dv[0].y) : make_float2(dv[0].z, dv[0].w)) : (g == 2 ? make_float2(dv[1].x, dv[1].y) : make_float2(dv[1].z, dv[1].w));
            const uint32_t xw[8] = {xv[2 * g].x, xv[2 * g].y, xv[2 * g].z, xv[2 * g].w, xv[2 * g + 1].x, xv[2 * g + 1].y, xv[2 * g + 1].z, xv[2 * g + 1].w};
            const uint32_t ww[4] = {w[g].x, w[g].y, w[g].z, w[g].w};
            int si = 0;
#pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                si = sdot4((int) ((ww[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi], si);
                si = sdot4((int) (((ww[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi + 1], si);
            }
            acc += (float) ((sc >> (4 * g)) & 0xF) * (ds.x * (float) si - 8.f * ds.y);
        }
    }
    acc += swap_rows(acc);
    __shared__ float red[NW][16];
    if (h == 0) red[wv][r] = acc;
    __syncthreads();
    if (wv == 0 && h == 0) {
        float a = 0.f;
#pragma unroll
        for (int q = 0; q < NW; q++) a += red[q][r];
        dst[(size_t) s * stride_dst + prow] = a * (pS * inv);
    }
}

// IK2_INT4_DENSE: the dense projections' decode matvec, single matrix [M][K], T=1, with the
// rot_q8x prologue folded in exactly as k_mv_grouped_rot (wave wv owns 256-block sb=wv; host
// guarantees K/256 == NW, instantiated for 9 (K=2304) and 12 (K=3072, wo)). ALT=1: the rows of
// a0's/a1's ranges take their own inputs - picked per TILE by its first row (the fused
// q|k|v|r|gate segment bounds are multiples of 16, so a tile never straddles two inputs).
// Epilogue: mv_conv_row per row, exactly as mv_q8p_kernel's (one writer thread per row).
// Same rotate_block/quant_rot8 math and accumulation order as the expert decode kernels.
template <int NW, int ALT>
__global__ __launch_bounds__(32 * NW) void k_mv_dense_rot(
        const uint8_t* __restrict__ Wp, const uint8_t* __restrict__ Wh, const float* __restrict__ S, float inv,
        const float* __restrict__ X, float* __restrict__ dst, MvConv c0, MvConv c1, MvAlt a0, MvAlt a1) {
    const int lane = threadIdx.x, r = lane & 15, h = lane >> 4, wv = threadIdx.y;
    const int ft = blockIdx.x;
    const int prow = ft * 16 + r;
    const float pS = S[prow];
    __shared__ __align__(16) int8_t lq[NW * 256];
    __shared__ __align__(16) float2 ld[NW * 8];
    // prologue: wave wv owns block sb=wv (K/256 == NW)
    const int sb = wv;
    const float* xtile = X;
    if (ALT) {
        const int row0 = ft * 16;
        xtile = (unsigned) (row0 - a0.r0) < (unsigned) a0.n   ? a0.x
                : (unsigned) (row0 - a1.r0) < (unsigned) a1.n ? a1.x
                                                              : X;
    }
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, r); a2[k] = h16(r, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    const float8v rot = rotate_block(xtile + sb * 256, lane, b1, a2);
    quant_rot8(rot, lane, lq + wv * 256, ld + wv * 8);
    __syncthreads();
    float acc = 0.f;
    {
        const size_t wo = ((size_t) (ft * NW + sb) * 8 + h * 4) * 256 + r * 16, so = ((size_t) (ft * NW + sb) * 16 + r) * 4 + h * 2;
        uint4 w[4];
#pragma unroll
        for (int g = 0; g < 4; g++) w[g] = nt128(Wp, wo + g * 256);
        const uint32_t sc = *(const uint16_t *) (Wh + so);
        const uint4* xq = (const uint4*) (lq + wv * 256 + h * 128);
        const float4* xd = (const float4*) (ld + wv * 8 + h * 4);
        uint4 xv[8]; float4 dv[2];
#pragma unroll
        for (int i = 0; i < 8; i++) xv[i] = xq[i];
        dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
        for (int g = 0; g < 4; g++) {
            const float2 ds = g < 2 ? (g == 0 ? make_float2(dv[0].x, dv[0].y) : make_float2(dv[0].z, dv[0].w)) : (g == 2 ? make_float2(dv[1].x, dv[1].y) : make_float2(dv[1].z, dv[1].w));
            const uint32_t xw[8] = {xv[2 * g].x, xv[2 * g].y, xv[2 * g].z, xv[2 * g].w, xv[2 * g + 1].x, xv[2 * g + 1].y, xv[2 * g + 1].z, xv[2 * g + 1].w};
            const uint32_t ww[4] = {w[g].x, w[g].y, w[g].z, w[g].w};
            int si = 0;
#pragma unroll
            for (int wi = 0; wi < 4; wi++) {
                si = sdot4((int) ((ww[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi], si);
                si = sdot4((int) (((ww[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi + 1], si);
            }
            acc += (float) ((sc >> (4 * g)) & 0xF) * (ds.x * (float) si - 8.f * ds.y);
        }
    }
    acc += swap_rows(acc);
    __shared__ float red[NW][16];
    if (h == 0) red[wv][r] = acc;
    __syncthreads();
    if (wv == 0 && h == 0) {
        float a = 0.f;
#pragma unroll
        for (int q = 0; q < NW; q++) a += red[q][r];
        const float res = a * (pS * inv);
        if (!mv_conv_row(c0, prow, res) && !mv_conv_row(c1, prow, res)) dst[prow] = res;
    }
}

// NC activation rows share each weight-row read. Each row runs the same rotate/quant/sdot/reduction
// order as k_mv_dense_rot (one row's lq/ld/red buffers reused per iteration, weights stay in
// registers across rows), so NC=1 is bit-exact with that kernel.
template <int NC, int NW, int ALT>
__global__ __launch_bounds__(32 * NW) void k_mv_dense_rot_rows(
        const uint8_t* __restrict__ Wp, const uint8_t* __restrict__ Wh, const float* __restrict__ S, float inv,
        const float* __restrict__ X, int K, int T, float* __restrict__ dst, int ldy,
        const MvConv* __restrict__ convs, int nconv) {
    const int lane = threadIdx.x, r = lane & 15, h = lane >> 4, wv = threadIdx.y;
    const int ft = blockIdx.x;
    const int prow = ft * 16 + r;
    const float pS = S[prow];
    __shared__ __align__(16) int8_t lq[NW * 256];
    __shared__ __align__(16) float2 ld[NW * 8];
    __shared__ float red[NW][16];
    (void) ALT;
    half16v b1, a2;
#pragma unroll
    for (int k = 0; k < 16; k++) { b1[k] = h16(k, r); a2[k] = h16(r, k < 8 ? 2 * k : 2 * (k - 8) + 1); }
    // wave wv owns block sb=wv (K/256 == NW); read the weight row ONCE for all NC activation rows.
    const size_t wo = ((size_t) (ft * NW + wv) * 8 + h * 4) * 256 + r * 16;
    const size_t so = ((size_t) (ft * NW + wv) * 16 + r) * 4 + h * 2;
    uint4 w[4];
#pragma unroll
    for (int g = 0; g < 4; g++) w[g] = nt128(Wp, wo + g * 256);
    const uint32_t sc = *(const uint16_t *) (Wh + so);
#pragma unroll
    for (int row = 0; row < NC; ++row) {
        const float8v rot = rotate_block(X + (size_t) row * K + wv * 256, lane, b1, a2);
        quant_rot8(rot, lane, lq + wv * 256, ld + wv * 8);
        __syncthreads();
        float acc = 0.f;
        {
            const uint4* xq = (const uint4*) (lq + wv * 256 + h * 128);
            const float4* xd = (const float4*) (ld + wv * 8 + h * 4);
            uint4 xv[8]; float4 dv[2];
#pragma unroll
            for (int i = 0; i < 8; i++) xv[i] = xq[i];
            dv[0] = xd[0]; dv[1] = xd[1];
#pragma unroll
            for (int g = 0; g < 4; g++) {
                const float2 ds = g < 2 ? (g == 0 ? make_float2(dv[0].x, dv[0].y) : make_float2(dv[0].z, dv[0].w)) : (g == 2 ? make_float2(dv[1].x, dv[1].y) : make_float2(dv[1].z, dv[1].w));
                const uint32_t xw[8] = {xv[2 * g].x, xv[2 * g].y, xv[2 * g].z, xv[2 * g].w, xv[2 * g + 1].x, xv[2 * g + 1].y, xv[2 * g + 1].z, xv[2 * g + 1].w};
                const uint32_t ww[4] = {w[g].x, w[g].y, w[g].z, w[g].w};
                int si = 0;
#pragma unroll
                for (int wi = 0; wi < 4; wi++) {
                    si = sdot4((int) ((ww[wi] & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi], si);
                    si = sdot4((int) (((ww[wi] >> 4) & 0x0F0F0F0Fu) ^ 0x08080808u), (int) xw[2 * wi + 1], si);
                }
                acc += (float) ((sc >> (4 * g)) & 0xF) * (ds.x * (float) si - 8.f * ds.y);
            }
        }
        acc += swap_rows(acc);
        if (h == 0) red[wv][r] = acc;
        __syncthreads();
        if (wv == 0 && h == 0) {
            float a = 0.f;
#pragma unroll
            for (int q = 0; q < NW; q++) a += red[q][r];
            const float res = a * (pS * inv);
            const int yrow = row * ldy + prow;
            if (row < nconv && !mv_conv_row(convs[row], prow, res)) dst[yrow] = res;
            else if (row >= nconv) dst[yrow] = res;
        }
        __syncthreads();
    }
}

// Gather per-slot INT4 weight pointers from the routing's expert ids. One thread per slot.
__global__ void int4_gather_slots_kernel(const int* __restrict__ top,
        const uint8_t* const* __restrict__ tabWp, const uint8_t* const* __restrict__ tabWh,
        const float* const* __restrict__ tabWS, int S,
        const uint8_t** __restrict__ Wp, const uint8_t** __restrict__ Wh, const float** __restrict__ WS) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= S) return;
    int e = top[s];
    Wp[s] = tabWp[e]; Wh[s] = tabWh[e]; WS[s] = tabWS[e];
}

// swiglu's fp16 output h16[S][F] -> f32 hf[S][Kpad], zero-padded beyond F.
__global__ void int4_prep_down_kernel(const uint16_t* __restrict__ h16, float* __restrict__ hf, int S, int F, int Kpad) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= Kpad) return;
    for (int s = 0; s < S; ++s) hf[(size_t) s * Kpad + c] = c < F ? h2f(h16[(size_t) s * F + c]) : 0.0f;
}
}  // namespace int4w

void int4_gather_slots(const int* top, const uint8_t* const* tabWp, const uint8_t* const* tabWh,
                       const float* const* tabWS, int S, const uint8_t** Wp, const uint8_t** Wh,
                       const float** WS, cudaStream_t s) {
    int4w::int4_gather_slots_kernel<<<(S + 63) / 64, 64, 0, s>>>(top, tabWp, tabWh, tabWS, S, Wp, Wh, WS);
    check("int4_gather_slots");
}
void int4_prep_down(const uint16_t* h16, float* hf, int S, int F, int Kpad, cudaStream_t s) {
    int4w::int4_prep_down_kernel<<<(Kpad + 255) / 256, 256, 0, s>>>(h16, hf, S, F, Kpad);
    check("int4_prep_down");
}

void int4_matvec_grouped(const uint8_t* const* Wp, const uint8_t* const* Wh, const float* const* WS,
                         int M, int K, const int8_t* Xq, const float2* Xd, int S,
                         float* out, int stride_dst, float inv, cudaStream_t s,
                         int xq_stride, int xd_stride) {
    if (S <= 0 || M % 16 != 0 || K % 256 != 0) return;
    if (xq_stride < 0) xq_stride = 0;
    if (xd_stride < 0) xd_stride = 0;
    // Right-size the wave count: one wave per 256-wide K block, capped at MV_NW. Down (K=1280) needs
    // only 5 waves, not 10; the extra 5 would idle through the reduction. Bit-exact: each wave owns
    // disjoint sb slices, the reduction just sums fewer partials.
    const int nw_auto = K / 256 < int4w::MV_NW ? K / 256 : int4w::MV_NW;
    // IK2_MV_NW overrides the wave count (1..MV_NW) for occupancy tuning. Bit-exact: waves own
    // disjoint K slices; fewer waves = more sb iters per wave, same partials summed. Default 0 = auto.
    static const int nw_env = getenv("IK2_MV_NW") ? std::atoi(getenv("IK2_MV_NW")) : 0;
    int nw = nw_auto;
    if (nw_env >= 1 && nw_env <= int4w::MV_NW) nw = nw_env;
    static const int slot_major = getenv("IK2_MV_SLOT_MAJOR") ? std::atoi(getenv("IK2_MV_SLOT_MAJOR")) : 0;
    static const int cache_ld = getenv("IK2_MV_CACHE") ? std::atoi(getenv("IK2_MV_CACHE")) : 0;
    const dim3 grid(slot_major ? S : M / 16, slot_major ? M / 16 : S);
#define MV_LAUNCH(NWV) do { \
    if (slot_major && cache_ld) int4w::k_mv_grouped<1, NWV, true, true><<<grid, dim3(32, NWV), 0, s>>>(Wp, Wh, WS, inv, Xq, Xd, out, M, K, stride_dst, xq_stride, xd_stride); \
    else if (slot_major) int4w::k_mv_grouped<1, NWV, true, false><<<grid, dim3(32, NWV), 0, s>>>(Wp, Wh, WS, inv, Xq, Xd, out, M, K, stride_dst, xq_stride, xd_stride); \
    else if (cache_ld) int4w::k_mv_grouped<1, NWV, false, true><<<grid, dim3(32, NWV), 0, s>>>(Wp, Wh, WS, inv, Xq, Xd, out, M, K, stride_dst, xq_stride, xd_stride); \
    else int4w::k_mv_grouped<1, NWV, false, false><<<grid, dim3(32, NWV), 0, s>>>(Wp, Wh, WS, inv, Xq, Xd, out, M, K, stride_dst, xq_stride, xd_stride); \
} while (0)
    switch (nw) {
        case 1: MV_LAUNCH(1); break;
        case 2: MV_LAUNCH(2); break;
        case 3: MV_LAUNCH(3); break;
        case 4: MV_LAUNCH(4); break;
        case 5: MV_LAUNCH(5); break;
        case 6: MV_LAUNCH(6); break;
        case 7: MV_LAUNCH(7); break;
        case 8: MV_LAUNCH(8); break;
        case 9: MV_LAUNCH(9); break;
        default: MV_LAUNCH(10); break;
    }
#undef MV_LAUNCH
    check("int4_matvec_grouped");
}

// IK2_MOE_GROUP_EXPERT verify MoE: group the S slots by their routed expert (device-side, one tiny
// deterministic build launch shared by gate_up and down), then run the grouped matvec with each
// expert's weights read once per 16-row tile. Wp/Wh/WS here are the FULL per-expert tables [E]
// (not per-slot gathered). Bit-exact vs int4_matvec_grouped; see k_mv_grouped_exp.
void int4_moe_group_build(const int* top, int S, int* gcnt, int* gslots, int* gexp, int* ngrp,
                          cudaStream_t s) {
    if (S <= 0 || S > 64) return;   // the verify-fast guard guarantees S <= 64
    int4w::k_moe_group_build<<<1, 32, 0, s>>>(top, S, gcnt, gslots, gexp, ngrp);
    check("int4_moe_group_build");
}
void int4_matvec_grouped_exp(const uint8_t* const* tabWp, const uint8_t* const* tabWh,
                             const float* const* tabWS, int M, int K, const int8_t* Xq,
                             const float2* Xd, int S, float* out, int stride_dst, float inv,
                             cudaStream_t s, int xq_stride, int xd_stride,
                             const int* gcnt, const int* gslots, const int* gexp, const int* ngrp) {
    if (S <= 0 || S > 64 || M % 16 != 0 || K % 256 != 0) return;
    if (xq_stride < 0) xq_stride = 0;
    if (xd_stride < 0) xd_stride = 0;
    const int nw_auto = K / 256 < int4w::MV_NW ? K / 256 : int4w::MV_NW;
    // Same IK2_MV_NW occupancy override as int4_matvec_grouped (bit-exact: disjoint sb slices).
    static const int nw_env = getenv("IK2_MV_NW") ? std::atoi(getenv("IK2_MV_NW")) : 0;
    int nw = nw_auto;
    if (nw_env >= 1 && nw_env <= int4w::MV_NW) nw = nw_env;
    // Expert-major grid (ft fastest): one expert's whole matrix streams through consecutively
    // launched blocks; grid.y = S is the fixed worst-case group count (empty groups early-exit).
    const dim3 grid(M / 16, S);
#define MVG_LAUNCH(NWV) int4w::k_mv_grouped_exp<NWV><<<grid, dim3(32, NWV), 0, s>>>( \
        tabWp, tabWh, tabWS, inv, Xq, Xd, out, M, K, stride_dst, xq_stride, xd_stride, \
        gcnt, gslots, gexp, ngrp)
    switch (nw) {
        case 1: MVG_LAUNCH(1); break;
        case 2: MVG_LAUNCH(2); break;
        case 3: MVG_LAUNCH(3); break;
        case 4: MVG_LAUNCH(4); break;
        case 5: MVG_LAUNCH(5); break;
        case 6: MVG_LAUNCH(6); break;
        case 7: MVG_LAUNCH(7); break;
        case 8: MVG_LAUNCH(8); break;
        case 9: MVG_LAUNCH(9); break;
        default: MVG_LAUNCH(10); break;
    }
#undef MVG_LAUNCH
    check("int4_matvec_grouped_exp");
}

// Phase-D fused grouped decode matvec (rot_q8x prologue inside k_mv_grouped_rot). H16 == null:
// shared f32 activation X[K] (T=1 gate_up). H16 != null: per-slot f16 H16[S][F] zero-padded to K
// (down side, replaces prep_down + rot_q8x). Requires K/256 <= MV_NW (one block per wave).
void int4_matvec_grouped_rot(const uint8_t* const* Wp, const uint8_t* const* Wh, const float* const* WS,
                             int M, int K, const float* X, const uint16_t* H16, int F, int S,
                             float* out, int stride_dst, float inv, cudaStream_t s) {
    if (S <= 0 || M % 16 != 0 || K % 256 != 0) return;
    const int nw = K / 256;
    if (nw > int4w::MV_NW) return;
    const dim3 grid(M / 16, S);
#define MVR_LAUNCH(NWV, SRCV) int4w::k_mv_grouped_rot<NWV, SRCV><<<grid, dim3(32, NWV), 0, s>>>(Wp, Wh, WS, inv, X, H16, F, out, M, K, stride_dst)
    if (H16) {
        switch (nw) {
            case 1: MVR_LAUNCH(1, 1); break;
            case 2: MVR_LAUNCH(2, 1); break;
            case 3: MVR_LAUNCH(3, 1); break;
            case 4: MVR_LAUNCH(4, 1); break;
            case 5: MVR_LAUNCH(5, 1); break;
            case 6: MVR_LAUNCH(6, 1); break;
            case 7: MVR_LAUNCH(7, 1); break;
            case 8: MVR_LAUNCH(8, 1); break;
            case 9: MVR_LAUNCH(9, 1); break;
            default: MVR_LAUNCH(10, 1); break;
        }
    } else {
        switch (nw) {
            case 1: MVR_LAUNCH(1, 0); break;
            case 2: MVR_LAUNCH(2, 0); break;
            case 3: MVR_LAUNCH(3, 0); break;
            case 4: MVR_LAUNCH(4, 0); break;
            case 5: MVR_LAUNCH(5, 0); break;
            case 6: MVR_LAUNCH(6, 0); break;
            case 7: MVR_LAUNCH(7, 0); break;
            case 8: MVR_LAUNCH(8, 0); break;
            case 9: MVR_LAUNCH(9, 0); break;
            default: MVR_LAUNCH(10, 0); break;
        }
    }
#undef MVR_LAUNCH
    check("int4_matvec_grouped_rot");
}

void int4_rot_pack8(const float* X, int N, int K, uint8_t* x4, uint8_t* xm, float* xr, cudaStream_t s) {
    if (N <= 0 || K % 256 != 0) return;
    const int Npad = ((N + int4w::TILE_N - 1) / int4w::TILE_N) * int4w::TILE_N;
    const size_t plane = (size_t) Npad * K / 2;
    const size_t smem = (size_t) K * 2 + (K / 256) * 4;
    static size_t attr_smem = 0;
    if (smem > attr_smem) { cudaFuncSetAttribute(int4w::k_rot_pack8, cudaFuncAttributeMaxDynamicSharedMemorySize, (int) smem); attr_smem = smem; }
    int4w::k_rot_pack8<<<N, 256, smem, s>>>(X, K, x4, x4 + plane, xm, xr);
    check("int4_rot_pack8");
}
void int4_gemm8(const uint8_t* Wp, const uint8_t* Wh, const float* WS, int M, int K,
                const uint8_t* x4, const uint8_t* xm, const float* xr, int N, int gw, int wn,
                float* out, const float* res, int sh, cudaStream_t s, int ldy) {
    if (N <= 0 || M % int4w::TILE_M != 0 || K % 256 != 0) return;
    const int ld = ldy > 0 ? ldy : M;
    const int Npad = ((N + int4w::TILE_N - 1) / int4w::TILE_N) * int4w::TILE_N;
    const size_t plane = (size_t) Npad * K / 2;
    const uint8_t* xh = x4; const uint8_t* xl = x4 + plane;
    const dim3 grid(M / int4w::TILE_M, Npad / (wn * int4w::NJ * 16));
#define I4_G8(GW, WN) int4w::k_gemm8w<GW, WN><<<grid, WN * 32, 0, s>>>(Wp, Wh, WS, xh, xl, xm, xr, out, M, N, K, res, sh, ld)
    if (wn == 4) { if (gw == 1) I4_G8(1, 4); else if (gw == 2) I4_G8(2, 4); else I4_G8(4, 4); }
    else { if (gw == 1) I4_G8(1, 8); else if (gw == 2) I4_G8(2, 8); else I4_G8(4, 8); }
#undef I4_G8
    check("int4_gemm8");
}
void int4_gemm8_moe_masked(const uint8_t* Wp, const uint8_t* Wh, const float* WS,
                           size_t wp_stride, size_t wh_stride, int ws_stride, int E,
                           const int* row_expert, int M, int K, const uint8_t* x4,
                           const uint8_t* xm, const float* xr, int N, int gw, int wn_,
                           float* out, const float* res, int sh, cudaStream_t s, int ldy,
                            const int* active_expert, const int* active_count) {
    if (N <= 0 || M % int4w::TILE_M != 0 || K % 256 != 0) return;
    const int ld = ldy > 0 ? ldy : M;
    const int Npad = ((N + int4w::TILE_N - 1) / int4w::TILE_N) * int4w::TILE_N;
    const size_t plane = (size_t) Npad * K / 2;
    const uint8_t* xh = x4; const uint8_t* xl = x4 + plane;
    const int planes = active_expert ? (M < E ? M : E) : E;
    const dim3 grid(M / int4w::TILE_M, Npad / (wn_ * int4w::NJ * 16), planes);
#define I4_G8M(GW, WN) int4w::k_gemm8w_moe_masked<GW, WN><<<grid, WN * 32, 0, s>>>(Wp, Wh, WS, wp_stride, wh_stride, ws_stride, E, row_expert, xh, xl, xm, xr, out, M, N, K, res, sh, ld, active_expert, active_count)
    if (wn_ == 4) { if (gw == 1) I4_G8M(1, 4); else if (gw == 2) I4_G8M(2, 4); else I4_G8M(4, 4); }
    else { if (gw == 1) I4_G8M(1, 8); else if (gw == 2) I4_G8M(2, 8); else I4_G8M(4, 8); }
#undef I4_G8M
    check("int4_gemm8_moe_masked");
}
void int4_rot_q8x(const float* X, int N, int K, int8_t* Xq, float2* Xd, cudaStream_t s) {
    if (N <= 0 || K % 256 != 0) return;
    int4w::k_rot_q8x<<<dim3(K / 256, N), 32, 0, s>>>(X, Xq, Xd, K);
    check("int4_rot_q8x");
}
void int4_mid(const float* gu, int S, int F, int Kpad, float gcap, float ucap, int8_t* Q, float2* D, cudaStream_t s) {
    if (S <= 0 || Kpad % 256 != 0) return;
    int4w::k_int4_mid<<<dim3(Kpad / 256, S), 32, 0, s>>>(gu, S, F, Kpad, gcap, ucap, Q, D);
    check("int4_mid");
}
void int4_matvec(const uint8_t* Wp, const uint8_t* Wh, const float* WS, int M, int K,
                 const int8_t* Xq, const float2* Xd, int N, float* out, int stride_dst,
                 const float* xb, float inv, cudaStream_t s) {
    if (N <= 0 || M % 16 != 0 || K % 256 != 0) return;
    // IK2_MV_DENSE_NW overrides the wave count for the single-matrix (dense proj) matvec, like IK2_MV_NW
    // does for the grouped MoE. NC=8 register pressure may make fewer waves faster. Default 0 = MV_NW.
    static const int dnw_env = getenv("IK2_MV_DENSE_NW") ? std::atoi(getenv("IK2_MV_DENSE_NW")) : 0;
    int dnw = int4w::MV_NW;
    if (dnw_env >= 1 && dnw_env <= int4w::MV_NW) dnw = dnw_env;
    const dim3 grid(M / 16);
#define I4_MV(NC, NWV) int4w::k_mv<NC, NWV><<<grid, dim3(32, NWV), 0, s>>>(Wp, Wh, WS, inv, Xq, Xd, out, xb, M, K, stride_dst)
#define I4_MV_N(NC) do { switch (dnw) { \
    case 1: I4_MV(NC,1); break; case 2: I4_MV(NC,2); break; case 3: I4_MV(NC,3); break; case 4: I4_MV(NC,4); break; \
    case 5: I4_MV(NC,5); break; case 6: I4_MV(NC,6); break; case 7: I4_MV(NC,7); break; case 8: I4_MV(NC,8); break; \
    case 9: I4_MV(NC,9); break; default: I4_MV(NC,10); break; } } while (0)
    switch (N) { case 1: I4_MV_N(1); break; case 2: I4_MV_N(2); break; case 3: I4_MV_N(3); break; case 4: I4_MV_N(4); break;
                 case 5: I4_MV_N(5); break; case 6: I4_MV_N(6); break; case 7: I4_MV_N(7); break; default: I4_MV_N(8); break; }
#undef I4_MV_N
#undef I4_MV
    check("int4_matvec");
}
// IK2_MV_LDS (default OFF): int4_matvec with the activation staged in LDS (k_mv_lds). Dynamic smem =
// NC*K + NC*(K/32)*8 bytes. Same IK2_MV_DENSE_NW wave-count knob and NC 1..8 x NW 1..10 dispatch as
// int4_matvec. Host guard: dynamic smem stays <= 48KB (no MaxDynamicSharedMemorySize opt-in needed on
// any accepted shape) and smem + the static red[NW][NC][16] tile (<= 5120B) fits the 64KB gfx1100 LDS
// budget; larger shapes fall back to int4_matvec. Bit-exact vs int4_matvec.
void int4_matvec_lds(const uint8_t* Wp, const uint8_t* Wh, const float* WS, int M, int K,
                     const int8_t* Xq, const float2* Xd, int N, float* out, int stride_dst,
                     const float* xb, float inv, cudaStream_t s) {
    if (N <= 0 || M % 16 != 0 || K % 256 != 0) return;
    static const int dnw_env = getenv("IK2_MV_DENSE_NW") ? std::atoi(getenv("IK2_MV_DENSE_NW")) : 0;
    int dnw = int4w::MV_NW;
    if (dnw_env >= 1 && dnw_env <= int4w::MV_NW) dnw = dnw_env;
    const int nc = N < 8 ? N : 8;
    const size_t smem = (size_t) nc * K + (size_t) nc * (K / 32) * 8;
    if (smem > 48 * 1024 || smem + 5120 > 64 * 1024) {
        int4_matvec(Wp, Wh, WS, M, K, Xq, Xd, N, out, stride_dst, xb, inv, s);
        return;
    }
    const dim3 grid(M / 16);
#define I4_MVL(NC, NWV) int4w::k_mv_lds<NC, NWV><<<grid, dim3(32, NWV), smem, s>>>(Wp, Wh, WS, inv, Xq, Xd, out, xb, M, K, stride_dst)
#define I4_MVL_N(NC) do { switch (dnw) { \
    case 1: I4_MVL(NC,1); break; case 2: I4_MVL(NC,2); break; case 3: I4_MVL(NC,3); break; case 4: I4_MVL(NC,4); break; \
    case 5: I4_MVL(NC,5); break; case 6: I4_MVL(NC,6); break; case 7: I4_MVL(NC,7); break; case 8: I4_MVL(NC,8); break; \
    case 9: I4_MVL(NC,9); break; default: I4_MVL(NC,10); break; } } while (0)
    switch (N) { case 1: I4_MVL_N(1); break; case 2: I4_MVL_N(2); break; case 3: I4_MVL_N(3); break; case 4: I4_MVL_N(4); break;
                 case 5: I4_MVL_N(5); break; case 6: I4_MVL_N(6); break; case 7: I4_MVL_N(7); break; default: I4_MVL_N(8); break; }
#undef I4_MVL_N
#undef I4_MVL
    check("int4_matvec_lds");
}
// IK2_INT4_MV_ALT: one fused-projection matvec where row ranges take different activations
// (msg branch q|k|v|r,gate over one wqkvrg sidecar). Same wave-count knob as int4_matvec.
// Boundaries must be 16-aligned and ordered; a violation is a wiring bug - fail loudly.
void int4_matvec_alt(const uint8_t* Wp, const uint8_t* Wh, const float* WS, int M, int K,
                     const int8_t* Xq0, const float2* Xd0, const int8_t* Xq1, const float2* Xd1,
                     const int8_t* Xq2, const float2* Xd2, int b0, int b1, int b2,
                     int N, float* out, int stride_dst, float inv, cudaStream_t s) {
    if (N <= 0 || M % 16 != 0 || K % 256 != 0) return;
    if ((b0 & 15) || (b1 & 15) || (b2 & 15) || b0 < 0 || b0 > b1 || b1 > b2 || b2 > M) {
        std::fprintf(stderr, "ik2::k::int4_matvec_alt: bad boundaries %d %d %d (M %d)\n", b0, b1, b2, M);
        std::abort();
    }
    static const int dnw_env = getenv("IK2_MV_DENSE_NW") ? std::atoi(getenv("IK2_MV_DENSE_NW")) : 0;
    int dnw = int4w::MV_NW;
    if (dnw_env >= 1 && dnw_env <= int4w::MV_NW) dnw = dnw_env;
    const dim3 grid(M / 16);
#define I4_MVA(NC, NWV) int4w::k_mv_alt<NC, NWV><<<grid, dim3(32, NWV), 0, s>>>(Wp, Wh, WS, inv, Xq0, Xd0, Xq1, Xd1, Xq2, Xd2, b0, b1, b2, out, nullptr, M, K, stride_dst)
#define I4_MVA_N(NC) do { switch (dnw) { \
    case 1: I4_MVA(NC,1); break; case 2: I4_MVA(NC,2); break; case 3: I4_MVA(NC,3); break; case 4: I4_MVA(NC,4); break; \
    case 5: I4_MVA(NC,5); break; case 6: I4_MVA(NC,6); break; case 7: I4_MVA(NC,7); break; case 8: I4_MVA(NC,8); break; \
    case 9: I4_MVA(NC,9); break; default: I4_MVA(NC,10); break; } } while (0)
    switch (N) { case 1: I4_MVA_N(1); break; case 2: I4_MVA_N(2); break; case 3: I4_MVA_N(3); break; case 4: I4_MVA_N(4); break;
                 case 5: I4_MVA_N(5); break; case 6: I4_MVA_N(6); break; case 7: I4_MVA_N(7); break; default: I4_MVA_N(8); break; }
#undef I4_MVA_N
#undef I4_MVA
    check("int4_matvec_alt");
}
// IK2_INT4_DENSE: single-matrix decode matvec with the rot prologue folded in (k_mv_dense_rot).
// Only the dense projections' shapes are instantiated: K=2304 (NW=9: wqkvrg/wqrg/wk/wv) and
// K=3072 (NW=12: wo). Any other shape is a wiring bug - fail loudly, never silently skip.
void int4_dense_rot(const uint8_t* Wp, const uint8_t* Wh, const float* WS, int M, int K,
                    const float* X, float* dst, const MvConv& c0, const MvConv& c1,
                    float inv, cudaStream_t s, const MvAlt& a0, const MvAlt& a1) {
    if (M % 16 != 0 || K % 256 != 0) {
        std::fprintf(stderr, "ik2::k::int4_dense_rot: M %d K %d not tile-aligned\n", M, K);
        std::abort();
    }
    const int nw = K / 256;
    const bool alt = a0.n > 0 || a1.n > 0;
    const dim3 grid(M / 16);
#define IDR_LAUNCH(NWV, AV) int4w::k_mv_dense_rot<NWV, AV><<<grid, dim3(32, NWV), 0, s>>>(Wp, Wh, WS, inv, X, dst, c0, c1, a0, a1)
    if (nw == 9) { if (alt) IDR_LAUNCH(9, 1); else IDR_LAUNCH(9, 0); }
    else if (nw == 12) { if (alt) IDR_LAUNCH(12, 1); else IDR_LAUNCH(12, 0); }
    else {
        std::fprintf(stderr, "ik2::k::int4_dense_rot: K %d (nw %d) not instantiated\n", K, nw);
        std::abort();
    }
#undef IDR_LAUNCH
    check("int4_dense_rot");
}
// IK2_BATCH_ROTMV: batched T=1-equivalent dense decode matvec - T activation rows share each
// weight-row read (k_mv_dense_rot_rows). Same rotate/quant/sdot/reduction order per row, so
// NC=1 is bit-exact with int4_dense_rot. convs[row] applies row `row`'s mv_conv_row epilogue;
// nconv=0 (or row >= nconv) writes the raw result to dst[row*ldy + prow].
void int4_dense_rot_rows(const uint8_t* Wp, const uint8_t* Wh, const float* WS, int M, int K,
                         const float* X, int T, float* dst, int ldy, float inv, cudaStream_t s,
                         const MvConv* convs, int nconv) {
    if (M % 16 != 0 || K % 256 != 0 || T < 1 || T > 8) {
        std::fprintf(stderr, "ik2::k::int4_dense_rot_rows: M %d K %d T %d bad shape\n", M, K, T);
        std::abort();
    }
    if (nconv < 0) nconv = 0;
    if (nconv > T) nconv = T;
    const int nw = K / 256;
    const dim3 grid(M / 16);
#define IDR_ROWS_LAUNCH(NC, NWV) int4w::k_mv_dense_rot_rows<NC, NWV, 0><<<grid, dim3(32, NWV), 0, s>>>(Wp, Wh, WS, inv, X, K, T, dst, ldy, convs, nconv)
#define IDR_ROWS_NW(NC) do { \
        if (nw == 9) IDR_ROWS_LAUNCH(NC, 9); \
        else if (nw == 12) IDR_ROWS_LAUNCH(NC, 12); \
        else { std::fprintf(stderr, "ik2::k::int4_dense_rot_rows: K %d (nw %d) not instantiated\n", K, nw); std::abort(); } \
    } while (0)
    switch (T) {
        case 1: IDR_ROWS_NW(1); break;
        case 2: IDR_ROWS_NW(2); break;
        case 3: IDR_ROWS_NW(3); break;
        case 4: IDR_ROWS_NW(4); break;
        case 5: IDR_ROWS_NW(5); break;
        case 6: IDR_ROWS_NW(6); break;
        case 7: IDR_ROWS_NW(7); break;
        default: IDR_ROWS_NW(8); break;
    }
#undef IDR_ROWS_NW
#undef IDR_ROWS_LAUNCH
    check("int4_dense_rot_rows");
}
#else
void int4_rot_pack8(const float*, int, int, uint8_t*, uint8_t*, float*, cudaStream_t) {}
void int4_gemm8(const uint8_t*, const uint8_t*, const float*, int, int, const uint8_t*, const uint8_t*, const float*, int, int, int, float*, const float*, int, cudaStream_t, int) {}
void int4_rot_q8x(const float*, int, int, int8_t*, float2*, cudaStream_t) {}
void int4_mid(const float*, int, int, int, float, float, int8_t*, float2*, cudaStream_t) {}
void int4_matvec(const uint8_t*, const uint8_t*, const float*, int, int, const int8_t*, const float2*, int, float*, int, const float*, float, cudaStream_t) {}
void int4_matvec_lds(const uint8_t*, const uint8_t*, const float*, int, int, const int8_t*, const float2*, int, float*, int, const float*, float, cudaStream_t) {}
void int4_matvec_alt(const uint8_t*, const uint8_t*, const float*, int, int, const int8_t*, const float2*, const int8_t*, const float2*, const int8_t*, const float2*, int, int, int, int, float*, int, float, cudaStream_t) {}
void int4_matvec_grouped(const uint8_t* const*, const uint8_t* const*, const float* const*, int, int, const int8_t*, const float2*, int, float*, int, float, cudaStream_t, int, int) {}
void int4_moe_group_build(const int*, int, int*, int*, int*, int*, cudaStream_t) {}
void int4_matvec_grouped_exp(const uint8_t* const*, const uint8_t* const*, const float* const*, int, int, const int8_t*, const float2*, int, float*, int, float, cudaStream_t, int, int, const int*, const int*, const int*, const int*) {}
void int4_matvec_grouped_rot(const uint8_t* const*, const uint8_t* const*, const float* const*, int, int, const float*, const uint16_t*, int, int, float*, int, float, cudaStream_t) {}
void int4_dense_rot(const uint8_t*, const uint8_t*, const float*, int, int, const float*, float*, const MvConv&, const MvConv&, float, cudaStream_t, const MvAlt&, const MvAlt&) {}
void int4_dense_rot_rows(const uint8_t*, const uint8_t*, const float*, int, int, const float*, int, float*, int, float, cudaStream_t, const MvConv*, int) {}
void int4_gather_slots(const int*, const uint8_t* const*, const uint8_t* const*, const float* const*, int, const uint8_t**, const uint8_t**, const float**, cudaStream_t) {}
void int4_prep_down(const uint16_t*, float*, int, int, int, cudaStream_t) {}
// IK2_DRAFT_CAND stubs: the drafter gates its calls on the same build flag, so these are never used;
// they keep the symbol set identical to the gfx11 build for other translation units.
void draft_base_topk_part(const float*, int, const uint16_t*, int, int, float*, int*, cudaStream_t) {}
void draft_base_topk_reduce(const float*, const int*, int, int, int*, float*, cudaStream_t) {}
void draft_cand_argmax(const uint16_t*, int, const float*, const float*, const uint16_t*, int, int, const int*, int, int*, cudaStream_t) {}
#endif

}  // namespace ik2::k
