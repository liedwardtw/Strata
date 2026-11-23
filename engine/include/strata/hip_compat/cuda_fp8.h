#pragma once
// Included only by STRATA_ENABLE_HIP builds. CUDA builds use NVIDIA's cuda_fp8.h.
// ik2's fp8 helpers (f32_to_e4m3fn / e4m3fn_to_f16 etc.) gate the __nv_cvt_*
// intrinsics behind __CUDA_ARCH__ >= 890, which is undefined under HIP, so they
// take their software bit-twiddle branch and never call these. This header only
// satisfies the #include <cuda_fp8.h> and names the types for that dead branch.
#if defined(__HIPCC__)
#include <hip/hip_fp8.h>
#define __nv_fp8_storage_t __hip_fp8_storage_t
#define __nv_fp8x2_storage_t __hip_fp8x2_storage_t
#define __nv_fp8_interpretation_t __hip_fp8_interpretation_t
#define __NV_E4M3 __HIP_E4M3
#define __NV_E5M2 __HIP_E5M2
#define __NV_SATFINITE __HIP_SATFINITE
#define __nv_cvt_float_to_fp8 __hip_cvt_float_to_fp8
#define __nv_cvt_float2_to_fp8x2 __hip_cvt_float2_to_fp8x2
#define __nv_cvt_fp8_to_halfraw __hip_cvt_fp8_to_halfraw
#define __nv_cvt_fp8x2_to_halfraw2 __hip_cvt_fp8x2_to_halfraw2
#endif
