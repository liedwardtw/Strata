# RX 7900 XTX — optimization log (ik2 engine)

What we changed in the Strata ik2 engine to run Inkling-2-Mini fast on gfx1100 (RDNA3). Each item is environment-gated (`IK2_*`), default-off where it changes numerics, and was kept only after A/B token-stream (greedy bit-exact) verification.

## 0. Engine selection

Surveyed alternatives before committing:
- **sglang-radeon**: dead — `sgl_kernel` gfx942-only wheel, fused_moe faults.
- **vllm-radiance**: dead — gfx1201/RDNA4 only, W4A8 needs RDNA4 FP8 WMMA.
- **hipfire** (user's fork): no Inkling arch, no NVFP4 dequant; demoted to fallback.
- **Avalon** (llama.cpp fork): tuned for Qwen3.8-Flash-Next, expert cache low-value when weights are VRAM-resident.
- **Strata ik2** (Niko1221/Strata PR #6506): NVIDIA 3090 120–124 tok/s, bit-identical; chosen as base and ported to HIP.

## 1. CUDA → HIP wave32 shim

`engine/include/strata/hip_compat/` maps cuBLAS/runtime/intrinsics to HIP; NVIDIA-only PTX helpers (mma16816 / wgmma / cp.async / ldsm) are guarded to trap stubs; device paths gated by `__HIP_DEVICE_COMPILE__`. `ik2_kernels_test`: 315 ok / 0 fail / 3 PTX-skip.

## 2. gfx1100 WMMA flash attention — largest prefill lever

NVIDIA `attn_flash_f16_kernel` needs 88064 B dynamic shared memory > gfx1100's 64 KB LDS, so it cannot run. Rewrote as `attn_flash_wmma_kernel<BN,NW>` with 4 waves, BN=32, ~20 KB LDS using `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`. Kept GQA-shared K/V and bias/rel-pos in log2 domain. Result: 16K prefill 1450 → 2661 tok/s (+83%). `IK2_FLASH_WMMA=1`, `IK2_FLASH_NW=8`, `IK2_FLASH_BN=32`.

## 3. Rotated INT4 weights

Offline convert NVFP4 (layers 0–19) and Q8_0 (layers 20–23) experts, dense projections, and output head to **rotated int4** (Hadamard H256 rotation + per-128 scales; `Wp` nibble tiles / `Wh` sub-scales / `WS` fp32 row scales). Decode kernels: `k_gemm8w` (WMMA iu4) for prefill, `k_mv_dense_rot` for single-token matvec with conv epilogue, `k_mv_grouped` for MoE. gfx1100 WMMA supports f16/bf16/int8/int4 but **not fp8**. Conversion tools in `tools/int4/`. `IK2_INT4_EXPERTS=1`, `IK2_INT4_DENSE=1`, `IK2_INT4_LAYERS=24`.

## 4. INT4 KV

RDNA3 has no FP8 WMMA. INT4 KV (K+V int4 codes + per-(token,KV-head) fp16 scale, ~1056 B/token/visit) halves bytes vs fp8 and makes 640K all-VRAM possible. Decode is scalar (bandwidth/latency bound), so dequant happens in the load path with hardware `sdot4`. `IK2_KV=int4`.

## 5. Attention decode reductions

- **GQA-shared** (`attn_split_int_gqa_kernel`, grid `(Hkv,P)`): each K/V head walked once and shared across R=3 Q heads → 256K decode 31.84 → 41.14 tok/s (+28%). `IK2_ATTN_GQA=1`.
- **DPP warp reduction**: 5-step `__shfl_xor_sync` → `permlanex16 + update_dpp` register reduction, +6% @ 256K, bit-exact. `IK2_ATTN_DPP=1`.
- **gate fusion** (fold output gate into attention combine epilogue, +1.5% short) and **rms_multi** (layer-0 3 norms → 1, neutral). `IK2_GATE_FUSE=1`, `IK2_RMS_FUSE=1`.

## 6. Resident path

128 experts fully resident via device pointer tables; single CUDA graph capture; expert cache sized to fit 24 GB; chunk kept ≤ 4096 at long context so work buffers don't evict resident experts. INT4 output-head sidecar avoids a ~60 ms/token PCIe head read (`IK2_INT4_HEAD=1`, `IK2_INT4_HEAD_DECODE=1`).

## 7. DSpark (native speculative decoding)

Drafter = model-native `mtp.safetensors`. Fast path: device-only verify (removes 48 D2H routing syncs/step), draft WMMA, slot-major MoE, gate-fused combine. Root-cause fix: `argmax` used device globals `g_argmax_v/i[1024]`; concurrent graphs overwrote each other → near-tie logit flips. Made argmax scratch per-Work (fresh for the cloned stream) → dual-stream bit-exact. `IK2_VERIFY_FAST=1`. See [DSPARK](XTX-7900XTX-DSPARK.md).

## 8. Serve / monitor / desktop APP

- Single-stream engine reports `batch_slots=0` so native serve uses its serialized FIFO; two concurrent requests queue instead of deadlocking (`IK2_REPORT_BATCH=0` default).
- `reasoning_effort=none` in the config = AR-direct (no thinking pass), the ~140 tok/s short-output mode; a client can still ask for thinking.
- `IK2_MONITOR_INFO=1` emits INFO keys (`expert_slots`, `expert_cache_mib`, `conversation_cache_mib/slots`) so the monitor page shows cache/VRAM correctly.
- Strata Desktop App loads ik2 as a native **shape-engine component** (`component:strata-inkling`) — the App's own Control runtime drives our engine, no external provider. `--version` prints `strata-inkling 0.1.64`; `--draft off` is a no-op.

## 9. What was measured and kept OFF (default)

WMMA single-token expert GEMV / QKᵀ (tensor cores waste 15/16 columns at N=1); fp8 KV for decode; chunk > 10240 (expert-cache thrash); resident-fp16 projection cache (breaks residency); NVIDIA-PTX attention/expert paths (`IK2_ATTN_MMA`, `IK2_EXPERT_MMA`) — these trap-stub or hang on HIP; non-bit-exact float-math relaxations (EXP2, batch-graph-exact) — see [QUALITY](XTX-7900XTX-QUALITY.md).

Dual-stream / K-step persistent graph experiments are in the source as opt-in default-off code but are **not** part of the final single-stream deliverable; see [LIMITS](XTX-7900XTX-LIMITS.md).
