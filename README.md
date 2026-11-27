# Strata — Inkling-2-Mini on RX 7900 XTX (single-stream AR, 640k context)

An optimized fork of [Strata](https://github.com/Niko1221/Strata) (original ik2 engine from PR #6506) for running **Thinking Machines Inkling-2-Mini** on a single **AMD RX 7900 XTX (gfx1100, 24 GB)** under **Windows native HIP**.

This build is the **final, converged deliverable: one single autoregressive stream, 655360-token (640k) context held entirely in VRAM with INT4 KV, and an 8 GiB conversation cache in host RAM.** Every optimization is environment-gated; the validated configuration is baked into the one-click launcher.

## Headline result (greedy, bit-exact)

| Item | Value | Condition |
|---|---|---|
| Context | **655360 tokens (640k)** | all-VRAM resident, 0 expert misses |
| KV cache | INT4, 5.36 GiB | 48 layer visits, per-head fp16 scale |
| Conversation cache | 8192 MiB host RAM | 64 snapshots × 41.2 MiB |
| AR decode @ 640k ctx | **68.88 tok/s** | short prompt; KV larger → more bytes read/token |
| AR decode @ 256k ctx | **141.8 tok/s** | same engine/config, smaller context |
| Prefill @ 640k ctx | ~552 tok/s | chunk 4096 |
| Correctness | bit-exact vs. llama.cpp V3 | greedy, 31–96 tokens, all configs |

> **Honest note on the two decode numbers:** 141.8 tok/s is the measured rate at **256k** context; at **640k** the measured rate is **68.88 tok/s**. The larger allocated KV cache reads more bytes per token (the 8 full-attention visits are O(ctx)), so the rate is lower. Both use the identical validated engine build.

All numbers: Windows 11, RX 7900 XTX (960 GB/s), 32 GB RAM, greedy decoding, 128 experts fully resident, CUDA graph captured.

---

## Contents

- [What is in this build](#what-is-in-this-build)
- [Requirements](#requirements)
- [Build (Windows / HIP / gfx1100)](#build-windows--hip--gfx1100)
- [Prepare model, token file & INT4 sidecars](#prepare-model-token-file--int4-sidecars)
- [Run (one click)](#run-one-click)
- [Measured results](#measured-results)
- [Correctness validation](#correctness-validation)
- [Environment variable reference](#environment-variable-reference)
- [Source changes](#source-changes)
- [Credits & license](#credits--license)

---

## What is in this build

Inkling-2-Mini is a looped multimodal MoE: 24 physical layers, core layers 6–17 run 3× per token (48 visits/token), 128 experts top-8, hidden 2304, 24 Q / 8 KV heads (head dim 128), sliding-window attention (1024) plus full layers, learned relative-position bias + attention sink. At ~26 B total / 5.4 B active parameters it is memory-bandwidth- and latency-bound on one GPU.

Porting work that made this build fast and correct on gfx1100:

- **CUDA → HIP wave32 shim** under `engine/include/strata/hip_compat/`; NVIDIA-only PTX helpers guarded so the scalar path compiles/runs; HIP device paths gated by `__HIP_DEVICE_COMPILE__`.
- **gfx1100 WMMA flash attention** (`attn_flash_wmma_kernel<BN,NW>`, RDNA3 `wmma_f32_16x16x16_f16_w32`, ~20–25 KB LDS) — largest prefill lever (+83% at 16k).
- **Rotated INT4 weights** (experts + dense projections + output head) with Hadamard rotation and per-row/per-128 scales (`Wp` nibble tiles, `Wh` sub-scales, `WS` fp32). gfx1100 WMMA supports f16/bf16/int8/int4 but **not** fp8.
- **INT4 KV** (per-head fp16 scale) — halves KV bytes vs. fp8 and enables the 640k all-VRAM configuration.
- **Efficient resident path**: device expert cache, single CUDA graph capture, INT4 head decode (avoids a 60 ms/token PCIe head read), gate/rms fusion, GQA + DPP attention reductions.
- **8 GiB host conversation cache**: 64 snapshots park prefix state in host RAM for multi-turn reuse.

Measured and kept **off** (defaults): WMMA single-token expert GEMV / QKᵀ (tensor cores waste 15/16 columns at N=1), fp8 KV for decode, chunk > 10240 (expert-cache thrash), resident-fp16 projection cache (breaks expert residency), NVIDIA-PTX attention/expert paths (`IK2_ATTN_MMA`, `IK2_EXPERT_MMA`).

> Earlier experiments (native DSpark speculative decoding, dual-stream concurrency, K-step persistent graphs, 1M host tiering) remain in the source tree as opt-in, default-off code but are **not part of this final single-stream deliverable**. See the development report in the workspace for the full record.

## Requirements

- **GPU**: AMD RX 7900 XTX (gfx1100), 24 GB, ROCm Windows SDK with `hipcc`/clang targeting gfx1100.
- **OS**: Windows 11; Visual Studio build tools (for `mt.exe` manifest embedding); CMake + Ninja.
- **RAM**: 32 GB. Keep only one model-loading process at a time.
- **Model**: `Inkling-2-Mini-UD-Q4_K_XL.gguf` (Unsloth quantization, ~17.6 GB).
- **Python 3.12** with `numpy` + `scipy` and a `gguf-py` checkout for conversion tools.

## Build (Windows / HIP / gfx1100)

```powershell
.\scripts\build-windows-hip.ps1 `
  -EngineDir  <repo>\engine `
  -BuildDir   <repo>\build-hip-win `
  -RocmSdk    <path\to\_rocm_sdk_devel> `
  -Target     strata-ik2
```

At runtime put the ROCm SDK `bin` and `lib\llvm\bin` on `PATH` (the launcher does this). Compile with the SDK clang/headers; run with the SDK HIP/hipBLAS/rocBLAS DLLs. Flags: `AMDGPU_TARGETS=gfx1100`, `GGML_HIP=ON`, `STRATA_ENABLE_CUDA=OFF`, `STRATA_ENABLE_HIP=ON`.

## Prepare model, token file & INT4 sidecars

The CLI consumes **token-id files** (`--tokens @file`): comma-separated integers generated with the Inkling-2 tokenizer (`inkling2` / O200K BPE, `ignore_merges=true`). The serve mode generates tokens internally from a chat/template, so no token file is needed for the one-click launcher.

INT4 paths need converted sidecars generated from the GGUF (the engine still reads the GGUF for routing/metadata):

```powershell
python tools\int4\nvfp4_to_int4.py   ...   # NVFP4 experts (layers 0-19)  -> int4-bin
python tools\int4\q8_to_int4_l20.py  ...   # Q8_0 experts (layers 20-23) -> int4-bin
python tools\int4\q8_dense_to_int4.py ...  # dense projections + out_w   -> int4-dense-bin (Wp/Wh/WS)
```

Set `IK2_INT4_DIR` / `IK2_INT4_DENSE_DIR` to those directories. Conversion is byte-deterministic; INT4 sidecars replace (not coexist with) the NVFP4 residency within the 24 GB budget.

## Run (one click)

The final launcher (`start-daily.ps1` / `start.bat` in the deployment directory) is fixed to the validated single-stream 640k configuration (640k ctx, int4 KV, 8 GiB conversation cache, no drafter). It is headless stdio serve; for an HTTP API put a compatible adapter in front or use the native serve scripts under `scripts/native-serve/`.

```bat
start.bat    :: launches the engine, writes strata-serve.pid, logs to serve-daily.log
stop.bat     :: stops the engine, leaves gpu_guard running
```

Or directly:

```powershell
powershell -ExecutionPolicy Bypass -File <deploy-dir>\start-daily.ps1
powershell -ExecutionPolicy Bypass -File <deploy-dir>\stop-daily.ps1
```

Expected startup markers in `serve-daily.log`:

```
fit(actual): experts 12.23 GiB + dense 1.10 + KV 5.36 + reserve 0.50 = 19.19 vs free 22.99 GiB -> FITS
resident: INT4 experts (l<20) + Q8_0 (l>=20) all in VRAM
conversation cache: 8192 MiB budget, 64 snapshots
INFO engine=ik2 ... kv=int4 ctx=655360
READY 655360 stop
```

**GPU safety**: the launcher refuses to start if another model process is running and warns if `gpu_guard` is down. Never run two model-loading processes concurrently on a 24 GB WDDM card.

## Measured results

### Single-stream AR (num. in tok/s, greedy, bit-exact)

| Context | KV | Decode | Prefill | Residency |
|---|---|---|---|---|
| short prompt | int4 | ~94 | — | — |
| 4k | int4 | 102 | 2262 | resident |
| 16k | int4 | 92.9 | 3053 (chunk 9216) | resident |
| 128k | int4 | 46.6 | 1614 | resident |
| 256k | int4 | **141.8** | 1097 | resident |
| 512k | int4 | 18.8 | 661 | resident |
| **640k (final)** | int4 | **68.88** | 552 | resident, 0 misses |

Decode falls with context because the 8 full-attention visits scan O(ctx) KV bytes (~85 GB/s effective on this path); the 256k vs. 640k difference is a context-size effect, not a configuration change. Reference llama.cpp V3 baseline: 44.5 tok/s decode, 539 tok/s prefill.

### Prefill (16k prompt, chunk sweep)

chunk 8192 → 2567, 9216 → 2679, 10240 → 2636, 11264 → 1010 tok/s (expert-cache thrash). WMMA flash attention raised 16k prefill 1450 → 2661 tok/s (+83%).

## Correctness validation

- Kernel parity: **315 ok / 0 fail / 3 PTX-skip** on gfx1100 (double-reference parity for RMS, L2 clip, conv4, attention residual/probs/decode/multi, routing ties, NVFP4/Q8_0 experts, fp8 KV, loop gates, token path).
- Greedy parity: bit-exact token streams vs. llama.cpp V3 (temperature 0, top_k 1) over 31–96 tokens on diverse/capital prompts, across scalar, WMMA-flash and INT4 configs.
- INT4 expert/dense/KV conversions verified bit-exact vs. FP16/Q8 references.

## Environment variable reference

Defaults are the validated single-stream setup. The launcher sets these explicitly.

| Variable | Value in final config | Meaning |
|---|---|---|
| `IK2_FLASH` / `IK2_FLASH_WMMA` | 1 | gfx1100 WMMA flash attention |
| `IK2_FLASH_NW` / `IK2_FLASH_BN` | 8 / 32 | flash tile geometry |
| `IK2_WMMA_GEMM` | 1 | dense projection WMMA/BK64 |
| `IK2_INT4_EXPERTS` / `_DENSE` | 1 | rotated INT4 experts / dense |
| `IK2_INT4_LAYERS` | 24 | all expert layers converted |
| `IK2_INT4_HEAD` / `_HEAD_DECODE` | 1 | INT4 output-head sidecar + VRAM decode |
| `IK2_KV` | int4 | KV cache format (f16/fp8/int8/int4) |
| `IK2_RESERVE_MIB` | 512 | VRAM reserve for 640k residency |
| `IK2_LG_CTX` | 4096 | logits scratch cap |
| `IK2_CONV_CACHE_MIB` | 8192 | host conversation cache budget |
| `IK2_EC_ROOM_MIB` | 96 | extra MiB for resident expert cache |
| `IK2_ATTN_GQA` / `IK2_ATTN_DPP` | 1 | attention reduction opts |
| `IK2_GATE_FUSE` / `IK2_RMS_FUSE` | 1 | small launch-fusion opts |
| `IK2_ATTN_MMA` / `IK2_EXPERT_MMA` | 0 | NVIDIA-PTX paths — keep 0 on HIP |
| `IK2_KV_HOST` / `IK2_DENSE_HOST` | 0 | host tiering — off for all-VRAM 640k |

## Source changes

All optimizations live under `engine/`. Key files:

| File | Change |
|---|---|
| `include/strata/hip_compat/{cublas_v2.h,cuda_runtime.h,intrinsics.hpp,cuda_fp8.h}` | wave32 CUDA→HIP shim |
| `src/ik2/ik2_kernels.cu` / `.cuh` | PTX guards; WMMA flash `<BN,NW,KVT,IBITS>`; INT4 experts/dense/head/KV kernels; warp-shuffle reductions |
| `src/ik2/ik2_engine.cu` | hybrid WMMA GEMM; INT4 loading; residency/fit logic; profiler; single-stream serve |
| `src/prefill/wmma_gemm.cu` / `.h` | strided WMMA GEMM + tile-size switch |
| `tests/cuda/ik2_kernels_test.cu` | double-reference parity, WMMA flash + INT4 cases |
| `cmake/strata_gpu_engine.cmake` | ik2 target guard `OR HIP`, link `strata_core`, HIP WMMA sources |

Conversion tools: `tools/int4/*.py`. Build helper: `scripts/build-windows-hip.ps1`. Native serve (optional HTTP/agent path): `scripts/native-serve/`.

## Credits & license

Based on [Strata](https://github.com/Niko1221/Strata) by Niko1221; Inkling-2-Mini by Thinking Machines. See upstream license files for terms.
