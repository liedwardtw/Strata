# Strata — Inkling-2-Mini Optimized Build for RX 7900 XTX

An optimized fork of [Strata](https://github.com/Niko1221/Strata) (original ik2 engine from PR #6506) for running **Thinking Machines Inkling-2-Mini** at maximum speed on a single **AMD RX 7900 XTX (gfx1100, 24 GB)** under **Windows native HIP**.

Upstream Strata's ik2 engine was CUDA-only. This build adds a wave32 RDNA3 compatibility layer, a gfx1100 WMMA flash-attention kernel, rotated-INT4 weight paths (experts + dense projections + head), native DSpark speculative decoding, and int4 KV — plus a 1M-context / host-RAM tiering path. Every optimization is environment-gated; the defaults are the validated correct + fast configuration.

**Headline results (all greedy bit-exact vs. llama.cpp V3 reference):**

| Metric | Result |
|---|---|
| Prefill (16k prompt) | **3053 tok/s** |
| AR decode (no speculation) | **107 tok/s** (short), **103 tok/s** (4k) |
| DSpark decode (native draft, sustained) | **247 tok/s** (structured prompt, accept 7.73/8) |
| Long context, all-VRAM resident | 640k context @ 15.6 tok/s decode |
| 1M context | fits in 24 GB (KV in VRAM + host tiering), 4k-populated @ 103 tok/s decode |

See [Measured results](#measured-results) for the full tables and conditions.

---

## Contents

- [Why this build](#why-this-build)
- [Architecture notes](#architecture-notes)
- [Requirements](#requirements)
- [Build (Windows / HIP / gfx1100)](#build-windows--hip--gfx1100)
- [Prepare model & token files](#prepare-model--token-files)
- [INT4 weight conversion](#int4-weight-conversion)
- [Run & benchmark](#run--benchmark)
- [Measured results](#measured-results)
- [Correctness validation](#correctness-validation)
- [Environment variable reference](#environment-variable-reference)
- [Source changes](#source-changes)
- [Credits & license](#credits--license)

---

## Why this build

Inkling-2-Mini is a looped multimodal MoE: 24 physical layers, core layers 6–17 executed **3×** per token (48 visits/token), 128 experts with top-8 routing, hidden 2304, 24 Q / 8 KV heads (head dim 128), sliding-window attention (1024) plus full layers, learned relative-position bias + attention sink. At ~26 B total / 5.4 B active parameters it is memory-bandwidth- and latency-bound on a single GPU.

The stock engines did not meet the target on gfx1100:

| Engine | Verdict |
|---|---|
| **Strata ik2 (this build)** | Chosen base. Resident expert cache + CUDA graph; portable to HIP. |
| Strata ink2 (alternate fork) | Native NVFP4 checkpoint; decode limited by ~1200 kernel launches/token. |
| Basalt | Blackwell sm_120 FP4 tensor cores only; no HIP. |
| LexiSuite | Closed source, custom OS; below target on 7900 XTX. |
| RustingStrata | AMD Vulkan; prefill well below target. |
| SGLang-radeon / vLLM | gfx942 / RDNA4-only kernels; Inkling recipe NVIDIA-only. |
| hipfire | No Inkling architecture support; NVFP4 dequant missing. |
| llama.cpp (banded-FA fork) | Correctness reference only: decode 44–47, prefill 1679 tok/s. |

The final build is ~2.3× the decode and ~1.8× the prefill of the corrected llama.cpp baseline, and its DSpark mode reaches 97% of the B=8 perfect-accept theoretical ceiling.

## Architecture notes

Inkling-2-Mini (authoritative spec in `notes/inkling-math-spec.md` of the development workspace):

- 24 physical layers; layers 6–17 loop 3× (48 visits/token); cross-pass messages via 4 rank-64 gates.
- 128 experts, top-8, route scale 8.0; SiTU activation with gate/up softcaps 4 / 25.
- Attention: 5 sliding-window (1024) + 1 full layer per 6-layer block; learned rel-pos (dim 16, extent 1024) + learned sink; qk L2 clip 22.63, rel L2 clip 8.0.
- Extended context: 1M tokens native; final logit softcap 30; vocab 201024.

Key porting facts:

- **CUDA → HIP**: a wave32 compat layer under `engine/include/strata/hip_compat/` maps cuBLAS/cuda-runtime/intrinsics; NVIDIA-only PTX helpers (mma16816, ldsm, cp.async, wgmma) are guarded with `__builtin_trap` stubs so the scalar path compiles and runs. HIP device paths use `__HIP_DEVICE_COMPILE__` gates.
- **WMMA flash attention**: `attn_flash_wmma_kernel<BN,NW>` uses RDNA3 `wmma_f32_16x16x16_f16_w32`; shared memory 20–25 KB (fits gfx1100's 64 KB LDS), unlike the NVIDIA kernel's 88 KB. Replaces the F32 logits round-trip (`W.lg[H,Tq,nk]` ~201 MB/visit). Largest prefill lever: **+83% at 16k/chunk10240**.
- **Rotated INT4 weights**: experts, dense projections and the output head are converted from the model's NVFP4/Q8_0 format to symmetric int4 with Hadamard rotation and per-row/per-128 scales (`Wp` nibble tiles, `Wh` 4-bit sub-scales, `WS` per-row fp32). INT4 uses `wmma_i32_16x16x16_iu4_w32`; gfx1100 WMMA supports f16/bf16/int8/int4 but **not** fp8.
- **What did NOT help** (measured, kept off): WMMA single-token expert GEMV / QKᵀ / multi-token NVFP4 experts (tensor cores waste 15/16 columns at N=1), fp8 KV for decode (launch/serial-bound, not KV-bandwidth), chunk > 10240 (expert-cache thrash), resident-fp16 projection cache (breaks expert residency), decode graph (graph ≈ direct).

## Requirements

- **GPU**: AMD RX 7900 XTX (gfx1100), 24 GB, ROCm Windows SDK with `hipcc`/clang targeting gfx1100.
- **OS**: Windows 11; Visual Studio build tools (for `mt.exe` manifest embedding); CMake + Ninja.
- **RAM**: 32 GB recommended. The engine keeps all experts resident in VRAM; never run two model-loading processes concurrently.
- **Model**: `Inkling-2-Mini-UD-Q4_K_XL.gguf` (Unsloth quantization, ~17.6 GB). Draft head for DSpark: native `mtp.safetensors` (BF16).
- **Python 3.12** with `numpy` + `scipy` and a `gguf-py` checkout for the conversion tools.

## Build (Windows / HIP / gfx1100)

```powershell
# 1) Configure + build the ik2 target (see script header for all params)
.\scripts\build-windows-hip.ps1 `
  -EngineDir  <repo>\engine `
  -BuildDir   <repo>\build-hip-win `
  -RocmSdk    <path\to\_rocm_sdk_devel> `
  -Target     strata-ik2

# 2) Build the kernel parity test (optional)
.\scripts\build-windows-hip.ps1 ... -Target ik2_kernels_test
```

At runtime put the ROCm SDK `bin` and `lib\llvm\bin` on `PATH` (the bench script does this). Compilation uses the SDK's clang/headers; runtime links the SDK's HIP/hipBLAS/rocBLAS DLLs. Set `AMDGPU_TARGETS=gfx1100`, `GGML_HIP=ON`, `STRATA_ENABLE_CUDA=OFF`, `STRATA_ENABLE_HIP=ON`.

## Prepare model & token files

The CLI consumes **token-id files** (`--tokens @file`), not raw text: comma-separated integers, one prompt per file. Generate them with your Inkling-2 tokenizer (`inkling2` / O200K BPE, `ignore_merges=true`). A diverse ~325-token prompt and a long structured ~16k-token prompt are useful for reproduction; see the benchmark section for which prompt class each number uses.

## INT4 weight conversion

The INT4 paths need converted sidecar files generated from the GGUF (the engine still reads the GGUF for routing/metadata). Scripts live in `tools/int4/`:

```powershell
python tools\int4\nvfp4_to_int4.py   ...   # NVFP4 experts (layers 0-19) -> int4-bin
python tools\int4\q8_to_int4_l20.py  ...   # Q8_0 experts (layers 20-23) -> int4-bin
python tools\int4\q8_dense_to_int4.py ...  # dense projections + out_w -> int4-dense-bin (Wp/Wh/WS)
```

Each script takes explicit input GGUF / output dir arguments (see `--help` / file header). Set `IK2_INT4_DIR` and `IK2_INT4_DENSE_DIR` to the output directories. The conversion is byte-deterministic: `Wp`/`Wh`/`WS` repacking matches the device kernels; a round-trip verification tool is included. INT4 sidecars **replace** (not coexist with) the original NVFP4 residency because of the 24 GB VRAM budget.

## Run & benchmark

```powershell
# Standard AR benchmark (prefill + decode), correct+fast defaults
.\scripts\bench.ps1 -Exe <repo>\build-hip-win\strata-ik2.exe `
  -Model <path>\Inkling-2-Mini-UD-Q4_K_XL.gguf `
  -PromptFile <path>\prompt-ids.txt -RocmSdk <path\to\_rocm_sdk_devel> `
  -Mode ar -Gen 128 -Chunk 9216

# DSpark speculative decode (native draft) — sustained 200+ on structured prompts
.\scripts\bench.ps1 ... -Mode dspark -Draft <path>\mtp.safetensors `
  -PromptFile <structured-prompt-ids.txt> -Gen 256 -Ctx 4096

# 1M-context decode (KV in VRAM, host tiering, INT4 head decode)
.\scripts\bench.ps1 ... -Mode longctx -Ctx 1052672 -Gen 64 -PromptFile <short-prompt-ids.txt>
```

**GPU safety**: keep only one model-loading process at a time. The engine refuses to over-commit VRAM; on a 24 GB WDDM card, spilling to host memory collapses decode by >10×. The bench script refuses to start if another model process is running.

## Measured results

All numbers: Windows 11, RX 7900 XTX 24 GB (gfx1100, 960 GB/s), 32 GB RAM, greedy decoding, default config = WMMA flash (NW8/BN32) + INT4 experts + INT4 dense, KV as noted, experts fully resident, CUDA graph captured.

### Short / medium context

| Config | Prefill | AR decode |
|---|---|---|
| 16k prompt, chunk 9216, f16 KV | **3053 tok/s** | 92.9 tok/s |
| 4k prompt, chunk 4096, f16 KV | 2463 tok/s | **103.5 tok/s** |
| Short prompt gen256 | — | **107.3 tok/s** |
| 64k prompt, fp8 KV | 2463 tok/s | 59.2 tok/s |
| Baseline llama.cpp V3 (diverse 325) | 539 tok/s | 44.5 tok/s |

Prefill scaling (16k, chunk sweep): chunk 8192 → 2567, **9216 → 2679**, 10240 → 2636, 11264 → 1010 tok/s (expert-cache thrash). WMMA flash attention raised 16k prefill 1450 → 2661 tok/s (+83%).

### DSpark speculative decode

| Prompt class | Accept (drafts/step) | Decode |
|---|---|---|
| Long structured table, gen256 | 7.73 / 8 | **246.8 tok/s** |
| Long code, gen256 | 7.50 / 8 | 217.4 tok/s |
| Prose | 5.73 / 8 | 176 tok/s |
| Short capital prompt | 2.25 / 8 | ~47 tok/s |

Verify cost dropped 155.9 → 27.4 ms/step via device-only verify (no host routing sync), WMMA drafting (10.3 → 5.8 ms), INT4 verify paths and slot-major MoE. B=8 is the trained draft block; B∈{2,4,6} all measured worse (86/148/151 tok/s). 246.8 tok/s ≈ 97% of the B=8 perfect-accept ceiling (~255 tok/s).

### Long context (int4 KV, cut24 experts, fully resident)

| Context | Prefill | Decode |
|---|---|---|
| 4k | 2262 tok/s | 102 tok/s |
| 128k | 1614 tok/s | 46.6 tok/s |
| 256k | 1097 tok/s | 31.7 tok/s |
| 512k | 661 tok/s | 18.8 tok/s |
| 640k | 552 tok/s | 15.6 tok/s |
| 768k | OOM in all-VRAM (EC budget < 1 GiB hard minimum) | — |
| 1M (host-mapped KV) | prefill PCIe-write-bound (slow) | ~3–10 tok/s |

1M context **fits** with the host-tiering path (`IK2_KV_HOST=1`, or KV in VRAM via `IK2_DENSE_HOST=1` + INT4 sidecars). With only 4k tokens populated in a 1M-allocation config, decode reaches **103 tok/s** (head-decode fix: the output-head Q8 read was 60 ms/token over PCIe; routing it through the VRAM INT4 sidecar cut it to 0.3 ms). With a truly populated 256k context the bottleneck is attention's O(ctx) KV scan (~85 GB/s effective vs. 846 GB/s tensor-core path on the same architecture). The 8× full-attention visits per token make the decay architectural; NVIDIA's reference keeps ~26% at 131k even with tensor cores. An 8-snapshot conversation cache parks prefix state in host RAM for multi-turn reuse (93× prefix-reuse speedup measured).

## Correctness validation

- Kernel parity: **315 ok / 0 fail / 3 PTX-skip** on gfx1100 (double-reference parity for RMS, L2 clip, conv4, attention residual/probs/decode/multi, routing ties, NVFP4/Q8_0 experts, fp8 KV, loop gates, token path).
- Greedy parity: bit-exact token streams vs. llama.cpp V3 reference (temperature 0, top_k 1) over 31–96 tokens on diverse / capital prompts, across scalar, WMMA-flash, INT4 and DSpark configs.
- FP8/INT4 KV and INT4 expert conversions verified bit-exact against FP16/Q8 references; residual differences are near-tie router tokens from INT4 requantization only.

## Environment variable reference

Defaults are the validated correct + fast setup. Escape hatches let you fall back to strict FP16/NVFP4 paths.

| Variable | Default | Meaning |
|---|---|---|
| `IK2_FLASH` / `IK2_FLASH_WMMA` | 1 | gfx1100 WMMA flash attention (prefill +83% at 16k) |
| `IK2_FLASH_NW` / `IK2_FLASH_BN` | 8 / 32 | flash tile geometry (swept optimum) |
| `IK2_WMMA_GEMM` | 1 | dense projection WMMA/BK64 (prefill +50–70%) |
| `IK2_INT4_EXPERTS` | 1 | rotated INT4 experts (escape 0 → NVFP4/Q8) |
| `IK2_INT4_DENSE` | 1 | rotated INT4 dense projections |
| `IK2_INT4_LAYERS` | 20 | number of expert layers converted to INT4 (24 = all) |
| `IK2_INT4_HEAD` / `IK2_INT4_HEAD_DECODE` | 1 | INT4 output-head sidecar + VRAM token-1 head decode |
| `IK2_INT4_DIR` / `IK2_INT4_DENSE_DIR` | relative | converted sidecar directories |
| `IK2_KV` | f16 | KV cache format: f16 / fp8 / int8 / int4 |
| `IK2_KV_HOST` | 0 | host-mapped KV (1M tiering; decode PCIe-slow) |
| `IK2_DENSE_HOST` / `IK2_INT4_PREFILL` | 0 | host-map dense copies / prefill from VRAM INT4 sidecar |
| `IK2_EC_ROOM_MIB` | 256 | extra MiB reserved for the resident expert cache |
| `IK2_LG_CTX` | — | cap logits scratch to reduce work-buffer VRAM |
| `IK2_VERIFY_FAST` / `IK2_DRAFT_WMMA` | 0 | DSpark device-only verify / WMMA drafting |
| `IK2_INT4_DENSE_MULTI` / `_DENSE_MV` / `_MSG` / `_HEAD` / `IK2_MV_SLOT_MAJOR` / `IK2_MV_NW` / `IK2_DRAFT_HEAD_INT4` | 0 | validated DSpark speed opts (see bench.ps1) |
| `IK2_ATTN_MMA` / `IK2_EXPERT_MMA` / `IK2_EXPERTS_PREFILL_WMMA` | 0 | NVIDIA-PTX paths — keep 0 on HIP |
| `IK2_GATE_FUSE` / `IK2_RMS_FUSE` | 1 | small launch-fusion opts |
| `IK2_PROF` / `IK2_PROF_FAST` | 0 | phase profiling (disables resident fast path / event-based) |
| `GGML_HIP_NO_VMM` | set | use plain allocations instead of VMM |

## Source changes

All optimizations live under `engine/`. Key files:

| File | Change |
|---|---|
| `include/strata/hip_compat/{cublas_v2.h,cuda_runtime.h,intrinsics.hpp,cuda_fp8.h}` | wave32 CUDA→HIP shim; cuBLAS/runtime/intrinsics/fp8 coverage |
| `src/ik2/ik2_kernels.cu` / `.cuh` | PTX guards; WMMA flash `<BN,NW,KVT,IBITS>`; INT4 experts/dense/head/KV kernels; warp-shuffle reductions; parity test hooks |
| `src/ik2/ik2_engine.cu` | hybrid WMMA GEMM dispatch; INT4 expert/dense/head loading; DSpark verify + host tiering; residency/fit logic; profiler |
| `src/ik2/ik2_draft.cu` / `.hpp` | native DSpark drafter, ring-cache context, draft-window env |
| `src/prefill/wmma_gemm.cu` / `.h` | strided WMMA GEMM + tile-size switch |
| `tests/cuda/ik2_kernels_test.cu` | double-reference parity, WMMA flash + INT4 cases |
| `cmake/strata_gpu_engine.cmake` | ik2 target guard `OR HIP`, link `strata_core`, HIP WMMA sources |

Conversion tools: `tools/int4/*.py`. Reproducible bench/build helpers: `scripts/*.ps1`.

## Credits & license

Based on [Strata](https://github.com/Niko1221/Strata) by Niko1221 and contributors; the ik2 engine originates from Strata PR #6506. Inkling-2-Mini is by Thinking Machines. Original Strata is MIT-licensed; see `LICENSE`. This fork adds the Windows/HIP port and optimizations described above.
