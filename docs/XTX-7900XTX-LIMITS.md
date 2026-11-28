# RX 7900 XTX — limits & falsified approaches (honest)

This document records what did **not** work, so nobody repeats it. All claims are from measurements on one RX 7900 XTX (gfx1100) under Windows HIP.

## 1. "Use the full 960 GB/s" is not the right target

Six independent analyses (bandwidth physics, attention kernel, expert kernels, dequant ALU, graph replay, upstream survey) reached the same conclusion:

- Effective HBM bandwidth of the decode path is only **13–38% of peak** (1.8–43 GB/s measured across 1K–640K). The machine is **not memory-bandwidth-bound**.
- Real bottlenecks: serial online-softmax with SFU `expf` (6 expf/key, 25 ops:byte in the attention kernel), INT4 dequant/rotate ALU, and the **48-visit RAW dependency chain** (attn → MoE → attn).
- Even at perfect HBM: 256K → ~200 tok/s, 640K → ~118 tok/s are the *bandwidth floors*; measured 43 / 23 tok/s is far below that because of ALU/serial limits.
- All bit-exact levers are small: PRE_LOAD_V +1.3%, preserve-order 8B load 0–3%, cache modifiers 0–3%, `IK2_FA_SPLIT` actually −1.3%.

## 2. Dual-stream concurrency stalls at ~148 tok/s

True batched forward and per-slot graph capture were implemented and verified bit-exact, but aggregate decode tops out at:

| Setup | Aggregate | Note |
| --- | ---: | --- |
| Serial P4 (4 slots) | ~18 tok/s | general forward path |
| Graph-per-slot, 2 streams | **147.9 tok/s** | 1.94× overlap, 0 token diffs |
| Dual + K-step persistent graph (K=2/4) | ~147 tok/s | no gain — graph node count grows linearly with K |
| `BATCH_GRAPH_EXACT` | 123 tok/s | 57–62/64 tokens different — rejected |

Root cause: gfx1100 HIP graph replay is host-driven per-node (~1.9 µs/node × ~1016 nodes ≈ 1.95 ms/step); there is no env flag to disable it. Merging graphs, host-thread submit, and flat fork/join graphs were all tried and were slower or non-exact. The theoretical zero-overhead ceiling is ~202 tok/s; reaching 185–195 needs K≥20 persistent graphs (large scratch, stop/max handling, graph-walk overhead unmeasured) — judged not worth it. The 200/185 target was withdrawn by the user.

## 3. Non-bit-exact speed does not survive quality checks

See [QUALITY](XTX-7900XTX-QUALITY.md). EXP2 (+4.9% speed) caused 84% prompt-level flips from a 2.38e-7 unit diff, amplified by the 48-visit loop. `BATCH_GRAPH_EXACT` (5.3×) diverged from token 1–3. There is no "large speed + small output change" lever on this model.

## 4. Fast model claim not reproduced

`Inkling-2-Mini-Fast-IQ4_NL` (Lucebox) advertises 2.01× end-to-end via weight-tuned concise reasoning (−50.2% thinking tokens, +3.8% accuracy) on R9700/Linux/luce_server with DSpark. On 7900XTX/ik2:

- 8 thinking-enabled reasoning questions: Fast's thinking length was equal or longer (mixed), all answers correct — no consistent reduction.
- CLI decode speed equal (~109 tok/s); with `reasoning_effort=none` our original model already runs AR-direct at ~140 tok/s short-output.
- Supporting work was done (native IQ4_NL decode with fp16-scale fix, prefix auto-detect, offline converter) but is **not used** — we kept the original UD-Q4_K_XL model.

## 5. 1M context fits but is slow

640K all-VRAM is the fast boundary. 1M requires host tiering (`IK2_KV_HOST=1` or `IK2_DENSE_HOST=1`); it loads and is resident, but prefill is PCIe-bound (minutes) and full-1M decode extrapolates to ~10–15 tok/s. The 8 full-attention visits are O(ctx); no flag removes that. See [VRAM](XTX-7900XTX-VRAM.md).

## 6. Platform caveats

- `gpu_guard` must run; never two model-loading processes on a 24 GB WDDM card.
- `IK2_ATTN_MMA` / `IK2_EXPERT_MMA` are NVIDIA-PTX paths — trap-stub or hang on HIP; keep 0.
- Wrong `IK2_INT4_DIR` / `IK2_INT4_DENSE_DIR` silently falls back to Q8 experts → ~3.9 tok/s.
- The stock signed Strata engine is untouched; our local engine is an unsigned per-user shape component (Windows Smart App Control may need a local allow).

## Measured ceiling and the long-context levers (2026-11-29)

All numbers on the RX 7900 XTX, this build, the daily 640K config (chunk 7168, INT4 KV, INT4 experts 24 layers), single stream, greedy, no other GPU work. Run-to-run noise on the same config is +-12%, so only a change above ~15% is resolvable.

**Prefill is at its ceiling for this design.** Eight environment knobs were swept at a 44,288-token prompt and every one landed inside the noise (2,490-2,613 tok/s): `IK2_FLASH_NW=16`/`=4`, `IK2_FLASH_BN=64`, `IK2_ATTN_GQA=0`, `IK2_EXPERTS_PREFILL_WMMA=1`, `IK2_INT4_PREFILL=1`, `IK2_GATE_FUSE=0`, `IK2_RMS_FUSE=0`. The ceiling by prompt length is 2,954 tok/s at 16K, 2,564 at 21.6K, 2,613 at 44.3K. The 3,053 recorded in [BENCHMARKS](XTX-7900XTX-BENCHMARKS.md) was a 16K prompt with FP16 KV and chunk 9216; with INT4 KV the flash path has no tensor-core MMA (`mma` is false when `kv_is_int()`), which is why a 7168-token chunk beats the old 9216.

What the profile still shows (per 4,096-token chunk): attention 1.03 s of which the input projection is 0.618 s and the output projection 0.313 s, experts 1.21 s (gather 0.393, gemm16 0.546, scatter 0.118), loop gates 0.244 s. The projection GEMM runs at ~11% of the card's FP16 peak, so the remaining headroom is kernel efficiency (fused loop gates, an expert-major GEMM that does not gather first), not memory.

**Decode by prompt length, single stream, 24-32 tokens after the prompt** (tok/s):

| prompt | before | park capped | capped + `IK2_FULL_WINDOW=32768` |
|---|---|---|---|
| 48K | 62.4 | 62.4 | 64.9 |
| 128K | 36.0 | 64.1 | 96.5 |
| 256K | 21.8 | 45.2 | 93.2 |
| 393K | 13.5 | 35.4 | 95.7 |

Two changes got there. The conversation park copied the full-attention KV to host RAM with no bound and that copy is synchronous: at 393K it stalled ~42 s a request, so it is now capped at `IK2_CONV_CACHE_PARK_MIB` (default 1024 MiB) and a longer conversation is simply not cached. `IK2_FULL_WINDOW=N` then makes the 8 full-attention visits read only the last N keys while decoding. It is lossy and off by default: with a 22,251-token prompt and the answer at the very start, N=32768 answers correctly, N=16384 and N=4096 do not.

At the 48K the agent traffic actually runs at, the conversation cache is the lever that matters: a second turn reuses 48,422 tokens and reads the prompt in 446 ms instead of 18,830 ms (42x), and an unrelated request in between does not evict it.
**A 600K prompt now reads at all.** Without the window the eight full-attention visits walk the whole KV once per chunk, so prefill is O(ctx^2): a 600,000-token prompt was still running after ten minutes and was killed twice. `IK2_FULL_WINDOW=32768` now applies to the prompt pass as well as decode, and the same prompt reads in 241.6 s (2,483 tok/s, 84 chunk passes, 4 generated tokens). The KV is still built in full; only the re-read is bounded, so nothing older than the window is available to the model, which is the same loss decode already took. Measured on this build with `ik2bench.py --tokens 600000`; the 11-case quality set still passes 11/11 with the window on, and at 48K the window is worth only 2,613 -> 2,757 tok/s because the full visits are a small share of the work there.