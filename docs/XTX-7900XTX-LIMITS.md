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
**The 8,192-token chunk cliff is VRAM paging, not an algorithm.** Chunk 7,168 reads a 44K prompt at 2,613 tok/s and chunk 8,192 at 1,435. `IK2_PF_PROFILE=1` (new: the prefill breakdown at the end of each served request, because the CLI cannot take a prompt longer than 4,096 tokens) shows where it goes at a 640K context: `loop_gates` 0.545 -> 5.755 s and `attn proj` 1.253 -> 5.202 s, while the tiled MoE phases get *faster* (gather 1.184 -> 0.868 s, gemm16 flat). The two phases that collapse are the ones streaming the largest scratch buffers. At a 32K context the same chunk 8,192 is faster than 7,168 (2,463 vs 2,276 tok/s), so the cliff is the 640K config sitting ~0.6 GiB from the VRAM budget. Consequence: a context the workload does not need costs speed. Paired, interleaved, 2 rounds at a 44,288-token prompt: ctx 655360 + chunk 7168 -> 2,610 / 2,621 tok/s; ctx 131072 + chunk 10240 -> 2,706 / 2,705 (+3.4%). `start-daily.ps1` now takes `-Ctx` and `-Chunk`, defaulting to the 640K pair.

## What the decode profile can and cannot tell you (2026-11-29)

A 48K prompt, one stream, greedy, 24 tokens, `ik2dec.py --tokens 48000 --gen 24 --env IK2_PROF_FAST=1 --env IK2_GRAPH=0`:

| phase | ms a token | share |
| --- | --- | --- |
| attn | 5.444 | 37.3% |
| experts | 3.189 | 21.8% |
| proj | 1.385 | 9.5% |
| input_norm | 1.008 | 6.9% |
| router_route | 0.851 | 5.8% |
| moe_norm | 0.779 | 5.3% |
| wo | 0.587 | 4.0% |
| loop_bank | 0.544 | 3.7% |
| head | 0.304 | 2.1% |
| prep_kv | 0.247 | 1.7% |
| combine | 0.163 | 1.1% |
| emb_norm | 0.104 | 0.7% |
| **total** | **14.604** | |

The same profile at a 2,048-token prompt reads 12.487 ms a token with `attn` at 3.336. So attention carries **3.34 ms that does not depend on the context at all** and only 2.1 ms that does (48K minus 2K).

That 3.34 ms is not GPU work. It is the launch cost of 48 layer visits x ~16 kernels x ~4 us, and the CUDA graph removes it: the same engine at a 64-token prompt decodes **112 tok/s with the graph and 82 tok/s without** (8.9 vs 12.1 ms a token), and the 3.2 ms difference is exactly the fixed part of the profile.

So the graph-off profile is only readable for the phases that scale with real work (attention over the context, the MoE, the head). Every small per-visit phase — the two norms, `prep_kv`, `combine`, `wo`, `loop_bank` — reads 10-20x its true cost, because with the graph off each of those launches is a real host launch. And the graph-on profile cannot be taken at all: `IK2_PROF_FAST` events never fire inside a captured graph, so every phase reads 0.000 with `IK2_GRAPH=1`.

The numbers to trust are end to end with the graph on: **112 tok/s at a 64-token prompt, 89 at 8K, 62 at 48K, 96 at 393K with `IK2_FULL_WINDOW=32768`**.

That table also settles what the window is for. At 48K, cutting the keys 6x (`IK2_FULL_WINDOW=8192`) moves attention from 5.668 to 4.108 ms and the token from 15.098 to 14.226 (−5.8%): at this depth the attention is not key-bound, so the window only pays where it removes most of the keys. At 393K it does (13.5 -> 95.7 tok/s); at 48K it does not, and the conversation cache is the lever there instead.

One caveat on measurement: process to process at 48K the decode rate moves by ±20% (62.4 in one run, 49-51 in another with the same config). Single samples at that depth mean nothing; only interleaved paired runs do.

**The park was inside the decode timer.** At a 48K conversation the conversation cache copies 405 MB of full-attention KV to host RAM after every reply. That copy sat between the last token and the `DONE` line, and the decode timer read `now()` after it, so a request that decoded at 89 tok/s reported 64. Two reps each, single stream, greedy, 32 tokens:

| 48K, 32 tokens | cache on | cache off |
| --- | --- | --- |
| park before `DONE` | 64.08 / 56.98 | 89.36 / 87.29 |
| park after `DONE` | 88.74 / 86.70 | 89.21 / 87.41 |

The park now runs after `DONE` is printed and the timer is taken before it, so the reported speed is the decode's own and the client gets the reply ~80 ms earlier. The copy still happens; the next request pays for it before its first token, which is where it belongs, because a cold read of the same conversation costs 18.5 s.

## The prefill ceiling, and why 4,000 tok/s is not reachable on this kernel shape (2026-11-29)

Measured with chunk 7,168, INT4 KV, every expert resident, two reps each: **2,954 tok/s at 16,384 tokens** (2,843 / 3,066), **2,613-2,649 at 44,288**, **2,483 at 600,000** (with `IK2_FULL_WINDOW=32768`).

The environment space is exhausted. At a 44,288-token prompt, one full read each, against a 2,613 baseline: `IK2_FLASH_NW=16` 2,490, `=4` 2,512, `IK2_FLASH_BN=64` 2,530, `IK2_ATTN_GQA=0` 2,609, `IK2_EXPERTS_PREFILL_WMMA=1` 2,605, `IK2_INT4_PREFILL=1` 2,582, `IK2_GATE_FUSE=0` 2,596, `IK2_RMS_FUSE=0` 2,602 — all inside the ±12% run-to-run noise. `--threads 8 / 16 / 32` gives 2,605.0 / 2,605.5 / 2,609.9, so the host is not the limit either.

The phase profile (read the shares, not the absolute times: `IK2_PF_PROFILE=1` syncs after every phase and costs 19%, 2,152 vs 2,649 tok/s) splits a 4,096-token chunk into attention 1.03 s (projections 0.618, output 0.313), experts 1.21 s (gather 0.393, gemm16 0.546, scatter 0.118), loop gates 0.244 s and router 0.03 s.

That is ~8.4 TFLOP of projection work and ~0.52 TFLOP of expert work per chunk. The projections run at ~13.5 TFLOPS, 21% of the card's 61 TFLOPS fp16 peak; the experts run at ~1 TFLOPS because the prefill MoE is 128 small GEMMs of ~256 rows each. Reaching 4,000 tok/s needs both of those to more than double, which is a rewrite of the INT4 GEMM and of the expert loop — not a switch, a chunk size or a thread count.

**But the chunk size does buy 7.3%, if the context is not 640K.** At a 44,288-token prompt, one rep each (repeats agree within 0.3%):

| ctx | chunk | tok/s |
| --- | --- | --- |
| 655360 | 7168 | 2,613 (the daily default) |
| 131072 | 10240 | 2,699 |
| 131072 | 12288 | 2,751 |
| 131072 | 14336 | 2,754 |
| 131072 | **16384** | **2,804** |
| 65536 | 16384 | 2,805 |
| 131072 | 20480 | 768 |
| 65536 | 20480 | 952 |
| 196608 | 16384 | 1,466 |
| 262144 | 16384 | 1,103 |
| 393216 | 16384 | 721 |

Two cliffs, both reproducible and both at the same place whatever the prompt:

- **chunk above 16,384 collapses** (768 tok/s at 20,480) at a 64K context and at a 131K context alike, so it is a hard limit of the buffers or the kernels, not of VRAM.
- **ctx above 131,072 with a 16K chunk pages** (1,466 at 196K, 721 at 393K), the same paging that makes the 8,192 chunk collapse at 640K: the 16K chunk's scratch plus the KV no longer fit.

The 640K config cannot use a 16K chunk, so the daily default stays at 7,168. `start-daily.ps1 -Ctx 131072 -Chunk 16384` is the faster configuration for a workload that fits in 131K: measured end to end through the server at a 48,422-token prompt, 17.63 s to first token against 18.83 s on the default (+6.8%), with the conversation cache unaffected (turns 2-4 reuse 48,422 / 48,439 / 48,457 tokens at 412-439 ms).

Also measured and rejected on the same prompt: `IK2_WMMA_GEMM=0` (the rocBLAS path) reads at 1,681 tok/s against 2,613, so the WMMA INT4 GEMM is already the fast path; `IK2_INT4_DENSE=0` 2,604 and `IK2_INT4_PREFILL=1` 2,593, both inside the noise.

## What a cached turn still costs, and where it goes (2026-11-29)

The conversation cache turns a 48,422-token turn from 19 s into a fraction of a second, but it is not free, and the engine now prints the price. Two lines per turn on stderr (the server's log), with the copy time and the bandwidth it achieved:

```
strata serve: conversation cache: restored 48439 tokens; parked=4 bytes=1637771520; 76.3 ms (5.93 GB/s)
strata serve: conversation cache: parked 48512 tokens; parked=4 bytes=1637771520; 143.0 ms (3.17 GB/s)
```

At 48,422 tokens a snapshot holds 452 MB (the sliding windows of the 40 sliding layers plus the whole KV of the 8 full-attention layers). Restoring it moves 452 MB at **5.93 GB/s** (76 ms); parking it moves 454 MB at **3.17 GB/s** (143 ms). Both are pageable-memory speeds: a pinned buffer on PCIe 4.0 x16 should reach 20-25 GB/s, so the snapshot buffers are the next thing to pin (they are plain std::vector<uint8_t> today). Pinning would take a turn's two copies from 219 ms to roughly 50 ms.

Two hypotheses were ruled out before that conclusion:

- **not the decode graph capture.** The graph is captured per request, so it was the prime suspect for a fixed per-turn cost. Running the server with `IK2_GRAPH=0` changed the cached turns from 409.3 / 444.1 / 425.4 ms to 400.5 / 404.1 / 419.0 ms, i.e. nothing.
- **not the restore.** The restore is only 76-80 ms of a 409-456 ms turn, so the rest is the prompt pass itself.

The rest is the price of reading a *short* chunk. Sent straight to the engine (`ik2cache.py`, the same 48,422-token prompt, then the same prompt plus N tokens, so the second request reuses 48,422 tokens), the new tokens cost:

| new tokens | chunk time | per token |
|---|---|---|
| 17 | 226 ms | 13.3 ms |
| 300 | 511 ms | 1.70 ms |
| 1,000 | 877 ms | 0.88 ms |
| 7,168 | 2,775 ms | 0.39 ms |

So a chunk carries a fixed cost of roughly 200-350 ms (48 layer visits of setup and small-kernel launches) plus 0.5-1.0 ms a token, and only a full 7,168-token chunk amortises the fixed part down to the 0.36 ms a token that the prefill table reports. A cached agent turn therefore reaches its first token in about 0.36 s when it adds 17 tokens and about 0.95 s when it adds 1,000 — against 19 s cold. The fixed part is the next lever: it is per chunk, not per token, so it would need the per-visit setup to be amortised across chunks (or a smaller, cheaper visit for short chunks).

`IK2_PF_PROFILE=1` is not the instrument for this. It syncs after every phase, which on a short chunk costs more than the phases it measures: a 17-token cached chunk reported attention 0.13 s plus experts 0.57 s, more than the 0.36 s the request took in total.
### The snapshot buffers are pinned now (same day)

The four host buffers of a snapshot (kv_k, kv_v and the two scale planes) are std::vector<uint8_t, PinAlloc>: PinAlloc locks the pages with `cudaHostAlloc` and falls back to the heap when the lock fails, so a large cache can never kill the engine, only slow it. The buffers are also sized when a conversation is parked rather than up front, so the 64 slots no longer reserve 2.6 GB of locked pages nobody uses.

Same 48,422-token conversation, same run shape:

| | pageable | pinned |
|---|---|---|
| restore 452 MB | 76.3 ms (5.93 GB/s) | **41.2 ms (10.98 GB/s)** |
| park 454 MB | 143.0 ms (3.17 GB/s) | **120.7 ms (3.75 GB/s)** |
| cached turn, prompt_ms (turns 2-4) | 409 / 444 / 425 ms | **372 / 360 / 360 ms** |

The restore nearly doubled; the park barely moved, so the park's limit is the device side (or WDDM's staging), not the host pages. A cached turn at 48k tokens is now 0.36-0.37 s against 19 s cold, and the remaining time is the short chunk's own cost (see the table above), not the cache.

Quality unchanged: `quality.py run pinned` 11/11 passed, and the four serve test files pass (39 passed, 3011 subtests).
### A chunk carries ~200-300 ms of fixed cost, and it is not the MoE, the projections or the attention (2026-11-29)

A short chunk is expensive out of proportion to its tokens. Sent straight to the engine (`ik2cache.py`), the same 8,000-token prompt followed by N more tokens costs:

| new tokens | chunk time | per token |
|---|---|---|
| 17 | 199 ms | 11.7 ms |
| 300 | 511 ms | 1.70 ms |
| 1,000 | 877 ms | 0.88 ms |
| 7,168 | 2,856 ms | 0.40 ms |

The same 17-token chunk costs 199 ms at an 8K context and 226 ms at a 48K one, and a *cold* 17-token prompt costs the same, so the cost does not depend on the context and does not come from the conversation cache. Fitting the two points inside one 8,000-token request (chunk 1 = 7,168 tokens in 2,856 ms, chunk 2 = 832 tokens in 594 ms) gives a marginal 0.357 ms a token and a fixed **297 ms per chunk**.

Six candidate causes were each measured and each ruled out:

| candidate | how it was tested | result |
|---|---|---|
| the MoE's per-expert loop | `IK2_EXPERTS_PREFILL_WMMA=1` / `IK2_INT4_LAYERS=48` | 197 / 215 ms against 211 — no change |
| hipBLAS call overhead in the projections | `IK2_INT4_PREFILL=1` | 197 ms against 199 — no change |
| the attention walk | `IK2_FULL_WINDOW=1024` / `IK2_FLASH_NW=4` / `IK2_ATTN_GQA=0` | 199 / 201 / 206 ms — no change |
| the route copy to the host | the staging buffers are already `cudaHostAlloc`-pinned | not it |
| the decode graph capture | `IK2_GRAPH=0` | no change |
| the cache restore | timed separately | 41 ms of a 360 ms turn |

So roughly 200-300 ms sits in the per-chunk setup of the 48 layer visits, in a part none of those switches reaches. It is worth about 2.6% of a 7,168-token prefill (2,856 → ~2,670 ms would be 2,684 tok/s) and about 190 ms on every short agent turn, which makes it a better target for the agent loop than for the prefill number. Finding it needs a per-visit timer that does not itself synchronise (`IK2_PF_PROFILE` does, and on a short chunk it reports more time than the request took).
### What the per-chunk floor is not: the GEMM stack, the launch count, or a per-visit sync (2026-11-29)

Three more candidates were measured after the eight above.

The projections already run on the tuned kernel, not hipBLAS. At a 44,288-token prompt, `IK2_WMMA_GEMM=1` reads 2,612 tok/s and `IK2_WMMA_GEMM=0` (hipBLAS fp16) reads 1,700, and the daily launcher already sets `=1`. So the ~13.5 TFLOPS the projections reach is the custom kernel's own throughput (22% of the card's fp16 peak), not call overhead, and `IK2_INT4_PREFILL=1` cannot help because it only swaps one fast path for another.

Removing launches does not help either, because a short chunk is not launch-bound: a 17-token continuation at an 8K context costs 201 ms at the baseline, 201 with `IK2_I4_ROT_IN=2` (three fewer launches a visit), 197 with `IK2_LOOPBANK_FUSE=1` and 221 with both.

The phase profiler cannot localise the floor. With `IK2_PROF=1` a chunk's attention reads 0.45 s at 17 tokens, 0.62 s at 1,024 and 1.01 s at 4,096, i.e. flat up to a thousand tokens — but the profiler itself synchronises twice a visit (96 syncs a chunk), which is the same order as the floor it is trying to measure. Its own numbers are 3-4x the unprofiled chunk.

The floor does explain the 16K-chunk result, which is the best check available: a 44,288-token prompt is 7 chunks at 7,168 and 3 at 16,384, and four fewer chunks at ~300 ms is 1.2 s of a 17 s read = 7%, against the measured +7.3% (2,804 against 2,613 tok/s). So the 16K chunk's gain is the floor avoided, not a better kernel.
### The floor is per visit, and it is the per-expert MoE's launch count (2026-11-29)

`IK2_VPROF4=1` prints a chunk's per-visit split from the CUDA events forward() already records (one sync for the whole chunk, so unlike `IK2_PROF` it does not change what it measures). Three chunks of the same conversation:

| chunk | total | per visit | slowest visit |
|---|---|---|---|
| 7,168 tokens | 2,457.8 ms | 51.20 ms | 415.2 ms (visit 0, cold) |
| 832 tokens | 492.7 ms | 10.27 ms | 16.0 ms |
| 17 tokens | 209.4 ms | 4.36 ms | 18.5 ms (visit 34) |

Fitting the three: **4.36 ms a visit + 0.0065 ms a token** (832 -> 9.8 predicted against 10.3 measured; 7,168 -> 51.0 against 51.2). So every visit carries ~4.4 ms that does not depend on how many tokens it is given.

That fixed part is the launch count. `run_prefill_moe` loops over the experts on the host and issues six kernels per expert (gather, rot_pack8, gemm8 gate/up, swiglu+pad, rot_pack8, gemm8 down, scatter), so a visit is ~540 launches at a 17-token chunk (S = 17x8 = 136 rows spread over 128 experts, so nearly every expert has a row) at ~8 us each. At 7,168 tokens the same 48 visits cost 48 x 4.36 = 209 ms of a 2,458 ms chunk, i.e. **8.5% of the read**, which is also why a bigger chunk pays: fewer chunks means fewer of these.

The fix that follows from this is the masked expert GEMM the batch path already uses (`IK2_BATCH_MOE_DEVICE`, engine/src/ik2/ik2_engine.cu:4423-4457): permutation, gather, rot_pack8, gemm8_masked, swiglu, pad, rot_pack8, gemm8_masked, scatter — nine launches a visit instead of ~540, with the row->expert map that `run_prefill_moe` already builds. Expected: ~+9% on a long read and 2-3x on a short continuation; not done yet because it needs a device row->expert array and its own quality check.
## The masked expert GEMM removes the launches and costs the read (2026-11-29)

The idea above was implemented and measured: `IK2_PF_MOE_BATCH` routed the whole prefill MoE through the batch path's masked expert GEMM (`k::int4_gemm8_moe_masked`, engine/src/ik2/ik2_engine.cu:4440) with a device row->expert array built in `run_prefill_moe`'s host grouping loop, tiled to the f32 scratch's row capacity. Nine launches a visit instead of ~540. Measured with `ik2cache.py --tokens 8000 --extra 17` and `IK2_VPROF4=1`, against the same run without the flag:

| chunk | per-expert loop | masked GEMM | read of the 8,000-token prompt |
| 17 tokens | 4.36 ms a visit | **3.27 ms** (-25%) | 209 -> 157 ms |
| 832 tokens | 10.27 ms | 11.00 ms (+7%) | 493 -> 528 ms |
| 7,168 tokens | 51.20 ms | 133.30 ms (+160%) | 2,458 -> 6,399 ms |

So it wins only on a nearly empty chunk and costs a real read 2.6x. The reason is the tile shape: the masked kernel launches a block per expert (`grid.z = c.E` = 128) and the prefill tiles rows by the scratch capacity, so a big chunk becomes hundreds of tiles that each pay 128 mostly-idle expert blocks, while the per-expert loop gives one expert all its rows in a properly tiled GEMM. The row->expert map and the branch were reverted rather than kept behind a flag.

The ~4.4 ms a visit is therefore the price of the launch count *and* the shape the experts are routed in, and it is not recoverable without a genuinely grouped GEMM: every expert's rows in one kernel, expert-major weights, no per-expert grid dimension. That is a new kernel, not a switch.

## Two more levers measured and closed (2026-11-29)

- **FP16 KV at a 131,072 context does not buy the tensor-core attention.** With `IK2_KV=f16` the KV is 6.66 GiB (against 1.35 GiB of INT4) and the read fell to **261 tok/s** against 2,804, because the extra 5 GiB pushes the resident expert sidecars out of the budget. The MMA attention path needs f16 KV, but on this card the memory it costs is worth more than the attention it saves.
- **`IK2_AR_KSTEP=1 IK2_AR_K=2` does nothing on the single-stream decode**: 106.48 tok/s against 106.90 at an 8,000-token prompt. The +3.2% recorded for it earlier was on the batched path, which the daily configuration does not use.
