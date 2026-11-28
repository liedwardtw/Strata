# RX 7900 XTX — DSpark (native speculative decoding)

The Inkling-2-Mini ships a native MTP/DSpark drafter (`mtp.safetensors`, BF16, 3 layers, block size 8). Our ik2 engine can drive it directly; with the model's structured prompts it is the fastest decode mode measured.

## Results (greedy bit-exact)

| Prompt | Gen | Aggregate tok/s | Accept (tokens/step) |
| --- | ---: | ---: | ---: |
| table-long | 256 | 243–247 | 7.7/8 |
| table-long (best single run) | 64 | **265** | 7.7/8 |
| code-long | 256 | 217 | 7.5/8 |
| prose | — | — | 5.73/8 |
| capital-style | — | — | 2.25/8 |
| table @ 256K context | — | ~34 (loses to AR) | 1.96/8 |

Accept rate is **workload-dependent**: repetitive/structured text sustains 7.5–7.7 of 8; prose and capital-style reasoning drop sharply. B=8 (the trained block size) is optimal: an `IK2_SPEC_N` sweep gave B=6→151, B=4→148, B=2→86 tok/s.

## What made it work on gfx1100

- **Device-only verify** (`IK2_VERIFY_FAST=1`): the default T=8 verify path took the general `forward()` route with 48 device→host routing syncs per step (~169 ms). Device-only routing cut verify to ~60 ms; graph capture of the verify step was neutral (GPU-execution bound, not launch bound).
- **Draft WMMA**: the drafter's small GEMMs run through the RDNA3 WMMA path (10 → 5.8 ms/step).
- **Slot-major MoE** + `MV_NW=6` → L2 locality and fewer waves.
- **INT4 draft head** (926 → 240 MB) frees VRAM for long-context residency.
- **argmax race fix**: `argmax_sum` used device globals `g_argmax_v/i[1024]`; with two concurrent CUDA graphs (or even one graph + a second verify stream) both wrote the same 197 partial slots → near-tie logit flips (e.g. slot0 diverged at token 16). Made the argmax partials per-Work (fresh allocation for the cloned stream) → dual-stream bit-exact 0 diffs.
- **Ring-cache drafter context** (`IK2_DRAFT_CTX_CAP=1`): local drafter layers use a 2048-row ring instead of ctx, saving VRAM at long context (bit-exact).

## When not to use it

At 256K context the accept rate collapses to 1.96/8 (~34 tok/s) and DSpark loses to plain AR. The 8-token batched verify over 256K KV costs more than 8 GQA-shared AR steps. Non-bit-exact "faster" variants were tried and rejected: `IK2_VERIFY_GQA` combined grid → 531 ms verify (3.7× slower); flat graph → 68 tok/s with token diffs; K-step persistent graphs → no speedup on gfx1100. See [LIMITS](XTX-7900XTX-LIMITS.md).

## Running

DSpark is enabled by the `--draft` CLI flag with the native `mtp.safetensors`. In serve mode, `IK2_SERVE_DSPEC=1` enables greedy DSpark in the request path (default 0 = AR). The Desktop App auto-adds `--draft off` for Inkling engines; our engine treats that as a no-op plain-AR request.
