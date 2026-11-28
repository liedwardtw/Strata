# RX 7900 XTX — correctness & quality methodology

We treat bit-exactness as a hard gate. A speed trick that changes the greedy output is not a speedup; it is a different model.

## Bit-exact checks

1. **Kernel parity (`ik2_kernels_test`)**: 315 ok / 0 fail / 3 PTX-skip on gfx1100. Double-reference parity for RMS, L2 clip, conv4, attention residual / probs / decode / multi, routing ties, NVFP4 / Q8_0 experts, fp8 KV, loop gates, token path.
2. **Greedy token-stream parity**: compare full token ids (not text summaries) against llama.cpp V3 (temperature 0, top_k 1) over 31–96 tokens on diverse and capital prompts, across scalar / WMMA-flash / INT4 / DSpark configs. All kept configs match exactly.
3. **Conversion parity**: INT4 expert / dense / head / KV conversions verified bit-exact (or worst-abs-diff 0) vs FP16 / Q8 references; rotated-int4 round-trip checks in `tools/int4/validate_int4*.py`.

## Prompt-level first-divergence flip rate

For "relaxed" (non-bit-exact) experiments we used **prompt-level first-divergence** as the acceptance metric, not token-level divergence rate: an early single-token flip gets diluted by a long output in token-level stats, but it changes the answer. Gate: <3% flip at 25 prompts (0/25 flips), plus hard gates of 0 tool-call failures, 0 structured-parse failures, and median speedup ≥1.5×.

19 quality prompts were built (capital / factual / math / near-tie / JSON / table / code / prose / tool-like / multilingual) plus 4 multi-step reasoning prompts.

## What the gate caught

- `IK2_FA_RDNA3` bitmask `EXP2` (replace expf with exp2-style math): unit worst-diff only 2.38e-7, +4.9% speed at 16K — but **16/19 prompts (84.2%) diverged** (first divergence at token 14–19). Rejected. Token-level divergence would have looked tiny; prompt-level caught it.
- `BATCH_GRAPH_EXACT` (the "fast" batched path): ~123 tok/s aggregate (5.3× vs serial 23) but token 1–3 already diverged, 57–62/64 tokens different. Catastrophic, rejected.
- `IK2_FA_SPLIT` (8B int4 load with shuffle): bit-exact 0 flips but −1.3% slower. Falsified.
- `PRE_LOAD_V` (move V dequant before the R-head loop): worst-diff 0, +1.3% — the only non-baseline lever kept (default-off, opt-in).
- `WMMA_TILE=1` single-token GEMV: changed greedy logits (idx3 20 vs 15) and was slower. Rejected.

Conclusion on this model/kernel: **there is no "non-bit-exact big speed" lever**. All defensible large speedups (dual AR 147.9, DSpark 265) are bit-exact; all relaxed variants either diverge from token 1–19 or are slower. See [LIMITS](XTX-7900XTX-LIMITS.md).

## Fast model note

Lucebox's `Inkling-2-Mini-Fast-IQ4_NL` claims "2.01× end-to-end, thinking tokens −50.2%, +3.8% accuracy" on R9700 + luce_server + DSpark. We downloaded it, ported IQ4_NL decode (found and fixed an fp16-scale bits-as-float bug), built an offline converter to the original layout, and ran 8 thinking-enabled reasoning questions on our hardware: Fast's thinking length was the same or **longer** (mixed), answers all correct. The claim did not reproduce on 7900XTX/ik2, and our deployment uses `reasoning_effort=none` anyway, so we kept the original model.
