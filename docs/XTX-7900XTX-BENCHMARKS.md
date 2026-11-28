# RX 7900 XTX — Full-depth Inkling-2-Mini benchmark results

Single machine, single RX 7900 XTX (gfx1100, 24 GB, ~960 GB/s HBM peak), Windows native HIP, 32 GB RAM. Every row: greedy decoding, INT4 experts + INT4 dense + INT4 KV, GQA + DPP attention, WMMA flash prefill, single CUDA graph, **resident with 0 expert misses** unless noted.

Honesty first: each data point was measured once or a few times, not a rigorous multi-run study. Treat numbers as order-of-magnitude. Speed fields were cross-checked as `completion_tokens / (wall - prefill)` where possible.

## 1. Full-depth table (long prompt actually fills the KV)

| Input tokens | Prefill tok/s | Decode tok/s | Resident |
| ---: | ---: | ---: | :---: |
| 1,024 | 818 | 101.4 | yes |
| 4,096 | 2,121 | 101.0 | yes |
| 16,384 | **3,021** (chunk 9216) | 93.6 | yes |
| 32,768 | 2,850 | 90.6 | yes |
| 65,536 | 2,428 | 77.9 | yes |
| 131,072 | 1,623 | 61.6 | yes |
| 262,144 | 819 | 43.3 | yes |
| 524,288 | 550 | 27.6 | yes |
| 655,360 (640K) | 472 | 23.1 | yes |

Decode falls with context because the 8 full-attention visits scan O(ctx) KV bytes. Prefill peaks at 16K then falls (longer prompts force smaller chunks / more KV traffic).

## 2. Prefill chunk sweep (16K prompt)

| chunk | 8192 | 9216 | 10240 | 11264 |
| --- | ---: | ---: | ---: | ---: |
| tok/s | 2567 | **2679** | 2636 | 1010 (expert-cache thrash) |

The WMMA flash-attention kernel raised 16K prefill 1450 → 2661 tok/s (+83%).

## 3. DSpark (native speculative decoding, greedy bit-exact)

Drafter = model-native `mtp.safetensors` (BF16). Accept rate depends heavily on prompt structure.

| Prompt | Gen | tok/s | accept |
| --- | ---: | ---: | ---: |
| table-long | 256 | 243–247 | 7.7/8 |
| table-long (single best) | 64 | **265** | 7.7/8 |
| code-long | 256 | 217 | 7.5/8 |
| prose / capital | — | — | 5.73 / 2.25 |
| table 256K | — | 34 | 1.96 |

At 256K the accept rate collapses to 1.96 and DSpark loses to plain AR (see [LIMITS](XTX-7900XTX-LIMITS.md)).

## 4. Correctness

- Kernel parity: 315 ok / 0 fail / 3 PTX-skip on gfx1100 (double reference).
- Greedy token streams bit-exact vs llama.cpp V3 over 31–96 tokens, across scalar / WMMA-flash / INT4 configs.
- Every speed lever in this fork is env-gated and was kept only after A/B token-stream comparison.

## 5. Baseline for comparison

llama.cpp V3 (Unsloth build, IQ4_NL-style quant) on the same card: ~44.5 tok/s decode, ~539 tok/s prefill (512 tokens). See the development report for the full engine survey.
