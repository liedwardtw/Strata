# RX 7900 XTX — VRAM budget & long context

How the 640K context + 8 GiB conversation cache fits in 24 GB, and why 1M is possible but slow.

## 640K all-VRAM budget (validated, resident, 0 misses)

| Component | Size | Notes |
| --- | ---: | --- |
| INT4 experts, 24 physical layers | ~12.23 GiB | fully resident, device pointer table, 0 misses |
| Dense projections (Q8 planar, host-mapped) | ~1.10 GiB | decode reads INT4 sidecar in VRAM, not host Q8 |
| KV cache @ 640K, INT4 | ~5.36 GiB | 48 layer visits, ~1056 B/token/visit (K+V int4 codes + fp16 scales) |
| ExpertCache + work buffers | ~2 GiB | chunk 4096; chunk > 10240 evicts resident experts (thrash) |
| Reserve | 0.5 GiB | `IK2_RESERVE_MIB=512` |
| **Total** | **~19.2 GiB** | fits the 23.7 GiB WDDM budget |

`fit(actual): experts 12.23 + dense 1.10 + KV 5.36 + reserve 0.50 = 19.19 vs free 22.99 GiB -> FITS`.

## KV format comparison (measured at 256K)

| KV format | Bytes | Decode tok/s | VRAM |
| --- | ---: | ---: | ---: |
| fp8 | ~4.5 GiB | 42.2 | baseline |
| int8 | ~4.5 GiB | 46.5 | +10% |
| **int4** | ~2.3 GiB | **47.6** | **+13%, half the VRAM** |

RDNA3 has no FP8 WMMA, so int4 KV is both faster (fewer bytes read) and smaller. int8 is more precise but costs 2× bytes with no speed win here.

## Conversation cache (host RAM)

`--conversation-cache-mib 8192` reserves 8192 MiB host RAM for **64 snapshots** (41.2 MiB each). Each snapshot parks the prefix state (conv histories in VRAM, sliding KV in host vectors), so agents taking turns can reuse a prior conversation. Measured: 247 of 259 prompt tokens reused, prompt-reuse 49%, 2/64 conversations parked. Full-visit KV is not snapshotted (only sliding window).

## Why chunk size matters more than ctx for residency

The work buffers scale with `chunk` (gu32 / blocks / h / r / tmp / res / xq / xk / xv / q / gate / o), not with `ctx`. At chunk 9216 + 128K, work buffers ate ~4.4 GiB and evicted resident experts (decode 14.6 tok/s). At chunk 4096 the same context stayed resident (41.6 tok/s). Chunk 10240 is the boundary; 11264 thrashes.

## 1M context

1M INT4 KV ≈ 8.4 GiB. Plus experts 12.23 + dense 1.1 + work ~2 = ~23.9 GiB, just over the 23.7 GiB WDDM budget — **all-VRAM 1M OOMs at EC init**. Two ways it fits:

- `IK2_KV_HOST=1` — host-map full-visit KV. 1M loads and is resident, but prefill is PCIe-write-bound (>7 min) and decode reads 8.4 GiB host KV/token (~3–10 tok/s).
- `IK2_DENSE_HOST=1` — host-map the prefill-only Q8 dense copies, freeing ~0.7 GiB so 1M KV stays in VRAM. Decode is fine (~14 tok/s with only 4K KV populated); full 1M prefill is still PCIe-read-bound.

Honest conclusion: 1M fits with RAM+VRAM cooperation, but **does not keep 640K-speed** — the 8 full visits are O(ctx) and no engine flag removes that. 640K all-VRAM is the fast boundary. See [LIMITS](XTX-7900XTX-LIMITS.md).
