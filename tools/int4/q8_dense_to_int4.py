"""Q8_0 dense projections -> rotated symmetric INT4 sidecars for Inkling-2-Mini decode (gfx1100).

The DENSE attention projections (wqkvrg / wqrg / wk / wv / wo) are Q8_0 in the GGUF and dominate
the proj+wo decode bandwidth (~24.6% of a token). This converts them to the SAME rotated-symmetric
int4 layout the experts already use (nvfp4_to_int4.py: H256 rotation per 256-block along K, codes
-7..7, fp16 scale per 32*GW=128 values, GW=4, SH8=12, k_repack Wp/Wh/WS), so the decode matvec
(k_mv_dense_rot) reads ~half the weight bytes. Prefill stays Q8_0 (the engine keeps q8 resident).

Reuses nvfp4_to_int4.quantize_q4_0_rot / repack / rot_matrix VERBATIM - the only new step is the
Q8_0 -> FP32 dequant and the per-role row concatenation that mirrors the engine's fused matrices:
  sliding: wqkvrg = [attn_q | attn_k | attn_v | attn_rel | attn_gate] rows  (N=8576, K=C=2304)
  full:    wqrg   = [attn_q | attn_rel | attn_gate] rows                    (N=6528, K=2304)
           wk = attn_k, wv = attn_v                                       (N=1024, K=2304)
  all:     wo = attn_output.weight                                        (N=C=2304, K=HD=3072)

Output: raw .bin per (layer, role, {Wp,Wh,WS}) + manifest.json in out_dir, loaded by the engine's
IK2_INT4_DENSE sidecar loader (ik2_engine.cu). No .npy (the loader reads raw bins).

--head: ALSO/INSTEAD convert the verify HEAD projection output.weight (Q8_0, [N=201024, K=2304])
to out_w_{Wp,Wh,WS}.bin for IK2_INT4_HEAD (run_head's T=2..8 verify GEMM). With --head only the
head is converted (the layer sidecars are left as-is) and manifest.json is MERGED, not replaced.
The head is converted in row chunks (multiples of 16): quantize_q4_0_rot is row-local and repack
is 16-row-block-local (Wp [ft,sb,g,r,widx], Wh [ft,sb,r,4], WS per row), so chunked output is
byte-identical to a whole-matrix conversion while bounding peak RAM.
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, r".")
sys.path.insert(0, r"gguf-py")
import nvfp4_to_int4 as C   # SH8, rot_matrix, quantize_q4_0_rot, repack (exact same math as experts)
from gguf import GGUFReader


def dequant_q8_0(t, N, K):
    """Q8_0 tensor data (uint8, N*K/32 blocks of 34 bytes) -> float32 [N, K].
    Block = 2-byte fp16 scale d + 32 int8 codes; value = d * q. Row-major over N rows."""
    raw = np.asarray(t.data, dtype=np.uint8).reshape(N, K // 32, 34)
    d = raw[..., 0:2].copy().view(np.float16).astype(np.float32)        # [N, K/32, 1]
    q = raw[..., 2:34].copy().view(np.int8).astype(np.float32)          # [N, K/32, 32]
    return (d * q).reshape(N, K)


def convert_one(w, gw, H, sh):
    """w [N, K] float32 -> (Wp u8, Wh u8, WS f32) in the expert k_repack layout."""
    N, K = w.shape
    assert K % 256 == 0 and K % (32 * gw) == 0 and N % 16 == 0, (N, K)
    blocks, dd = C.quantize_q4_0_rot(w, gw, H)
    Wp, Wh, WS = C.repack(blocks, dd, N, K, sh)
    assert Wp.size == N * K // 2 and Wh.size == (N // 16) * (K // 256) * 64 and WS.size == N
    assert np.isfinite(WS).all() and (WS >= 0).all()
    return Wp, Wh, WS


def convert_head(tmap, out_dir, gw, sh, H, chunk=4096):
    """output.weight [N, K] Q8_0 -> out_w_{Wp,Wh,WS}.bin, chunked in multiples of 16 rows.
    Byte-identical to convert_one on the whole matrix (repack is ft-block-local); returns the
    manifest entry. chunk=4096 rows -> ~38 MB float32 + ~19 MB Wp per chunk, peak RAM ~1 GB."""
    t = tmap["output.weight"]
    assert int(t.tensor_type) == 8, ("output.weight", int(t.tensor_type))   # Q8_0
    K = int(t.shape[0]); N = int(t.shape[1])
    assert K % 256 == 0 and K % (32 * gw) == 0 and N % 16 == 0 and chunk % 16 == 0, (N, K, chunk)
    raw = np.asarray(t.data, dtype=np.uint8).reshape(N, K // 32, 34)
    wp_f = open(os.path.join(out_dir, "out_w_Wp.bin"), "wb")
    wh_f = open(os.path.join(out_dir, "out_w_Wh.bin"), "wb")
    ws_f = open(os.path.join(out_dir, "out_w_WS.bin"), "wb")
    t0 = time.time()
    ws_min, ws_max = float("inf"), 0.0
    try:
        for r0 in range(0, N, chunk):
            n = min(chunk, N - r0)
            d = raw[r0:r0 + n, :, 0:2].copy().view(np.float16).astype(np.float32)
            q = raw[r0:r0 + n, :, 2:34].copy().view(np.int8).astype(np.float32)
            w = (d * q).reshape(n, K)
            Wp, Wh, WS = convert_one(w, gw, H, sh)
            wp_f.write(Wp.tobytes()); wh_f.write(Wh.tobytes())
            ws_f.write(WS.astype(np.float32).tobytes())
            ws_min = min(ws_min, float(WS.min())); ws_max = max(ws_max, float(WS.max()))
            if (r0 // chunk) % 8 == 0 or r0 + n == N:
                print(f"head rows {r0+n}/{N} ({time.time()-t0:.1f}s)", flush=True)
    finally:
        wp_f.close(); wh_f.close(); ws_f.close()
    entry = {"layer": -1, "role": "out_w", "M": N, "K": K,
             "Wp_bytes": N * K // 2, "Wh_bytes": (N // 16) * (K // 256) * 64, "WS_bytes": N * 4}
    print(f"out_w: N={N} K={K} Wp={entry['Wp_bytes']}B Wh={entry['Wh_bytes']}B WS={N}f "
          f"WS[min,max]=[{ws_min:.4g},{ws_max:.4g}] in {time.time()-t0:.1f}s", flush=True)
    return entry


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--gw", type=int, default=4)
    ap.add_argument("--sh", type=int, default=C.SH8)
    ap.add_argument("--layers", default="0-23")
    ap.add_argument("--head", action="store_true", help="convert only the head output.weight (merge manifest)")
    ap.add_argument("--head-chunk", type=int, default=4096, help="head conversion rows per chunk (multiple of 16)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    H = C.rot_matrix()
    r = GGUFReader(a.src)
    tmap = {t.name: t for t in r.tensors}

    if a.head:
        entry = convert_head(tmap, a.out_dir, a.gw, a.sh, H, a.head_chunk)
        mpath = os.path.join(a.out_dir, "manifest.json")
        manifest = {"gw": a.gw, "sh": a.sh, "rotation_block": 256, "source": "Q8_0", "layers": []}
        if os.path.exists(mpath):
            with open(mpath) as f:
                manifest = json.load(f)
            manifest["layers"] = [e for e in manifest["layers"] if e.get("role") != "out_w"]
        manifest["layers"].append(entry)
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"wrote out_w sidecar to {a.out_dir} (manifest merged)", flush=True)
        return

    lo, hi = (int(x) for x in a.layers.split("-")) if "-" in a.layers else (int(a.layers), int(a.layers))

    # sliding pattern -> per-layer role set (mirrors ik2_engine.cu config parse)
    def g(k):
        f = r.get_field("inkling2." + k)
        if f is None:
            return None
        v = f.contents()
        return v[0] if isinstance(v, list) else v
    n_visit = int(g("block_count")); n_pass = int(g("num_loops") or 1)
    core0 = int(g("loop.core_start") or 0); core1 = int(g("loop.core_end") or 0)
    n_core = core1 - core0
    n_layer = n_visit - (n_pass - 1) * n_core
    pat = r.get_field("inkling2.attention.sliding_window_pattern").contents()
    assert isinstance(pat, list) and len(pat) == n_visit, (type(pat), len(pat) if isinstance(pat, list) else pat)
    sliding = [pat[l if l < core1 else l + (n_pass - 1) * n_core] != 0 for l in range(n_layer)]
    print(f"n_layer={n_layer} full={[l for l in range(n_layer) if not sliding[l]]}", flush=True)

    manifest = {"gw": a.gw, "sh": a.sh, "rotation_block": 256, "source": "Q8_0", "layers": []}
    t0 = time.time()
    for l in range(lo, hi + 1):
        b = f"blk.{l}."
        tq = tmap[b + "attn_q.weight"]; tk = tmap[b + "attn_k.weight"]; tv = tmap[b + "attn_v.weight"]
        tr = tmap[b + "attn_rel.weight"]; tg = tmap[b + "attn_gate.weight"]; to = tmap[b + "attn_output.weight"]
        for t in (tq, tk, tv, tr, tg, to):
            assert int(t.tensor_type) == 8, (t.name, int(t.tensor_type))   # Q8_0
        K = int(tq.shape[0]); HD = int(tq.shape[1]); KD = int(tk.shape[1]); HR = int(tr.shape[1])
        wq = dequant_q8_0(tq, HD, K); wk = dequant_q8_0(tk, KD, K); wv = dequant_q8_0(tv, KD, K)
        wr = dequant_q8_0(tr, HR, K); wg = dequant_q8_0(tg, HD, K)
        wo = dequant_q8_0(to, int(to.shape[1]), int(to.shape[0]))          # [C, HD]
        roles = []
        if sliding[l]:
            roles.append(("wqkvrg", np.concatenate([wq, wk, wv, wr, wg], axis=0)))
        else:
            roles.append(("wqrg", np.concatenate([wq, wr, wg], axis=0)))
            roles.append(("wk", wk)); roles.append(("wv", wv))
        roles.append(("wo", wo))
        for role, w in roles:
            N, KK = w.shape
            Wp, Wh, WS = convert_one(w, a.gw, H, a.sh)
            tag = f"L{l:02d}_{role}"
            Wp.tofile(os.path.join(a.out_dir, tag + "_Wp.bin"))
            Wh.tofile(os.path.join(a.out_dir, tag + "_Wh.bin"))
            WS.astype(np.float32).tofile(os.path.join(a.out_dir, tag + "_WS.bin"))
            manifest["layers"].append({"layer": l, "role": role, "M": N, "K": KK,
                                       "Wp_bytes": int(Wp.size), "Wh_bytes": int(Wh.size), "WS_bytes": int(WS.size * 4)})
            if l == lo:
                print(f"L{l} {role}: N={N} K={KK} Wp={Wp.size}B Wh={Wh.size}B WS={WS.size}f "
                      f"WS[min,max]=[{WS.min():.4g},{WS.max():.4g}]", flush=True)
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"wrote {len(manifest['layers'])} sidecars to {a.out_dir} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()