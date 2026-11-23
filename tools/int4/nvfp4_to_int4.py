"""NVFP4 -> rotated symmetric INT4 expert-weight converter for Inkling-2-Mini on gfx1100.

Reads the unsloth GGUF (NVFP4 type-40 experts L0-19), dequantizes exactly as ik2's
decode_value/ue4m3 (ik2_kernels.cu:2229/2268), applies the H256 Hadamard rotation
per 256-block along K, quantizes symmetric int4 (codes -7..7, fp16 scale per 32*GW
values, GW=4 default), folds the per-expert outer scale into the row scale, and
writes the operand-ready repacked layout (Wp K-slices + Wh u16 sub-scales + WS row
scale) consumed by the ported k_gemm8w / k_mv kernels.

Output: a binary sidecar per (layer, role) plus a manifest. The ik2 engine loads
these at startup instead of decoding NVFP4 every visit.

Math source: repos/llama.cpp-rdna3-int4 docs/rdna3/quantization/rotate_quantize.py
and ggml/src/ggml-cuda/int4-wmma.cu (k_repack:91, k_row_scale:71, P:63).
"""
import argparse, json, struct, sys, time
import numpy as np
sys.path.insert(0, r"gguf-py")
from gguf import GGUFReader
from scipy.linalg import hadamard

SH8 = 12  # 8-bit activation fixed-point bits (quality mode)

def signs256():
    j = np.arange(256, dtype=np.uint64)
    return np.where(((j * np.uint64(2654435761)) & np.uint64(0xFFFFFFFF)) >> np.uint64(31), -1.0, 1.0).astype(np.float32)

def rot_matrix():
    return (hadamard(256).astype(np.float32) / 16.0) * signs256()[None, :]

# E2M1 LUT x2 (ik2 decode_value kv[16])
KV16 = np.array([0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12], dtype=np.float32)

def ue4m3(x):
    """ik2 ue4m3 (ik2_kernels.cu:2229): e+119 => scale x0.5 (cancels KV x2)."""
    x = np.asarray(x, dtype=np.uint8).astype(np.uint32)
    e = (x >> 3) & 0xF
    m = x & 7
    bits = ((e + 119) << 23) | (m << 20)
    norm = bits.view(np.float32) if bits.dtype == np.uint32 else np.frombuffer(bits.astype(np.uint32).tobytes(), dtype=np.float32)
    sub = m.astype(np.float32) * (2.0 ** -10)
    v = np.where(e == 0, sub, norm)
    v = np.where(x == 0x7F, 0.0, v)
    return v.astype(np.float32)

def dequant_nvfp4(raw, M, K):
    """raw: uint8 [M, K*9/16] NVFP4 blocks for ONE expert. Returns float32 [M, K].
    36-byte group = 4 E4M3 scale bytes + 32 packed code bytes (64 values).
    value = KV16[nibble] * ue4m3(scale_byte)."""
    raw = np.asarray(raw, dtype=np.uint8).reshape(M, K * 9 // 16)
    ng = K // 64
    out = np.empty((M, K), dtype=np.float32)
    for g in range(ng):
        grp = raw[:, g*36:(g+1)*36]
        sc = ue4m3(grp[:, 0:4])                 # [M, 4] scales for 4 sub-blocks of 16
        codes = grp[:, 4:36]                    # [M, 32] bytes = 64 nibbles
        lo = (codes & 0xF).astype(np.int32)
        hi = (codes >> 4).astype(np.int32)
        # j<8 -> low nibble of byte j; j>=8 -> high nibble of byte j-8
        # sub-block sb (16 values): bytes sb*8..sb*8+7, low nibbles j=0..7, high j=8..15
        for sb in range(4):
            b = codes[:, sb*8:(sb+1)*8]
            l = (b & 0xF).astype(np.int32)
            h = (b >> 4).astype(np.int32)
            vals = np.concatenate([KV16[l], KV16[h]], axis=1)  # [M, 16]
            out[:, g*64 + sb*16 : g*64 + sb*16 + 16] = vals * sc[:, sb:sb+1]
    return out

def quantize_q4_0_rot(w, gw, H):
    """w [M, K] float32 -> rotate per 256-block along K, symmetric int4 Q4_0 blocks.
    Returns (blocks uint8 [M, K/32*18], d_f32 [M, K/32]) with scale shared over gw groups."""
    M, K = w.shape
    assert K % 256 == 0 and K % (32*gw) == 0, (M, K)
    wr = (w.reshape(M, K//256, 256) @ H.T).reshape(M, K)
    g = wr.reshape(M, K//(32*gw), 32*gw)
    d = (np.abs(g).max(-1, keepdims=True) / 7.0).astype(np.float16)
    df = np.maximum(d.astype(np.float32), 1e-30)
    q = np.clip(np.rint(g / df), -7, 7).astype(np.int8)
    q = q.reshape(M, K//32, 32)
    nib = (q + 8).astype(np.uint8)
    qs = nib[..., :16] | (nib[..., 16:] << 4)
    dd = np.repeat(d.reshape(M, K//(32*gw), 1), gw, axis=1).reshape(M, K//32).astype(np.float16)
    blocks = np.empty((M, K//32, 18), dtype=np.uint8)
    blocks[..., 0:2] = dd.view(np.uint8).reshape(M, K//32, 2)
    blocks[..., 2:] = qs
    return blocks.reshape(M, K//32*18), dd.astype(np.float32)

def P(s):
    if s < 8:
        return (4 + (s >> 1)) if (s & 1) else (s >> 1)
    t = s - 8
    return 8 + ((4 + (t >> 1)) if (t & 1) else (t >> 1))

def _repack_idx():
    """Gather index [4 words, 8 nibbles] for one 32-code group -> 4 u32 K-slices.
    word widx=sl*2+dw, nibble s2 reads code e=16*sl+P(dw*8+s2)."""
    idx = np.zeros((4, 8), dtype=np.int32)
    for sl in range(2):
        for dw in range(2):
            for s2 in range(8):
                idx[sl*2 + dw, s2] = 16*sl + P(dw*8 + s2)
    return idx

_RIDX = _repack_idx()
_SHIFT = (np.arange(8, dtype=np.uint32) * 4)

def repack(blocks, d_f32, M, K, sh):
    """Q4_0 blocks -> operand-ready Wp (int4 K-slices) + Wh (u16 sub-scales) + WS (row scale).
    Mirrors k_row_scale + k_repack (int4-wmma.cu:71/91). Vectorized over (m, sb, g)."""
    nb = K // 32
    nsb = K // 256
    b = blocks.reshape(M, nb, 18)
    dd = d_f32.reshape(M, nb)
    WS = np.abs(dd).max(axis=1) * ((1 << sh) / 15.0)   # [M]
    WS_safe = np.where(WS > 0, WS, 1.0)
    qs = b[..., 2:]
    lo = (qs & 0xF).astype(np.int32)
    hi = (qs >> 4).astype(np.int32)
    q = np.concatenate([lo, hi], axis=-1) - 8          # [M, nb, 32]
    sgn = np.where(dd < 0, -1, 1)
    q = np.where(sgn[..., None] < 0, np.clip(-q, -7, 7), q)
    qg = (q & 15).astype(np.uint32).reshape(M, nsb, 8, 32)   # [M, nsb, g, 32 codes]
    # words[m, sb, g, widx] = OR_s2 qg[..., idx[widx,s2]] << (4*s2)
    picked = qg[..., _RIDX]                            # [M, nsb, 8, 4, 8]
    words = np.bitwise_or.reduce(picked << _SHIFT, axis=-1)  # [M, nsb, 8, 4] u32
    # Wp u32 layout [ft, sb, g, r, widx]; m = ft*16+r
    Wp = words.reshape(M//16, 16, nsb, 8, 4).transpose(0, 2, 3, 1, 4)
    Wp = np.ascontiguousarray(Wp).reshape(-1).view(np.uint8)
    # sub-scales: |dd|/WS*(1<<sh) clamp 0..15, two groups per byte
    sub = np.clip(np.rint(np.abs(dd) / WS_safe[:, None] * (1 << sh)), 0, 15).astype(np.uint8)
    sub = sub.reshape(M, nsb, 4, 2)
    whb = sub[..., 0] | (sub[..., 1] << 4)             # [M, nsb, 4]
    Wh = whb.reshape(M//16, 16, nsb, 4).transpose(0, 2, 1, 3)
    Wh = np.ascontiguousarray(Wh).reshape(-1)
    return Wp, Wh, WS.astype(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("out_dir")
    ap.add_argument("--gw", type=int, default=4)
    ap.add_argument("--layers", default="0-19")
    ap.add_argument("--sh", type=int, default=SH8)
    ap.add_argument("--max-experts", type=int, default=0, help="0=all; else first N experts (test)")
    a = ap.parse_args()
    lo, hi = (int(x) for x in a.layers.split("-")) if "-" in a.layers else (int(a.layers), int(a.layers))
    H = rot_matrix()
    r = GGUFReader(a.src)
    tmap = {t.name: t for t in r.tensors}
    import os
    os.makedirs(a.out_dir, exist_ok=True)
    manifest = {"gw": a.gw, "sh": a.sh, "rotation_block": 256, "layers": []}
    t0 = time.time()
    for l in range(lo, hi+1):
        for role, nm, M, K in [("gate_up", f"blk.{l}.ffn_gate_up_exps", 2304, 2304),
                                ("down", f"blk.{l}.ffn_down_exps", 2304, 1152)]:
            t = tmap[nm + ".weight"]
            assert t.tensor_type == 40, (nm, t.tensor_type)
            E = int(t.shape[2])
            ne = E if a.max_experts == 0 else min(E, a.max_experts)
            raw = np.asarray(t.data).reshape(E, M, K*9//16)
            sc = np.asarray(tmap[nm + ".scale"].data, dtype=np.float32)  # [E]
            # down K=1152 not %256: pad to 1280 with zeros (rotation needs 256-blocks)
            Kpad = ((K + 255)//256)*256
            Wp_all, Wh_all, WS_all = [], [], []
            for e in range(ne):
                w = dequant_nvfp4(raw[e], M, K)          # [M, K]
                w = w * sc[e]                            # fold per-expert outer scale
                if Kpad != K:
                    wp = np.zeros((M, Kpad), dtype=np.float32); wp[:, :K] = w; w = wp
                blocks, dd = quantize_q4_0_rot(w, a.gw, H)
                Wp, Wh, WS = repack(blocks, dd, M, Kpad, a.sh)
                Wp_all.append(Wp); Wh_all.append(Wh); WS_all.append(WS)
                if e == 0:
                    print(f"L{l} {role} e0: M={M} K={K} Kpad={Kpad} Wp={Wp.size}B Wh={Wh.size}B WS={WS.size}f", flush=True)
            tag = f"L{l:02d}_{role}"
            np.save(os.path.join(a.out_dir, tag + "_Wp.npy"), np.concatenate(Wp_all))
            np.save(os.path.join(a.out_dir, tag + "_Wh.npy"), np.concatenate(Wh_all))
            np.save(os.path.join(a.out_dir, tag + "_WS.npy"), np.concatenate(WS_all))
            manifest["layers"].append({"layer": l, "role": role, "M": M, "K": K, "Kpad": Kpad,
                                        "E": ne, "Wp_bytes_per_expert": int(Wp_all[0].size),
                                        "Wh_bytes_per_expert": int(Wh_all[0].size)})
            print(f"  wrote {tag} ({ne} experts) {time.time()-t0:.1f}s", flush=True)
    with open(os.path.join(a.out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"DONE {time.time()-t0:.1f}s -> {a.out_dir}", flush=True)

if __name__ == "__main__":
    main()