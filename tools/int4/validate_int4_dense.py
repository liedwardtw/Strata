"""Validate q8_dense_to_int4.py: Q8_0 dequant, repack layout (independent inverse) and end-to-end matvec.
Run after the converter: python tools/validate_int4_dense.py [out_dir] [gguf] [layer]"""
import os, sys
import numpy as np
sys.path.insert(0, r".")
sys.path.insert(0, r"gguf-py")
import nvfp4_to_int4 as C
import q8_dense_to_int4 as Q
from gguf import GGUFReader

out = sys.argv[1] if len(sys.argv) > 1 else r"int4-dense-bin"
gguf = sys.argv[2] if len(sys.argv) > 2 else r"Inkling-2-Mini-UD-Q4_K_XL.gguf"
L = int(sys.argv[3]) if len(sys.argv) > 3 else 0
gwp = 4; sh = C.SH8
H = C.rot_matrix()
r = GGUFReader(gguf)
tmap = {t.name: t for t in r.tensors}

def dequant_q8(name, N, K):
    return Q.dequant_q8_0(tmap[name], N, K)

def reconstruct(Wp, Wh, WS, M, K, sh):
    """Inverse of k_repack, read exactly as k_mv_dense_rot does -> w' [M,K] float32.
    Wp u32 [ft,sb,g8,r,widx]; word widx=sl*2+dw, nibble s2 -> code
    e=16*sl+P(dw*8+s2) in a 32-code group. Wh[ft,sb,r,g8//2] packs two 4-bit subs.
    Weight code = (n^8)-8 (the sdot4 offset trick); scale = WS[m]*sub/2^sh."""
    nsb = K // 256
    Wp = np.asarray(Wp, dtype=np.uint8).view(np.uint32).reshape(M//16, nsb, 8, 16, 4)
    Wh = np.asarray(Wh, dtype=np.uint8).reshape(M//16, nsb, 16, 4)
    WS = np.asarray(WS, dtype=np.float32).reshape(M)
    codes = np.zeros((M//16, 16, nsb, 8, 32), dtype=np.int32)   # [ft, r, sb, g8, e]
    for sl in range(2):
        for dw in range(2):
            for s2 in range(8):
                e = 16*sl + C.P(dw*8 + s2)
                nib = (Wp[:, :, :, :, sl*2+dw] >> (4*s2)) & 0xF        # [ft,sb,g8,r]
                codes[:, :, :, :, e] = (((nib ^ 8).astype(np.int32)) - 8).transpose(0, 3, 1, 2)
    sub = np.zeros((M//16, 16, nsb, 8), dtype=np.float32)
    for g8 in range(8):
        byte = Wh[:, :, :, g8//2]                                      # [ft,sb,r]
        sub[:, :, :, g8] = ((byte >> (4*(g8 % 2))) & 0xF).astype(np.float32).transpose(0, 2, 1)
    sc = WS.reshape(M//16, 16, 1, 1) * sub / float(1 << sh)            # [ft,r,sb,g8]
    return (codes * sc[..., None]).reshape(M, K)

def check(name, w, tag):
    M, K = w.shape
    blocks, dd = C.quantize_q4_0_rot(w, gwp, H)
    # ideal rotated int4 weight (dd * q), q = code - 8 per 32
    b = blocks.reshape(M, K//32, 18)
    ddv = dd.reshape(M, K//32)
    qs = b[..., 2:]
    q = np.concatenate([(qs & 15).astype(np.int32), (qs >> 4).astype(np.int32)], -1) - 8
    ideal = (ddv[:, :, None] * q).reshape(M, K)
    Wp = np.fromfile(os.path.join(out, tag + "_Wp.bin"), dtype=np.uint8)
    Wh = np.fromfile(os.path.join(out, tag + "_Wh.bin"), dtype=np.uint8)
    WS = np.fromfile(os.path.join(out, tag + "_WS.bin"), dtype=np.float32)
    assert Wp.size == M*K//2 and Wh.size == (M//16)*(K//256)*64 and WS.size == M, (tag, Wp.size, Wh.size, WS.size)
    rec = reconstruct(Wp, Wh, WS, M, K, sh)
    rel = np.linalg.norm(rec - ideal) / np.linalg.norm(ideal)
    print(f"{name} {tag}: w'[{M},{K}] reconstruct vs ideal int4 rel err = {rel*100:.3f}%", flush=True)
    # end-to-end: W x ~= w' (H x quantized int8 per 256)
    x = np.random.default_rng(1234).standard_normal(K).astype(np.float32)
    y_ref = w @ x
    # kernel: x' = H @ x where H = hadamard(256)/16 * signs (quant_rot8's *0.0625 cancels rotate_regs' *16)
    xr = (x.reshape(K//256, 256) @ H.T).reshape(K)
    # quant_rot8: 8 groups of 32 per 256-block, each with its own amax/127 scale
    xv = xr.reshape(K//256, 8, 32)
    d = np.abs(xv).max(-1, keepdims=True) / 127.0
    xq = (np.clip(np.rint(xv / d), -127, 127).astype(np.float32) * d).reshape(K)
    y_int4 = rec @ xq
    rel2 = np.linalg.norm(y_int4 - y_ref) / np.linalg.norm(y_ref)
    print(f"{name} {tag}: end-to-end W@x rel err = {rel2*100:.3f}% (int4 weights + int8 act)", flush=True)
    # independent WS spot check
    WS_ref = np.abs(ddv).max(axis=1) * ((1 << sh)/15.0)
    print(f"  WS max diff = {np.abs(WS - WS_ref).max():.4g}", flush=True)
    # Same 11-12% round-trip error as the expert path (validate_int4.py: 11.73%): the int4 weights'
    # per-128 group quant plus the per-32 int8 activation quant, not a layout error. The exact
    # rotation identity (unquantized W'H x' vs Wx) is ~1e-7; an inverted repack gives ~100%.
    assert rel < 0.05 and rel2 < 0.15, (rel, rel2)

# sliding L: wqkvrg = q|k|v|rel|gate; full L: wqrg = q|rel|gate, wk, wv
b = f"blk.{L}."
full = L in (5, 11, 17, 23)
Kc = int(tmap[b+"attn_q.weight"].shape[0]); HD = int(tmap[b+"attn_q.weight"].shape[1])
KD = int(tmap[b+"attn_k.weight"].shape[1]); HR = int(tmap[b+"attn_rel.weight"].shape[1])
wq = dequant_q8(b+"attn_q.weight", HD, Kc); wk = dequant_q8(b+"attn_k.weight", KD, Kc)
wv = dequant_q8(b+"attn_v.weight", KD, Kc); wr = dequant_q8(b+"attn_rel.weight", HR, Kc)
wg = dequant_q8(b+"attn_gate.weight", HD, Kc)
if full:
    check("wqrg", np.concatenate([wq, wr, wg], axis=0), f"L{L:02d}_wqrg")
    check("wk", wk, f"L{L:02d}_wk")
    check("wv", wv, f"L{L:02d}_wv")
else:
    check("wqkvrg", np.concatenate([wq, wk, wv, wr, wg], axis=0), f"L{L:02d}_wqkvrg")
# wo
to = tmap[b+"attn_output.weight"]
wo = dequant_q8(b+"attn_output.weight", int(to.shape[1]), int(to.shape[0]))
check("wo", wo, f"L{L:02d}_wo")
print("VALIDATION DONE", flush=True)