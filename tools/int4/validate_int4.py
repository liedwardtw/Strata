"""Validate nvfp4_to_int4 converter: round-trip error + independent repack cross-check."""
import sys, numpy as np
sys.path.insert(0, r".")
sys.path.insert(0, r"gguf-py")
import nvfp4_to_int4 as C
from gguf import GGUFReader
from scipy.linalg import hadamard

r = GGUFReader(r"Inkling-2-Mini-UD-Q4_K_XL.gguf")
tmap = {t.name: t for t in r.tensors}
t = tmap["blk.0.ffn_gate_up_exps.weight"]
E = int(t.shape[2]); M, K = 2304, 2304
raw = np.asarray(t.data).reshape(E, M, K*9//16)
sc = np.asarray(tmap["blk.0.ffn_gate_up_exps.scale"].data, dtype=np.float32)

# 1. dequant expert 0
w = C.dequant_nvfp4(raw[0], M, K) * sc[0]
print("dequant w stats: min %.4g max %.4g mean %.4g std %.4g  nan=%d" % (w.min(), w.max(), w.mean(), w.std(), np.isnan(w).sum()))

# 2. rotate + quant + dequant round trip (no repack) to measure int4 quant error
H = C.rot_matrix()
gw = 4
blocks, dd = C.quantize_q4_0_rot(w, gw, H)
# dequant the Q4_0 blocks back (rotated space)
b = blocks.reshape(M, K//32, 18)
d = b[..., 0:2].copy().view(np.float16).astype(np.float32).reshape(M, K//32, 1)
qs = b[..., 2:]
q = np.concatenate([(qs & 15).astype(np.int8), (qs >> 4).astype(np.int8)], -1) - 8
deq_rot = (q * d).reshape(M, K)
# un-rotate: W = W' (H D)  since W' = W (H D)^T and (H D) orthogonal => W = W' (H D)
wr_orig = (w.reshape(M, K//256, 256) @ H.T).reshape(M, K)
rel_rot = np.linalg.norm(deq_rot - wr_orig) / np.linalg.norm(wr_orig)
print("int4 quant rel error in rotated space: %.4f (%.2f%%)" % (rel_rot, rel_rot*100))
# un-rotate deq
deq = (deq_rot.reshape(M, K//256, 256) @ H).reshape(M, K)   # (H D)^T inverse = (H D) since orthogonal? H symmetric, D diag signs
rel = np.linalg.norm(deq - w) / np.linalg.norm(w)
print("int4 round-trip rel error vs NVFP4 dequant: %.4f (%.2f%%)" % (rel, rel*100))

# 3. independent repack cross-check (re-implement k_repack directly, compare to converter repack)
Wp, Wh, WS = C.repack(blocks, dd, M, K, 12)
# independent: compute WS
nb = K//32; nsb = K//256
WS_ref = np.abs(dd.reshape(M, nb)).max(axis=1) * ((1<<12)/15.0)
print("WS match:", np.allclose(WS, WS_ref, rtol=1e-5), "max diff", np.abs(WS-WS_ref).max())
# spot-check one (m=0, sb=0, g=0) tile against k_repack logic
def P(s):
    if s < 8: return (4+(s>>1)) if (s&1) else (s>>1)
    t=s-8; return 8+((4+(t>>1)) if (t&1) else (t>>1))
m, sb, g = 0, 0, 0
ft, rr = m>>4, m&15
blk = b[m, sb*8+g]
ddv = float(dd.reshape(M,nb)[m, sb*8+g])
sgn = -1 if ddv < 0 else 1
words=[0,0,0,0]
for sl in range(2):
    for dw in range(2):
        word=0
        for s2 in range(8):
            e=16*sl+P(dw*8+s2)
            nib = int(blk[2+e]) if e<16 else (int(blk[2+e-16])>>4)
            # careful: qs byte e holds low nibble elem e, high nibble elem e+16
            if e<16: nib = int(blk[2+e]) & 0xF
            else: nib = int(blk[2+e-16]) >> 4
            qq = nib-8
            if sgn<0: qq = max(-7,min(7,-qq))
            word |= (qq & 15) << (4*s2)
        words[sl*2+dw]=word
off = ((ft*nsb+sb)*8+g)*256 + rr*16
got = Wp.reshape(-1,256)[off//256, off%256:off%256+16].view(np.uint32)
print("repack tile m0sb0g0 match:", list(got)==words, "got", list(got), "ref", words)
print("VALIDATION DONE")