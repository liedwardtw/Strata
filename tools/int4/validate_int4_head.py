"""Validate the IK2_INT4_HEAD out_w sidecar (CPU only, no GPU runs).

Checks the k_repack layout of out_w_{Wp,Wh,WS}.bin against a fresh Q8_0 dequant
of output.weight from the GGUF, including rows spanning the converter's chunk
boundaries (a chunked conversion must be byte-identical to a whole-matrix one).

Usage: python tools/validate_int4_head.py [out_dir] [gguf]
"""
import os, sys
import numpy as np
sys.path.insert(0, r".")
sys.path.insert(0, r"gguf-py")
import nvfp4_to_int4 as C
import q8_dense_to_int4 as Q
from gguf import GGUFReader

out = sys.argv[1] if len(sys.argv) > 1 else r"int4-dense-bin"
gguf = sys.argv[2] if len(sys.argv) > 2 else r"Inkling-2-Mini-UD-Q4_K_XL.gguf"
gw, sh = 4, C.SH8
H = C.rot_matrix()
r = GGUFReader(gguf)
t = next(x for x in r.tensors if x.name == "output.weight")
K, N = int(t.shape[0]), int(t.shape[1])
print(f"output.weight Q8_0 type={int(t.tensor_type)} shape=[{K},{N}]", flush=True)
assert int(t.tensor_type) == 8 and K == 2304 and N == 201024

Wp = np.fromfile(os.path.join(out, "out_w_Wp.bin"), dtype=np.uint8)
Wh = np.fromfile(os.path.join(out, "out_w_Wh.bin"), dtype=np.uint8)
WS = np.fromfile(os.path.join(out, "out_w_WS.bin"), dtype=np.float32)
assert Wp.size == N * K // 2 and Wh.size == (N // 16) * (K // 256) * 64 and WS.size == N, \
    (Wp.size, Wh.size, WS.size)


def reconstruct_rows(rows, M, K, sh):
    """Independent inverse of k_repack for the given 16-row-aligned ft block indices.
    Returns [M, K] float32 reconstructed from the sidecar, exactly as the kernel reads it."""
    nsb = K // 256
    ft0 = rows[0]
    Wpb = Wp.view(np.uint32).reshape(N // 16, nsb, 8, 16, 4)[ft0:ft0+M]
    Whb = Wh.reshape(N // 16, nsb, 16, 4)[ft0:ft0+M]
    ws = WS.reshape(N)[ft0*16:(ft0+M)*16]
    codes = np.zeros((M, 16, nsb, 8, 32), dtype=np.int32)
    for sl in range(2):
        for dw in range(2):
            for s2 in range(8):
                e = 16 * sl + C.P(dw * 8 + s2)
                nib = (Wpb[:, :, :, :, sl*2+dw] >> (4*s2)) & 0xF    # [ft,sb,g8,r]
                codes[:, :, :, :, e] = (((nib ^ 8).astype(np.int32)) - 8).transpose(0, 3, 1, 2)
    sub = np.zeros((M, 16, nsb, 8), dtype=np.float32)
    for g8 in range(8):
        byte = Whb[:, :, :, g8//2]
        sub[:, :, :, g8] = ((byte >> (4*(g8 % 2))) & 0xF).astype(np.float32).transpose(0, 2, 1)
    sc = ws.reshape(M, 16, 1, 1) * sub / float(1 << sh)
    return (codes * sc[..., None]).reshape(M * 16, K)


def check(ft):
    r0 = ft * 16
    raw = np.asarray(t.data, dtype=np.uint8)[r0:r0+16].reshape(16, K//32, 34)
    d = raw[..., 0:2].copy().view(np.float16).astype(np.float32)
    q = raw[..., 2:34].copy().view(np.int8).astype(np.float32)
    w = (d * q).reshape(16, K)
    blocks, dd = C.quantize_q4_0_rot(w, gw, H)
    b = blocks.reshape(16, K//32, 18)
    ddv = dd.reshape(16, K//32)
    qs = b[..., 2:]
    code = np.concatenate([(qs & 15).astype(np.int32), (qs >> 4).astype(np.int32)], -1) - 8
    ideal = (ddv[:, :, None] * code).reshape(16, K)
    rec = reconstruct_rows([ft], 1, K, sh)
    rel = np.linalg.norm(rec - ideal) / np.linalg.norm(ideal)
    # end-to-end: int4 sidecar * int8-rotated activation vs Q8_0 weight * activation
    x = np.random.default_rng(1234 + ft).standard_normal(K).astype(np.float32)
    y_ref = w @ x
    xr = (x.reshape(K//256, 256) @ H.T).reshape(K)
    xv = xr.reshape(K//256, 8, 32)
    dx = np.abs(xv).max(-1, keepdims=True) / 127.0
    xq = (np.clip(np.rint(xv / dx), -127, 127).astype(np.float32) * dx).reshape(K)
    rel2 = np.linalg.norm(rec @ xq - y_ref) / np.linalg.norm(y_ref)
    rel2_ideal = np.linalg.norm(ideal @ xq - y_ref) / np.linalg.norm(y_ref)
    # the sidecar's own error must match the ideal int4 weight's end-to-end error (repack sub-scales add only
    # a little); the absolute value varies by row distribution (head vocab rows differ from attention rows).
    print(f"out_w ft={ft}: rel(layout)={rel*100:.3f}% rel(end2end)={rel2*100:.3f}% ideal={rel2_ideal*100:.3f}%", flush=True)
    assert rel < 0.05 and rel2 < 0.25 and rel2 <= rel2_ideal + 0.02, (ft, rel, rel2, rel2_ideal)


# first/last rows and both sides of two converter chunk boundaries (chunk=4096 rows = 256 ft blocks)
for ft in (0, 1, 255, 256, 257, 511, 512, 513, N//16 - 2, N//16 - 1):
    check(ft)
print(f"VALIDATION DONE: out_w Wp={Wp.size}B Wh={Wh.size}B WS={WS.size*4}B", flush=True)
