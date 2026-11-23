# Extend the INT4 expert conversion to layers 20-23 (Q8_0 -> rotated INT4) for the 1M-context VRAM budget.
# Reuses the validated nvfp4_to_int4.py rotation/repack, but dequantizes Q8_0 blocks instead of NVFP4.
# Output goes to the SAME int4-bin dir so the engine's l<24 INT4 load picks it up.
import sys, os, json, time
import numpy as np
sys.path.insert(0, r"gguf-py")
sys.path.insert(0, r".")
from gguf import GGUFReader
import nvfp4_to_int4 as C   # rot_matrix, quantize_q4_0_rot, repack, SH8

SRC = r"Inkling-2-Mini-UD-Q4_K_XL.gguf"
OUT = r"int4-bin"
GW, SH = 4, C.SH8
H = C.rot_matrix()

def dequant_q8_0(raw, M, K):
    """Q8_0 block = 2-byte fp16 scale + 32 int8 codes (34 bytes / 32 values)."""
    nb = K // 32
    b = np.asarray(raw, dtype=np.uint8).reshape(M, nb, 34)
    d = b[:, :, 0:2].copy().view(np.float16).astype(np.float32).reshape(M, nb)   # [M, nb]
    q = b[:, :, 2:].view(np.int8).astype(np.float32)                            # [M, nb, 32]
    return (q * d[:, :, None]).reshape(M, K)

def main():
    r = GGUFReader(SRC)
    tmap = {t.name: t for t in r.tensors}
    os.makedirs(OUT, exist_ok=True)
    man_path = os.path.join(OUT, "manifest.json")
    manifest = json.load(open(man_path)) if os.path.exists(man_path) else {"gw": GW, "sh": SH, "rotation_block": 256, "layers": []}
    t0 = time.time()
    for l in range(20, 24):
        for role, nm, M, K in [("gate_up", f"blk.{l}.ffn_gate_up_exps", 2304, 2304),
                               ("down", f"blk.{l}.ffn_down_exps", 2304, 1152)]:
            t = tmap[nm + ".weight"]
            tt = int(t.tensor_type)
            E = int(t.shape[2])
            Kpad = ((K + 255) // 256) * 256
            raw = np.asarray(t.data)
            Wp_all, Wh_all, WS_all = [], [], []
            for e in range(E):
                if tt == 8:   # Q8_0
                    w = dequant_q8_0(raw.reshape(E, M, K * 34 // 32)[e], M, K)
                elif tt == 40:  # NVFP4 (fallback)
                    w = C.dequant_nvfp4(raw.reshape(E, M, K * 9 // 16)[e], M, K)
                    w = w * np.asarray(tmap[nm + ".scale"].data, dtype=np.float32)[e]
                else:
                    raise SystemExit(f"L{l} {role}: unexpected tensor_type {tt}")
                if Kpad != K:
                    wp = np.zeros((M, Kpad), dtype=np.float32); wp[:, :K] = w; w = wp
                blocks, dd = C.quantize_q4_0_rot(w, GW, H)
                Wp, Wh, WS = C.repack(blocks, dd, M, Kpad, SH)
                Wp_all.append(Wp); Wh_all.append(Wh); WS_all.append(WS)
                if e == 0:
                    print(f"L{l} {role} e0: type={tt} M={M} K={K} Kpad={Kpad} Wp={Wp.size}B Wh={Wh.size}B WS={WS.size}f", flush=True)
            tag = f"L{l:02d}_{role}"
            np.save(os.path.join(OUT, tag + "_Wp.npy"), np.concatenate(Wp_all))
            np.save(os.path.join(OUT, tag + "_Wh.npy"), np.concatenate(Wh_all))
            np.save(os.path.join(OUT, tag + "_WS.npy"), np.concatenate(WS_all))
            manifest["layers"] = [x for x in manifest["layers"] if not (x["layer"] == l and x["role"] == role)]
            manifest["layers"].append({"layer": l, "role": role, "M": M, "K": K, "Kpad": Kpad, "E": E,
                                       "Wp_bytes_per_expert": int(Wp_all[0].size), "Wh_bytes_per_expert": int(Wh_all[0].size)})
            print(f"  wrote {tag} ({E} experts) {time.time()-t0:.1f}s", flush=True)
    json.dump(manifest, open(man_path, "w"), indent=2)
    print(f"DONE {time.time()-t0:.1f}s -> {OUT}", flush=True)

if __name__ == "__main__":
    main()