"""Convert int4-all .npy sidecars to raw little-endian .bin for the C++ ik2 loader.
Keeps the exact same filenames/layout; nothing else changes. Idempotent."""
import os, sys
import numpy as np

SRC = r"int4-all"
DST = r"int4-bin"
os.makedirs(DST, exist_ok=True)

count = 0
total = 0
for fn in sorted(os.listdir(SRC)):
    if not fn.endswith(".npy"):
        continue
    arr = np.load(os.path.join(SRC, fn))
    assert arr.flags.c_contiguous, fn
    out = os.path.join(DST, fn[:-4] + ".bin")
    arr.tofile(out)
    n = arr.nbytes
    total += n
    count += 1
    if count % 20 == 0:
        print(f"converted {count} files, {total/1e9:.2f} GB", flush=True)

print(f"DONE {count} files {total/1e9:.3f} GB -> {DST}", flush=True)
