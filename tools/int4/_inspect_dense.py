import sys
sys.path.insert(0, r"gguf-py")
from gguf import GGUFReader
r = GGUFReader(r"Inkling-2-Mini-UD-Q4_K_XL.gguf")
def g(k):
    f = r.get_field("inkling2." + k)
    return f.contents() if f else None
for k in ["block_count","embedding_length","attention.head_count","attention.head_count_kv",
          "attention.key_length","attention.relative_length","num_loops","loop.core_start",
          "loop.core_end","expert_feed_forward_length"]:
    print(k, "=", g(k))
pat = g("attention.sliding_window_pattern")
print("pattern len", len(pat) if pat else None, "first 12", pat[:12] if pat else None)
n_visit = g("block_count"); n_pass = g("num_loops") or 1
core0 = g("loop.core_start") or 0; core1 = g("loop.core_end") or 0
n_core = core1 - core0
n_layer = n_visit - (n_pass-1)*n_core
print("n_visit", n_visit, "n_pass", n_pass, "core0", core0, "core1", core1, "n_layer", n_layer)
sliding = [pat[l if l < core1 else l + (n_pass-1)*n_core] != 0 for l in range(n_layer)]
print("full layers:", [l for l in range(n_layer) if not sliding[l]])
tmap = {t.name: t for t in r.tensors}
for l in [0, 5]:
    for nm in ["attn_q.weight","attn_k.weight","attn_v.weight","attn_rel.weight","attn_r.weight",
               "attn_gate.weight","attn_output.weight"]:
        t = tmap.get(f"blk.{l}.{nm}")
        if t is None:
            print(f"L{l} {nm}: MISSING"); continue
        print(f"L{l} {nm}: type={int(t.tensor_type)} shape={list(t.shape)} nbytes={t.n_bytes}")