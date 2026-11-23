import sys, struct
import numpy as np
p = sys.argv[1] if len(sys.argv)>1 else r'pop-pop16k.bin'
raw = open(p,'rb').read()
nl, E = struct.unpack('<ii', raw[:8])
a = np.frombuffer(raw[8:], dtype='<u8').reshape(nl, E)
print(f'layers={nl} experts={E} total_picks={a.sum()}')
for H in (16,32,64,96,128):
    # per-layer top-H mass, then worst layer
    part = np.sort(a, axis=1)[:, ::-1][:, :H].sum(axis=1) / a.sum(axis=1).clip(min=1)
    print(f'top-{H:3d}: mean={part.mean()*100:6.2f}%  min-layer={part.min()*100:6.2f}%  layer-with-min={part.argmin()}')
# global (all layers pooled) top-H
g = np.sort(a.reshape(-1))[::-1]; tot=g.sum()
for H in (16,32,64,96):
    print(f'pooled top-{H:3d}: {g[:H].sum()/tot*100:6.2f}%')
# per-layer most/least used expert
for l in (0,5,6,11,17,23):
    idx=np.argsort(a[l])[::-1]
    print(f'layer {l:2d}: top5 ids={idx[:5].tolist()} counts={a[l][idx[:5]].tolist()} min={a[l].min()} zeros={(a[l]==0).sum()}')