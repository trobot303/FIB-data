"""Score crack post-processing variants (seeds, growth threshold, length limit, width refinement) on the held-out
crack probability maps written by scripts/eval_cracks_loso.py --out results/loso_v3, and render
docs/overlays/LCO70_cycled_cracks_current_vs_relaxed.png.  Run from the repository root."""
import sys, json, glob, importlib.util
import numpy as np, tifffile
from scipy import ndimage as ndi
sys.path.insert(0, ".")
from fib_rf_seg.scribbles import rasterise
spec = importlib.util.spec_from_file_location("g", "scripts/grow_cracks_3d.py"); g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
S = {"0015": "LCO70_cycled_slice-0015_cracks_user.json", "0022": "LCO70_cycled_slice-0022_cracks_user.json", "0038": "LCO70_cycled_slice-0038_cracks_user.json"}
def load(s):
    d = f"results/loso_v3/LCO70_cycled_slice-{s}"
    pc = tifffile.imread(glob.glob(d + "/crack_proba/*")[0])
    l2 = tifffile.imread(glob.glob(d + "/labels/*")[0])
    st = json.load(open("data/labels/" + S[s]))
    for k in st["strokes"]: k["brush"] = 0
    line = rasterise(st, pc.shape) == 4
    return pc, l2, line, d
def variant(pc, l2, raw_dir, seed_t, low_t, min_len, refine, seed_from_model=False):
    lab = l2.copy()[None]
    if not seed_from_model:            # seeds straight from the probability, no shape filter
        lab[(lab == 4)] = 2
        lab[0][(pc > seed_t * 255) & (np.isin(l2, (2, 3, 4)))] = 4
    orig = lab.copy()
    grown = g.grow(lab, pc[None], int(low_t * 255), 26)
    lab[grown & (lab != 4)] = 4
    if min_len: lab = g.drop_short(lab, orig, min_len, 26)
    if refine:
        stem = glob.glob(raw_dir + "/input/*")[0].split("/")[-1].rsplit(".", 1)[0]
        from pathlib import Path
        lab = g.refine_width(lab, orig, Path(raw_dir + "/input"), [stem])
    return lab[0] == 4
V = {"current chain (model seeds, grow .2, len 25, refine)": dict(seed_t=None, low_t=.2, min_len=25, refine=True, seed_from_model=True),
     "seeds p>.5, grow .2, len 25, refine": dict(seed_t=.5, low_t=.2, min_len=25, refine=True),
     "seeds p>.5, grow .2, len 25, no refine": dict(seed_t=.5, low_t=.2, min_len=25, refine=False),
     "seeds p>.5, grow .2, len 12, no refine": dict(seed_t=.5, low_t=.2, min_len=12, refine=False),
     "seeds p>.4, grow .15, len 25, no refine": dict(seed_t=.4, low_t=.15, min_len=25, refine=False),
     "seeds p>.4, grow .15, len 25, refine": dict(seed_t=.4, low_t=.15, min_len=25, refine=True)}
data = {s: load(s) for s in S}
for name, kw in V.items():
    rec, out, px = [], [], []
    for s, (pc, l2, line, d) in data.items():
        m = variant(pc, l2, d, **kw)
        rec.append((ndi.binary_dilation(m, iterations=2) & line).sum() / line.sum())
        out.append((m & ~ndi.binary_dilation(line, iterations=4)).sum()); px.append(m.sum())
    print(f"{name:55s} recall {np.mean(rec):.3f}  ({' '.join(f'{r:.2f}' for r in rec)})  crack px {np.mean(px):6.0f}  outside {np.mean(out):6.0f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(3, 3, figsize=(15, 15.6))
for r, s in enumerate(S):
    pc, l2, line, d = data[s]
    raw = tifffile.imread(glob.glob(d + "/input/*")[0]).astype(np.float32)
    lo, hi = np.percentile(raw, [0.5, 99.5]); gr = np.clip((raw - lo) / (hi - lo), 0, 1)
    cur = variant(pc, l2, d, **V["current chain (model seeds, grow .2, len 25, refine)"])
    rel = variant(pc, l2, d, **V["seeds p>.5, grow .2, len 12, no refine"])
    y0, x0 = 250, 250; sl = (slice(y0, y0 + 500), slice(x0, x0 + 500))
    for c, (m, t) in enumerate(((None, "SEM"), (cur, f"current chain: {int(cur.sum())} crack px"), (rel, f"relaxed (len 12, no width trim): {int(rel.sum())} px"))):
        rgb = np.repeat(gr[..., None], 3, -1)
        if m is not None:
            rgb[ndi.binary_dilation(m)] = (0.92, 0.41, 0.20)
            rgb[ndi.binary_dilation(line) & ~ndi.binary_dilation(m, iterations=2)] = (0.16, 0.47, 0.84)
        ax[r, c].imshow(rgb[sl]); ax[r, c].set_axis_off(); ax[r, c].set_title(f"slice {s} (centre 500x500 px = 10 um): {t}", fontsize=9)
fig.suptitle("orange = detected crack, blue = annotated crack NOT detected (models never saw the slice)", fontsize=11)
fig.tight_layout(); fig.savefig("docs/overlays/LCO70_cycled_cracks_current_vs_relaxed.png", dpi=110)
