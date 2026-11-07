"""Leave-one-slice-out test of a crack model recipe on annotated slices.

    python scripts/eval_cracks_loso.py --out results/loso_v3 [--held-out LCO70_cycled/slice-0015 ...]
                                       [--recipe "<extra train args>"] [--post "<extra predict args>"]

For each held-out annotated slice: train the recipe on all the other annotated slices (LCO50 cycled ds_1 +
LCO70 cycled), predict the held-out slice, run the cycled post-processing chain (crack shape filter, then
grow_cracks_3d.py with growth, length limit and width refinement on the single slice), and score each stage
against the user's crack strokes:
  recall    share of the annotated crack line pixels with a predicted crack pixel within 2 px
  outside   predicted crack pixels farther than 4 px from any stroke (unannotated cracks or false alarms)
It also scores the raw crack probability at several thresholds (crack_proba > t), which shows how much is
lost to the probability cut alone.  Results: <out>/loso.csv and per-slice PNGs <out>/<slice>_eval.png.
"""
import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fib_rf_seg.scribbles import rasterise  # noqa: E402

ANNOTATED = {   # slice -> crack strokes
    "LCO50_cycled/slice-045": "LCO50_cycled_slice-045_cracks_user.json",
    "LCO50_cycled/slice-053": "LCO50_cycled_slice-053_cracks_user.json",
    "LCO50_cycled/slice-262": "LCO50_cycled_slice-262_cracks_user.json",
    "LCO50_cycled/slice-272": "LCO50_cycled_slice-272_cracks_user.json",
    "LCO70_cycled/slice-0015": "LCO70_cycled_slice-0015_cracks_user.json",
    "LCO70_cycled/slice-0022": "LCO70_cycled_slice-0022_cracks_user.json",
    "LCO70_cycled/slice-0038": "LCO70_cycled_slice-0038_cracks_user.json",
}
RECIPE_V3 = ("--auto-seeds --dense --pore-level auto --norm phase --smooth 0.7 --destripe --class-weight 4 "
             "--max-per-class 20000 --trees 200 --max-depth 16 --oriented --curtaining-negatives --shape ridge")
POST = "--min-size 6 --crack-min-length 12 --crack-min-aspect 3"
THRESHOLDS = (0.5, 0.4, 0.3, 0.2, 0.1)


def run(cmd):
    print("  $", " ".join(shlex.quote(c) for c in cmd[:6]), "...", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT, stdout=subprocess.DEVNULL)


def score(pred, line, near2, far4):
    found = ndi.binary_dilation(pred, iterations=2) & line
    return {"recall": float(found.sum() / max(line.sum(), 1)), "outside": int((pred & far4).sum()),
            "crack_px": int(pred.sum())}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--held-out", nargs="+", default=[k for k in ANNOTATED if k.startswith("LCO70")])
    ap.add_argument("--recipe", default=RECIPE_V3, help="train arguments (default: the rf_cracks_lco50_v3 recipe)")
    ap.add_argument("--post", default=POST, help="predict post-processing arguments")
    ap.add_argument("--low", type=float, default=0.2, help="growth threshold of grow_cracks_3d.py")
    ap.add_argument("--min-length", type=float, default=25)
    a = ap.parse_args()
    out = (ROOT / a.out) if not Path(a.out).is_absolute() else Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for h in a.held_out:
        name = h.replace("/", "_")
        d = out / name
        model = d / "model.joblib"
        d.mkdir(parents=True, exist_ok=True)
        if not model.exists():
            train = []
            for k, j in ANNOTATED.items():
                if k != h:
                    train += ["--image", str(next((ROOT / "data").glob(k + ".tif*"))), "--scribbles", str(ROOT / "data/labels" / j)]
            run([sys.executable, "-m", "fib_rf_seg", "train", *train, *shlex.split(a.recipe), "--model", str(model)])
        img = next((ROOT / "data").glob(h + ".tif*"))
        inp = d / "input"; inp.mkdir(parents=True, exist_ok=True)
        link = inp / img.name
        if not link.exists():
            link.symlink_to(img)
        run([sys.executable, "-m", "fib_rf_seg", "predict", "--model", str(model), "--input", str(inp), "--out", str(d),
             *shlex.split(a.post), "--crack-proba", "--overwrite"])
        run([sys.executable, "scripts/grow_cracks_3d.py", "--labels", str(d / "labels"), "--proba", str(d / "crack_proba"),
             "--out", str(d / "labels_3d"), "--low", str(a.low), "--min-length", str(a.min_length), "--raw", str(inp),
             "--refine-width", "--no-stacks"])
        stem = img.stem
        lab2 = tifffile.imread(str(d / "labels" / f"{stem}_labels.tif"))
        lab3 = tifffile.imread(str(d / "labels_3d" / f"{stem}_labels.tif"))
        pc = tifffile.imread(str(d / "crack_proba" / f"{stem}_crackp.tif")) / 255.0
        strokes = json.load(open(ROOT / "data/labels" / ANNOTATED[h]))
        for s in strokes["strokes"]:
            s["brush"] = 0
        ann = rasterise(strokes, lab2.shape)
        line = ann == 4
        far4 = ~ndi.binary_dilation(line, iterations=4)
        res = {"slice": h, "annotated_px": int(line.sum())}
        for t in THRESHOLDS:
            res.update({f"p>{t}_{k}": v for k, v in score(pc > t, line, None, far4).items()})
        res.update({f"model_{k}": v for k, v in score(lab2 == 4, line, None, far4).items()})
        res.update({f"chain_{k}": v for k, v in score(lab3 == 4, line, None, far4).items()})
        rows.append(res)
        print(f"{h}: annotated {res['annotated_px']} px | p>0.5 recall {res['p>0.5_recall']:.2f} | "
              f"p>0.2 {res['p>0.2_recall']:.2f} | p>0.1 {res['p>0.1_recall']:.2f} | after model+filter "
              f"{res['model_recall']:.2f} | full chain {res['chain_recall']:.2f} (outside {res['chain_outside']} px)", flush=True)
        figure(out / f"{name}_eval.png", tifffile.imread(str(img)), line, lab3 == 4, pc, h)
    keys = list(rows[0])
    with open(out / "loso.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys); w.writeheader(); w.writerows(rows)
        mean = {k: (np.mean([r[k] for r in rows]) if k != "slice" else "mean") for k in keys}
        w.writerow(mean)
    print(f"mean recall: p>0.5 {mean['p>0.5_recall']:.3f}, p>0.2 {mean['p>0.2_recall']:.3f}, p>0.1 {mean['p>0.1_recall']:.3f}, "
          f"model+filter {mean['model_recall']:.3f}, full chain {mean['chain_recall']:.3f}; "
          f"outside px (chain) {mean['chain_outside']:.0f}")
    print("wrote", out / "loso.csv")


def figure(path, raw, line, pred, pc, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = raw.astype(np.float32); lo, hi = np.percentile(g, [0.5, 99.5]); g = np.clip((g - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(g[..., None], 3, -1)
    hit = ndi.binary_dilation(pred, iterations=2)
    missed = line & ~hit
    found = line & hit
    extra = pred & ~ndi.binary_dilation(line, iterations=4)
    rgb[ndi.binary_dilation(extra)] = (0.92, 0.41, 0.20)     # orange: detected, not annotated
    rgb[ndi.binary_dilation(found)] = (0.16, 0.47, 0.84)     # blue: annotated and found
    rgb[ndi.binary_dilation(missed)] = (0.89, 0.29, 0.28)    # red: annotated, missed
    fig, ax = plt.subplots(1, 2, figsize=(16, 8.4))
    ax[0].imshow(rgb); ax[0].set_title(f"{title}: blue = annotated & found, red = annotated & MISSED, orange = detected only", fontsize=9)
    ax[1].imshow(pc, cmap="magma", vmin=0, vmax=1); ax[1].set_title("crack probability (0-1)", fontsize=9)
    for x in ax:
        x.set_axis_off()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()
