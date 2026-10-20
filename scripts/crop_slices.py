"""Crop every slice of a stack folder to a fixed size (e.g. the 1001 x 1001 px of LCO70 cycled).

    python scripts/crop_slices.py --in <folder> --out <folder> --size 1001 1001 [--anchor center | --origin Y0 X0]
                                  [--z Z0 Z1] [--dry-run] [--preview preview.png]

--anchor center (default) centres the window in each slice, --origin gives the top-left corner
(row, column) explicitly.  --z keeps only slices Z0..Z1-1 (0-based, by position in the sorted
list).  --dry-run reports the slice count, the slice size and the crop window without writing.
--preview draws the window on the first slice.  Slices keep their file names; TIFFs keep their dtype.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import tifffile


def natural_key(p: Path):
    m = re.findall(r"\d+", p.stem)
    return (int(m[-1]) if m else -1, p.name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, nargs=2, metavar=("H", "W"), default=(1001, 1001))
    ap.add_argument("--anchor", choices=("center", "top-left"), default="center")
    ap.add_argument("--origin", type=int, nargs=2, metavar=("Y0", "X0"), default=None, help="top-left corner of the window (overrides --anchor)")
    ap.add_argument("--z", type=int, nargs=2, metavar=("Z0", "Z1"), default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--preview", default=None, help="PNG of the first slice with the crop window")
    a = ap.parse_args()

    files = sorted([p for p in Path(a.inp).iterdir() if p.suffix.lower() in (".tif", ".tiff")], key=natural_key)
    if not files:
        raise SystemExit(f"no TIFF slices in {a.inp}")
    nz = len(files)
    if a.z:
        files = files[a.z[0]:a.z[1]]
    first = tifffile.imread(str(files[0]))
    H, W = first.shape[:2]
    h, w = a.size
    if h > H or w > W:
        raise SystemExit(f"crop {h}x{w} larger than the slices ({H}x{W})")
    if a.origin:
        y0, x0 = a.origin
    elif a.anchor == "center":
        y0, x0 = (H - h) // 2, (W - w) // 2
    else:
        y0, x0 = 0, 0
    if y0 < 0 or x0 < 0 or y0 + h > H or x0 + w > W:
        raise SystemExit(f"window rows {y0}-{y0 + h}, cols {x0}-{x0 + w} does not fit in {H}x{W}")
    print(f"{nz} slices of {H}x{W} px ({first.dtype}) in {a.inp}; keeping {len(files)} slices")
    print(f"crop window: rows {y0}-{y0 + h - 1}, cols {x0}-{x0 + w - 1} ({h}x{w} px)")
    if a.preview:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
        g = first.astype(np.float32)
        lo, hi = np.percentile(g, [0.5, 99.5])
        fig, ax = plt.subplots(figsize=(10, 10 * H / W))
        ax.imshow(g, cmap="gray", vmin=lo, vmax=hi)
        ax.add_patch(Rectangle((x0, y0), w, h, fill=False, edgecolor="red", linewidth=2))
        ax.set_title(f"{files[0].name}: crop rows {y0}-{y0 + h - 1}, cols {x0}-{x0 + w - 1}")
        ax.axis("off"); plt.tight_layout(); plt.savefig(a.preview, dpi=100); plt.close(fig)
        print(f"preview: {a.preview}")
    if a.dry_run:
        return
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files, 1):
        im = tifffile.imread(str(f))
        if im.shape[:2] != (H, W):
            raise SystemExit(f"{f.name}: shape {im.shape[:2]} differs from the first slice {H}x{W}")
        tifffile.imwrite(str(out / f.name), np.ascontiguousarray(im[y0:y0 + h, x0:x0 + w]))
        if i % 100 == 0 or i == len(files):
            print(f"  {i}/{len(files)}")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
