"""Write an RGB overlay stack: SEM grey slices with the crack segmentation painted in red.

    python scripts/overlay_stack.py --raw <slice folder> --labels <labels folder> --out <folder>
                                    [--phases crack|all] [--alpha 0.5] [--dilate 1] [--stretch global|slice]
                                    [--scale-bar --voxel-um 0.02 --bar-um 5] [--no-slices]

Writes <out>/overlay_stack.tif (multi-page RGB uint8, one page per slice, ImageJ-readable) and, unless
--no-slices, <out>/slices/<stem>_overlay.tif per slice.  Cracks (label 4) are painted fully opaque
after a 1 px dilation so they stay visible (--dilate 0 for pixel-exact); the other phases
(--phases all) are blended with --alpha.  Grey stretch: 0.5-99.5 percentiles of a sample of slices
("global", default, same contrast through the stack) or of each slice ("slice").
"""
import argparse
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

COLORS = {1: (30, 30, 220), 2: (30, 190, 30), 3: (240, 50, 50), 4: (255, 0, 0)}   # pore, LCO, LLZTO, crack


def natural_key(p: Path):
    m = re.findall(r"\d+", p.stem)
    return (int(m[-1]) if m else -1, p.name)


def list_tifs(d: Path):
    return sorted([p for p in Path(d).iterdir() if p.suffix.lower() in (".tif", ".tiff")], key=natural_key)


def grey_bounds(files, n=10):
    pick = files[:: max(1, len(files) // n)][:n]
    vals = np.concatenate([tifffile.imread(str(f)).astype(np.float32).ravel()[::16] for f in pick])
    vals = vals[vals > 0] if (vals > 0).any() else vals
    return np.percentile(vals, [0.5, 99.5])


def overlay_rgb(grey, lab, phases, alpha, dilate, lo, hi):
    g = grey.astype(np.float32)
    if lo is None:
        v = g[g > 0] if (g > 0).any() else g
        lo, hi = np.percentile(v, [0.5, 99.5])
    g8 = np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(g8[..., None], 3, axis=-1).astype(np.float32)
    for k in phases:
        m = lab == k
        if k == 4:
            if dilate > 0:
                m = ndi.binary_dilation(m, iterations=dilate)
            rgb[m] = COLORS[k]
        else:
            rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(COLORS[k], dtype=np.float32)
    return rgb.astype(np.uint8)


def add_scale_bar(rgb, px_um, bar_um):
    h, w = rgb.shape[:2]
    L = int(round(bar_um / px_um)); bh = max(3, h // 120); pad = w // 40
    x1, y1 = w - pad, h - pad
    rgb[y1 - 3 * bh:y1 + bh // 2 + 1, x1 - L - bh:x1 + bh + 1] = 90
    rgb[y1 - bh:y1 + 1, x1 - L:x1 + 1] = 255
    return rgb


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder with the SEM slices")
    ap.add_argument("--labels", required=True, help="folder with <stem>_labels.tif")
    ap.add_argument("--out", required=True)
    ap.add_argument("--phases", choices=("crack", "all"), default="crack")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--dilate", type=int, default=1, help="dilate cracks by N px for visibility (0 = exact)")
    ap.add_argument("--stretch", choices=("global", "slice"), default="global")
    ap.add_argument("--scale-bar", action="store_true")
    ap.add_argument("--voxel-um", type=float, default=0.02)
    ap.add_argument("--bar-um", type=float, default=5.0)
    ap.add_argument("--no-slices", action="store_true", help="write only the stack, not the per-slice files")
    ap.add_argument("--stack-name", default="overlay_stack.tif")
    a = ap.parse_args()

    phases = [4] if a.phases == "crack" else [1, 2, 3, 4]
    raw_dir, lab_dir, out = Path(a.raw), Path(a.labels), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = [f for f in list_tifs(raw_dir) if (lab_dir / f"{f.stem}_labels.tif").exists()]
    if not files:
        raise SystemExit(f"no slices in {raw_dir} with labels in {lab_dir}")
    lo = hi = None
    if a.stretch == "global":
        lo, hi = grey_bounds(files)
        print(f"grey stretch {lo:.0f}-{hi:.0f} (0.5-99.5 % of a sample of slices)")
    if not a.no_slices:
        (out / "slices").mkdir(exist_ok=True)
    shape = tifffile.imread(str(files[0])).shape
    big = len(files) * shape[0] * shape[1] * 3 > 3.5e9
    with tifffile.TiffWriter(str(out / a.stack_name), bigtiff=big, imagej=not big) as tw:
        for i, f in enumerate(files, 1):
            rgb = overlay_rgb(tifffile.imread(str(f)), tifffile.imread(str(lab_dir / f"{f.stem}_labels.tif")),
                              phases, a.alpha, a.dilate, lo, hi)
            if a.scale_bar:
                rgb = add_scale_bar(rgb, a.voxel_um, a.bar_um)
            tw.write(rgb, photometric="rgb", contiguous=True)
            if not a.no_slices:
                tifffile.imwrite(str(out / "slices" / f"{f.stem}_overlay.tif"), rgb, photometric="rgb")
            if i % 50 == 0 or i == len(files):
                print(f"  {i}/{len(files)} slices")
    print(f"done: {out / a.stack_name} ({len(files)} pages, {shape[0]}x{shape[1]} RGB)")


if __name__ == "__main__":
    main()
