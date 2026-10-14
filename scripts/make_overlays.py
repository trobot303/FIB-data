"""Write overlay images (SEM grey + segmentation) for a slice folder and its label folder.

    python scripts/make_overlays.py <block_folder> [--raw raw] [--labels labels_v2] [--out overlays_v2]
                                    [--phases crack | all] [--every 1] [--alpha 0.6] [--voxel-um 0.02] [--bar-um 5]

For every raw/slice-XXXX.tif with a labels/slice-XXXX_labels.tif, writes <out>/slice-XXXX.png:
the grey slice (0.5-99.5 percentile stretch) with cracks in red (--phases crack, default) or all
four phases coloured (--phases all), plus a scale bar.  Pass several block folders to do them all.
"""
import argparse
import re
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image, ImageDraw

COLORS = {1: (30, 30, 220), 2: (30, 190, 30), 3: (240, 50, 50), 4: (255, 40, 40)}   # pore, LCO, LLZTO, crack


def natural_key(p: Path):
    m = re.findall(r"\d+", p.stem)
    return (int(m[-1]) if m else -1, p.name)


def overlay_png(grey, lab, out, phases, alpha, px_um, bar_um):
    g = grey.astype(np.float32)
    lo, hi = np.percentile(g[g > 0], [0.5, 99.5]) if (g > 0).any() else (g.min(), g.max())
    g8 = np.clip((g - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    rgb = np.repeat(g8[..., None], 3, axis=-1).astype(np.float32)
    for k in phases:
        m = lab == k
        if k == 4:                                            # cracks: make them visible (dilate 1 px), fully opaque
            from scipy import ndimage as ndi
            m = ndi.binary_dilation(m, iterations=1)
            rgb[m] = COLORS[k]
        else:
            rgb[m] = (1 - alpha) * rgb[m] + alpha * np.array(COLORS[k], dtype=np.float32)
    im = Image.fromarray(rgb.astype(np.uint8))
    # scale bar on a grey box, lower right
    h, w = g8.shape
    L = int(round(bar_um / px_um)); bh = max(3, h // 120); pad = w // 40
    d = ImageDraw.Draw(im)
    x1, y1 = w - pad, h - pad
    d.rectangle([x1 - L - bh, y1 - 4 * bh, x1 + bh, y1 + bh // 2], fill=(90, 90, 90))
    d.rectangle([x1 - L, y1 - bh, x1, y1], fill=(255, 255, 255))
    d.text((x1 - L, y1 - 3.6 * bh), f"{bar_um:g} um", fill=(255, 255, 255))
    im.save(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folders", nargs="+")
    ap.add_argument("--raw", default="raw"); ap.add_argument("--labels", default="labels_v2"); ap.add_argument("--out", default="overlays_v2")
    ap.add_argument("--phases", choices=("crack", "all"), default="crack")
    ap.add_argument("--every", type=int, default=1, help="write every N-th slice")
    ap.add_argument("--alpha", type=float, default=0.45)
    ap.add_argument("--voxel-um", type=float, default=0.02, help="in-plane pixel size in um")
    ap.add_argument("--bar-um", type=float, default=5.0)
    a = ap.parse_args()
    phases = [4] if a.phases == "crack" else [1, 2, 3, 4]
    for folder in a.folders:
        folder = Path(folder)
        raw_dir = folder / a.raw if (folder / a.raw).is_dir() else folder
        lab_dir = folder / a.labels
        out_dir = folder / a.out; out_dir.mkdir(exist_ok=True)
        files = sorted([p for p in raw_dir.iterdir() if p.suffix.lower() in (".tif", ".tiff")], key=natural_key)[::a.every]
        n = 0
        for f in files:
            lp = lab_dir / f"{f.stem}_labels.tif"
            if not lp.exists():
                continue
            overlay_png(tifffile.imread(str(f)), tifffile.imread(str(lp)), out_dir / f"{f.stem}.png", phases, a.alpha, a.voxel_um, a.bar_um)
            n += 1
        print(f"{folder.name}: {n} overlays -> {out_dir}")


if __name__ == "__main__":
    main()
