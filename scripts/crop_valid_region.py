"""Crop an aligned slice stack to the region that is valid (non-black) in every slice.

    python scripts/crop_valid_region.py <aligned_folder> [--out <folder>_cropped] [--threshold 0]
                                        [--margin 2] [--dry-run]

Alignment shifts leave black (zero) borders that differ from slice to slice.  The excluded
area of a slice is taken as the black (<= threshold) regions that touch the image border and
are larger than --min-area pixels, so black pores inside the image are not mistaken for it.
The script intersects the valid area of ALL slices, takes the largest axis-aligned rectangle
inside it (works for shifts and rotations), shrinks it by --margin pixels, and writes every
slice cropped to it with the same file names.  --dry-run only reports the rectangle.
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


def excluded_area(im: np.ndarray, threshold: float, min_area: int) -> np.ndarray:
    """Black regions that touch the border and are at least min_area pixels (alignment padding)."""
    black = im <= threshold
    lab, n = ndi.label(black)
    if n == 0:
        return black
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    border = border[border > 0]
    sizes = np.bincount(lab.ravel())
    keep = border[sizes[border] >= min_area]
    return np.isin(lab, keep)


def natural_key(p: Path):
    import re
    m = re.findall(r"\d+", p.stem)
    return (int(m[-1]) if m else -1, p.name)


def largest_rectangle(valid: np.ndarray):
    """Largest all-True axis-aligned rectangle (y0, y1, x0, x1) in a boolean matrix (histogram method)."""
    h, w = valid.shape
    heights = np.zeros(w, dtype=np.int64)
    best = (0, 0, 0, 0, 0)        # area, y0, y1, x0, x1
    for y in range(h):
        heights = np.where(valid[y], heights + 1, 0)
        stack = []
        for x in range(w + 1):
            cur = heights[x] if x < w else 0
            start = x
            while stack and stack[-1][1] > cur:
                i, hh = stack.pop()
                area = hh * (x - i)
                if area > best[0]:
                    best = (area, y - hh + 1, y + 1, i, x)
                start = i
            stack.append((start, cur))
    return best[1:]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default=None)
    ap.add_argument("--threshold", type=float, default=0, help="pixels <= threshold count as excluded (black)")
    ap.add_argument("--margin", type=int, default=2, help="extra pixels removed on every side")
    ap.add_argument("--min-area", type=int, default=2000, help="smallest border-touching black region counted as excluded area (px)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    src = Path(a.folder)
    files = sorted([p for p in src.iterdir() if p.suffix.lower() in (".tif", ".tiff")], key=natural_key)
    if not files:
        raise SystemExit(f"no TIFF slices in {src}")
    valid = None
    for i, f in enumerate(files):
        im = tifffile.imread(str(f))
        if im.ndim == 3:
            im = im[..., 0]
        v = ~excluded_area(im, a.threshold, a.min_area)
        valid = v if valid is None else (valid & v)
        if i % 100 == 0:
            print(f"  scanned {i + 1}/{len(files)} slices, valid pixels so far {valid.mean() * 100:.1f}%", flush=True)
    h, w = valid.shape
    y0, y1, x0, x1 = largest_rectangle(valid)
    y0, x0 = y0 + a.margin, x0 + a.margin
    y1, x1 = y1 - a.margin, x1 - a.margin
    print(f"{len(files)} slices of {h} x {w}; valid in all slices: {valid.mean() * 100:.1f}% of pixels")
    print(f"crop rectangle: rows {y0}:{y1}, cols {x0}:{x1}  ->  {y1 - y0} x {x1 - x0} px  ({(y1 - y0) * 0.05:.1f} x {(x1 - x0) * 0.05:.1f} um at 50 nm/px)")
    if a.dry_run:
        return
    out = Path(a.out) if a.out else src.parent / (src.name + "_cropped")
    out.mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(files):
        im = tifffile.imread(str(f))
        tifffile.imwrite(str(out / f.name), im[y0:y1, x0:x1])
        if i % 100 == 0:
            print(f"  wrote {i + 1}/{len(files)}", flush=True)
    with open(out / "crop_info.txt", "w") as fh:
        fh.write(f"source {src}\nrows {y0}:{y1}\ncols {x0}:{x1}\nthreshold {a.threshold}\nmin_area {a.min_area}\nmargin {a.margin}\n")
    print("done:", out)


if __name__ == "__main__":
    main()
