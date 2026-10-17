"""
Merge two already-registered BSE and SE(TLD) FIB-SEM stacks into a single
weighted-blend stack. No curtaining removal, no denoising, no segmentation -
just the blend:

    merged = bse_weight * BSE + (1 - bse_weight) * SE

Self-contained (doesn't depend on any sibling script) - portable to drop into
any FIB-SEM working directory.

Usage
-----
    python merge_bse_se.py
    python merge_bse_se.py --bse-weight 0.75
    python merge_bse_se.py --bit-depth 8
    python merge_bse_se.py --bse-dir "..." --se-dir "..." --out-dir "..."

Output
------
    <out-dir>/  the blended stack, same filenames as the BSE input
"""

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile

# Both stacks must already be registered/cropped to the same size, with
# filenames ending in a slice number (e.g. "..._0001.tif").
SLICE_RE = re.compile(r"(\d{3,6})\.tiff?$", re.IGNORECASE)

DEFAULT_BSE_DIR = '/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/LE BSE'
DEFAULT_SE_DIR = '/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/E-T'


def index_files(folder: Path) -> dict[int, Path]:
    """Map slice index (parsed from filename) -> file path."""
    files = {}
    for f in sorted(folder.glob("*.tif*")):
        m = SLICE_RE.search(f.name)
        if m:
            files[int(m.group(1))] = f
    return files


def to_output_dtype(img_float01: np.ndarray, bit_depth: int) -> np.ndarray:
    img_float01 = np.clip(img_float01, 0.0, 1.0)
    if bit_depth == 8:
        return (img_float01 * 255).astype(np.uint8)
    return (img_float01 * 65535).astype(np.uint16)


def img_as_float01(img: np.ndarray) -> np.ndarray:
    """Scale an integer image to float in [0, 1] based on its dtype's max value."""
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img, 0.0, 1.0).astype(np.float64)
    info = np.iinfo(img.dtype)
    return img.astype(np.float64) / info.max


def merge_bse_se(bse: np.ndarray, se: np.ndarray, bse_weight: float) -> np.ndarray:
    return bse_weight * bse + (1 - bse_weight) * se


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bse-dir", default=DEFAULT_BSE_DIR)
    parser.add_argument("--se-dir", default=DEFAULT_SE_DIR)
    parser.add_argument("--out-dir", default=None, help="Default: <bse-dir>/../Merged")
    parser.add_argument("--bse-weight", type=float, default=0.75,
                         help="Weight given to BSE in the blend; SE gets 1 - this. Default 0.5 (75%% BSE / 25%% SE).")
    parser.add_argument("--bit-depth", type=int, choices=[8, 16], default=16)
    args = parser.parse_args()

    bse_dir = Path(args.bse_dir)
    se_dir = Path(args.se_dir)
    out_dir = Path(args.out_dir) if args.out_dir else bse_dir.parent / "Merged"
    out_dir.mkdir(parents=True, exist_ok=True)

    bse_files = index_files(bse_dir)
    se_files = index_files(se_dir)
    common = sorted(set(bse_files) & set(se_files))
    if not common:
        raise SystemExit("No matching slice indices found between the two folders.")
    print(f"Found {len(common)} matched slice pairs.")

    for i, idx in enumerate(common):
        bse = img_as_float01(tifffile.imread(bse_files[idx]))
        se = img_as_float01(tifffile.imread(se_files[idx]))
        if bse.shape != se.shape:
            raise SystemExit(
                f"Slice {idx}: BSE shape {bse.shape} != SE shape {se.shape} - "
                "both stacks must already be registered/cropped to the same size."
            )

        merged = merge_bse_se(bse, se, args.bse_weight)
        tifffile.imwrite(out_dir / bse_files[idx].name, to_output_dtype(merged, args.bit_depth))

        if (i + 1) % 100 == 0 or i == len(common) - 1:
            print(f"  merged {i + 1}/{len(common)} slices...")

    print(f"Done. Merged ({args.bse_weight:.0%} BSE / {1 - args.bse_weight:.0%} SE) stack: {out_dir}")


if __name__ == "__main__":
    main()
