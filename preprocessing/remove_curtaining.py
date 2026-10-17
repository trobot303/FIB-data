"""
Remove FIB milling curtaining stripes from the merged BSE/SE FIB-SEM stack
(output of merge_bse_se.py) via an FFT notch filter. No denoising, no
segmentation. Applied once to the merged stack, not separately to BSE/SE.

A stripe pattern that's constant along one image axis and varies along the
other concentrates its energy into a thin band through the origin of the 2D
FFT, oriented along the axis the stripes DON'T vary over (e.g. vertical
curtains, constant along columns, put all their energy at zero row-
frequency, spread across every column-frequency). Damping a narrow Gaussian
band centered on that line removes the stripe pattern while leaving
everything else - including a small disk around DC, left untouched so real
large-scale brightness/contrast survives.

Self-contained (doesn't depend on any sibling script) - portable to drop into
any FIB-SEM working directory.

Usage
-----
    python remove_curtaining.py
    python remove_curtaining.py --curtain-sigma 20
    python remove_curtaining.py --curtain-axis horizontal
    python remove_curtaining.py --mask-path fft_mask.npy
    python remove_curtaining.py --in-dir "..." --out-dir "..."

Output
------
    <out-dir>/  curtaining-corrected merged tiffs, same filenames as input
"""

import argparse
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import gaussian_filter

SLICE_RE = re.compile(r"(\d{3,6})\.tiff?$", re.IGNORECASE)

# Defaults to merge_bse_se.py's default output location (<BSE dir>/../Merged).
DEFAULT_IN_DIR = '/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_5050_cycled/A970_5to5_cycled/try3/Dataset 1/Merged'

# Hand-picked in select_fft_filter.py. Used automatically whenever this file
# is present and matches the image shape; otherwise falls back to the
# automatic Gaussian notch below.
DEFAULT_MASK_PATH = str(Path(__file__).resolve().parent / "fft_mask.npy")


def index_files(folder: Path) -> dict[int, Path]:
    """Map slice index (parsed from filename) -> file path."""
    files = {}
    for f in sorted(folder.glob("*.tif*")):
        m = SLICE_RE.search(f.name)
        if m:
            files[int(m.group(1))] = f
    return files


def img_as_float01(img: np.ndarray) -> np.ndarray:
    """Scale an integer image to float in [0, 1] based on its dtype's max value."""
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(img, 0.0, 1.0).astype(np.float64)
    info = np.iinfo(img.dtype)
    return img.astype(np.float64) / info.max


def to_output_dtype(img_float01: np.ndarray, bit_depth: int) -> np.ndarray:
    img_float01 = np.clip(img_float01, 0.0, 1.0)
    if bit_depth == 8:
        return (img_float01 * 255).astype(np.uint8)
    return (img_float01 * 65535).astype(np.uint16)


def apply_fft_weight(img: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """Multiply an image's (fftshifted) spectrum by a per-frequency weight in
    [0, 1] (0 = fully removed, 1 = fully kept) and invert back to image space."""
    img_f = img_as_float01(img)
    spectrum = np.fft.fftshift(np.fft.fft2(img_f)) * weight
    result = np.real(np.fft.ifft2(np.fft.ifftshift(spectrum)))
    return np.clip(result, 0.0, 1.0).astype(np.float32)


def remove_curtaining(img: np.ndarray, sigma: float, axis: str, dc_radius: float) -> np.ndarray:
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]

    perp = (yy - cy) if axis == "vertical" else (xx - cx)
    weight = 1.0 - np.exp(-(perp.astype(np.float64) ** 2) / (2 * sigma ** 2))
    dc_disk = (yy - cy) ** 2 + (xx - cx) ** 2 <= dc_radius ** 2
    weight[dc_disk] = 1.0

    return apply_fft_weight(img, weight)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--in-dir", default=DEFAULT_IN_DIR, help="Merged BSE/SE stack (output of merge_bse_se.py)")
    parser.add_argument("--out-dir", default=None, help="Default: <in-dir>/../Merged_clean")
    parser.add_argument("--bit-depth", type=int, choices=[8, 16], default=16)

    parser.add_argument("--curtain-axis", choices=["vertical", "horizontal"], default="vertical",
                         help="Axis the stripes are constant along (vertical = tall thin streaks).")
    parser.add_argument("--curtain-sigma", type=float, default=12.0,
                         help="Width (in FFT pixels) of the suppressed frequency band. "
                              "Calibrate visually: too small leaves stripes, too large blurs real detail.")
    parser.add_argument("--curtain-dc-radius", type=float, default=8.0,
                         help="Radius around the true DC term (overall brightness) that's always kept.")

    default_mask = DEFAULT_MASK_PATH if Path(DEFAULT_MASK_PATH).exists() else None
    parser.add_argument("--mask-path", default=default_mask,
                         help="Boolean .npy mask from select_fft_filter.py. Used by default when "
                              "fft_mask.npy exists next to this script; pass --mask-path '' to fall back "
                              "to the automatic Gaussian notch (--curtain-*) instead.")
    parser.add_argument("--mask-feather", type=float, default=2.0,
                         help="Gaussian blur (in FFT pixels) applied to the loaded mask's edges to reduce ringing.")
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir) if args.out_dir else in_dir.parent / "Merged_clean"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = index_files(in_dir)
    if not files:
        raise SystemExit(f"No slices found in {in_dir}")
    print(f"Found {len(files)} slices.")

    custom_weight = None
    if args.mask_path:
        sample_shape = tifffile.imread(files[sorted(files)[0]]).shape[:2]
        delete_mask = np.load(args.mask_path)
        if delete_mask.shape == sample_shape:
            custom_weight = (~delete_mask).astype(np.float64)
            if args.mask_feather > 0:
                custom_weight = gaussian_filter(custom_weight, sigma=args.mask_feather)
            print(f"Using hand-picked FFT mask from {args.mask_path}")
        else:
            print(f"NOTE: {Path(args.mask_path).name} shape {delete_mask.shape} doesn't match this "
                  f"stack's image shape {sample_shape} (it was picked on a different dataset) - "
                  "falling back to the automatic Gaussian notch. Run select_fft_filter.py on this "
                  "stack if you want a hand-picked mask for it instead.")

    for i, idx in enumerate(sorted(files)):
        raw = tifffile.imread(files[idx])
        if custom_weight is not None:
            cleaned = apply_fft_weight(raw, custom_weight)
        else:
            cleaned = remove_curtaining(raw, args.curtain_sigma, args.curtain_axis, args.curtain_dc_radius)
        tifffile.imwrite(out_dir / files[idx].name, to_output_dtype(cleaned, args.bit_depth))

        if (i + 1) % 100 == 0 or i == len(files) - 1:
            print(f"  processed {i + 1}/{len(files)} slices...")

    print(f"Done. Cleaned merged stack: {out_dir}")


if __name__ == "__main__":
    main()
