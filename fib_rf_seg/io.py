"""Image loading, normalisation and label I/O."""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import tifffile


def _natural_key(p: Path):
    m = re.findall(r"\d+", p.stem)
    return (int(m[-1]) if m else -1, p.name)


def list_slices(path: str | Path) -> list[Path]:
    """Return the TIFF slices under *path* (a file or a directory), sorted by trailing number."""
    path = Path(path)
    if path.is_file():
        return [path]
    files = [p for p in path.iterdir() if p.suffix.lower() in (".tif", ".tiff")]
    return sorted(files, key=_natural_key)


def read_slice(path: str | Path) -> np.ndarray:
    """Read one 2-D grey slice as float32 without any scaling."""
    im = tifffile.imread(str(path))
    if im.ndim == 3:  # RGB or multi-page; take first channel/page
        im = im[..., 0] if im.shape[-1] in (3, 4) else im[0]
    return im.astype(np.float32)


def normalise(im: np.ndarray, lo: float | None = None, hi: float | None = None,
              p_lo: float = 0.1, p_hi: float = 99.9) -> np.ndarray:
    """Map grey values to [0, 1] using fixed or percentile bounds (clipped).

    Normalising per slice makes the features robust against brightness drift along z.
    Pass explicit *lo*/*hi* to use the same bounds for a whole stack instead.
    """
    if lo is None or hi is None:
        lo_p, hi_p = np.percentile(im, [p_lo, p_hi])
        lo = lo_p if lo is None else lo
        hi = hi_p if hi is None else hi
    out = (im - lo) / max(hi - lo, 1e-6)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def read_labels(path: str | Path) -> np.ndarray:
    """Read a label image (0 = unlabelled, 1 = pore, 2 = LCO, 3 = LLZTO) as uint8."""
    path = Path(path)
    if path.suffix.lower() == ".npy":
        lab = np.load(path)
    else:
        lab = tifffile.imread(str(path)) if path.suffix.lower() in (".tif", ".tiff") else None
        if lab is None:
            from skimage.io import imread
            lab = imread(str(path))
    if lab.ndim == 3:
        lab = lab[..., 0]
    return lab.astype(np.uint8)


def write_labels(path: str | Path, lab: np.ndarray) -> None:
    tifffile.imwrite(str(path), lab.astype(np.uint8))
