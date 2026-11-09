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


# Normalised grey medians of the LCO60 training data (bounds 2500-50000); used as
# anchors by normalise_phases so that other stacks land on the same scale.
PHASE_ANCHORS = {"pore": 0.18, "LLZTO": 0.875}


def normalise_phases(im: np.ndarray, smooth: float = 1.0, anchors: dict | None = None) -> np.ndarray:
    """Map grey values to [0, 1] so that the pore and LLZTO medians of *this slice*
    land on the same values as in the training data.

    The medians come from a multi-Otsu split of the smoothed slice, so the
    mapping adapts to brightness/contrast drift along z and between stacks.
    """
    from scipy import ndimage as ndi
    from skimage.filters import threshold_multiotsu
    anchors = anchors or PHASE_ANCHORS
    g = ndi.gaussian_filter(im, smooth) if smooth > 0 else im
    t1, t2 = threshold_multiotsu(g, classes=3)
    m_pore, m_llz = float(np.median(g[g < t1])), float(np.median(g[g >= t2]))
    a_p, a_l = anchors["pore"], anchors["LLZTO"]
    scale = (a_l - a_p) / max(m_llz - m_pore, 1e-6)
    out = a_p + (im - m_pore) * scale
    return np.clip(out, 0.0, 1.0).astype(np.float32)


# Pixel size (nm) of the LCO60 training slices (TIFF resolution 200000 px/cm, and
# the same grain sizes in px as the TESCAN stacks whose metadata say 50 nm).
# Used for models saved before the pixel size was stored in the model file.
DEFAULT_MODEL_PIXEL_NM = 50.0


def read_pixel_size(path: str | Path) -> float | None:
    """Return the in-plane pixel size of a slice in nm, or None if unknown.

    Looks at, in order: the TESCAN metadata tag (``PixelSizeX`` in metres),
    ImageJ's ``unit=`` + resolution tags, and the TIFF resolution tags with a
    cm/inch unit.
    """
    import re
    try:
        with tifffile.TiffFile(str(path)) as tf:
            page = tf.pages[0]
            tags = page.tags
            tescan = tags.get(50431)
            if tescan is not None:
                raw = tescan.value
                txt = raw.decode("latin1", "ignore") if isinstance(raw, (bytes, bytearray)) else str(raw)
                m = re.search(r"PixelSizeX=([0-9.eE+-]+)", txt)
                if m and float(m.group(1)) > 0:
                    return float(m.group(1)) * 1e9
            xres = tags.get("XResolution")
            if xres is None:
                return None
            num, den = xres.value if isinstance(xres.value, tuple) else (xres.value, 1)
            px_per_unit = num / den if den else 0.0
            if px_per_unit <= 0:
                return None
            desc = tags.get("ImageDescription")
            if desc is not None and "ImageJ" in str(desc.value):
                m = re.search(r"unit=(\S+)", str(desc.value))
                unit_nm = {"nm": 1.0, "nanometer": 1.0, "micron": 1e3, "um": 1e3, "\\u00B5m": 1e3, "µm": 1e3,
                           "mm": 1e6, "cm": 1e7}.get(m.group(1).strip() if m else "", None)
                if unit_nm is not None:
                    return unit_nm / px_per_unit
            unit = tags.get("ResolutionUnit")
            unit = int(unit.value) if unit is not None else 2
            if unit == 3:      # centimetre
                return 1e7 / px_per_unit
            if unit == 2 and px_per_unit > 1000:   # inch; small values are just screen dpi
                return 2.54e7 / px_per_unit
    except Exception:
        return None
    return None


def resample(im: np.ndarray, shape, order: int = 3) -> np.ndarray:
    """Resize a 2-D float image to *shape* (anti-aliased when shrinking)."""
    from skimage.transform import resize
    if tuple(shape) == im.shape:
        return im
    shrink = shape[0] < im.shape[0] or shape[1] < im.shape[1]
    return resize(im, shape, order=order, mode="edge", anti_aliasing=shrink,
                  preserve_range=True).astype(np.float32)


def resample_labels(lab: np.ndarray, shape, classes=(1, 2, 3)) -> np.ndarray:
    """Resize a label image to *shape* by an area-weighted vote between classes."""
    if tuple(shape) == lab.shape:
        return lab
    votes = np.stack([resample((lab == k).astype(np.float32), shape, order=1) for k in classes], axis=-1)
    return np.asarray(classes, dtype=np.uint8)[np.argmax(votes, axis=-1)]


def crop(im: np.ndarray, box) -> np.ndarray:
    """Crop to (y0, y1, x0, x1); None leaves the image unchanged."""
    if box is None:
        return im
    y0, y1, x0, x1 = box
    return im[y0:y1, x0:x1]


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
