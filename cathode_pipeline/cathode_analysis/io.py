"""Input/output: slice discovery, stack loading, label volumes and tables.

Design notes
* ``discover_slices`` handles both a folder of 2D TIFFs and a single multipage
  TIFF. Files are natural-sorted by the trailing integer in the file name.
* ``load_stack`` reads slice by slice into a pre-allocated array so that at no
  point two copies of the volume exist. With ``scale=(lo, hi)`` each slice is
  converted to uint8 on the fly, so a uint16 stack never exists in memory as a
  whole (peak = 1 byte/voxel + a few slices).
* Label volumes are stored as ``.npy`` (memory-mappable with ``mmap_mode="r"``)
  or ``.tif`` (uint8, zlib compressed), chosen by ``runtime.label_format``.
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterator, Sequence

import numpy as np
import tifffile

LOG = logging.getLogger(__name__)

_TRAILING_INT = re.compile(r"(\d+)(?=\.[^.]+$)")


def natural_key(path: Path) -> tuple[int, str]:
    """Sort key: trailing integer before the extension, then the name."""
    m = _TRAILING_INT.search(path.name)
    return (int(m.group(1)) if m else -1, path.name)


# --------------------------------------------------------------------------- #
# discovery
# --------------------------------------------------------------------------- #
def discover_slices(input_dir: str | Path, pattern: str) -> tuple[list[Path], bool]:
    """Return (sorted files, is_multipage).

    ``is_multipage`` is True when exactly one file matches and it contains more
    than one page; that file is then treated as the whole stack.
    """
    input_dir = Path(input_dir)
    if not input_dir.is_dir():
        raise FileNotFoundError(f"input_dir does not exist: {input_dir}")
    files = sorted((p for p in input_dir.glob(pattern) if p.is_file()), key=natural_key)
    if not files:
        raise FileNotFoundError(f"no files match {pattern!r} in {input_dir}")
    if len(files) == 1:
        with tifffile.TiffFile(files[0]) as tf:
            if len(tf.pages) > 1:
                return files, True
    return files, False


class _Source:
    """Uniform access to slices from a file sequence or a multipage TIFF."""

    def __init__(self, files: list[Path], multipage: bool):
        self.files = files
        self.multipage = multipage
        self._tf: tifffile.TiffFile | None = None
        if multipage:
            self._tf = tifffile.TiffFile(files[0])
            self.n = len(self._tf.pages)
            page = self._tf.pages[0]
            self.shape2d, self.dtype = tuple(page.shape), np.dtype(page.dtype)
        else:
            self.n = len(files)
            with tifffile.TiffFile(files[0]) as tf:
                page = tf.pages[0]
                self.shape2d, self.dtype = tuple(page.shape), np.dtype(page.dtype)
        if len(self.shape2d) != 2:
            raise ValueError(f"expected 2D grayscale slices, got page shape {self.shape2d}")

    def read(self, i: int) -> np.ndarray:
        if self._tf is not None:
            return self._tf.pages[i].asarray()
        return tifffile.imread(self.files[i])

    def close(self) -> None:
        if self._tf is not None:
            self._tf.close()


def describe_stack(input_dir: str | Path, pattern: str) -> dict[str, Any]:
    """Cheap header-only description: n_slices, shape (z, y, x), dtype, mode."""
    files, multipage = discover_slices(input_dir, pattern)
    src = _Source(files, multipage)
    try:
        return {
            "mode": "multipage" if multipage else "sequence",
            "n_slices": src.n,
            "shape_zyx": (src.n, *src.shape2d),
            "dtype": str(src.dtype),
            "first_file": files[0].name,
            "last_file": files[-1].name,
            "gigavoxels": round(src.n * src.shape2d[0] * src.shape2d[1] / 1e9, 3),
        }
    finally:
        src.close()


# --------------------------------------------------------------------------- #
# range helpers
# --------------------------------------------------------------------------- #
def _slice_of(rng: Sequence[int | None] | None, n: int) -> slice:
    if rng is None:
        return slice(0, n)
    a, b = rng
    a = 0 if a is None else (a if a >= 0 else n + a)
    b = n if b is None else (b if b >= 0 else n + b)
    if not (0 <= a < b <= n):
        raise ValueError(f"range {list(rng)} invalid for extent {n}")
    return slice(a, b)


def _block_mean_2d(img: np.ndarray, k: int) -> np.ndarray:
    """Average k x k blocks (drops the incomplete border)."""
    if k == 1:
        return img
    h, w = (img.shape[0] // k) * k, (img.shape[1] // k) * k
    return img[:h, :w].reshape(h // k, k, w // k, k).mean(axis=(1, 3), dtype=np.float32)


# --------------------------------------------------------------------------- #
# loading
# --------------------------------------------------------------------------- #
def iter_slices(
    input_dir: str | Path,
    pattern: str,
    slice_range: Sequence[int | None] = (None, None),
    crop: dict[str, Sequence[int | None]] | None = None,
    step: int = 1,
) -> Iterator[np.ndarray]:
    """Yield cropped 2D slices (native dtype) - for histogram estimation."""
    files, multipage = discover_slices(input_dir, pattern)
    src = _Source(files, multipage)
    crop = crop or {}
    try:
        zs = _slice_of(slice_range, src.n)
        zs = _slice_of(crop.get("z"), zs.stop - zs.start) if crop.get("z") else slice(0, zs.stop - zs.start)
        z0 = _slice_of(slice_range, src.n).start
        ys = _slice_of(crop.get("y"), src.shape2d[0])
        xs = _slice_of(crop.get("x"), src.shape2d[1])
        for i in range(zs.start + z0, zs.stop + z0, step):
            yield src.read(i)[ys, xs]
    finally:
        src.close()


def load_stack(
    input_dir: str | Path,
    pattern: str,
    slice_range: Sequence[int | None] = (None, None),
    crop: dict[str, Sequence[int | None]] | None = None,
    downsample: int = 1,
    scale: tuple[float, float] | None = None,
    per_slice_scale: Callable[[np.ndarray], tuple[float, float]] | None = None,
    n_workers: int = 1,
) -> np.ndarray:
    """Load the (z, y, x) volume.

    * ``slice_range`` then ``crop['z']`` select slices; ``crop['y'|'x']`` crop
      each slice at read time.
    * ``downsample = k`` averages k x k x k blocks (k consecutive slices, then
      k x k pixels).
    * ``scale = (lo, hi)`` converts to uint8 on the fly: values are clipped to
      [lo, hi] and rescaled to 0..255. ``per_slice_scale(slice) -> (lo, hi)``
      does the same with a per-slice range. Without either, the native dtype
      is kept.
    """
    files, multipage = discover_slices(input_dir, pattern)
    src = _Source(files, multipage)
    crop = crop or {}
    k = int(downsample)
    try:
        zsel = _slice_of(slice_range, src.n)
        zc = _slice_of(crop.get("z"), zsel.stop - zsel.start)
        z_indices = list(range(zsel.start + zc.start, zsel.start + zc.stop))
        ys = _slice_of(crop.get("y"), src.shape2d[0])
        xs = _slice_of(crop.get("x"), src.shape2d[1])
        ny, nx = ys.stop - ys.start, xs.stop - xs.start
        nz = len(z_indices) // k
        ny, nx = ny // k, nx // k
        if nz == 0 or ny == 0 or nx == 0:
            raise ValueError("empty volume after slice_range/crop/downsample")

        to_uint8 = scale is not None or per_slice_scale is not None
        out_dtype = np.uint8 if to_uint8 else (np.float32 if k > 1 else src.dtype)
        out = np.empty((nz, ny, nx), dtype=out_dtype)
        LOG.info("loading %d slices -> %s %s (%.2f GB)", len(z_indices), out.shape, out.dtype, out.nbytes / 1e9)

        def convert(img: np.ndarray) -> np.ndarray:
            if not to_uint8:
                return img
            lo, hi = scale if scale is not None else per_slice_scale(img)
            f = img.astype(np.float32, copy=False)
            f -= lo
            f *= 255.0 / max(hi - lo, 1e-9)
            np.clip(f, 0, 255, out=f)
            return f.astype(np.uint8)

        def read_block(j: int) -> np.ndarray:
            if k == 1:
                return src.read(z_indices[j])[ys, xs]
            acc = None
            for i in z_indices[j * k:(j + 1) * k]:
                img = _block_mean_2d(src.read(i)[ys, xs], k)
                acc = img.astype(np.float32) if acc is None else acc + img
            return acc / k

        if multipage or n_workers <= 1:
            for j in range(nz):
                out[j] = convert(read_block(j))
        else:
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                for j, img in enumerate(pool.map(read_block, range(nz), chunksize=4)):
                    out[j] = convert(img)
        return out
    finally:
        src.close()


# --------------------------------------------------------------------------- #
# label volumes and small files
# --------------------------------------------------------------------------- #
def save_labels(labels: np.ndarray, path: str | Path, fmt: str = "npy") -> Path:
    """Save a label volume as .npy or compressed .tif (path extension is set from fmt)."""
    path = Path(path).with_suffix(f".{fmt}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if fmt == "npy":
        np.save(path, labels)
    elif fmt == "tif":
        tifffile.imwrite(path, labels, compression="zlib", photometric="minisblack")
    else:
        raise ValueError(f"unknown label format {fmt!r}")
    return path


def load_labels(path: str | Path, mmap: bool = True) -> np.ndarray:
    """Load a label volume; .npy is memory-mapped read-only when ``mmap``."""
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r" if mmap else None)
    return tifffile.imread(path)


def save_json(data: dict[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=_json_default), encoding="utf-8")
    return path


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)}")
