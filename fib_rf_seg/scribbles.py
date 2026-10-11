"""Training labels: automatic high-confidence seeds and hand-drawn scribbles.

Scribble files are JSON::

    {"brush": 3,
     "strokes": [{"phase": 1, "points": [[y, x], [y, x], ...]},
                 {"phase": 3, "points": [[y, x]]}]}

Each stroke is a polyline in (row, col) pixel coordinates rasterised with a
round brush of radius ``brush`` (per-stroke ``"brush"`` overrides it).  Scribbles
drawn in ImageJ/napari/GIMP can instead be supplied as a label image.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import ndimage as ndi
from skimage.draw import line, disk
from skimage.filters import threshold_multiotsu
from skimage.filters import sobel
from skimage.morphology import erosion, disk as sk_disk
from skimage.segmentation import watershed, find_boundaries


def rasterise(strokes: dict, shape: tuple[int, int]) -> np.ndarray:
    lab = np.zeros(shape, dtype=np.uint8)
    default_brush = int(strokes.get("brush", 3))
    for s in strokes["strokes"]:
        phase = int(s["phase"])
        r = int(s.get("brush", default_brush))
        pts = [tuple(map(int, p)) for p in s["points"]]
        mask = np.zeros(shape, dtype=bool)
        if len(pts) == 1:
            pts = pts * 2
        for (y0, x0), (y1, x1) in zip(pts[:-1], pts[1:]):
            rr, cc = line(y0, x0, y1, x1)
            for y, x in zip(rr, cc):
                dr, dc = disk((y, x), r + 0.5, shape=shape)
                mask[dr, dc] = True
        lab[mask] = phase
    return lab


def load_scribbles(path: str | Path, shape: tuple[int, int]) -> np.ndarray:
    with open(path) as f:
        return rasterise(json.load(f), shape)


def auto_seeds(im_norm: np.ndarray, erode: int = 3, margin: float = 0.35,
               smooth: float = 1.0, keep_small: int = 200) -> tuple[np.ndarray, tuple[float, float]]:
    """High-confidence seed labels from multi-Otsu thresholds.

    A pixel is a seed only if its smoothed grey value is at least ``margin`` of
    the class width away from both thresholds *and* it survives erosion with a
    disk of radius ``erode`` (so seeds stay inside objects, never on edges).
    Small objects would lose their seed entirely to the erosion and then be
    flooded with the surrounding phase by the watershed, so connected
    components of the un-eroded core smaller than ``keep_small`` px (and at
    least 4 px) are kept whole.  Everything else stays 0 (unlabelled) and is
    left to the classifier.
    """
    g = ndi.gaussian_filter(im_norm, smooth) if smooth > 0 else im_norm
    t1, t2 = threshold_multiotsu(g, classes=3)
    w1, w2, w3 = t1 - 0.0, t2 - t1, 1.0 - t2
    lo_core = g < t1 - margin * w1
    mid_core = (g > t1 + margin * w2) & (g < t2 - margin * w2)
    hi_core = g > t2 + margin * w3
    seeds = np.zeros(im_norm.shape, dtype=np.uint8)
    se = sk_disk(erode) if erode > 0 else None
    for k, m in ((1, lo_core), (2, mid_core), (3, hi_core)):
        if se is not None:
            eroded = erosion(m, se)
            if keep_small > 0:
                cc, n = ndi.label(m)
                sizes = np.bincount(cc.ravel())
                small = (sizes >= 4) & (sizes < keep_small)
                small[0] = False
                eroded = eroded | small[cc]
            m = eroded
        seeds[m] = k
    return seeds, (float(t1), float(t2))


def densify(im_norm: np.ndarray, seeds: np.ndarray, smooth: float = 1.0,
            drop_boundary: int = 1) -> np.ndarray:
    """Propagate sparse seeds to every pixel with a marker watershed on the gradient.

    Watershed lines sit on grey-level edges, so the dense labels include the
    boundary pixels the eroded seeds leave out.  Pixels within ``drop_boundary``
    of a watershed line are set back to 0 so that uncertain pixels do not train
    the classifier.
    """
    g = ndi.gaussian_filter(im_norm, smooth) if smooth > 0 else im_norm
    dense = watershed(sobel(g), markers=seeds.astype(np.int32)).astype(np.uint8)
    if drop_boundary > 0:
        b = find_boundaries(dense, mode="thick")
        if drop_boundary > 1:
            b = ndi.binary_dilation(b, iterations=drop_boundary - 1)
        dense[b] = 0
    return dense


def merge_labels(*labels: np.ndarray) -> np.ndarray:
    """Later label images override earlier ones wherever they are non-zero."""
    out = np.zeros_like(labels[0])
    for lab in labels:
        out[lab > 0] = lab[lab > 0]
    return out
