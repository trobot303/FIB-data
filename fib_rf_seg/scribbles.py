"""Training labels: automatic high-confidence seeds and hand-drawn scribbles.

Scribble files are JSON::

    {"brush": 3,
     "strokes": [{"phase": 1, "points": [[y, x], [y, x], ...]},
                 {"phase": 3, "points": [[y, x]]}]}

Each stroke is a polyline in (row, col) pixel coordinates rasterised with a
round brush of radius ``brush`` (per-stroke ``"brush"`` overrides it).  Scribbles
drawn in ImageJ/napari/GIMP can instead be supplied as a label image.
Phase 5 is a sentinel meaning "not a crack": the pixel keeps the phase the
watershed assigns and is always included in training.

``annotation_to_scribbles`` converts an RGB copy of a slice painted in a
drawing program (pure red = crack, light green = not crack) into this format,
snapping red strokes onto the local dark-line centre.
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


def crack_candidates(im_norm: np.ndarray, width: int = 3, contrast: float = 0.35, smooth: float = 1.0,
                     min_len: int = 12) -> np.ndarray:
    """Thin dark lines: a black top-hat (closing with a disk of radius ``width`` minus the image)
    above ``contrast`` x (LCO - pore grey difference), excluding dark regions wider than
    2*width (those survive an opening and are pores) and components shorter than ``min_len`` px."""
    from skimage.morphology import closing, opening, remove_small_objects
    g = ndi.gaussian_filter(im_norm, smooth) if smooth > 0 else im_norm
    t1, t2 = threshold_multiotsu(g, classes=3)
    lo, mid = float(np.median(g[g < t1])), float(np.median(g[(g >= t1) & (g < t2)]))
    tophat = closing(g, sk_disk(width)) - g
    cand = tophat > contrast * (mid - lo)
    thick = opening(g < t1, sk_disk(width))
    cand &= ~ndi.binary_dilation(thick, iterations=2)
    cand &= g < t2                                  # never inside LLZTO
    return remove_small_objects(cand, max_size=min_len - 1, connectivity=2) if min_len > 1 else cand


def auto_seeds(im_norm: np.ndarray, erode: int = 3, margin: float = 0.35,
               smooth: float = 1.0, keep_small: int = 200, crack_width: int = 0,
               crack_contrast: float = 0.35) -> tuple[np.ndarray, tuple[float, float]]:
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
    if crack_width > 0:
        cr = crack_candidates(im_norm, width=crack_width, contrast=crack_contrast, smooth=smooth)
        seeds[cr] = 4                              # cracks override pore/LCO seeds
        seeds[ndi.binary_dilation(cr, iterations=2) & ~cr & (seeds != 3)] = 0   # unlabel the fuzzy rim
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


def annotation_to_scribbles(annotated_path, im_norm: np.ndarray, out_json, snap: int = 2):
    """Red strokes (crack, snapped + skeletonised) and light-green strokes (not crack) -> scribble JSON."""
    import json
    import tifffile
    from skimage.morphology import closing, skeletonize
    from skimage.io import imread
    p = str(annotated_path)
    a = tifffile.imread(p) if p.lower().endswith((".tif", ".tiff")) else imread(p)
    a = a[..., :3].astype(int)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    scale = 256 if a.max() > 255 else 1
    red = ((r - np.maximum(g, b)) > 60 * scale) & (r > 140 * scale)
    green = ((g - np.maximum(r, b)) > 40 * scale) & (g > 140 * scale)
    th = closing(im_norm, sk_disk(3)) - im_norm
    snapped = np.zeros_like(red)
    for y, x in zip(*np.nonzero(red)):
        y0, y1, x0, x1 = max(0, y - snap), min(th.shape[0], y + snap + 1), max(0, x - snap), min(th.shape[1], x + snap + 1)
        w = th[y0:y1, x0:x1]; iy, ix = np.unravel_index(np.argmax(w), w.shape); snapped[y0 + iy, x0 + ix] = True
    snapped = skeletonize(ndi.binary_closing(snapped, iterations=1))
    strokes = []
    for phase, mask in ((4, snapped), (5, green)):
        lab, n = ndi.label(mask, np.ones((3, 3)))
        for i in range(1, n + 1):
            yy, xx = np.nonzero(lab == i)
            strokes.append({"phase": phase, "brush": 0, "points": [[int(u), int(v)] for u, v in zip(yy, xx)]})
    with open(out_json, "w") as fh:
        json.dump({"brush": 0, "strokes": strokes}, fh)
    return int(snapped.sum()), int(green.sum())
