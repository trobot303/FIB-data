"""Optional label clean-up after classification."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label


def clean_labels(lab: np.ndarray, min_size: int = 25, majority_radius: int = 0) -> np.ndarray:
    """Remove islands smaller than *min_size* pixels per phase; optional majority filter.

    Removed islands take the phase of the nearest remaining pixel.
    """
    out = lab.copy()
    keep = np.zeros_like(lab, dtype=bool)
    for cls in np.unique(lab):
        if cls == 0:
            continue
        cc = label(lab == cls, connectivity=1)
        sizes = np.bincount(cc.ravel())
        keep |= (cc > 0) & (sizes[cc] >= min_size)
    if not keep.all():
        _, (iy, ix) = ndi.distance_transform_edt(~keep, return_indices=True)
        out = out[iy, ix]
    if majority_radius > 0:
        from skimage.filters.rank import majority
        from skimage.morphology import disk
        out = majority(out.astype(np.uint8), disk(majority_radius))
    return out.astype(np.uint8)


def phase_fractions(lab: np.ndarray) -> dict[int, float]:
    n = lab.size
    return {int(c): float((lab == c).sum()) / n for c in (1, 2, 3)}
