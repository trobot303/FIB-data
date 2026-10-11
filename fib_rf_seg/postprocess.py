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


def snap_to_edges(lab: np.ndarray, im_norm: np.ndarray, erode: int = 2, smooth: float = 0.7) -> np.ndarray:
    """Move phase boundaries onto the grey-level edges.

    Each phase is eroded by *erode* px to form markers, then a watershed on the
    gradient magnitude of the (lightly smoothed) image re-grows them so that the
    boundaries sit on the gradient maxima.  Regions thinner than 2*erode+1 px
    lose their marker and are absorbed by their neighbours, so keep *erode* small.
    """
    from skimage.filters import sobel
    from skimage.morphology import erosion, disk
    from skimage.segmentation import watershed
    g = ndi.gaussian_filter(im_norm, smooth) if smooth > 0 else im_norm
    markers = np.zeros_like(lab, dtype=np.int32)
    se = disk(erode)
    for cls in (1, 2, 3):
        m = erosion(lab == cls, se) if erode > 0 else lab == cls
        markers[m] = cls
    out = watershed(sobel(g), markers=markers)
    return out.astype(np.uint8)


def otsu_levels(im_norm: np.ndarray, smooth: float = 1.0) -> tuple[float, float]:
    """Multi-Otsu (pore/solid, LCO/LLZTO) thresholds of the smoothed slice, normalised grey units."""
    from skimage.filters import threshold_multiotsu
    g = ndi.gaussian_filter(im_norm, smooth) if smooth > 0 else im_norm
    t1, t2 = threshold_multiotsu(g, classes=3)
    return float(t1), float(t2)


def pore_level_auto(im_norm: np.ndarray, smooth: float = 1.0) -> float:
    return otsu_levels(im_norm, smooth)[0]


def solid_level_auto(im_norm: np.ndarray, smooth: float = 1.0) -> float:
    return otsu_levels(im_norm, smooth)[1]


def place_solid_edges(lab: np.ndarray, im_norm: np.ndarray, level: float, band: int = 3,
                      smooth: float = 1.0) -> np.ndarray:
    """Place LCO/LLZTO edges at a fixed grey level (same idea as :func:`trim_pores`).

    Within ``band`` px of an LCO/LLZTO boundary a pixel is LLZTO iff its smoothed
    grey value is at least ``level``, else LCO.  Pore pixels are never changed.
    """
    g = ndi.gaussian_filter(im_norm, smooth) if smooth > 0 else im_norm
    lco, llz = lab == 2, lab == 3
    if not lco.any() or not llz.any():
        return lab.copy()
    d_lco = ndi.distance_transform_edt(~lco)
    d_llz = ndi.distance_transform_edt(~llz)
    zone = (lco & (d_llz <= band)) | (llz & (d_lco <= band))
    out = lab.copy()
    out[zone & (g >= level)] = 3
    out[zone & (g < level)] = 2
    return out


def trim_pores(lab: np.ndarray, im_norm: np.ndarray, level: float, band: int = 3,
               smooth: float = 1.0, grow: bool = True) -> np.ndarray:
    """Place pore edges at a fixed grey level instead of the transition midpoint.

    Within ``band`` px of every pore boundary, a pixel is pore iff its smoothed
    grey value is below ``level``.  Pore pixels above the level take the label
    of the nearest solid pixel (so no LCO halo appears between pore and LLZTO);
    with ``grow``, solid pixels below the level become pore.  Pore regions far
    from any boundary (e.g. dark-grey pore back walls) are left to the
    classifier.
    """
    g = ndi.gaussian_filter(im_norm, smooth) if smooth > 0 else im_norm
    pore = lab == 1
    out = lab.copy()
    if not pore.any() or pore.all():
        return out
    d_in, (iy, ix) = ndi.distance_transform_edt(pore, return_indices=True)
    nearest_solid = lab[iy, ix]
    trim = pore & (d_in <= band) & (g >= level)
    out[trim] = nearest_solid[trim]
    if grow:
        d_out = ndi.distance_transform_edt(~pore)
        out[~pore & (d_out <= band) & (g < level)] = 1
    return out


def phase_fractions(lab: np.ndarray) -> dict[int, float]:
    n = lab.size
    return {int(c): float((lab == c).sum()) / n for c in (1, 2, 3)}
