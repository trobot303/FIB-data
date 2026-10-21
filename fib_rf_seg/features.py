"""Per-pixel feature stack for random-forest classification.

The features are the usual "ilastik-style" set: Gaussian-smoothed intensity,
gradient magnitude, Laplacian of Gaussian, Hessian eigenvalues and local
standard deviation at several scales, plus a difference-of-Gaussians.  Texture
and context at larger scales let the classifier tell real pore (black with
pore back-wall structure shining through) from genuinely dark LCO.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

DEFAULT_SIGMAS = (0.5, 1.0, 2.0, 4.0, 8.0)


SHAPE_SIGMAS = (1.0, 2.0, 4.0)
# (top-hat disk radius, depth threshold in normalised grey units); 0 radius = global dark mask (below the pore level)
SHAPE_MASKS = ((3, 0.02), (3, 0.05), (8, 0.05), (8, 0.15), (0, 0.0))
RIDGE_MASKS = ((1.0, 0.7, 0.004), (2.0, 0.7, 0.002))   # (sigma, min anisotropy, min dark-line curvature)


def feature_names(sigmas=DEFAULT_SIGMAS, oriented: bool = False, shape=False) -> list[str]:
    names = ["intensity"]
    for s in sigmas:
        names += [f"gauss_{s}", f"gradmag_{s}", f"log_{s}", f"hess_max_{s}", f"hess_min_{s}", f"std_{s}"]
        if oriented:
            names += [f"hess_rr_{s}", f"hess_cc_{s}", f"absgrad_r_{s}", f"absgrad_c_{s}"]
    for a, b in zip(sigmas[:-1], sigmas[1:]):
        names.append(f"dog_{a}_{b}")
    if shape:
        names += [f"ridge_aniso_{s}" for s in SHAPE_SIGMAS]
    if shape and shape != "ridge":
        for r, t in SHAPE_MASKS:
            tag = f"dark_r{r}_t{t:g}" if r else "pore_mask"
            names += [f"{tag}_aspect", f"{tag}_width", f"{tag}_logarea"]
        for sg, _, cm in RIDGE_MASKS:
            names += [f"ridge_s{sg:g}_c{cm:g}_aspect", f"ridge_s{sg:g}_c{cm:g}_width", f"ridge_s{sg:g}_c{cm:g}_logarea"]
    return names


def _component_shape(mask):
    """Per-pixel aspect ratio, mean width (px) and log10(area) of the 8-connected component of ``mask``."""
    from skimage.measure import label, regionprops
    from skimage.morphology import skeletonize
    asp = np.zeros(mask.shape, np.float32); wid = np.zeros(mask.shape, np.float32); lar = np.zeros(mask.shape, np.float32)
    if mask.any():
        cc = label(mask, connectivity=2)
        sk_len = np.bincount(cc[skeletonize(mask)], minlength=cc.max() + 1)
        a_ = np.zeros(cc.max() + 1, np.float32); w_ = a_.copy(); l_ = a_.copy()
        for p in regionprops(cc):
            a_[p.label] = min(p.axis_major_length / max(p.axis_minor_length, 1.0), 30.0)
            w_[p.label] = p.area / max(sk_len[p.label], 1)
            l_[p.label] = np.log10(p.area)
        asp, wid, lar = a_[cc], w_[cc], l_[cc]
    return [asp, wid, lar]


def shape_features(im: np.ndarray, ridge_only: bool = False) -> list[np.ndarray]:
    """Object-shape features for telling cracks (thin, elongated dark lines) from pores (compact dark blobs).

    * ridge anisotropy per scale: sign(L) (|L| - |l|) / (|L| + |l|) with L the Hessian eigenvalue of
      larger magnitude and l the other; ~+1 on a dark line, ~0 on a dark blob (pore) or flat area,
      ~-1 on a bright line;
    * for each dark-object mask (black top-hat deeper than a threshold at two disk radii, and the
      global below-pore-level mask): the connected component each pixel belongs to gives its aspect
      ratio (major / minor axis, capped at 30), its mean width (area / centre-line length, px) and
      log10(area); 0 outside the masks.  A pore gets aspect ~1-2 and a large width, a crack a high
      aspect and a width of a few px, whatever their grey level;
    * the same three numbers for the components of the dark-ridge masks (anisotropy above a limit
      and a minimum dark-line curvature at sigma 1 and 2), which also catch very faint but continuous
      cracks that no grey-level threshold separates from the noise.
    """
    from skimage.filters import threshold_multiotsu
    from skimage.morphology import closing, disk
    feats = []
    eps = 1e-6
    aniso, bigs = {}, {}
    for s in sorted(set(SHAPE_SIGMAS) | {r[0] for r in RIDGE_MASKS}):
        ev = hessian_matrix_eigvals(hessian_matrix(im, sigma=s, order="rc", use_gaussian_derivatives=False))
        a0, a1 = np.abs(ev[0]), np.abs(ev[1])
        big = np.where(a0 >= a1, ev[0], ev[1])
        aniso[s] = (np.sign(big) * np.abs(a0 - a1) / (a0 + a1 + eps)).astype(np.float32)
        bigs[s] = big
    feats += [aniso[s] for s in SHAPE_SIGMAS]
    if ridge_only:
        return feats
    g = ndi.gaussian_filter(im, 1.0)
    t1 = threshold_multiotsu(g, classes=3)[0]
    tophat = {r: closing(g, disk(r)) - g for r, _ in SHAPE_MASKS if r}
    for r, t in SHAPE_MASKS:
        feats += _component_shape((tophat[r] > t) if r else (g < t1))
    for sg, amin, cmin in RIDGE_MASKS:
        feats += _component_shape((aniso[sg] > amin) & (bigs[sg] > cmin))
    return feats


def compute_features(im: np.ndarray, sigmas=DEFAULT_SIGMAS, oriented: bool = False, shape=False) -> np.ndarray:
    """Return an (H, W, n_features) float32 array for a normalised 2-D image.

    ``oriented`` adds the row/column Hessian diagonal and the absolute row/column
    gradients per scale, so that the classifier can tell vertical structures
    (FIB curtaining) from inclined ones (cracks).  ``shape`` adds the object-shape
    features of :func:`shape_features` (aspect ratio and width of the dark object a
    pixel belongs to), so that cracks and pores, both dark, are told apart by shape;
    ``shape="ridge"`` adds only the continuous ridge-anisotropy maps.
    """
    im = im.astype(np.float32)
    feats = [im]
    gauss = {}
    for s in sigmas:
        g = ndi.gaussian_filter(im, s)
        gauss[s] = g
        gy, gx = np.gradient(g)
        gradmag = np.sqrt(gx * gx + gy * gy)
        log = ndi.gaussian_laplace(im, s)
        H = hessian_matrix(im, sigma=s, order="rc", use_gaussian_derivatives=False)
        ev = hessian_matrix_eigvals(H)  # (2, H, W), sorted descending
        mean_sq = ndi.gaussian_filter(im * im, s)
        std = np.sqrt(np.clip(mean_sq - g * g, 0, None))
        feats += [g, gradmag, log, ev[0], ev[1], std]
        if oriented:
            feats += [H[0], H[2], np.abs(gy), np.abs(gx)]
    for a, b in zip(sigmas[:-1], sigmas[1:]):
        feats.append(gauss[a] - gauss[b])
    if shape:
        feats += shape_features(im, ridge_only=(shape == "ridge"))
    return np.stack(feats, axis=-1).astype(np.float32)
