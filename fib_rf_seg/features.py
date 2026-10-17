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


def feature_names(sigmas=DEFAULT_SIGMAS, oriented: bool = False) -> list[str]:
    names = ["intensity"]
    for s in sigmas:
        names += [f"gauss_{s}", f"gradmag_{s}", f"log_{s}", f"hess_max_{s}", f"hess_min_{s}", f"std_{s}"]
        if oriented:
            names += [f"hess_rr_{s}", f"hess_cc_{s}", f"absgrad_r_{s}", f"absgrad_c_{s}"]
    for a, b in zip(sigmas[:-1], sigmas[1:]):
        names.append(f"dog_{a}_{b}")
    return names


def compute_features(im: np.ndarray, sigmas=DEFAULT_SIGMAS, oriented: bool = False) -> np.ndarray:
    """Return an (H, W, n_features) float32 array for a normalised 2-D image.

    ``oriented`` adds the row/column Hessian diagonal and the absolute row/column
    gradients per scale, so that the classifier can tell vertical structures
    (FIB curtaining) from inclined ones (cracks).
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
    return np.stack(feats, axis=-1).astype(np.float32)
