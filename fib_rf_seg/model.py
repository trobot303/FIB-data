"""Training and prediction with a scikit-learn random forest."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from .features import DEFAULT_SIGMAS, compute_features, feature_names


@dataclass
class TrainingSet:
    """Accumulated (feature, label) samples, possibly from several slices."""
    X: list = field(default_factory=list)
    y: list = field(default_factory=list)

    def add(self, im_norm: np.ndarray, labels: np.ndarray, sigmas=DEFAULT_SIGMAS,
            max_per_class: int | None = 20000, rng: np.random.Generator | None = None,
            boundary_frac: float = 0.0, boundary_dist: int = 4, force: np.ndarray | None = None,
            oriented: bool = False, shape: bool = False) -> None:
        """Sample up to *max_per_class* labelled pixels per phase.

        With ``boundary_frac`` > 0, that share of each class's samples is drawn
        from pixels within ``boundary_dist`` px of another phase, so the forest
        sees enough edge pixels to place boundaries precisely.
        """
        rng = rng or np.random.default_rng(0)
        F = compute_features(im_norm, sigmas, oriented=oriented, shape=shape)
        Ff = F.reshape(-1, F.shape[-1])
        near = None
        if boundary_frac > 0:
            from scipy import ndimage as ndi
            from skimage.segmentation import find_boundaries
            b = find_boundaries(labels, mode="thick") & (labels > 0)
            near = ndi.binary_dilation(b, iterations=boundary_dist).ravel()
        forced = force.ravel() if force is not None else None
        for cls in np.unique(labels):
            if cls == 0:
                continue
            idx = np.flatnonzero(labels.ravel() == cls)
            keep_idx = idx[forced[idx]] if forced is not None else idx[:0]   # always-included pixels (e.g. "not crack" marks)
            idx = idx[~forced[idx]] if forced is not None else idx
            if max_per_class and idx.size > max_per_class:
                if near is not None:
                    nb = idx[near[idx]]; nin = idx[~near[idx]]
                    n_b = min(nb.size, int(round(max_per_class * boundary_frac)))
                    n_i = min(nin.size, max_per_class - n_b)
                    idx = np.concatenate([rng.choice(nb, n_b, replace=False) if n_b else nb[:0],
                                          rng.choice(nin, n_i, replace=False) if n_i else nin[:0]])
                else:
                    idx = rng.choice(idx, max_per_class, replace=False)
            idx = np.concatenate([idx, keep_idx])
            self.X.append(Ff[idx])
            self.y.append(np.full(idx.size, cls, dtype=np.uint8))

    def arrays(self):
        return np.concatenate(self.X), np.concatenate(self.y)


def train(ts: TrainingSet, n_estimators: int = 200, max_depth: int | None = 20,
          class_weight: str | None = "balanced", n_jobs: int = -1, random_state: int = 0,
          max_features="sqrt"):
    X, y = ts.arrays()
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=2,
        max_features=max_features, class_weight=class_weight, n_jobs=n_jobs,
        random_state=random_state, oob_score=True,
    )
    rf.fit(X, y)
    return rf


def predict(rf, im_norm: np.ndarray, sigmas=DEFAULT_SIGMAS, chunk_rows: int = 128, oriented: bool | None = None,
            shape: bool | None = None):
    """Return (labels uint8 HxW, probabilities HxWxC) for one normalised slice."""
    if oriented is None:
        oriented = bool(getattr(rf, "oriented_features_", False))
    if shape is None:
        shape = bool(getattr(rf, "shape_features_", False))
    F = compute_features(im_norm, sigmas, oriented=oriented, shape=shape)
    H, W, n = F.shape
    classes = rf.classes_
    proba = np.empty((H, W, len(classes)), dtype=np.float32)
    for r in range(0, H, chunk_rows):
        blk = F[r:r + chunk_rows].reshape(-1, n)
        proba[r:r + chunk_rows] = rf.predict_proba(blk).reshape(-1, W, len(classes))
    labels = classes[np.argmax(proba, axis=-1)].astype(np.uint8)
    return labels, proba


def save_model(rf, path: str | Path, sigmas=DEFAULT_SIGMAS, norm: dict | None = None, oriented: bool = False,
               shape: bool = False) -> None:
    rf.oriented_features_ = oriented
    rf.shape_features_ = shape
    joblib.dump({"rf": rf, "sigmas": tuple(sigmas), "feature_names": feature_names(sigmas, oriented, shape),
                 "norm": norm or {}, "oriented": oriented, "shape": shape}, path, compress=3)


def load_model(path: str | Path):
    d = joblib.load(path)
    rf = d["rf"]
    rf.oriented_features_ = bool(d.get("oriented", False))
    rf.shape_features_ = bool(d.get("shape", False))
    return rf, d["sigmas"], d.get("norm", {})
