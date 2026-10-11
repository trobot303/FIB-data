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
            max_per_class: int | None = 20000, rng: np.random.Generator | None = None) -> None:
        rng = rng or np.random.default_rng(0)
        F = compute_features(im_norm, sigmas)
        for cls in np.unique(labels):
            if cls == 0:
                continue
            idx = np.flatnonzero(labels.ravel() == cls)
            if max_per_class and idx.size > max_per_class:
                idx = rng.choice(idx, max_per_class, replace=False)
            self.X.append(F.reshape(-1, F.shape[-1])[idx])
            self.y.append(np.full(idx.size, cls, dtype=np.uint8))

    def arrays(self):
        return np.concatenate(self.X), np.concatenate(self.y)


def train(ts: TrainingSet, n_estimators: int = 200, max_depth: int | None = 20,
          class_weight: str | None = "balanced", n_jobs: int = -1, random_state: int = 0):
    X, y = ts.arrays()
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=2,
        max_features="sqrt", class_weight=class_weight, n_jobs=n_jobs,
        random_state=random_state, oob_score=True,
    )
    rf.fit(X, y)
    return rf


def predict(rf, im_norm: np.ndarray, sigmas=DEFAULT_SIGMAS, chunk_rows: int = 128):
    """Return (labels uint8 HxW, probabilities HxWxC) for one normalised slice."""
    F = compute_features(im_norm, sigmas)
    H, W, n = F.shape
    classes = rf.classes_
    proba = np.empty((H, W, len(classes)), dtype=np.float32)
    for r in range(0, H, chunk_rows):
        blk = F[r:r + chunk_rows].reshape(-1, n)
        proba[r:r + chunk_rows] = rf.predict_proba(blk).reshape(-1, W, len(classes))
    labels = classes[np.argmax(proba, axis=-1)].astype(np.uint8)
    return labels, proba


def save_model(rf, path: str | Path, sigmas=DEFAULT_SIGMAS, norm: dict | None = None) -> None:
    joblib.dump({"rf": rf, "sigmas": tuple(sigmas), "feature_names": feature_names(sigmas),
                 "norm": norm or {}}, path, compress=3)


def load_model(path: str | Path):
    d = joblib.load(path)
    return d["rf"], d["sigmas"], d.get("norm", {})
