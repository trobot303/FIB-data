"""Command-line interface.

    python -m fib_rf_seg train   --image SLICE.tif [--scribbles S.json | --labels L.tif]
                                 [--auto-seeds] --model models/rf.joblib
    python -m fib_rf_seg predict --model models/rf.joblib --input SLICE_OR_DIR --out OUT_DIR
    python -m fib_rf_seg review  --model models/rf.joblib --image SLICE.tif --out review.png
                                 [--region y0 y1 x0 x1]

Training examples from several slices: pass --image/--scribbles pairs repeatedly
in the same order (zip).  Labels: 1 = pore, 2 = LCO, 3 = LLZTO, 0 = unlabelled.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import PHASES
from .io import list_slices, normalise, read_labels, read_slice, write_labels
from .model import TrainingSet, load_model, predict, save_model, train
from .postprocess import clean_labels, phase_fractions
from .scribbles import auto_seeds, densify, load_scribbles, merge_labels
from .viz import review_figure


def _norm_args(p):
    p.add_argument("--lo", type=float, default=None, help="fixed lower grey bound (default: 0.1th percentile per slice)")
    p.add_argument("--hi", type=float, default=None, help="fixed upper grey bound (default: 99.9th percentile per slice)")


def _load_norm(path, lo, hi):
    return normalise(read_slice(path), lo, hi)


def build_training_labels(im_norm, scribble_path=None, label_path=None, auto=False,
                          auto_erode=3, auto_margin=0.35, dense=False, drop_boundary=1):
    parts = []
    if auto:
        seeds, (t1, t2) = auto_seeds(im_norm, erode=auto_erode, margin=auto_margin)
        print(f"  auto seeds: thresholds {t1:.3f}/{t2:.3f}; "
              + ", ".join(f"{PHASES[k]}={int((seeds == k).sum())}" for k in PHASES))
        parts.append(seeds)
    if label_path:
        parts.append(read_labels(label_path))
    if scribble_path:
        parts.append(load_scribbles(scribble_path, im_norm.shape))
    if not parts:
        sys.exit("no training labels given (use --auto-seeds, --scribbles or --labels)")
    lab = merge_labels(*parts)
    if dense:
        lab = densify(im_norm, lab, drop_boundary=drop_boundary)
        print("  densified by watershed: " + ", ".join(f"{PHASES[k]}={int((lab == k).sum())}" for k in PHASES))
    return lab


def cmd_train(a):
    ts = TrainingSet()
    scr = a.scribbles or [None] * len(a.image)
    labs = a.labels or [None] * len(a.image)
    for img, s, l in zip(a.image, scr, labs):
        print(f"[train] {img}")
        im = _load_norm(img, a.lo, a.hi)
        lab = build_training_labels(im, s, l, a.auto_seeds, a.auto_erode, a.auto_margin, a.dense, a.drop_boundary)
        print("  labels used: " + ", ".join(f"{PHASES[k]}={int((lab == k).sum())}" for k in PHASES))
        ts.add(im, lab, max_per_class=a.max_per_class)
    t0 = time.time()
    rf = train(ts, n_estimators=a.trees, max_depth=a.max_depth)
    print(f"[train] {len(ts.arrays()[1])} samples, {a.trees} trees, {time.time() - t0:.1f}s, OOB accuracy {rf.oob_score_:.4f}")
    save_model(rf, a.model, norm={"lo": a.lo, "hi": a.hi})
    print(f"[train] saved {a.model}")


def cmd_predict(a):
    rf, sigmas, norm = load_model(a.model)
    lo = a.lo if a.lo is not None else norm.get("lo")
    hi = a.hi if a.hi is not None else norm.get("hi")
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = list_slices(a.input)
    summary = {}
    for f in files:
        im = _load_norm(f, lo, hi)
        lab, proba = predict(rf, im, sigmas)
        if a.min_size > 0:
            lab = clean_labels(lab, min_size=a.min_size, majority_radius=a.majority)
        write_labels(out / f"{f.stem}_labels.tif", lab)
        if a.save_proba:
            np.save(out / f"{f.stem}_proba.npy", proba)
        fr = phase_fractions(lab)
        summary[f.name] = fr
        print(f"[predict] {f.name}: " + ", ".join(f"{PHASES[k]}={fr[k]:.3f}" for k in PHASES))
    with open(out / "fractions.json", "w") as fh:
        json.dump(summary, fh, indent=2)


def cmd_review(a):
    rf, sigmas, norm = load_model(a.model)
    lo = a.lo if a.lo is not None else norm.get("lo")
    hi = a.hi if a.hi is not None else norm.get("hi")
    im = _load_norm(a.image, lo, hi)
    lab, proba = predict(rf, im, sigmas)
    if a.min_size > 0:
        lab = clean_labels(lab, min_size=a.min_size, majority_radius=a.majority)
    scr = load_scribbles(a.scribbles, im.shape) if a.scribbles else None
    region = tuple(a.region) if a.region else None
    fr = phase_fractions(lab)
    title = "  ".join(f"{PHASES[k]} {fr[k]*100:.1f}%" for k in PHASES)
    review_figure(im, lab, scr, proba, title=title, path=a.out, region=region)
    print(f"[review] {title}\n[review] wrote {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="fib_rf_seg", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train")
    t.add_argument("--image", action="append", required=True)
    t.add_argument("--scribbles", action="append", help="JSON scribble file (one per --image)")
    t.add_argument("--labels", action="append", help="label image (one per --image)")
    t.add_argument("--auto-seeds", action="store_true", help="add high-confidence multi-Otsu seeds")
    t.add_argument("--auto-erode", type=int, default=3)
    t.add_argument("--auto-margin", type=float, default=0.35)
    t.add_argument("--dense", action="store_true", help="propagate seeds/scribbles to all pixels with a watershed before sampling")
    t.add_argument("--drop-boundary", type=int, default=1, help="with --dense: unlabel pixels this close to watershed lines")
    t.add_argument("--max-per-class", type=int, default=20000)
    t.add_argument("--trees", type=int, default=200)
    t.add_argument("--max-depth", type=int, default=20)
    t.add_argument("--model", required=True)
    _norm_args(t)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True)
    pr.add_argument("--input", required=True, help="slice file or directory of slices")
    pr.add_argument("--out", required=True)
    pr.add_argument("--min-size", type=int, default=0, help="remove islands smaller than this (pixels)")
    pr.add_argument("--majority", type=int, default=0, help="majority filter radius (0 = off)")
    pr.add_argument("--save-proba", action="store_true")
    _norm_args(pr)
    pr.set_defaults(func=cmd_predict)

    r = sub.add_parser("review")
    r.add_argument("--model", required=True)
    r.add_argument("--image", required=True)
    r.add_argument("--scribbles")
    r.add_argument("--out", required=True)
    r.add_argument("--region", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    r.add_argument("--min-size", type=int, default=0)
    r.add_argument("--majority", type=int, default=0)
    _norm_args(r)
    r.set_defaults(func=cmd_review)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
