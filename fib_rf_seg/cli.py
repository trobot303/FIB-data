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

import matplotlib

matplotlib.use("Agg")          # headless: review figures are written to files
import numpy as np

from . import PHASES
from .io import crop, destripe, list_slices, normalise, normalise_phases, read_labels, read_slice, write_labels
from .features import DEFAULT_SIGMAS
from .model import TrainingSet, load_model, predict, save_model, train
from .postprocess import clean_labels, drop_vertical_cracks, phase_fractions, place_solid_edges, pore_level_auto, snap_to_edges, solid_level_auto, trim_pores
from .scribbles import auto_seeds, curtaining_negatives, densify, load_scribbles, merge_labels
from .viz import review_figure


def _norm_args(p):
    p.add_argument("--lo", type=float, default=None, help="fixed lower grey bound (default: 0.1th percentile per slice)")
    p.add_argument("--hi", type=float, default=None, help="fixed upper grey bound (default: 99.9th percentile per slice)")
    p.add_argument("--smooth", type=float, default=None, help="Gaussian sigma (px) applied to the raw slice before everything else, for noisy (un-denoised) stacks")
    p.add_argument("--norm", choices=("bounds", "phase"), default=None,
                   help="'bounds': linear map with --lo/--hi (or per-slice percentiles); 'phase': per-slice map that puts the pore and LLZTO medians on the training scale (robust to drift along z and to other stacks)")
    p.add_argument("--crop", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"), default=None,
                   help="crop every slice to rows Y0:Y1 and columns X0:X1 before processing (pixel indices, end exclusive)")
    p.add_argument("--destripe", action="store_true", help="remove vertical curtaining stripes before normalisation")


def _load_norm(path, lo, hi, smooth=None, norm="bounds", box=None, strip=False):
    im = crop(read_slice(path), box)
    if smooth:
        from scipy import ndimage as ndi
        im = ndi.gaussian_filter(im, smooth)
    if strip:
        im = destripe(im)
    if norm == "phase":
        return normalise_phases(im)
    return normalise(im, lo, hi)


def _level(arg, im_norm, auto_fn):
    """Resolve a --*-level argument: None (off), 'auto' (multi-Otsu) or a number."""
    if arg is None:
        return None
    if str(arg).lower() == "auto":
        return auto_fn(im_norm)
    return float(arg)


def _pore_level(arg, im_norm):
    return _level(arg, im_norm, pore_level_auto)


def _solid_level(arg, im_norm):
    return _level(arg, im_norm, solid_level_auto)


def apply_edge_levels(lab, im_norm, pore_level=None, solid_level=None, band=3, verbose=False):
    """Re-place LCO/LLZTO edges and then pore edges at fixed grey levels (either may be None)."""
    s = _solid_level(solid_level, im_norm)
    if s is not None:
        lab = place_solid_edges(lab, im_norm, s, band=band)
        if verbose:
            print(f"  LCO/LLZTO edges set at grey level {s:.3f}")
    p = _pore_level(pore_level, im_norm)
    if p is not None:
        lab = trim_pores(lab, im_norm, p, band=band)
        if verbose:
            print(f"  pore edges set at grey level {p:.3f}")
    return lab


def build_training_labels(im_norm, scribble_path=None, label_path=None, auto=False,
                          auto_erode=3, auto_margin=0.35, dense=False, drop_boundary=1,
                          pore_level=None, band=3, solid_level=None, crack_width=0, crack_contrast=0.35,
                          curtain_neg=False):
    parts = []
    if auto:
        seeds, (t1, t2) = auto_seeds(im_norm, erode=auto_erode, margin=auto_margin,
                                     crack_width=crack_width, crack_contrast=crack_contrast)
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
    if curtain_neg:
        cn = curtaining_negatives(im_norm)
        cn &= lab != 4
        lab[cn] = 5
        print(f"  curtaining negatives: {int(cn.sum())} px")
    cracks = lab == 4
    notcrack = lab == 5                      # "not crack" marks: take the phase the watershed gives, always train on them
    force = np.zeros(lab.shape, dtype=bool)
    if dense:
        from scipy import ndimage as ndi
        lab[cracks | notcrack] = 0
        lab = densify(im_norm, lab, drop_boundary=0)
        if notcrack.any():
            dense0 = lab.copy()
        print("  densified by watershed: " + ", ".join(f"{PHASES[k]}={int((lab == k).sum())}" for k in PHASES))
        lab = apply_edge_levels(lab, im_norm, pore_level, solid_level, band, verbose=True)
        if drop_boundary > 0:
            from skimage.segmentation import find_boundaries
            b = find_boundaries(lab, mode="thick")
            if drop_boundary > 1:
                b = ndi.binary_dilation(b, iterations=drop_boundary - 1)
            lab[b] = 0
        if cracks.any():                      # crack scribbles: 1 px wide label, 3 px unlabelled rim
            lab[ndi.binary_dilation(cracks, iterations=3)] = 0
            lab[ndi.binary_dilation(cracks, iterations=1)] = 4
            print(f"  crack scribbles: {int((lab == 4).sum())} px")
        if notcrack.any():
            lab[notcrack] = dense0[notcrack]
            force = notcrack & (lab > 0) & (lab != 4)
            print(f"  not-crack marks: {int(force.sum())} px -> " + ", ".join(f"{PHASES[k]}={int((lab[force] == k).sum())}" for k in (1, 2, 3)))
    else:
        lab[notcrack] = 0
    return lab, force


def cmd_train(a):
    ts = TrainingSet()
    sigmas = tuple(a.sigmas) if a.sigmas else DEFAULT_SIGMAS
    n = len(a.image)
    scr = list(a.scribbles or []) + [None] * n          # pad: fewer --scribbles than --image is fine
    labs = list(a.labels or []) + [None] * n
    for img, s, l in zip(a.image, scr[:n], labs[:n]):
        print(f"[train] {img}")
        im = _load_norm(img, a.lo, a.hi, a.smooth, a.norm or "bounds", a.crop, a.destripe)
        lab, force = build_training_labels(im, s, l, a.auto_seeds, a.auto_erode, a.auto_margin, a.dense, a.drop_boundary,
                                           a.pore_level, a.band, a.solid_level, a.crack_width, a.crack_contrast,
                                           a.curtaining_negatives)
        print("  labels used: " + ", ".join(f"{PHASES[k]}={int((lab == k).sum())}" for k in PHASES))
        ts.add(im, lab, sigmas=sigmas, max_per_class=a.max_per_class, boundary_frac=a.boundary_frac, force=force,
               oriented=a.oriented)
    t0 = time.time()
    mf = a.max_features
    mf = None if mf == "all" else (float(mf) if mf.replace(".", "", 1).isdigit() else mf)
    cw = "balanced" if a.class_weight == "balanced" else (None if a.class_weight == "none" else {1: 1, 2: 1, 3: 1, 4: float(a.class_weight)})
    rf = train(ts, n_estimators=a.trees, max_depth=a.max_depth, max_features=mf, class_weight=cw)
    print(f"[train] {len(ts.arrays()[1])} samples, {a.trees} trees, {time.time() - t0:.1f}s, OOB accuracy {rf.oob_score_:.4f}")
    save_model(rf, a.model, sigmas=sigmas, norm={"lo": a.lo, "hi": a.hi, "smooth": a.smooth, "norm": a.norm or "bounds", "destripe": a.destripe},
               oriented=a.oriented)
    print(f"[train] saved {a.model}")


def _postprocess(lab, im, a):
    if a.snap >= 0:
        lab = snap_to_edges(lab, im, erode=a.snap)
    lab = apply_edge_levels(lab, im, a.pore_level, a.solid_level, a.band)
    if a.min_size > 0:
        lab = clean_labels(lab, min_size=a.min_size, majority_radius=a.majority)
    if a.drop_vertical is not None and a.drop_vertical > 0:
        lab = drop_vertical_cracks(lab, max_angle_deg=a.drop_vertical)
    return lab


def cmd_predict(a):
    import tifffile

    rf, sigmas, norm = load_model(a.model)
    lo = a.lo if a.lo is not None else norm.get("lo")
    hi = a.hi if a.hi is not None else norm.get("hi")
    smooth = a.smooth if a.smooth is not None else norm.get("smooth")
    mode = a.norm or norm.get("norm", "bounds")
    strip = a.destripe or bool(norm.get("destripe", False))
    box = tuple(a.crop) if a.crop else None
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    files = list_slices(a.input)
    if not files:
        sys.exit(f"no .tif/.tiff slices found in {a.input}")
    print(f"[predict] {len(files)} slices from {a.input}; norm={mode} lo={lo} hi={hi}; pre-smooth={smooth}; crop={box}")
    ldir = a.labels_subdir
    stack_name = a.stack_name or ("labels_stack.tif" if ldir == "labels" else f"{ldir}_stack.tif")
    frac_prefix = "fractions" if ldir == "labels" else f"{ldir}_fractions"
    (out / ldir).mkdir(exist_ok=True)
    phase_dirs = {}
    if a.per_phase:
        for k, name in PHASES.items():
            phase_dirs[k] = out / name
            phase_dirs[k].mkdir(exist_ok=True)
    hi_val = 255 if a.binary_value == 255 else 1
    stacks = {k: [] for k in list(PHASES) + [0]} if a.stack else None
    summary = {}
    t_start = time.time()
    for i, f in enumerate(files, 1):
        if a.overwrite or not (out / ldir / f"{f.stem}_labels.tif").exists():
            im = _load_norm(f, lo, hi, smooth, mode, box, strip)
            lab, proba = predict(rf, im, sigmas)
            lab = _postprocess(lab, im, a)
            write_labels(out / ldir / f"{f.stem}_labels.tif", lab)
            if a.save_proba:
                np.save(out / f"{f.stem}_proba.npy", proba)
        else:
            lab = read_labels(out / ldir / f"{f.stem}_labels.tif")
        for k, d in phase_dirs.items():
            tifffile.imwrite(str(d / f"{f.stem}_{PHASES[k]}.tif"), ((lab == k) * hi_val).astype(np.uint8))
        if stacks is not None:
            stacks[0].append(lab)
            for k in PHASES:
                stacks[k].append(((lab == k) * hi_val).astype(np.uint8))
        fr = phase_fractions(lab)
        summary[f.name] = fr
        el = time.time() - t_start
        eta = el / i * (len(files) - i)
        print(f"[predict] {i}/{len(files)} {f.name}: " + ", ".join(f"{PHASES[k]}={fr[k]:.3f}" for k in PHASES)
              + f"   ({el/i:.1f} s/slice, ETA {eta/60:.1f} min)", flush=True)
    with open(out / f"{frac_prefix}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(out / f"{frac_prefix}.csv", "w") as fh:
        fh.write("slice," + ",".join(PHASES[k] for k in PHASES) + "\n")
        for name, fr in summary.items():
            fh.write(name + "," + ",".join(f"{fr[k]:.5f}" for k in PHASES) + "\n")
    if stacks is not None:
        tifffile.imwrite(str(out / stack_name), np.stack(stacks[0]), compression="zlib", photometric="minisblack")
        for k in PHASES:
            tifffile.imwrite(str(out / f"{PHASES[k]}_stack.tif"), np.stack(stacks[k]), compression="zlib", photometric="minisblack")
        print(f"[predict] wrote {stack_name} and one multipage stack per phase to {out}")
    mean = {PHASES[k]: float(np.mean([fr[k] for fr in summary.values()])) for k in PHASES}
    print("[predict] mean fractions over stack: " + ", ".join(f"{n}={v:.3f}" for n, v in mean.items()))


def cmd_review(a):
    rf, sigmas, norm = load_model(a.model)
    lo = a.lo if a.lo is not None else norm.get("lo")
    hi = a.hi if a.hi is not None else norm.get("hi")
    smooth = a.smooth if a.smooth is not None else norm.get("smooth")
    mode = a.norm or norm.get("norm", "bounds")
    strip = a.destripe or bool(norm.get("destripe", False))
    im = _load_norm(a.image, lo, hi, smooth, mode, tuple(a.crop) if a.crop else None, strip)
    lab, proba = predict(rf, im, sigmas)
    lab = _postprocess(lab, im, a)
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
    t.add_argument("--crack-width", type=int, default=0, help="add automatic crack seeds (class 4): max half-width in px of a dark line counted as crack; 0 = off")
    t.add_argument("--crack-contrast", type=float, default=0.35, help="crack seed depth as a fraction of the LCO-pore grey difference")
    t.add_argument("--curtaining-negatives", action="store_true", help="label vertical dark ridges (FIB curtaining) automatically as 'not crack'")
    t.add_argument("--oriented", action="store_true", help="add orientation-sensitive features (row/column Hessian and gradients) so vertical curtaining and inclined cracks can be told apart")
    t.add_argument("--dense", action="store_true", help="propagate seeds/scribbles to all pixels with a watershed before sampling")
    t.add_argument("--drop-boundary", type=int, default=1, help="with --dense: unlabel pixels this close to watershed lines")
    t.add_argument("--max-per-class", type=int, default=20000)
    t.add_argument("--pore-level", default=None, help="grey level (0-1) of the pore edge in the training labels, or 'auto' (multi-Otsu); default: transition midpoint")
    t.add_argument("--solid-level", default=None, help="grey level (0-1) of the LCO/LLZTO edge in the training labels, or 'auto' (multi-Otsu)")
    t.add_argument("--band", type=int, default=3, help="width in px of the boundary band the edge levels are applied in")
    t.add_argument("--boundary-frac", type=float, default=0.5, help="share of samples per class drawn within 4 px of a phase boundary")
    t.add_argument("--sigmas", type=float, nargs="+", help=f"feature scales in px (default {DEFAULT_SIGMAS})")
    t.add_argument("--trees", type=int, default=200)
    t.add_argument("--max-depth", type=int, default=20)
    t.add_argument("--class-weight", default="balanced", help="'balanced' (default; not for crack models), 'none', or a number = weight of the crack class")
    t.add_argument("--max-features", default="sqrt", help="features tried per split: sqrt, log2, a fraction (0.5) or all")
    t.add_argument("--model", required=True)
    _norm_args(t)
    t.set_defaults(func=cmd_train)

    pr = sub.add_parser("predict")
    pr.add_argument("--model", required=True)
    pr.add_argument("--input", required=True, help="slice file or directory of slices")
    pr.add_argument("--out", required=True)
    pr.add_argument("--min-size", type=int, default=0, help="remove islands smaller than this (pixels)")
    pr.add_argument("--majority", type=int, default=0, help="majority filter radius (0 = off)")
    pr.add_argument("--pore-level", default=None, help="re-place pore edges at this grey level (0-1) or 'auto' (multi-Otsu)")
    pr.add_argument("--solid-level", default=None, help="re-place LCO/LLZTO edges at this grey level (0-1) or 'auto' (multi-Otsu)")
    pr.add_argument("--band", type=int, default=3)
    pr.add_argument("--drop-vertical", type=float, default=None, metavar="DEG", help="relabel crack components within DEG of vertical as LCO (FIB curtaining streaks)")
    pr.add_argument("--snap", type=int, default=-1, metavar="ERODE", help="snap boundaries to grey edges by watershed (erode radius for markers, e.g. 2; -1 = off)")
    pr.add_argument("--per-phase", action="store_true", help="also write one binary mask per phase and slice into <out>/pore, <out>/LCO, <out>/LLZTO")
    pr.add_argument("--stack", action="store_true", help="also write one multipage TIFF per phase (and labels_stack.tif) for the whole stack")
    pr.add_argument("--binary-value", type=int, default=255, choices=(1, 255), help="foreground value of the binary masks")
    pr.add_argument("--overwrite", action="store_true", help="recompute slices whose labels already exist in <out>/labels (default: resume)")
    pr.add_argument("--labels-subdir", default="labels", help="name of the per-slice label folder inside <out> (e.g. labels_v2)")
    pr.add_argument("--stack-name", default=None, help="file name of the label stack (default: <labels-subdir>_stack.tif)")
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
    r.add_argument("--pore-level", default=None, help="re-place pore edges at this grey level (0-1) or 'auto' (multi-Otsu)")
    r.add_argument("--solid-level", default=None, help="re-place LCO/LLZTO edges at this grey level (0-1) or 'auto' (multi-Otsu)")
    r.add_argument("--band", type=int, default=3)
    r.add_argument("--drop-vertical", type=float, default=None, metavar="DEG", help="relabel crack components within DEG of vertical as LCO (FIB curtaining streaks)")
    r.add_argument("--snap", type=int, default=-1, metavar="ERODE", help="snap boundaries to grey edges by watershed (erode radius for markers, e.g. 2; -1 = off)")
    _norm_args(r)
    r.set_defaults(func=cmd_review)

    a = p.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
