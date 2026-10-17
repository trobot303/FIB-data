"""Crop segmented volumes (the output folders of ``predict``) to a common size.

    python -m fib_rf_seg.cropvol report  DIR [DIR ...]
    python -m fib_rf_seg.cropvol crop    DIR [DIR ...] --size Z Y X [--z-anchor start|center|end | --z-start N]
                                         [--xy-anchor center|start] [--suffix _cropped] [--stack]
    python -m fib_rf_seg.cropvol crop    DIR [DIR ...] --match REF_DIR [...]

``report`` prints the number of slices and the slice shape of each folder.
``crop`` writes ``<DIR><suffix>`` with the same layout (labels/, pore/, LCO/,
LLZTO/, fractions.csv, optional *_stack.tif) containing only the chosen
slices, each cut to Y x X pixels.  ``--match`` takes the size from a reference
folder (e.g. the LCO60_2 output).  Slices are never resampled: all stacks are
assumed to share the same voxel size (50 nm isotropic here).
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import tifffile

from . import PHASES
from .io import list_slices

PHASE_DIRS = ["labels"] + [PHASES[k] for k in sorted(PHASES)]


def volume_info(d: Path):
    """Return (n_slices, (H, W), subdirs present) for a predict output folder."""
    subs = [s for s in PHASE_DIRS if (d / s).is_dir()]
    if not subs:
        sys.exit(f"{d}: no labels/pore/LCO/LLZTO subfolder found (is this a predict output folder?)")
    files = list_slices(d / subs[0])
    if not files:
        sys.exit(f"{d / subs[0]}: no TIFF slices")
    shape = tifffile.imread(str(files[0])).shape[:2]
    return len(files), shape, subs


def cmd_report(a):
    for d in a.dirs:
        d = Path(d)
        n, (h, w), subs = volume_info(d)
        print(f"{d}: {n} slices x {h} rows x {w} cols  ({w*a.px:.1f} x {h*a.px:.1f} x {n*a.px:.1f} um at {a.px} um/px); folders: {', '.join(subs)}")


def cmd_crop(a):
    if a.match:
        n_ref, (h_ref, w_ref), _ = volume_info(Path(a.match))
        Z, Y, X = n_ref, h_ref, w_ref
        print(f"[crop] target size from {a.match}: {Z} x {Y} x {X}")
    elif a.size:
        Z, Y, X = a.size
    else:
        sys.exit("give --size Z Y X or --match REF_DIR")
    for d in a.dirs:
        d = Path(d)
        n, (h, w), subs = volume_info(d)
        if n < Z or h < Y or w < X:
            sys.exit(f"{d} is {n} x {h} x {w}, smaller than the target {Z} x {Y} x {X}")
        if a.z_start is not None:
            z0 = a.z_start
        else:
            z0 = {"start": 0, "center": (n - Z) // 2, "end": n - Z}[a.z_anchor]
        if z0 + Z > n:
            sys.exit(f"{d}: z-start {z0} + {Z} slices exceeds {n}")
        y0 = (h - Y) // 2 if a.xy_anchor == "center" else 0
        x0 = (w - X) // 2 if a.xy_anchor == "center" else 0
        out = d.parent / (d.name + a.suffix)
        out.mkdir(parents=True, exist_ok=True)
        print(f"[crop] {d}: {n} x {h} x {w} -> {Z} x {Y} x {X}; slices {z0}..{z0+Z-1}, rows {y0}:{y0+Y}, cols {x0}:{x0+X} -> {out}")
        stacks = {s: [] for s in subs} if a.stack else None
        fractions = {}
        for s in subs:
            (out / s).mkdir(exist_ok=True)
            files = list_slices(d / s)[z0:z0 + Z]
            for f in files:
                im = tifffile.imread(str(f))[y0:y0 + Y, x0:x0 + X]
                tifffile.imwrite(str(out / s / f.name), im)
                if stacks is not None:
                    stacks[s].append(im)
                if s == "labels":
                    fractions[f.name] = {int(k): float((im == k).mean()) for k in PHASES}
        if fractions:
            with open(out / "fractions.csv", "w") as fh:
                fh.write("slice,pore,LCO,LLZTO\n")
                for name, fr in fractions.items():
                    fh.write(f"{name},{fr[1]:.5f},{fr[2]:.5f},{fr[3]:.5f}\n")
            with open(out / "fractions.json", "w") as fh:
                json.dump(fractions, fh, indent=2)
        if stacks is not None:
            for s, arrs in stacks.items():
                name = "labels_stack.tif" if s == "labels" else f"{s}_stack.tif"
                tifffile.imwrite(str(out / name), np.stack(arrs), compression="zlib", photometric="minisblack")
        with open(out / "crop_info.json", "w") as fh:
            json.dump({"source": str(d), "source_size_zyx": [n, h, w], "target_size_zyx": [Z, Y, X],
                       "z_start": z0, "y_start": y0, "x_start": x0, "voxel_um": a.px}, fh, indent=2)
        print(f"[crop] done: {out}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report"); r.add_argument("dirs", nargs="+"); r.add_argument("--px", type=float, default=0.05, help="voxel size in um"); r.set_defaults(func=cmd_report)
    c = sub.add_parser("crop"); c.add_argument("dirs", nargs="+")
    c.add_argument("--size", type=int, nargs=3, metavar=("Z", "Y", "X"))
    c.add_argument("--match", help="take Z Y X from this predict output folder")
    c.add_argument("--z-anchor", choices=("start", "center", "end"), default="start", help="which slices to keep (default: the first Z)")
    c.add_argument("--z-start", type=int, default=None, help="explicit first slice index (overrides --z-anchor)")
    c.add_argument("--xy-anchor", choices=("center", "start"), default="center")
    c.add_argument("--suffix", default="_cropped")
    c.add_argument("--stack", action="store_true", help="also write one multipage TIFF per phase")
    c.add_argument("--px", type=float, default=0.05)
    c.set_defaults(func=cmd_crop)
    a = ap.parse_args(argv)
    a.func(a)


if __name__ == "__main__":
    main()
