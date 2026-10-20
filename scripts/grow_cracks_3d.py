"""Grow the segmented cracks along their length and through the slices (3D hysteresis).

    python scripts/grow_cracks_3d.py --labels <out>/labels --proba <out>/crack_proba --out <out>/labels_3d
                                     [--low 0.2] [--connectivity 26] [--min-seed 1] [--z-blocks 0]

The random forest decides pixel by pixel, so a crack is often found only on its clearest parts
and may be missed in a neighbouring slice although it is still there.  This script takes the
final crack pixels of `predict` (label 4, after --min-size / shape filters) as seeds and adds
every voxel that has crack probability > --low and is 3D-connected to a seed (26-connectivity:
the 8 in-plane neighbours and the 9+9 in the slices below/above).  Pixels with crack probability
above --low that touch no seed in any slice stay as they are, so curtaining noise is not added.

Inputs: <labels>/<stem>_labels.tif (1 pore, 2 LCO, 3 LLZTO, 4 crack) and <proba>/<stem>_crackp.tif
(uint8, crack probability x 255, from `predict --crack-proba`).  The slices are ordered by the
number in their name.  Output: <out>/<stem>_labels.tif with the grown cracks, <out>/../labels_3d_stack.tif,
per-phase binary stacks (255 = phase) in <out>/.. and fractions_3d.csv.

Memory: the whole stack is held as uint8 (labels) + uint8 (probability) + an int32 label volume,
about 6 bytes per voxel (500 slices of 1001x1001 px = 3 GB).  --z-blocks N processes the stack in
N overlapping blocks (growth across a block edge is limited to the overlap of 10 slices).
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

PHASES = {1: "pore", 2: "LCO", 3: "LLZTO", 4: "crack"}


def natural_key(p: Path):
    m = re.findall(r"\d+", p.stem)
    return (int(m[-1]) if m else -1, p.name)


def grow(lab, pc, low, conn):
    """lab, pc: (Z, Y, X).  Returns the boolean mask of grown cracks."""
    seeds = lab == 4
    cand = (pc > low) | seeds
    st = ndi.generate_binary_structure(3, 3 if conn == 26 else (2 if conn == 18 else 1))
    cc, n = ndi.label(cand, structure=st)
    if n == 0:
        return seeds
    keep = np.zeros(n + 1, dtype=bool)
    keep[np.unique(cc[seeds])] = True
    keep[0] = False
    return keep[cc]


def drop_short(lab, orig, min_len, conn):
    """Remove 3D crack components whose longest 2D piece (major axis, per slice) is < min_len px.
    Removed voxels get their label from before the growth if that was not crack, else the nearest
    non-crack label in the slice."""
    from skimage.measure import regionprops
    st = ndi.generate_binary_structure(3, 3 if conn == 26 else (2 if conn == 18 else 1))
    cc, n = ndi.label(lab == 4, structure=st)
    if n == 0:
        return lab
    longest = np.zeros(n + 1)
    for z in range(lab.shape[0]):
        sl = cc[z]
        if not sl.any():
            continue
        pieces, _ = ndi.label(sl > 0, structure=np.ones((3, 3)))
        for r in regionprops(pieces, intensity_image=sl):
            cid = int(r.intensity_max)
            longest[cid] = max(longest[cid], r.axis_major_length)
    drop = (longest < min_len)[cc] & (cc > 0)
    out = lab.copy()
    for z in np.nonzero(drop.any(axis=(1, 2)))[0]:
        d = drop[z]
        prev = orig[z]
        out[z][d & (prev != 4)] = prev[d & (prev != 4)]
        rest = d & (prev == 4)
        if rest.any():
            src = (out[z] != 4) & ~d
            _, (ir, ic) = ndi.distance_transform_edt(~src, return_indices=True)
            out[z][rest] = out[z][ir[rest], ic[rest]]
    print(f"  {int((longest[1:] < min_len).sum())} of {n} 3D crack components shorter than {min_len:g} px dropped")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--proba", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--low", type=float, default=0.2, help="crack probability above which a voxel connected to a seed is added")
    ap.add_argument("--connectivity", type=int, choices=(6, 18, 26), default=26)
    ap.add_argument("--z-blocks", type=int, default=0, help="process the stack in N overlapping blocks (0 = whole stack)")
    ap.add_argument("--no-stacks", action="store_true", help="write only the per-slice labels")
    ap.add_argument("--min-length", type=float, default=0.0, metavar="PX",
                    help="after growth, drop 3D crack components whose longest in-plane piece is shorter than PX "
                         "(short dark spots between particles); pieces of a crack that is long in another slice are kept")
    ap.add_argument("--tag", default="3d", help="name tag of the outputs next to <out>: labels_<tag>_stack.tif, <phase>_<tag>_stack.tif, fractions_<tag>.csv")
    a = ap.parse_args()

    lab_dir, pr_dir, out = Path(a.labels), Path(a.proba), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted([p for p in lab_dir.glob("*_labels.tif")], key=natural_key)
    stems = [p.name[:-len("_labels.tif")] for p in files]
    missing = [s for s in stems if not (pr_dir / f"{s}_crackp.tif").exists()]
    if missing:
        raise SystemExit(f"{len(missing)} slices have no crack probability in {pr_dir} (e.g. {missing[0]}); run predict with --crack-proba")
    Z = len(files)
    shape = tifffile.imread(str(files[0])).shape
    print(f"{Z} slices of {shape[0]}x{shape[1]} px; low threshold {a.low}; {a.connectivity}-connectivity")
    lab = np.empty((Z,) + shape, dtype=np.uint8)
    pc = np.empty((Z,) + shape, dtype=np.uint8)
    for i, (f, s) in enumerate(zip(files, stems)):
        lab[i] = tifffile.imread(str(f))
        pc[i] = tifffile.imread(str(pr_dir / f"{s}_crackp.tif"))
    low8 = int(round(a.low * 255))
    before = int((lab == 4).sum())
    grown = np.zeros(lab.shape, dtype=bool)
    nb = max(1, a.z_blocks)
    edges = np.linspace(0, Z, nb + 1).astype(int)
    ov = 10 if nb > 1 else 0
    for b in range(nb):
        z0, z1 = max(0, edges[b] - ov), min(Z, edges[b + 1] + ov)
        g = grow(lab[z0:z1], pc[z0:z1], low8, a.connectivity)
        c0, c1 = edges[b] - z0, edges[b + 1] - z0
        grown[edges[b]:edges[b + 1]] = g[c0:c1]
        if nb > 1:
            print(f"  block {b + 1}/{nb}: slices {edges[b]}-{edges[b + 1] - 1}")
    del pc
    new = grown & (lab != 4)
    orig = lab.copy() if a.min_length > 0 else None
    lab[new] = 4
    after = int((lab == 4).sum())
    print(f"crack voxels: {before} -> {after} (+{100 * (after - before) / max(before, 1):.0f} %)")
    if a.min_length > 0:
        lab = drop_short(lab, orig, a.min_length, a.connectivity)
        kept = int((lab == 4).sum())
        print(f"length filter >= {a.min_length:g} px: crack voxels {after} -> {kept} (-{100 * (after - kept) / max(after, 1):.0f} %)")

    with open(out.parent / f"fractions_{a.tag}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["slice"] + list(PHASES.values()))
        for i, s in enumerate(stems):
            tifffile.imwrite(str(out / f"{s}_labels.tif"), lab[i])
            w.writerow([s] + [f"{(lab[i] == k).mean():.5f}" for k in PHASES])
        w.writerow(["mean"] + [f"{(lab == k).mean():.5f}" for k in PHASES])
    print("mean fractions: " + ", ".join(f"{PHASES[k]}={(lab == k).mean():.3f}" for k in PHASES))
    if not a.no_stacks:
        big = lab.nbytes > 3.5e9
        tifffile.imwrite(str(out.parent / f"labels_{a.tag}_stack.tif"), lab, photometric="minisblack", bigtiff=big)
        for k, name in PHASES.items():
            tifffile.imwrite(str(out.parent / f"{name}_{a.tag}_stack.tif"), ((lab == k) * 255).astype(np.uint8), photometric="minisblack", bigtiff=big)
        print(f"stacks written to {out.parent}")
    print(f"done: {out}")


if __name__ == "__main__":
    main()
