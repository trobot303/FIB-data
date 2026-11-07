"""How much LCO would be cut off from the collector if every thin LCO neck were cracked?  An upper bound for cracks
that the segmentation misses at particle contacts / grain boundaries.

    python scripts/lco_neck_sensitivity.py <segmented_folder> [--tag 3d] [--nm 20 40 60 100 150]
                                           [--px-um 0.02] [--z-um 0.05] [--collector top|bottom]

Reads <folder>/labels_<tag>_stack.tif.  For each half-width d, LCO (cracks as void) is opened with an ellipsoid of
radius d (d / px in x and y, d / z-step in z): every LCO neck or protrusion thinner than about 2 d is removed, i.e.
treated as cracked.  Reported: the share of LCO removed, and the share of the remaining LCO that is no longer
connected to the collector face (6-connectivity, as in lco_crack_connectivity.py).  If even large d cut off little,
LCO connectivity cannot be lowered much by missed cracks at contacts.
Writes lco_neck_sensitivity_<tag>.csv next to the stack.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


def connected(mask, face):
    cc, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    ref = np.unique(cc[:, 0, :] if face == "top" else cc[:, -1, :]); ref = ref[ref > 0]
    lut = np.zeros(n + 1, bool); lut[ref] = True
    return lut[cc]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--tag", default="3d")
    ap.add_argument("--nm", type=float, nargs="+", default=[20, 40, 60, 100, 150])
    ap.add_argument("--px-um", type=float, default=0.02)
    ap.add_argument("--z-um", type=float, default=0.05)
    ap.add_argument("--collector", choices=("top", "bottom"), default="top")
    a = ap.parse_args()
    d = Path(a.folder)
    lab = tifffile.imread(str(d / (f"labels_{a.tag}_stack.tif" if a.tag else "labels_stack.tif")))
    lco = lab == 2
    del lab
    n = int(lco.sum())
    base = connected(lco, a.collector)
    rows = [(0, 0.0, 1 - (base & lco).sum() / n)]
    print(f"{lco.shape} (z, y, x); detected cracks only: {100 * rows[0][2]:.2f} % of LCO cut off")
    del base
    for nm in a.nm:
        r, rz = nm / (a.px_um * 1000), nm / (a.z_um * 1000)
        iy, iz = int(r), int(rz)
        z, y, x = np.mgrid[-iz:iz + 1, -iy:iy + 1, -iy:iy + 1]
        fp = (z / max(rz, 1e-9)) ** 2 + (y / r) ** 2 + (x / r) ** 2 <= 1 + 1e-9
        opened = ndi.binary_opening(lco, structure=fp)
        removed = 1 - opened.sum() / n
        cut = (opened & ~connected(opened, a.collector)).sum() / n
        rows.append((nm, removed, cut))
        print(f"  necks < ~{2 * nm:g} nm cracked: {100 * removed:5.2f} % of LCO removed, "
              f"{100 * cut:5.2f} % of LCO cut off from the collector", flush=True)
    sfx = "" if a.tag in ("3d", "") else f"_{a.tag}"
    with open(d / f"lco_neck_sensitivity{sfx or '_3d'}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["neck_halfwidth_nm", "LCO_removed_frac", "LCO_cut_off_frac"]); w.writerows(rows)


if __name__ == "__main__":
    main()
