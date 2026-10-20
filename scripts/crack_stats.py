"""Crack statistics of one or more segmented cycled samples, side by side.

    python scripts/crack_stats.py NAME=<segmented folder> [NAME=<folder> ...] [--labels labels_3d] [--px-um 0.02]
                                  [--z-um 0.02 ...] [--out crack_stats.csv]

Reads <folder>/<labels>/*_labels.tif (1 pore, 2 LCO, 3 LLZTO, 4 crack; the final cracks of the cycled scripts are in
labels_3d, for the LCO70 aligned blocks give each block folder with --labels labels_v3_3d).  Per sample:
  phase fractions (pore, LCO, LLZTO, crack) of the whole volume,
  host phase of the cracks: each crack pixel is assigned to the nearest non-crack phase in its slice,
  cracked fraction of LCO and of LLZTO = crack pixels hosted by the phase / (phase + those crack pixels),
  crack length density = total crack centre-line length per section area (um / um^2), mean over slices,
  mean crack width (nm) = crack area / centre-line length,
  number of 3D crack objects (26-connected) per 1000 um^3 (only meaningful for contiguous stacks; needs --z-um).
"""
import argparse
import csv
import re
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.morphology import skeletonize


def natural_key(p: Path):
    m = re.findall(r"\d+", p.stem)
    return (int(m[-1]) if m else -1, p.name)


def stats(folder, labels, px, zum):
    files = sorted(Path(folder, labels).glob("*_labels.tif"), key=natural_key)
    if not files:
        raise SystemExit(f"no *_labels.tif in {Path(folder, labels)}")
    n = np.zeros(5); host = np.zeros(4); skel_len = []; area_px = 0; skel_tot = 0; crack3d = []
    for f in files:
        lab = tifffile.imread(str(f))
        n += np.bincount(lab.ravel(), minlength=5)[:5]
        m = lab == 4
        crack3d.append(m)
        if m.any():
            _, (ir, ic) = ndi.distance_transform_edt(m, return_indices=True)
            h = lab[ir[m], ic[m]]
            host += np.bincount(h, minlength=4)[:4]
            sk = skeletonize(m).sum()
        else:
            sk = 0
        skel_tot += sk
        area_px += m.sum()
        skel_len.append(sk * px / (lab.size * px * px))
    tot = n[1:].sum()
    r = {"slices": len(files)}
    for k, name in ((1, "pore"), (2, "LCO"), (3, "LLZTO"), (4, "crack")):
        r[f"{name}_vf_%"] = 100 * n[k] / tot
    r["crack_in_LCO_%"] = 100 * host[2] / max(host.sum(), 1)
    r["crack_in_LLZTO_%"] = 100 * host[3] / max(host.sum(), 1)
    r["crack_at_pore_%"] = 100 * host[1] / max(host.sum(), 1)
    r["LCO_cracked_%"] = 100 * host[2] / max(n[2] + host[2], 1)
    r["LLZTO_cracked_%"] = 100 * host[3] / max(n[3] + host[3], 1)
    r["length_density_um/um2"] = float(np.mean(skel_len))
    r["mean_width_nm"] = 1000 * px * area_px / max(skel_tot, 1)
    if zum:
        c = np.stack(crack3d)
        _, nobj = ndi.label(c, structure=ndi.generate_binary_structure(3, 3))
        vol = c.size * px * px * zum
        r["objects_per_1000um3"] = 1000 * nobj / vol
    return r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("samples", nargs="+", help="NAME=folder")
    ap.add_argument("--labels", default="labels_3d")
    ap.add_argument("--px-um", type=float, default=0.02)
    ap.add_argument("--z-um", type=float, nargs="*", default=None, help="slice spacing per sample (um), enables 3D object counts")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    rows = []
    for i, s in enumerate(a.samples):
        name, folder = s.split("=", 1) if "=" in s else (Path(s).name, s)
        z = a.z_um[i] if a.z_um and i < len(a.z_um) else None
        r = {"sample": name, **stats(folder, a.labels, a.px_um, z)}
        rows.append(r)
    keys = list(rows[0].keys()) + [k for r in rows for k in r if k not in rows[0]]
    keys = list(dict.fromkeys(keys))
    w = max(len(k) for k in keys)
    for k in keys:
        print(f"{k:<{w}}  " + "  ".join(f"{(r.get(k) if isinstance(r.get(k), str) else (f'{r[k]:.3f}' if k in r else '-')):>14}" for r in rows))
    if a.out:
        with open(a.out, "w", newline="") as fh:
            cw = csv.DictWriter(fh, fieldnames=keys); cw.writeheader(); cw.writerows(rows)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
