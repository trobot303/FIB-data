"""Inflate the cracks of a segmented stack by a given distance (nm) into the LCO, to test how sensitive LCO
connectivity / tortuosity are to the crack width.

    python scripts/inflate_cracks.py <segmented_folder> --nm 20 40 60 [--tag 3d] [--px-um 0.02] [--z-um 0.05]

Reads <folder>/labels_<tag>_stack.tif (1 pore, 2 LCO, 3 LLZTO, 4 crack).  For each distance d, every LCO voxel
within d of a crack voxel becomes crack (ellipsoidal neighbourhood in physical units: d / px in x and y, d / z-step
in z, so with 20 nm pixels and 50 nm slices, 20 nm grows the cracks 1 px in-plane and not in z, 50 nm also one
slice in z).  Pore and LLZTO are not changed.  Writes, next to the input:
  labels_<tag>_infl<d>nm_stack.tif and <phase>_<tag>_infl<d>nm_stack.tif (phase = pore, LCO, LLZTO, crack)
so that lco_crack_connectivity.py / tortuosity_y.py / connectivity_y.py read it with --tag <tag>_infl<d>nm.
Inflation also closes gaps narrower than 2 d between crack pieces: if connectivity drops sharply at a small d,
the cracks are nearly closed surfaces with small gaps (which may be missed crack, or real ligaments).
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

PHASES = {1: "pore", 2: "LCO", 3: "LLZTO", 4: "crack"}


def footprint(nm, px_nm, z_nm):
    ry, rz = nm / px_nm, nm / z_nm
    iy, iz = int(np.floor(ry)), int(np.floor(rz))
    z, y, x = np.mgrid[-iz:iz + 1, -iy:iy + 1, -iy:iy + 1]
    return (z / max(rz, 1e-9)) ** 2 + (y / ry) ** 2 + (x / ry) ** 2 <= 1.0 + 1e-9


def inflate(lab, fp, chunk=64):
    """Dilate the crack class (4) with footprint fp into LCO (2), in z chunks to bound memory."""
    out = lab.copy()
    hz = fp.shape[0] // 2
    Z = lab.shape[0]
    for z0 in range(0, Z, chunk):
        z1 = min(Z, z0 + chunk)
        a0, a1 = max(0, z0 - hz), min(Z, z1 + hz)
        grown = ndi.binary_dilation(lab[a0:a1] == 4, structure=fp)[z0 - a0: z1 - a0]
        blk = out[z0:z1]
        blk[grown & (blk == 2)] = 4
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--nm", type=float, nargs="+", required=True, help="inflation distance(s) in nm")
    ap.add_argument("--tag", default="3d", help="read labels_<tag>_stack.tif (default 3d = grown cracks)")
    ap.add_argument("--px-um", type=float, default=0.02, help="pixel size in x and y (um); LCO70 cycled: 0.02")
    ap.add_argument("--z-um", type=float, default=0.05, help="slice spacing (um); LCO70 cycled: 0.05")
    a = ap.parse_args()
    d = Path(a.folder)
    src = d / (f"labels_{a.tag}_stack.tif" if a.tag else "labels_stack.tif")
    if not src.exists():
        raise SystemExit(f"{src} not found")
    lab = tifffile.imread(str(src))
    n_lco, n_cr = int((lab == 2).sum()), int((lab == 4).sum())
    print(f"{src.name}: {lab.shape} (z, y, x); LCO {n_lco / lab.size:.4f}, crack {n_cr / lab.size:.4f} of the volume; "
          f"pixel {a.px_um * 1000:g} nm, slice {a.z_um * 1000:g} nm")
    big = lab.nbytes > 3.5e9
    for nm in a.nm:
        fp = footprint(nm, a.px_um * 1000, a.z_um * 1000)
        out = inflate(lab, fp)
        tag = f"{a.tag}_infl{nm:g}nm" if a.tag else f"infl{nm:g}nm"
        c = int((out == 4).sum())
        print(f"  +{nm:g} nm (footprint {fp.shape[2]} px in x/y, {fp.shape[0]} slice(s) in z): crack {n_cr} -> {c} voxels "
              f"(x{c / max(n_cr, 1):.2f}), crack {c / out.size:.4f} of the volume, LCO {(out == 2).sum() / out.size:.4f}")
        tifffile.imwrite(str(d / f"labels_{tag}_stack.tif"), out, photometric="minisblack", bigtiff=big, compression="zlib")
        for k, name in PHASES.items():
            tifffile.imwrite(str(d / f"{name}_{tag}_stack.tif"), ((out == k) * 255).astype(np.uint8),
                             photometric="minisblack", bigtiff=big, compression="zlib")
        print(f"    wrote labels_{tag}_stack.tif (+ per-phase stacks); use --tag {tag}")


if __name__ == "__main__":
    main()
