"""Through-thickness (y) connectivity of LCO and LLZTO from a segmented stack, with and without cracks.

    python scripts/connectivity_y.py <segmented_folder> [--collector top|bottom] [--out results.json]

<segmented_folder> is a predict output folder holding pore_stack.tif, LCO_stack.tif, LLZTO_stack.tif
and (optionally) crack_stack.tif (255 = phase).  Arrays are (z, y, x); y = image rows is the
through-thickness direction; the current collector is at the top image edge (y = 0) unless
--collector bottom.  LCO must reach the collector face, LLZTO the separator face (6-connectivity).

Reported per phase: volume fraction, fraction of the phase connected to its reference face,
fraction connected to both faces (percolating), number of clusters.  LCO is evaluated twice:
cracks treated as void (they interrupt LCO), and cracks filled back into LCO (as if uncracked),
so the loss of connectivity due to cracking is the difference.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

STRUCT = ndi.generate_binary_structure(3, 1)   # 6-connectivity


def connected_stats(mask: np.ndarray, ref_face: str) -> dict:
    """mask (z, y, x) bool; ref_face 'top' (y=0) or 'bottom' (y=-1)."""
    lab, n = ndi.label(mask, structure=STRUCT)
    top = np.unique(lab[:, 0, :]); bot = np.unique(lab[:, -1, :])
    top = top[top > 0]; bot = bot[bot > 0]
    ref = top if ref_face == "top" else bot
    both = np.intersect1d(top, bot)
    sizes = np.bincount(lab.ravel())
    vol = int(mask.sum())
    return {"volume_fraction": vol / mask.size,
            "connected_fraction": float(sizes[ref].sum() / vol) if vol else 0.0,
            "percolating_fraction": float(sizes[both].sum() / vol) if vol else 0.0,
            "percolates": bool(both.size > 0),
            "n_clusters": int(n),
            "largest_cluster_fraction": float(sizes[1:].max() / vol) if n else 0.0}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--collector", choices=("top", "bottom"), default="top", help="image edge of the current collector (y direction)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--z", type=int, nargs=2, default=None, metavar=("Z0", "Z1"), help="slice range to analyse (default: all)")
    ap.add_argument("--tag", default="", help="use <phase>_<tag>_stack.tif (e.g. 3d for the grown cracks of grow_cracks_3d.py)")
    a = ap.parse_args()
    d = Path(a.folder)
    sfx = f"_{a.tag}_stack.tif" if a.tag else "_stack.tif"
    def load(name):
        p = d / f"{name}{sfx}"
        if not p.exists():
            return None
        arr = tifffile.imread(str(p))
        if a.z:
            arr = arr[a.z[0]:a.z[1]]
        return arr > 0
    pore, lco, llz, crack = (load(n) for n in ("pore", "LCO", "LLZTO", "crack"))
    print(f"stack {lco.shape} (z, y, x); collector at {a.collector}; crack mask {'present' if crack is not None else 'absent'}")
    sep = "bottom" if a.collector == "top" else "top"
    res = {"folder": str(d), "shape_zyx": list(lco.shape), "collector_face": a.collector,
           "crack_volume_fraction": float(crack.mean()) if crack is not None else 0.0}
    res["LCO_cracks_as_void"] = connected_stats(lco, a.collector)
    if crack is not None:
        res["LCO_cracks_filled"] = connected_stats(lco | crack, a.collector)
    res["LLZTO"] = connected_stats(llz, sep)
    for k, v in res.items():
        if isinstance(v, dict):
            print(f"{k:22s} vf={v['volume_fraction']:.4f}  connected={v['connected_fraction']:.4f}  percolating={v['percolating_fraction']:.4f}  percolates={v['percolates']}  clusters={v['n_clusters']}")
    out = Path(a.out) if a.out else d / "connectivity_y.json"
    json.dump(res, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
