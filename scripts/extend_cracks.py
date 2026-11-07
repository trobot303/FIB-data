"""Measure the intact LCO "ligament" at every crack tip, and extend the cracks across it (upper-bound scenario
for unsegmented crack sections).

    python scripts/extend_cracks.py <segmented_folder> [--tag 3d] [--max-nm 200 500 1000] [--px-um 0.02]
                                    [--raw <grey slice folder>] [--max-search-nm 2000]

Reads <folder>/labels_<tag>_stack.tif (1 pore, 2 LCO, 3 LLZTO, 4 crack).  In every slice, each crack trace is
skeletonised; at each end point the local crack direction (principal axis of the skeleton within 8 px) is
followed outward through LCO until the path meets pore, LLZTO or another crack ("closing" ligament), the image
edge, or the search limit ("open").  The path length is the ligament: how much intact LCO separates the crack tip
from the next boundary along the crack's own direction.

For each --max-nm L, every closing ligament up to L is turned into crack (a 1 px, 8-connected line, which blocks
the 4-connected in-plane / 6-connected 3D LCO paths) and written as
  labels_<tag>_ext<L>nm_stack.tif and <phase>_<tag>_ext<L>nm_stack.tif
for lco_crack_connectivity.py / tortuosity_y.py (--tag <tag>_ext<L>nm).  This assumes each crack really continues to
the next boundary, i.e. that the ligament is unsegmented crack: an upper bound, to be checked on the figure.

Also writes crack_ligaments_<tag>.csv (one row per tip) and crack_ligaments_<tag>.png (length histogram and, with
--raw, the slice with most tips: SEM, cracks in red, closing ligaments <= 1 um in cyan).
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi
from skimage.draw import line
from skimage.morphology import skeletonize

PHASES = {1: "pore", 2: "LCO", 3: "LLZTO", 4: "crack"}
END = {1: "pore", 3: "LLZTO", 4: "crack"}


def tips(lab2, max_px):
    """Return a list of (y, x, y_end, x_end, length_px, end) for the crack tips of one slice."""
    crack = lab2 == 4
    if not crack.any():
        return []
    sk = skeletonize(crack)
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant") - 1
    cc, _ = ndi.label(crack, structure=np.ones((3, 3)))
    H, W = lab2.shape
    ys, xs = np.nonzero(sk & (nb == 1))
    sy, sx = np.nonzero(sk)
    out = []
    for y, x in zip(ys, xs):
        comp = cc[y, x]
        near = (np.abs(sy - y) <= 8) & (np.abs(sx - x) <= 8)
        py, px = sy[near], sx[near]
        keep = cc[py, px] == comp
        py, px = py[keep].astype(float), px[keep].astype(float)
        if py.size < 4:
            continue
        P = np.stack([py - py.mean(), px - px.mean()], 1)
        v = np.linalg.svd(P, full_matrices=False)[2][0]
        if (y - py.mean()) * v[0] + (x - px.mean()) * v[1] < 0:
            v = -v
        end, L, ye, xe = "open", max_px, y, x
        for t in range(1, max_px + 1):
            qy, qx = int(round(y + t * v[0])), int(round(x + t * v[1]))
            if not (0 <= qy < H and 0 <= qx < W):
                end, L = "edge", t - 1
                break
            val = lab2[qy, qx]
            if val == 4 and cc[qy, qx] == comp:
                continue                              # still inside the own crack (bent end)
            if val != 2:
                end, L = END.get(int(val), "other"), t - 1
                break
            ye, xe = qy, qx
        out.append((y, x, ye, xe, L, end))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--tag", default="3d")
    ap.add_argument("--max-nm", type=float, nargs="*", default=[200, 500, 1000], help="extend closing ligaments up to these lengths")
    ap.add_argument("--max-search-nm", type=float, default=2000)
    ap.add_argument("--px-um", type=float, default=0.02)
    ap.add_argument("--raw", default=None, help="grey slice folder (same order as the labels) for the figure")
    a = ap.parse_args()
    d = Path(a.folder)
    src = d / (f"labels_{a.tag}_stack.tif" if a.tag else "labels_stack.tif")
    lab = tifffile.imread(str(src))
    px_nm = a.px_um * 1000
    max_px = int(round(a.max_search_nm / px_nm))
    rows = []
    per_slice = []
    for z in range(lab.shape[0]):
        t = tips(lab[z], max_px)
        per_slice.append(t)
        rows += [(z, *r) for r in t]
        if (z + 1) % 50 == 0 or z + 1 == lab.shape[0]:
            print(f"  {z + 1}/{lab.shape[0]} slices, {len(rows)} crack tips", flush=True)
    sfx = "" if a.tag in ("3d", "") else f"_{a.tag}"
    with open(d / f"crack_ligaments{sfx or '_3d'}.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["z", "tip_y", "tip_x", "end_y", "end_x", "ligament_px", "ligament_nm", "ends_at"])
        for z, y, x, ye, xe, L, e in rows:
            w.writerow([z, y, x, ye, xe, L, f"{L * px_nm:.0f}", e])
    Ls = np.array([r[5] for r in rows], float) * px_nm
    ends = np.array([r[6] for r in rows])
    closing = np.isin(ends, ("pore", "LLZTO", "crack"))
    print(f"{len(rows)} crack tips in {lab.shape[0]} slices; ends at: " +
          ", ".join(f"{k} {np.mean(ends == k) * 100:.0f} %" for k in ("pore", "LLZTO", "crack", "edge", "open")))
    if closing.any():
        q = np.percentile(Ls[closing], [25, 50, 75, 90])
        print(f"closing ligament length (nm): 25 % {q[0]:.0f}, median {q[1]:.0f}, 75 % {q[2]:.0f}, 90 % {q[3]:.0f}")
        for L in (200, 500, 1000):
            print(f"  tips closing within {L} nm: {np.mean(closing & (Ls <= L)) * 100:.0f} % of all tips")

    big = lab.nbytes > 3.5e9
    for Lmax in a.max_nm:
        lim = int(round(Lmax / px_nm))
        out = lab.copy()
        n = 0
        for z, t in enumerate(per_slice):
            for y, x, ye, xe, L, e in t:
                if e in ("pore", "LLZTO", "crack") and L <= lim and L > 0:
                    rr, cc_ = line(y, x, ye, xe)
                    sl = out[z]
                    m = sl[rr, cc_] == 2
                    sl[rr[m], cc_[m]] = 4
                    n += 1
        tag = f"{a.tag}_ext{Lmax:g}nm"
        c0, c1 = int((lab == 4).sum()), int((out == 4).sum())
        print(f"  ext {Lmax:g} nm: {n} ligaments closed, crack voxels {c0} -> {c1} (+{100 * (c1 - c0) / max(c0, 1):.0f} %); use --tag {tag}")
        tifffile.imwrite(str(d / f"labels_{tag}_stack.tif"), out, photometric="minisblack", bigtiff=big, compression="zlib")
        for k, name in PHASES.items():
            tifffile.imwrite(str(d / f"{name}_{tag}_stack.tif"), ((out == k) * 255).astype(np.uint8),
                             photometric="minisblack", bigtiff=big, compression="zlib")
    figure(d / f"crack_ligaments{sfx or '_3d'}.png", Ls, ends, lab, per_slice, a, px_nm)


def figure(path, Ls, ends, lab, per_slice, a, px_nm):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    closing = np.isin(ends, ("pore", "LLZTO", "crack"))
    ncol = 2 if a.raw else 1
    fig, ax = plt.subplots(1, ncol, figsize=(7 * ncol, 6.5), gridspec_kw={"width_ratios": [1, 1.3][:ncol]})
    ax = np.atleast_1d(ax)
    bins = np.arange(0, a.max_search_nm + 100, 100)
    ax[0].hist([Ls[closing & (ends == "pore")], Ls[closing & (ends == "LLZTO")], Ls[closing & (ends == "crack")]],
               bins=bins, stacked=True, color=["#2a78d6", "#eda100", "#eb6834"], label=["ends at pore", "ends at LLZTO", "ends at another crack"])
    ax[0].set_xlabel("intact LCO ahead of the crack tip, along the crack (nm)")
    ax[0].set_ylabel("crack tips")
    no = int((~closing).sum())
    ax[0].set_title(f"{closing.sum()} closing ligaments ({no} tips run > {a.max_search_nm:g} nm or to the image edge)", fontsize=9, loc="left")
    ax[0].legend(frameon=False); ax[0].spines[["top", "right"]].set_visible(False)
    if a.raw:
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from fib_rf_seg.io import list_slices
        files = list_slices(a.raw)
        z = int(np.argmax([len(t) for t in per_slice]))
        g = tifffile.imread(str(files[z])).astype(np.float32)
        lo, hi = np.percentile(g, [0.5, 99.5]); g = np.clip((g - lo) / (hi - lo), 0, 1)
        rgb = np.repeat(g[..., None], 3, -1)
        rgb[lab[z] == 4] = (0.89, 0.29, 0.28)
        lim = int(round(1000 / px_nm))
        for y, x, ye, xe, L, e in per_slice[z]:
            if e in ("pore", "LLZTO", "crack") and 0 < L <= lim:
                rr, cc = line(y, x, ye, xe)
                rgb[rr, cc] = (0.10, 0.80, 0.90)
        ax[1].imshow(rgb); ax[1].set_axis_off()
        ax[1].set_title(f"slice {files[z].name}: cracks red, ligaments <= 1 um cyan (zoom in)", fontsize=9, loc="left")
    fig.tight_layout(); fig.savefig(path, dpi=150); plt.close(fig)
    print("wrote", path)


if __name__ == "__main__":
    main()
