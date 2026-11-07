"""How much LCO do the cracks cut off from the current collector?  y-connectivity with cracks vs cracks filled.

    python scripts/lco_crack_connectivity.py <segmented_folder> [--tag 3d] [--collector top|bottom]
                                             [--px-um 0.02] [--z Z0 Z1] [--out <folder>]

<segmented_folder> is a predict / grow_cracks_3d output folder with labels_<tag>_stack.tif
(1 pore, 2 LCO, 3 LLZTO, 4 crack; --tag 3d = grown cracks, --tag "" = labels_stack.tif).
Arrays are (z, y, x); y = image rows is the through-thickness direction, the current collector is at
the top image edge (y = 0) unless --collector bottom.  6-connectivity, as in scripts/connectivity_y.py.

Two cases:
  cracks as void   LCO alone is labelled; a crack interrupts the LCO network.
  cracks filled    LCO + crack voxels are labelled together, as if the particles were uncracked.
Every fraction is taken over the LCO voxels only (crack voxels are never counted as LCO), so both cases
share the same denominator and the difference is exactly the LCO the cracks disconnect:
  cut off by cracks = LCO connected to the collector when filled, but not with cracks as void.

Writes to <out> (default: the segmented folder):
  lco_crack_connectivity.json      summary of both cases and the difference
  lco_crack_connectivity_y.csv     connected fraction per depth row y (both cases)
  lco_crack_connectivity_z.csv     connected fraction per slice z (both cases)
  lco_crack_connectivity.png       figure: summary bars, depth profile, per-slice profile, map of one slice
  LCO_cut_by_cracks_stack.tif      255 = LCO cut off by cracks (open in Fiji next to the label stack)
For a tag other than 3d / "" (e.g. 3d_infl20nm from scripts/inflate_cracks.py) the tag is added to every name.
Memory: about 6 bytes per voxel (int32 cluster labels + masks), e.g. ~4 GB for 600 x 1001 x 1001.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi

STRUCT = ndi.generate_binary_structure(3, 1)   # 6-connectivity
PORE, LCO, LLZTO, CRACK = 1, 2, 3, 4


def face_connected(mask: np.ndarray, ref_face: str):
    """Label *mask*; return (connected to ref face, connected to both faces, number of clusters)."""
    lab, n = ndi.label(mask, structure=STRUCT)
    top = np.unique(lab[:, 0, :]); bot = np.unique(lab[:, -1, :])
    top = top[top > 0]; bot = bot[bot > 0]
    ref = top if ref_face == "top" else bot
    lut = np.zeros(n + 1, bool); lut[ref] = True
    conn = lut[lab]
    lut[:] = False; lut[np.intersect1d(top, bot)] = True
    perc = lut[lab]
    del lab
    return conn, perc, int(n)


def case_stats(lco, conn, perc, n):
    v = int(lco.sum())
    return {"connected_fraction": float((conn & lco).sum() / v) if v else 0.0,
            "percolating_fraction": float((perc & lco).sum() / v) if v else 0.0,
            "percolates": bool((perc & lco).any()),
            "n_clusters": n}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--tag", default="3d", help="read labels_<tag>_stack.tif (default 3d = grown cracks; '' = labels_stack.tif)")
    ap.add_argument("--collector", choices=("top", "bottom"), default="top", help="image edge of the current collector")
    ap.add_argument("--px-um", type=float, default=None, help="pixel size in y (um) for the depth axis, e.g. 0.02 for LCO70 cycled")
    ap.add_argument("--z", type=int, nargs=2, default=None, metavar=("Z0", "Z1"), help="slice range (0-based, end exclusive)")
    ap.add_argument("--slice", type=int, default=None, help="slice shown in the map (default: the one with most LCO cut off)")
    ap.add_argument("--out", default=None, help="output folder (default: the segmented folder)")
    a = ap.parse_args()

    d = Path(a.folder)
    stack = d / (f"labels_{a.tag}_stack.tif" if a.tag else "labels_stack.tif")
    if not stack.exists():
        raise SystemExit(f"{stack} not found: run the segmentation (and grow_cracks_3d for --tag 3d) first")
    lab = tifffile.imread(str(stack))
    if lab.ndim == 2:
        lab = lab[None]
    if a.z:
        lab = lab[a.z[0]:a.z[1]]
    out = Path(a.out) if a.out else d
    out.mkdir(parents=True, exist_ok=True)
    nz, ny, nx = lab.shape
    sfx = "" if a.tag in ("3d", "") else f"_{a.tag}"           # e.g. _3d_infl20nm: one set of outputs per tag
    pre = f"lco_crack_connectivity{sfx}"
    lco = lab == LCO
    crack = lab == CRACK
    print(f"{stack.name}: {nz} x {ny} x {nx} (z, y, x); collector at {a.collector}; "
          f"LCO {lco.mean():.4f}, crack {crack.mean():.4f} of the volume")

    conn_v, perc_v, n_v = face_connected(lco, a.collector)
    void = case_stats(lco, conn_v, perc_v, n_v); del perc_v
    conn_f, perc_f, n_f = face_connected(lco | crack, a.collector)
    filled = case_stats(lco, conn_f, perc_f, n_f); del perc_f
    conn_v &= lco; conn_f &= lco
    cut = conn_f & ~conn_v

    n_lco = int(lco.sum())
    res = {"stack": str(stack), "shape_zyx": [nz, ny, nx], "collector_face": a.collector,
           "LCO_volume_fraction": float(lco.mean()), "crack_volume_fraction": float(crack.mean()),
           "crack_fraction_of_LCO_plus_crack": float(crack.sum() / max(n_lco + int(crack.sum()), 1)),
           "LCO_cracks_as_void": void, "LCO_cracks_filled": filled,
           "difference": {
               "LCO_cut_off_by_cracks_fraction": float(cut.sum() / n_lco) if n_lco else 0.0,
               "connected_fraction_drop": filled["connected_fraction"] - void["connected_fraction"],
               "connected_fraction_relative_drop": (1 - void["connected_fraction"] / filled["connected_fraction"])
                                                   if filled["connected_fraction"] else 0.0,
               "percolating_fraction_drop": filled["percolating_fraction"] - void["percolating_fraction"],
               "extra_clusters_from_cracks": n_v - n_f}}

    # profiles: connected share of the LCO in each depth row and in each slice
    lco_y = lco.sum(axis=(0, 2)); lco_z = lco.sum(axis=(1, 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        py_v = conn_v.sum(axis=(0, 2)) / lco_y; py_f = conn_f.sum(axis=(0, 2)) / lco_y
        pz_v = conn_v.sum(axis=(1, 2)) / lco_z; pz_f = conn_f.sum(axis=(1, 2)) / lco_z
    with open(out / f"{pre}_y.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["y_px", "y_um", "LCO_voxels", "connected_cracks_filled", "connected_cracks_as_void", "cut_off_by_cracks"])
        for y in range(ny):
            w.writerow([y, f"{y * a.px_um:.4f}" if a.px_um else "", int(lco_y[y]),
                        f"{py_f[y]:.5f}", f"{py_v[y]:.5f}", f"{py_f[y] - py_v[y]:.5f}"])
    with open(out / f"{pre}_z.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["z", "LCO_voxels", "connected_cracks_filled", "connected_cracks_as_void", "cut_off_by_cracks"])
        for z in range(nz):
            w.writerow([z + (a.z[0] if a.z else 0), int(lco_z[z]), f"{pz_f[z]:.5f}", f"{pz_v[z]:.5f}", f"{pz_f[z] - pz_v[z]:.5f}"])
    tifffile.imwrite(str(out / f"LCO_cut_by_cracks{sfx}_stack.tif"), (cut * 255).astype(np.uint8),
                     photometric="minisblack", bigtiff=cut.nbytes > 2**31)

    zs = a.slice if a.slice is not None else int(np.argmax(cut.sum(axis=(1, 2))))
    res["map_slice"] = zs + (a.z[0] if a.z else 0)
    json.dump(res, open(out / f"{pre}.json", "w"), indent=2)

    dif = res["difference"]
    print(f"{'':22s} {'connected':>10s} {'percolating':>12s} {'clusters':>9s}")
    for k, v in (("cracks filled", filled), ("cracks as void", void)):
        print(f"{k:22s} {v['connected_fraction']:10.4f} {v['percolating_fraction']:12.4f} {v['n_clusters']:9d}")
    print(f"{'filled − void':22s} {dif['connected_fraction_drop']:10.4f} {dif['percolating_fraction_drop']:12.4f} {-dif['extra_clusters_from_cracks']:9d}")
    print(f"LCO cut off from the collector by cracks: {100 * dif['LCO_cut_off_by_cracks_fraction']:.2f} % of all LCO "
          f"({100 * dif['connected_fraction_relative_drop']:.1f} % of the connected LCO)")

    figure(out / f"{pre}.png", res, py_f, py_v, pz_f, pz_v, lab[zs], conn_v[zs], conn_f[zs], a, zs)
    print("wrote", out / f"{pre}.json", f"and the .png / .csv / LCO_cut_by_cracks{sfx}_stack.tif next to it")


def figure(path, res, py_f, py_v, pz_f, pz_v, lab2, cv2, cf2, a, zs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator

    INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    C_FILL, C_VOID = "#2a78d6", "#eb6834"          # categorical slots 1-2 (validated pair)
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                         "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
    fig, ax = plt.subplots(2, 2, figsize=(12, 9), gridspec_kw={"width_ratios": [1, 1.25]})
    f, v, d = res["LCO_cracks_filled"], res["LCO_cracks_as_void"], res["difference"]
    nz, ny, nx = res["shape_zyx"]
    z0 = a.z[0] if a.z else 0

    # (a) summary
    b = ax[0, 0]
    names = ["connected to\ncollector", "percolating\n(both faces)"]
    xs = np.arange(2); wdt = 0.36
    for off, vals, c, lbl in ((-wdt / 2 - 0.01, (f["connected_fraction"], f["percolating_fraction"]), C_FILL, "cracks filled"),
                              (wdt / 2 + 0.01, (v["connected_fraction"], v["percolating_fraction"]), C_VOID, "cracks as void")):
        bars = b.bar(xs + off, [100 * x for x in vals], wdt, color=c, label=lbl)
        for r, val in zip(bars, vals):
            b.text(r.get_x() + r.get_width() / 2, r.get_height() + 1, f"{100 * val:.1f} %", ha="center", va="bottom", color=INK, fontsize=8)
    b.set_xticks(xs, names); b.set_ylabel("share of LCO volume (%)"); b.set_ylim(0, 124)
    b.yaxis.grid(True, color=GRID, lw=0.8); b.set_axisbelow(True)
    b.legend(frameon=False, loc="upper center", ncols=2)
    b.set_title(f"(a) LCO y-connectivity: cracks cut off {100 * d['LCO_cut_off_by_cracks_fraction']:.2f} % of the LCO\n"
                f"clusters {f['n_clusters']} filled → {v['n_clusters']} with cracks; crack = "
                f"{100 * res['crack_fraction_of_LCO_plus_crack']:.2f} % of LCO+crack", loc="left", fontsize=9, color=INK)

    # (b) depth profile
    p = ax[1, 0]
    yy = np.arange(ny) * (a.px_um or 1)
    p.plot(100 * py_f, yy, color=C_FILL, lw=2, label="cracks filled")
    p.plot(100 * py_v, yy, color=C_VOID, lw=2, label="cracks as void")
    p.fill_betweenx(yy, 100 * py_v, 100 * py_f, color=C_VOID, alpha=0.15, lw=0, label="cut off by cracks")
    p.set_ylim(yy[-1], yy[0])
    lo = np.nanmin(np.r_[py_v, py_f])                   # zoom: the curves usually sit close to 100 %
    p.set_xlim(max(0.0, np.floor(100 * lo) - 1), 100.5)
    p.set_ylabel(f"depth from the {a.collector} edge ({'µm' if a.px_um else 'px'})"
                 + ("  (collector at 0)" if a.collector == "top" else "  (collector at the bottom)"))
    p.set_xlabel("LCO at this depth connected to the collector (%)")
    p.xaxis.grid(True, color=GRID, lw=0.8); p.set_axisbelow(True)
    p.legend(frameon=False, loc="upper left")
    p.set_title("(b) connected LCO vs depth", loc="left", fontsize=9, color=INK)

    # (c) per-slice profile
    q = ax[0, 1]
    zz = np.arange(nz) + z0
    mk = dict(marker="o", ms=4) if nz <= 30 else {}
    q.plot(zz, 100 * pz_f, color=C_FILL, lw=2, label="cracks filled", **mk)
    q.plot(zz, 100 * pz_v, color=C_VOID, lw=2, label="cracks as void", **mk)
    q.fill_between(zz, 100 * pz_v, 100 * pz_f, color=C_VOID, alpha=0.15, lw=0)
    q.axvline(zs + z0, color=INK2, lw=0.8, ls=":")
    q.xaxis.set_major_locator(MaxNLocator(integer=True))
    q.text(zs + z0, q.get_ylim()[1], f" map slice {zs + z0}", color=INK2, fontsize=8, va="top")
    q.set_xlabel("slice z (index in the stack)"); q.set_ylabel("LCO in the slice connected (%)")
    q.yaxis.grid(True, color=GRID, lw=0.8); q.set_axisbelow(True)
    q.legend(frameon=False, loc="lower right", ncols=2)
    q.set_title("(c) connected LCO per slice", loc="left", fontsize=9, color=INK)

    # (d) map of one slice: what each LCO pixel is
    m = ax[1, 1]
    cat = np.zeros(lab2.shape, np.uint8)                # 0 other phases
    lco2 = lab2 == LCO
    cat[lco2] = 3                                       # isolated in both cases
    cat[lco2 & cf2] = 2                                 # cut off by cracks
    cat[lco2 & cv2] = 1                                 # connected in both cases
    cat[lab2 == CRACK] = 4
    cmap = ListedColormap([SURF, "#9ec5f4", C_VOID, "#b9b8b2", INK])
    ext = [0, nx * (a.px_um or 1), ny * (a.px_um or 1), 0]
    m.imshow(cat, cmap=cmap, vmin=0, vmax=4, interpolation="nearest", extent=ext)
    m.set_xlabel("x (µm)" if a.px_um else "x (px)"); m.set_ylabel("y (µm)" if a.px_um else "y (px)")
    m.spines[["top", "right"]].set_visible(True)
    n_cut = int((cat == 2).sum())
    cl, nc = ndi.label(cat == 2)                        # ring every cut-off piece: they are often only a few px
    if 0 < nc <= 300:
        s = a.px_um or 1
        cy, cx = np.array(ndi.center_of_mass(cat == 2, cl, range(1, nc + 1))).T
        m.scatter((cx + 0.5) * s, (cy + 0.5) * s, s=90, facecolors="none", edgecolors=C_VOID, linewidths=1.3)
    m.set_title(f"(d) slice {zs + z0}: {n_cut} LCO px cut off by cracks; collector at the {a.collector}", loc="left", fontsize=9, color=INK)
    m.legend(handles=[Patch(color="#9ec5f4", label="LCO connected"), Patch(color=C_VOID, label="LCO cut off by cracks (ringed)"),
                      Patch(color="#b9b8b2", label="LCO isolated anyway"), Patch(color=INK, label="crack"),
                      Patch(facecolor=SURF, edgecolor=INK2, label="pore / LLZTO")],
             frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=5, fontsize=8)

    fig.suptitle(f"{Path(res['stack']).parent.name}: {nz} slices × {ny} × {nx} px", x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
