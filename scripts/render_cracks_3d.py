"""Dimetric 3D rendering of a segmented (cycled) volume: LCO semi-transparent, cracks red.

    python scripts/render_cracks_3d.py <segmented_folder> [--out render.png] [--size 1000]
        [--lco-step 4] [--crack-step 1] [--lco-opacity 0.15] [--show-llzto] [--interactive]

Reads LCO_stack.tif and crack_stack.tif (255 = phase) from a `predict` output folder, takes the
first --size slices/rows/columns (default 1000 x 1000 x 1000 voxels = 50 x 50 x 50 um), builds a
smoothed iso-surface of LCO on a coarse grid (--lco-step) and the crack surface at full or
near-full resolution (--crack-step), and renders them with a parallel (dimetric) camera.
Requires: pip install pyvista
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


def load(folder, name, size, step):
    """(z, y, x) bool mask from <folder>/<name>_stack.tif; size = 0 keeps the whole stack."""
    p = Path(folder) / f"{name}_stack.tif"
    with tifffile.TiffFile(str(p)) as t:
        pages = t.series[0].shape[0]
        n = pages if size <= 0 else min(size, pages)
        lim = None if size <= 0 else size
        slices = [t.pages[i].asarray()[:lim:step, :lim:step] > 0 for i in range(0, n, step)]
    return np.stack(slices)


a_ytop = a_z0 = a_z1 = None


def load_blocks(base, name, step, stack_name):
    """Assemble the z-blocks of a shear-corrected stack into one canvas.

    Expects <base>/block*/ each holding <stack_name>.tif-style phase stacks written by
    fib_rf_seg predict, and <base>/blocks.csv (realign_stack.py) with columns z0, z1,
    y_top: block b covers slices z0..z1-1 and its rows start at canvas row y_top."""
    import csv
    import re
    base = Path(base)
    dirs = sorted(d for d in base.iterdir() if d.is_dir() and d.name.startswith("block"))
    rows = list(csv.DictReader(open(base / "blocks.csv")))
    if len(rows) != len(dirs):
        raise SystemExit(f"{len(dirs)} block folders but {len(rows)} rows in blocks.csv")

    def pick(r, names, what):
        for n in names:
            for k in r:
                if k.strip().lower() == n:
                    return int(float(r[k]))
        raise SystemExit(f"blocks.csv has no column for {what}; columns are {list(r)}. "
                         f"Pass --y-top-col / --z0-col / --z1-col with the right names.")
    ycols = [a_ytop] + ["y_top", "ytop", "y0", "row0", "row_start", "rows_start", "top", "y_min", "ymin", "y_start"]
    z0cols = [a_z0] + ["z0", "z_start", "zstart", "z_min", "zmin", "slice0", "first_slice"]
    z1cols = [a_z1] + ["z1", "z_end", "zend", "z_max", "zmax", "slice1", "last_slice"]
    stacks = [load(d, name, 0, step) for d in dirs]
    info = []
    for d, r, st in zip(dirs, rows, stacks):
        y0 = pick(r, [c for c in ycols if c], "the first canvas row of the block (y_top)")
        m = re.search(r"z(\d+)-(\d+)", d.name)
        try:
            z0 = pick(r, [c for c in z0cols if c], "z0")
        except SystemExit:
            if not m:
                raise
            z0 = int(m.group(1)) - 1                   # folder names are 1-based global slice numbers
        info.append((z0, y0, st))
    H = max(y0 // step + st.shape[1] for _, y0, st in info)
    W = max(st.shape[2] for st in stacks)
    Z = max(z0 // step + st.shape[0] for z0, _, st in info)
    vol = np.zeros((Z, H, W), dtype=bool)
    for z0, y0, st in info:
        z0, y0 = z0 // step, y0 // step
        vol[z0:z0 + st.shape[0], y0:y0 + st.shape[1], :st.shape[2]] = st
        print(f"  {name}: block z {z0 * step}-{(z0 + st.shape[0]) * step} placed at rows {y0 * step}:{(y0 + st.shape[1]) * step}", flush=True)
    return vol


def surface(mask, spacing, smooth_sigma=1.0, level=0.5, chunk=120, reduction=0.0, z0=0):
    """Iso-surface of a (z, y, x) mask in a coordinate frame where X = image columns,
    Y = image rows (through-thickness, drawn vertical) and Z = slice index (milling).
    Large volumes are processed in z-chunks (with overlap) and optionally decimated
    (``reduction`` = fraction of triangles removed) so that a 1000^3 LCO surface fits in memory."""
    import pyvista as pv
    if mask.shape[0] > chunk:
        parts = []
        for a in range(0, mask.shape[0], chunk):
            b = min(mask.shape[0], a + chunk + 1)
            parts.append(surface(mask[a:b], spacing, smooth_sigma, level, chunk=10**9, reduction=reduction, z0=z0 + a))
            print(f"    LCO surface chunk {a}-{b}: {parts[-1].n_cells} triangles", flush=True)
        return pv.merge(parts) if len(parts) > 1 else parts[0]
    vol = ndi.gaussian_filter(mask.astype(np.float32), smooth_sigma) if smooth_sigma > 0 else mask.astype(np.float32)
    vol = np.ascontiguousarray(vol.transpose(2, 1, 0))           # (x, y, z)
    grid = pv.ImageData(dimensions=np.array(vol.shape) + 1, spacing=spacing, origin=(0, 0, z0 * spacing[2]))
    grid.cell_data["v"] = vol.ravel(order="F")
    surf = grid.cell_data_to_point_data().contour([level], scalars="v")
    if reduction > 0 and surf.n_cells > 0:
        surf = surf.triangulate().decimate_pro(reduction, preserve_topology=True)
    return surf


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--out", default="render_cracks_3d.png")
    ap.add_argument("--size", type=int, default=1000)
    ap.add_argument("--lco-step", type=int, default=1, help="downsampling of the LCO surface (1 = same grid as the cracks; 4 -> 250^3 grid, much faster)")
    ap.add_argument("--lco-decimate", type=float, default=0.8, help="fraction of LCO triangles removed after extraction (0 = keep all)")
    ap.add_argument("--crack-step", type=int, default=1, help="downsampling of the crack surface")
    ap.add_argument("--crack-dilate", type=int, default=1, help="dilate cracks by this many voxels so 1-px cracks are visible")
    ap.add_argument("--lco-opacity", type=float, default=0.15)
    ap.add_argument("--show-llzto", action="store_true", help="also draw LLZTO (light, semi-transparent)")
    ap.add_argument("--voxel-um", type=float, nargs="+", default=[0.05], help="voxel size in um: one value, or three (x y z)")
    ap.add_argument("--blocks", action="store_true", help="<folder> holds block*/ sub-stacks plus blocks.csv (shear-corrected stack); assemble them")
    ap.add_argument("--y-top-col", default=None, help="blocks.csv column with the first canvas row of each block (auto-detected: y_top, y0, row0, ...)")
    ap.add_argument("--z0-col", default=None, help="blocks.csv column with the first slice index (0-based) of each block; else taken from the folder name block*_zAAAA-BBBB")
    ap.add_argument("--z1-col", default=None)
    ap.add_argument("--window", type=int, nargs=2, default=(1800, 1600))
    ap.add_argument("--interactive", action="store_true", help="open an interactive window instead of writing a PNG")
    a = ap.parse_args()
    import pyvista as pv

    global a_ytop, a_z0, a_z1
    a_ytop, a_z0, a_z1 = a.y_top_col, a.z0_col, a.z1_col
    v = a.voxel_um if len(a.voxel_um) == 3 else a.voxel_um * 3
    vx, vy, vz = v
    if a.blocks:
        lco = load_blocks(a.folder, "LCO", a.lco_step, "LCO_stack.tif")
    else:
        lco = load(a.folder, "LCO", a.size, a.lco_step)
    print("LCO grid", lco.shape)
    extent = (lco.shape[2] * a.lco_step * vx, lco.shape[1] * a.lco_step * vy, lco.shape[0] * a.lco_step * vz)
    lco_surf = surface(lco, (vx * a.lco_step, vy * a.lco_step, vz * a.lco_step), smooth_sigma=1.0, reduction=a.lco_decimate)
    del lco
    crack = load_blocks(a.folder, "crack", a.crack_step, "crack_stack.tif") if a.blocks else load(a.folder, "crack", a.size, a.crack_step)
    if a.crack_dilate > 0:
        crack = ndi.binary_dilation(crack, iterations=a.crack_dilate)
    print("crack grid", crack.shape, f"{crack.mean()*100:.2f}% of voxels")
    crack_surf = surface(crack, (vx * a.crack_step, vy * a.crack_step, vz * a.crack_step), smooth_sigma=0.5)
    print("triangles: LCO", lco_surf.n_cells, "crack", crack_surf.n_cells)

    pl = pv.Plotter(off_screen=not a.interactive, window_size=list(a.window))
    pl.set_background("white")
    Lx, Ly, Lz = extent
    pl.add_mesh(pv.Cube(center=(Lx / 2, Ly / 2, Lz / 2), x_length=Lx, y_length=Ly, z_length=Lz).extract_all_edges(),
                color="black", line_width=1.5)
    pl.add_mesh(lco_surf, color=(0.55, 0.6, 0.65), opacity=a.lco_opacity, smooth_shading=True, specular=0.2)
    if a.show_llzto:
        llz = load_blocks(a.folder, "LLZTO", a.lco_step, "LLZTO_stack.tif") if a.blocks else load(a.folder, "LLZTO", a.size, a.lco_step)
        pl.add_mesh(surface(llz, (vx * a.lco_step, vy * a.lco_step, vz * a.lco_step)), color=(0.95, 0.9, 0.7), opacity=0.08, smooth_shading=True)
    pl.add_mesh(crack_surf, color="red", opacity=1.0, smooth_shading=True)
    # dimetric: parallel projection, viewed from a direction where two axes are equally foreshortened
    pl.camera.parallel_projection = True
    L = max(extent)
    pl.camera.position = (Lx / 2 + 2.2 * L, Ly / 2 + 1.0 * L, Lz / 2 + 1.6 * L)   # two axes equally foreshortened
    pl.camera.focal_point = (Lx / 2, Ly / 2, Lz / 2)
    pl.camera.up = (0, -1, 0)                                                 # y (image rows) grows downwards: collector on top
    pl.reset_camera()
    pl.camera.zoom(1.3)
    pl.add_axes(xlabel="x", ylabel="y (thickness)", zlabel="z (milling)")
    pl.add_text(f"LCO (grey, {a.lco_opacity:.0%} opaque) and cracks (red)   {Lx:.0f} x {Ly:.0f} x {Lz:.0f} um (x, y, z)", font_size=12, color="black")
    if a.interactive:
        pl.show()
    else:
        pl.screenshot(a.out, transparent_background=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
