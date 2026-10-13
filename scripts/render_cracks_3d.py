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
    p = Path(folder) / f"{name}_stack.tif"
    with tifffile.TiffFile(str(p)) as t:
        n = len(t.series[0].shape) and t.series[0].shape[0]
        pages = t.series[0].shape[0]
        zs = range(0, min(size, pages), step)
        slices = [t.pages[i].asarray()[:size:step, :size:step] > 0 for i in zs]
    return np.stack(slices)


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
    ap.add_argument("--voxel-um", type=float, default=0.05)
    ap.add_argument("--window", type=int, nargs=2, default=(1800, 1600))
    ap.add_argument("--interactive", action="store_true", help="open an interactive window instead of writing a PNG")
    a = ap.parse_args()
    import pyvista as pv

    um = a.voxel_um
    lco = load(a.folder, "LCO", a.size, a.lco_step)
    print("LCO grid", lco.shape)
    lco_surf = surface(lco, (um * a.lco_step,) * 3, smooth_sigma=1.0, reduction=a.lco_decimate)
    del lco
    crack = load(a.folder, "crack", a.size, a.crack_step)
    if a.crack_dilate > 0:
        crack = ndi.binary_dilation(crack, iterations=a.crack_dilate)
    print("crack grid", crack.shape, f"{crack.mean()*100:.2f}% of voxels")
    crack_surf = surface(crack, (um * a.crack_step,) * 3, smooth_sigma=0.5)
    print("triangles: LCO", lco_surf.n_cells, "crack", crack_surf.n_cells)

    pl = pv.Plotter(off_screen=not a.interactive, window_size=list(a.window))
    pl.set_background("white")
    L = a.size * um
    pl.add_mesh(pv.Cube(center=(L / 2, L / 2, L / 2), x_length=L, y_length=L, z_length=L).extract_all_edges(),
                color="black", line_width=1.5)
    pl.add_mesh(lco_surf, color=(0.55, 0.6, 0.65), opacity=a.lco_opacity, smooth_shading=True, specular=0.2)
    if a.show_llzto:
        llz = load(a.folder, "LLZTO", a.size, a.lco_step)
        pl.add_mesh(surface(llz, (um * a.lco_step,) * 3), color=(0.95, 0.9, 0.7), opacity=0.08, smooth_shading=True)
    pl.add_mesh(crack_surf, color="red", opacity=1.0, smooth_shading=True)
    # dimetric: parallel projection, viewed from a direction where two axes are equally foreshortened
    pl.camera.parallel_projection = True
    pl.camera.position = (L / 2 + 2.2 * L, L / 2 + 1.0 * L, L / 2 + 1.6 * L)   # two axes equally foreshortened
    pl.camera.focal_point = (L / 2, L / 2, L / 2)
    pl.camera.up = (0, -1, 0)                                                 # y (image rows) grows downwards: collector on top
    pl.reset_camera()
    pl.camera.zoom(1.3)
    pl.add_axes(xlabel="x", ylabel="y (thickness)", zlabel="z (milling)")
    pl.add_text(f"LCO (grey, {a.lco_opacity:.0%} opaque) and cracks (red)   {L:.0f} x {L:.0f} x {L:.0f} um", font_size=12, color="black")
    if a.interactive:
        pl.show()
    else:
        pl.screenshot(a.out, transparent_background=False)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
