"""Stage ``grains``: split each phase into individual grains / particles.

Methods
    snow            porespy.filters.snow_partitioning (falls back to watershed_hmax
                    with a warning when porespy is not installed)
    watershed_hmax  Euclidean distance transform -> gaussian smoothing -> h-maxima
                    markers -> marker-controlled watershed (scikit-image only)

Grains smaller than ``min_grain_voxels`` are merged into the neighbouring grain
they share the most faces with (or dropped if they have no grain neighbour).
Per grain: volume, equivalent sphere diameter, centroid, surface (corrected
voxel faces), sphericity, contact area with the configured phases.

Memory: the distance transform is float64 (8 bytes/voxel) over the whole
volume; use ``grains.downsample`` for large stacks.

Outputs (per phase in ``grains.phases``)
    grains_<phase>.npy | .tif    int32 grain labels (0 = not this phase)
    grains_<phase>.csv           one row per grain
    grains.json                  summary statistics and size distributions
    qc/grain_size_<phase>.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from . import PHASE_CODE, PHASES
from . import io as cio
from . import volume_ops as vo
from .pipeline import StageContext
from .tortuosity import AXIS_INDEX, block_reduce_mask

NAME = "grains"
SCOPE = "sample"
DEPENDS_ON = ("segment",)
LOG = logging.getLogger(__name__)


def expected_outputs(ctx: StageContext) -> list[Path]:
    ext = ctx.cfg["runtime"]["label_format"]
    outs = [ctx.path("grains.json")]
    for ph in ctx.cfg["grains"]["phases"]:
        outs += [ctx.path(f"grains_{ph}.{ext}"), ctx.path(f"grains_{ph}.csv")]
    return outs


# --------------------------------------------------------------------------- #
# partitioning
# --------------------------------------------------------------------------- #
def partition_watershed_hmax(mask: np.ndarray, h: float, sigma: float = 0.0) -> np.ndarray:
    from skimage.morphology import h_maxima
    from skimage.segmentation import watershed

    dt = ndi.distance_transform_edt(mask)
    if sigma > 0:
        ndi.gaussian_filter(dt, sigma=sigma, output=dt)
    peaks = h_maxima(dt, h)
    markers, n = ndi.label(peaks)
    del peaks
    regions = watershed(-dt, markers, mask=mask)
    del dt, markers
    return regions.astype(np.int32)


def partition_snow(mask: np.ndarray, sigma: float, r_max: int) -> np.ndarray:
    import porespy as ps

    res = ps.filters.snow_partitioning(mask, sigma=sigma, r_max=r_max)
    return np.asarray(res.regions, dtype=np.int32)


def merge_small_regions(regions: np.ndarray, min_size: int) -> tuple[np.ndarray, int]:
    """Merge regions smaller than ``min_size`` into their best-connected neighbour."""
    from skimage.segmentation import relabel_sequential

    for _ in range(3):
        n = int(regions.max())
        sizes = np.bincount(regions.ravel(), minlength=n + 1)
        small = sizes < min_size
        small[0] = False
        if not small.any():
            break
        keys = []
        for ax in range(3):
            a = np.moveaxis(regions, ax, 0)[:-1]
            b = np.moveaxis(regions, ax, 0)[1:]
            sel = (a != b) & (a > 0) & (b > 0) & (small[a] | small[b])
            if sel.any():
                aa, bb = a[sel].astype(np.int64), b[sel].astype(np.int64)
                keys.append(np.concatenate([aa * (n + 1) + bb, bb * (n + 1) + aa]))
        mapping = np.arange(n + 1, dtype=np.int32)
        if keys:
            k, cnt = np.unique(np.concatenate(keys), return_counts=True)
            src, dst = k // (n + 1), k % (n + 1)
            sel = small[src] & ~small[dst]
            src, dst, cnt = src[sel], dst[sel], cnt[sel]
            order = np.lexsort((-cnt, src))
            src, dst = src[order], dst[order]
            first = np.unique(src, return_index=True)[1]
            mapping[src[first]] = dst[first]
        unresolved = small & (mapping == np.arange(n + 1))
        mapping[unresolved] = 0
        regions = mapping[regions]
    regions, _, _ = relabel_sequential(regions)
    return regions.astype(np.int32), int(regions.max())


# --------------------------------------------------------------------------- #
# measurements
# --------------------------------------------------------------------------- #
def grain_faces(regions: np.ndarray, labels: np.ndarray, contact_codes: dict[str, int], chunk_z: int = 64
                ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Per grain: number of boundary voxel faces, and faces touching each contact phase."""
    n = int(regions.max())
    surface = np.zeros(n + 1, dtype=np.int64)
    contact = {p: np.zeros(n + 1, dtype=np.int64) for p in contact_codes}
    nz = regions.shape[0]
    for z0, z1, _, _ in vo.iter_z_chunks(nz, chunk_z):
        ze = min(nz, z1 + 1)
        r, l = regions[z0:ze], labels[z0:ze]
        for ax in range(3):
            ra, rb = np.moveaxis(r, ax, 0)[:-1], np.moveaxis(r, ax, 0)[1:]
            la, lb = np.moveaxis(l, ax, 0)[:-1], np.moveaxis(l, ax, 0)[1:]
            if ax != 0:
                ra, rb, la, lb = ra[:z1 - z0], rb[:z1 - z0], la[:z1 - z0], lb[:z1 - z0]
            diff = ra != rb
            surface += np.bincount(ra[diff], minlength=n + 1)
            surface += np.bincount(rb[diff], minlength=n + 1)
            for p, code in contact_codes.items():
                contact[p] += np.bincount(ra[diff & (lb == code)], minlength=n + 1)
                contact[p] += np.bincount(rb[diff & (la == code)], minlength=n + 1)
    return surface, contact


def measure(regions: np.ndarray, labels: np.ndarray, vox: float, contact_codes: dict[str, int],
            corr: float, tt_axis: int, cc_face: str) -> pd.DataFrame:
    from skimage.measure import regionprops_table

    props = regionprops_table(regions, properties=("label", "area", "centroid", "bbox"))
    df = pd.DataFrame(props).rename(columns={"area": "volume_voxels", "centroid-0": "centroid_z",
                                             "centroid-1": "centroid_y", "centroid-2": "centroid_x"})
    df["volume_um3"] = df["volume_voxels"] * vox ** 3
    df["eq_diameter_um"] = (6 * df["volume_um3"] / np.pi) ** (1 / 3)
    n_axis = regions.shape[tt_axis]
    c = df[["centroid_z", "centroid_y", "centroid_x"][tt_axis]]
    df["depth_from_collector_um"] = (c if cc_face == "min" else (n_axis - 1 - c)) * vox
    surface, contact = grain_faces(regions, labels, contact_codes)
    lab = df["label"].to_numpy()
    df["surface_um2"] = surface[lab] * vox ** 2 * corr
    df["sphericity"] = np.pi ** (1 / 3) * (6 * df["volume_um3"]) ** (2 / 3) / df["surface_um2"].replace(0, np.nan)
    for p in contact_codes:
        df[f"contact_{p}_um2"] = contact[p][lab] * vox ** 2 * corr
        df[f"contact_{p}_fraction"] = df[f"contact_{p}_um2"] / df["surface_um2"].replace(0, np.nan)
    shp = regions.shape
    df["touches_boundary"] = ((df["bbox-0"] == 0) | (df["bbox-1"] == 0) | (df["bbox-2"] == 0) |
                              (df["bbox-3"] == shp[0]) | (df["bbox-4"] == shp[1]) | (df["bbox-5"] == shp[2]))
    return df.drop(columns=[c for c in df.columns if c.startswith("bbox-")])


def size_distribution(df: pd.DataFrame, n_bins: int) -> dict[str, Any]:
    d = df["eq_diameter_um"].to_numpy()
    v = df["volume_um3"].to_numpy()
    if d.size == 0:
        return {}
    edges = np.linspace(0, float(d.max()) * 1.001, n_bins + 1)
    n_hist, _ = np.histogram(d, bins=edges)
    v_hist, _ = np.histogram(d, bins=edges, weights=v)
    order = np.argsort(d)
    cum_v = np.cumsum(v[order]) / v.sum()
    dv = {f"d{q}_volume_weighted_um": float(np.interp(q / 100, cum_v, d[order])) for q in (10, 50, 90)}
    return {"bin_edges_um": edges.tolist(), "number_fraction": (n_hist / n_hist.sum()).tolist(),
            "volume_fraction": (v_hist / v_hist.sum()).tolist(),
            "d_mean_um": float(d.mean()), "d_median_um": float(np.median(d)),
            **{f"d{q}_number_um": float(np.percentile(d, q)) for q in (10, 50, 90)}, **dv}


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
def run(ctx: StageContext) -> dict[str, Any]:
    cfg, log = ctx.cfg, ctx.log
    g = cfg["grains"]
    ext = cfg["runtime"]["label_format"]
    k = int(g.get("downsample", 1))
    vox = ctx.voxel_um * int(cfg["sample"].get("downsample", 1)) * k
    labels = np.asarray(cio.load_labels(ctx.path(f"labels.{ext}")))
    if k > 1:
        # majority-vote downsample of the label volume, phase by phase
        small = np.zeros(tuple(s // k for s in labels.shape), dtype=np.uint8)
        for code in (1, 2):
            small[block_reduce_mask(labels == code, k)] = code
        labels = small
    corr = float(cfg["surface"]["voxel_faces"]["correction_factor"])
    tt_axis = AXIS_INDEX[cfg["voxel"]["through_thickness_axis"]]
    cc_face = cfg["voxel"]["current_collector_face"]
    contact_codes = {p: PHASE_CODE[p] for p in PHASES}   # contact measured against every other phase

    from . import plotting

    summary: dict[str, Any] = {"sample": ctx.sample, "method": g["method"], "voxel_size_um": vox, "phases": {}}
    ret = {}
    for phase in g["phases"]:
        mask = labels == PHASE_CODE[phase]
        method = g["method"]
        if method == "snow":
            try:
                regions = partition_snow(mask, float(g["snow"]["sigma"]), int(g["snow"]["r_max"]))
            except ImportError:
                log.warning("porespy not installed; falling back to watershed_hmax")
                method = "watershed_hmax"
        if method == "watershed_hmax":
            regions = partition_watershed_hmax(mask, float(g["watershed_hmax"]["h"]), float(g["snow"]["sigma"]))
        del mask
        n_raw = int(regions.max())
        regions, n = merge_small_regions(regions, int(g["min_grain_voxels"]))
        log.info("%s: %d regions -> %d grains after merging < %d voxels", phase, n_raw, n, g["min_grain_voxels"])

        df = measure(regions, labels, vox, {p: c for p, c in contact_codes.items() if p != phase}, corr,
                     tt_axis, cc_face)
        df.insert(0, "phase", phase)
        df.to_csv(ctx.path(f"grains_{phase}.csv"), index=False, float_format="%.5g")
        cio.save_labels(regions, ctx.path(f"grains_{phase}"), ext)
        dist = size_distribution(df, int(g["measure"]["n_bins_size_distribution"]))
        stats = {
            "n_grains": int(n), "n_regions_before_merge": n_raw,
            "mean_sphericity": float(df["sphericity"].mean()) if n else float("nan"),
            "interior_grains": int((~df["touches_boundary"]).sum()),
            **{f"mean_contact_{p}_fraction": float(df[f"contact_{p}_fraction"].mean())
               for p in contact_codes if p != phase and n},
            "size_distribution": dist,
        }
        summary["phases"][phase] = stats
        ret[phase] = {"n_grains": n, "d50_um": dist.get("d50_number_um")}
        log.info("%s: d50 (number) = %s um, d50 (volume) = %s um", phase,
                 f"{dist.get('d50_number_um', float('nan')):.3f}", f"{dist.get('d50_volume_weighted_um', float('nan')):.3f}")
        if dist:
            plotting.size_distribution(np.asarray(dist["bin_edges_um"]), np.asarray(dist["number_fraction"]),
                                       np.asarray(dist["volume_fraction"]), cfg["plotting"]["colors"][phase],
                                       ctx.path("qc") / f"grain_size_{phase}.png",
                                       title=f"{ctx.sample}: {phase} grain size (n={n})")
        del regions
    cio.save_json(summary, ctx.path("grains.json"))
    return ret
