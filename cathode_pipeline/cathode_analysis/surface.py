"""Stage ``surface``: interface areas between phases, total and active.

Two estimators:

* ``voxel_faces``: count voxel faces shared by phase a and b, times
  voxel_size^2 times an empirical staircase correction factor.
* ``marching_cubes`` (default): mesh each phase's smoothed mask with marching
  cubes (z-chunked, no boundary caps) to get the single-phase surface S_p; for
  three phases the pairwise interface is exactly
      A_ab = (S_a + S_b - S_c) / 2 .

The *active* interface counts only boundary where both phases belong to their
connected cluster (codes 0-2 in labels_connected). It is obtained from the
voxel-face count ratio active/total applied to the chosen estimator.

Areas are reported in um^2, as specific area per total volume (1/um) and per
volume of each phase of the pair.

Outputs
    surface.json
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from . import ISOLATED_CODE, PHASE_CODE, PHASES
from . import io as cio
from . import volume_ops as vo
from .pipeline import StageContext

NAME = "surface"
SCOPE = "sample"
DEPENDS_ON = ("segment", "connectivity")
LOG = logging.getLogger(__name__)


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [ctx.path("surface.json")]


def marching_cubes_area(labels: np.ndarray, code: int, sigma: float, step_size: int = 1,
                        chunk_z: int | None = 64) -> float:
    """Surface area (voxel units^2) of the smoothed iso-surface of ``labels == code``."""
    from skimage.measure import marching_cubes, mesh_surface_area

    nz = labels.shape[0]
    margin = int(np.ceil(3 * sigma)) + 1 if sigma > 0 else 1
    area = 0.0
    for z0, z1, e0, e1 in vo.iter_z_chunks(nz, chunk_z, margin):
        block = (labels[e0:e1] == code).astype(np.float32)
        if sigma > 0:
            ndi.gaussian_filter(block, sigma=sigma, output=block, mode="nearest")
        # include one extra slice so the cell between z1-1 and z1 is meshed exactly once
        core = block[z0 - e0:min(z1 + 1, nz) - e0]
        if core.shape[0] < 2 or core.min() > 0.5 or core.max() < 0.5:
            continue
        try:
            verts, faces, _, _ = marching_cubes(core, level=0.5, step_size=step_size, allow_degenerate=False)
        except (ValueError, RuntimeError):
            continue
        area += float(mesh_surface_area(verts, faces))
    return area


def run(ctx: StageContext) -> dict[str, Any]:
    cfg, log = ctx.cfg, ctx.log
    surf = cfg["surface"]
    ext = cfg["runtime"]["label_format"]
    vox = ctx.voxel_um * int(cfg["sample"].get("downsample", 1))
    chunk_z = surf.get("chunk_z") or 64

    labels = cio.load_labels(ctx.path(f"labels.{ext}"))
    conn_path = ctx.path(f"labels_connected.{ext}")
    have_conn = conn_path.exists()
    conn = cio.load_labels(conn_path) if have_conn else None
    total_vol_um3 = labels.size * vox ** 3
    counts = vo.count_codes(labels, 3)
    phase_vol_um3 = {p: counts[PHASE_CODE[p]] * vox ** 3 for p in PHASES}

    # voxel faces between all codes (0-2 on labels; 0-5 on connected labels)
    M_total = vo.face_counts(np.asarray(labels), 3, chunk_z)
    M_conn = vo.face_counts(np.asarray(conn), 6, chunk_z) if have_conn else None
    corr = float(surf["voxel_faces"]["correction_factor"])

    S_mc: dict[str, float] = {}
    if surf["method"] == "marching_cubes":
        mc = surf["marching_cubes"]
        for p in PHASES:
            S_mc[p] = marching_cubes_area(np.asarray(labels), PHASE_CODE[p], float(mc["smoothing_sigma"]),
                                          int(mc["step_size"]), chunk_z) * vox ** 2
            log.info("marching cubes surface of %-5s: %.1f um^2", p, S_mc[p])

    results: dict[str, Any] = {"sample": ctx.sample, "method": surf["method"], "voxel_size_um": vox,
                               "total_volume_um3": total_vol_um3, "phase_volume_um3": phase_vol_um3,
                               "single_phase_surface_mc_um2": S_mc, "interfaces": {}}
    for a, b in surf["interfaces"]:
        ca, cb = PHASE_CODE[a], PHASE_CODE[b]
        n_faces = int(M_total[ca, cb])
        area_vf = n_faces * vox ** 2 * corr
        if surf["method"] == "marching_cubes":
            c = next(p for p in PHASES if p not in (a, b))
            area = max(0.0, (S_mc[a] + S_mc[b] - S_mc[c]) / 2)
        else:
            area = area_vf
        entry: dict[str, Any] = {
            "faces_total": n_faces,
            "area_um2": area,
            "area_voxel_faces_um2": area_vf,
            "specific_area_per_total_volume_1_um": area / total_vol_um3,
            f"area_per_{a}_volume_1_um": area / phase_vol_um3[a] if phase_vol_um3[a] else float("nan"),
            f"area_per_{b}_volume_1_um": area / phase_vol_um3[b] if phase_vol_um3[b] else float("nan"),
        }
        if have_conn and surf["active_requires_connected"]:
            ia, ib = ISOLATED_CODE[a], ISOLATED_CODE[b]
            n_active = int(M_conn[ca, cb])
            n_all = int(M_conn[ca, cb] + M_conn[ca, ib] + M_conn[ia, cb] + M_conn[ia, ib])
            frac = n_active / n_all if n_all else float("nan")
            entry.update({
                "faces_active": n_active,
                "active_fraction": frac,
                "active_area_um2": area * frac if n_all else float("nan"),
                "active_specific_area_per_total_volume_1_um": area * frac / total_vol_um3 if n_all else float("nan"),
                f"active_area_per_{a}_volume_1_um": area * frac / phase_vol_um3[a] if n_all and phase_vol_um3[a] else float("nan"),
                f"active_area_per_{b}_volume_1_um": area * frac / phase_vol_um3[b] if n_all and phase_vol_um3[b] else float("nan"),
            })
        results["interfaces"][f"{a}-{b}"] = entry
        log.info("%s-%s: area %.1f um^2 (%.4f 1/um), active fraction %s", a, b, area, area / total_vol_um3,
                 f"{entry.get('active_fraction', float('nan')):.3f}")

    cio.save_json(results, ctx.path("surface.json"))
    return {k: v["specific_area_per_total_volume_1_um"] for k, v in results["interfaces"].items()}
