"""Stage ``profiles``: quantities resolved along the through-thickness axis.

The volume is cut into slabs of ``profiles.slab_thickness_um`` along
``voxel.through_thickness_axis``, counted from the current-collector face.
Per slab and phase: volume fraction, connected fraction (from
labels_connected), LCO-LLZTO and LCO-pore interface density (1/um, voxel
faces with the staircase correction) and grain count / density (from the
grains CSVs when present).

Outputs
    profiles.csv     tidy table: slab, position_um, position_norm, phase, quantity, value
    profiles.<fmt>   one panel per quantity
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import ISOLATED_CODE, PHASE_CODE, PHASES
from . import io as cio
from . import volume_ops as vo
from .pipeline import StageContext
from .tortuosity import AXIS_INDEX

NAME = "profiles"
SCOPE = "sample"
DEPENDS_ON = ("segment", "connectivity")


def expected_outputs(ctx: StageContext) -> list[Path]:
    fmt = ctx.cfg["report"]["figures"]["format"]
    return [ctx.path("profiles.csv"), ctx.path(f"profiles.{fmt}")]


def run(ctx: StageContext) -> dict[str, Any]:
    cfg, log = ctx.cfg, ctx.log
    prof = cfg["profiles"]
    ext = cfg["runtime"]["label_format"]
    vox = ctx.voxel_um * int(cfg["sample"].get("downsample", 1))
    axis = AXIS_INDEX[cfg["voxel"]["through_thickness_axis"]]
    from_max = cfg["voxel"]["current_collector_face"] == "max"
    corr = float(cfg["surface"]["voxel_faces"]["correction_factor"])

    labels = cio.load_labels(ctx.path(f"labels.{ext}"))
    conn_path = ctx.path(f"labels_connected.{ext}")
    conn = cio.load_labels(conn_path) if conn_path.exists() else None
    n_axis = labels.shape[axis]
    slab = max(1, int(round(float(prof["slab_thickness_um"]) / vox)))
    thickness_um = n_axis * vox
    log.info("axis=%s length=%d voxels (%.1f um), slab=%d voxels", cfg["voxel"]["through_thickness_axis"], n_axis,
             thickness_um, slab)

    grain_depths: dict[str, np.ndarray] = {}
    for p in cfg["grains"]["phases"]:
        csv = ctx.path(f"grains_{p}.csv")
        if csv.exists():
            grain_depths[p] = pd.read_csv(csv)["depth_from_collector_um"].to_numpy()

    rows: list[dict[str, Any]] = []
    quantities = list(prof["quantities"])
    conn_phases = set(cfg["connectivity"]["reference_faces"])
    for s, start in enumerate(range(0, n_axis, slab)):
        stop = min(n_axis, start + slab)
        # index range along the axis, measured from the collector face
        lo, hi = (n_axis - stop, n_axis - start) if from_max else (start, stop)
        sl = [slice(None)] * 3
        sl[axis] = slice(lo, hi)
        block = np.asarray(labels[tuple(sl)])
        pos_um = (start + stop) / 2 * vox
        base = {"slab": s, "position_um": pos_um, "position_norm": pos_um / thickness_um,
                "slab_volume_um3": block.size * vox ** 3}
        counts = np.bincount(block.ravel(), minlength=3)[:3]
        cblock = np.asarray(conn[tuple(sl)]) if conn is not None else None
        ccounts = np.bincount(cblock.ravel(), minlength=6)[:6] if cblock is not None else None
        M = vo.face_counts(block, 3, chunk_z=None) if "interface_density" in quantities else None
        for phase in prof["phases"]:
            code = PHASE_CODE[phase]
            if "volume_fraction" in quantities:
                rows.append({**base, "phase": phase, "quantity": "volume_fraction", "value": counts[code] / block.size})
            if "connected_fraction" in quantities and ccounts is not None and phase in conn_phases:
                tot = ccounts[code] + ccounts[ISOLATED_CODE[phase]]
                rows.append({**base, "phase": phase, "quantity": "connected_fraction",
                             "value": ccounts[code] / tot if tot else np.nan})
            if "grain_count" in quantities and phase in grain_depths:
                d = grain_depths[phase]
                n_g = int(((d >= start * vox) & (d < stop * vox)).sum())
                rows.append({**base, "phase": phase, "quantity": "grain_count", "value": n_g})
                rows.append({**base, "phase": phase, "quantity": "grain_density_per_um3",
                             "value": n_g / base["slab_volume_um3"]})
        if M is not None:
            for a, b in cfg["surface"]["interfaces"]:
                area = M[PHASE_CODE[a], PHASE_CODE[b]] * vox ** 2 * corr
                rows.append({**base, "phase": f"{a}-{b}", "quantity": "interface_density",
                             "value": area / base["slab_volume_um3"]})

    df = pd.DataFrame(rows)
    df.to_csv(ctx.path("profiles.csv"), index=False, float_format="%.6g")

    from . import plotting

    fmt = cfg["report"]["figures"]["format"]
    colors = dict(cfg["plotting"]["colors"])
    colors.update(cfg["plotting"].get("interface_colors", {}))
    panels = [q for q in ("volume_fraction", "connected_fraction", "interface_density", "grain_count") if q in set(df["quantity"])]
    plotting.profile_panels(df, panels, colors, ctx.path(f"profiles.{fmt}"), title=f"{ctx.sample} through-thickness profiles",
                            dpi=int(cfg["report"]["figures"]["dpi"]))
    return {"n_slabs": int(df["slab"].max()) + 1 if len(df) else 0, "thickness_um": thickness_um}
