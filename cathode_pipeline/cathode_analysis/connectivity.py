"""Stage ``connectivity``: percolation of each phase to its reference faces.

For every phase: label connected components (scipy.ndimage.label with the
configured neighbourhood), find the components that touch at least one of the
faces listed in ``connectivity.reference_faces.<phase>``, and report. Phases without
an entry in ``reference_faces`` are not analysed (e.g. pores when only their
volume fraction is of interest); they keep their code in labels_connected.

    volume fraction total / connected / isolated
    number of clusters, size of the N largest clusters
    whether the phase percolates collector -> separator

Memory: one int32 component array of the full volume exists at a time
(4 bytes/voxel) plus the uint8 labels. Use ``samples[].downsample`` when the
workstation cannot hold that.

Outputs
    connectivity.json
    labels_connected.npy | .tif   (optional) codes 0-2 connected, 3-5 isolated
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from . import ISOLATED_CODE, PHASE_CODE, PHASES
from . import io as cio
from . import volume_ops as vo
from .config import face_to_axis_side
from .pipeline import StageContext

NAME = "connectivity"
SCOPE = "sample"
DEPENDS_ON = ("segment",)

AXIS_INDEX = {"z": 0, "y": 1, "x": 2}


def expected_outputs(ctx: StageContext) -> list[Path]:
    outs = [ctx.path("connectivity.json")]
    if ctx.cfg["connectivity"]["save_connected_labels"]:
        outs.append(ctx.path(f"labels_connected.{ctx.cfg['runtime']['label_format']}"))
    return outs


def _plane(lab: np.ndarray, axis: str, side: str) -> np.ndarray:
    idx = 0 if side == "min" else -1
    return np.take(lab, idx, axis=AXIS_INDEX[axis])


def analyse_phase(labels: np.ndarray, code: int, faces: list[str], cfg: dict, n_largest: int = 10
                  ) -> tuple[dict[str, Any], np.ndarray | None]:
    """Return (stats, isolated_mask). ``isolated_mask`` is None when nothing is isolated."""
    st = vo.structure(cfg["connectivity"]["structure"])
    mask = labels == code
    n_phase = int(mask.sum())
    total = labels.size
    if n_phase == 0:
        return {"volume_fraction": 0.0, "n_clusters": 0, "connected_fraction": float("nan"),
                "percolates_collector_to_separator": False}, None
    lab, n = ndi.label(mask, structure=st)
    del mask
    sizes = np.bincount(lab.ravel(), minlength=n + 1)
    sizes[0] = 0

    touch = np.zeros(n + 1, dtype=bool)
    per_face: dict[str, int] = {}
    for face in faces:
        ids = np.unique(_plane(lab, *face_to_axis_side(face, cfg)))
        ids = ids[ids > 0]
        touch[ids] = True
        per_face[face] = int(sizes[ids].sum())

    # percolation through the thickness, independent of the reference faces
    cc_ids = np.unique(_plane(lab, *face_to_axis_side("collector", cfg)))
    sep_ids = np.unique(_plane(lab, *face_to_axis_side("separator", cfg)))
    spanning = np.intersect1d(cc_ids[cc_ids > 0], sep_ids[sep_ids > 0])

    connected_voxels = int(sizes[touch].sum())
    order = np.argsort(sizes)[::-1][:n_largest]
    largest = [{"id": int(i), "voxels": int(sizes[i]), "fraction_of_phase": float(sizes[i] / n_phase),
                "connected": bool(touch[i])} for i in order if sizes[i] > 0]

    stats = {
        "volume_fraction": n_phase / total,
        "volume_fraction_connected": connected_voxels / total,
        "volume_fraction_isolated": (n_phase - connected_voxels) / total,
        "connected_fraction": connected_voxels / n_phase,
        "reference_faces": faces,
        "voxels_touching_face": per_face,
        "n_clusters": int(n),
        "n_clusters_connected": int(touch.sum()),
        "percolates_collector_to_separator": bool(spanning.size > 0),
        "spanning_cluster_fraction_of_phase": float(sizes[spanning].sum() / n_phase),
        "largest_clusters": largest,
    }
    isolated = None
    if connected_voxels < n_phase:
        keep = touch.copy()
        keep[0] = True  # background is not "isolated"
        isolated = ~keep[lab]
    del lab
    return stats, isolated


def run(ctx: StageContext) -> dict[str, Any]:
    cfg, log = ctx.cfg, ctx.log
    ext = cfg["runtime"]["label_format"]
    labels = np.asarray(cio.load_labels(ctx.path(f"labels.{ext}"), mmap=False))
    conn = cfg["connectivity"]
    save = bool(conn["save_connected_labels"])
    out_labels = labels.copy() if save else None

    results: dict[str, Any] = {"sample": ctx.sample, "structure": conn["structure"], "phases": {}}
    analysed = [p for p in PHASES if p in conn["reference_faces"]]
    skipped = [p for p in PHASES if p not in analysed]
    if skipped:
        log.info("no reference_faces configured for %s -> connectivity not analysed", skipped)
        results["phases_not_analysed"] = skipped
    for phase in analysed:
        code = PHASE_CODE[phase]
        faces = list(conn["reference_faces"][phase])
        stats, isolated = analyse_phase(labels, code, faces, cfg, int(conn["report_largest_n_clusters"]))
        results["phases"][phase] = stats
        log.info("%-5s vf=%.4f connected=%.3f clusters=%d percolates=%s", phase, stats["volume_fraction"],
                 stats.get("connected_fraction", float("nan")), stats.get("n_clusters", 0),
                 stats["percolates_collector_to_separator"])
        if isolated is not None and out_labels is not None:
            out_labels[isolated] = ISOLATED_CODE[phase]
        del isolated

    if out_labels is not None:
        p = cio.save_labels(out_labels, ctx.path("labels_connected"), ext)
        results["labels_connected_file"] = str(p)
        log.info("saved %s", p)
    cio.save_json(results, ctx.path("connectivity.json"))
    return {p: {"connected_fraction": results["phases"][p].get("connected_fraction")} for p in analysed}
