"""Stage ``segment``: grayscale stack -> three-phase uint8 label volume.

Steps (all parameters from ``segmentation`` in the config)
1. estimate the grey-level range on a slice subsample, then load the stack and
   convert to uint8 on the fly (never holds the uint16 stack)
2. optional 3D / 2D smoothing (preprocess)
3. exact 256-bin histogram; thresholds by multi-Otsu, k-means or manual
4. map grey classes to phases via ``intensity_order`` (in-place LUT)
5. post-processing: remove small islands per phase, fill small enclosed pore
   cavities (never enclosed particles of another phase), majority vote
6. QC: histogram with thresholds, overlay PNGs at chosen z positions

Pre-segmented input (sample has ``masks`` instead of ``pattern``)
    One binary TIFF stack per phase (non-zero = phase present) is imported
    instead of steps 1-4: masks are combined by ``segmentation.masks.priority``
    (first entry wins where masks overlap), voxels covered by no mask are
    filled by the majority of their neighbours or a fixed phase
    (``segmentation.masks.unassigned``), and step 5 only runs when
    ``segmentation.masks.postprocess`` is true. QC shows the phase maps.

Outputs
    labels.npy | labels.tif      uint8 (z, y, x), codes 0/1/2
    segment.json                 thresholds, volume fractions, shape, voxel size
    qc/histogram.png, qc/overlay_z*.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from . import PHASE_CODE, PHASES
from . import io as cio
from . import volume_ops as vo
from .pipeline import StageContext

NAME = "segment"
SCOPE = "sample"
DEPENDS_ON: tuple[str, ...] = ()

LOG = logging.getLogger(__name__)


def expected_outputs(ctx: StageContext) -> list[Path]:
    ext = ctx.cfg["runtime"]["label_format"]
    return [ctx.path(f"labels.{ext}"), ctx.path("segment.json")]


# --------------------------------------------------------------------------- #
# thresholds
# --------------------------------------------------------------------------- #
def thresholds_from_histogram(hist: np.ndarray, method: str, n_classes: int = 3,
                              manual: list | None = None, seed: int = 0) -> list[float]:
    """Return n_classes-1 increasing thresholds on the 0..255 scale."""
    if method == "manual":
        return [float(t) for t in manual]
    centers = np.arange(256, dtype=np.float64)
    if method == "multi_otsu":
        from skimage.filters import threshold_multiotsu

        return [float(t) for t in threshold_multiotsu(hist=(hist.astype(np.float64), centers), classes=n_classes)]
    if method == "kmeans":
        return _kmeans_1d(hist, centers, n_classes, seed)
    raise ValueError(f"unknown threshold method {method!r}")


def _kmeans_1d(hist: np.ndarray, centers: np.ndarray, k: int, seed: int) -> list[float]:
    w = hist.astype(np.float64)
    cdf = np.cumsum(w) / w.sum()
    means = np.array([centers[np.searchsorted(cdf, (i + 0.5) / k)] for i in range(k)], dtype=np.float64)
    for _ in range(100):
        assign = np.abs(centers[:, None] - means[None, :]).argmin(axis=1)
        new = np.array([np.average(centers[assign == j], weights=w[assign == j]) if w[assign == j].sum() > 0
                        else means[j] for j in range(k)])
        if np.allclose(new, means):
            break
        means = new
    means.sort()
    return [float((means[i] + means[i + 1]) / 2) for i in range(k - 1)]


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
UNASSIGNED = 255   # sentinel label used while imported masks are combined


def run(ctx: StageContext) -> dict[str, Any]:
    if ctx.cfg["sample"].get("input_mode", "grayscale") == "masks":
        return _run_masks(ctx)
    return _run_grayscale(ctx)


def _run_grayscale(ctx: StageContext) -> dict[str, Any]:
    cfg, log = ctx.cfg, ctx.log
    seg = cfg["segmentation"]
    sample = cfg["sample"]
    rt = cfg["runtime"]
    chunk_z = int(seg["postprocess"].get("chunk_z") or 128)
    margin = int(seg["postprocess"].get("chunk_margin") or 16)

    info = cio.describe_stack(sample["input_dir"], sample["pattern"])
    log.info("input: %s", info)

    # ---- 1. grey-level range and loading ----------------------------------
    norm = seg["normalize"]
    scale, per_slice = None, None
    if norm["method"] == "none":
        maxval = np.iinfo(np.dtype(info["dtype"])).max if np.dtype(info["dtype"]).kind == "u" else 255
        scale = (0.0, float(maxval))
    elif norm["per_slice"]:
        p = norm["percentiles"] if norm["method"] == "percentile" else [0, 100]
        per_slice = lambda img: tuple(np.percentile(img[::4, ::4], p))  # noqa: E731
    else:
        step = max(1, int(seg["thresholds"]["histogram_subsample"]))
        samples = [img[::4, ::4] for img in cio.iter_slices(sample["input_dir"], sample["pattern"],
                                                             sample["slice_range"], sample["crop"], step=step)]
        arr = np.concatenate([s.ravel() for s in samples])
        del samples
        if norm["method"] == "percentile":
            scale = tuple(float(v) for v in np.percentile(arr, norm["percentiles"]))
        else:
            scale = (float(arr.min()), float(arr.max()))
        del arr
        log.info("grey-level range (%s): %.1f .. %.1f", norm["method"], *scale)

    vol = cio.load_stack(sample["input_dir"], sample["pattern"], sample["slice_range"], sample["crop"],
                         downsample=int(sample.get("downsample", 1)), scale=scale, per_slice_scale=per_slice,
                         n_workers=int(rt["n_workers"]))
    log.info("volume %s uint8, %.2f GB", vol.shape, vol.nbytes / 1e9)

    # ---- 2. preprocessing ------------------------------------------------
    pre = seg["preprocess"]
    if pre["filter"] != "none":
        log.info("preprocess: %s (3d=%s)", pre["filter"], pre["in_3d"])
        if pre["filter"] == "gaussian":
            sig = float(pre["sigma"])
            sigma = (sig, sig, sig) if pre["in_3d"] else (0, sig, sig)
            ndi.gaussian_filter(vol, sigma=sigma, output=vol, mode="nearest")
        elif pre["filter"] == "median":
            sz = int(pre["size"])
            size = (sz, sz, sz) if pre["in_3d"] else (1, sz, sz)
            for z0, z1, e0, e1 in vo.iter_z_chunks(vol.shape[0], chunk_z, sz):
                vol[z0:z1] = ndi.median_filter(vol[e0:e1], size=size, mode="nearest")[z0 - e0:z1 - e0]

    # keep a few raw slices for QC overlays before the volume is turned into labels
    qc_fracs = seg["qc"]["overlay_slices"]
    qc_idx = sorted({min(vol.shape[0] - 1, int(round(f * (vol.shape[0] - 1)))) for f in qc_fracs})
    qc_gray = {i: vol[i].copy() for i in qc_idx}

    # ---- 3. histogram + thresholds ---------------------------------------
    hist = vo.histogram_uint8(vol)
    th = seg["thresholds"]
    thresholds = thresholds_from_histogram(hist, th["method"], int(th["n_classes"]), th["manual"],
                                           int(rt["random_seed"]))
    log.info("thresholds (%s): %s", th["method"], [round(t, 1) for t in thresholds])

    # ---- 4. grey class -> phase code -------------------------------------
    order = seg["intensity_order"]
    lut = np.zeros(256, dtype=np.uint8)
    grey_class = np.searchsorted(np.asarray(thresholds), np.arange(256), side="right")
    for cls, phase in enumerate(order):
        lut[grey_class == cls] = PHASE_CODE[phase]
    labels = vo.apply_lut_inplace(vol, lut)
    del vol
    raw_fractions = _fractions(labels)
    log.info("raw volume fractions: %s", _fmt(raw_fractions))

    # ---- 5. post-processing ----------------------------------------------
    stats = _postprocess(ctx, labels, chunk_z, margin)
    fractions = _fractions(labels)
    log.info("final volume fractions: %s", _fmt(fractions))

    # ---- 6. outputs ------------------------------------------------------
    out_labels = cio.save_labels(labels, ctx.path("labels"), rt["label_format"])
    log.info("saved %s", out_labels)

    qc_dir = ctx.path("qc")
    qc_dir.mkdir(parents=True, exist_ok=True)
    from . import plotting

    colors = cfg["plotting"]["colors"]
    if seg["qc"]["save_histogram"]:
        plotting.histogram_with_thresholds(hist, thresholds, order, colors, qc_dir / "histogram.png",
                                           title=f"{ctx.sample}: grey-level histogram")
    for i, gray in qc_gray.items():
        plotting.overlay_slice(gray, labels[i], colors, float(cfg["plotting"]["overlay_alpha"]),
                               qc_dir / f"overlay_z{i:04d}.png", title=f"{ctx.sample} z={i}")

    summary = {
        "sample": ctx.sample,
        "mode": "grayscale",
        "input": info,
        "shape_zyx": list(labels.shape),
        "voxel_size_um": ctx.voxel_um * int(sample.get("downsample", 1)),
        "downsample": int(sample.get("downsample", 1)),
        "grey_scale": list(scale) if scale else "per_slice",
        "thresholds_uint8": thresholds,
        "intensity_order": order,
        "volume_fractions_raw": raw_fractions,
        "volume_fractions": fractions,
        "nominal_lco_vol_fraction": sample.get("nominal_lco_vol_fraction"),
        "postprocess": stats,
        "histogram": hist.tolist(),
        "labels_file": str(out_labels),
    }
    cio.save_json(summary, ctx.path("segment.json"))
    return {"volume_fractions": fractions, "thresholds": thresholds}


# --------------------------------------------------------------------------- #
# pre-segmented input: one binary stack per phase
# --------------------------------------------------------------------------- #
def _run_masks(ctx: StageContext) -> dict[str, Any]:
    cfg, log = ctx.cfg, ctx.log
    seg = cfg["segmentation"]
    opts = seg["masks"]
    sample = cfg["sample"]
    rt = cfg["runtime"]
    chunk_z = int(seg["postprocess"].get("chunk_z") or 128)
    margin = int(seg["postprocess"].get("chunk_margin") or 16)
    k = int(sample.get("downsample", 1))
    priority: list[str] = list(opts["priority"])

    # ---- 1. describe the three stacks and check they agree ----------------
    infos = {ph: cio.describe_stack(sample["input_dir"], sample["masks"][ph]) for ph in PHASES}
    for ph, info in infos.items():
        log.info("mask %-5s: %s", ph, info)
    shapes = {tuple(i["shape_zyx"]) for i in infos.values()}
    if len(shapes) != 1:
        raise ValueError("mask stacks differ in shape: %s" % {ph: i["shape_zyx"] for ph, i in infos.items()})

    # ---- 2. combine: lowest priority first so the highest wins on overlap --
    labels: np.ndarray | None = None
    overlap: dict[str, int] = {}
    for ph in reversed(priority):
        code = PHASE_CODE[ph]
        mask = cio.load_stack(sample["input_dir"], sample["masks"][ph], sample["slice_range"], sample["crop"],
                              downsample=k, n_workers=int(rt["n_workers"]))
        if labels is None:
            labels = np.full(mask.shape, UNASSIGNED, dtype=np.uint8)
            log.info("label volume %s uint8, %.2f GB", labels.shape, labels.nbytes / 1e9)
        n_over = 0
        for z0, z1, _, _ in vo.iter_z_chunks(labels.shape[0], chunk_z):
            fg = mask[z0:z1] > (0.5 if mask.dtype.kind == "f" else 0)
            n_over += int((fg & (labels[z0:z1] != UNASSIGNED)).sum())
            labels[z0:z1][fg] = code
        del mask
        overlap[ph] = n_over
        log.info("mask %-5s imported: %.4f of the volume; %d voxels overwrote a lower-priority phase",
                 ph, float(vo.count_codes(labels, 256)[code] / labels.size), n_over)
    assert labels is not None

    # ---- 3. voxels covered by no mask ------------------------------------
    n_unassigned = int(vo.count_codes(labels, 256)[UNASSIGNED])
    frac_unassigned = n_unassigned / labels.size
    rule = opts["unassigned"]
    codes = tuple(PHASE_CODE[p] for p in PHASES)
    n_fallback = 0
    if n_unassigned:
        if rule == "majority":
            fallback = PHASE_CODE[priority[0]]
            n_fallback = vo.fill_unassigned(labels, UNASSIGNED, codes, fallback, chunk_z=chunk_z, margin=margin)
            log.info("%d voxels (%.4f) covered by no mask -> neighbour majority (%d fell back to %s)",
                     n_unassigned, frac_unassigned, n_fallback, priority[0])
        else:
            lut = np.arange(256, dtype=np.uint8)
            lut[UNASSIGNED] = PHASE_CODE[rule]
            vo.apply_lut_inplace(labels, lut)
            log.info("%d voxels (%.4f) covered by no mask -> %s", n_unassigned, frac_unassigned, rule)
    else:
        log.info("every voxel is covered by at least one mask")
    if frac_unassigned > 0.05:
        log.warning("%.1f%% of the voxels were covered by no mask - check the input stacks", 100 * frac_unassigned)
    raw_fractions = _fractions(labels)
    log.info("imported volume fractions: %s", _fmt(raw_fractions))

    # ---- 4. optional post-processing --------------------------------------
    stats: dict[str, Any] = {"removed_voxels": {}, "filled_voxels": {}}
    if opts["postprocess"]:
        stats = _postprocess(ctx, labels, chunk_z, margin)
    fractions = _fractions(labels)
    log.info("final volume fractions: %s", _fmt(fractions))

    # ---- 5. outputs -------------------------------------------------------
    out_labels = cio.save_labels(labels, ctx.path("labels"), rt["label_format"])
    log.info("saved %s", out_labels)

    qc_dir = ctx.path("qc")
    qc_dir.mkdir(parents=True, exist_ok=True)
    from . import plotting

    colors = cfg["plotting"]["colors"]
    qc_fracs = seg["qc"]["overlay_slices"]
    qc_idx = sorted({min(labels.shape[0] - 1, int(round(f * (labels.shape[0] - 1)))) for f in qc_fracs})
    for i in qc_idx:
        plotting.label_slice(labels[i], colors, qc_dir / f"phases_z{i:04d}.png", title=f"{ctx.sample} z={i}")

    summary = {
        "sample": ctx.sample,
        "mode": "masks",
        "input": {ph: infos[ph] for ph in PHASES},
        "mask_files": {ph: sample["masks"][ph] for ph in PHASES},
        "shape_zyx": list(labels.shape),
        "voxel_size_um": ctx.voxel_um * k,
        "downsample": k,
        "grey_scale": None,
        "thresholds_uint8": None,
        "intensity_order": None,
        "masks": {
            "priority": priority,
            "overlap_voxels": overlap,
            "unassigned_voxels": n_unassigned,
            "unassigned_fraction": frac_unassigned,
            "unassigned_rule": rule,
            "unassigned_fallback_voxels": n_fallback,
        },
        "volume_fractions_raw": raw_fractions,
        "volume_fractions": fractions,
        "nominal_lco_vol_fraction": sample.get("nominal_lco_vol_fraction"),
        "postprocess": stats,
        "histogram": None,
        "labels_file": str(out_labels),
    }
    cio.save_json(summary, ctx.path("segment.json"))
    return {"volume_fractions": fractions, "unassigned_fraction": frac_unassigned}


def _postprocess(ctx: StageContext, labels: np.ndarray, chunk_z: int, margin: int) -> dict[str, Any]:
    """Island removal, cavity filling and majority vote from ``segmentation.postprocess`` (in place)."""
    log = ctx.log
    post = ctx.cfg["segmentation"]["postprocess"]
    codes = tuple(PHASE_CODE[p] for p in PHASES)
    stats: dict[str, Any] = {"removed_voxels": {}, "filled_voxels": {}}
    for phase, min_size in post["min_object_voxels"].items():
        code = PHASE_CODE[phase]
        others = tuple(c for c in codes if c != code)
        n = vo.remove_small_objects(labels, code, int(min_size), others, conn=6, chunk_z=chunk_z, margin=margin)
        stats["removed_voxels"][phase] = n
        log.info("removed %d voxels of %s islands < %d voxels", n, phase, min_size)
    fh = post["fill_holes"]
    if fh["enabled"]:
        for phase in ("lco", "llzto"):
            n = vo.fill_holes(labels, PHASE_CODE[phase], int(fh["max_voxels"]), PHASE_CODE["pore"],
                              chunk_z=chunk_z, margin=margin)
            stats["filled_voxels"][phase] = n
            log.info("filled %d enclosed pore voxels (cavities <= %d voxels) in %s", n, fh["max_voxels"], phase)
    if int(post["majority_filter_radius"]) > 0:
        vo.majority_filter(labels, int(post["majority_filter_radius"]), codes, chunk_z=chunk_z)
    return stats


def _fractions(labels: np.ndarray) -> dict[str, float]:
    counts = vo.count_codes(labels, 3)
    total = counts.sum()
    return {p: float(counts[PHASE_CODE[p]] / total) for p in PHASES}


def _fmt(d: dict[str, float]) -> str:
    return ", ".join(f"{k}={v:.4f}" for k, v in d.items())
