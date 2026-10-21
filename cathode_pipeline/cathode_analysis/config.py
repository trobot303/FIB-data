"""Configuration loading, validation and resolution.

The YAML file is the single source of truth. This module

* loads it,
* validates types / allowed values / cross-references (fail early, before any
  multi-GB stack is touched),
* resolves relative paths against the YAML file location,
* applies per-sample ``overrides`` to produce one fully resolved config per
  sample,
* writes the resolved config next to the results for provenance.

No third-party schema library is used on purpose: the validator below is
small, explicit and has no extra dependency on the workstation.
"""

from __future__ import annotations

import copy
import datetime as _dt
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from . import PHASES

SCHEMA_VERSION = 1

STAGES = ("segment", "connectivity", "surface", "tortuosity", "grains", "profiles", "report")

# Allowed values for enumerated options: (dotted key) -> allowed set
_ENUMS: dict[str, tuple[Any, ...]] = {
    "runtime.log_level": ("DEBUG", "INFO", "WARNING", "ERROR"),
    "runtime.device": ("auto", "cpu", "cuda"),
    "runtime.label_format": ("npy", "tif"),
    "voxel.through_thickness_axis": ("x", "y", "z"),
    "voxel.current_collector_face": ("min", "max"),
    "segmentation.normalize.method": ("none", "percentile", "minmax"),
    "segmentation.preprocess.filter": ("none", "gaussian", "median"),
    "segmentation.thresholds.method": ("multi_otsu", "manual", "kmeans"),
    "connectivity.structure": (6, 18, 26),
    "surface.method": ("marching_cubes", "voxel_faces"),
    "tortuosity.backend": ("taufactor", "porespy"),
    "grains.method": ("snow", "watershed_hmax"),
    "report.figures.format": ("png", "pdf", "svg"),
}

# Required top-level sections
_SECTIONS = (
    "project", "paths", "runtime", "voxel", "samples", "segmentation", "connectivity",
    "surface", "tortuosity", "grains", "profiles", "report", "plotting",
)

_FACE_ALIASES = ("collector", "separator")


class ConfigError(ValueError):
    """Raised for any invalid or inconsistent configuration."""


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _get(cfg: Mapping[str, Any], dotted: str) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, Mapping) or part not in node:
            raise ConfigError(f"missing config key: {dotted}")
        node = node[part]
    return node


def _deep_update(base: dict, patch: Mapping) -> dict:
    """Recursively merge ``patch`` into a copy of ``base``."""
    out = copy.deepcopy(base)
    for k, v in patch.items():
        if isinstance(v, Mapping) and isinstance(out.get(k), Mapping):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _check_range_pair(value: Any, where: str) -> None:
    if not (isinstance(value, (list, tuple)) and len(value) == 2):
        raise ConfigError(f"{where} must be a 2-element list [start, stop], got {value!r}")
    for v in value:
        if v is not None and not isinstance(v, int):
            raise ConfigError(f"{where} entries must be int or null, got {v!r}")


def _check_phases(values: Iterable[Any], where: str) -> None:
    for v in values:
        if v not in PHASES:
            raise ConfigError(f"{where}: unknown phase {v!r}; allowed: {PHASES}")


def _check_faces(values: Iterable[Any], where: str) -> None:
    for v in values:
        if v in _FACE_ALIASES:
            continue
        if not (isinstance(v, str) and len(v) == 5 and v[0] in "xyz" and v[1] == "_" and v[2:] in ("min", "max")):
            raise ConfigError(f"{where}: bad face {v!r}; use collector | separator | <axis>_min | <axis>_max")


# --------------------------------------------------------------------------- #
# validation
# --------------------------------------------------------------------------- #
def validate(cfg: Mapping[str, Any]) -> None:
    """Validate the raw (unresolved) config. Raises :class:`ConfigError`."""
    if not isinstance(cfg, Mapping):
        raise ConfigError("config root must be a mapping")
    if cfg.get("schema_version") != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}, got {cfg.get('schema_version')!r}")
    for sec in _SECTIONS:
        if sec not in cfg:
            raise ConfigError(f"missing config section: {sec}")

    for key, allowed in _ENUMS.items():
        val = _get(cfg, key)
        if val not in allowed:
            raise ConfigError(f"{key} = {val!r}; allowed: {allowed}")

    if not (isinstance(_get(cfg, "voxel.size_nm"), (int, float)) and _get(cfg, "voxel.size_nm") > 0):
        raise ConfigError("voxel.size_nm must be a positive number")
    if not (isinstance(_get(cfg, "runtime.n_workers"), int) and _get(cfg, "runtime.n_workers") >= 1):
        raise ConfigError("runtime.n_workers must be an integer >= 1")

    samples = cfg["samples"]
    if not isinstance(samples, list) or not samples:
        raise ConfigError("samples must be a non-empty list")
    names = set()
    for i, s in enumerate(samples):
        where = f"samples[{i}]"
        for req in ("name", "input_dir"):
            if req not in s:
                raise ConfigError(f"{where}.{req} is required")
        has_pattern, has_masks = "pattern" in s, "masks" in s
        if has_pattern == has_masks:
            raise ConfigError(f"{where}: give exactly one of 'pattern' (grayscale slices) or "
                              "'masks' (pre-segmented binary stack per phase)")
        if has_masks:
            masks = s["masks"]
            if not (isinstance(masks, Mapping) and sorted(masks) == sorted(PHASES)
                    and all(isinstance(v, str) and v for v in masks.values())):
                raise ConfigError(f"{where}.masks must map every phase {PHASES} to a file name / glob "
                                  f"relative to input_dir, got {masks!r}")
        if s["name"] in names:
            raise ConfigError(f"duplicate sample name {s['name']!r}")
        names.add(s["name"])
        frac = s.get("nominal_lco_vol_fraction")
        if frac is not None and not (0.0 <= float(frac) <= 1.0):
            raise ConfigError(f"{where}.nominal_lco_vol_fraction must be within [0, 1]")
        _check_range_pair(s.get("slice_range", [None, None]), f"{where}.slice_range")
        crop = s.get("crop", {})
        for ax in ("z", "y", "x"):
            _check_range_pair(crop.get(ax, [None, None]), f"{where}.crop.{ax}")
        if not isinstance(s.get("overrides", {}), Mapping):
            raise ConfigError(f"{where}.overrides must be a mapping")
        ds = s.get("downsample", 1)
        if not (isinstance(ds, int) and ds >= 1):
            raise ConfigError(f"{where}.downsample must be an integer >= 1")

    av = cfg.get("analysis_volume")
    if av is not None:
        shape = av.get("shape_zyx")
        if not (isinstance(shape, list) and len(shape) == 3 and all(isinstance(v, int) and v > 0 for v in shape)):
            raise ConfigError("analysis_volume.shape_zyx must be three positive integers [z, y, x]")
        if av.get("enforce", True):
            for s in samples:
                crop = s.get("crop", {})
                for ax, want in zip("zyx", shape):
                    rng = crop.get(ax, [None, None])
                    if rng[0] is None or rng[1] is None:
                        raise ConfigError(f"samples[{s['name']}].crop.{ax} must be explicit [start, stop] when "
                                          "analysis_volume is enforced")
                    if rng[1] - rng[0] != want:
                        raise ConfigError(f"samples[{s['name']}].crop.{ax} spans {rng[1] - rng[0]} voxels, "
                                          f"analysis_volume.shape_zyx requires {want}")

    seg = cfg["segmentation"]
    order = seg.get("intensity_order", [])
    if sorted(order) != sorted(PHASES):
        raise ConfigError(f"segmentation.intensity_order must be a permutation of {PHASES}, got {order}")
    if seg["thresholds"]["method"] == "manual":
        man = seg["thresholds"].get("manual")
        if not (isinstance(man, list) and len(man) == 2 and all(isinstance(t, (int, float)) for t in man)):
            raise ConfigError("segmentation.thresholds.manual needs two numeric thresholds when method = manual")
        if not man[0] < man[1]:
            raise ConfigError("segmentation.thresholds.manual must be increasing")
    _check_phases(seg["postprocess"]["min_object_voxels"].keys(), "segmentation.postprocess.min_object_voxels")
    masks = seg.get("masks", {})
    if not isinstance(masks, Mapping):
        raise ConfigError("segmentation.masks must be a mapping")
    prio = masks.get("priority", list(PHASES))
    if sorted(prio) != sorted(PHASES):
        raise ConfigError(f"segmentation.masks.priority must be a permutation of {PHASES}, got {prio}")
    unassigned = masks.get("unassigned", "majority")
    if unassigned not in ("majority",) + PHASES:
        raise ConfigError(f"segmentation.masks.unassigned must be 'majority' or one of {PHASES}, got {unassigned!r}")
    if not isinstance(masks.get("postprocess", False), bool):
        raise ConfigError("segmentation.masks.postprocess must be a bool")
    fh = seg["postprocess"].get("fill_holes")
    if not (isinstance(fh, Mapping) and isinstance(fh.get("enabled"), bool) and isinstance(fh.get("max_voxels"), int)):
        raise ConfigError("segmentation.postprocess.fill_holes must be {enabled: bool, max_voxels: int}")

    conn = cfg["connectivity"]
    _check_phases(conn["reference_faces"].keys(), "connectivity.reference_faces")
    for ph, faces in conn["reference_faces"].items():
        _check_faces(faces, f"connectivity.reference_faces.{ph}")

    for pair in cfg["surface"]["interfaces"]:
        if not (isinstance(pair, list) and len(pair) == 2):
            raise ConfigError("surface.interfaces entries must be [phase_a, phase_b]")
        _check_phases(pair, "surface.interfaces")
        if pair[0] == pair[1]:
            raise ConfigError(f"surface.interfaces: pair {pair} has identical phases")

    tor = cfg["tortuosity"]
    _check_phases(tor["phases"], "tortuosity.phases")
    for d in tor["directions"]:
        if d not in ("x", "y", "z"):
            raise ConfigError(f"tortuosity.directions: bad axis {d!r}")
    if not (isinstance(tor["downsample"], int) and tor["downsample"] >= 1):
        raise ConfigError("tortuosity.downsample must be an integer >= 1")

    _check_phases(cfg["grains"]["phases"], "grains.phases")

    prof = cfg["profiles"]
    _check_phases(prof["phases"], "profiles.phases")
    if not (isinstance(prof["slab_thickness_um"], (int, float)) and prof["slab_thickness_um"] > 0):
        raise ConfigError("profiles.slab_thickness_um must be > 0")

    _check_phases(cfg["plotting"]["colors"].keys(), "plotting.colors")


# --------------------------------------------------------------------------- #
# loading / resolution
# --------------------------------------------------------------------------- #
def load_config(path: str | Path) -> dict[str, Any]:
    """Load, validate and path-resolve a YAML config.

    Adds a ``_meta`` section with ``config_path``, ``config_dir`` and
    ``loaded_at`` (ISO timestamp). Relative ``paths.*`` entries become absolute.
    """
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    validate(cfg)

    cfg_dir = path.parent
    for key in ("data_root", "output_root", "log_dir"):
        cfg["paths"][key] = str((cfg_dir / cfg["paths"][key]).resolve())

    cfg["_meta"] = {
        "config_path": str(path),
        "config_dir": str(cfg_dir),
        "loaded_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    return cfg


def resolve_sample(cfg: Mapping[str, Any], sample_name: str) -> dict[str, Any]:
    """Return a config specialised for one sample.

    The result is a deep copy of ``cfg`` with the sample's ``overrides`` merged
    into the top-level sections, plus a ``sample`` section containing the
    sample entry with an absolute ``input_dir`` and an ``output_dir``.
    """
    entry = next((s for s in cfg["samples"] if s["name"] == sample_name), None)
    if entry is None:
        raise ConfigError(f"unknown sample {sample_name!r}")
    out = _deep_update(dict(cfg), entry.get("overrides", {}))
    sample = copy.deepcopy(entry)
    sample["input_dir"] = str((Path(cfg["paths"]["data_root"]) / entry["input_dir"]).resolve())
    sample["output_dir"] = str((Path(cfg["paths"]["output_root"]) / entry["name"]).resolve())
    sample.setdefault("slice_range", [None, None])
    sample.setdefault("downsample", 1)
    sample.setdefault("crop", {})
    for ax in ("z", "y", "x"):
        sample["crop"].setdefault(ax, [None, None])
    sample.pop("overrides", None)
    sample["input_mode"] = "masks" if "masks" in sample else "grayscale"
    out["sample"] = sample
    seg = out.setdefault("segmentation", {})
    masks = dict(seg.get("masks") or {})
    masks.setdefault("priority", list(PHASES))
    masks.setdefault("unassigned", "majority")
    masks.setdefault("postprocess", False)
    seg["masks"] = masks
    return out


def select_samples(cfg: Mapping[str, Any], names: Iterable[str] | None) -> list[str]:
    """Return sample names to process: all, or the given subset (validated)."""
    all_names = [s["name"] for s in cfg["samples"]]
    if not names:
        return all_names
    bad = [n for n in names if n not in all_names]
    if bad:
        raise ConfigError(f"unknown sample(s) {bad}; available: {all_names}")
    return [n for n in all_names if n in set(names)]


def save_resolved(cfg: Mapping[str, Any], dest: str | Path) -> Path:
    """Write the resolved config (with ``_meta``) as YAML for provenance."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with open(dest, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dict(cfg), fh, sort_keys=False, default_flow_style=False)
    return dest


def face_to_axis_side(face: str, cfg: Mapping[str, Any]) -> tuple[str, str]:
    """Map a face name (``collector``, ``separator``, ``y_min``...) to (axis, side)."""
    if face in _FACE_ALIASES:
        axis = cfg["voxel"]["through_thickness_axis"]
        cc = cfg["voxel"]["current_collector_face"]
        if face == "collector":
            return axis, cc
        return axis, ("max" if cc == "min" else "min")
    axis, side = face.split("_")
    return axis, side
