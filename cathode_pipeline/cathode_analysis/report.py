"""Stage ``report`` (global): collect per-sample results into one summary
table and the comparison figures across LCO50 / LCO60 / LCO70.

Missing per-sample files are tolerated (NaN in the table), so the report can
be produced after any subset of stages.

Outputs (in paths.output_root)
    summary.csv        one row per sample (wide)
    summary_long.csv   sample, metric, value
    summary.json
    figures/volume_fractions, connectivity, interface_area, tortuosity, grain_size, profiles_<quantity>
"""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import PHASES
from . import io as cio
from .pipeline import StageContext

NAME = "report"
SCOPE = "global"
DEPENDS_ON: tuple[str, ...] = ()


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [ctx.path("summary.csv"), ctx.path("summary.json")]


def _load(path: Path) -> dict[str, Any]:
    return cio.load_json(path) if path.exists() else {}


def collect_sample(sdir: Path) -> dict[str, Any]:
    """Flatten a sample's JSON results into metric -> value."""
    m: dict[str, Any] = {}
    seg = _load(sdir / "segment.json")
    if seg:
        m["shape_z"], m["shape_y"], m["shape_x"] = seg["shape_zyx"]
        m["voxel_size_um"] = seg["voxel_size_um"]
        m["nominal_lco_vol_fraction"] = seg.get("nominal_lco_vol_fraction")
        for p in PHASES:
            m[f"vf_{p}"] = seg["volume_fractions"][p]
        th = seg.get("thresholds_uint8") or [None, None]
        m["threshold_1"], m["threshold_2"] = th[0], th[1]
        m["segmentation_mode"] = seg.get("mode", "grayscale")
    conn = _load(sdir / "connectivity.json")
    for p, st in conn.get("phases", {}).items():
        m[f"connected_fraction_{p}"] = st.get("connected_fraction")
        m[f"vf_connected_{p}"] = st.get("volume_fraction_connected")
        m[f"percolates_{p}"] = st.get("percolates_collector_to_separator")
        m[f"n_clusters_{p}"] = st.get("n_clusters")
    surf = _load(sdir / "surface.json")
    for pair, st in surf.get("interfaces", {}).items():
        key = pair.replace("-", "_")
        m[f"area_{key}_um2"] = st["area_um2"]
        m[f"specific_area_{key}_1_um"] = st["specific_area_per_total_volume_1_um"]
        m[f"active_fraction_{key}"] = st.get("active_fraction")
        m[f"active_area_{key}_um2"] = st.get("active_area_um2")
        m[f"active_specific_area_{key}_1_um"] = st.get("active_specific_area_per_total_volume_1_um")
        for k, v in st.items():
            if k.startswith("area_per_") or k.startswith("active_area_per_"):
                m[f"{key}_{k}"] = v
    tor = _load(sdir / "tortuosity.json")
    for p, st in tor.get("phases", {}).items():
        for d, e in st["directions"].items():
            m[f"tau_{p}_{d}"] = e.get("tau")
            m[f"D_rel_{p}_{d}"] = e.get("D_rel")
            m[f"tau_converged_{p}_{d}"] = e.get("converged")
    gr = _load(sdir / "grains.json")
    for p, st in gr.get("phases", {}).items():
        m[f"n_grains_{p}"] = st["n_grains"]
        m[f"mean_sphericity_{p}"] = st.get("mean_sphericity")
        for k, v in st.get("size_distribution", {}).items():
            if k.startswith("d") and not k.startswith("d_") and isinstance(v, (int, float)):
                m[f"{k}_{p}"] = v
            if k in ("d_mean_um", "d_median_um"):
                m[f"{k}_{p}"] = v
        for k, v in st.items():
            if k.startswith("mean_contact_"):
                m[f"{k}_{p}"] = v
    return m


def run(ctx: StageContext) -> dict[str, Any]:
    cfg, log = ctx.cfg, ctx.log
    fig_cfg = cfg["report"]["figures"]
    fmt, dpi = fig_cfg["format"], int(fig_cfg["dpi"])
    colors = cfg["plotting"]["colors"]
    sample_dirs = ctx.sample_output_dirs
    names = ctx.extra.get("samples") or list(sample_dirs)
    nominal = {s["name"]: s.get("nominal_lco_vol_fraction") for s in cfg["samples"]}

    rows = {}
    for name in names:
        m = collect_sample(sample_dirs[name])
        if not m:
            log.warning("no results found for %s in %s", name, sample_dirs[name])
        m.setdefault("nominal_lco_vol_fraction", nominal.get(name))
        rows[name] = m
    wide = pd.DataFrame.from_dict(rows, orient="index")
    wide.index.name = "sample"
    wide.to_csv(ctx.path("summary.csv"), float_format=cfg["report"]["tables"]["float_format"])
    long = wide.reset_index().melt(id_vars="sample", var_name="metric", value_name="value")
    long.to_csv(ctx.path("summary_long.csv"), index=False)
    cio.save_json({"samples": rows}, ctx.path("summary.json"))
    log.info("summary: %d samples x %d metrics", len(wide), wide.shape[1])

    from . import plotting

    fdir = ctx.path("figures")
    fdir.mkdir(parents=True, exist_ok=True)
    made = []

    def series(prefix: str, keys: list[str], suffix: str = "") -> dict[str, list[float]]:
        return {k: [rows[n].get(f"{prefix}{k}{suffix}", math.nan) for n in names] for k in keys}

    if any(f"vf_{p}" in wide.columns for p in PHASES):
        ref = {"lco": [nominal.get(n) if nominal.get(n) is not None else math.nan for n in names]}
        made.append(plotting.bar_by_sample(names, series("vf_", list(PHASES)), colors, "volume fraction",
                                           fdir / f"volume_fractions.{fmt}", "Volume fractions",
                                           reference=ref if cfg["report"]["compare_with_nominal"] else None, dpi=dpi))
    if any(c.startswith("connected_fraction_") for c in wide.columns):
        made.append(plotting.bar_by_sample(names, series("connected_fraction_", list(PHASES)), colors,
                                           "connected fraction of phase", fdir / f"connectivity.{fmt}",
                                           "Connectivity to reference faces", dpi=dpi))
    pairs = [c[len("specific_area_"):-len("_1_um")] for c in wide.columns if c.startswith("specific_area_")]
    if pairs:
        icolors = cfg["plotting"].get("interface_colors", {})
        pair_colors = {p: icolors.get(p.replace("_", "-"), "#7a7a7a") for p in pairs}
        made.append(plotting.bar_by_sample(names, series("specific_area_", pairs, "_1_um"), pair_colors,
                                           "interface area / volume (1/um)", fdir / f"interface_area.{fmt}",
                                           "Specific interface area", dpi=dpi))
        active = [p for p in pairs if f"active_specific_area_{p}_1_um" in wide.columns]
        if active:
            made.append(plotting.bar_by_sample(names, series("active_specific_area_", active, "_1_um"), pair_colors,
                                               "active interface area / volume (1/um)",
                                               fdir / f"active_interface_area.{fmt}", "Active interface area", dpi=dpi))
    taus = [c[len("tau_"):] for c in wide.columns if re.fullmatch(r"tau_(pore|lco|llzto)_[xyz]", c)]
    if taus:
        tau_colors = {t: colors[t.split("_")[0]] for t in taus}
        vals = series("tau_", taus)
        vals = {k: [v if (v is not None and math.isfinite(v)) else math.nan for v in vs] for k, vs in vals.items()}
        made.append(plotting.bar_by_sample(names, vals, tau_colors, "tortuosity factor", fdir / f"tortuosity.{fmt}",
                                           "Tortuosity (inf = not percolating, shown as gap)", dpi=dpi))
    d50 = [c[len("d50_number_um_"):] for c in wide.columns if c.startswith("d50_number_um_")]
    if d50:
        made.append(plotting.bar_by_sample(names, series("d50_number_um_", d50), colors, "d50 (number) um",
                                           fdir / f"grain_size.{fmt}", "Grain size", dpi=dpi))

    # profiles across samples
    prof = {n: pd.read_csv(sample_dirs[n] / "profiles.csv") for n in names if (sample_dirs[n] / "profiles.csv").exists()}
    if prof:
        sample_colors = cfg["plotting"].get("sample_colors") or {}
        for q, phase in (("volume_fraction", "lco"), ("volume_fraction", "llzto"), ("volume_fraction", "pore"),
                         ("connected_fraction", "lco"), ("connected_fraction", "llzto"),
                         ("interface_density", "lco-llzto")):
            ser, pos = {}, None
            for n, df in prof.items():
                d = df[(df["quantity"] == q) & (df["phase"] == phase)]
                if len(d):
                    ser[n] = d["value"].to_numpy()
                    pos = d["position_norm"].to_numpy() if pos is None or len(d) > len(pos) else pos
            if ser:
                made.append(plotting.profile_lines(ser, {n: d[(d["quantity"] == q) & (d["phase"] == phase)]["position_norm"].to_numpy()
                                                          for n, d in prof.items() if n in ser},
                                                   sample_colors, f"{q} ({phase})", fdir / f"profile_{q}_{phase}.{fmt}",
                                                   title=f"{phase}: {q} through thickness", dpi=dpi))
    # figures in the style of Clausnitzer et al. 2023 (through-thickness direction only)
    from . import paper_figures

    axis = cfg["voxel"]["through_thickness_axis"]
    made += paper_figures.make_paper_figures(rows, names, fdir, axis, fmt, dpi,
                                             cfg["plotting"].get("paper_colors"))

    log.info("figures: %s", [Path(p).name for p in made])
    return {"n_samples": len(names), "n_metrics": int(wide.shape[1]), "figures": len(made)}
