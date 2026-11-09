"""Figures in the style of Clausnitzer et al., Batteries & Supercaps 2023, e202300167.

All quantities are taken from the per-sample metric rows collected by the
report stage and plotted against the solid volume fraction of LCO

    SVF_LCO = V_LCO / (V_LCO + V_LLZTO)            (their Eq. 3, porosity independent)

Tortuosity and effective conductivity use ONE direction only: the
through-thickness axis (``voxel.through_thickness_axis``), i.e. the direction
between current collector and separator.

Figures (written to <output_root>/figures/):
    clausnitzer_fig2_connectivity_tortuosity   connectivity of SE and CAM (triangles, left axis)
                                               and 1/tau (dashed lines, right axis) vs SVF_LCO
    clausnitzer_fig3_effective_conductivity    normalised effective ionic (SE) and electronic (CAM)
                                               conductivity sigma_eff/sigma = eps/tau vs phase fraction
    clausnitzer_fig7_tortuosity_active_area    tau of the SE (left axis) and active LCO-LLZTO area
                                               per cathode volume in 1/cm (right axis) vs SVF_LCO

Conventions: SE = LLZTO (blue), CAM = LCO (red), as in the paper. tau is the
tortuosity factor of taufactor, tau = eps * D0 / D_eff; a phase that does not
percolate has tau = inf and is drawn at 1/tau = 0.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .plotting import _save, _style, plt  # noqa: F401  (Agg backend is forced in plotting)

PAPER_COLORS = {"se": "#1f77b4", "cam": "#d62728"}
PHASE_OF = {"se": "llzto", "cam": "lco"}
LABEL = {"se": "SE (LLZTO)", "cam": "CAM (LCO)"}


def _val(row: Mapping[str, Any], key: str) -> float:
    v = row.get(key)
    if v is None or (isinstance(v, str)):
        return math.nan
    try:
        return float(v)
    except (TypeError, ValueError):
        return math.nan


def _finite(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=float)


def sample_table(rows: Mapping[str, Mapping[str, Any]], names: Sequence[str], axis: str) -> dict[str, np.ndarray]:
    """Per-sample arrays (in ``names`` order sorted by SVF_LCO) of everything the figures need."""
    vf_lco = np.array([_val(rows[n], "vf_lco") for n in names])
    vf_llzto = np.array([_val(rows[n], "vf_llzto") for n in names])
    vf_pore = np.array([_val(rows[n], "vf_pore") for n in names])
    svf = vf_lco / (vf_lco + vf_llzto) * 100.0
    order = np.argsort(svf)
    names = [names[i] for i in order]

    def col(key: str) -> np.ndarray:
        return np.array([_val(rows[n], key) for n in names])

    t = {
        "names": np.array(names),
        "svf_lco": svf[order],
        "density": (1.0 - vf_pore[order]) * 100.0,          # sinter density = 1 - porosity, in %
        "vf_lco_pct": vf_lco[order] * 100.0,
        "vf_llzto_pct": vf_llzto[order] * 100.0,
        "conn_se": col("connected_fraction_llzto") * 100.0,
        "conn_cam": col("connected_fraction_lco") * 100.0,
        "tau_se": col(f"tau_llzto_{axis}"),
        "tau_cam": col(f"tau_lco_{axis}"),
        "D_rel_se": col(f"D_rel_llzto_{axis}"),
        "D_rel_cam": col(f"D_rel_lco_{axis}"),
        # solver did not converge -> tau is a lower bound; drawn as open symbols
        "open_se": np.array([rows[n].get(f"tau_converged_llzto_{axis}") is False for n in names]),
        "open_cam": np.array([rows[n].get(f"tau_converged_lco_{axis}") is False for n in names]),
        # active LCO-LLZTO area per total cathode volume: 1/um -> 1/cm
        "a_act_per_vca_1_cm": col("active_specific_area_lco_llzto_1_um") * 1e4,
        "a_tot_per_vca_1_cm": col("specific_area_lco_llzto_1_um") * 1e4,
    }
    return t


def _inv_tau(tau: np.ndarray) -> np.ndarray:
    out = np.full(tau.shape, math.nan)
    for i, v in enumerate(tau):
        if v is None or math.isnan(v):
            continue
        out[i] = 0.0 if math.isinf(v) or v <= 0 else 1.0 / v
    return out


def _annotate(ax: plt.Axes, x: np.ndarray, y: np.ndarray, names: np.ndarray) -> None:
    for xi, yi, n in zip(x, y, names):
        if np.isfinite(xi) and np.isfinite(yi):
            ax.annotate(str(n), (xi, yi), textcoords="offset points", xytext=(4, 4), fontsize=7, color="#444444")


def _has(v: np.ndarray) -> bool:
    return bool(np.isfinite(v).any())


def _line(ax: plt.Axes, x: np.ndarray, y: np.ndarray, open_mask: np.ndarray, color: str, label: str,
          marker: str = "o", linestyle: str = "--") -> None:
    """Line with filled markers; points flagged in ``open_mask`` get open (white-filled) markers."""
    ax.plot(x, y, marker=marker, linestyle=linestyle, color=color, label=label)
    if open_mask.any():
        ax.plot(x[open_mask], y[open_mask], marker=marker, linestyle="none", color=color, markerfacecolor="white",
                markeredgewidth=1.5)


def _open_note(ax: plt.Axes, t: Mapping[str, np.ndarray]) -> None:
    if t["open_se"].any() or t["open_cam"].any():
        ax.text(0.99, 0.01, "open symbols: solver not converged (tau = lower bound)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=7, color="#666666")


def fig2_connectivity_tortuosity(t: Mapping[str, np.ndarray], path: Path, axis: str, colors: Mapping[str, str],
                                 dpi: int = 200) -> Path | None:
    if not (_has(t["conn_se"]) or _has(t["conn_cam"])):
        return None
    x = t["svf_lco"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(x, t["conn_se"], marker="^", linestyle="none", markersize=8, color=colors["se"], label="Connectivity SE")
    ax.plot(x, t["conn_cam"], marker="^", linestyle="none", markersize=8, color=colors["cam"], label="Connectivity CAM")
    _annotate(ax, x, t["conn_se"], t["names"])
    ax.set_xlabel(r"SVF$_{LCO}$ / %")
    ax.set_ylabel("Connectivity / %")
    ax.set_ylim(0, 105)
    _style(ax)
    ax2 = ax.twinx()
    inv_se, inv_cam = _inv_tau(t["tau_se"]), _inv_tau(t["tau_cam"])
    if _has(inv_se) or _has(inv_cam):
        _line(ax2, x, inv_se, t["open_se"], colors["se"], "1/Tortuosity SE")
        _line(ax2, x, inv_cam, t["open_cam"], colors["cam"], "1/Tortuosity CAM")
        ax2.set_ylabel(f"1/Tortuosity ({axis} direction) / -")
        ax2.set_ylim(0, 1.05)
        _open_note(ax2, t)
    else:
        ax2.set_yticks([])
        ax2.set_ylabel("1/Tortuosity (tortuosity stage not run)")
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=8, frameon=True, framealpha=0.9)
    ax.set_title("Connectivity and tortuosity vs. LCO solid volume fraction (cf. Clausnitzer 2023, Fig. 2)",
                 fontsize=9)
    return _save(fig, path, dpi)


def fig3_effective_conductivity(t: Mapping[str, np.ndarray], path: Path, axis: str, colors: Mapping[str, str],
                                dpi: int = 200) -> Path | None:
    if not (_has(t["D_rel_se"]) or _has(t["D_rel_cam"])):
        return None
    fig, axes = plt.subplots(2, 1, figsize=(5.2, 6.4))
    for ax, key, xkey, lab, xlabel, c in (
        (axes[0], "D_rel_se", "vf_llzto_pct", "Ionic", "SE fraction / vol%", colors["se"]),
        (axes[1], "D_rel_cam", "vf_lco_pct", "Electronic", "CAM fraction / vol%", colors["cam"]),
    ):
        y = np.where(np.isfinite(t[key]) & (t[key] > 0), t[key], np.nan)
        ax.plot(t[xkey], y, marker="o", linestyle="none", markersize=7, color=c, label="FIB-SEM reconstruction")
        _annotate(ax, t[xkey], y, t["names"])
        ax.set_yscale("log")
        ax.set_xlim(0, 100)
        ax.set_ylim(1e-3, 1.2)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Norm. conductivity / -")
        ax.text(0.03, 0.92, lab, transform=ax.transAxes, fontsize=11, color="#777777")
        ax.legend(loc="lower right", fontsize=8, frameon=True)
        _style(ax)
    fig.suptitle(f"sigma_eff / sigma = eps / tau ({axis} direction)  (cf. Clausnitzer 2023, Fig. 3)", fontsize=9)
    return _save(fig, path, dpi)


def fig7_tortuosity_active_area(t: Mapping[str, np.ndarray], path: Path, axis: str, colors: Mapping[str, str],
                                dpi: int = 200) -> Path | None:
    if not (_has(t["tau_se"]) or _has(t["a_act_per_vca_1_cm"])):
        return None
    x = t["svf_lco"]
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    tau = np.where(np.isfinite(t["tau_se"]), t["tau_se"], np.nan)
    _line(ax, x, tau, t["open_se"], colors["se"], f"Tortuosity SE ({axis})")
    _annotate(ax, x, tau, t["names"])
    _open_note(ax, t)
    ax.set_xlabel(r"SVF$_{LCO}$ / %")
    ax.set_ylabel(r"Tortuosity $\tau$ / -")
    if _has(tau) and np.nanmax(tau) / max(np.nanmin(tau), 1e-9) > 20:
        ax.set_yscale("log")                                  # a barely percolating phase would squash the others
    else:
        ax.set_ylim(bottom=1.0 if np.nanmin(tau) >= 1.0 else 0.0)
    _style(ax)
    ax2 = ax.twinx()
    ax2.plot(x, t["a_act_per_vca_1_cm"], marker="^", linestyle="-", color=colors["cam"],
             label=r"$A_{act}/V_{Ca}$ (LCO-LLZTO, both sides connected)")
    ax2.plot(x, t["a_tot_per_vca_1_cm"], marker="v", linestyle=":", color=colors["cam"], alpha=0.6,
             label=r"$A_{tot}/V_{Ca}$ (LCO-LLZTO, all)")
    ax2.set_ylabel(r"$A_{act}/V_{Ca}$ / 1/cm")
    ax2.set_ylim(bottom=0)
    ax2.spines["top"].set_visible(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=8, frameon=False)
    dens = ", ".join(f"{n}: {d:.1f} %" for n, d in zip(t["names"], t["density"]) if np.isfinite(d))
    ax.set_title(f"SE tortuosity and active interface area (cf. Clausnitzer 2023, Fig. 7)\nsinter density = 1 - porosity: {dens}",
                 fontsize=8)
    return _save(fig, path, dpi)


def make_paper_figures(rows: Mapping[str, Mapping[str, Any]], names: Sequence[str], out_dir: Path, axis: str,
                       fmt: str = "png", dpi: int = 200, colors: Mapping[str, str] | None = None) -> list[Path]:
    """Write the three figures; returns the paths that were produced."""
    names = [n for n in names if rows.get(n)]
    if not names:
        return []
    c = dict(PAPER_COLORS, **(colors or {}))
    t = sample_table(rows, names, axis)
    out_dir = Path(out_dir)
    made = []
    for fn, name in ((fig2_connectivity_tortuosity, "clausnitzer_fig2_connectivity_tortuosity"),
                     (fig3_effective_conductivity, "clausnitzer_fig3_effective_conductivity"),
                     (fig7_tortuosity_active_area, "clausnitzer_fig7_tortuosity_active_area")):
        p = fn(t, out_dir / f"{name}.{fmt}", axis, c, dpi)
        if p is not None:
            made.append(p)
    return made
