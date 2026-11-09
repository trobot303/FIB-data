"""Headless matplotlib helpers shared by the stages (backend forced to Agg).

Conventions: phase colours are fixed identities from ``plotting.colors`` and
never re-assigned; one y-axis per plot; thin marks; recessive grid.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import ListedColormap, to_rgb  # noqa: E402

from . import PHASE_CODE, PHASES  # noqa: E402

PHASE_LABEL = {"pore": "Pore", "lco": "LCO", "llzto": "LLZTO"}


def _style(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#d0d0d0", linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, path: str | Path, dpi: int = 150) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return path


def phase_cmap(colors: Mapping[str, str], n: int = 6) -> ListedColormap:
    """Colormap for codes 0..n-1; isolated codes (3-5) get a lighter tint."""
    cols = []
    for code in range(n):
        phase = PHASES[code % 3]
        rgb = np.array(to_rgb(colors[phase]))
        if code >= 3:
            rgb = 0.5 * rgb + 0.5
        cols.append(rgb)
    return ListedColormap(cols)


def overlay_slice(gray: np.ndarray, labels: np.ndarray, colors: Mapping[str, str], alpha: float,
                  path: str | Path, title: str = "", dpi: int = 120) -> Path:
    h, w = gray.shape
    fig, axes = plt.subplots(1, 2, figsize=(min(16, 2 * 7 * w / h + 1), 7 * 1.0))
    axes[0].imshow(gray, cmap="gray", interpolation="nearest")
    axes[0].set_title(f"{title} grey")
    axes[1].imshow(gray, cmap="gray", interpolation="nearest")
    axes[1].imshow(labels, cmap=phase_cmap(colors), vmin=0, vmax=5, alpha=alpha, interpolation="nearest")
    axes[1].set_title("phases")
    for ax in axes:
        ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in PHASES]
    axes[1].legend(handles, [PHASE_LABEL[p] for p in PHASES], loc="lower right", fontsize=8, framealpha=0.9)
    return _save(fig, path, dpi)


def label_slice(labels: np.ndarray, colors: Mapping[str, str], path: str | Path, title: str = "",
                dpi: int = 120) -> Path:
    """Phase map of one label slice (used when no grey image exists, e.g. imported masks)."""
    h, w = labels.shape
    fig, ax = plt.subplots(figsize=(min(12, 7 * w / h + 1), 7))
    ax.imshow(labels, cmap=phase_cmap(colors), vmin=0, vmax=5, interpolation="nearest")
    ax.set_title(f"{title} phases")
    ax.set_axis_off()
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[p]) for p in PHASES]
    ax.legend(handles, [PHASE_LABEL[p] for p in PHASES], loc="lower right", fontsize=8, framealpha=0.9)
    return _save(fig, path, dpi)


def histogram_with_thresholds(hist: np.ndarray, thresholds: Sequence[float], order: Sequence[str],
                              colors: Mapping[str, str], path: str | Path, title: str = "") -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(256)
    edges = [0, *thresholds, 256]
    for i, phase in enumerate(order):
        sel = (x >= edges[i]) & (x < edges[i + 1])
        ax.bar(x[sel], hist[sel], width=1.0, color=colors[phase], label=PHASE_LABEL[phase])
    for t in thresholds:
        ax.axvline(t, color="#333333", linewidth=1, linestyle="--")
        ax.text(t, ax.get_ylim()[1] * 0.95, f"{t:.0f}", ha="center", va="top", fontsize=8)
    ax.set_xlabel("grey level (uint8)")
    ax.set_ylabel("voxels")
    ax.set_yscale("log")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return _save(fig, path)


def bar_by_sample(samples: Sequence[str], series: Mapping[str, Sequence[float]], colors: Mapping[str, str],
                  ylabel: str, path: str | Path, title: str = "", reference: Mapping[str, Sequence[float]] | None = None,
                  fmt: str = "png", dpi: int = 200) -> Path:
    """Grouped bars: one group per sample, one bar per series key (a phase or metric)."""
    fig, ax = plt.subplots(figsize=(1.6 * len(samples) + 3, 3.8))
    keys = list(series)
    n = len(keys)
    width = 0.8 / max(n, 1)
    xs = np.arange(len(samples))
    for j, key in enumerate(keys):
        vals = np.asarray(series[key], dtype=float)
        pos = xs - 0.4 + width * (j + 0.5)
        ax.bar(pos, vals, width=width * 0.92, color=colors.get(key, "#7a7a7a"), label=PHASE_LABEL.get(key, key))
        for p, v in zip(pos, vals):
            if np.isfinite(v):
                ax.text(p, v, f"{v:.3g}", ha="center", va="bottom", fontsize=7, color="#333333")
        if reference and key in reference:
            ax.scatter(pos, reference[key], marker="_", s=300, color="#111111", zorder=3,
                       label="nominal" if j == 0 else None)
    ax.set_xticks(xs, samples)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if n > 1 or reference:
        ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return _save(fig, path, dpi)


def profile_plot(position: np.ndarray, series: Mapping[str, np.ndarray], colors: Mapping[str, str],
                 ylabel: str, path: str | Path, title: str = "", xlabel: str = "distance from current collector (um)",
                 dpi: int = 200) -> Path:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    for key, vals in series.items():
        ax.plot(position, vals, color=colors.get(key, "#7a7a7a"), linewidth=1.8, label=PHASE_LABEL.get(key, key))
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if len(series) > 1:
        ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return _save(fig, path, dpi)


def size_distribution(bin_edges: np.ndarray, number_frac: np.ndarray, volume_frac: np.ndarray, color: str,
                      path: str | Path, title: str = "", dpi: int = 200) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    w = np.diff(bin_edges)
    for ax, y, lab in ((axes[0], number_frac, "number fraction"), (axes[1], volume_frac, "volume fraction")):
        ax.bar(centers, y, width=w * 0.92, color=color)
        ax.set_xlabel("equivalent diameter (um)")
        ax.set_ylabel(lab)
        _style(ax)
    fig.suptitle(title)
    return _save(fig, path, dpi)


def profile_panels(df, quantities: Sequence[str], colors: Mapping[str, str], path: str | Path, title: str = "",
                   dpi: int = 200) -> Path:
    """One panel per quantity; lines per phase (tidy profiles.csv as DataFrame)."""
    n = len(quantities)
    fig, axes = plt.subplots(n, 1, figsize=(7, 2.6 * n), sharex=True, squeeze=False)
    for ax, q in zip(axes[:, 0], quantities):
        d = df[df["quantity"] == q]
        for phase, g in d.groupby("phase", sort=False):
            ax.plot(g["position_um"], g["value"], color=colors.get(phase, "#7a7a7a"), linewidth=1.8,
                    label=PHASE_LABEL.get(phase, phase))
        ax.set_ylabel(q.replace("_", " "))
        if d["phase"].nunique() > 1:
            ax.legend(frameon=False, fontsize=8, ncol=3)
        _style(ax)
    axes[-1, 0].set_xlabel("distance from current collector (um)")
    fig.suptitle(title)
    return _save(fig, path, dpi)


def profile_lines(series: Mapping[str, np.ndarray], positions: Mapping[str, np.ndarray], colors: Mapping[str, str],
                  ylabel: str, path: str | Path, title: str = "", dpi: int = 200) -> Path:
    """One line per sample against normalised depth (0 = collector, 1 = separator)."""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    fallback = ["#4477aa", "#ee6677", "#228833", "#ccbb44", "#66ccee", "#aa3377"]
    for i, (name, vals) in enumerate(series.items()):
        ax.plot(positions[name], vals, color=colors.get(name, fallback[i % len(fallback)]), linewidth=1.8, label=name)
    ax.set_xlabel("normalised depth (0 = current collector, 1 = separator)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    _style(ax)
    return _save(fig, path, dpi)
