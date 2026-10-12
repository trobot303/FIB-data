"""Overlay and review figures."""
from __future__ import annotations

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from . import PHASES, PHASE_COLORS

CMAP = ListedColormap([(0, 0, 0, 0)] + [PHASE_COLORS[k] for k in (1, 2, 3)])


def overlay(im_norm, lab, alpha=0.45):
    rgb = np.repeat(im_norm[..., None], 3, axis=-1).astype(np.float32)
    for k, c in PHASE_COLORS.items():
        m = lab == k
        rgb[m] = (1 - alpha) * rgb[m] + alpha * np.asarray(c, dtype=np.float32)
    return np.clip(rgb, 0, 1)


def add_scale_bar(ax, shape, px_um=0.05, length_um=10.0, color="white", loc="lower right", pad_frac=0.03,
                  height_px=None, background="0.35", fontsize=12):
    """Draw a scale bar of ``length_um`` with its label on a grey backing box (image in pixel coordinates)."""
    from matplotlib.patches import Rectangle
    h, w = shape[:2]
    L = length_um / px_um
    bh = height_px or max(3, int(0.008 * h))
    text_h = 2.6 * bh
    box_w, box_h = L + 2.0 * bh, bh + text_h + 2.0 * bh
    x1 = w - pad_frac * w if "right" in loc else pad_frac * w + box_w
    y1 = h - pad_frac * h if "lower" in loc else pad_frac * h + box_h
    x0, y0 = x1 - box_w, y1 - box_h
    ax.add_patch(Rectangle((x0, y0), box_w, box_h, color=background, alpha=0.85, zorder=10, lw=0))
    ax.add_patch(Rectangle((x0 + bh, y1 - bh - bh), L, bh, color=color, zorder=11, lw=0))
    ax.text(x0 + bh + L / 2, y1 - 2.4 * bh, f"{length_um:g} \u00b5m", color=color, ha="center", va="bottom",
            fontsize=fontsize, fontweight="bold", zorder=12)


def review_figure(im_norm, lab, scribbles=None, proba=None, title="", path=None, region=None):
    """Side-by-side: grey image, overlay, label map, and (optionally) max-probability map."""
    if region is not None:
        y0, y1, x0, x1 = region
        sl = (slice(y0, y1), slice(x0, x1))
        im_norm, lab = im_norm[sl], lab[sl]
        scribbles = None if scribbles is None else scribbles[sl]
        proba = None if proba is None else proba[sl]
    ncol = 3 + (proba is not None)
    h, w = im_norm.shape
    fig, axes = plt.subplots(ncol, 1, figsize=(14, 14 * h / w * ncol * 0.9))
    axes = np.atleast_1d(axes)
    axes[0].imshow(im_norm, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("grey" + (" + scribbles" if scribbles is not None else ""))
    if scribbles is not None:
        axes[0].imshow(np.ma.masked_equal(scribbles, 0), cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
    axes[1].imshow(overlay(im_norm, lab)); axes[1].set_title("overlay  (blue=pore, green=LCO, red=LLZTO)")
    axes[2].imshow(lab, cmap=CMAP, vmin=0, vmax=3, interpolation="nearest"); axes[2].set_title("labels")
    if proba is not None:
        conf = proba.max(axis=-1)
        a = axes[3].imshow(conf, cmap="magma", vmin=0.34, vmax=1); axes[3].set_title("classifier confidence (max probability)")
        fig.colorbar(a, ax=axes[3], fraction=0.02)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=110)
        plt.close(fig)
    return fig
