"""Stage ``tortuosity``: geometric tortuosity factor per phase and direction.

Backends
    taufactor  - GPU/CPU finite-difference solver (torch); default
    porespy    - porespy.simulations.tortuosity_fd (CPU, scipy sparse solver)

The solve uses the connected cluster only when ``tortuosity.use_connected_only``
is true and labels_connected exists. Results per (phase, direction): tau, the
volume fraction of the solved mask, D_eff/D_0 = vf / tau, and solver status.
A phase that does not span the volume in a direction gets tau = inf.

Downsampling (``tortuosity.downsample = k``) block-averages the phase mask and
thresholds the block means at the quantile that keeps the phase's volume
fraction unchanged (a plain majority vote erodes thin necks of a minority
phase, raises tau and can even break percolation).

taufactor is run in chunks of ``taufactor.check_every`` iterations so that
progress (iteration, current tau, flux error) is logged; ``taufactor.max_seconds``
(optional) stops a solve after that wall time and reports the current tau with
``converged = false`` instead of running to ``max_iterations``.

Outputs
    tortuosity.json
"""

from __future__ import annotations

import contextlib
import io
import logging
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage as ndi

from . import PHASE_CODE
from . import io as cio
from .pipeline import StageContext

NAME = "tortuosity"
SCOPE = "sample"
DEPENDS_ON = ("segment", "connectivity")
LOG = logging.getLogger(__name__)

AXIS_INDEX = {"z": 0, "y": 1, "x": 2}


def expected_outputs(ctx: StageContext) -> list[Path]:
    return [ctx.path("tortuosity.json")]


def resolve_device(pref: str) -> str:
    if pref == "cpu":
        return "cpu"
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if pref == "cuda":
            LOG.warning("device=cuda requested but CUDA is not available; using cpu")
    except ImportError:
        pass
    return "cpu"


def block_reduce_mask(mask: np.ndarray, k: int, preserve_vf: bool = True) -> np.ndarray:
    """Downsample a boolean mask by k x k x k blocks.

    ``preserve_vf=True`` (default): threshold the block occupancy at the level
    that reproduces the volume fraction of the full-resolution mask (ties are
    resolved so that the fraction is matched as closely as possible from
    above). ``preserve_vf=False``: plain majority vote (> half of the block).
    """
    if k == 1:
        return mask
    nz, ny, nx = (s // k * k for s in mask.shape)
    m = mask[:nz, :ny, :nx]
    target = float(m.mean())
    counts = m.reshape(nz // k, k, ny // k, k, nx // k, k).sum(axis=(1, 3, 5), dtype=np.int32)
    if not preserve_vf:
        return counts * 2 > k ** 3
    # counts take values 0..k^3: pick the threshold t whose fraction of blocks with
    # count >= t is closest to the target volume fraction
    hist = np.bincount(counts.ravel(), minlength=k ** 3 + 1)
    frac_ge = np.cumsum(hist[::-1])[::-1] / counts.size          # fraction of blocks with count >= t
    # smallest t (1..k^3) whose fraction does not exceed the target, then fill up with part of the tied blocks
    t = int(np.argmax(frac_ge[1:] <= target) + 1) if (frac_ge[1:] <= target).any() else k ** 3
    out = counts >= t
    n_missing = int(round(target * counts.size)) - int(out.sum())
    tied = np.flatnonzero(counts.ravel() == t - 1)
    if n_missing > 0 and tied.size:
        rng = np.random.default_rng(0)                              # reproducible tie-breaking
        pick = rng.choice(tied, size=min(n_missing, tied.size), replace=False)
        out.ravel()[pick] = True
    return out


def spans_axis(mask: np.ndarray) -> bool:
    """True if one connected component of ``mask`` touches both faces along axis 0."""
    lab, n = ndi.label(mask)
    if n == 0:
        return False
    a, b = np.unique(lab[0]), np.unique(lab[-1])
    return bool(np.intersect1d(a[a > 0], b[b > 0]).size)


def solve_taufactor(mask: np.ndarray, device: str, conv: float, iter_limit: int, check_every: int = 1000,
                    max_seconds: float | None = None, log: logging.Logger | None = None,
                    tag: str = "") -> dict[str, Any]:
    """Run taufactor in chunks of ``check_every`` iterations, logging progress.

    taufactor keeps its state between ``solve`` calls, so calling it with an
    increasing ``iter_limit`` continues the same solve. A ``max_seconds`` budget
    ends the solve early with the current tau and ``converged = False``.
    """
    import taufactor as tf

    log = log or LOG
    solver = tf.Solver(mask.astype(np.uint8), device=device)
    step = max(100, int(check_every) // 100 * 100)          # taufactor checks convergence every 100 iterations
    t0 = time.perf_counter()
    timed_out = False
    limit = 0
    while True:
        limit = min(iter_limit, limit + step)
        with contextlib.redirect_stdout(io.StringIO()):       # taufactor prints "not converged" after every chunk
            solver.solve(verbose=False, conv_crit=conv, iter_limit=limit)
        it = int(np.asarray(getattr(solver, "iter", 0)).ravel()[0] or 0)
        converged = bool(np.asarray(getattr(solver, "converged", False)).ravel()[0])
        elapsed = time.perf_counter() - t0
        tau_now = _f(getattr(solver, "tau", None))
        if converged or it >= iter_limit:
            break
        log.info("%s iter %d/%d  tau=%s  %.0f s (%.1f ms/iter)", tag, it, iter_limit,
                 f"{tau_now:.4f}" if tau_now is not None else "-", elapsed, 1000 * elapsed / max(it, 1))
        if max_seconds is not None and elapsed > float(max_seconds):
            timed_out = True
            log.warning("%s time budget of %.0f s exhausted after %d iterations; reporting the current tau "
                        "(converged=false)", tag, float(max_seconds), it)
            break
    if not converged and hasattr(solver, "compute_metrics"):
        try:
            solver.tau, _ = solver.compute_metrics()
        except Exception:  # pragma: no cover - metric refresh is best effort
            pass
    tau = _f(getattr(solver, "tau", None))
    return {"tau": tau if tau is not None else math.nan, "D_eff": _f(getattr(solver, "D_eff", None)),
            "D_rel_solver": _f(getattr(solver, "D_rel", None)), "vf_solver": _f(getattr(solver, "VF", None)),
            "iterations": it, "converged": converged, "timed_out": timed_out,
            "seconds": round(time.perf_counter() - t0, 1), "solver": "taufactor", "device": device}


def solve_porespy(mask: np.ndarray, tol: float) -> dict[str, Any]:
    import porespy as ps

    r = ps.simulations.tortuosity_fd(im=mask, axis=0)
    return {"tau": float(r.tortuosity), "D_eff": _f(getattr(r, "effective_diffusivity", None)),
           "converged": True, "solver": "porespy"}


def _f(v: Any) -> float | None:
    """float() for scalars, 0-d / 1-element arrays and torch tensors."""
    if v is None:
        return None
    try:
        if hasattr(v, "detach"):
            v = v.detach().cpu().numpy()
        return float(np.asarray(v, dtype=np.float64).ravel()[0])
    except (TypeError, ValueError, IndexError):
        return None


def run(ctx: StageContext) -> dict[str, Any]:
    cfg, log = ctx.cfg, ctx.log
    tor = cfg["tortuosity"]
    ext = cfg["runtime"]["label_format"]
    conn_path = ctx.path(f"labels_connected.{ext}")
    use_conn = bool(tor["use_connected_only"]) and conn_path.exists()
    labels = cio.load_labels(conn_path if use_conn else ctx.path(f"labels.{ext}"))
    device = resolve_device(cfg["runtime"]["device"]) if tor["backend"] == "taufactor" else "cpu"
    k = int(tor["downsample"])
    log.info("backend=%s device=%s connected_only=%s downsample=%d", tor["backend"], device, use_conn, k)

    results: dict[str, Any] = {"sample": ctx.sample, "backend": tor["backend"], "device": device,
                               "connected_only": use_conn, "downsample": k, "phases": {}}
    for phase in tor["phases"]:
        code = PHASE_CODE[phase]
        full = np.asarray(labels) == code
        vf_full = float(full.mean())
        mask = block_reduce_mask(full, k, preserve_vf=bool(tor.get("preserve_volume_fraction", True)))
        del full
        vf = float(mask.mean())
        if k > 1:
            log.info("%-5s downsample %d: volume fraction %.4f -> %.4f", phase, k, vf_full, vf)
        results["phases"][phase] = {"volume_fraction_solved": vf, "volume_fraction_full": vf_full, "directions": {}}
        for d in tor["directions"]:
            arr = np.ascontiguousarray(np.moveaxis(mask, AXIS_INDEX[d], 0))
            entry: dict[str, Any] = {"axis": d, "shape": list(arr.shape)}
            if not spans_axis(arr):
                entry.update({"tau": math.inf, "D_rel": 0.0, "converged": True, "note": "phase does not span this axis"})
            else:
                try:
                    if tor["backend"] == "taufactor":
                        t = tor["taufactor"]
                        ms = t.get("max_seconds")
                        entry.update(solve_taufactor(arr, device, float(t["convergence"]), int(t["max_iterations"]),
                                                     int(t.get("check_every") or 1000),
                                                     float(ms) if ms is not None else None, log,
                                                     tag=f"{phase}/{d}"))
                    else:
                        entry.update(solve_porespy(arr, float(tor["porespy"]["solver_tol"])))
                    tau = entry["tau"]
                    entry["D_rel"] = vf / tau if tau and math.isfinite(tau) and tau > 0 else 0.0
                except Exception as exc:  # solver failure must not kill the run
                    log.exception("tortuosity solve failed for %s/%s", phase, d)
                    entry.update({"tau": math.nan, "D_rel": math.nan, "converged": False, "error": str(exc)})
            results["phases"][phase]["directions"][d] = entry
            log.info("%-5s %s: tau=%.3f D_rel=%.4f converged=%s iterations=%s seconds=%s", phase, d, entry["tau"],
                     entry["D_rel"], entry.get("converged"), entry.get("iterations", "-"), entry.get("seconds", "-"))
        del mask

    cio.save_json(results, ctx.path("tortuosity.json"))
    return {p: {d: v["tau"] for d, v in r["directions"].items()} for p, r in results["phases"].items()}
