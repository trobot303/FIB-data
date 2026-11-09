"""Through-thickness (y) tortuosity factor and normalised effective conductivity of LCO and LLZTO
from a segmented stack, with and without cracks (taufactor finite-difference solver).

    python scripts/tortuosity_y.py <segmented_folder> [--downsample 2] [--collector top|bottom]
                                   [--device auto|cpu|cuda|mps] [--max-iter 20000] [--conv 0.02]

Reports per phase: tau (tortuosity factor, taufactor definition tau = eps * D0 / D_eff) and
eps/tau = sigma_eff/sigma.  LCO is solved twice: cracks as void and cracks filled back into
LCO, so the effect of cracking on the electronic path is the ratio of the two eps/tau values.
Only the part of each phase connected to its reference face is solved (as in the pipeline).
Requires: pip install taufactor torch
"""
import argparse
import io
import contextlib
import json
import time
from pathlib import Path

import numpy as np
import tifffile
from scipy import ndimage as ndi


def connected_part(mask, ref_face):
    lab, n = ndi.label(mask, structure=ndi.generate_binary_structure(3, 1))
    face = lab[:, 0, :] if ref_face == "top" else lab[:, -1, :]
    ids = np.unique(face); ids = ids[ids > 0]
    return np.isin(lab, ids)


def block_reduce(mask, k):
    """k x k x k block mean, re-binarised at the threshold that preserves the volume fraction."""
    if k == 1:
        return mask
    z, y, x = (s // k * k for s in mask.shape)
    m = mask[:z, :y, :x].reshape(z // k, k, y // k, k, x // k, k).mean(axis=(1, 3, 5))
    vf = mask.mean()
    thr = np.quantile(m, 1 - vf) if 0 < vf < 1 else 0.5
    return m > thr


def solve(mask_zyx, device, conv, max_iter, check_every=500):
    import taufactor as tf
    arr = np.ascontiguousarray(np.moveaxis(mask_zyx, 1, 0)).astype(np.uint8)   # solve along axis 0 = y
    s = tf.Solver(arr, device=device)
    t0 = time.time(); it = 0; converged = False
    while it < max_iter and not converged:
        with contextlib.redirect_stdout(io.StringIO()):
            s.solve(verbose=False, conv_crit=conv, iter_limit=min(max_iter, it + check_every))
        it = int(np.asarray(getattr(s, "iter", it)).ravel()[0]); converged = bool(np.asarray(getattr(s, "converged", False)).ravel()[0])
        tau = float(np.asarray(s.tau).ravel()[0]) if getattr(s, "tau", None) is not None else float("nan")
        print(f"    iter {it:6d}  tau={tau:.4f}  {time.time()-t0:6.0f} s", flush=True)
    tau = float(np.asarray(s.tau).ravel()[0])
    return {"tau": tau, "converged": converged, "iterations": it, "seconds": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder"); ap.add_argument("--downsample", type=int, default=2)
    ap.add_argument("--collector", choices=("top", "bottom"), default="top")
    ap.add_argument("--device", default="auto"); ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--conv", type=float, default=0.02); ap.add_argument("--out", default=None)
    ap.add_argument("--z", type=int, nargs=2, default=None, metavar=("Z0", "Z1"))
    ap.add_argument("--tag", default="", help="use <phase>_<tag>_stack.tif (e.g. 3d for the grown cracks)")
    ap.add_argument("--phases", nargs="+", default=["lco", "llzto"], choices=("lco", "llzto"), help="which phases to solve (lco = both crack cases)")
    a = ap.parse_args()
    import torch
    device = a.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    d = Path(a.folder)
    sfx = f"_{a.tag}_stack.tif" if a.tag else "_stack.tif"
    def load(n):
        p = d / f"{n}{sfx}"
        if not p.exists(): return None
        arr = tifffile.imread(str(p)); arr = arr[a.z[0]:a.z[1]] if a.z else arr
        return arr > 0
    lco, crack = load("LCO"), load("crack")
    llz = load("LLZTO") if "llzto" in a.phases else None
    sep = "bottom" if a.collector == "top" else "top"
    print(f"stack {lco.shape} (z, y, x), downsample {a.downsample}, device {device}, collector {a.collector}")
    cases = {}
    if "lco" in a.phases:
        cases["LCO_cracks_as_void"] = (lco, a.collector)
        if crack is not None:
            cases["LCO_cracks_filled"] = (lco | crack, a.collector)
    if "llzto" in a.phases:
        cases["LLZTO"] = (llz, sep)
    res = {"folder": str(d), "shape_zyx": list(lco.shape), "downsample": a.downsample, "device": device, "direction": "y", "phases": {}}
    for name, (mask, face) in cases.items():
        vf_full = float(mask.mean())
        m = block_reduce(connected_part(mask, face), a.downsample)
        vf = float(m.mean())
        print(f"{name}: vf {vf_full:.4f} (connected, downsampled {vf:.4f}); solving ...")
        if not (m[:, 0, :].any() and m[:, -1, :].any()):
            r = {"tau": float("inf"), "converged": True, "note": "does not span y"}
        else:
            r = solve(m, device, a.conv, a.max_iter)
        r.update({"volume_fraction": vf_full, "volume_fraction_solved": vf,
                  "eps_over_tau": (vf / r["tau"]) if np.isfinite(r["tau"]) and r["tau"] > 0 else 0.0})
        res["phases"][name] = r
        print(f"  -> tau={r['tau']:.3f}  eps/tau={r['eps_over_tau']:.4f}  converged={r['converged']}")
    if "LCO_cracks_filled" in res["phases"]:
        a_, b_ = res["phases"]["LCO_cracks_as_void"]["eps_over_tau"], res["phases"]["LCO_cracks_filled"]["eps_over_tau"]
        res["LCO_conductivity_ratio_cracked_over_uncracked"] = a_ / b_ if b_ else float("nan")
        print(f"LCO effective conductivity in y, cracked / uncracked = {res['LCO_conductivity_ratio_cracked_over_uncracked']:.3f}")
    out = Path(a.out) if a.out else d / "tortuosity_y.json"
    json.dump(res, open(out, "w"), indent=2); print("wrote", out)


if __name__ == "__main__":
    main()
