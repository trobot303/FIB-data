"""Minimal interactive scribble tool (matplotlib, runs on your own machine).

    python -m fib_rf_seg.label --image data/LCO60_Processed001.tiff \
        --out data/labels/LCO60_Processed001_scribbles.json [--model models/rf.joblib]

Keys:  1 = pore, 2 = LCO, 3 = LLZTO (choose the phase to paint)
       [ / ]  smaller / larger brush      u = undo last stroke
       o = toggle model overlay (if --model given)      s = save      q = quit
Drag with the left mouse button to paint.  Existing strokes in --out are loaded
and kept, so you can add corrections over several sessions.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from . import PHASES, PHASE_COLORS
from .io import normalise, read_slice
from .scribbles import rasterise
from .viz import CMAP, overlay


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", help="show the current model's prediction as overlay")
    ap.add_argument("--region", type=int, nargs=4, metavar=("Y0", "Y1", "X0", "X1"))
    a = ap.parse_args(argv)

    im = normalise(read_slice(a.image))
    out = Path(a.out)
    data = json.load(open(out)) if out.exists() else {"brush": 3, "strokes": []}
    state = {"phase": 1, "brush": int(data.get("brush", 3)), "cur": None, "show_pred": True}

    pred = None
    if a.model:
        from .model import load_model, predict
        rf, sigmas, _ = load_model(a.model)
        pred, _ = predict(rf, im, sigmas)

    fig, ax = plt.subplots(figsize=(16, 16 * im.shape[0] / im.shape[1]))
    base = ax.imshow(overlay(im, pred, 0.35) if pred is not None else im, cmap="gray", vmin=0, vmax=1)
    scr_art = ax.imshow(np.ma.masked_equal(rasterise(data, im.shape), 0), cmap=CMAP, vmin=0, vmax=3, interpolation="nearest")
    if a.region:
        y0, y1, x0, x1 = a.region
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0)
    (cur_line,) = ax.plot([], [], "-", lw=2, color=PHASE_COLORS[1])

    def title():
        ax.set_title(f"phase {state['phase']} ({PHASES[state['phase']]})   brush {state['brush']}   strokes {len(data['strokes'])}   [s]ave [u]ndo [o]verlay [q]uit")
        fig.canvas.draw_idle()

    def refresh():
        scr_art.set_data(np.ma.masked_equal(rasterise(data, im.shape), 0))
        title()

    def on_press(ev):
        if ev.inaxes != ax or ev.button != 1:
            return
        state["cur"] = [[int(round(ev.ydata)), int(round(ev.xdata))]]
        cur_line.set_color(PHASE_COLORS[state["phase"]])

    def on_move(ev):
        if state["cur"] is None or ev.inaxes != ax:
            return
        state["cur"].append([int(round(ev.ydata)), int(round(ev.xdata))])
        pts = np.array(state["cur"]); cur_line.set_data(pts[:, 1], pts[:, 0]); fig.canvas.draw_idle()

    def on_release(ev):
        if state["cur"] is None:
            return
        data["strokes"].append({"phase": state["phase"], "brush": state["brush"], "points": state["cur"]})
        state["cur"] = None; cur_line.set_data([], []); refresh()

    def on_key(ev):
        k = ev.key
        if k in ("1", "2", "3"):
            state["phase"] = int(k); title()
        elif k == "[":
            state["brush"] = max(1, state["brush"] - 1); title()
        elif k == "]":
            state["brush"] += 1; title()
        elif k == "u" and data["strokes"]:
            data["strokes"].pop(); refresh()
        elif k == "o" and pred is not None:
            state["show_pred"] = not state["show_pred"]
            base.set_data(overlay(im, pred, 0.35) if state["show_pred"] else im); fig.canvas.draw_idle()
        elif k == "s":
            out.parent.mkdir(parents=True, exist_ok=True)
            json.dump(data, open(out, "w"), indent=1)
            print(f"saved {len(data['strokes'])} strokes to {out}")
        elif k == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("key_press_event", on_key)
    title()
    plt.show()


if __name__ == "__main__":
    main()
