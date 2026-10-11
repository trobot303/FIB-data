# FIB-data: random-forest segmentation of FIB-SEM cathode slices

`fib_rf_seg` is a small pixel-classification package (scikit-learn random forest,
ilastik-style multi-scale features) that labels each pixel of a FIB-SEM slice as

| code | phase | appearance |
|-----:|-------|------------|
| 1 | pore  | black (pore back walls shining through are still pore) |
| 2 | LCO   | grey |
| 3 | LLZTO | almost white |

The codes match the `cathode_analysis` convention described in
`docs/cathode_pipeline_methods.docx`, so the label TIFFs can feed that pipeline's
pre-segmented mode.

## Install

```bash
pip install -r requirements.txt
```

## Workflow

1. **Train** on one or more slices.  Labels come from any mix of
   automatic high-confidence seeds (`--auto-seeds`, multi-Otsu cores away from
   edges), hand scribbles (`--scribbles file.json`, see below) and full label
   images (`--labels file.tif`, 0 = unlabelled).  `--dense` propagates the
   seeds/scribbles to all pixels with a marker watershed so that boundary
   pixels are also learnt (this removes the thin LCO halo around LLZTO grains
   that otherwise appears).

   ```bash
   python -m fib_rf_seg train --image data/LCO60_Processed001.tiff \
       --scribbles data/labels/LCO60_Processed001_scribbles.json \
       --auto-seeds --dense --max-per-class 40000 --model models/rf_v2.joblib
   ```

2. **Review** the prediction (grey, overlay, labels, confidence map):

   ```bash
   python -m fib_rf_seg review --model models/rf_v2.joblib --image data/LCO60_Processed001.tiff \
       --min-size 20 --out docs/review_v2_full.png [--region Y0 Y1 X0 X1]
   ```

3. **Correct**: paint scribbles where the model is wrong and retrain.
   Either draw a label image in ImageJ/napari (values 1/2/3, 0 elsewhere) or use
   the built-in tool on your own machine:

   ```bash
   python -m fib_rf_seg.label --image data/LCO60_Processed001.tiff \
       --out data/labels/LCO60_Processed001_scribbles.json --model models/rf_v2.joblib
   ```

   Keys `1`/`2`/`3` choose the phase, drag to paint, `[`/`]` change the brush,
   `u` undoes, `o` toggles the model overlay, `s` saves, `q` quits.  Scribbles
   override auto-seeds wherever they overlap, so a few strokes on a mistake are
   enough.  Commit the JSON and retrain (step 1).

4. **Predict** a whole stack (a directory of slices or one file):

   ```bash
   python -m fib_rf_seg predict --model models/rf_v2.joblib --input /path/to/slices \
       --out results/stack --min-size 20 [--save-proba]
   ```

   Output: `<slice>_labels.tif` per slice plus `fractions.json`.  Each slice is
   normalised with its own 0.1/99.9 percentiles by default (robust against
   brightness drift along z); pass `--lo/--hi` for fixed grey bounds.

## Scribble file format

```json
{"brush": 2,
 "strokes": [{"phase": 1, "points": [[y, x], [y, x]]},
             {"phase": 3, "brush": 4, "points": [[y, x]]}]}
```

Points are `[row, col]` pixel coordinates; a stroke is a polyline painted with a
round brush.

## Features

Per pixel, at sigmas 0.7, 1.5, 3, 6 and 12 px: Gaussian intensity, gradient
magnitude, Laplacian of Gaussian, both Hessian eigenvalues, local standard
deviation, plus differences of Gaussians between neighbouring scales and the
raw intensity (35 features).  The large scales give the context needed to call
the dark-grey pore back walls "pore" rather than LCO.

## Current state (slice LCO60_Processed001)

Model `rf_v2` (auto-seeds + 8 scribbles, dense watershed labels, 200 trees):
pore 36.1 %, LCO 40.3 %, LLZTO 23.6 %.  See `docs/overlay_v2_full.png` and
`docs/review_v2_full.png`.  Known issue: bright charging artefacts at the top
corners of the slice are labelled LLZTO; crop them away.

Trained models are not committed (30 MB); the train command above reproduces
them deterministically (`random_state=0`).
