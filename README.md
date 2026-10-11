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
   python -m fib_rf_seg train \
       --image data/LCO60_Processed001.tiff --scribbles data/labels/LCO60_Processed001_scribbles.json \
       --image data/LCO60_Processed451.tiff \
       --image data/LCO60_Processed564.tiff \
       --image data/LCO60_Processed632.tiff \
       --auto-seeds --dense --pore-level auto --lo 2500 --hi 50000 --model models/rf_v4.joblib
   ```

   **Grey normalisation.** `--lo/--hi` map raw grey values to 0–1 for every
   slice with the same bounds (here the 0.1/99.9 percentiles of a clean slice
   of this stack; raw medians are pore ≈ 11000, LCO ≈ 23000, LLZTO ≈ 41000).
   The bounds are stored in the model and reused by `predict`/`review`.
   Without them each slice is scaled by its own percentiles, which the
   charging artefacts on slice 001 distort (its LLZTO landed at 0.71 instead
   of 0.87), inflating LLZTO grains by 1–2 px on the other slices.  Use
   per-slice scaling only for a stack whose brightness really drifts along z.

   **Pore edge position.** A watershed (and a pixel classifier trained on it)
   puts each pore edge at the midpoint of the grey transition, which is 0.28
   next to LCO but 0.40 next to LLZTO; that made pores 1–2 px too wide all
   round.  `--pore-level L` instead places every pore edge where the lightly
   smoothed grey value crosses `L` (0–1 normalised grey; `auto` = the
   multi-Otsu pore/solid threshold, 0.256 on this slice).  The classifier still
   decides *which* regions are pore (dark-grey pore back walls stay pore) and
   which solid phase lies on the other side (no LCO halo between pore and
   LLZTO); the level only moves the edge within a `--band` (default 3 px)
   around it.  Lower `L` shrinks pores, higher `L` grows them.  The same option
   on `predict`/`review` re-applies the rule after classification.
   `--solid-level L` does the same for the LCO/LLZTO edge (`auto` = multi-Otsu
   threshold, 0.65).  The watershed already puts that edge within half a pixel
   of 0.65, so it is off by default, but because LLZTO grains are small that
   half pixel is worth about 5 % of the image area (LLZTO 30.0 % → 24.9 % on
   slice 451 with `--solid-level auto`).

2. **Review** the prediction (grey, overlay, labels, confidence map):

   ```bash
   python -m fib_rf_seg review --model models/rf_v4.joblib --image data/LCO60_Processed001.tiff \
       --pore-level auto --min-size 20 --out docs/review_v4_001.png [--region Y0 Y1 X0 X1]
   ```

3. **Correct**: paint scribbles where the model is wrong and retrain.
   Either draw a label image in ImageJ/napari (values 1/2/3, 0 elsewhere) or use
   the built-in tool on your own machine:

   ```bash
   python -m fib_rf_seg.label --image data/LCO60_Processed001.tiff \
       --out data/labels/LCO60_Processed001_scribbles.json --model models/rf_v4.joblib
   ```

   Keys `1`/`2`/`3` choose the phase, drag to paint, `[`/`]` change the brush,
   `u` undoes, `o` toggles the model overlay, `s` saves, `q` quits.  Scribbles
   override auto-seeds wherever they overlap, so a few strokes on a mistake are
   enough.  Commit the JSON and retrain (step 1).

4. **Predict** a whole stack (a directory of slices or one file) and export
   each phase separately:

   ```bash
   python -m fib_rf_seg predict --model models/rf_lco60.joblib --input /path/to/slices \
       --out /path/to/output --pore-level auto --min-size 20 --per-phase --stack
   ```

   Output layout:

   | path | content |
   |------|---------|
   | `labels/<slice>_labels.tif` | 8-bit label image, 1 = pore, 2 = LCO, 3 = LLZTO |
   | `pore/<slice>_pore.tif`, `LCO/<slice>_LCO.tif`, `LLZTO/<slice>_LLZTO.tif` | one binary mask (0/255) per phase and slice (`--per-phase`; `--binary-value 1` for 0/1) |
   | `pore_stack.tif`, `LCO_stack.tif`, `LLZTO_stack.tif`, `labels_stack.tif` | the same as one multipage TIFF per phase (`--stack`) |
   | `fractions.csv`, `fractions.json` | area fraction of each phase per slice |

   The per-phase folders are what `cathode_analysis`' pre-segmented mode
   expects (one binary stack per phase, slices sorted by trailing number).
   The run resumes from existing `labels/` files if interrupted
   (`--overwrite` recomputes).  The grey bounds stored in the model are used;
   `--lo/--hi` override them, and a model trained without bounds scales each
   slice by its own 0.1/99.9 percentiles.  Speed is about 9 s per 1702×700
   slice on 4 cores.

## Running on the LCO60 stack (Windows)

`models/rf_lco60.joblib` is committed (100 trees, depth 14, 6 MB; it agrees
with the full-size `rf_v4` on 99.4 % of pixels).  In a terminal in the
repository folder:

```bat
pip install -r requirements.txt
python -m fib_rf_seg predict --model models\rf_lco60.joblib ^
    --input "C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\A970_6040\processed_LCO60" ^
    --out "C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\A970_6040\processed_LCO60_segmented" ^
    --pore-level auto --min-size 20 --per-phase --stack
```

or double-click `scripts\segment_lco60_stack.bat`, which runs the same
command.  The quotes around the paths are required because of the space in
`PlasmaFIB_A970 Data`.  Drop `--stack` if the stack is too large to hold in
memory as four uint8 volumes (about 1.2 GB per 1000 slices each).

## Scribble file format

```json
{"brush": 2,
 "strokes": [{"phase": 1, "points": [[y, x], [y, x]]},
             {"phase": 3, "brush": 4, "points": [[y, x]]}]}
```

Points are `[row, col]` pixel coordinates; a stroke is a polyline painted with a
round brush.

## Features

Per pixel, at sigmas 0.5, 1, 2, 4 and 8 px: Gaussian intensity, gradient
magnitude, Laplacian of Gaussian, both Hessian eigenvalues, local standard
deviation, plus differences of Gaussians between neighbouring scales and the
raw intensity (35 features).  The large scales give the context needed to call
the dark-grey pore back walls "pore" rather than LCO.

Half of each class's training samples are drawn within 4 px of a phase
boundary (`--boundary-frac`, default 0.5) so edges are learnt precisely.

## Current state (sample LCO60, slices 001, 451, 564, 632)

Model `rf_v4` (and its committed compact twin `rf_lco60`): auto-seeds on all
four slices + 8 scribbles on 001, dense watershed labels with pore edges at
the multi-Otsu level, fixed grey bounds 2500–50000, 200 trees, OOB accuracy
0.947.

| slice | pore | LCO | LLZTO |
|------:|-----:|----:|------:|
| 001 | 29.4 % | 42.4 % | 28.1 % |
| 451 | 28.8 % | 41.2 % | 30.0 % |
| 564 | 31.2 % | 40.2 % | 28.6 % |
| 632 | 29.5 % | 40.9 % | 29.5 % |

`docs/overlay_v4_all.png` shows all four slices with their overlays,
`docs/review_v4_<slice>.png` the per-slice review figures, and
`docs/pore_edge_v2_vs_v3.png` the pore-edge fix on zoomed regions (v2 pores
were 1–2 px too wide).  Known issue: bright charging artefacts at the top
corners of slice 001 are labelled LLZTO; crop them away.

History: v2 = watershed edges (pore 36 % on slice 001, inflated); v3 = pore
edges at the multi-Otsu level (29 %); v4 = v3 retrained on four slices with
fixed grey bounds.  Only `rf_lco60.joblib` is committed; the train command
above (with `--trees 100 --max-depth 14` for the compact one) reproduces the
models deterministically (`random_state=0`).
