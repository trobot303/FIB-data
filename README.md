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

## Other stacks of the same sample type

The committed model was tested on `data/stackB/slice_0011.tif`, a slice from a
different, larger (2800×1201) and un-denoised volume of the same cathode type.
Its raw grey levels match LCO60 (phase medians 11.7k / 23.7k / 43.8k vs
11.2k / 23.2k / 44.9k), so the stored bounds apply.  Because the slice is
noisy, pre-smooth it with `--smooth 1.0` (Gaussian sigma in px, applied before
normalisation):

```bash
python -m fib_rf_seg predict --model models/rf_lco60.joblib --input /path/to/other_stack \
    --out /path/to/other_stack_segmented --smooth 1.0 --pore-level auto --min-size 20 --per-phase --stack
```

A model retrained on that slice alone agrees with the LCO60 model on 97.6 % of
pixels; the difference is a 1 px wider LLZTO edge in the retrained one.  See
`docs/stackB_slice_0011_overlay.png`.  For a new stack, check that (a) the
raw phase grey levels are similar (`--lo/--hi` otherwise) and (b) the pixel
size is similar, since feature scales are in pixels; then inspect one review
figure before running the whole volume.

### LCO60_2: a raw, uncropped stack with different contrast

`data/LCO60_2/slice-4.tif` (3000×1500 px, 50 nm/px) differs from LCO60 in
three ways, each handled by an option:

| difference | measurement | option |
|------------|-------------|--------|
| not cropped: cathode spans rows ≈60–1430, cols ≈150–2750 | 85 × 35 µm = 1700 × 700 px | `--crop 400 1100 600 2300` (rows Y0 Y1, cols X0 X1; see `docs/LCO60_2_crop_box.png`) |
| raw, noisy | noise std 3400 vs 460 in processed LCO60 | `--smooth 1.5` |
| different contrast, brightness drifts along z | phase medians 5.8k / 17.1k / 41.0k vs 11.2k / 23.2k / 44.9k | `--norm phase` |

`--norm phase` normalises each slice so that its pore and LLZTO medians (from
a multi-Otsu split of the smoothed slice) land on the training values 0.18
and 0.875, which makes the model independent of the stack's contrast and of
drift along z.  With these options the committed model agrees on 98.4 % of
pixels with a model trained on this slice itself (`docs/LCO60_2_slice-4_overlay.png`):
pore 22.9 %, LCO 48.6 %, LLZTO 28.5 % in the crop.  `scripts\\segment_lco60_2_stack.bat`
runs it on `...\\FIB data\\processed_LCO60_2`.

### LCO70: small LLZTO particles, and the combined model `rf_all`

The LCO70 slices (`data/LCO70/`, already cropped to 1700×700, HistBal-Gauss-
Kuwahara processed, same grey scale as LCO60) contain many LLZTO particles of
only 3–8 px.  The LCO60 model labelled about half of them LCO.  The cause was
in the training-label generation: auto-seeds were eroded by 3 px, so objects
smaller than ~7 px never received a seed and the watershed flooded them with
the surrounding phase, teaching the forest "small bright blob = LCO".
`auto_seeds` now keeps small connected cores whole (`keep_small`, default
200 px), which raises the share of small bright blobs labelled LLZTO on LCO70
from 0.45 to 0.90.

`models/rf_all.joblib` is trained with the fixed seeds on the four LCO60 and
the three LCO70 slices (`--norm phase --smooth 1.0`, 120 trees, depth 14,
OOB 0.985).  It is the model to use for LCO70 (`scripts\\segment_lco70_stack.bat`)
and for any future stack.  On LCO60 it agrees with `rf_lco60` on 95.6 % of
pixels; the difference is mostly the recovered small particles and a slightly
tighter LLZTO edge.

| LCO70 slice | pore | LCO | LLZTO |
|------------:|-----:|----:|------:|
| 001 | 19.5 % | 58.0 % | 22.5 % |
| 075 | 19.6 % | 58.8 % | 21.6 % |
| 125 | 21.0 % | 56.4 % | 22.6 % |

See `docs/overlay_LCO70_rf_all.png`.  Because LCO70's LLZTO grains are small,
the LLZTO fraction is very sensitive to the edge convention: `--solid-level
auto` (edge exactly at the multi-Otsu threshold, about one pixel further in)
gives 18–19 % instead of 22 %.  Use the same convention for every sample you
compare.

Note on earlier models: a bug in `train` (one `--scribbles` file given with
several `--image` files truncated the image list) meant `rf_v4`/`rf_lco60`
were in fact trained on slice 001 only.  Fixed; the results on LCO60 were
still fine because the slices are alike.

## Cracks in cycled samples (class 4)

`fib_rf_seg` has a fourth class, `4 = crack`, for cycled electrodes (thin dark
lines through and between the LCO grains).  Cracks cannot be seeded
automatically with enough reliability (a black top-hat finds grain-edge
shadows as readily as cracks), so they are trained from scribbles drawn with
`python -m fib_rf_seg.label` (key `4`) or supplied as a JSON file.  For a crack
model use

```bash
python -m fib_rf_seg train --image data/LCO70_cycled/slice-0015.tif \
    --scribbles data/labels/LCO70_cycled_slice-0015_cracks.json \
    --auto-seeds --dense --norm phase --smooth 0.7 --destripe --class-weight 4 \
    --max-per-class 25000 --model models/rf_cracks_v1.joblib
```

* `--destripe` removes the vertical FIB curtaining (column offsets that survive
  a long smoothing along the rows); without it the dark stripes are confused
  with cracks.  The flag is stored in the model and applied by `predict`.
* `--auto-seeds --dense` label pore / LCO / LLZTO everywhere by watershed;
  crack scribbles are then painted on top 1 px wide with a 3 px unlabelled rim.
* `--class-weight 4` gives the (rare) crack class a fixed weight; the default
  `balanced` makes the forest call every dark stripe and grain rim a crack.

`rf_cracks_v1` was trained on two verified cracks of slice-0015 (0.4 % crack
area); see `docs/overlays/LCO70_cycled_slice-0015_cracks_v1.png`.  It finds the
prominent cracks but misses faint ones; more scribbles on several slices are
needed for a production model.

## Cropping segmented volumes to a common size

All stacks share a 50 nm isotropic voxel, but LCO60_2 has fewer slices than
LCO70 and LCO50.  `fib_rf_seg.cropvol` trims the *segmented* output folders
to one size without resampling:

```bash
python -m fib_rf_seg.cropvol report  OUT_LCO60_2 OUT_LCO70 OUT_LCO50        # sizes in voxels and um
python -m fib_rf_seg.cropvol crop    OUT_LCO70 OUT_LCO50 --match OUT_LCO60_2 --z-anchor start --stack
```

`--match` takes Z (slices), Y and X from the reference folder; `--size Z Y X`
sets them directly.  `--z-anchor start|center|end` or `--z-start N` chooses
which slices to keep; the x/y window is cut from the centre of each slice
(`--xy-anchor start` for the top-left corner).  Each input gets a
`<folder>_cropped` twin with the same layout (`labels/`, `pore/`, `LCO/`,
`LLZTO/`, `fractions.csv`, optional phase stacks) plus `crop_info.json`
recording the offsets.  `scripts\\crop_volumes_to_LCO60_2.bat` runs this for
the three samples.

## Running the 3D analysis (cathode_analysis) in the cloud

`cathode_pipeline/` is the `cathode_analysis` workstation package (v2).  It can
run in a Claude cloud session (4 cores, 15 GB RAM, no GPU) on the
random-forest segmentations.  Only PyPI and GitHub are reachable from the
session, so the data travel through the repository:

1. **On the workstation**, after `predict --stack` (and `cropvol crop --stack`
   for the volumes trimmed to the LCO60_2 size), run
   `scripts\\push_label_stacks.bat`.  It copies `labels_stack.tif` of the four
   samples from `...\\FIB data\\segmented\\` into `data/segmented/<sample>/`
   and pushes them (zlib-compressed label volumes, roughly 50–60 MB per
   700-slice sample).
2. **In the session**: `bash scripts/setup_cloud_pipeline.sh` builds the
   virtual environment (`cathode_pipeline/requirements_cloud.txt`; the pinned
   `requirements.txt` is for Python 3.10–3.12 on the workstation) and runs the
   50 pipeline tests.  Then, per sample,
   `python scripts/labels_to_masks.py data/segmented/<sample>/labels_stack.tif`
   writes the `pores.tiff` / `LCO.tiff` / `LLZTO.tiff` mask stacks the mask
   mode expects, and

   ```bash
   cd cathode_pipeline && . .venv/bin/activate
   python -m cathode_analysis.cli validate --config config/cloud_segmented.yaml
   python -m cathode_analysis.cli run      --config config/cloud_segmented.yaml --stage all
   ```

   `config/cloud_segmented.yaml` is `workstation_segmented.yaml` with the four
   samples, `device: cpu`, `n_workers: 4` and `tortuosity.downsample: 4`;
   set `analysis_volume.shape_zyx` and the sample crops to the common volume
   before running.  Results land in `results_pipeline/` (summary tables,
   per-sample JSON/CSV, figures) and are committed back to the repository.

Verified in the session: all 50 tests pass; a 3-slice LCO70 volume runs
through every stage in under 2 minutes.

### Tortuosity at full resolution

`cathode_pipeline\run_tortuosity_full.bat` re-runs only the tortuosity stage
with `tortuosity.downsample: 1` (50 nm voxels) and no wall-time limit, keeps
the earlier result as `tortuosity_downsampled.json`, then refreshes the report.
Cost measured with taufactor on CPU: 16 B per voxel and 1.8 s per iteration
per 71 Mvoxel on 4 threads, so a 700 x 700 x 1700 volume needs about 14 GB
and roughly 20 s per iteration on CPU; with the tens of thousands of
iterations a 700-voxel-long path needs, that is days per phase on CPU and
hours on a GPU.  `downsample: 2` (100 nm) costs 1/8 of the memory and about
1/30 of the time and changes tau by a few percent.

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

## Preprocessing (BSE/SE merge, FFT curtaining removal)

Jehad's preprocessing scripts live in `preprocessing/` (`merge_bse_se.py`, `remove_curtaining.py`,
`select_fft_filter.py`, hand-picked `fft_mask.npy` for 1001×1001 px slices).
`scripts/preprocess_lco50_cycled.sh` runs them on the LCO50 cycled BSE stack on the Mac
(`ds_1_BSE` → `ds_1_BSE_clean`); pass a registered SE folder as 2nd argument to blend first.
Example before/after: `docs/overlays/preprocess_lco50_cycled_slice001.png`.

Crack post-processing options of `predict`/`review`: `--drop-vertical DEG` (vertical streaks → surrounding
phase), `--crack-min-length PX` and `--crack-min-aspect RATIO` (drop dot-like crack components; e.g. 12 and 3
for the heavily curtained LCO50 cycled data, `scripts/segment_lco50_cycled.sh`).
`scripts/overlay_stack.py --raw <slices> --labels <labels> --out <folder>` writes an RGB overlay stack
(`overlay_stack.tif`, one page per slice, cracks in red) plus per-slice overlay TIFFs; `--phases all`
colours all phases, `--scale-bar` adds a bar. `scripts/segment_lco50_cycled.sh` runs it at the end.

### LCO50 cycled crack model

`models/rf_cracks_lco50_v2.joblib` (v1: slices 262/272 only): trained on the FFT-cleaned LCO50 cycled slices 045, 053, 262 and 272
(`data/LCO50_cycled/`, red crack annotations converted with `annotation_to_scribbles` to
`data/labels/LCO50_cycled_slice-2xx_cracks_user.json`) together with the three annotated LCO70 cycled
slices, same recipe as rf_cracks_v5 (`--auto-seeds --dense --pore-level auto --norm phase --smooth 0.7
--destripe --class-weight 4 --max-per-class 20000 --trees 200 --max-depth 16 --oriented
--curtaining-negatives`). Used by `scripts/segment_lco50_cycled.sh` without `--drop-vertical` (the
sample has near-vertical cracks) and with the dot filter `--crack-min-length 12 --crack-min-aspect 3`.
Comparison with rf_cracks_v5: `docs/overlays/LCO50_cycled_model_comparison.png`.

**Crack growth in 3D** (`scripts/grow_cracks_3d.py`): the per-pixel classifier finds a crack only on
its clearest parts and may miss it in the next slice. `predict --crack-proba` writes the crack
probability per slice (`crack_proba/<stem>_crackp.tif`, uint8); the growth script then adds every voxel
with probability > `--low` (0.2) that is 26-connected in 3D to a detected crack, so cracks are completed
along their length and propagate through neighbouring slices, while isolated low-probability pixels
(curtaining) are not added. Output `labels_3d/`, `labels_3d_stack.tif`, `<phase>_3d_stack.tif`,
`fractions_3d.csv`. `scripts/segment_lco50_cycled.sh` runs it (LOW=0.15 for more growth) and writes
both overlay stacks (`overlay/overlay_stack.tif` per-slice, `overlay_3d/overlay_stack.tif` grown).
Example: `docs/overlays/LCO50_cycled_crack_growth.png`.

**Common post-processing for the cycled samples (reviewer point).** LCO70 cycled was first segmented with
island removal only (labels_v2); LCO50 cycled with the crack shape filter and 3D growth. Both LCO70 scripts
(`scripts/segment_lco70_cycled.sh`, `scripts/segment_aligned_blocks.sh`, output `labels_v3` / `labels_v3_3d`)
now use the same chain as LCO50: `--min-size 6 --crack-min-length 12 --crack-min-aspect 3`, no
`--drop-vertical`, then `grow_cracks_3d.py --low 0.2`. Per-phase stacks written with a label subfolder
other than `labels` are named `<phase>_<subfolder>_stack.tif`.

### LCO50 cycled: full analysis (`scripts/analyse_lco50_cycled.sh`)

Same analysis as for LCO70 cycled, with the LCO50 model: segmentation + common post-processing + 3D crack
growth + overlay stacks (`segment_lco50_cycled.sh`), y-connectivity of LCO (cracks as void / filled) and
LLZTO (`connectivity_y.py --tag 3d`), dimetric 3D rendering (`render_cracks_3d.py --tag 3d`), optional
y-tortuosity (`TORTUOSITY=1`). `--tag 3d` makes these scripts read the grown-crack stacks
`<phase>_3d_stack.tif` (use `--tag v3_3d` for the LCO70 aligned blocks). Set `VOXEL="x y z"` (um) to the
real voxel size of the dataset (LCO50 cycled ds_1: 20 nm isotropic, default) and `COLLECTOR=top|bottom`.

`scripts/crop_slices.py --in <folder> --out <folder> --size 1001 1001 [--anchor center | --origin Y0 X0] [--z Z0 Z1] [--dry-run] [--preview p.png]`
crops raw slices to a fixed size (e.g. LCO50 cycled ds_2 to the 1001 x 1001 px of LCO70 cycled) before segmentation.

**Crack length limit** (`grow_cracks_3d.py --min-length PX`, default 25 px = 0.5 um in the cycled scripts,
`MIN_LENGTH=...` to change): after the 3D growth, crack components whose longest in-plane piece is shorter
than the limit are removed (dark spots at particle contacts, curtaining bits); a short piece that belongs to
a crack that is long in another slice is kept. On the four annotated LCO50 slices 25 px keeps 97 % of the
annotated crack area and removes 81 % of the unmarked short detections. Applied to LCO50 and LCO70 alike.

### LCO50 cycled ds_2 crack model and crack width refinement

`models/rf_cracks_lco50_ds2_v1.joblib`: for the BSE/SE-merged ds_2 stack, whose cracks are ~2 px wide and only
~8 % of the LCO-pore grey step deep. Trained on the annotated ds_2 slice 0364 (`data/LCO50_cycled_ds2/`, strokes
converted with `annotation_to_scribbles(..., snap=0)`, i.e. as drawn) plus the four annotated ds_1 slices, same
recipe as the other crack models but `--class-weight 8`. Two-fold spatial cross-validation on slice 0364
(train on one half, test on the other): weight 8 finds 61-69 % of the annotated crack pixels after growth vs
53-55 % for weight 4; the ds_1 model finds almost none. Run: `scripts/analyse_lco50_cycled_ds2.sh`.

**Width refinement** (`grow_cracks_3d.py --raw <slices> --refine-width`, default in all cycled scripts,
`REFINE_WIDTH=0` to switch off): the classifier marks a 5-7 px band around a 2 px crack. After growth and the
length limit, each crack is trimmed to the pixels at least half as dark (black top-hat, disk radius 4 px) as the
darkest crack pixel within 3 px, i.e. the full width at half depth, plus the centre line. Mean width on ds_2:
7 px -> 2.1 px; on ds_1: 4.4 px -> ~3 px. Applied to LCO50 ds_1, ds_2 and LCO70 alike.
