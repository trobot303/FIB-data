---
title: "Random-forest segmentation of FIB-SEM cathodes and crack analysis of cycled LCO50/LCO70"
date: 2026-11-09
tags:
  - fib-sem
  - segmentation
  - random-forest
  - cracks
  - lco
  - llzto
  - session-notes
aliases:
  - FIB crack segmentation session
  - LCO50 LCO70 cycled cracks
---

# Random-forest segmentation of FIB-SEM cathodes and crack analysis of cycled LCO50/LCO70

> [!NOTE]
> **For a local Claude picking this up on Jehad's Mac.**
> - Repository: `trobot303/FIB-data` (GitHub). Local clone: `/Users/jehad/Documents/FIB-data`.
> - Branch to pull: `claude/vigilant-cerf-7etcfg` (`git pull origin claude/vigilant-cerf-7etcfg`). Everything below is on that branch.
> - Data and outputs are **not** in the repo. They live under `/Users/jehad/Desktop/work/paper 2/` (LCO50 cycled, LCO70 aligned blocks) and `/Users/jehad/Desktop/work/PlasmaFIB_Data/` (raw/merged LCO70 cycled). Full paths in section 3.
> - Setup: Miniforge, conda env `fibseg` (`conda activate fibseg`), zsh, run all commands from the repo root. Apple GPU: `scripts/tortuosity_y.py --device mps` (only the tortuosity solver uses a GPU; nothing else does).
> - The repo also holds `cathode_pipeline/` (Jehad's `cathode_analysis` package), used earlier on a Windows workstation (`D:\microscopy\Jehad\FIB data\segmented`).
> - The cloud session that wrote this could not read the Mac; every number below for full stacks came from files Jehad uploaded.

## 1. TL;DR

- One random-forest pixel classifier (`fib_rf_seg`) labels every pixel as pore, LCO, LLZTO or crack. Separate crack models per dataset; one common post-processing chain for all cycled samples.
- LCO50 cycled ds_1, slices 311–611 (50 × 50 × 15 µm, model v3, final chain): pore 16.1 %, LCO 42.8 %, LLZTO 41.0 %, crack 0.133 %.
- Cracks barely affect LCO connectivity in y: 99.85 % (cracks as void) vs 99.87 % (cracks filled); clusters 6520 vs 5013. Same finding as LCO70 earlier.
- Crack fraction in ds_1 z311–611 decreases with z: 0.173 % → 0.122 % → 0.105 % (thirds of 100 slices). LCO falls 43 % → 20 % over the same block (strong composition gradient).
- ds_1 model v3 (ridge-anisotropy features) finds 57 % of annotated crack pixels on a held-out slice, same as v2 (56 %), with 154 vs 217 false pixels.
- Object-shape features (aspect ratio of dark objects) failed: recall 38 %.
- ds_2 (BSE/SE merged, 20 nm) has very thin (~2 px, 42 nm) cracks; the ds_2 model overestimates: 0.61 % on slice 364, 59 % of detected pixels lie on 16 lines Jehad did not mark. Retraining with those lines as "not crack" killed recall (2–27 %). Pending: new annotations from slices 0–300.
- The LCO70 model finds 0.00 % cracks on ds_2 (crack probability median 0.07 on annotated cracks): crack numbers depend strongly on model and imaging.
- Reviewer point (LCO70/LCO50 ratio 10–11 vs 9 or 8 with a common size filter) was real: the two samples had different post-processing. All scripts now share one chain; LCO70 must be rerun with it before the ratio is recomputed (pending).

## 2. Data, paths and conventions

### 2.1 Datasets on the Mac (not in the repo)

| dataset | input | output (segmented) | status |
|---|---|---|---|
| LCO70 cycled, merged + cleaned | `/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/Merged_clean` | `.../Dataset 2/Merged_clean_segmented` | run with old chain (labels only); rerun pending |
| LCO70 cycled, 300 slices | `.../Dataset 2/Merged_clean_z311-611` | `.../Dataset 2/Merged_clean_z311-611_segmented` | command given, not confirmed run |
| LCO70 cycled, aligned blocks | `/Users/jehad/Desktop/work/paper 2/LCO70_cycled/Merged_clean_aligned/block<k>_z…/raw/` (+ `blocks.csv`, `shift_table.csv`) | in each block: `labels_v2/` (old), `labels_v3/`, `labels_v3_3d/` (common chain) | v2 done; v3 pending |
| LCO50 cycled ds_1, raw BSE | `/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_1_BSE` | `.../ds_1_BSE_clean` (FFT curtaining removed) | done |
| LCO50 cycled ds_1, full | `.../LCO50_cycled/ds_1_BSE_clean` | `.../ds_1_BSE_clean_segmented` | run with older versions |
| LCO50 cycled ds_1, slices 311–611 | `.../LCO50_cycled/ds_1_BSE_clean_z311-611` | `.../ds_1_BSE_clean_z311-611_segmented` | **done, final** (v3, 10 px limit) |
| LCO50 cycled ds_2, merged | `.../LCO50_cycled/ds_2_Merged_clean` | cropped to `.../ds_2_Merged_clean_1001` | done |
| LCO50 cycled ds_2, ds_2 model | `.../ds_2_Merged_clean_1001` | `.../ds_2_Merged_clean_1001_segmented` | status unknown; model overestimates |
| LCO50 cycled ds_2, LCO70 model | same | `.../ds_2_Merged_clean_1001_segmented_LCO70model` | folder did not exist at last check; not run |
| LCO50 cycled ds_2, slices 0–300 | `.../ds_2_Merged_clean_1001_z300` | `.../ds_2_Merged_clean_1001_z300_segmented` | planned; waits for new annotations |

Earlier (pristine) samples, segmented on Windows: LCO60, LCO50, LCO60_2, LCO70 in `C:\Users\303je\Desktop\work\PlasmaFIB_A970 Data\FIB data\processed_*` and `D:\microscopy\Jehad\FIB data\segmented\processed_LCO{50,60,70}_segmented`.

Preprocessing code originally in `/Users/jehad/Desktop/work/preprocessing` (now copied to `preprocessing/` in the repo).

### 2.2 Voxel sizes

| sample | pixel | slice | source | note |
|---|---|---|---|---|
| LCO60, LCO50, LCO60_2, LCO70 (pristine) | 50 nm | 50 nm | Jehad | LCO60_2 cropped to 85 × 35 µm = 1700 × 700 px |
| LCO70 cycled (aligned blocks) | 20 nm | 50 nm | Jehad | 1001 × 1001 px slices |
| LCO50 cycled ds_1 | 50 nm | 50 nm | `dataset_info.hdr` (A970_5to5_cycled, dynamic_1; 728 slices; 109.121 × 45.8368 µm) | 1001 px crop = 50 µm. **Inconsistency, unresolved:** 45.8 µm / 50 nm = 917 px rows < 1001 px |
| LCO50 cycled ds_2 | 20 nm | 20 nm | `dataset_info_5050.hdr` (A970_5050_cycled, try3, Dataset 1; 2384 slices; 53.1978 × 31.3283 µm = 2660 × 1566 px) | 1001 px crop = 20 µm |

Depth of 300 slices: 15 µm at 50 nm, 6 µm at 20 nm.

### 2.3 Label values and file layout

| value | phase in `fib_rf_seg` labels | `cathode_analysis` code |
|---|---|---|
| 1 | pore | 0 |
| 2 | LCO | 1 |
| 3 | LLZTO | 2 |
| 4 | crack | – |

- Binary masks: 255 = phase.
- Phases are assigned by grey level: black = pore, grey = LCO, almost white = LLZTO. In LCO50 ds_1 slices 262/272 the large cracked grain is bright, so it is in the LLZTO mask (Jehad never confirmed whether it is LLZTO).
- Output folder of the cycled scripts: `labels/` (per-slice model), `crack_proba/` (uint8, probability × 255), `labels_3d/` (final, after growth + length limit + width refinement), `<phase>_stack.tif` and `<phase>_3d_stack.tif`, `fractions.csv`, `fractions_3d.csv`, `connectivity_y_3d.json`, `render_cracks_3d.png`, `overlay/overlay_stack.tif` (per-slice), `overlay_3d/overlay_stack.tif` (final), `overlays_3d/*.png` (every 50th slice).
- Aligned blocks use `labels_v3`, `labels_v3_3d`, `<phase>_labels_v3_stack.tif`, `<phase>_v3_3d_stack.tif`, `fractions_v3_3d.csv`, `overlay_v3_3d/`.

### 2.4 Directions

- Arrays are (z, y, x). z = slice / milling direction. y = image rows = through-thickness. x = columns.
- Current collector at the top image edge (y = 0) unless `COLLECTOR=bottom`.
- FIB curtaining runs vertically (along columns).
- Connectivity: 6-connectivity, LCO must reach the collector face, LLZTO the separator face.

## 3. Scripts and models

### 3.1 Package `fib_rf_seg` (`python -m fib_rf_seg ...`)

| command / module | does | key options |
|---|---|---|
| `train` | trains the random forest from auto-seeds, scribble JSONs, label images | `--image/--scribbles` (repeat per image), `--auto-seeds --dense`, `--pore-level auto`, `--norm phase`, `--smooth`, `--destripe`, `--class-weight N`, `--oriented`, `--curtaining-negatives`, `--shape [ridge]`, `--crack-llzto`, `--crop`, `--rescale`, `--trees`, `--max-depth`, `--max-per-class` |
| `predict` | segments a slice folder | `--min-size 6`, `--crack-min-length 12 --crack-min-aspect 3`, `--drop-vertical DEG`, `--crack-proba`, `--per-phase --stack`, `--overwrite`, `--labels-subdir`, `--rescale`, `--crop` |
| `review` | review figure for one slice | as predict |
| `fib_rf_seg.label` | interactive scribble tool (keys 1–4) | did not open on the Mac; annotations were drawn in a drawing program instead |
| `fib_rf_seg.cropvol` | crop segmented volumes to a common size | `report`, `crop --match REF --z-anchor` |
| `scribbles.annotation_to_scribbles` | red strokes → crack, light green → not crack | `snap=0` keeps thin strokes as drawn (use this), `snap=2` snaps to darkest pixel (fragments faint cracks) |

### 3.2 Scripts (`scripts/`)

| script | does | key options / variables |
|---|---|---|
| `scripts/preprocess_lco50_cycled.sh` | FFT curtaining removal (+ optional BSE/SE blend) using `preprocessing/` | args: BSE dir, SE dir, out dir; `MASK=` |
| `scripts/crop_slices.py` | crop slices to fixed size / slice range | `--size 1001 1001`, `--z Z0 Z1` (0-based positions, end exclusive), `--origin`, `--dry-run --preview` |
| `scripts/segment_lco50_cycled.sh` | predict + 3D growth + overlays (LCO50 ds_1 default) | `MODEL`, `LOW` (0.2), `MIN_LENGTH` (25 px default here), `REFINE_WIDTH` (1), `RESUME=1` |
| `scripts/analyse_lco50_cycled.sh` | segment + connectivity + 3D render (+ tortuosity) | `MODEL`, `VOXEL` ("0.05 0.05 0.05"), `MIN_LENGTH` (10), `COLLECTOR`, `SIZE`, `SKIP_SEG=1`, `TORTUOSITY=1`, `DOWNSAMPLE` |
| `scripts/analyse_lco50_cycled_ds2.sh` | same with ds_2 model, 20 nm voxels, 25 px | args: in, out |
| `scripts/segment_lco70_cycled.sh` | LCO70 cycled with common chain | `MODEL`, `LOW`, `MIN_LENGTH`, `REFINE_WIDTH` |
| `scripts/segment_aligned_blocks.sh` | LCO70 aligned blocks, common chain, writes `labels_v3*` | `MODEL`, `LABELS`, `LOW`, `MIN_LENGTH`, `RESUME` |
| `scripts/grow_cracks_3d.py` | 3D hysteresis growth, length limit, width refinement | `--low 0.2`, `--min-length PX`, `--raw DIR --refine-width`, `--connectivity 26`, `--z-blocks N`, `--tag`, `--no-stacks` |
| `scripts/overlay_stack.py` | RGB overlay stack, cracks red | `--phases crack/all`, `--dilate`, `--scale-bar`, `--no-slices` |
| `scripts/make_overlays.py` | PNG overlays with scale bar | `--labels`, `--every`, `--voxel-um`, `--phases` |
| `scripts/connectivity_y.py` | y-connectivity LCO (void / filled) and LLZTO | `--tag 3d`, `--collector`, `--z` |
| `scripts/tortuosity_y.py` | taufactor tortuosity, ε/τ | `--tag`, `--downsample`, `--phases lco llzto`, `--device auto/cpu/cuda/mps` |
| `scripts/render_cracks_3d.py` | PyVista dimetric render | `--tag`, `--size`, `--voxel-um x y z`, `--blocks`, `--lco-step`, `--interactive` |
| `scripts/crack_stats.py` | side-by-side crack statistics | `NAME=folder ...`, `--labels`, `--px-um` (one value for all samples), `--z-um` per sample |
| `scripts/crop_valid_region.py` | crop away black padding of aligned stacks | `--mode per-slice`, `--blocks N` |

### 3.3 Models (`models/`, ~50–60 MB each)

| model | for | training data | recipe |
|---|---|---|---|
| `models/rf_lco60.joblib` | pristine LCO60 | LCO60 slices (effectively 001 only, bug) | fixed bounds 2500–50000, 100 trees, depth 14 |
| `models/rf_all.joblib` | pristine LCO70 and later stacks | 4 LCO60 + 3 LCO70 slices | `--norm phase --smooth 1.0`, 120 trees, depth 14, OOB 0.985 |
| `models/rf_cracks_v5.joblib` | **LCO70 cycled** | LCO70 cycled slices 15, 22, 38 (red + light green) | `--auto-seeds --dense --norm phase --smooth 0.7 --destripe --class-weight 4 --max-per-class 20000 --trees 200 --max-depth 16 --oriented --curtaining-negatives` |
| `models/rf_cracks_lco50_v1.joblib` | ds_1 (superseded) | ds_1 262, 272 + LCO70 15/22/38 | as v5 |
| `models/rf_cracks_lco50_v2.joblib` | ds_1 (superseded) | ds_1 045, 053, 262, 272 + LCO70 | as v5 |
| `models/rf_cracks_lco50_v3.joblib` | **LCO50 ds_1** | as v2 | v2 + `--shape ridge` |
| `models/rf_cracks_lco50_ds2_v1.joblib` | **LCO50 ds_2** | ds_2 slice 0364 (snap=0) + ds_1 4 slices | as v5 but `--class-weight 8` |

Training data in repo: `data/LCO70_cycled/`, `data/LCO50_cycled/`, `data/LCO50_cycled_ds2/`, scribbles in `data/labels/` (`*_cracks_user.json`). `data/labels/LCO50_cycled_ds2_slice-0364_notcrack_auto.json` holds the 75 unmarked detections as candidate not-crack marks (not used by any model).

### 3.4 Common post-processing chain (all cycled samples)

1. `predict --min-size 6 --crack-min-length 12 --crack-min-aspect 3 --crack-proba` (no `--drop-vertical`).
2. `grow_cracks_3d.py --low 0.2` (26-connected hysteresis from detected cracks).
3. `--min-length` = 0.5 µm: 10 px at 50 nm (ds_1), 25 px at 20 nm (ds_2, LCO70). 3D-aware: a piece is kept if its crack is long in any slice.
4. `--refine-width`: keep pixels at least half as dark as the crack's darkest point within 3 px (black top-hat, disk 4 px) + centre line.

## 4. What we tried

### 4.1 Pristine samples (phase segmentation)

| step | result |
|---|---|
| v2 watershed edges | pore 36 % on LCO60 slice 001 (inflated) |
| v3 pore edge at multi-Otsu level | 29 % |
| v4 four slices, fixed bounds | LCO60 001/451/564/632: pore 29.4/28.8/31.2/29.5 %, OOB 0.947 |
| stackB slice 0011 with `--smooth 1.0` | retrained model agrees on 97.6 % of pixels |
| LCO60_2 (`--crop --smooth 1.5 --norm phase`) | 98.4 % agreement; pore 22.9, LCO 48.6, LLZTO 28.5 % |
| LCO70 small LLZTO particles (3–8 px) | `keep_small` raised small-blob LLZTO share 0.45 → 0.90; `rf_all` |
| LCO70 slices 001/075/125 | pore 19.5/19.6/21.0, LCO 58.0/58.8/56.4, LLZTO 22.5/21.6/22.6 % |
| `--solid-level auto` on LCO70 | LLZTO 18–19 % instead of 22 % (edge-convention sensitivity) |

### 4.2 LCO70 cycled crack models v1–v5

| version | change | outcome |
|---|---|---|
| v1 | 2 verified cracks on slice 15 | 0.4 % crack area; misses faint cracks |
| `class_weight=balanced` | – | swallowed stripes and rims (23 % crack) |
| v2–v4 | dense labels, destripe, fixed weight, Jehad's red annotations (15, 22) | vertical curtaining "shot lines" still counted as cracks, esp. slice 15 |
| v5 | + slice 38 with light-green not-crack, `--oriented`, `--curtaining-negatives` | final LCO70 model |

Connectivity of LCO in y was insensitive to cracks (no numbers recorded in this note's source).

### 4.3 New large slice (2660 × 1567, later identified as a ds_2-type full frame)

| run | crack | note |
|---|---|---|
| v5 native | 0.1 % | pore 18.3, LCO 39.1, LLZTO 42.5 % |
| v5 `--rescale 0.5` | 0.1 % | |
| v5 `--rescale 0.33` | 0.0 % | |
| auto-seed model with `--crack-llzto 0.15` | 1.0 % | mostly false (grain edges, pore necks) |

### 4.4 LCO50 cycled ds_1

| step | result |
|---|---|
| v5 (LCO70 model) on cleaned slice 001 | 0.1 % crack, misses many; lowering threshold adds curtaining |
| shape filter 12 px / aspect 3 | components 37 → 9 |
| v1 (262/272 + LCO70) | finds particle cracks; 262: 0.19 %, 272: 0.33 % (v5: 0.00 %) |
| `--drop-vertical 20` | removed a real near-vertical crack in 272 → dropped for LCO50 |
| v2 (+ 045/053) | 045: 0.03 → 0.09 %, 053: 0.02 → 0.05 %, 001: 0.13 → 0.14 % |
| 3D growth (slice 272) | 0.31 → 0.48 % (low 0.2), 0.72 % (low 0.1) |
| 3D growth (slice 001) | 0.14 → 0.23 % (0.2), 0.31 % (0.1) |

Length limit on the four annotated slices (2D, after growth):

| limit | annotated area kept | unmarked area kept |
|---|---|---|
| 12 px | 100 % | 94 % |
| 15 px | 100 % | 88 % |
| 20 px | 98 % | 49 % |
| 25 px | 97 % | 19 % |
| 30 px | 96 % | 19 % |
| 40 px | 89 % | 19 % |

Leave-one-slice-out test (train on 3 ds_1 + 3 LCO70 slices, test on the 4th, full chain with 25 px limit):

| variant | change | mean recall | outside px | 045 | 053 | 262 | 272 |
|---|---|---|---|---|---|---|---|
| A | v2 recipe | 56 % | 217 | 66 | 58 | 70 | 32 |
| B | strokes as drawn (snap=0) | 56 % | 404 | | | | |
| C | B + class weight 8 | 58 % | 538 | 73 | 59 | 66 | 34 |
| D | C + full shape features | 48 % | 460 | 43 | 41 | 89 | 19 |
| E | A + full shape features | 38 % | 65 | 43 | 41 | 59 | 9 |
| F | A + ridge anisotropy only (= v3) | 57 % | 154 | 66 | 58 | 70 | 32 |

Why object-shape features failed: faint cracks fall below the dark-object thresholds, get "no object", and the model learned that cracks always have one.

### 4.5 LCO50 cycled ds_2

| measurement | value |
|---|---|
| crack contrast (unsmoothed) | 914 grey vs noise 400; LCO–pore step 11 291 (≈ 8 %) |
| ds_1 model v2 on slice 364 | 0.01 % |
| annotation conversion with snapping | 119 fragments → use snap=0 (10 strokes) |

Two-fold spatial test on slice 364 (train one half, test other):

| setting | recall per-slice (R / L) | recall after growth + 25 px (R / L) |
|---|---|---|
| smooth 0.7, weight 4 | 47 / 37 | 53 / 55 |
| smooth 0.3, weight 4 | 38 / 52 | – |
| smooth 0, weight 4 | 33 / 51 | – |
| smooth 0.7, weight 8 | 50 / 48 | 61 / 69 |
| weight 8 + ridge | 51 / 48 | 61 / 63 |
| weight 8 + all unmarked lines as not-crack | 12 / 2 | 27 / 8 |

Width refinement: ds_2 mean crack width 7 px → 2.1 px; ds_1 4.4 px → ~3 px.

Overestimate diagnosis on slice 364 (ds_2 model, final chain):

| | value |
|---|---|
| crack area | 0.61 % |
| detected pixels off Jehad's strokes | 59 %, on 16 lines |
| median crack probability, strokes vs off-stroke (held-out) | 0.63 vs 0.50 |
| share with P > 0.7, both groups | ~ 25 % |
| median length, marked vs unmarked lines | 136 px vs 60 px |
| grey / depth / distance to pore and LLZTO | same for both groups |

LCO70 model on ds_2 slice 364: crack 0.00 %, 6 % of stroke pixels with P > 0.5, median 0.07.

### 4.6 Preprocessing

`preprocessing/remove_curtaining.py` with Jehad's `preprocessing/fft_mask.npy` (1001 × 1001, 16-row band) weakens but does not remove stripes; cracks survive.

## 5. Results

### 5.1 LCO50 cycled ds_1, slices 311–611 (final: v3, 10 px = 0.5 µm limit)

| phase | mean | range across slices |
|---|---|---|
| pore | 16.1 % | 10.1–26.0 % |
| LCO | 42.8 % | 19.6–57.7 % |
| LLZTO | 41.0 % | 29.9–54.4 % |
| crack | 0.133 % | 0.039–0.310 % |

| slices | crack |
|---|---|
| 312–411 | 0.173 % |
| 412–511 | 0.122 % |
| 512–611 | 0.105 % |

| y-connectivity (6-conn, collector top) | volume | connected | percolating | clusters |
|---|---|---|---|---|
| LCO, cracks as void | 42.8 % | 99.85 % | 99.85 % | 6520 |
| LCO, cracks filled | 42.9 % | 99.87 % | 99.87 % | 5013 |
| LLZTO | 41.0 % | 98.6 % | 98.4 % | 606 |

How to read / caveats:
- Per-slice model alone: 0.043 % crack; the 3D chain triples it (mostly growth through slices).
- With the earlier 25 px (1.25 µm) limit: 0.123 %. The 0.5 µm limit adds 8 %.
- Composition changes by a factor of two over 15 µm; a mean over this block mixes different material.
- Shape statistics (host phase, length density, width, 3D objects) not yet computed for this block.
- 50 nm pixels cannot resolve the ~40 nm cracks seen in ds_2.

### 5.2 Preliminary single-slice comparisons (training slices; trend only)

| | LCO70 (3 slices) | ds_1 v3 (5 slices) | ds_2 (1 slice) |
|---|---|---|---|
| crack, % area | 0.13 | 0.16 | 0.61 |
| LCO cracked, % | 0.19 | 0.20 | 1.22 |
| LLZTO cracked, % | 0.00 | 0.15 | 0.00 |
| cracks in LCO / LLZTO, % | 94 / 0 | 49 / 46 | 100 / 0 |
| length density, µm/µm² | 0.026 | 0.027 | 0.147 |
| mean width | 50 nm* | 60 nm* | 42 nm |

\* computed at 20 nm/px; ds_1 is 50 nm/px, so its width is really ~150 nm (3 px). ds_2 0.61 % is an overestimate (section 4.5).

### 5.3 Other numbers

| item | value |
|---|---|
| LCO70 aligned-block test (3 slices, common chain) | crack voxels 5487 → 7586 after growth; 0.25 % |
| cathode_analysis in cloud | 50 tests pass; 300-slice scale test 34 min, peak 7.9 GB |
| taufactor CPU cost | 16 B/voxel, 1.8 s per iteration per 71 Mvoxel (4 threads); 700 × 700 × 1700 ≈ 14 GB |
| predict speed | ~10 s per 1001² slice (cloud, 4 cores); ~14 s on ds_2 |

## 6. Interpretation

Can be claimed:
- Cracking in cycled LCO50 ds_1 is a small volume fraction (≈ 0.13 %) and does not break LCO percolation in y; it fragments LCO into more small clusters.
- Cracks occur in both LCO and the large LLZTO particles in ds_1 (single-slice evidence only).
- Crack fraction varies along z within ds_1.

Avoid:
- Comparing crack fractions across datasets processed with different models or post-processing without saying so. ds_2 with the LCO70 model gives 0 %, with its own model 0.61 %.
- Treating ds_2 crack numbers as final (overestimated).
- Comparing ds_1 (50 nm) and ds_2 (20 nm) crack widths or densities as material differences: part is resolution.
- Quoting the 10–11 LCO70/LCO50 ratio: the reviewer was right that post-processing differed (LCO50 had no objects < 15 px; a quarter of LCO70 objects were < 10 px).

Paper / reviewer strategy discussed:
- Use one chain for all samples and state it in methods: islands < 6 px removed, line filter 12 px / aspect 3, 3D growth at probability 0.2, 0.5 µm minimum crack length, half-depth width refinement, model per sample (state which).
- Rerun LCO70 with the common chain, then recompute the ratio (expected to differ from both 10–11 and 8–9).
- Report connectivity and, if computed, tortuosity (ε/τ) as the transport-relevant metrics; crack density as the damage metric.
- Use full stacks or several blocks; one 300-slice block can be unrepresentative.

## 7. Open items (zsh, macOS, from the repo root)

Setup once per terminal:

```zsh
cd /Users/jehad/Documents/FIB-data
git checkout claude/vigilant-cerf-7etcfg && git pull origin claude/vigilant-cerf-7etcfg
conda activate fibseg
```

1. LCO70 cycled, 300 slices with the common chain:

```zsh
IN="/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/Merged_clean"
python scripts/crop_slices.py --in "$IN" --out "${IN}_z311-611" --size 1001 1001 --z 311 611
MODEL=models/rf_cracks_v5.joblib VOXEL="0.02 0.02 0.05" MIN_LENGTH=25 \
  bash scripts/analyse_lco50_cycled.sh "${IN}_z311-611" "${IN}_z311-611_segmented"
```

2. LCO70 aligned blocks, common chain (writes `labels_v3`, `labels_v3_3d`):

```zsh
bash scripts/segment_aligned_blocks.sh "/Users/jehad/Desktop/work/paper 2/LCO70_cycled/Merged_clean_aligned"
```

3. Shape statistics, one call per pixel size:

```zsh
python scripts/crack_stats.py ds1_z311-611="/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_1_BSE_clean_z311-611_segmented" \
  --px-um 0.05 --z-um 0.05 --out ~/Desktop/crack_stats_ds1_z311-611.csv
python scripts/crack_stats.py LCO70_z311-611="/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/Merged_clean_z311-611_segmented" \
  --px-um 0.02 --z-um 0.05 --out ~/Desktop/crack_stats_LCO70_z311-611.csv
```

4. ds_2 retraining (blocked on Jehad): 2–3 slices from slices 0–300 of `ds_2_Merged_clean_1001` (e.g. ~50, 150, 250), originals + annotated copies with red on every crack and light green on the faint parallel non-crack lines; plus the visual criterion that separates them. Then train only on them (snap=0) and run:

```zsh
python scripts/crop_slices.py --in "/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_2_Merged_clean_1001" \
  --out "/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_2_Merged_clean_1001_z300" --size 1001 1001 --z 0 300
bash scripts/analyse_lco50_cycled_ds2.sh "/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_2_Merged_clean_1001_z300" \
  "/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_2_Merged_clean_1001_z300_segmented"
```

5. ds_2 with the LCO70 model (model-dependence check, never run):

```zsh
MODEL=models/rf_cracks_v5.joblib bash scripts/analyse_lco50_cycled_ds2.sh \
  "/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_2_Merged_clean_1001" \
  "/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_2_Merged_clean_1001_segmented_LCO70model"
```

6. Tortuosity of LCO with and without cracks (Apple GPU):

```zsh
python scripts/tortuosity_y.py "/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_1_BSE_clean_z311-611_segmented" \
  --tag 3d --phases lco --downsample 2 --device mps --out tortuosity_y_3d.json
```

7. Resolve the ds_1 voxel/size inconsistency (917 px frame height vs 1001 px slices): check which acquisition and processing `ds_1_BSE_clean` comes from.
8. Confirm whether the bright cracked grain in ds_1 slices 262/272 is LLZTO.
9. After items 1–3: recompute the LCO70/LCO50 ratio with the common chain for the reviewer reply.

## 8. Pitfalls

| pitfall | avoid by |
|---|---|
| stats/script "file not found" on the Mac | `git pull` first; check the branch with `git branch --show-current` |
| stats error "no *_labels.tif" | the output folder was never created; run the analysis before `crack_stats.py` |
| old labels kept after a model change | scripts now overwrite by default; `RESUME=1` only after an interrupted run |
| ds_2 run with the ds_1 model | `analyse_lco50_cycled_ds2.sh` sets the model; check the `== model:` first line of every run |
| wrong voxel size (20 vs 50 nm for ds_1) | ds_1 = 50 nm (`dataset_info.hdr`), ds_2 = 20 nm; scale `MIN_LENGTH` to 0.5 µm (10 vs 25 px) |
| `segment_lco50_cycled.sh` called directly defaults to 25 px | set `MIN_LENGTH=10` for ds_1, or use `analyse_lco50_cycled.sh` |
| `crack_stats.py` has one `--px-um` | one call per pixel size |
| `crop_slices.py --z` counts positions from 0 | `--z 311 611` = files 312–611 (300 slices) |
| `--drop-vertical` deletes near-vertical real cracks | not used for cycled LCO50/LCO70 common chain |
| snapping thin strokes fragments them | `annotation_to_scribbles(..., snap=0)` |
| annotated TIFF loses grey under strokes | always send the original slice with the annotated copy |
| 3D growth across non-contiguous slices is meaningless | grow only contiguous stacks / per block |
| `fib_rf_seg.label` window did not open on Mac | annotate in a drawing program (red / light green) |
| `No module named skimage` | `pip install -r requirements.txt` in `fibseg` |
| `py` not found on Mac/Windows Miniforge | use `python` |
| Windows: `cd d:` does not change drive | `cd /d D:\...` |
| Windows: script "not recognized" | it was not pulled; no git → download zip |
| blocks.csv columns differ from expected | `render_cracks_3d.py --blocks` reads `shift_table.csv` |
| GitHub warns on > 50 MB model files | pushes still work; models need `!models/<name>` in `.gitignore` |
| cloud session cannot read Mac folders | upload the small result files, or run Claude on the Mac |
| leave-one-out training is slow | ~15 min per training on 4 cloud cores |

Related: [README](../../README.md), [README: cracks in cycled samples](../../README.md#cracks-in-cycled-samples-class-4), [README: preprocessing](../../README.md#preprocessing-bsese-merge-fft-curtaining-removal), [README: LCO50 cycled crack model](../../README.md#lco50-cycled-crack-model), [README: LCO50 full analysis](../../README.md#lco50-cycled-full-analysis-scriptsanalyse_lco50_cycledsh), [README: ds_2 model and width refinement](../../README.md#lco50-cycled-ds_2-crack-model-and-crack-width-refinement), [README: ds_1 v3 and leave-one-out test](../../README.md#ds_1-model-v3-and-the-leave-one-slice-out-test), figures: [ds_1 model comparison](../overlays/LCO50_cycled_model_comparison.png), [ds_1 v1 vs v2](../overlays/LCO50_cycled_v1_vs_v2.png), [ds_1 slice 272 v5 vs new](../overlays/LCO50_cycled_slice272_v5_vs_new.png), [crack growth](../overlays/LCO50_cycled_crack_growth.png), [length limit](../overlays/LCO_cycled_crack_length_limit.png), [shape filter](../overlays/LCO50_cycled_slice001_crack_filter.png), [ds_2 model](../overlays/LCO50_cycled_ds2_model.png), [preprocessing](../overlays/preprocess_lco50_cycled_slice001.png), [LCO70 cycled v5](../overlays/LCO70_cycled_slice-0015_cracks_v5.png), [LCO70 render demo](../overlays/LCO70_cycled_render_demo.png). No slides were made.
