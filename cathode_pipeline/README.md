# cathode_analysis

Headless Python pipeline for the 3D microstructure analysis of all-solid-state
battery composite cathodes (LCO active material, LLZTO garnet electrolyte,
pores) from FIB-SEM slice stacks. It reproduces the geometric analysis of
Clausnitzer et al., *Batteries & Supercaps* 2023, e202300167: volume fractions,
connectivity, active interface area, tortuosity, grain segmentation and
through-thickness profiles.

No Jupyter, no display, no prompts. Everything is driven by one YAML file.

## Layout

```
cathode_pipeline/
  cathode_analysis/
    cli.py           entry point:  python -m cathode_analysis.cli ...
    config.py        YAML loading, validation, per-sample resolution
    pipeline.py      stage registry, StageContext, skip/force, logging, peak RSS
    io.py            slice discovery (2D TIFF sequence or multipage), streaming uint8 load, label I/O
    volume_ops.py    z-chunked label operations (island removal, cavity filling, face counts)
    segmentation.py  stage segment      -> labels.npy, segment.json, qc/
    connectivity.py  stage connectivity -> connectivity.json, labels_connected.npy
    surface.py       stage surface      -> surface.json
    tortuosity.py    stage tortuosity   -> tortuosity.json
    grains.py        stage grains       -> grains_<phase>.npy/.csv, grains.json, qc/
    profiles.py      stage profiles     -> profiles.csv/.png
    report.py        stage report       -> summary.csv/json, figures/
    paper_figures.py Clausnitzer-style figures (Fig. 2 / 3 / 7) made by the report stage
    plotting.py      Agg-only matplotlib helpers
  config/workstation.yaml  the production run: all samples, identical analysis volume, pores = VF only
  config/workstation_segmented.yaml  the same analysis on ALREADY SEGMENTED stacks (one binary TIFF per phase)
  config/default.yaml      reference schema with every parameter commented
  config/dev_crop.yaml     small crops of the stacks for quick local checks
  tests/                pytest; includes a synthetic phantom that runs every stage
  requirements.txt      pinned versions
  environment.yml       Miniforge / conda environment (python 3.11 + requirements.txt)
  run_all.sh            venv + full run (Linux / macOS / Git Bash)
  run_all.ps1           the same for Windows PowerShell
  run_segmented.sh/.ps1 the same with config/workstation_segmented.yaml
```

## Conventions

* Arrays are `(z, y, x)`; z = slice index (milling direction), y = image rows,
  x = image columns. Voxels are isotropic (`voxel.size_nm`).
* Phase codes in every label volume: `0 = pore, 1 = LCO, 2 = LLZTO`.
  After the connectivity stage, `labels_connected` uses `3 / 4 / 5` for the
  isolated part of pore / LCO / LLZTO.
* Through-thickness direction: `voxel.through_thickness_axis` (default `y`)
  with the current collector at `voxel.current_collector_face` (default `min`,
  i.e. the top image edge) and the LLZTO separator at the opposite end. Faces
  in the config can be named `collector`, `separator` or `<axis>_min|max`.
* Relative paths in `paths` resolve against the YAML file. Sample `input_dir`
  is relative to `paths.data_root`; outputs go to `paths.output_root/<sample>/`.

## Same analysis volume for every sample

`analysis_volume.shape_zyx` fixes the number of voxels every sample is cropped
to; the validator refuses a config where any sample's `crop` spans a different
extent. `config/workstation.yaml` uses 500 x 680 x 1680 voxels (25 x 34 x 84 um)
centred in the usable region of each stack (margins with mounting material or
bright edge strips are excluded).

## Already segmented input (binary mask stacks)

A sample can skip the grey-level segmentation entirely by pointing at one
binary TIFF stack per phase (non-zero = phase present) instead of a slice
pattern:

```yaml
samples:
  - name: LCO50
    input_dir: segmented_LCO50           # relative to paths.data_root
    masks: {pore: pores.tiff, lco: LCO.tiff, llzto: LLZTO.tiff}   # file name or glob per phase
    crop: {z: [0, 700], y: [0, 700], x: [0, 1700]}
    downsample: 1

segmentation:
  masks:
    priority: [llzto, lco, pore]   # where masks overlap, the first listed phase wins
    unassigned: majority           # majority | pore | lco | llzto  for voxels covered by no mask
    postprocess: false             # true = also apply segmentation.postprocess (island removal, hole filling)
```

The segment stage then imports the masks into the usual `labels.npy` (codes
0/1/2) and writes `segment.json` with `mode: masks`, the per-phase overlap
counts, the number and fraction of voxels covered by no mask and how they were
filled; QC phase maps go to `qc/phases_z*.png`. Every later stage runs
unchanged. `slice_range`, `crop` and `downsample` apply to the masks as they do
to grey slices (block-averaged masks are re-binarised at 0.5). `validate` and
`inspect` list the three stacks; the stage refuses stacks of different shape.
`config/workstation_segmented.yaml` is set up for the `segmented_LCO50` folder
(700 x 700 x 1700 voxels, no overlaps, about 1 % uncovered voxels) and contains
a commented template for adding more samples.

## Pores: volume fraction only

Leave `pore` out of `connectivity.reference_faces`, list only `[lco, llzto]` in
`surface.interfaces`, and keep `tortuosity.phases` / `grains.phases` at
`[lco, llzto]`. The pore volume fraction is still reported by the segment stage,
in the summary table and in the through-thickness profile.

## Usage

```bash
python -m pip install -r requirements.txt
python -m cathode_analysis.cli validate --config config/default.yaml
python -m cathode_analysis.cli inspect  --config config/default.yaml         # slices found, shape, dtype
python -m cathode_analysis.cli run --config config/default.yaml --stage all
python -m cathode_analysis.cli run --config config/default.yaml --stage segment --sample LCO50 --force
python -m cathode_analysis.cli run --config config/default.yaml --stage all --dry-run
```

Stages: `all | segment | connectivity | surface | tortuosity | grains | profiles | report`.
A stage is skipped when all its outputs exist, unless `--force` is given.
Each run writes `results/logs/run_<timestamp>.log` and `results/config_resolved.yaml`;
each sample folder gets a `stage_status.json` with wall time and peak RSS per stage.
Exit codes: 0 ok, 2 config error, 3 stage not implemented.

## What each stage computes

| stage | method | key outputs |
|---|---|---|
| segment | percentile-scaled uint8, exact histogram, multi-Otsu / k-means / manual thresholds, `intensity_order` maps grey classes to phases, island removal with neighbour-majority refill, filling of small enclosed pore cavities. With `masks`: import of one binary stack per phase, priority on overlap, neighbour-majority fill of uncovered voxels | `labels.npy`, volume fractions, thresholds (or mask statistics), QC histogram and overlays (or phase maps) |
| connectivity | `scipy.ndimage.label` per phase; a cluster is *connected* if it touches a reference face (LCO: collector, LLZTO: separator, pores: either) | connected / isolated fractions, cluster counts, percolation collector-to-separator, `labels_connected.npy` |
| surface | marching cubes on smoothed phase masks (z-chunked). For three phases the pairwise interface is `A_ab = (S_a + S_b - S_c) / 2`. The *active* fraction is the ratio of voxel faces where both sides are connected | interface areas (um^2, 1/um), active areas |
| tortuosity | taufactor (torch, GPU if available) or porespy on the connected cluster, per phase and axis; non-spanning phases get `inf` | tau, D_eff/D_0 |
| grains | h-maxima marker watershed on the distance transform (or porespy SNOW), small-region merging | per-grain volume, equivalent diameter, sphericity, contact area with each other phase, size distributions |
| profiles | slabs along the through-thickness axis | volume fraction, connected fraction, interface density, grain count vs depth |
| report | collects all JSON/CSV results | `summary.csv`, `summary_long.csv`, comparison figures |

## Memory

Peak RSS per stage is logged. Rough per-voxel costs: segment ~2 B, connectivity
~6 B (one int32 component array), surface ~5 B per chunk, grains ~13 B (float64
distance transform + int32 regions), tortuosity ~5 B on CPU (GPU memory on CUDA).
For the full LCO50 stack (3.3 Gvoxel) that means 20 GB for connectivity and
~45 GB for grains. Use `samples[].downsample` (whole pipeline), `grains.downsample`
or `tortuosity.downsample` (stage only), or `crop` when the workstation cannot
hold that. Tortuosity on CPU is slow (minutes per direction for 40 Mvoxel);
use a CUDA build of torch on the workstation.

## Transfer to the workstation

Copy the `cathode_pipeline/` folder (without `results*/` and `.venv/`), place the
`processed_LCO*` (grey slices) or `segmented_LCO*` (mask stacks) folders where
`paths.data_root` points (default: the parent folder of the project), then
`./run_all.sh` or `.\run_all.ps1` for grey slices and `./run_segmented.sh` or
`.\run_segmented.ps1` for pre-segmented stacks (Windows PowerShell: use
`powershell -ExecutionPolicy Bypass -File .\run_segmented.ps1` if scripts are
blocked). The pinned requirements target
Python 3.10-3.12 (torch wheels); install a CUDA torch first if a GPU is present
(see the comment in `requirements.txt`).

## Miniforge / conda instead of the venv scripts

```bash
cd cathode_pipeline
mamba env create -f environment.yml        # or: conda env create -f environment.yml
mamba activate cathode
python -m cathode_analysis.cli validate --config config/workstation_segmented.yaml
python -m cathode_analysis.cli run --config config/workstation_segmented.yaml --stage all
```

For a GPU, activate the environment, install the CUDA torch build first (see
`requirements.txt`), then `pip install -r requirements.txt`.

## Tortuosity: direction, speed and convergence

`tortuosity.directions` selects the axes; `[y]` (through-thickness, collector
to separator) is all the Clausnitzer-style figures need and cuts the number of
solves by three. The solver (taufactor, SOR) needs a number of iterations that
grows with the length of the volume in voxels and with the tortuosity of the
phase, so a near-percolation minority phase (LLZTO in LCO70) is by far the
slowest. Levers, in order of effect:

1. **GPU**: a CUDA torch build makes each iteration 20-50x faster.
2. `tortuosity.downsample`: 4 (200 nm voxels) is ~30x faster than 2 and changes
   tau by a few percent. Downsampling keeps every phase's volume fraction
   (`preserve_volume_fraction: true`); a majority vote would erode thin necks
   and break the percolation of a minority phase.
3. `taufactor.max_seconds`: wall-time budget per solve; when exceeded, the
   current tau is written with `converged: false` and `timed_out: true`.
4. `taufactor.convergence` (default 0.02 = 2 % flux mismatch) is already loose;
   loosening it further saves little and risks stopping on a still-drifting
   field. taufactor additionally requires tau to change by < 0.002 between
   checks (hard-coded).

Progress (iteration, current tau, ms/iteration) is logged every
`taufactor.check_every` iterations.

## Figures in the style of Clausnitzer et al. 2023

The report stage writes, next to the bar charts, three figures that use only
the through-thickness direction (`voxel.through_thickness_axis`):

| file | content | paper |
|---|---|---|
| `figures/clausnitzer_fig2_connectivity_tortuosity` | connectivity of SE (LLZTO) and CAM (LCO) as triangles, 1/tau as dashed lines, vs SVF_LCO = V_LCO/(V_LCO+V_LLZTO) | Fig. 2 |
| `figures/clausnitzer_fig3_effective_conductivity` | normalised effective ionic / electronic conductivity sigma_eff/sigma = eps/tau vs phase fraction | Fig. 3 |
| `figures/clausnitzer_fig7_tortuosity_active_area` | tau of the SE and active LCO-LLZTO area per cathode volume (1/cm) vs SVF_LCO; sinter density = 1 - porosity in the title | Fig. 7 |

Samples are sorted by SVF_LCO and labelled by name. A phase that does not
percolate is drawn at 1/tau = 0. tau is the tortuosity factor of taufactor
(tau = eps D0/D_eff). Re-create only the figures with
`... run --config <cfg> --stage report --force`.

## Tests

```bash
python -m pytest tests -q
```

The end-to-end test builds a synthetic phantom (LCO matrix, LLZTO spheres,
straight pore channels, one closed pore) and checks segmentation accuracy,
connectivity, sphere surface area, tortuosity, grain count and size, profiles,
report and the resume logic.
