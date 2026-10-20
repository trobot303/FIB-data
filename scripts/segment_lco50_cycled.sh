#!/usr/bin/env bash
# Segment the LCO50 cycled stack (pore / LCO / LLZTO / crack) on the Mac with the crack model
# trained on the annotated LCO50 cycled slices 045/053/262/272 plus the LCO70 cycled slices 15/22/38
# (models/rf_cracks_lco50_v2.joblib). No --drop-vertical: this sample has near-vertical cracks and the
# model (oriented features + curtaining negatives) separates them from curtaining itself.
#   cd ~/Documents/FIB-data && conda activate fibseg
#   bash scripts/segment_lco50_cycled.sh [INPUT_DIR] [OUTPUT_DIR]
# Input: the curtaining-corrected slices (scripts/preprocess_lco50_cycled.sh output).
# Destriping, smoothing and phase normalisation are stored in the model and applied automatically.
# --crack-min-length 12 --crack-min-aspect 3 drop crack components shorter than 12 px or not
# line-like (dots/blobs from curtaining and noise); remove them if short cracks are expected.
# Output: <OUT>/labels/<slice>_labels.tif (1 pore, 2 LCO, 3 LLZTO, 4 crack), <OUT>/labels_stack.tif,
#         <OUT>/<phase>/ and <OUT>/<phase>_stack.tif binary masks (255 = phase), fractions csv/json
#         = the per-slice model output; <OUT>/labels_3d/, labels_3d_stack.tif, <phase>_3d_stack.tif and
#         fractions_3d.csv = after growing the cracks along their length and through neighbouring slices
#         (scripts/grow_cracks_3d.py, crack probability > LOW, default 0.2, connected to a detected crack).
#         Use the *_3d results for the crack analysis.
# Existing label files in OUT are recomputed (so a change of model takes effect); set RESUME=1 to skip
# slices already done, e.g. after an interrupted run.  MODEL=models/rf_cracks_lco50_v1.joblib selects
# the earlier model (trained on slices 262/272 only).
# MIN_LENGTH (default 25 px = 0.5 um): after growth, drop cracks whose longest in-plane piece is shorter
# (spots between particles); a short piece of a crack that is long in another slice is kept.
# REFINE_WIDTH (default 1): trim the cracks to the dark core of the line (full width at half depth),
# so the crack width is defined the same way in every sample; REFINE_WIDTH=0 keeps the detected band.
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:-/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_1_BSE_clean}"
OUT="${2:-/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_1_BSE_clean_segmented}"
OW="--overwrite"; [ "${RESUME:-0}" = "1" ] && OW=""
echo "== model: ${MODEL:-models/rf_cracks_lco50_v2.joblib}"
python -m fib_rf_seg predict --model "${MODEL:-models/rf_cracks_lco50_v2.joblib}" --input "$IN" --out "$OUT" \
    --min-size 6 --crack-min-length 12 --crack-min-aspect 3 --per-phase --stack --crack-proba $OW
echo "== overlays (every 50th slice) -> $OUT/overlays"
python scripts/make_overlays.py "$OUT" --raw "$IN" --labels labels --out "$OUT/overlays" --phases crack --every 50 --voxel-um 0.02
echo "== growing cracks along their length and through the slices (3D hysteresis, low ${LOW:-0.2}) -> $OUT/labels_3d"
python scripts/grow_cracks_3d.py --labels "$OUT/labels" --proba "$OUT/crack_proba" --out "$OUT/labels_3d" --low "${LOW:-0.2}" --min-length "${MIN_LENGTH:-25}" --raw "$IN" $( [ "${REFINE_WIDTH:-1}" = "1" ] && echo --refine-width )
echo "== RGB overlay stacks (cracks in red on the SEM slices) -> $OUT/overlay (per-slice model) and $OUT/overlay_3d (grown)"
python scripts/overlay_stack.py --raw "$IN" --labels "$OUT/labels" --out "$OUT/overlay" --no-slices
python scripts/overlay_stack.py --raw "$IN" --labels "$OUT/labels_3d" --out "$OUT/overlay_3d"
