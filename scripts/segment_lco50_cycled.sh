#!/usr/bin/env bash
# Segment the LCO50 cycled stack (pore / LCO / LLZTO / crack) on the Mac with the crack model
# trained on LCO70 cycled (models/rf_cracks_v5.joblib, the latest version).
#   cd ~/Documents/FIB-data && conda activate fibseg
#   bash scripts/segment_lco50_cycled.sh [INPUT_DIR] [OUTPUT_DIR]
# Input: the curtaining-corrected slices (scripts/preprocess_lco50_cycled.sh output).
# Destriping, smoothing and phase normalisation are stored in the model and applied automatically.
# --drop-vertical 20 relabels crack components within 20 deg of vertical (residual curtaining
# streaks) as the surrounding phase; --crack-min-length 12 --crack-min-aspect 3 drop crack components
# shorter than 12 px or not line-like (dots/blobs from curtaining and noise). Remove them if short or
# vertical cracks are expected.
# Output: <OUT>/labels/<slice>_labels.tif (1 pore, 2 LCO, 3 LLZTO, 4 crack), <OUT>/labels_stack.tif,
#         <OUT>/<phase>/ and <OUT>/<phase>_stack.tif binary masks (255 = phase), fractions csv/json.
# The run resumes if interrupted (existing label files are skipped).
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:-/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_1_BSE_clean}"
OUT="${2:-/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_1_BSE_clean_segmented}"
python -m fib_rf_seg predict --model models/rf_cracks_v5.joblib --input "$IN" --out "$OUT" \
    --min-size 6 --drop-vertical 20 --crack-min-length 12 --crack-min-aspect 3 --per-phase --stack
echo "== overlays (every 50th slice) -> $OUT/overlays"
python scripts/make_overlays.py "$OUT" --raw "$IN" --labels labels --out "$OUT/overlays" --phases crack --every 50 --voxel-um 0.02
echo "== RGB overlay stack (cracks in red on the SEM slices) -> $OUT/overlay"
python scripts/overlay_stack.py --raw "$IN" --labels "$OUT/labels" --out "$OUT/overlay"
