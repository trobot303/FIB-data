#!/usr/bin/env bash
# Segment the LCO70 cycled stack (pore / LCO / LLZTO / crack) on the Mac with the crack model.
#   conda activate fibseg && bash scripts/segment_lco70_cycled.sh [INPUT_DIR] [OUTPUT_DIR]
# Destriping, smoothing and phase normalisation are stored in the model and applied automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:-/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/Merged_clean}"
OUT="${2:-/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/Merged_clean_segmented}"
python -m fib_rf_seg predict --model models/rf_cracks_v5.joblib --input "$IN" --out "$OUT" \
    --min-size 6 --per-phase --stack
