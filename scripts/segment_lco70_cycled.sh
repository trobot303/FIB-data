#!/usr/bin/env bash
# Segment the LCO70 cycled stack (pore / LCO / LLZTO / crack) on the Mac with the crack model rf_cracks_v5,
# with the same post-processing chain as LCO50 cycled: island removal < 6 px, crack shape filter
# (--crack-min-length 12 --crack-min-aspect 3), 3D crack growth (scripts/grow_cracks_3d.py, LOW = 0.2).
#   conda activate fibseg && bash scripts/segment_lco70_cycled.sh [INPUT_DIR] [OUTPUT_DIR]
# Destriping, smoothing and phase normalisation are stored in the model and applied automatically.
# Output layout as in scripts/segment_lco50_cycled.sh (labels/ = per-slice model, labels_3d/ = grown cracks,
# use the *_3d results for the crack analysis).  Existing labels are recomputed; RESUME=1 keeps them.
# MIN_LENGTH (default 25 px = 0.5 um): after growth, drop cracks whose longest in-plane piece is shorter
# (spots between particles); a short piece of a crack that is long in another slice is kept.
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:-/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/Merged_clean}"
OUT="${2:-/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/Merged_clean_segmented}"
MODEL="${MODEL:-models/rf_cracks_v5.joblib}"
OW="--overwrite"; [ "${RESUME:-0}" = "1" ] && OW=""
echo "== model: $MODEL; growth threshold: ${LOW:-0.2}"
python -m fib_rf_seg predict --model "$MODEL" --input "$IN" --out "$OUT" \
    --min-size 6 --crack-min-length 12 --crack-min-aspect 3 --per-phase --stack --crack-proba $OW
python scripts/grow_cracks_3d.py --labels "$OUT/labels" --proba "$OUT/crack_proba" --out "$OUT/labels_3d" --low "${LOW:-0.2}" --min-length "${MIN_LENGTH:-25}"
python scripts/overlay_stack.py --raw "$IN" --labels "$OUT/labels_3d" --out "$OUT/overlay_3d"
