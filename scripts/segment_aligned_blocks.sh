#!/usr/bin/env bash
# Apply the LCO70-cycled crack model (rf_cracks_v5) to the aligned, padding-free z-blocks with the same
# post-processing chain as LCO50 cycled (scripts/segment_lco50_cycled.sh):
#   island removal < 6 px, crack shape filter (--crack-min-length 12 --crack-min-aspect 3), then 3D growth of
#   the cracks along their length and through the slices (scripts/grow_cracks_3d.py, probability > LOW = 0.2).
#   conda activate fibseg && bash scripts/segment_aligned_blocks.sh ["/Users/jehad/Desktop/work/paper 2/LCO70_cycled/Merged_clean_aligned"]
# For every block*/ folder: reads block*/raw/slice-XXXX.tif and writes
#   block*/labels_v3/slice-XXXX_labels.tif     per-slice model output after the filters (1 pore, 2 LCO, 3 LLZTO, 4 crack)
#   block*/labels_v3_stack.tif, <phase>_labels_v3_stack.tif, labels_v3_fractions.csv
#   block*/crack_proba/slice-XXXX_crackp.tif   crack probability (uint8 x255)
#   block*/labels_v3_3d/slice-XXXX_labels.tif  after 3D crack growth  <- use these for the crack analysis
#   block*/labels_v3_3d_stack.tif, crack_v3_3d_stack.tif (and the other phases), fractions_v3_3d.csv
#   block*/overlay_v3_3d/overlay_stack.tif     RGB overlay of the grown cracks on the SEM slices
# The earlier labels_v2 results (no shape filter, no growth) are left untouched.
# Variables: MODEL (default models/rf_cracks_v5.joblib), LOW (0.2), RESUME=1 to keep existing labels_v3 slices.
# MIN_LENGTH (default 25 px = 0.5 um): after growth, drop cracks whose longest in-plane piece is shorter
# (spots between particles); a short piece of a crack that is long in another slice is kept.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="${1:-/Users/jehad/Desktop/work/paper 2/LCO70_cycled/Merged_clean_aligned}"
MODEL="${MODEL:-models/rf_cracks_v5.joblib}"
L="${LABELS:-labels_v3}"
OW="--overwrite"; [ "${RESUME:-0}" = "1" ] && OW=""
echo "== model: $MODEL; labels: $L; growth threshold: ${LOW:-0.2}"
for b in "$BASE"/block*/; do
    [ -d "$b/raw" ] || { echo "skip $b (no raw/ folder)"; continue; }
    echo "== $b"
    python -m fib_rf_seg predict --model "$MODEL" --input "$b/raw" --out "$b" \
        --labels-subdir "$L" --min-size 6 --crack-min-length 12 --crack-min-aspect 3 --stack --crack-proba $OW
    python scripts/grow_cracks_3d.py --labels "$b/$L" --proba "$b/crack_proba" --out "$b/${L}_3d" --low "${LOW:-0.2}" --min-length "${MIN_LENGTH:-25}" --tag "${L#labels_}_3d"
    python scripts/overlay_stack.py --raw "$b/raw" --labels "$b/${L}_3d" --out "$b/overlay_${L#labels_}_3d" --no-slices
done
echo "== done. Crack fractions per block: $BASE/block*/fractions_${L#labels_}_3d.csv"
