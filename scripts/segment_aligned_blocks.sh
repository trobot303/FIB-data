#!/usr/bin/env bash
# Apply the LCO70-cycled crack model (rf_cracks_v5) to the aligned, padding-free z-blocks.
#   conda activate fibseg && bash scripts/segment_aligned_blocks.sh ["/Users/jehad/Desktop/work/paper 2/LCO70_cycled/Merged_clean_aligned"]
# For every block*/ folder: reads block*/raw/slice-XXXX.tif and writes
#   block*/labels_v2/slice-XXXX_labels.tif   (uint8: 1 pore, 2 LCO, 3 LLZTO, 4 crack; same size as the raw slice)
#   block*/labels_v2_stack.tif               (multipage stack of the block)
#   block*/labels_v2_fractions.csv           (phase fractions per slice)
# Destriping, smoothing and phase normalisation are stored in the model and applied automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
BASE="${1:-/Users/jehad/Desktop/work/paper 2/LCO70_cycled/Merged_clean_aligned}"
MODEL="${MODEL:-models/rf_cracks_v5.joblib}"
for b in "$BASE"/block*/; do
    [ -d "$b/raw" ] || { echo "skip $b (no raw/ folder)"; continue; }
    echo "== $b"
    python -m fib_rf_seg predict --model "$MODEL" --input "$b/raw" --out "$b" \
        --labels-subdir labels_v2 --min-size 6 --stack
done
