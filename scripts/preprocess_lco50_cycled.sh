#!/bin/bash
# Preprocess the LCO50 cycled BSE stack on the Mac: FFT curtaining removal with the
# hand-picked mask preprocessing/fft_mask.npy (picked on this dataset, 1001x1001 px).
# If a registered SE stack is given as 2nd argument, BSE and SE are blended first
# (75 % BSE / 25 % SE, merge_bse_se.py) and the blend is cleaned instead.
#
#   cd ~/Documents/FIB-data && conda activate fibseg
#   bash scripts/preprocess_lco50_cycled.sh                      # ds_1_BSE -> ds_1_BSE_clean
#   bash scripts/preprocess_lco50_cycled.sh "<BSE dir>" "<SE dir>"   # merge, then clean -> <BSE dir>_clean
#   bash scripts/preprocess_lco50_cycled.sh "<BSE dir>" "" "<out dir>"
#
# To pick a new mask for another dataset:  python preprocessing/select_fft_filter.py --image-path <slice> --mask-out preprocessing/fft_mask.npy
# To use the automatic Gaussian notch instead of the mask, add:  MASK="" bash scripts/preprocess_lco50_cycled.sh
set -e
cd "$(dirname "$0")/.."
BSE="${1:-/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_1_BSE}"
SE="${2:-}"
OUT="${3:-${BSE}_clean}"
MASK="${MASK-preprocessing/fft_mask.npy}"
if [ ! -d "$BSE" ]; then echo "BSE folder not found: $BSE"; exit 1; fi
IN="$BSE"
if [ -n "$SE" ]; then
  IN="${BSE}_merged"
  echo "== merging BSE + SE -> $IN"
  python preprocessing/merge_bse_se.py --bse-dir "$BSE" --se-dir "$SE" --out-dir "$IN" --bse-weight 0.75
fi
echo "== removing curtaining: $IN -> $OUT"
python preprocessing/remove_curtaining.py --in-dir "$IN" --out-dir "$OUT" --mask-path "$MASK" --bit-depth 16
echo "== done: $OUT"
