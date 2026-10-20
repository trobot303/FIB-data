#!/usr/bin/env bash
# Full crack analysis of LCO50 cycled ds_2 (BSE/SE merged, FFT-cleaned, cropped to 1001 x 1001 px) with the
# ds_2 crack model rf_cracks_lco50_ds2_v1 (trained on the annotated ds_2 slice 0364 plus the four annotated
# ds_1 slices; crack weight 8 for the very thin ds_2 cracks).  Same chain as for ds_1 and LCO70 cycled:
# shape filter, 3D growth, 25 px length limit, width refinement, overlays, y-connectivity, 3D rendering.
#   cd ~/Documents/FIB-data && conda activate fibseg
#   bash scripts/analyse_lco50_cycled_ds2.sh [INPUT_DIR] [OUTPUT_DIR]
# Voxel 20 nm isotropic (dataset_info_5050.hdr).  Other variables as in scripts/analyse_lco50_cycled.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:-/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_2_Merged_clean_1001}"
OUT="${2:-/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_2_Merged_clean_1001_segmented}"
MODEL="${MODEL:-models/rf_cracks_lco50_ds2_v1.joblib}" bash scripts/analyse_lco50_cycled.sh "$IN" "$OUT"
