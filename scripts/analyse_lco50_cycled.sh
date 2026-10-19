#!/usr/bin/env bash
# Full crack analysis of an LCO50 cycled stack, as done for LCO70 cycled, with the LCO50 model:
#   1. segmentation with rf_cracks_lco50_v2 + common post-processing chain + 3D crack growth + overlay stacks
#      (scripts/segment_lco50_cycled.sh)
#   2. through-thickness (y) connectivity of LCO (cracks as void / filled) and LLZTO, 6-connectivity
#      (scripts/connectivity_y.py on the grown cracks, <OUT>/connectivity_y_3d.json)
#   3. dimetric 3D rendering, LCO semi-transparent and cracks red (scripts/render_cracks_3d.py,
#      <OUT>/render_cracks_3d.png; needs pyvista: pip install pyvista)
#   4. optional: y tortuosity / effective conductivity of LCO with and without cracks
#      (TORTUOSITY=1, scripts/tortuosity_y.py, needs taufactor + torch; DOWNSAMPLE=2 by default)
#   cd ~/Documents/FIB-data && conda activate fibseg
#   bash scripts/analyse_lco50_cycled.sh [INPUT_DIR] [OUTPUT_DIR]
# Variables: VOXEL="x y z" voxel size in um (default "0.02 0.02 0.05": 20 nm pixels, 50 nm slices - adjust!),
#            COLLECTOR=top|bottom (image edge of the current collector, default top), SIZE (render edge in
#            voxels, 0 = whole stack), SKIP_SEG=1 to reuse an existing segmentation, TORTUOSITY=1, DOWNSAMPLE.
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:-/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_1_BSE_clean}"
OUT="${2:-/Users/jehad/Desktop/work/paper 2/LCO50_cycled/ds_1_BSE_clean_segmented}"
VOXEL="${VOXEL:-0.02 0.02 0.05}"
COLLECTOR="${COLLECTOR:-top}"
if [ "${SKIP_SEG:-0}" != "1" ]; then
    bash scripts/segment_lco50_cycled.sh "$IN" "$OUT"
fi
echo "== connectivity in y (grown cracks) -> $OUT/connectivity_y_3d.json"
python scripts/connectivity_y.py "$OUT" --tag 3d --collector "$COLLECTOR" --out "$OUT/connectivity_y_3d.json"
echo "== 3D rendering -> $OUT/render_cracks_3d.png"
python scripts/render_cracks_3d.py "$OUT" --tag 3d --size "${SIZE:-0}" --lco-step 1 --lco-decimate 0.8 \
    --voxel-um $VOXEL --out "$OUT/render_cracks_3d.png"
if [ "${TORTUOSITY:-0}" = "1" ]; then
    echo "== tortuosity in y (grown cracks, downsample ${DOWNSAMPLE:-2}) -> $OUT/tortuosity_y_3d.json"
    python scripts/tortuosity_y.py "$OUT" --tag 3d --collector "$COLLECTOR" --downsample "${DOWNSAMPLE:-2}" \
        --phases lco --out "$OUT/tortuosity_y_3d.json"
fi
echo "== done: $OUT"
