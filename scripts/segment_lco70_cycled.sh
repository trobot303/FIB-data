#!/usr/bin/env bash
# Segment the LCO70 cycled stack (pore / LCO / LLZTO / crack) on the Mac with the crack model rf_cracks_v6,
# with the same post-processing chain as LCO50 cycled: island removal < 6 px, crack shape filter
# (--crack-min-length 12 --crack-min-aspect 3), 3D crack growth (scripts/grow_cracks_3d.py, LOW = 0.15).
# rf_cracks_v6: v3 recipe + --ignore-unmarked 0.2 on the three annotated LCO70 cycled slices (conservative);
# MODEL=models/rf_cracks_v6_all.joblib adds the four LCO50 slices (finds more cracks, more false alarms in the
# fine-particle regions and at LLZTO).  Leave-one-slice-out, full 3D chain: share of the annotated crack found
# 0.91 (old recipe) -> 0.95 (v6) / 0.98 (v6_all); crack pixels per slice 1740 -> 2600 / 5570.
# LOW = 0.2 reproduces the earlier growth threshold.
#   conda activate fibseg && bash scripts/segment_lco70_cycled.sh [INPUT_DIR] [OUTPUT_DIR]
# Destriping, smoothing and phase normalisation are stored in the model and applied automatically.
# Output layout as in scripts/segment_lco50_cycled.sh (labels/ = per-slice model, labels_3d/ = grown cracks,
# use the *_3d results for the crack analysis).  Existing labels are recomputed; RESUME=1 keeps them.
# MIN_LENGTH (default 25 px = 0.5 um): after growth, drop cracks whose longest in-plane piece is shorter
# (spots between particles); a short piece of a crack that is long in another slice is kept.
# REFINE_WIDTH (default 1): trim the cracks to the dark core of the line (full width at half depth),
# so the crack width is defined the same way in every sample; REFINE_WIDTH=0 keeps the detected band.
# SLICES="FIRST LAST" (e.g. SLICES="1 600"): segment only these slices, counted 1-based in file-name order
# (inclusive).  They are linked into <OUTPUT_DIR>/input_slices and the whole chain (model, 3D growth, overlays)
# runs on that subset.  Without OUTPUT_DIR the default output folder gets the suffix _slices<FIRST>-<LAST>,
# so the 3D growth does not pick up labels of a full-stack run.
set -euo pipefail
cd "$(dirname "$0")/.."
IN="${1:-/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/Merged_clean}"
OUT="${2:-/Users/jehad/Desktop/work/PlasmaFIB_Data/A970_7030_cycled/try2/Dataset 2/Merged_clean_segmented}"
MODEL="${MODEL:-models/rf_cracks_v6.joblib}"
if [ -n "${SLICES:-}" ]; then
    read -r FIRST LAST <<< "$SLICES"
    [ $# -lt 2 ] && OUT="${OUT}_slices${FIRST}-${LAST}"
    SUB="$OUT/input_slices"
    python - "$IN" "$SUB" "$FIRST" "$LAST" <<'EOF'
import sys, os
from pathlib import Path
from fib_rf_seg.io import list_slices
src, sub, first, last = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4])
files = list_slices(src)
if not 1 <= first <= last:
    sys.exit(f"SLICES: need 1 <= FIRST <= LAST, got {first} {last}")
if last > len(files):
    print(f"== SLICES: only {len(files)} slices in {src}; using {first}-{len(files)}")
sel = files[first - 1:last]
if not sel:
    sys.exit(f"SLICES: no slices selected ({len(files)} slices in {src})")
sub.mkdir(parents=True, exist_ok=True)
for p in sub.iterdir():
    if p.is_symlink():
        p.unlink()
for p in sel:
    os.symlink(p.resolve(), sub / p.name)
print(f"== SLICES {first}-{first + len(sel) - 1} of {len(files)}: {sel[0].name} ... {sel[-1].name}")
EOF
    IN="$SUB"
fi
OW="--overwrite"; [ "${RESUME:-0}" = "1" ] && OW=""
echo "== model: $MODEL; growth threshold: ${LOW:-0.15}; output: $OUT"
python -m fib_rf_seg predict --model "$MODEL" --input "$IN" --out "$OUT" \
    --min-size 6 --crack-min-length 12 --crack-min-aspect 3 --per-phase --stack --crack-proba $OW
python scripts/grow_cracks_3d.py --labels "$OUT/labels" --proba "$OUT/crack_proba" --out "$OUT/labels_3d" --low "${LOW:-0.15}" --min-length "${MIN_LENGTH:-25}" --raw "$IN" $( [ "${REFINE_WIDTH:-1}" = "1" ] && echo --refine-width )
python scripts/overlay_stack.py --raw "$IN" --labels "$OUT/labels_3d" --out "$OUT/overlay_3d"
