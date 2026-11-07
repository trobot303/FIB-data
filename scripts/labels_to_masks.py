"""Split a labels_stack.tif (1 = pore, 2 = LCO, 3 = LLZTO) into the three binary mask stacks
(pores.tiff, LCO.tiff, LLZTO.tiff, non-zero = phase) that cathode_analysis' mask mode reads.

    python scripts/labels_to_masks.py data/segmented/LCO70/labels_stack.tif  [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import tifffile

PHASES = {1: "pores", 2: "LCO", 3: "LLZTO"}

ap = argparse.ArgumentParser()
ap.add_argument("labels")
ap.add_argument("--out", default=None, help="output folder (default: next to the labels file)")
a = ap.parse_args()
src = Path(a.labels)
out = Path(a.out) if a.out else src.parent
out.mkdir(parents=True, exist_ok=True)
lab = tifffile.imread(str(src))
print(f"{src}: {lab.shape} {lab.dtype}, codes {np.unique(lab)}")
for k, name in PHASES.items():
    m = (lab == k).astype(np.uint8) * 255
    tifffile.imwrite(str(out / f"{name}.tiff"), m, compression="zlib", photometric="minisblack")
    print(f"  {name}.tiff: fraction {m.mean() / 255:.4f}")
