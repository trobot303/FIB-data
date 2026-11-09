"""Synthetic phantom shared by the end-to-end tests.

Phantom (z, y, x) = (40, 64, 72), y = through-thickness axis:
  * LCO matrix everywhere (grey 30000)
  * LLZTO: 6 isolated spheres of radius 6 (grey 60000) -> not connected to the separator
  * pores: 3 straight square channels along y (grey 1500) -> open, percolating, tau_y = 1
  * one closed pore cube not touching any face
Written as a uint16 TIFF sequence plus a config YAML pointing at it.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "config" / "default.yaml"

SHAPE = (40, 64, 72)
SPHERE_CENTRES = [(10, 16, 14), (10, 46, 14), (28, 16, 40), (28, 46, 40), (14, 32, 60), (30, 32, 20)]
SPHERE_R = 6
CHANNELS = [(4, 4), (4, 30), (30, 62)]   # (z0, x0) of 6x6 channels running the whole y extent
CHANNEL_W = 6
CLOSED_PORE = ((20, 26), (54, 60), (2, 8))  # z, y, x ranges


def make_phantom_labels() -> np.ndarray:
    lab = np.ones(SHAPE, dtype=np.uint8)  # LCO
    zz, yy, xx = np.indices(SHAPE)
    for cz, cy, cx in SPHERE_CENTRES:
        lab[(zz - cz) ** 2 + (yy - cy) ** 2 + (xx - cx) ** 2 <= SPHERE_R ** 2] = 2
    for z0, x0 in CHANNELS:
        lab[z0:z0 + CHANNEL_W, :, x0:x0 + CHANNEL_W] = 0
    (z0, z1), (y0, y1), (x0, x1) = CLOSED_PORE
    lab[z0:z1, y0:y1, x0:x1] = 0
    return lab


@pytest.fixture(scope="session")
def phantom(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("phantom")
    data = root / "processed_PHANTOM"
    data.mkdir()
    lab = make_phantom_labels()
    rng = np.random.default_rng(0)
    grey = np.select([lab == 0, lab == 1, lab == 2], [1500, 30000, 60000]).astype(np.float32)
    grey += rng.normal(0, 1500, size=lab.shape).astype(np.float32)
    grey = np.clip(grey, 0, 65535).astype(np.uint16)
    for i in range(SHAPE[0]):
        tifffile.imwrite(data / f"slice_{i + 1:04d}.tif", grey[i])

    with open(DEFAULT, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {"data_root": ".", "output_root": "results", "log_dir": "results/logs"}
    cfg["runtime"]["n_workers"] = 2
    cfg["runtime"]["device"] = "cpu"
    cfg["samples"] = [{"name": "PHANTOM", "nominal_lco_vol_fraction": None, "input_dir": "processed_PHANTOM",
                       "pattern": "slice_*.tif", "slice_range": [None, None],
                       "crop": {"z": [None, None], "y": [None, None], "x": [None, None]}, "downsample": 1,
                       "overrides": {}}]
    cfg["voxel"]["size_nm"] = 100.0
    cfg["segmentation"]["postprocess"]["min_object_voxels"] = {"pore": 8, "lco": 8, "llzto": 8}
    cfg["segmentation"]["postprocess"]["fill_holes"] = {"enabled": True, "max_voxels": 100}
    cfg["segmentation"]["postprocess"]["chunk_z"] = 16
    cfg["segmentation"]["postprocess"]["chunk_margin"] = 6
    cfg["surface"]["chunk_z"] = 16
    cfg["tortuosity"]["taufactor"]["max_iterations"] = 3000
    cfg["grains"]["method"] = "watershed_hmax"
    cfg["grains"]["min_grain_voxels"] = 30
    cfg["profiles"]["slab_thickness_um"] = 0.8
    cfg["plotting"]["sample_colors"] = {"PHANTOM": "#4477aa"}
    cfg_path = root / "phantom.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return {"root": root, "config": cfg_path, "labels": lab, "results": root / "results" / "PHANTOM",
            "output_root": root / "results"}
