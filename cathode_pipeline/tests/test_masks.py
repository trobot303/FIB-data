"""Import of pre-segmented binary stacks (sample ``masks`` instead of ``pattern``).

The phantom labels are written as three multipage binary TIFFs. To mimic real
exports, ~1 % of the voxels are removed from every mask (covered by nothing)
and a small region is claimed by two masks at once.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import tifffile
import yaml

from cathode_analysis import cli
from cathode_analysis import io as cio
from cathode_analysis.config import ConfigError, load_config, resolve_sample, validate
from tests.conftest import make_phantom_labels

ROOT = Path(__file__).resolve().parents[1]
SEGMENTED = ROOT / "config" / "workstation_segmented.yaml"

OVERLAP = ((5, 9), (10, 20), (20, 30))   # z, y, x block present in both the LCO and the LLZTO mask


@pytest.fixture(scope="session")
def mask_phantom(tmp_path_factory) -> dict:
    root = tmp_path_factory.mktemp("mask_phantom")
    data = root / "segmented_PHANTOM"
    data.mkdir()
    truth = make_phantom_labels()
    rng = np.random.default_rng(1)
    holes = rng.random(truth.shape) < 0.01
    masks = {"pore": truth == 0, "lco": truth == 1, "llzto": truth == 2}
    (z0, z1), (y0, y1), (x0, x1) = OVERLAP
    masks["llzto"] = masks["llzto"].copy()
    masks["llzto"][z0:z1, y0:y1, x0:x1] = True          # overlaps the LCO matrix there
    for name, m in masks.items():
        m = m & ~holes
        tifffile.imwrite(data / f"{name}.tiff", (m.astype(np.uint8) * 255), photometric="minisblack")

    with open(SEGMENTED, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {"data_root": ".", "output_root": "results", "log_dir": "results/logs"}
    cfg["runtime"]["n_workers"] = 2
    cfg["runtime"]["device"] = "cpu"
    cfg["analysis_volume"] = {"shape_zyx": list(truth.shape), "enforce": True}
    cfg["samples"] = [{"name": "PHANTOM", "nominal_lco_vol_fraction": None, "input_dir": "segmented_PHANTOM",
                       "masks": {"pore": "pore.tiff", "lco": "lco.tiff", "llzto": "llzto.tiff"},
                       "slice_range": [None, None],
                       "crop": {"z": [0, truth.shape[0]], "y": [0, truth.shape[1]], "x": [0, truth.shape[2]]},
                       "downsample": 1, "overrides": {}}]
    cfg["voxel"]["size_nm"] = 100.0
    cfg["segmentation"]["masks"] = {"priority": ["lco", "llzto", "pore"], "unassigned": "majority",
                                    "postprocess": False}
    cfg["segmentation"]["postprocess"]["chunk_z"] = 16
    cfg["segmentation"]["postprocess"]["chunk_margin"] = 6
    cfg["surface"]["chunk_z"] = 16
    cfg["tortuosity"]["taufactor"]["max_iterations"] = 3000
    cfg["tortuosity"]["downsample"] = 1
    cfg["grains"]["downsample"] = 1
    cfg["grains"]["min_grain_voxels"] = 30
    cfg["profiles"]["slab_thickness_um"] = 0.8
    cfg["plotting"]["sample_colors"] = {"PHANTOM": "#4477aa"}
    cfg_path = root / "phantom_masks.yaml"
    with open(cfg_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg, fh, sort_keys=False)
    return {"root": root, "config": cfg_path, "truth": truth, "holes": holes,
            "results": root / "results" / "PHANTOM", "output_root": root / "results"}


# --------------------------------------------------------------------------- #
# config
# --------------------------------------------------------------------------- #
def test_segmented_config_validates_and_resolves():
    cfg = load_config(SEGMENTED)
    s = resolve_sample(cfg, "LCO50")
    assert s["sample"]["input_mode"] == "masks"
    assert sorted(s["sample"]["masks"]) == ["lco", "llzto", "pore"]
    assert s["segmentation"]["masks"]["unassigned"] == "majority"


def test_grayscale_sample_gets_mask_defaults():
    cfg = load_config(ROOT / "config" / "default.yaml")
    s = resolve_sample(cfg, "LCO50")
    assert s["sample"]["input_mode"] == "grayscale"
    assert s["segmentation"]["masks"] == {"priority": ["pore", "lco", "llzto"], "unassigned": "majority",
                                          "postprocess": False}


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda c: c["samples"][0].pop("masks"), "exactly one"),
        (lambda c: c["samples"][0].update(pattern="*.tif"), "exactly one"),
        (lambda c: c["samples"][0]["masks"].pop("lco"), "every phase"),
        (lambda c: c["segmentation"]["masks"].update(priority=["lco", "lco", "pore"]), "priority"),
        (lambda c: c["segmentation"]["masks"].update(unassigned="crack"), "unassigned"),
        (lambda c: c["segmentation"]["masks"].update(postprocess="yes"), "postprocess"),
    ],
)
def test_invalid_mask_configs_are_rejected(mutate, match):
    with open(SEGMENTED, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    mutate(raw)
    with pytest.raises(ConfigError, match=match):
        validate(raw)


def test_validate_and_inspect_commands_on_segmented_config(capsys):
    assert cli.main(["validate", "--config", str(SEGMENTED)]) == 0
    assert "masks=" in capsys.readouterr().out


# --------------------------------------------------------------------------- #
# stage
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def imported(mask_phantom):
    rc = cli.main(["run", "--config", str(mask_phantom["config"]), "--stage", "segment", "--force"])
    assert rc == 0
    return mask_phantom


def test_masks_are_combined_into_labels(imported):
    truth, holes = imported["truth"], imported["holes"]
    labels = np.asarray(cio.load_labels(imported["results"] / "labels.npy"))
    assert labels.shape == truth.shape and labels.dtype == np.uint8
    assert set(np.unique(labels)) <= {0, 1, 2}
    # voxels covered by exactly one mask are copied verbatim
    (z0, z1), (y0, y1), (x0, x1) = OVERLAP
    clean = ~holes.copy()
    clean[z0:z1, y0:y1, x0:x1] = False
    assert (labels[clean] == truth[clean]).all()
    # the overlap goes to the first phase in priority (lco)
    assert (labels[z0:z1, y0:y1, x0:x1] == 1).all()
    # scattered uncovered voxels are recovered from their neighbours
    assert (labels[holes] == truth[holes]).mean() > 0.97


def test_segment_json_reports_mask_statistics(imported):
    seg = json.loads((imported["results"] / "segment.json").read_text())
    assert seg["mode"] == "masks"
    assert seg["thresholds_uint8"] is None and seg["histogram"] is None
    m = seg["masks"]
    assert m["priority"] == ["lco", "llzto", "pore"]
    assert m["unassigned_rule"] == "majority"
    assert 0.005 < m["unassigned_fraction"] < 0.015
    assert m["overlap_voxels"]["lco"] > 0 and m["overlap_voxels"]["pore"] == 0
    truth = imported["truth"]
    for p, code in (("pore", 0), ("lco", 1), ("llzto", 2)):
        assert abs(seg["volume_fractions"][p] - (truth == code).mean()) < 0.01
    assert any(p.name.startswith("phases_z") for p in (imported["results"] / "qc").iterdir())


def test_fixed_phase_rule_for_unassigned(mask_phantom, tmp_path):
    with open(mask_phantom["config"], encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["paths"] = {"data_root": str(mask_phantom["root"]), "output_root": str(tmp_path / "res"),
                    "log_dir": str(tmp_path / "res" / "logs")}
    cfg["segmentation"]["masks"]["unassigned"] = "pore"
    cfg_path = tmp_path / "fixed.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    assert cli.main(["run", "--config", str(cfg_path), "--stage", "segment"]) == 0
    labels = np.asarray(cio.load_labels(tmp_path / "res" / "PHANTOM" / "labels.npy"))
    assert (labels[mask_phantom["holes"]] == 0).all()


def test_downstream_stages_run_on_imported_labels(imported):
    rc = cli.main(["run", "--config", str(imported["config"]), "--stage", "all"])
    assert rc == 0
    status = json.loads((imported["results"] / "stage_status.json").read_text())
    assert set(status) >= {"segment", "connectivity", "surface", "tortuosity", "grains", "profiles"}
    conn = json.loads((imported["results"] / "connectivity.json").read_text())["phases"]
    assert conn["lco"]["percolates_collector_to_separator"]
    summary = (imported["output_root"] / "summary.csv").read_text()
    assert "PHANTOM" in summary and "segmentation_mode" in summary
