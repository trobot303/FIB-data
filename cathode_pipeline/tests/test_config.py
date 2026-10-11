import copy
from pathlib import Path

import pytest
import yaml

from cathode_analysis.config import ConfigError, face_to_axis_side, load_config, resolve_sample, validate

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "config" / "default.yaml"


@pytest.fixture
def raw():
    with open(DEFAULT, encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def test_default_config_validates(raw):
    validate(raw)


def test_load_resolves_paths():
    cfg = load_config(DEFAULT)
    assert Path(cfg["paths"]["data_root"]).is_absolute()
    assert Path(cfg["paths"]["output_root"]).is_absolute()
    assert cfg["_meta"]["config_path"] == str(DEFAULT)


def test_resolve_sample_applies_overrides():
    cfg = load_config(DEFAULT)
    cfg["samples"][0]["overrides"] = {"segmentation": {"thresholds": {"method": "manual", "manual": [60, 150]}}}
    s = resolve_sample(cfg, cfg["samples"][0]["name"])
    assert s["segmentation"]["thresholds"]["method"] == "manual"
    assert s["segmentation"]["normalize"]["method"] == cfg["segmentation"]["normalize"]["method"]
    assert s["sample"]["output_dir"].endswith(cfg["samples"][0]["name"])
    assert "overrides" not in s["sample"]
    # the shared config is untouched
    assert cfg["segmentation"]["thresholds"]["method"] == "multi_otsu"


@pytest.mark.parametrize(
    "patch, match",
    [
        ({"schema_version": 99}, "schema_version"),
        ({"voxel": {"through_thickness_axis": "w"}}, "through_thickness_axis"),
        ({"segmentation": {"intensity_order": ["pore", "lco", "lco"]}}, "intensity_order"),
        ({"segmentation": {"thresholds": {"method": "manual", "manual": [None, None]}}}, "manual"),
        ({"connectivity": {"reference_faces": {"lco": ["top"]}}}, "bad face"),
        ({"tortuosity": {"directions": ["q"]}}, "bad axis"),
        ({"surface": {"interfaces": [["lco", "lco"]]}}, "identical"),
    ],
)
def test_invalid_configs_are_rejected(raw, patch, match):
    from cathode_analysis.config import _deep_update

    bad = _deep_update(copy.deepcopy(raw), patch)
    with pytest.raises(ConfigError, match=match):
        validate(bad)


def test_duplicate_sample_names_rejected(raw):
    raw["samples"].append(copy.deepcopy(raw["samples"][0]))
    with pytest.raises(ConfigError, match="duplicate"):
        validate(raw)


def test_face_aliases(raw):
    assert face_to_axis_side("collector", raw) == ("y", "min")
    assert face_to_axis_side("separator", raw) == ("y", "max")
    assert face_to_axis_side("z_max", raw) == ("z", "max")


def test_analysis_volume_enforced(raw):
    raw["analysis_volume"] = {"shape_zyx": [100, 200, 300], "enforce": True}
    for s in raw["samples"]:
        s["crop"] = {"z": [0, 100], "y": [10, 210], "x": [5, 305]}
    validate(raw)
    raw["samples"][1]["crop"]["x"] = [5, 300]
    with pytest.raises(ConfigError, match="analysis_volume"):
        validate(raw)
    raw["samples"][1]["crop"]["x"] = [None, None]
    with pytest.raises(ConfigError, match="explicit"):
        validate(raw)
