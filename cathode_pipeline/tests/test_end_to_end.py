"""Run every stage on the synthetic phantom and check the physics."""
import json
import math

import numpy as np
import pandas as pd
import pytest

from cathode_analysis import cli
from cathode_analysis import io as cio
from tests.conftest import SPHERE_CENTRES, SPHERE_R


@pytest.fixture(scope="session")
def ran(phantom):
    rc = cli.main(["run", "--config", str(phantom["config"]), "--stage", "all", "--force"])
    assert rc == 0
    return phantom


def _json(ran, name):
    return json.loads((ran["results"] / name).read_text())


def test_segmentation_recovers_phantom(ran):
    truth = ran["labels"]
    labels = np.asarray(cio.load_labels(ran["results"] / "labels.npy"))
    assert labels.shape == truth.shape and labels.dtype == np.uint8
    agreement = (labels == truth).mean()
    assert agreement > 0.98, agreement
    seg = _json(ran, "segment.json")
    for p, code in (("pore", 0), ("lco", 1), ("llzto", 2)):
        assert abs(seg["volume_fractions"][p] - (truth == code).mean()) < 0.01
    assert (ran["results"] / "qc" / "histogram.png").exists()


def test_connectivity(ran):
    c = _json(ran, "connectivity.json")["phases"]
    assert c["lco"]["percolates_collector_to_separator"]
    assert c["lco"]["connected_fraction"] > 0.99
    assert c["pore"]["percolates_collector_to_separator"]
    # the closed cube is the only isolated pore
    assert 0.9 < c["pore"]["connected_fraction"] < 1.0
    # spheres never touch the separator face
    assert c["llzto"]["connected_fraction"] == 0.0
    assert not c["llzto"]["percolates_collector_to_separator"]
    conn = np.asarray(cio.load_labels(ran["results"] / "labels_connected.npy"))
    assert set(np.unique(conn)) <= {0, 1, 2, 3, 5}
    assert (conn == 5).sum() == (np.asarray(cio.load_labels(ran["results"] / "labels.npy")) == 2).sum()


def test_surface_area_of_spheres(ran):
    s = _json(ran, "surface.json")
    vox = s["voxel_size_um"]
    expected = len(SPHERE_CENTRES) * 4 * math.pi * (SPHERE_R * vox) ** 2
    area = s["interfaces"]["lco-llzto"]["area_um2"]
    assert abs(area - expected) / expected < 0.12, (area, expected)
    # LLZTO is entirely isolated -> nothing active
    assert s["interfaces"]["lco-llzto"]["active_fraction"] == 0.0
    assert s["interfaces"]["lco-pore"]["active_fraction"] > 0.9


def test_tortuosity(ran):
    t = _json(ran, "tortuosity.json")["phases"]
    # LLZTO does not span anything -> inf
    assert all(math.isinf(d["tau"]) for d in t["llzto"]["directions"].values())
    # LCO matrix is well connected but obstructed -> tau modestly above 1
    for d in t["lco"]["directions"].values():
        assert 1.0 <= d["tau"] < 2.0, d


def test_grains(ran):
    g = _json(ran, "grains.json")["phases"]["llzto"]
    assert g["n_grains"] == len(SPHERE_CENTRES)
    df = pd.read_csv(ran["results"] / "grains_llzto.csv")
    vox = _json(ran, "surface.json")["voxel_size_um"]
    assert np.allclose(df["eq_diameter_um"], 2 * SPHERE_R * vox, rtol=0.1)
    assert (df["sphericity"] > 0.6).all()
    # spheres sit in the LCO matrix; two of them overlap a pore channel in the phantom
    assert (df["contact_lco_fraction"] > 0.9).all()
    assert np.allclose(df["contact_lco_fraction"] + df["contact_pore_fraction"], 1.0, atol=1e-6)


def test_profiles_and_report(ran):
    df = pd.read_csv(ran["results"] / "profiles.csv")
    vf = df[(df.quantity == "volume_fraction") & (df.phase == "pore")]
    assert len(vf) > 3 and vf["position_norm"].between(0, 1).all()
    # channels run through the whole thickness: pore fraction is nearly constant
    assert vf["value"].std() < 0.03
    summary = pd.read_csv(ran["output_root"] / "summary.csv").set_index("sample")
    assert "PHANTOM" in summary.index
    assert summary.loc["PHANTOM", "n_grains_llzto"] == len(SPHERE_CENTRES)
    assert (ran["output_root"] / "figures").exists()
    assert (ran["output_root"] / "config_resolved.yaml").exists()


def test_resume_skips_everything(ran, capsys):
    rc = cli.main(["run", "--config", str(ran["config"]), "--stage", "all"])
    assert rc == 0
    status = json.loads((ran["results"] / "stage_status.json").read_text())
    assert set(status) >= {"segment", "connectivity", "surface", "tortuosity", "grains", "profiles"}
