"""Clausnitzer-style figures from summary rows (including inf / missing tortuosity)."""
import math

import numpy as np

from cathode_analysis import paper_figures as pf


def _rows():
    return {
        "LCO50": {"vf_pore": 0.22, "vf_lco": 0.46, "vf_llzto": 0.32, "connected_fraction_lco": 0.99,
                  "connected_fraction_llzto": 0.97, "tau_lco_y": 1.6, "tau_llzto_y": 2.4, "D_rel_lco_y": 0.29,
                  "D_rel_llzto_y": 0.13, "specific_area_lco_llzto_1_um": 0.9,
                  "active_specific_area_lco_llzto_1_um": 0.8},
        "LCO60": {"vf_pore": 0.27, "vf_lco": 0.46, "vf_llzto": 0.27, "connected_fraction_lco": 0.99,
                  "connected_fraction_llzto": 0.90, "tau_lco_y": 1.5, "tau_llzto_y": 4.0, "D_rel_lco_y": 0.31,
                  "D_rel_llzto_y": 0.07, "specific_area_lco_llzto_1_um": 0.8,
                  "active_specific_area_lco_llzto_1_um": 0.6},
        "LCO70": {"vf_pore": 0.21, "vf_lco": 0.60, "vf_llzto": 0.19, "connected_fraction_lco": 1.0,
                  "connected_fraction_llzto": 0.40, "tau_lco_y": 1.3, "tau_llzto_y": 413.0, "D_rel_lco_y": 0.46,
                  "D_rel_llzto_y": 0.0005, "tau_converged_llzto_y": False, "specific_area_lco_llzto_1_um": 0.6,
                  "active_specific_area_lco_llzto_1_um": 0.2},
    }


def test_sample_table_sorts_by_svf_and_handles_inf():
    t = pf.sample_table(_rows(), ["LCO70", "LCO50", "LCO60"], "y")
    assert list(t["names"]) == ["LCO50", "LCO60", "LCO70"]
    assert abs(t["svf_lco"][0] - 0.46 / 0.78 * 100) < 1e-9
    assert abs(t["density"][2] - 79.0) < 1e-9
    assert pf._inv_tau(t["tau_se"]).tolist() == [1 / 2.4, 1 / 4.0, 1 / 413.0]
    assert t["open_se"].tolist() == [False, False, True] and not t["open_cam"].any()
    assert pf._inv_tau(np.array([math.inf, math.nan]))[0] == 0.0
    assert abs(t["a_act_per_vca_1_cm"][0] - 8000.0) < 1e-9


def test_three_figures_are_written(tmp_path):
    made = pf.make_paper_figures(_rows(), list(_rows()), tmp_path, "y", "png", 100)
    assert sorted(p.name for p in made) == [
        "clausnitzer_fig2_connectivity_tortuosity.png",
        "clausnitzer_fig3_effective_conductivity.png",
        "clausnitzer_fig7_tortuosity_active_area.png",
    ]
    assert all(p.stat().st_size > 1000 for p in made)


def test_without_tortuosity_only_fig2_is_made(tmp_path):
    rows = {n: {k: v for k, v in r.items() if not (k.startswith("tau_") or k.startswith("D_rel_"))
                and not k.endswith("_1_um")} for n, r in _rows().items()}
    made = pf.make_paper_figures(rows, list(rows), tmp_path, "y")
    assert [p.name for p in made] == ["clausnitzer_fig2_connectivity_tortuosity.png"]


def test_single_sample_does_not_crash(tmp_path):
    rows = {"LCO50": _rows()["LCO50"]}
    assert len(pf.make_paper_figures(rows, ["LCO50"], tmp_path, "y")) == 3
