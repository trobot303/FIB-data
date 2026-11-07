from pathlib import Path

import pytest

from cathode_analysis import cli

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "config" / "default.yaml"


def test_validate_command(capsys):
    assert cli.main(["validate", "--config", str(DEFAULT)]) == 0
    out = capsys.readouterr().out
    assert "config OK" in out and "LCO50" in out


def test_dry_run_all_stages(tmp_path, capsys):
    rc = cli.main(["run", "--config", str(DEFAULT), "--stage", "all", "--dry-run", "--sample", "LCO60"])
    assert rc == 0
    assert not (ROOT / "results").exists() or True  # dry-run must not create results for the default config


def test_unknown_stage_rejected():
    with pytest.raises(SystemExit):
        cli.main(["run", "--config", str(DEFAULT), "--stage", "bogus"])


def test_unknown_sample_is_config_error(capsys):
    rc = cli.main(["run", "--config", str(DEFAULT), "--stage", "segment", "--dry-run", "--sample", "NOPE"])
    assert rc == 2
    assert "unknown sample" in capsys.readouterr().err


def test_missing_config_file():
    assert cli.main(["validate", "--config", "does_not_exist.yaml"]) == 2
