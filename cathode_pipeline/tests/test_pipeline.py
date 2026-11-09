"""Resume / force logic and stage registry, exercised with a dummy stage."""
import types
from pathlib import Path

import pytest

from cathode_analysis import pipeline
from cathode_analysis.config import STAGES, load_config

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ROOT / "config" / "default.yaml"


def _dummy_stage(calls):
    mod = types.ModuleType("dummy")
    mod.NAME = "dummy"
    mod.SCOPE = "sample"
    mod.DEPENDS_ON = ()
    mod.expected_outputs = lambda ctx: [ctx.path("dummy.json")]

    def run(ctx):
        calls.append(ctx.sample)
        ctx.path("dummy.json").write_text("{}")
        return {"ok": True}

    mod.run = run
    return mod


@pytest.fixture
def ctx(tmp_path):
    cfg = load_config(DEFAULT)
    return pipeline.StageContext(cfg=cfg, stage="dummy", output_dir=tmp_path / "S", sample="S")


def test_stage_runs_then_skips_unless_forced(ctx):
    calls = []
    mod = _dummy_stage(calls)

    first = pipeline.run_stage(mod, ctx)
    assert first["status"] == "done" and calls == ["S"]
    assert "peak_rss_gb" in first
    assert (ctx.output_dir / "stage_status.json").exists()

    second = pipeline.run_stage(mod, ctx)
    assert second["status"] == "skipped" and calls == ["S"]

    ctx.force = True
    third = pipeline.run_stage(mod, ctx)
    assert third["status"] == "done" and calls == ["S", "S"]


def test_dry_run_touches_nothing(ctx):
    calls = []
    ctx.dry_run = True
    res = pipeline.run_stage(_dummy_stage(calls), ctx)
    assert res["status"] == "dry-run" and calls == [] and not ctx.output_dir.exists()


def test_all_stage_modules_follow_contract():
    for name in STAGES:
        mod = pipeline.load_stage(name)
        assert mod.NAME == name
        assert mod.SCOPE in ("sample", "global")
        assert all(dep in STAGES for dep in mod.DEPENDS_ON)
        assert callable(mod.run) and callable(mod.expected_outputs)


def test_missing_dependency_is_reported(tmp_path):
    cfg = load_config(DEFAULT)
    mod = pipeline.load_stage("connectivity")
    ctx = pipeline.StageContext(cfg=cfg, stage="connectivity", output_dir=tmp_path, sample="X")
    with pytest.raises(RuntimeError, match="segment"):
        pipeline.check_dependencies(mod, ctx)
