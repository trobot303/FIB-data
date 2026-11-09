"""Stage orchestration: registry, resumability, logging and memory tracking.

Stage contract
--------------
Every stage module exposes

    NAME: str                      unique stage name (matches the CLI)
    SCOPE: "sample" | "global"     run once per sample, or once over all samples
    DEPENDS_ON: tuple[str, ...]    stages whose outputs must exist first
    expected_outputs(ctx) -> list[Path]
    run(ctx) -> dict               returns a small JSON-serialisable summary

A stage is skipped when all of ``expected_outputs`` already exist, unless the
run was started with ``force=True``.
"""

from __future__ import annotations

import datetime as _dt
import importlib
import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Mapping

from . import __version__
from .config import STAGES, resolve_sample, save_resolved, select_samples

LOG = logging.getLogger("cathode_analysis")

# stage name -> module path; order defines execution order for --stage all
_STAGE_MODULES = {
    "segment": "cathode_analysis.segmentation",
    "connectivity": "cathode_analysis.connectivity",
    "surface": "cathode_analysis.surface",
    "tortuosity": "cathode_analysis.tortuosity",
    "grains": "cathode_analysis.grains",
    "profiles": "cathode_analysis.profiles",
    "report": "cathode_analysis.report",
}
assert tuple(_STAGE_MODULES) == STAGES


# --------------------------------------------------------------------------- #
# context handed to every stage
# --------------------------------------------------------------------------- #
@dataclass
class StageContext:
    cfg: dict[str, Any]                 # fully resolved config (sample overrides applied)
    stage: str
    output_dir: Path                    # <output_root>/<sample>  or <output_root> for global stages
    sample: str | None = None           # None for global stages
    force: bool = False
    dry_run: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def log(self) -> logging.Logger:
        tag = f"{self.stage}:{self.sample}" if self.sample else self.stage
        return logging.getLogger(f"cathode_analysis.{tag}")

    @property
    def voxel_um(self) -> float:
        return float(self.cfg["voxel"]["size_nm"]) / 1000.0

    @property
    def sample_output_dirs(self) -> dict[str, Path]:
        """Per-sample output folders (used by global stages such as report)."""
        root = Path(self.cfg["paths"]["output_root"])
        return {s["name"]: root / s["name"] for s in self.cfg["samples"]}

    def path(self, name: str) -> Path:
        return self.output_dir / name


# --------------------------------------------------------------------------- #
# logging
# --------------------------------------------------------------------------- #
def setup_logging(log_dir: str | Path, level: str = "INFO") -> Path:
    """Console + one file per run. Returns the log file path."""
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"run_{stamp}.log"

    root = logging.getLogger()
    root.setLevel(getattr(logging, level))
    for h in list(root.handlers):
        root.removeHandler(h)
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s", "%H:%M:%S")
    for handler in (logging.StreamHandler(), logging.FileHandler(log_file, encoding="utf-8")):
        handler.setFormatter(fmt)
        root.addHandler(handler)
    LOG.info("cathode_analysis %s | log file: %s", __version__, log_file)
    return log_file


# --------------------------------------------------------------------------- #
# peak memory monitor
# --------------------------------------------------------------------------- #
class PeakMemory:
    """Samples the process RSS in a background thread; reports the peak in GB.

    Uses psutil when available, otherwise falls back to resource.ru_maxrss
    (process-lifetime maximum, so a lower bound for later stages).
    """

    def __init__(self, interval_s: float = 0.25):
        self.interval = interval_s
        self.peak_bytes = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        try:
            import psutil  # noqa: WPS433

            self._proc = psutil.Process(os.getpid())
        except Exception:  # pragma: no cover - psutil missing
            self._proc = None

    def _rss(self) -> int:
        if self._proc is not None:
            return int(self._proc.memory_info().rss)
        try:
            import resource

            return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024
        except Exception:
            return 0

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.peak_bytes = max(self.peak_bytes, self._rss())
            self._stop.wait(self.interval)

    def __enter__(self) -> "PeakMemory":
        self.peak_bytes = self._rss()
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self.peak_bytes = max(self.peak_bytes, self._rss())

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / 1e9


# --------------------------------------------------------------------------- #
# running stages
# --------------------------------------------------------------------------- #
def load_stage(name: str) -> ModuleType:
    if name not in _STAGE_MODULES:
        raise KeyError(f"unknown stage {name!r}; choose from {list(_STAGE_MODULES)}")
    return importlib.import_module(_STAGE_MODULES[name])


def stages_to_run(stage: str) -> list[str]:
    return list(STAGES) if stage == "all" else [stage]


def outputs_exist(module: ModuleType, ctx: StageContext) -> bool:
    outs = module.expected_outputs(ctx)
    return bool(outs) and all(Path(p).exists() for p in outs)


def check_dependencies(module: ModuleType, ctx: StageContext) -> None:
    missing = []
    for dep in getattr(module, "DEPENDS_ON", ()):
        dep_mod = load_stage(dep)
        dep_ctx = StageContext(cfg=ctx.cfg, stage=dep, output_dir=ctx.output_dir, sample=ctx.sample)
        if not outputs_exist(dep_mod, dep_ctx):
            missing.append(dep)
    if missing:
        raise RuntimeError(
            f"stage {module.NAME!r} for sample {ctx.sample!r} needs outputs of {missing}; "
            "run those stages first (or --stage all)"
        )


def run_stage(module: ModuleType, ctx: StageContext) -> dict[str, Any]:
    """Run one stage with skip/force logic, timing and peak-memory logging."""
    log = ctx.log
    status: dict[str, Any] = {"stage": module.NAME, "sample": ctx.sample}

    if ctx.dry_run:
        log.info("[dry-run] would produce: %s", [str(p) for p in module.expected_outputs(ctx)])
        status["status"] = "dry-run"
        return status

    if outputs_exist(module, ctx) and not ctx.force:
        log.info("outputs exist -> skipping (use --force to recompute)")
        status["status"] = "skipped"
        return status

    check_dependencies(module, ctx)
    ctx.output_dir.mkdir(parents=True, exist_ok=True)
    log.info("start")
    t0 = time.perf_counter()
    with PeakMemory() as mem:
        summary = module.run(ctx) or {}
    dt = time.perf_counter() - t0
    if ctx.cfg["runtime"].get("memory_log", True):
        log.info("done in %.1f s | peak RSS %.2f GB", dt, mem.peak_gb)
    else:
        log.info("done in %.1f s", dt)

    status.update({"status": "done", "seconds": round(dt, 2), "peak_rss_gb": round(mem.peak_gb, 3), "summary": summary})
    _append_status(ctx.output_dir / "stage_status.json", status)
    return status


def _append_status(path: Path, status: Mapping[str, Any]) -> None:
    data: dict[str, Any] = {}
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            data = {}
    data[str(status["stage"])] = {**status, "finished_at": _dt.datetime.now().isoformat(timespec="seconds")}
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def run_pipeline(
    cfg: dict[str, Any],
    stage: str = "all",
    samples: list[str] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Run the requested stage(s) for the requested sample(s)."""
    output_root = Path(cfg["paths"]["output_root"])
    if not dry_run:
        output_root.mkdir(parents=True, exist_ok=True)
        save_resolved(cfg, output_root / "config_resolved.yaml")

    names = select_samples(cfg, samples)
    results: list[dict[str, Any]] = []
    for stage_name in stages_to_run(stage):
        module = load_stage(stage_name)
        if module.SCOPE == "global":
            ctx = StageContext(cfg=cfg, stage=stage_name, output_dir=output_root, force=force, dry_run=dry_run,
                               extra={"samples": names})
            results.append(run_stage(module, ctx))
            continue
        for name in names:
            scfg = resolve_sample(cfg, name)
            ctx = StageContext(cfg=scfg, stage=stage_name, output_dir=Path(scfg["sample"]["output_dir"]),
                               sample=name, force=force, dry_run=dry_run)
            results.append(run_stage(module, ctx))
    return results
