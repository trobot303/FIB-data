"""Command-line entry point.

    python -m cathode_analysis.cli run --config config/default.yaml --stage all
    python -m cathode_analysis.cli run --config c.yaml --stage segment --sample LCO50 --force
    python -m cathode_analysis.cli run --config c.yaml --stage all --dry-run
    python -m cathode_analysis.cli validate --config c.yaml
    python -m cathode_analysis.cli inspect --config c.yaml        # list slices found per sample

Headless by design: no interactive prompts, no GUI backend.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Sequence

# Force a non-interactive matplotlib backend before anything imports pyplot.
os.environ.setdefault("MPLBACKEND", "Agg")

from . import __version__
from .config import STAGES, ConfigError, load_config, resolve_sample, select_samples

STAGE_CHOICES = ("all",) + STAGES


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m cathode_analysis.cli",
                                description="3D microstructure analysis of LCO/LLZTO composite cathodes")
    p.add_argument("--version", action="version", version=f"cathode_analysis {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run one stage or the whole pipeline")
    run.add_argument("--config", required=True, help="YAML config file")
    run.add_argument("--stage", default="all", choices=STAGE_CHOICES)
    run.add_argument("--sample", action="append", default=None, metavar="NAME",
                     help="restrict to this sample (repeatable); default: all samples in the config")
    run.add_argument("--force", action="store_true", help="recompute even if stage outputs exist")
    run.add_argument("--dry-run", action="store_true", help="resolve config and list planned outputs, run nothing")
    run.add_argument("--log-level", default=None, choices=("DEBUG", "INFO", "WARNING", "ERROR"),
                     help="override runtime.log_level")

    val = sub.add_parser("validate", help="load and validate a config, print the resolved paths")
    val.add_argument("--config", required=True)

    ins = sub.add_parser("inspect", help="discover input slices per sample and print shape/dtype")
    ins.add_argument("--config", required=True)
    ins.add_argument("--sample", action="append", default=None, metavar="NAME")
    return p


def _cmd_validate(args: argparse.Namespace) -> int:
    cfg = load_config(args.config)
    print(f"config OK: {cfg['_meta']['config_path']}")
    for k, v in cfg["paths"].items():
        print(f"  paths.{k}: {v}")
    for name in select_samples(cfg, None):
        s = resolve_sample(cfg, name)["sample"]
        if s["input_mode"] == "masks":
            print(f"  sample {name}: input_dir={s['input_dir']} masks={s['masks']} -> {s['output_dir']}")
        else:
            print(f"  sample {name}: input_dir={s['input_dir']} pattern={s['pattern']!r} -> {s['output_dir']}")
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    from . import io as cio

    cfg = load_config(args.config)
    for name in select_samples(cfg, args.sample):
        s = resolve_sample(cfg, name)["sample"]
        if s["input_mode"] == "masks":
            for phase, pattern in s["masks"].items():
                print(f"{name} [{phase}]: {cio.describe_stack(s['input_dir'], pattern)}")
        else:
            print(f"{name}: {cio.describe_stack(s['input_dir'], s['pattern'])}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from .pipeline import run_pipeline, setup_logging

    cfg = load_config(args.config)
    level = args.log_level or cfg["runtime"]["log_level"]
    if args.dry_run:
        logging.basicConfig(level=level, format="%(levelname)-7s | %(name)s | %(message)s", stream=sys.stdout)
    else:
        setup_logging(cfg["paths"]["log_dir"], level)
    log = logging.getLogger("cathode_analysis")
    log.info("config: %s | stage: %s | samples: %s | force=%s dry_run=%s",
             cfg["_meta"]["config_path"], args.stage, args.sample or "all", args.force, args.dry_run)

    results = run_pipeline(cfg, stage=args.stage, samples=args.sample, force=args.force, dry_run=args.dry_run)
    for r in results:
        log.info("%-12s %-8s %s", r["stage"], r.get("sample") or "-", r["status"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return _cmd_run(args)
        if args.command == "validate":
            return _cmd_validate(args)
        if args.command == "inspect":
            return _cmd_inspect(args)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    except NotImplementedError as exc:
        print(f"not implemented yet: {exc}", file=sys.stderr)
        return 3
    return 1


if __name__ == "__main__":
    sys.exit(main())
