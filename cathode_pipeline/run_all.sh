#!/usr/bin/env bash
# Headless end-to-end run on the workstation.
#   ./run_all.sh                     # all samples, all stages, resume where possible
#   ./run_all.sh --force             # recompute everything
#   CONFIG=config/default.yaml ./run_all.sh --sample LCO50   # other config
set -euo pipefail
cd "$(dirname "$0")"

CONFIG="${CONFIG:-config/workstation.yaml}"
PYTHON="${PYTHON:-python3}"
VENV="${VENV:-.venv}"

if [ ! -d "$VENV" ]; then
    "$PYTHON" -m venv "$VENV"
    "$VENV/bin/python" -m pip install --upgrade pip
    "$VENV/bin/python" -m pip install -r requirements.txt
fi

"$VENV/bin/python" -m cathode_analysis.cli validate --config "$CONFIG"
"$VENV/bin/python" -m cathode_analysis.cli run --config "$CONFIG" --stage all "$@"
