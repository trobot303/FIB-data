#!/usr/bin/env bash
# Headless run on ALREADY SEGMENTED stacks (config/workstation_segmented.yaml).
#   ./run_segmented.sh                     # all samples in the config, all stages, resume
#   ./run_segmented.sh --force             # recompute everything
#   ./run_segmented.sh --sample LCO50      # one sample
set -euo pipefail
cd "$(dirname "$0")"
CONFIG="config/workstation_segmented.yaml" exec ./run_all.sh "$@"
