#!/usr/bin/env bash
# Create the virtual environment for cathode_analysis in a cloud session and run its tests.
#   bash scripts/setup_cloud_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/../cathode_pipeline"
python3 -m venv .venv
. .venv/bin/activate
pip install -q --upgrade pip
pip install -q torch                      # ~5 GB with the CUDA libraries that PyPI bundles; runs on CPU
pip install -q -r requirements_cloud.txt
python -m pytest tests -q
