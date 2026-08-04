#!/usr/bin/env bash
set -euo pipefail

VENV_PATH="${VENV_PATH:-$HOME/specdec.venv}"
python3 -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python scripts/gpu_preflight.py --minimum-vram-gb "${MINIMUM_VRAM_GB:-20}"
