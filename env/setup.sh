#!/usr/bin/env bash
set -euo pipefail

module purge
unset LD_LIBRARY_PATH || true
module load cudnn cuda

VENV_PATH="${VENV_PATH:-$HOME/specdec.venv}"
python -m venv "$VENV_PATH"
source "$VENV_PATH/bin/activate"
python -m pip install --upgrade pip
python -m pip install -e '.[dev]'

python - <<'PY'
import torch

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"gpu={torch.cuda.get_device_name(0)}")
PY
