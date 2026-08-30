#!/usr/bin/env bash
set -euo pipefail

module purge
unset LD_LIBRARY_PATH || true
module load cudnn cuda "${PYTHON_MODULE:-python/3.11.11-5e66}"

VENV_PATH="${VENV_PATH:-$HOME/specdec.venv}"
if [[ -d "$VENV_PATH" ]]; then
    existing_version="$("$VENV_PATH/bin/python" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
    if [[ "$existing_version" != "3.11" && "$existing_version" != "3.12" && "$existing_version" != "3.13" ]]; then
        echo "$VENV_PATH uses unsupported Python $existing_version; remove it and rerun setup." >&2
        exit 1
    fi
fi
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
