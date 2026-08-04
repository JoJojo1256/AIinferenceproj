#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 baseline|speculative [experiment arguments...]" >&2
    exit 2
fi

experiment="$1"
shift

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

source "${VENV_PATH:-$HOME/specdec.venv}/bin/activate"
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN must be set in the environment." >&2
    exit 1
fi

python scripts/gpu_preflight.py --minimum-vram-gb "${MINIMUM_VRAM_GB:-20}"
mkdir -p results/raw results/logs

case "$experiment" in
    baseline)
        exec python -u experiments/run_baseline.py "$@"
        ;;
    speculative)
        exec python -u experiments/run_speculative.py "$@"
        ;;
    *)
        echo "Unknown experiment '$experiment'; choose baseline or speculative." >&2
        exit 2
        ;;
esac
