#!/usr/bin/env bash
#SBATCH --job-name=specdec-cached-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=06:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
mkdir -p results/logs results/raw

module purge
unset LD_LIBRARY_PATH || true
module load cudnn cuda "${PYTHON_MODULE:-python/3.11.11-5e66}"
source "${VENV_PATH:-$HOME/specdec.venv}/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN must be exported before submitting this job." >&2
    exit 1
fi

python -u experiments/run_sweep.py \
    --draft-model meta-llama/Llama-3.2-1B-Instruct \
    --workload code \
    --workload qa \
    --speculation-length 3 \
    --speculation-length 4 \
    --speculation-length 5 \
    --dtype bfloat16 \
    --max-new-tokens 128 \
    --warmup-runs 3 \
    --trials 5 \
    --output results/raw/cached_specdec_smoke.json
