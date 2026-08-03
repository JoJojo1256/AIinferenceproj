#!/usr/bin/env bash
#SBATCH --job-name=specdec-baseline
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=04:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
mkdir -p results/logs results/raw

module purge
unset LD_LIBRARY_PATH || true
module load cudnn cuda
source "${VENV_PATH:-$HOME/specdec.venv}/bin/activate"

export HF_HOME="${HF_HOME:-$HOME/scratch/hf_cache}"
export TOKENIZERS_PARALLELISM=false

if [[ -z "${HF_TOKEN:-}" ]]; then
    echo "HF_TOKEN must be exported before submitting this job." >&2
    exit 1
fi

python -u experiments/run_baseline.py \
    --workload qa \
    --dtype bfloat16 \
    --max-new-tokens 128 \
    --warmup-runs 3 \
    --trials 10
