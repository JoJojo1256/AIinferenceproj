#!/usr/bin/env bash
#SBATCH --job-name=specdec-optimized-smoke
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=results/logs/%x_%j.out
#SBATCH --error=results/logs/%x_%j.err

set -euo pipefail

cd "${SLURM_SUBMIT_DIR:?Submit this script with sbatch}"
mkdir -p results/logs results/raw

module purge
unset LD_LIBRARY_PATH || true
module load cudnn cuda python/3.11.11-5e66
source "$HOME/specdec.venv/bin/activate"

: "${HF_TOKEN:?HF_TOKEN must be exported before submitting this job}"
: "${HF_HOME:?HF_HOME must point to the Hugging Face cache}"
export TOKENIZERS_PARALLELISM=false

COMMON_ARGS=(
    --draft-model meta-llama/Llama-3.2-1B-Instruct
    --workload code
    --workload qa
    --dtype bfloat16
    --max-new-tokens 128
    --warmup-runs 3
    --trials 5
)

python -u experiments/run_sweep.py \
    "${COMMON_ARGS[@]}" \
    --speculation-length 5 \
    --output results/raw/optimized_smoke_dynamic_fixed_k5.json

python -u experiments/run_sweep.py \
    "${COMMON_ARGS[@]}" \
    --speculation-length 5 \
    --compile-draft \
    --draft-cache-implementation static \
    --static-cache-max-length 512 \
    --output results/raw/optimized_smoke_compiled_static_fixed_k5.json

python -u experiments/run_sweep.py \
    "${COMMON_ARGS[@]}" \
    --speculation-length 5 \
    --compile-draft \
    --draft-cache-implementation static \
    --static-cache-max-length 512 \
    --adaptive-speculation \
    --max-speculation-length 15 \
    --output results/raw/optimized_smoke_compiled_static_adaptive_k5.json
