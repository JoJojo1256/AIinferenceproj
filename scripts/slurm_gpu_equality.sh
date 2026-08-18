#!/usr/bin/env bash
#SBATCH --job-name=specdec-gpu-equality
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
module load cudnn cuda python/3.11.11-5e66
source "$HOME/specdec.venv/bin/activate"

: "${HF_TOKEN:?HF_TOKEN must be exported before submitting this job}"
: "${HF_HOME:?HF_HOME must point to the Hugging Face cache}"
export TOKENIZERS_PARALLELISM=false

EXTRA_ARGS=()
if [[ "${NO_CLONE_LOGITS:-0}" == "1" ]]; then
    EXTRA_ARGS+=(--no-clone-logits)
fi

python -u scripts/gpu_greedy_equality.py \
    --target-model meta-llama/Llama-3.1-8B-Instruct \
    --draft-model meta-llama/Llama-3.2-1B-Instruct \
    --dtype bfloat16 \
    --max-new-tokens 128 \
    --static-cache-max-length 512 \
    --compile-warmups 3 \
    --replays-per-prompt 3 \
    --stochastic-seeds 64 \
    --output "results/raw/gpu_equality_${SLURM_JOB_ID}.json" \
    "${EXTRA_ARGS[@]}"
