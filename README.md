# Speculative Decoding from First Principles

This project implements speculative decoding on top of Hugging Face Transformers, verifies that it preserves the target model's behavior, and measures when draft-model speculation improves or hurts inference performance.

The implementation is being developed in phases. See [`PROJECT.md`](PROJECT.md) for the complete build specification and definitions of done.
See [`GPU_ACCESS.md`](GPU_ACCESS.md) for Oscar and standalone Linux GPU workflows.

## Current status

Phases 0 and 1 implement the reusable benchmark harness, draft proposal, batched target verification, modified rejection sampling, corrected-token resampling, greedy decoding, and acceptance-rate logging. Phase 2 includes deterministic greedy-equality and sampled-distribution regression tests using local toy models. The Phase 3 runner sweeps draft models, workloads, and speculation lengths and the analysis script generates the four required performance figures. Oscar GPU runs are still required to populate those figures with full Llama results.

## Oscar quick start

```bash
git clone https://github.com/JoJojo1256/AIinferenceproj.git
cd AIinferenceproj

# The exploratory account permits four CPU cores and standard GPUs.
# Build the environment from an Ampere GPU compute node, not a login node.
interact -q gpu -g 1 -f ampere -m 40g -n 4
bash env/setup.sh

export HF_TOKEN="<your-token>"
sbatch scripts/slurm_baseline.sh
sbatch scripts/slurm_specdec.sh
sbatch scripts/slurm_sweep.sh
```

Accept the applicable Llama licenses on Hugging Face before submitting the job. The free account without a PI has no persistent `~/data` directory, so copy important raw results out of `~/scratch`. Never commit the token or model weights.

After the Phase 3 sweep completes, generate the figures with:

```bash
python analysis/make_figures.py results/raw/phase3_sweep_*.json
```

`env/setup.sh` loads Oscar's Python 3.11 module by default. Set `PYTHON_MODULE` before running it if Oscar replaces that module version.
