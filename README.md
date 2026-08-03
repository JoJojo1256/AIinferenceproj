# Speculative Decoding from First Principles

This project implements speculative decoding on top of Hugging Face Transformers, verifies that it preserves the target model's behavior, and measures when draft-model speculation improves or hurts inference performance.

The implementation is being developed in phases. See [`PROJECT.md`](PROJECT.md) for the complete build specification and definitions of done.

## Current status

Phase 0 is scaffolded: model loading, baseline autoregressive decoding, workload definitions, measurement aggregation, provenance capture, and an Oscar Slurm entrypoint. A GPU-produced baseline result is still required to complete the phase.

## Oscar quick start

```bash
git clone https://github.com/JoJojo1256/AIinferenceproj.git
cd AIinferenceproj

# Build the environment from an Ampere GPU compute node, not a login node.
interact -q gpu -g 1 -f ampere -m 40g -n 4
bash env/setup.sh

export HF_TOKEN="<your-token>"
sbatch scripts/slurm_baseline.sh
```

Accept the applicable Llama licenses on Hugging Face before submitting the job. The free account without a PI has no persistent `~/data` directory, so copy important raw results out of `~/scratch`. Never commit the token or model weights.
