# Speculative Decoding from First Principles

This project implements speculative decoding on top of Hugging Face Transformers, verifies that it preserves the target model's behavior, and measures when draft-model speculation improves or hurts inference performance.

The implementation is being developed in phases. See [`PROJECT.md`](PROJECT.md) for the complete build specification and definitions of done.
See [`GPU_ACCESS.md`](GPU_ACCESS.md) for Oscar and standalone Linux GPU workflows.

## Current status

Phases 0 and 1 implement the reusable benchmark harness, draft proposal, batched target verification, modified rejection sampling, corrected-token resampling, greedy decoding, and acceptance-rate logging. Phase 2 includes deterministic greedy-equality and sampled-distribution regression tests using local toy models. The Phase 3 A40 sweep found every original speculative configuration slower than baseline because both models repeatedly processed the full prefix. Speculative generation now prefills each model once, carries persistent KV caches, verifies only new proposals, and explicitly crops and repairs both caches after rejection. The original Phase 3 figures remain the honest uncached result.

## Oscar quick start

```bash
git clone https://github.com/JoJojo1256/AIinferenceproj.git
cd AIinferenceproj

# The exploratory account permits four CPU cores and standard GPUs.
# Build the environment from an Ampere GPU compute node, not a login node.
interact -q gpu -g 1 -f ampere -m 40g -n 4
bash env/setup.sh

export HF_TOKEN="<your-token>"
export HF_HOME="$HOME/scratch/hf_cache"
sbatch scripts/slurm_baseline.sh
sbatch scripts/slurm_specdec.sh
sbatch scripts/slurm_sweep.sh
sbatch scripts/slurm_cached_smoke.sh
sbatch scripts/slurm_optimized_smoke.sh
sbatch scripts/slurm_gpu_equality.sh
```

Accept the applicable Llama licenses on Hugging Face before submitting the job. The free account without a PI has no persistent `~/data` directory, so copy important raw results out of `~/scratch`. Never commit the token or model weights.

After the Phase 3 sweep completes, generate the figures with:

```bash
python analysis/make_figures.py results/raw/phase3_sweep_*.json
```

`env/setup.sh` loads Oscar's Python 3.11 module by default. Set `PYTHON_MODULE` before running it if Oscar replaces that module version.

The gated cached rerun uses the 1B draft, code and QA workloads, `k=3/4/5`,
three warmups, and five measured trials:

```bash
python -u experiments/run_sweep.py \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --workload code --workload qa \
  --speculation-length 3 --speculation-length 4 --speculation-length 5 \
  --dtype bfloat16 --max-new-tokens 128 \
  --warmup-runs 3 --trials 5 \
  --output results/raw/cached_specdec_smoke.json
```

Speculative result JSON retains the original acceptance and block metrics and
adds prefill, draft-proposal, target-verification, sampling/overhead, and
target/draft processed-token totals.

## Draft decode optimization attribution

The optimized path keeps fixed-`k`, eager execution, and `DynamicCache` as the
defaults so historical results remain comparable. The new options are:

- `--compile-draft`: compiles only the draft model's one-token decode forward
  with `torch.compile(mode="reduce-overhead")`. It requires draft
  `StaticCache`, a fixed `--static-cache-max-length`, and at least one warmup.
- `--draft-cache-implementation static`: uses an explicitly positioned,
  fixed-capacity draft cache. `--target-cache-implementation static` is
  separate and opt-in because preallocating the 8B target cache consumes
  additional VRAM on a 24 GB RTX 3090.
- `--adaptive-speculation`: starts at `--speculation-length`, increases `k` by
  two after full acceptance, and decreases it by one after rejection with a
  floor of one. `--max-speculation-length` optionally caps growth.

Each speculative trial records these switches, the initial `k`, all realized
block lengths, mean/median realized `k`, stage timings, and processed-token
counts. Submit `scripts/slurm_optimized_smoke.sh` for the baseline, eager
DynamicCache control, compiled StaticCache fixed-`k`, and compiled StaticCache
adaptive attribution matrix.

The real-model CUDA correctness gate runs baseline, eager DynamicCache,
compiled StaticCache, and compiled StaticCache plus adaptive speculation over
eight code/QA prompts. It profiles a replay and fails if no CUDA graph launch
is observed, checks cloned output-buffer lifetime directly, repeats each
compiled generation three times, and compares sampled first-token
distributions:

```bash
sbatch scripts/slurm_gpu_equality.sh
```

Run the precision diagnostic separately to distinguish reduced-precision
shape effects from algorithmic errors:

```bash
DTYPE=float32 sbatch scripts/slurm_gpu_equality.sh
```

The real 8B + 1B pair requires substantially more than 24 GB in float32; run
that arm on a high-memory GPU (for example, an 80 GB A100), not an RTX 3090.

The gate records same-path determinism, target-only incremental-versus-batched
logit differences, top-two logit gaps at every first divergence, and the
overall/by-workload/by-variant distribution of first-divergence indices. In
floating-point arithmetic, batched verification and one-token baseline
decoding can use different reduction orders; greedy equality is therefore
reported as an empirical hardware/dtype property rather than overstated as
bitwise universal when near-tied logits can flip argmax.

The buffer-level alias probe observed `cudaGraphLaunch` on an RTX 3090, reuse
of the raw compiled output storage, overwrite of an uncloned retained
reference, and survival of the production clone. The current generation loop
consumes each logit before the next replay, so removing the clone did not
change end-to-end tokens: the hazard is real, while the clone makes safety
explicit rather than incidental. The optional negative control is expected to
fail the buffer-lifetime assertion:

```bash
NO_CLONE_LOGITS=1 sbatch scripts/slurm_gpu_equality.sh
```
