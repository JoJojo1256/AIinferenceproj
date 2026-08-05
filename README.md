# LLM Inference Systems from First Principles

This repository contains two independent, correctness-first Hugging Face inference projects:

- **Speculative decoding** implements draft proposal and target verification without changing the target model's output distribution.
- **Continuous batching** implements an iteration-level scheduler that immediately replaces completed sequences, improving GPU utilization under mixed request lengths.

Both projects use PyTorch and Transformers rather than custom CUDA kernels. [`PROJECT.md`](PROJECT.md) remains the detailed speculative-decoding build specification, and [`GPU_ACCESS.md`](GPU_ACCESS.md) covers Oscar and standalone Linux GPU workflows.

## Continuous batching

### Architecture and motivation

`src/continuous_batching` separates concerns so every scheduling policy exercises the same inference behavior:

```text
FastAPI -> bounded request queue -> BatchScheduler
                                  -> ModelEngine.prefill/decode_step
                                  -> fixed-slot KVCacheManager
                                  -> per-request Sampler
                                  -> JSONL metrics
```

`sequential` admits one request at a time. `static` admits a fixed group and waits for the whole group to finish before admitting another. `continuous` performs FCFS admission every decoding iteration, samples one token per active sequence, evicts EOS/max-token completions, and fills their slots immediately. Admission reserves each sequence's full prompt-plus-output token capacity, so a low cache budget queues work instead of overallocating.

Ragged batches have explicit attention masks and position IDs. Device and dtype selection live only in `ModelEngine`. Greedy generation is token-identical across sequential, static, and continuous scheduling regardless of batch composition or request arrival ordering; the schedulers choose *when* a sequence advances, never *how* its token is computed.

The cache manager provides deterministic fixed-slot ownership, capacity accounting, free-list reuse, and complete state clearing on eviction. The portable Transformers reference engine currently recomputes each active sequence's visible token prefix on each iteration while tracking logical cache occupancy; it does not implement paged attention or custom fused KV-cache kernels.

### Setup and server

Python 3.11 and one CUDA GPU are the target runtime. Install the pinned environment:

```bash
python -m venv .venv
.venv\Scripts\activate
python -m pip install -e ".[dev]"
```

Run a swappable Hugging Face causal model:

```bash
continuous-batching-server \
  --model-name meta-llama/Llama-3.1-8B-Instruct \
  --device cuda --dtype bfloat16 \
  --mode continuous --max-batch-size 16 \
  --max-seq-len 2048 --kv-cache-budget-tokens 32768 \
  --metrics-path results/raw/server.jsonl
```

The API exposes:

- `POST /generate_sync` for one JSON response.
- `POST /generate` for server-sent token events.
- `GET /metrics` for throughput, configurable SLO goodput, p50/p99 latency, and active-batch trace/histogram/time average.

The queue is bounded. Overload returns HTTP 429; invalid requests and sequences that cannot fit the configured context or total token budget return HTTP 422.

### Benchmark methodology

Keep prompts fixed, discard warmup requests, and run at least three measured trials. The load generator supports closed-loop concurrency and open-loop Poisson arrivals plus uniform, bimodal, and seeded sampled output lengths:

```bash
continuous-batching-loadgen \
  --url http://127.0.0.1:8000 \
  --prompts bench/prompts.txt \
  --arrival closed --concurrency 8 \
  --length-workload bimodal \
  --warmup-requests 4 --trials 3 \
  --output results/raw/continuous_c8_bimodal.jsonl

continuous-batching-analyze \
  results/raw/continuous_c8_bimodal.jsonl \
  --output-dir results/continuous_c8_bimodal
```

`bench/sweep.py` covers static and continuous servers at concurrency 1, 2, 4, 8, 16, and 32 for all three output-length workloads. Raw JSONL is gitignored. `bench/analyze.py` derives summary JSON/CSV and latency plots solely from raw JSONL. Records include arrival, first-token and finish timestamps, TTFT, TPOT, end-to-end latency, token counts, active batch size, exact configuration, package versions, git state, CUDA runtime, and hardware.

Expected limits are one model on one GPU, host-side Python scheduling, no distributed execution, no paged attention, no custom kernels, and no prefix cache. GPU measurements still require model access, warmup, and representative production prompts.

## Speculative decoding

The existing speculative-decoding implementation remains under `src/specdec` and is unchanged. It includes standard autoregressive generation, draft-model proposal, batched target verification, modified rejection sampling, corrected-token resampling, greedy decoding, acceptance metrics, workload definitions, and Oscar scripts.

```bash
interact -q gpu -g 1 -f ampere -m 40g -n 4
bash env/setup.sh
export HF_TOKEN="<your-token>"
sbatch scripts/slurm_baseline.sh
sbatch scripts/slurm_specdec.sh
```

Accept the applicable model licenses before submitting jobs. Keep Hugging Face tokens and model weights out of the repository.

## Correctness tests

Tests use tiny local fake models and require neither a GPU nor network access:

```bash
python -m pytest
```

They cover exact ragged masks/positions, greedy equality across all schedulers and arrival orders, reproducible per-request sampling, slot reuse without leakage, budget-limited queuing, continuous replacement, one output per input without duplicates, and in-process FastAPI streaming/synchronous behavior.
