# Speculative Decoding on Oscar — Project Build Spec

> **How to use this file:** This is a build specification for Claude Code (or any coding agent). Open your empty repo, drop this file in as `PROJECT.md`, and work through it **phase by phase** — do NOT ask the agent to build everything at once. At the start of each phase, tell the agent: *"Read PROJECT.md. Implement Phase N only. Stop when the phase's Definition of Done is met and show me the results before continuing."* Verify each phase (especially the correctness phase) before moving on.

---

## 0. What this project is

An implementation of **speculative decoding** from the algorithm up, plus a rigorous correctness proof and a performance study characterizing when it helps and when it doesn't.

**The pitch (for a résumé line / interview):** *"I implemented speculative decoding from scratch, proved it preserves the target model's output distribution exactly, and characterized how draft-model choice, speculation length, and workload drive real speedup — including the regimes where it produces a net slowdown."*

**Why this design:** the goal is to demonstrate genuine understanding of LLM inference performance, not just the ability to flip a library flag. The value is concentrated in three things a coding agent must NOT shortcut:
1. A **correct, distribution-preserving** accept/reject rule (the subtle part).
2. A **hard correctness proof** (greedy output must be token-for-token identical to standard decoding).
3. An **honest performance study** with plots, including a negative result.

**Non-goals:** This project does NOT write custom CUDA/attention kernels. It builds on top of HuggingFace `transformers`. Kernel work is explicitly out of scope.

---

## 1. Target environment

- **Cluster:** Brown CCV "Oscar", Slurm-scheduled. All GPU work runs as batch jobs or interactive Slurm sessions — never on a login node.
- **GPUs:** Ampere generation. RTX 3090 (24 GB) via the `3090-gcondo` partition is the assumed default; A100 partitions may also be available (check with `sinfo` / your allocation). A 24 GB card runs a 7–8B model in FP16 comfortably.
- **Modules:** Ampere GPUs require CUDA 11+ modules. Do a `module purge` first, then use a fresh venv. Do not rely on system CUDA.
- **Filesystem:** Use `/oscar/scratch/$USER` for model weights and run outputs (large, fast, scratch). Keep the repo in `/oscar/home/$USER` or `/oscar/data`.

### Models
- **Target model:** `meta-llama/Llama-3.1-8B-Instruct` (gated — the user must accept the HF license and set `HF_TOKEN`).
- **Draft candidates:** `meta-llama/Llama-3.2-1B-Instruct` and `meta-llama/Llama-3.2-3B-Instruct`. Same tokenizer family as the target — this is a hard requirement for speculative decoding to be correct.
- Cache all weights to `/oscar/scratch/$USER/hf_cache` via `HF_HOME`. Never re-download per job.

---

## 2. Repository layout

The agent should create this structure:

```
specdec/
├── README.md                  # the final writeup — see Phase 6
├── PROJECT.md                 # this file
├── pyproject.toml             # or requirements.txt; pinned versions
├── .gitignore                 # exclude weights, outputs, __pycache__, *.out
├── env/
│   └── setup.sh               # module purge + venv + pip install
├── src/specdec/
│   ├── __init__.py
│   ├── models.py              # model/tokenizer loading, dtype, device
│   ├── baseline.py            # standard autoregressive generation
│   ├── speculative.py         # THE CORE: draft-propose + target-verify + accept/reject
│   ├── sampling.py            # rejection-sampling rule, isolated + unit-tested
│   ├── harness.py             # measurement harness: warmup, trials, timing, stats
│   ├── metrics.py             # tokens/sec, TTFT, ITL, acceptance rate, p50/p99
│   └── workloads.py           # prompt sets for different domains
├── scripts/
│   ├── slurm_baseline.sh
│   ├── slurm_specdec.sh
│   └── slurm_sweep.sh
├── experiments/
│   ├── run_baseline.py
│   ├── run_correctness.py     # the equality + distribution tests
│   ├── run_sweep.py           # k-sweep, draft-model sweep, workload sweep
│   └── run_profile.py         # PyTorch profiler / Nsight capture
├── tests/
│   ├── test_sampling.py       # unit tests for the accept/reject rule
│   └── test_correctness.py    # greedy-equality regression test
├── results/
│   ├── raw/                   # JSON per run (gitignored if large; keep a sample)
│   └── figures/               # generated plots (committed)
└── analysis/
    └── make_figures.py        # reads results/raw/*.json -> results/figures/*.png
```

---

## 3. Global engineering principles (apply to every phase)

- **Reproducibility:** every experiment is a committed Slurm script + a Python entrypoint. Fixed seeds. Record git commit hash, GPU model, driver/CUDA version, and model revisions in every results JSON.
- **Measurement rigor (this is the whole point):** always warm up before timing (discard the first N runs — CUDA graph capture, cache, JIT). Run multiple trials. Report **median and p99**, never a bare mean. Use `torch.cuda.synchronize()` around timed regions or timings are meaningless. Separate TTFT (prefill) from per-token decode latency.
- **Correctness before speed:** no performance number is reported until the correctness tests in Phase 2 pass. A fast wrong implementation is worthless and worse than useless in an interview.
- **Everything is a Slurm batch job.** Provide a documented interactive-session command for debugging, but all recorded results come from batch jobs with logged resource requests.
- **Honesty:** report negative results. Finding where the optimization loses is a feature, not a failure.

---

## 4. Phases

Each phase has a **Definition of Done (DoD)**. Do not proceed until it's met.

### Phase 0 — Environment & baseline
**Goal:** get on the GPU, establish the sacred baseline, and build the measurement harness *first*.

Tasks:
- `env/setup.sh`: `module purge`, create venv, install pinned `torch`, `transformers`, `accelerate`, `numpy`, `matplotlib`, `pytest`. Verify with a Slurm job that runs `torch.cuda.is_available()` and prints the GPU name.
- `src/specdec/models.py`: load target + draft with correct dtype (fp16/bf16) on GPU, shared tokenizer, from the scratch HF cache.
- `src/specdec/harness.py` + `metrics.py`: the measurement core. Warmup runs, configurable trial count, `cuda.synchronize()` timing, median + p99 aggregation, JSON output with full provenance.
- `experiments/run_baseline.py` + `scripts/slurm_baseline.sh`: standard autoregressive greedy + sampled generation over a fixed prompt set. Record tokens/sec, TTFT, inter-token latency.

**DoD:** a committed `results/raw/baseline_*.json` produced by a Slurm batch job, with median + p99 tokens/sec and TTFT for the 8B target model. Harness is reusable (baseline and specdec both call it).

### Phase 1 — Core speculative algorithm
**Goal:** implement the draft-propose / target-verify / accept-reject loop.

Tasks:
- `src/specdec/speculative.py`: draft model autoregressively proposes `k` tokens; target model verifies all `k+1` positions in a **single forward pass**; apply the accept/reject rule; on rejection, resample the corrected token from the adjusted distribution and continue.
- `src/specdec/sampling.py`: isolate the **modified rejection sampling** rule. Given draft probs `q(x)` and target probs `p(x)` for a proposed token: accept with probability `min(1, p(x)/q(x))`; on reject, sample from the normalized positive part of `(p - q)`. Keep this pure and separately testable.
- Handle the greedy case (temperature → argmax) as well as temperature sampling.

**DoD:** speculative generation runs end-to-end and produces coherent text. Acceptance rate is logged per run. (Correctness is proven in Phase 2, not assumed here.)

### Phase 2 — Correctness (CRITICAL — do not skip)
**Goal:** prove the implementation preserves the target model's output distribution.

Tasks:
- `tests/test_correctness.py`: **greedy equality test** — with temperature 0, speculative output must be **token-for-token identical** to standard greedy decoding from `baseline.py`, for every prompt in a fixed set. This is a hard assertion.
- Distribution test — with temperature > 0 and a fixed seed regime, run many generations and show the speculative output token-distribution matches the baseline's within statistical tolerance (e.g. compare token-frequency histograms / a chi-square or KL check at the first divergence point).
- `tests/test_sampling.py`: unit-test the accept/reject rule directly on hand-constructed `p`/`q` distributions with known expected acceptance behavior.

**DoD:** `pytest` passes. The greedy-equality test is green across the whole prompt set. This section will be quoted verbatim in the README — it's the credibility centerpiece.

### Phase 3 — Performance study (the centerpiece)
**Goal:** characterize *when* speculative decoding helps.

Experiments (each writes JSON to `results/raw/`):
- **Acceptance rate vs. speedup:** for each draft model (1B, 3B), measure empirical acceptance rate and wall-clock speedup vs. baseline.
- **Speculation length k:** sweep `k ∈ {2,3,4,5,7}`; find the optimum (too low underuses the target pass; too high wastes draft compute on rejected tokens). This is a mini Pareto result.
- **Workload sensitivity:** define ≥3 workloads in `workloads.py` — e.g. straightforward Q&A (high draft agreement), code generation, and multi-step reasoning (lower agreement). Show acceptance rate and speedup vary by workload.
- **Negative result:** identify a regime where specdec produces a **net slowdown** (e.g. 3B draft + low-acceptance workload). Report it explicitly.

**DoD:** `analysis/make_figures.py` regenerates all plots from raw JSON into `results/figures/`: (1) acceptance vs. speedup, (2) k vs. speedup with an annotated optimum, (3) speedup by workload, (4) the negative-result case. Plots are committed.

### Phase 4 — Profiling (stretch)
**Goal:** show *where* the time goes.

Tasks:
- `experiments/run_profile.py`: profile one representative specdec run with PyTorch Profiler; attribute time to draft forward passes vs. target verification vs. overhead. Attempt Nsight Systems (`nsys`) capture on Oscar if available.
- Add a short "where the time goes" section with the breakdown.

**DoD:** a profile trace + a paragraph explaining the bottleneck. Stretch — the project stands without it.

### Phase 5 — vLLM cross-check (stretch)
**Goal:** validate your from-scratch numbers against a production engine.

Tasks:
- Run vLLM's built-in speculative decoding with the same model pair and workloads; compare speedup to your implementation. Explain any gap (vLLM has continuous batching, paged attention, CUDA graphs — expect it to be faster in absolute terms; what matters is that your *relative* speedup trends agree).
- If during this you hit a genuine bug or doc gap in vLLM, that is a natural, low-stakes first open-source contribution — note it, don't force it.

**DoD:** a comparison table + honest commentary. Stretch.

### Phase 6 — Writeup (this is a deliverable, not an afterthought)
**Goal:** the README is half the value. Baseten's team communicates performance results for a living.

The `README.md` must contain, in order:
1. One-paragraph summary + the pitch line.
2. How to reproduce (setup, Slurm commands, model access).
3. The algorithm, explained clearly, with the accept/reject rule written out.
4. **Correctness:** state the greedy-equality guarantee and show the passing test.
5. **Results:** the four figures with interpretation. Lead with acceptance-vs-speedup.
6. **When it helps / when it doesn't:** the workload and negative-result findings.
7. **What I learned about inference performance:** a short, specific reflection.
8. Hardware/repro footnote (GPU, CUDA, model revisions, commit).

**DoD:** a README a hiring engineer can skim in three minutes and understand exactly what you built, that it's correct, and what you learned.

---

## 5. Scope guard

- **Core project = Phases 0–3 + 6.** If you stop there, you have a complete, honest, impressive artifact.
- **Phases 4 and 5 are stretch.** Do not let them block the core.
- Resist scope creep: no custom kernels, no multi-node, no serving-framework rewrite. The depth is in correctness and measurement rigor, not surface area.

## 6. Starter Slurm template (reference)

```bash
#!/bin/bash
#SBATCH -p 3090-gcondo,gpu --gres=gpu:1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH --mem=40g
#SBATCH --time=04:00:00
#SBATCH -o results/logs/%x_%j.out

module purge
source env/pytorch.venv/bin/activate
export HF_HOME=/oscar/scratch/$USER/hf_cache
export HF_TOKEN=...   # set via secret, do not commit

python -u experiments/run_baseline.py --config configs/baseline.yaml
```

## 7. First message to give Claude Code

> Read `PROJECT.md`. Set up the repo skeleton from section 2 and implement **Phase 0 only** (environment, model loading, and the measurement harness). Use pinned dependency versions. Do not implement the speculative algorithm yet. When done, show me `env/setup.sh`, `src/specdec/harness.py`, and the baseline Slurm script, and stop.
