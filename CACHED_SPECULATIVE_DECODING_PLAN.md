# Cached Speculative Decoding Plan

## Measured problem

The Phase 3 A40 sweep on commit `ecfa92c` tested both Llama 3.2 draft models,
three workloads, and `k` in `{2, 3, 4, 5, 7}`. All 30 speculative
configurations were slower than the target-only baseline. The best observed
case was the 1B draft on code at about `0.95x`; the worst was the 3B draft on
reasoning at `0.41x` (59% slower). Median speedup peaked near `0.61x` for the
1B draft at `k=5`.

These results remain an honest measurement of the original uncached
implementation. They must not be overwritten or relabeled as cached results.

## Root cause

`src/specdec/speculative.py` currently starts each block from token IDs rather
than model state:

- Draft proposal receives the complete committed prefix and prefills it again.
- Target verification concatenates the complete prefix and proposal and runs
  with `use_cache=False`.
- The cost therefore grows with total sequence length on every block and
  overwhelms any saving from verifying several tokens in one target pass.

The correction is to prefill each model exactly once, persist its KV cache, and
send only incremental tokens after prefill.

## Cache state model

Each model has a committed state:

- `past_key_values`: a KV cache for every committed prompt/output token except
  an optional final pending corrected/bonus token.
- `next_logits`: logits produced by the final cached token, representing the
  distribution of the next token.
- `pending_token`: the final committed token when it has not yet been consumed
  by either model. It is processed by the draft before its next proposal and
  prepended to the target's next verification batch.
- `cache_length`: required to equal the committed attention-mask length minus
  one when a pending token exists, and equal it otherwise.

The production path accepts the Transformers 4.53.1 `DynamicCache` returned by
Llama. Tests may use the legacy tuple-of-layer `(key, value)` cache. Typed
helpers will expose sequence length, validate cache shape, and crop a cache.
Unsupported cache types and missing caches are errors; there is no uncached
full-prefix fallback.

Transformers 4.53.1 semantics were checked directly: Llama creates a
`DynamicCache` when `use_cache=True`, derives `cache_position` from
`get_seq_length()`, appends the new sequence to that cache, and rejects legacy
tuples. `DynamicCache.crop(max_length)` updates `_seen_tokens` and slices each
layer on the sequence dimension. Legacy tuple cropping is implemented only for
cache-aware test doubles.

## Logits alignment

For a committed prefix of length `L` and proposal `x[0:k]`:

1. `target_state.next_logits` predicts `x[0]`.
2. One target call consumes only `x[0:k]` with the prefix cache.
3. Verification output logits at index `i - 1` predict `x[i]` for `i > 0`.
4. Verification output logits at index `k - 1` predict the bonus token.

The draft follows the same state invariant. It samples from stored next-token
logits, feeds only the sampled token, and carries the returned cache/logits to
the next proposal.

## Commit and rollback

Target verification and draft proposal temporarily advance both caches through
all proposed tokens.

- **Full acceptance:** both caches already include the proposal. Sample the
  target bonus token from the final verification logits and commit it as the
  pending token. The draft consumes it before its next proposal; the target
  consumes it in the next verification batch, avoiding a second serial target
  pass.
- **Rejection at proposal index `r`:** crop both caches to
  `L + r`, retaining the accepted proposal prefix only. Commit the corrected
  token as pending, then consume it through both model paths at the start of
  the next block.
- **Accepted EOS/eot or max-token boundary:** commit the accepted proposal
  cache and do not emit a bonus token.

At termination, a final pending token need not be physically processed because
there is no subsequent distribution to compute. After every block, both cache
lengths must match the logical committed length under the pending-token rule.

## Implementation phases

1. Add typed cache helpers for `DynamicCache` and legacy tuple caches.
2. Replace full-prefix proposal/verification with prefill-once cached model
   states and incremental calls.
3. Add explicit full-accept and rejection commit paths with cache assertions.
4. Extend speculative metrics with stage timings and processed-token counts.
5. Upgrade fake models and add focused cache, alignment, rollback, sampling,
   equality, and termination tests.
6. Add a gated Oscar smoke sweep for the 1B draft on code/QA at `k=3/4/5`.

## Correctness gates

- Greedy output is token-for-token equal to cached baseline decoding in exact
  arithmetic and the deterministic CPU test path. Real-GPU equality is reported
  separately by dtype because serial and batched low-precision kernels can
  choose different argmax values near ties.
- Sampled decoding retains modified rejection sampling and matches target
  distributions within the existing statistical test.
- Rejection is tested at the first, middle, and final proposal positions.
- Full acceptance, bonus emission, EOS, eot, and exact max-token boundaries
  are tested.
- Cache lengths equal committed lengths after crop/commit operations.
- Fake-model processed-token counts grow with prompt plus incremental inputs,
  never repeated full prefixes.
- Target verification receives proposal tokens only.

## Profiling and metrics

Keep existing acceptance, throughput, target-forward, and block-latency fields.
Add cumulative:

- draft proposal time;
- target verification time;
- sampling/cache-update overhead time;
- target processed-token count;
- draft processed-token count.

CUDA stage timing uses events recorded around regions and the existing
block-boundary synchronization, avoiding a new synchronization per stage. CPU
tests use `perf_counter`. Harness summaries aggregate the new counters and
times while old result readers remain valid.

## Oscar smoke benchmark

Run from an Ampere GPU node with the same target, dtype, prompt sets, and
token budget as Phase 3:

```bash
python -u experiments/run_sweep.py \
  --draft-model meta-llama/Llama-3.2-1B-Instruct \
  --workload code --workload qa \
  --speculation-length 3 --speculation-length 4 --speculation-length 5 \
  --dtype bfloat16 --max-new-tokens 128 \
  --warmup-runs 3 --trials 5 \
  --output results/raw/cached_specdec_smoke.json
```

The committed Slurm smoke script wraps this exact gated matrix. The original
`scripts/slurm_sweep.sh` and Phase 3 figures remain unchanged as the uncached
historical baseline.

An optional cross-check may run Hugging Face assisted generation with the same
target/draft pair and prompts. It is diagnostic only because `generate()` has
different orchestration and timing boundaries.

## Measured outcome

The cached smoke improved the best result to about `0.78x`, but did not meet the
throughput goal. Stage timing identified the new bottleneck: 26.7 seconds in
the serial 1B draft proposal loop versus 7.6 seconds in target verification.
That measurement motivated the compile, StaticCache, and adaptive-depth
follow-up below.

The optimized full sweep met the performance goal. On an RTX 3090 in bf16, the
1B draft on code reached `1.587x` at fixed `k=9` and `1.834x` with adaptive
speculation starting at five. QA and reasoning remained slower than baseline,
and the worst optimized case was the 3B draft on reasoning at `k=1` (`0.424x`).

Correctness diagnostics established a precise finite-precision boundary:

- modified rejection sampling remains exactly distribution-preserving
  mathematically;
- the real 8B+1B fp32 arm passed all 73 gate checks, including all 64
  cross-path greedy comparisons;
- bf16 serial and batched target shapes can flip near-tied argmax values even
  without speculative decoding. Per-divergence logit differences were
  comparable to the top-two margins, while repeated runs were deterministic
  within each path. The bf16 and fp32 arms used different GPU architectures,
  so their cross-arm delta magnitudes retain that hardware confound.

Historical uncached and cached-negative measurements remain unchanged.

## Draft decode optimization follow-up

The cached RTX 3090 smoke run moved the bottleneck to the serial 1B draft
proposal loop. Three independently recorded options target that bottleneck
without changing modified rejection sampling:

1. Compile the draft's one-token decode forward with
   `torch.compile(mode="reduce-overhead")`. The compiled path clones logits
   before the next replay because reduce-overhead CUDA graphs may reuse output
   buffers.
2. Use a fixed-capacity `StaticCache` for the draft, with explicit
   `cache_position`, a padded fixed-shape attention mask, logical cache length,
   rejected-slot clearing on rollback, and storage reuse across warmup and
   measured generations so CUDA graphs see stable addresses. Target StaticCache
   remains opt-in to avoid unnecessary 8B cache preallocation pressure on a
   24 GB RTX 3090.
3. Adapt `k` using the Transformers 4.53.1 assisted-generation heuristic:
   increase by two after full proposal acceptance; otherwise decrease by one
   with a floor of one and an optional upper bound.

The implementation follows the installed Transformers 4.53.1 source:
`StaticCache(config, max_batch_size, max_cache_len, device, dtype)` preallocates
and marks layer tensors at static addresses, `update()` requires
`cache_position`, and `get_seq_length()` scans nonzero slots rather than
tracking a logical length. The local abstraction therefore owns logical length
and does not use `get_seq_length()` to position or roll back StaticCache.

`scripts/slurm_optimized_smoke.sh` runs the attribution matrix over code and QA
with three warmups and five trials:

1. target-only baseline;
2. eager DynamicCache at fixed `k=5`;
3. compiled draft StaticCache at fixed `k=5`;
4. compiled draft StaticCache with adaptive `k`, starting at five and capped at
   fifteen.

`scripts/slurm_optimized_sweep.sh` runs the full measured matrix: both draft
models, code/QA/reasoning, fixed `k` in `{1, 2, 3, 5, 7, 9}`, and adaptive
speculation starting at five with a cap of fifteen.

The result schema records compile/cache/adaptive switches, initial and realized
per-block `k`, mean/median realized `k`, existing stage timings, and processed
token counts. Historical uncached Phase 3 and cached smoke files remain
unchanged. The optimized sweep reached `1.834x` on 1B/code. Real-GPU greedy
equality is reported by dtype: all 64 cross-path comparisons passed in fp32,
while bf16 exposed deterministic shape-dependent argmax flips that a
target-only control reproduced independently of speculative decoding.
