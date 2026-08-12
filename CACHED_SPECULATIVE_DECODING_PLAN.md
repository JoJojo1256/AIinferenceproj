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

- Greedy output is token-for-token equal to cached baseline decoding.
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

## Success criteria

- All targeted and repository tests pass.
- The implementation never executes a full-prefix model call after prefill.
- Processed-token metrics confirm incremental scaling.
- The 1B/code cached smoke sweep exceeds `1.0x` baseline median throughput for
  at least one of `k=3/4/5`, without correctness failures.

## Fallback if no speedup appears

If correctness and processed-token counts pass but no configuration exceeds
`1.0x`, collect stage timings and compare:

1. draft proposal share;
2. target verification share;
3. sampling/cache-update overhead;
4. acceptance and accepted tokens per target pass.

Cross-check one case with Hugging Face assisted generation. If both paths are
slow, report model-pair/hardware economics (draft latency or acceptance) as the
remaining negative result. If Hugging Face is faster, profile Python launch
overhead, attention backend, cache format/copies, and synchronization before
changing the algorithm or claiming a speedup.
