from specdec.harness import run_benchmark
from specdec.metrics import (
    GenerationMetrics,
    SpeculativeGenerationMetrics,
    percentile,
    summarize_trials,
)


def test_percentile_empty_sequence_is_zero() -> None:
    assert percentile([], 99) == 0.0


def test_trial_summary_reports_expected_medians() -> None:
    trials = [
        GenerationMetrics("a", "x", [1, 2], 10.0, [4.0], 20.0),
        GenerationMetrics("b", "y", [3, 4], 20.0, [8.0], 40.0),
        GenerationMetrics("c", "z", [5, 6], 30.0, [12.0], 80.0),
    ]

    summary = summarize_trials(trials)

    assert summary["trial_count"] == 3
    assert summary["ttft_ms_median"] == 20.0
    assert summary["decode_latency_ms_median"] == 8.0
    assert summary["tokens_per_second_median"] == 50.0


def test_harness_summarizes_speculative_acceptance() -> None:
    def generate(prompt: str, seed: int) -> SpeculativeGenerationMetrics:
        del seed
        return SpeculativeGenerationMetrics(
            prompt=prompt,
            output_text="x",
            output_token_ids=[1],
            ttft_ms=1.0,
            decode_latencies_ms=[],
            total_latency_ms=2.0,
            proposed_tokens=4,
            accepted_tokens=3,
            target_forward_passes=2,
            block_latencies_ms=[2.0],
        )

    results = run_benchmark(generate, ["prompt"], warmup_runs=0, trials=2, seed=1)

    assert results["summary"]["proposed_tokens"] == 8
    assert results["summary"]["accepted_tokens"] == 6
    assert results["summary"]["acceptance_rate"] == 0.75
    assert results["summary"]["target_forward_passes"] == 4
