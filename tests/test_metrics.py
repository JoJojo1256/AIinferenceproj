from specdec.metrics import GenerationMetrics, percentile, summarize_trials


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
