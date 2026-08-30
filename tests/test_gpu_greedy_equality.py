from collections import Counter

import torch

from scripts.gpu_greedy_equality import (
    _divergence_index_summary,
    _first_divergence,
    _mismatch_details,
    _top_two,
    _total_variation,
)


class _Tokenizer:
    def decode(self, token_ids: list[int], **_: object) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


def test_first_divergence_detects_token_and_length_mismatches() -> None:
    assert _first_divergence([1, 2, 3], [1, 4, 3]) == 1
    assert _first_divergence([1, 2], [1, 2, 3]) == 2
    assert _first_divergence([1, 2], [1, 2]) is None


def test_mismatch_details_reports_token_and_text_windows() -> None:
    details = _mismatch_details(
        [1, 2, 3, 4],
        [1, 2, 7, 4],
        _Tokenizer(),
    )

    assert details["first_divergent_index"] == 2
    assert details["expected_token_ids"] == [1, 2, 3, 4]
    assert details["actual_token_ids"] == [1, 2, 7, 4]
    assert details["expected_text"] == "1 2 3 4"
    assert details["actual_text"] == "1 2 7 4"


def test_total_variation_compares_empirical_token_distributions() -> None:
    assert _total_variation(Counter({1: 2}), Counter({1: 2}), 2) == 0.0
    assert _total_variation(Counter({1: 2}), Counter({2: 2}), 2) == 1.0


def test_top_two_reports_ids_values_and_absolute_gap() -> None:
    result = _top_two(torch.tensor([-1.0, 3.0, 2.5]))

    assert result == {
        "token_ids": [1, 2],
        "logits": [3.0, 2.5],
        "absolute_gap": 0.5,
    }


def test_divergence_summary_reports_distribution_and_groups() -> None:
    records = [
        {
            "passed": False,
            "first_divergent_index": 7,
            "workload": "code",
            "variant": "eager",
        },
        {
            "passed": False,
            "first_divergent_index": 28,
            "workload": "code",
            "variant": "adaptive",
        },
        {
            "passed": False,
            "first_divergent_index": 7,
            "workload": "qa",
            "variant": "adaptive",
        },
        {
            "passed": True,
            "workload": "qa",
            "variant": "eager",
        },
    ]

    summary = _divergence_index_summary(records)

    assert summary["failure_count"] == 3
    assert summary["histogram"] == {"7": 2, "28": 1}
    assert summary["minimum"] == 7
    assert summary["median"] == 7.0
    assert summary["maximum"] == 28
    assert summary["by_workload"] == {
        "code": {"7": 1, "28": 1},
        "qa": {"7": 1},
    }
    assert summary["by_variant"] == {
        "adaptive": {"7": 1, "28": 1},
        "eager": {"7": 1},
    }
