from collections import Counter

from scripts.gpu_greedy_equality import (
    _first_divergence,
    _mismatch_details,
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
