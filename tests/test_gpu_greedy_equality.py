from collections import Counter
import json

import pytest
import torch

from scripts.gpu_greedy_equality import (
    _clean_reference_cases,
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


def test_clean_reference_requires_schema_v2_output_tokens(tmp_path) -> None:
    reference = tmp_path / "clean.json"
    reference.write_text(
        json.dumps(
            {
                "greedy_equality": [
                    {
                        "workload": "code",
                        "prompt_index": 0,
                        "variant": "compiled-static-fixed",
                        "passed": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="actual_output_token_ids"):
        _clean_reference_cases(reference)
