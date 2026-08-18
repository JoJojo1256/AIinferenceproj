import pytest
import torch

from specdec.baseline import generate_baseline
from specdec.speculative import generate_speculative

from test_phase1_speculative import make_bundle


def test_greedy_output_matches_baseline_for_fixed_prompts() -> None:
    prompts = (
        "Explain why the sky is blue.",
        "Write a Python function that adds two integers.",
        "If Alice has three apples and buys two more, how many does she have?",
    )

    for prompt in prompts:
        baseline = generate_baseline(
            make_bundle("target"),
            prompt,
            max_new_tokens=6,
            temperature=0,
        )
        speculative = generate_speculative(
            make_bundle("target"),
            make_bundle("draft"),
            prompt,
            max_new_tokens=6,
            speculation_length=2,
            temperature=0,
        )

        assert speculative.output_token_ids == baseline.output_token_ids


@pytest.mark.parametrize(
    "options",
    (
        {
            "draft_cache_implementation": "static",
            "static_cache_max_length": 16,
        },
        {
            "draft_cache_implementation": "static",
            "target_cache_implementation": "static",
            "static_cache_max_length": 16,
        },
        {
            "adaptive_speculation": True,
            "max_speculation_length": 5,
        },
        {
            "compile_draft": True,
            "draft_cache_implementation": "static",
            "static_cache_max_length": 16,
        },
        {
            "compile_draft": True,
            "draft_cache_implementation": "static",
            "static_cache_max_length": 16,
            "adaptive_speculation": True,
            "max_speculation_length": 5,
        },
    ),
)
def test_greedy_optimization_paths_match_baseline_for_fixed_prompts(
    options: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "compile", lambda forward, **_: forward)
    prompts = (
        "Explain why the sky is blue.",
        "Write a Python function that adds two integers.",
        "If Alice has three apples and buys two more, how many does she have?",
    )

    for prompt in prompts:
        baseline = generate_baseline(
            make_bundle("target"),
            prompt,
            max_new_tokens=6,
            temperature=0,
        )
        speculative = generate_speculative(
            make_bundle("target"),
            make_bundle("draft"),
            prompt,
            max_new_tokens=6,
            speculation_length=2,
            temperature=0,
            **options,
        )

        assert speculative.output_token_ids == baseline.output_token_ids


def test_sampled_first_token_distribution_matches_baseline() -> None:
    baseline_counts = torch.zeros(8, dtype=torch.int64)
    speculative_counts = torch.zeros(8, dtype=torch.int64)

    for seed in range(256):
        baseline = generate_baseline(
            make_bundle("target"),
            "prompt",
            max_new_tokens=1,
            temperature=1.0,
            seed=seed,
        )
        speculative = generate_speculative(
            make_bundle("target"),
            make_bundle("draft", increment=2),
            "prompt",
            max_new_tokens=1,
            speculation_length=1,
            temperature=1.0,
            seed=seed,
        )
        baseline_counts[baseline.output_token_ids[0]] += 1
        speculative_counts[speculative.output_token_ids[0]] += 1

    assert torch.equal(speculative_counts, baseline_counts)


@pytest.mark.parametrize(
    "options",
    (
        {
            "draft_cache_implementation": "static",
            "static_cache_max_length": 8,
        },
        {
            "adaptive_speculation": True,
            "max_speculation_length": 4,
        },
        {
            "compile_draft": True,
            "draft_cache_implementation": "static",
            "static_cache_max_length": 8,
            "adaptive_speculation": True,
            "max_speculation_length": 4,
        },
    ),
)
def test_sampled_optimization_paths_preserve_first_token_distribution(
    options: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "compile", lambda forward, **_: forward)
    baseline_counts = torch.zeros(8, dtype=torch.int64)
    speculative_counts = torch.zeros(8, dtype=torch.int64)

    for seed in range(64):
        baseline = generate_baseline(
            make_bundle("target"),
            "prompt",
            max_new_tokens=1,
            temperature=1.0,
            seed=seed,
        )
        speculative = generate_speculative(
            make_bundle("target"),
            make_bundle("draft", increment=2),
            "prompt",
            max_new_tokens=1,
            speculation_length=1,
            temperature=1.0,
            seed=seed,
            **options,
        )
        baseline_counts[baseline.output_token_ids[0]] += 1
        speculative_counts[speculative.output_token_ids[0]] += 1

    assert torch.equal(speculative_counts, baseline_counts)
