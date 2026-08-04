import pytest
import torch

from specdec.sampling import (
    acceptance_probability,
    corrected_distribution,
    modified_rejection_sample,
)


def test_acceptance_probability_uses_clipped_probability_ratio() -> None:
    target = torch.tensor([0.6, 0.4])
    draft = torch.tensor([0.3, 0.7])

    assert acceptance_probability(target, draft, 0).item() == 1.0
    assert acceptance_probability(target, draft, 1).item() == pytest.approx(4 / 7)


def test_corrected_distribution_normalizes_positive_difference() -> None:
    target = torch.tensor([0.6, 0.1, 0.3])
    draft = torch.tensor([0.2, 0.5, 0.3])

    corrected = corrected_distribution(target, draft)

    assert torch.allclose(corrected, torch.tensor([1.0, 0.0, 0.0]))


def test_corrected_distribution_rejects_identical_distributions() -> None:
    probabilities = torch.tensor([0.25, 0.75])

    with pytest.raises(ValueError, match="undefined"):
        corrected_distribution(probabilities, probabilities)


def test_rejection_samples_from_positive_target_difference() -> None:
    target = torch.tensor([0.0, 1.0])
    draft = torch.tensor([1.0, 0.0])

    accepted, token_id = modified_rejection_sample(
        target,
        draft,
        proposed_token_id=0,
        generator=torch.Generator().manual_seed(7),
    )

    assert accepted is False
    assert token_id == 1
