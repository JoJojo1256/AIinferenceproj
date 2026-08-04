from __future__ import annotations

import torch


def probabilities_from_logits(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("temperature must be positive when constructing a sampling distribution")
    if logits.ndim != 1:
        raise ValueError("logits must be a one-dimensional vocabulary vector")
    return torch.softmax(logits.float() / temperature, dim=-1)


def sample_distribution(
    probabilities: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> int:
    _validate_distribution(probabilities)
    return int(torch.multinomial(probabilities, num_samples=1, generator=generator).item())


def acceptance_probability(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
    proposed_token_id: int,
) -> torch.Tensor:
    _validate_matching_distributions(target_probabilities, draft_probabilities)
    if not 0 <= proposed_token_id < target_probabilities.numel():
        raise ValueError("proposed_token_id is outside the vocabulary")

    draft_probability = draft_probabilities[proposed_token_id]
    target_probability = target_probabilities[proposed_token_id]
    if draft_probability <= 0:
        return torch.ones((), device=target_probabilities.device, dtype=torch.float32)
    return torch.clamp(target_probability / draft_probability, max=1.0)


def corrected_distribution(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
) -> torch.Tensor:
    _validate_matching_distributions(target_probabilities, draft_probabilities)
    positive_difference = torch.clamp(target_probabilities - draft_probabilities, min=0)
    mass = positive_difference.sum()
    if mass <= torch.finfo(positive_difference.dtype).eps:
        raise ValueError("The corrected distribution is undefined when p and q have no positive difference")
    return positive_difference / mass


def modified_rejection_sample(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
    proposed_token_id: int,
    *,
    generator: torch.Generator | None = None,
) -> tuple[bool, int]:
    accept_probability = acceptance_probability(
        target_probabilities,
        draft_probabilities,
        proposed_token_id,
    )
    random_value = torch.rand(
        (),
        device=target_probabilities.device,
        generator=generator,
        dtype=torch.float32,
    )
    if random_value < accept_probability:
        return True, proposed_token_id

    replacement = sample_distribution(
        corrected_distribution(target_probabilities, draft_probabilities),
        generator=generator,
    )
    return False, replacement


def _validate_matching_distributions(
    target_probabilities: torch.Tensor,
    draft_probabilities: torch.Tensor,
) -> None:
    _validate_distribution(target_probabilities)
    _validate_distribution(draft_probabilities)
    if target_probabilities.shape != draft_probabilities.shape:
        raise ValueError("Target and draft distributions must have matching shapes")
    if target_probabilities.device != draft_probabilities.device:
        raise ValueError("Target and draft distributions must be on the same device")


def _validate_distribution(probabilities: torch.Tensor) -> None:
    if probabilities.ndim != 1:
        raise ValueError("Probability distributions must be one-dimensional")
    if not probabilities.is_floating_point():
        raise ValueError("Probability distributions must use a floating-point dtype")
    if not torch.isfinite(probabilities).all():
        raise ValueError("Probability distributions must contain only finite values")
    if (probabilities < 0).any():
        raise ValueError("Probability distributions cannot contain negative values")
    if not torch.isclose(
        probabilities.sum(),
        torch.ones((), device=probabilities.device, dtype=probabilities.dtype),
        atol=1e-5,
        rtol=1e-5,
    ):
        raise ValueError("Probability distributions must sum to one")
