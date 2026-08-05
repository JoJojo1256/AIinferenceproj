from __future__ import annotations

import torch

from continuous_batching.models import SequenceState


class Sampler:
    def sample(self, logits: torch.Tensor, state: SequenceState) -> int:
        request = state.request
        if request.temperature == 0:
            return int(torch.argmax(logits).item())
        probabilities = torch.softmax(logits.float() / request.temperature, dim=-1)
        if request.top_p < 1:
            sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
            cumulative = torch.cumsum(sorted_probabilities, dim=-1)
            remove = cumulative - sorted_probabilities >= request.top_p
            sorted_probabilities[remove] = 0
            sorted_probabilities /= sorted_probabilities.sum()
            probabilities = torch.zeros_like(probabilities).scatter(
                0, sorted_indices, sorted_probabilities
            )
        if state.sampler_state is None:
            state.sampler_state = torch.Generator(device=logits.device).manual_seed(
                request.seed
            )
        return int(
            torch.multinomial(
                probabilities,
                num_samples=1,
                generator=state.sampler_state,
            ).item()
        )

