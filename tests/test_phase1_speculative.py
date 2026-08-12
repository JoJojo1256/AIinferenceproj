from types import SimpleNamespace

import pytest
import torch
from transformers import BatchEncoding

from specdec.models import ModelBundle
from specdec.speculative import generate_speculative


class FakeTokenizer:
    eos_token_id = 7
    pad_token_id = 0
    chat_template = None

    def __init__(self, *, include_eot: bool = True) -> None:
        self.include_eot = include_eot

    def __call__(self, text: str, **_: object) -> BatchEncoding:
        del text
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[0]], dtype=torch.long),
                "attention_mask": torch.tensor([[1]], dtype=torch.long),
            }
        )

    def decode(self, token_ids: list[int], **_: object) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def get_vocab(self) -> dict[str, int]:
        vocabulary = {str(token_id): token_id for token_id in range(8)}
        if self.include_eot:
            vocabulary["<|eot_id|>"] = 6
        return vocabulary


class IncrementModel(torch.nn.Module):
    def __init__(
        self,
        increment: int = 1,
        vocab_size: int = 8,
        *,
        wrong_after: set[int] | None = None,
    ) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.increment = increment
        self.vocab_size = vocab_size
        self.wrong_after = wrong_after or set()
        self.processed_tokens = 0
        self.input_lengths: list[int] = []
        self.cache_lengths_before: list[int] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] | None = None,
        use_cache: bool = False,
        **_: object,
    ) -> SimpleNamespace:
        if past_key_values is None:
            cached_ids = input_ids[:, :0]
        else:
            cached_ids = past_key_values[0][0][:, 0, :, 0].to(dtype=input_ids.dtype)
        self.processed_tokens += input_ids.shape[1]
        self.input_lengths.append(input_ids.shape[1])
        self.cache_lengths_before.append(cached_ids.shape[1])

        increments = torch.full_like(input_ids, self.increment)
        for token_id in self.wrong_after:
            increments = torch.where(input_ids == token_id, self.increment + 1, increments)
        next_ids = (input_ids + increments) % self.vocab_size
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids.unsqueeze(-1), 100.0)
        updated_ids = torch.cat((cached_ids, input_ids), dim=1)
        key = updated_ids[:, None, :, None]
        cache = ((key, key.clone()),) if use_cache else None
        return SimpleNamespace(logits=logits, past_key_values=cache)


def make_bundle(
    model_id: str,
    *,
    increment: int = 1,
    wrong_after: set[int] | None = None,
    include_eot: bool = True,
) -> ModelBundle:
    return ModelBundle(
        model=IncrementModel(increment=increment, wrong_after=wrong_after),
        tokenizer=FakeTokenizer(include_eot=include_eot),
        model_id=model_id,
        revision=None,
    )


def test_greedy_speculative_generation_runs_end_to_end() -> None:
    result = generate_speculative(
        make_bundle("target"),
        make_bundle("draft"),
        "prompt",
        max_new_tokens=6,
        speculation_length=2,
        temperature=0,
    )

    assert result.output_token_ids == [1, 2, 3, 4, 5, 6]
    assert result.accepted_tokens == 4
    assert result.proposed_tokens == 4
    assert result.acceptance_rate == 1.0
    assert result.target_forward_passes == 2
    assert result.target_processed_tokens == 6
    assert result.draft_processed_tokens == 6
    assert result.output_text == "1 2 3 4 5 6"


def test_sampled_speculative_generation_runs_end_to_end() -> None:
    result = generate_speculative(
        make_bundle("target"),
        make_bundle("draft"),
        "prompt",
        max_new_tokens=4,
        speculation_length=3,
        temperature=1.0,
        seed=11,
    )

    assert result.output_token_ids == [1, 2, 3, 4]
    assert result.acceptance_rate == 1.0


def test_greedy_rejection_emits_target_token_and_stops_block() -> None:
    result = generate_speculative(
        make_bundle("target"),
        make_bundle("draft", increment=2),
        "prompt",
        max_new_tokens=4,
        speculation_length=2,
        temperature=0,
    )

    assert result.output_token_ids == [1, 2, 3, 4]
    assert result.accepted_tokens == 0
    assert result.proposed_tokens == 7
    assert result.target_forward_passes == 4


def test_sampled_rejection_uses_corrected_target_distribution() -> None:
    result = generate_speculative(
        make_bundle("target"),
        make_bundle("draft", increment=2),
        "prompt",
        max_new_tokens=3,
        speculation_length=2,
        temperature=1.0,
        seed=3,
    )

    assert result.output_token_ids == [1, 2, 3]
    assert result.accepted_tokens == 0


def test_generation_stops_on_llama_end_of_turn_token() -> None:
    result = generate_speculative(
        make_bundle("target"),
        make_bundle("draft"),
        "prompt",
        max_new_tokens=7,
        speculation_length=2,
        temperature=0,
    )

    assert result.output_token_ids == [1, 2, 3, 4, 5, 6]


def test_generation_stops_on_eos_token() -> None:
    result = generate_speculative(
        make_bundle("target", include_eot=False),
        make_bundle("draft", include_eot=False),
        "prompt",
        max_new_tokens=10,
        speculation_length=3,
        temperature=0,
    )

    assert result.output_token_ids == [1, 2, 3, 4, 5, 6, 7]


def test_models_reuse_cache_and_only_process_incremental_tokens() -> None:
    target = make_bundle("target")
    draft = make_bundle("draft")

    result = generate_speculative(
        target,
        draft,
        "prompt",
        max_new_tokens=6,
        speculation_length=2,
        temperature=0,
    )

    target_model = target.model
    draft_model = draft.model
    assert isinstance(target_model, IncrementModel)
    assert isinstance(draft_model, IncrementModel)
    assert target_model.input_lengths == [1, 2, 3]
    assert draft_model.input_lengths == [1, 1, 1, 1, 1, 1]
    assert target_model.processed_tokens == result.target_processed_tokens
    assert draft_model.processed_tokens == result.draft_processed_tokens


@pytest.mark.parametrize("rejection_index", [0, 1, 2])
def test_rejection_crops_both_caches_to_the_accepted_prefix(
    rejection_index: int,
) -> None:
    target = make_bundle("target")
    draft = make_bundle("draft", wrong_after={rejection_index})

    result = generate_speculative(
        target,
        draft,
        "prompt",
        max_new_tokens=rejection_index + 2,
        speculation_length=3,
        temperature=0,
    )

    target_model = target.model
    draft_model = draft.model
    assert isinstance(target_model, IncrementModel)
    assert isinstance(draft_model, IncrementModel)
    assert result.output_token_ids == list(range(1, rejection_index + 3))
    assert result.accepted_tokens == rejection_index + 1
    assert target_model.cache_lengths_before[2] == 1 + rejection_index
    pending_draft_call = 1 + min(3, rejection_index + 2)
    assert draft_model.cache_lengths_before[pending_draft_call] == 1 + rejection_index


def test_exact_max_token_boundary_does_not_emit_or_process_bonus() -> None:
    target = make_bundle("target")
    draft = make_bundle("draft")

    result = generate_speculative(
        target,
        draft,
        "prompt",
        max_new_tokens=2,
        speculation_length=4,
        temperature=0,
    )

    target_model = target.model
    assert isinstance(target_model, IncrementModel)
    assert result.output_token_ids == [1, 2]
    assert target_model.input_lengths == [1, 2]
    assert result.target_processed_tokens == 3
