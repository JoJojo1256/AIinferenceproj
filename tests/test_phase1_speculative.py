from types import SimpleNamespace

import torch
from transformers import BatchEncoding

from specdec.models import ModelBundle
from specdec.speculative import generate_speculative


class FakeTokenizer:
    eos_token_id = 7
    pad_token_id = 0
    chat_template = None

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
        vocabulary["<|eot_id|>"] = 6
        return vocabulary


class IncrementModel(torch.nn.Module):
    def __init__(self, increment: int = 1, vocab_size: int = 8) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.increment = increment
        self.vocab_size = vocab_size

    def forward(self, input_ids: torch.Tensor, **_: object) -> SimpleNamespace:
        next_ids = (input_ids + self.increment) % self.vocab_size
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids.unsqueeze(-1), 100.0)
        return SimpleNamespace(logits=logits, past_key_values=None)


def make_bundle(model_id: str, *, increment: int = 1) -> ModelBundle:
    return ModelBundle(
        model=IncrementModel(increment=increment),
        tokenizer=FakeTokenizer(),
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
