from __future__ import annotations

from types import SimpleNamespace

import torch


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 31
    eos_token = "<eos>"
    pad_token = "<pad>"
    chat_template = None

    def __call__(self, text: str, **_: object) -> dict[str, list[int]]:
        return {"input_ids": [ord(character) % 20 + 1 for character in text]}

    def decode(self, token_ids: list[int], **_: object) -> str:
        return " ".join(str(token_id) for token_id in token_ids)


class IncrementModel(torch.nn.Module):
    def __init__(self, vocab_size: int = 32) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.vocab_size = vocab_size
        self.forward_batches: list[list[list[int]]] = []

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        **_: object,
    ) -> SimpleNamespace:
        self.forward_batches.append(input_ids.tolist())
        assert torch.equal(
            position_ids * attention_mask,
            torch.arange(input_ids.shape[1]).expand_as(input_ids) * attention_mask,
        )
        next_ids = (input_ids + 1) % self.vocab_size
        logits = torch.full(
            (*input_ids.shape, self.vocab_size),
            -100.0,
            device=input_ids.device,
        )
        logits.scatter_(2, next_ids.unsqueeze(-1), 100.0)
        return SimpleNamespace(logits=logits)

