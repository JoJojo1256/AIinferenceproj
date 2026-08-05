from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from continuous_batching.cache import KVCacheManager
from continuous_batching.config import InferenceConfig
from continuous_batching.models import SequenceState


def resolve_dtype(name: str) -> torch.dtype:
    dtypes = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    try:
        return dtypes[name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {name!r}; choose one of {sorted(dtypes)}") from exc


@dataclass(frozen=True)
class RaggedBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    position_ids: torch.Tensor
    sequence_lengths: torch.Tensor


def build_ragged_batch(
    token_rows: Sequence[Sequence[int]],
    *,
    pad_token_id: int,
    device: torch.device | str,
) -> RaggedBatch:
    if not token_rows or any(not row for row in token_rows):
        raise ValueError("token_rows must contain at least one non-empty sequence")
    max_length = max(len(row) for row in token_rows)
    input_ids = torch.full(
        (len(token_rows), max_length),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    position_ids = torch.zeros_like(input_ids)
    lengths: list[int] = []
    for row_index, row in enumerate(token_rows):
        length = len(row)
        lengths.append(length)
        input_ids[row_index, :length] = torch.tensor(row, dtype=torch.long, device=device)
        attention_mask[row_index, :length] = 1
        position_ids[row_index, :length] = torch.arange(length, device=device)
    return RaggedBatch(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        sequence_lengths=torch.tensor(lengths, dtype=torch.long, device=device),
    )


class ModelEngine:
    """One model forward path shared by every scheduling policy."""

    def __init__(
        self,
        config: InferenceConfig,
        *,
        model: Any | None = None,
        tokenizer: Any | None = None,
        cache_manager: KVCacheManager | None = None,
        cache_dir: str | Path | None = None,
        revision: str | None = None,
        token: str | None = None,
    ) -> None:
        self.config = config
        self.device = torch.device(config.device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("A CUDA device was requested, but CUDA is unavailable")
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(
            config.model_name,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            use_fast=True,
        )
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = model or AutoModelForCausalLM.from_pretrained(
            config.model_name,
            cache_dir=cache_dir,
            revision=revision,
            token=token,
            torch_dtype=resolve_dtype(config.dtype),
            low_cpu_mem_usage=True,
        )
        self.model.to(self.device)
        self.model.eval()
        self.cache_manager = cache_manager or KVCacheManager(
            config.max_batch_size,
            config.kv_cache_budget_tokens,
        )

    def tokenize(self, prompt: str) -> list[int]:
        text = prompt
        add_special_tokens = True
        if getattr(self.tokenizer, "chat_template", None):
            text = self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
            add_special_tokens = False
        encoded = self.tokenizer(text, add_special_tokens=add_special_tokens)
        token_ids = encoded["input_ids"]
        if token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        return [int(token_id) for token_id in token_ids]

    @torch.inference_mode()
    def prefill(self, states: Sequence[SequenceState]) -> torch.Tensor:
        return self._forward(states)

    @torch.inference_mode()
    def decode_step(self, states: Sequence[SequenceState]) -> torch.Tensor:
        return self._forward(states)

    def decode(self, token_ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(token_ids), skip_special_tokens=True)

    def metadata(self) -> dict[str, Any]:
        model_config = getattr(self.model, "config", None)
        return {
            "model_class": type(self.model).__qualname__,
            "tokenizer_class": type(self.tokenizer).__qualname__,
            "model_config": (
                model_config.to_dict()
                if model_config is not None and hasattr(model_config, "to_dict")
                else None
            ),
        }

    def _forward(self, states: Sequence[SequenceState]) -> torch.Tensor:
        if not states:
            raise ValueError("At least one sequence is required")
        batch = build_ragged_batch(
            [state.all_token_ids for state in states],
            pad_token_id=int(self.tokenizer.pad_token_id),
            device=self.device,
        )
        outputs = self.model(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask,
            position_ids=batch.position_ids,
            use_cache=False,
        )
        row_indices = torch.arange(len(states), device=self.device)
        final_indices = batch.sequence_lengths - 1
        logits = outputs.logits[row_indices, final_indices]
        for state in states:
            if state.slot is None:
                raise RuntimeError("An active sequence has no cache slot")
            state.position = len(state.all_token_ids)
            state.cache_length = state.position
            self.cache_manager.update(state.slot, state.request_id, state.all_token_ids)
        return logits
