from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BatchEncoding,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)


@dataclass(frozen=True)
class ModelBundle:
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    model_id: str
    revision: str | None


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


def load_model(
    model_id: str,
    *,
    revision: str | None = None,
    dtype: str = "bfloat16",
    device: str = "cuda",
    cache_dir: str | Path | None = None,
    token: str | None = None,
) -> ModelBundle:
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("A CUDA device was requested, but torch.cuda.is_available() is false")

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        cache_dir=cache_dir,
        token=token,
        torch_dtype=resolve_dtype(dtype),
        low_cpu_mem_usage=True,
    )
    model.to(device)
    model.eval()

    return ModelBundle(model=model, tokenizer=tokenizer, model_id=model_id, revision=revision)


def validate_shared_tokenizer(target: ModelBundle, draft: ModelBundle) -> None:
    target_vocab = target.tokenizer.get_vocab()
    draft_vocab = draft.tokenizer.get_vocab()
    if target_vocab != draft_vocab:
        raise ValueError("Target and draft tokenizers must have identical token-to-id mappings")


def encode_prompt(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    device: torch.device,
) -> BatchEncoding:
    text = prompt
    add_special_tokens = True
    if tokenizer.chat_template:
        text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        add_special_tokens = False

    encoded = tokenizer(
        text,
        return_tensors="pt",
        add_special_tokens=add_special_tokens,
    )
    return encoded.to(device)
