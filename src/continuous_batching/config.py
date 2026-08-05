from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class InferenceConfig:
    model_name: str
    dtype: str = "bfloat16"
    max_batch_size: int = 16
    max_seq_len: int = 2048
    kv_cache_budget_tokens: int | None = None
    default_max_new_tokens: int = 128
    device: str = "cuda"
    mode: Literal["sequential", "static", "continuous"] = "continuous"
    static_batch_size: int = 8

    def __post_init__(self) -> None:
        if not self.model_name:
            raise ValueError("model_name cannot be empty")
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if self.max_seq_len < 2:
            raise ValueError("max_seq_len must be at least 2")
        if self.default_max_new_tokens < 1:
            raise ValueError("default_max_new_tokens must be positive")
        if self.default_max_new_tokens >= self.max_seq_len:
            raise ValueError("default_max_new_tokens must be smaller than max_seq_len")
        if self.kv_cache_budget_tokens is not None and self.kv_cache_budget_tokens < 1:
            raise ValueError("kv_cache_budget_tokens must be positive when set")
        if self.static_batch_size < 1 or self.static_batch_size > self.max_batch_size:
            raise ValueError("static_batch_size must be between 1 and max_batch_size")
        if self.mode not in {"sequential", "static", "continuous"}:
            raise ValueError(f"Unsupported scheduling mode: {self.mode!r}")

