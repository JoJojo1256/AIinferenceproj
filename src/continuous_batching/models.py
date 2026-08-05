from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any
from uuid import uuid4


class SequenceStatus(StrEnum):
    QUEUED = "queued"
    ACTIVE = "active"
    FINISHED = "finished"
    FAILED = "failed"


@dataclass(frozen=True)
class GenerationRequest:
    prompt: str
    request_id: str = field(default_factory=lambda: uuid4().hex)
    max_new_tokens: int | None = None
    temperature: float = 0.0
    top_p: float = 1.0
    seed: int = 0
    arrival_time: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id cannot be empty")
        if not self.prompt:
            raise ValueError("prompt cannot be empty")
        if self.max_new_tokens is not None and self.max_new_tokens < 1:
            raise ValueError("max_new_tokens must be positive")
        if self.temperature < 0:
            raise ValueError("temperature must be non-negative")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")


@dataclass
class SequenceState:
    request: GenerationRequest
    prompt_token_ids: list[int]
    max_new_tokens: int
    generated_token_ids: list[int] = field(default_factory=list)
    slot: int | None = None
    status: SequenceStatus = SequenceStatus.QUEUED
    admitted_time: float | None = None
    first_token_time: float | None = None
    finish_time: float | None = None
    position: int = 0
    cache_length: int = 0
    error: str | None = None
    sampler_state: Any = field(default=None, repr=False)

    @property
    def request_id(self) -> str:
        return self.request.request_id

    @property
    def all_token_ids(self) -> list[int]:
        return [*self.prompt_token_ids, *self.generated_token_ids]

    @property
    def output_tokens(self) -> int:
        return len(self.generated_token_ids)

