from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from threading import RLock
from typing import Iterable

from continuous_batching.config import InferenceConfig
from continuous_batching.engine import ModelEngine
from continuous_batching.metrics import MetricsCollector
from continuous_batching.models import GenerationRequest, SequenceState, SequenceStatus
from continuous_batching.sampling import Sampler


class QueueFullError(RuntimeError):
    pass


@dataclass(frozen=True)
class TokenEvent:
    request_id: str
    token_id: int
    text: str
    delta: str
    finished: bool


class BatchScheduler:
    def __init__(
        self,
        config: InferenceConfig,
        engine: ModelEngine,
        *,
        max_queue_size: int = 1024,
        metrics: MetricsCollector | None = None,
        sampler: Sampler | None = None,
    ) -> None:
        if max_queue_size < 1:
            raise ValueError("max_queue_size must be positive")
        self.config = config
        self.engine = engine
        self.max_queue_size = max_queue_size
        self.metrics = metrics or MetricsCollector(config)
        self.sampler = sampler or Sampler()
        self.pending: deque[SequenceState] = deque()
        self.active: list[SequenceState] = []
        self.completed: deque[SequenceState] = deque()
        self.token_events: deque[TokenEvent] = deque()
        self._request_ids: set[str] = set()
        self._lock = RLock()

    @property
    def has_work(self) -> bool:
        with self._lock:
            return bool(self.pending or self.active)

    def submit(self, request: GenerationRequest) -> str:
        with self._lock:
            if len(self.pending) >= self.max_queue_size:
                raise QueueFullError("The inference queue is full")
            if request.request_id in self._request_ids:
                raise ValueError(f"Duplicate request_id {request.request_id!r}")
            token_ids = self.engine.tokenize(request.prompt)
            max_new_tokens = request.max_new_tokens or self.config.default_max_new_tokens
            capacity = len(token_ids) + max_new_tokens
            if capacity > self.config.max_seq_len:
                raise ValueError(
                    f"Request needs {capacity} tokens but max_seq_len is "
                    f"{self.config.max_seq_len}"
                )
            budget = self.engine.cache_manager.token_budget
            if budget is not None and capacity > budget:
                raise ValueError(
                    f"Request needs {capacity} cache tokens but the total budget is {budget}"
                )
            self.pending.append(
                SequenceState(
                    request=request,
                    prompt_token_ids=token_ids,
                    max_new_tokens=max_new_tokens,
                )
            )
            self._request_ids.add(request.request_id)
            return request.request_id

    def step(self) -> bool:
        with self._lock:
            admitted = self._admit()
            if not self.active:
                self.metrics.record_batch_size(0)
                return admitted
            self.metrics.record_batch_size(len(self.active))
            if all(state.output_tokens == 0 for state in self.active):
                logits = self.engine.prefill(self.active)
            else:
                logits = self.engine.decode_step(self.active)
            now = time.time()
            finished: list[SequenceState] = []
            eos_token_id = self.engine.tokenizer.eos_token_id
            for state, row_logits in zip(self.active, logits):
                token_id = self.sampler.sample(row_logits, state)
                previous_text = self.engine.decode(state.generated_token_ids)
                state.generated_token_ids.append(token_id)
                if state.first_token_time is None:
                    state.first_token_time = now
                if state.slot is None:
                    raise RuntimeError("An active sequence has no cache slot")
                self.engine.cache_manager.update(
                    state.slot, state.request_id, state.all_token_ids
                )
                is_finished = (
                    token_id == eos_token_id
                    or state.output_tokens >= state.max_new_tokens
                )
                full_text = self.engine.decode(state.generated_token_ids)
                delta = (
                    full_text[len(previous_text) :]
                    if full_text.startswith(previous_text)
                    else full_text
                )
                self.token_events.append(
                    TokenEvent(
                        request_id=state.request_id,
                        token_id=token_id,
                        text=full_text,
                        delta=delta,
                        finished=is_finished,
                    )
                )
                if is_finished:
                    state.status = SequenceStatus.FINISHED
                    state.finish_time = now
                    finished.append(state)
            for state in finished:
                self._evict(state)
            self.metrics.record_batch_size(len(self.active), now)
            return True

    def drain_completed(self) -> list[SequenceState]:
        with self._lock:
            states = list(self.completed)
            self.completed.clear()
            self._request_ids.difference_update(state.request_id for state in states)
            return states

    def drain_token_events(self) -> list[TokenEvent]:
        with self._lock:
            events = list(self.token_events)
            self.token_events.clear()
            return events

    def run_until_complete(
        self, requests: Iterable[GenerationRequest]
    ) -> list[SequenceState]:
        request_list = list(requests)
        for request in request_list:
            self.submit(request)
        results: dict[str, SequenceState] = {}
        while self.has_work:
            self.step()
            for state in self.drain_completed():
                results[state.request_id] = state
        return [results[request.request_id] for request in request_list]

    def fail_all(self, error: Exception) -> None:
        with self._lock:
            now = time.time()
            for state in self.active:
                if state.slot is not None:
                    self.engine.cache_manager.free(state.slot, state.request_id)
                    state.slot = None
                state.status = SequenceStatus.FAILED
                state.error = str(error)
                state.finish_time = now
            for state in self.pending:
                state.status = SequenceStatus.FAILED
                state.error = str(error)
                state.finish_time = now
            self.active.clear()
            self.pending.clear()
            self.completed.clear()
            self.token_events.clear()
            self._request_ids.clear()
            self.metrics.record_batch_size(0, now)

    def _admit(self) -> bool:
        if not self.pending:
            return False
        if self.config.mode in {"sequential", "static"} and self.active:
            return False
        if self.config.mode == "sequential":
            target = 1
        elif self.config.mode == "static":
            target = self.config.static_batch_size
        else:
            target = self.config.max_batch_size
        admitted = False
        while self.pending and len(self.active) < target:
            state = self.pending[0]
            capacity = len(state.prompt_token_ids) + state.max_new_tokens
            slot = self.engine.cache_manager.allocate(state.request_id, capacity)
            if slot is None:
                break
            self.pending.popleft()
            state.slot = slot
            state.status = SequenceStatus.ACTIVE
            state.admitted_time = time.time()
            self.active.append(state)
            admitted = True
        return admitted

    def _evict(self, state: SequenceState) -> None:
        if state.slot is None:
            raise RuntimeError("Cannot evict a sequence without a cache slot")
        self.engine.cache_manager.free(state.slot, state.request_id)
        state.slot = None
        self.active.remove(state)
        self.completed.append(state)
        self.metrics.record_request(state)
