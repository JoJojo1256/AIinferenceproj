from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class SlotCache:
    request_id: str | None = None
    capacity_tokens: int = 0
    token_ids: list[int] = field(default_factory=list)
    past_key_values: Any = None

    def clear(self) -> None:
        self.request_id = None
        self.capacity_tokens = 0
        self.token_ids.clear()
        self.past_key_values = None


class KVCacheManager:
    """Owns fixed request slots and reserves a bounded token budget."""

    def __init__(self, max_slots: int, token_budget: int | None = None) -> None:
        if max_slots < 1:
            raise ValueError("max_slots must be positive")
        if token_budget is not None and token_budget < 1:
            raise ValueError("token_budget must be positive when set")
        self.max_slots = max_slots
        self.token_budget = token_budget
        self._slots = [SlotCache() for _ in range(max_slots)]
        self._free = list(range(max_slots - 1, -1, -1))
        self._lock = Lock()

    @property
    def free_slots(self) -> int:
        with self._lock:
            return len(self._free)

    @property
    def used_tokens(self) -> int:
        with self._lock:
            return sum(slot.capacity_tokens for slot in self._slots if slot.request_id is not None)

    def allocate(self, request_id: str, token_budget: int) -> int | None:
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        with self._lock:
            if any(slot.request_id == request_id for slot in self._slots):
                raise ValueError(f"Request {request_id!r} already owns a cache slot")
            used = sum(
                slot.capacity_tokens for slot in self._slots if slot.request_id is not None
            )
            if not self._free or (
                self.token_budget is not None and used + token_budget > self.token_budget
            ):
                return None
            slot_index = self._free.pop()
            slot = self._slots[slot_index]
            slot.clear()
            slot.request_id = request_id
            slot.capacity_tokens = token_budget
            return slot_index

    def update(
        self,
        slot_index: int,
        request_id: str,
        token_ids: list[int],
        past_key_values: Any = None,
    ) -> None:
        with self._lock:
            slot = self._slot(slot_index)
            if slot.request_id != request_id:
                raise ValueError("Cache slot ownership mismatch")
            if len(token_ids) > slot.capacity_tokens:
                raise ValueError("Sequence exceeded its reserved cache capacity")
            slot.token_ids = list(token_ids)
            slot.past_key_values = past_key_values

    def snapshot(self, slot_index: int) -> SlotCache:
        with self._lock:
            slot = self._slot(slot_index)
            return SlotCache(
                request_id=slot.request_id,
                capacity_tokens=slot.capacity_tokens,
                token_ids=list(slot.token_ids),
                past_key_values=slot.past_key_values,
            )

    def free(self, slot_index: int, request_id: str) -> None:
        with self._lock:
            slot = self._slot(slot_index)
            if slot.request_id != request_id:
                raise ValueError("Cache slot ownership mismatch")
            slot.clear()
            self._free.append(slot_index)

    def _slot(self, slot_index: int) -> SlotCache:
        if not 0 <= slot_index < self.max_slots:
            raise IndexError(f"Invalid cache slot {slot_index}")
        return self._slots[slot_index]

