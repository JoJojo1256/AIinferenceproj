from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from transformers.cache_utils import DynamicCache, StaticCache

LegacyLayerCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]
LegacyCache: TypeAlias = tuple[LegacyLayerCache, ...]
KVCache: TypeAlias = DynamicCache | StaticCache | LegacyCache
CacheImplementation: TypeAlias = Literal["dynamic", "static"]


@dataclass(frozen=True)
class CacheState:
    cache: KVCache
    sequence_length: int

    def __post_init__(self) -> None:
        if self.sequence_length < 0:
            raise ValueError("Cache sequence length must be non-negative")
        if isinstance(self.cache, StaticCache) and self.sequence_length > self.cache.max_cache_len:
            raise ValueError(
                f"Cache sequence length {self.sequence_length} exceeds StaticCache capacity "
                f"{self.cache.max_cache_len}"
            )

    @property
    def implementation(self) -> CacheImplementation:
        return "static" if isinstance(self.cache, StaticCache) else "dynamic"


def require_kv_cache(value: object) -> KVCache:
    if isinstance(value, (DynamicCache, StaticCache)):
        return value
    if not isinstance(value, tuple):
        raise TypeError(
            "Expected transformers.DynamicCache, transformers.StaticCache, "
            "or a legacy tuple cache, "
            f"got {type(value).__name__}"
        )

    sequence_length: int | None = None
    for layer_index, layer in enumerate(value):
        if not isinstance(layer, tuple) or len(layer) != 2:
            raise TypeError(f"Legacy cache layer {layer_index} must be a (key, value) tuple")
        key, item_value = layer
        if not isinstance(key, torch.Tensor) or not isinstance(item_value, torch.Tensor):
            raise TypeError(f"Legacy cache layer {layer_index} must contain tensors")
        if key.ndim < 2 or item_value.ndim < 2:
            raise ValueError(f"Legacy cache layer {layer_index} tensors must have a sequence dimension")
        if key.shape[-2] != item_value.shape[-2]:
            raise ValueError(f"Legacy cache layer {layer_index} key/value lengths differ")
        if sequence_length is None:
            sequence_length = key.shape[-2]
        elif key.shape[-2] != sequence_length:
            raise ValueError("All legacy cache layers must have the same sequence length")
    return value


def cache_sequence_length(cache: KVCache) -> int:
    if isinstance(cache, (DynamicCache, StaticCache)):
        return int(cache.get_seq_length())
    if not cache:
        return 0
    return cache[0][0].shape[-2]


def crop_cache(cache: KVCache, max_length: int) -> KVCache:
    if max_length < 0:
        raise ValueError(f"Cannot crop cache to invalid length {max_length}")
    if isinstance(cache, StaticCache):
        if max_length > cache.max_cache_len:
            raise ValueError(
                f"Cannot crop StaticCache with capacity {cache.max_cache_len} "
                f"to invalid length {max_length}"
            )
        for key, value in zip(cache.key_cache, cache.value_cache, strict=True):
            key[..., max_length:, :].zero_()
            value[..., max_length:, :].zero_()
        return cache
    current_length = cache_sequence_length(cache)
    if max_length > current_length:
        raise ValueError(
            f"Cannot crop cache of length {current_length} to invalid length {max_length}"
        )
    if isinstance(cache, DynamicCache):
        cache.crop(max_length)
        return cache
    return tuple(
        (
            key[..., :max_length, :],
            value[..., :max_length, :],
        )
        for key, value in cache
    )


def validate_cache_length(cache: KVCache, expected_length: int) -> None:
    actual_length = cache_sequence_length(cache)
    if actual_length != expected_length:
        raise RuntimeError(
            f"Model returned cache length {actual_length}; expected {expected_length}"
        )


def validate_cache_state(state: CacheState, expected_length: int) -> None:
    if state.sequence_length != expected_length:
        raise RuntimeError(
            f"Logical cache length {state.sequence_length}; expected {expected_length}"
        )
    if not isinstance(state.cache, StaticCache):
        validate_cache_length(state.cache, expected_length)


def advance_cache_state(
    state: CacheState,
    updated_cache: object,
    processed_tokens: int,
) -> CacheState:
    if processed_tokens < 1:
        raise ValueError("Cache advancement must process at least one token")
    cache = require_kv_cache(updated_cache)
    if isinstance(state.cache, StaticCache) and cache is not state.cache:
        raise RuntimeError("StaticCache model calls must update and return the same cache object")
    updated_state = CacheState(cache, state.sequence_length + processed_tokens)
    validate_cache_state(updated_state, updated_state.sequence_length)
    return updated_state


def rollback_cache_state(state: CacheState, max_length: int) -> CacheState:
    if max_length < 0 or max_length > state.sequence_length:
        raise ValueError(
            f"Cannot roll back cache of length {state.sequence_length} "
            f"to invalid length {max_length}"
        )
    if isinstance(state.cache, StaticCache):
        for key, value in zip(
            state.cache.key_cache,
            state.cache.value_cache,
            strict=True,
        ):
            key[..., max_length : state.sequence_length, :].zero_()
            value[..., max_length : state.sequence_length, :].zero_()
        cache = state.cache
    else:
        cache = crop_cache(state.cache, max_length)
    rolled_back = CacheState(cache, max_length)
    validate_cache_state(rolled_back, max_length)
    return rolled_back


def cache_position(
    state: CacheState,
    processed_tokens: int,
    device: torch.device,
) -> torch.Tensor:
    if processed_tokens < 1:
        raise ValueError("Cache position requires at least one token")
    end = state.sequence_length + processed_tokens
    if isinstance(state.cache, StaticCache) and end > state.cache.max_cache_len:
        raise ValueError(
            f"StaticCache capacity {state.cache.max_cache_len} is too small for "
            f"sequence length {end}"
        )
    return torch.arange(state.sequence_length, end, device=device)


def model_attention_mask(
    state: CacheState,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    if not isinstance(state.cache, StaticCache):
        return attention_mask
    if attention_mask.shape[1] > state.cache.max_cache_len:
        raise ValueError(
            f"Attention mask length {attention_mask.shape[1]} exceeds StaticCache "
            f"capacity {state.cache.max_cache_len}"
        )
    if attention_mask.shape[1] == state.cache.max_cache_len:
        return attention_mask
    padding = torch.zeros(
        (attention_mask.shape[0], state.cache.max_cache_len - attention_mask.shape[1]),
        dtype=attention_mask.dtype,
        device=attention_mask.device,
    )
    return torch.cat((attention_mask, padding), dim=1)
