from __future__ import annotations

from typing import TypeAlias

import torch
from transformers.cache_utils import DynamicCache

LegacyLayerCache: TypeAlias = tuple[torch.Tensor, torch.Tensor]
LegacyCache: TypeAlias = tuple[LegacyLayerCache, ...]
KVCache: TypeAlias = DynamicCache | LegacyCache


def require_kv_cache(value: object) -> KVCache:
    if isinstance(value, DynamicCache):
        return value
    if not isinstance(value, tuple):
        raise TypeError(
            "Expected transformers.DynamicCache or a legacy tuple cache, "
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
    if isinstance(cache, DynamicCache):
        return cache.get_seq_length()
    if not cache:
        return 0
    return cache[0][0].shape[-2]


def crop_cache(cache: KVCache, max_length: int) -> KVCache:
    current_length = cache_sequence_length(cache)
    if max_length < 0 or max_length > current_length:
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
