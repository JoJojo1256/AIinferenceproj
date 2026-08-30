import pytest
import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache, StaticCache
from transformers.masking_utils import create_causal_mask

from specdec.cache import (
    CacheState,
    cache_position,
    cache_sequence_length,
    crop_cache,
    model_attention_mask,
    require_kv_cache,
    rollback_cache_state,
)
from specdec.baseline import generate_baseline
from specdec.models import ModelBundle
from specdec.speculative import generate_speculative


def _states(length: int) -> tuple[torch.Tensor, torch.Tensor]:
    key = torch.arange(length, dtype=torch.float32).reshape(1, 1, length, 1)
    return key, key.clone()


def test_dynamic_cache_crop_updates_sequence_length() -> None:
    cache = DynamicCache()
    key, value = _states(5)
    cache.update(key, value, layer_idx=0)

    cropped = crop_cache(cache, 3)

    assert cropped is cache
    assert cache_sequence_length(cropped) == 3
    assert cache.key_cache[0].shape[-2] == 3


def test_legacy_cache_crop_preserves_prefix() -> None:
    key, value = _states(5)
    cache = require_kv_cache(((key, value),))

    cropped = crop_cache(cache, 2)

    assert cache_sequence_length(cropped) == 2
    assert torch.equal(cropped[0][0], key[..., :2, :])


def test_static_cache_rollback_clears_rejected_slots_and_reuses_positions() -> None:
    config = LlamaConfig(
        hidden_size=1,
        intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=8,
    )
    cache = StaticCache(config, max_batch_size=1, max_cache_len=8)
    key, value = _states(5)
    cache.update(
        key,
        value,
        layer_idx=0,
        cache_kwargs={"cache_position": torch.arange(5)},
    )

    rolled_back = rollback_cache_state(CacheState(cache, 5), 3)

    assert rolled_back.cache is cache
    assert rolled_back.sequence_length == 3
    assert torch.equal(cache.key_cache[0][..., :3, :], key[..., :3, :])
    assert torch.count_nonzero(cache.key_cache[0][..., 3:, :]) == 0
    assert torch.equal(
        cache_position(rolled_back, 2, torch.device("cpu")),
        torch.tensor([3, 4]),
    )


def test_static_cache_consecutive_rejections_preserve_prefix_and_clear_stale_slots() -> None:
    config = LlamaConfig(
        hidden_size=1,
        intermediate_size=4,
        num_hidden_layers=1,
        num_attention_heads=1,
        num_key_value_heads=1,
        max_position_embeddings=12,
    )
    cache = StaticCache(config, max_batch_size=1, max_cache_len=12)
    initial_states = torch.tensor([1.0, 2.0]).reshape(1, 1, 2, 1)
    cache.update(
        initial_states,
        initial_states.clone(),
        layer_idx=0,
        cache_kwargs={"cache_position": torch.arange(2)},
    )
    first_verification = torch.arange(3.0, 8.0).reshape(1, 1, 5, 1)
    cache.update(
        first_verification,
        first_verification.clone(),
        layer_idx=0,
        cache_kwargs={"cache_position": torch.arange(2, 7)},
    )

    first_rollback = rollback_cache_state(CacheState(cache, 7), 3)
    fresh_after_first = StaticCache(config, max_batch_size=1, max_cache_len=12)
    fresh_first_prefix = torch.tensor([1.0, 2.0, 3.0]).reshape(1, 1, 3, 1)
    fresh_after_first.update(
        fresh_first_prefix,
        fresh_first_prefix.clone(),
        layer_idx=0,
        cache_kwargs={"cache_position": torch.arange(3)},
    )
    assert first_rollback.sequence_length == 3
    assert torch.equal(
        cache.key_cache[0][..., :3, :],
        fresh_after_first.key_cache[0][..., :3, :],
    )
    assert torch.count_nonzero(cache.key_cache[0][..., 3:, :]) == 0

    second_verification = torch.arange(8.0, 14.0).reshape(1, 1, 6, 1)
    cache.update(
        second_verification,
        second_verification.clone(),
        layer_idx=0,
        cache_kwargs={"cache_position": torch.arange(3, 9)},
    )
    second_rollback = rollback_cache_state(CacheState(cache, 9), 4)
    fresh_after_second = StaticCache(config, max_batch_size=1, max_cache_len=12)
    fresh_second_prefix = torch.tensor([1.0, 2.0, 3.0, 8.0]).reshape(1, 1, 4, 1)
    fresh_after_second.update(
        fresh_second_prefix,
        fresh_second_prefix.clone(),
        layer_idx=0,
        cache_kwargs={"cache_position": torch.arange(4)},
    )

    assert second_rollback.sequence_length == 4
    assert torch.equal(
        cache.key_cache[0][..., :4, :],
        fresh_after_second.key_cache[0][..., :4, :],
    )
    assert torch.equal(
        cache.value_cache[0][..., :4, :],
        fresh_after_second.value_cache[0][..., :4, :],
    )
    assert torch.count_nonzero(cache.key_cache[0][..., 4:, :]) == 0
    assert torch.count_nonzero(cache.value_cache[0][..., 4:, :]) == 0

    padded_mask = model_attention_mask(
        second_rollback,
        torch.ones((1, 5), dtype=torch.long),
    )
    assert torch.equal(padded_mask[:, :5], torch.ones((1, 5), dtype=torch.long))
    assert torch.count_nonzero(padded_mask[:, 5:]) == 0
    causal_mask = create_causal_mask(
        config=config,
        input_embeds=torch.zeros((1, 1, 1)),
        attention_mask=padded_mask,
        cache_position=torch.tensor([4]),
        past_key_values=cache,
        position_ids=torch.tensor([[4]]),
    )
    assert causal_mask is not None
    assert torch.count_nonzero(causal_mask[..., :5]) == 0
    assert torch.all(causal_mask[..., 5:] == torch.finfo(torch.float32).min)


def test_cache_helpers_reject_missing_or_unknown_cache() -> None:
    with pytest.raises(TypeError, match="Expected"):
        require_kv_cache(None)
    key, _ = _states(2)
    with pytest.raises(TypeError, match="key, value"):
        require_kv_cache(((key,),))


class _TinyTokenizer:
    eos_token_id = 7
    pad_token_id = 0
    chat_template = None

    def __call__(self, text: str, **_: object) -> BatchEncoding:
        del text
        return BatchEncoding(
            {
                "input_ids": torch.tensor([[0, 1]], dtype=torch.long),
                "attention_mask": torch.tensor([[1, 1]], dtype=torch.long),
            }
        )

    def decode(self, token_ids: list[int], **_: object) -> str:
        return " ".join(str(token_id) for token_id in token_ids)

    def get_vocab(self) -> dict[str, int]:
        return {str(token_id): token_id for token_id in range(8)}


def _tiny_llama_bundle(model_id: str, state_dict: dict[str, torch.Tensor]) -> ModelBundle:
    config = LlamaConfig(
        vocab_size=8,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    model = LlamaForCausalLM(config)
    model.load_state_dict(state_dict)
    model.eval()
    return ModelBundle(
        model=model,
        tokenizer=_TinyTokenizer(),
        model_id=model_id,
        revision=None,
    )


def test_tiny_llama_dynamic_cache_matches_greedy_baseline() -> None:
    torch.manual_seed(7)
    source = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    state_dict = source.state_dict()

    baseline = generate_baseline(
        _tiny_llama_bundle("target", state_dict),
        "prompt",
        max_new_tokens=4,
        temperature=0,
    )
    speculative = generate_speculative(
        _tiny_llama_bundle("target", state_dict),
        _tiny_llama_bundle("draft", state_dict),
        "prompt",
        max_new_tokens=4,
        speculation_length=2,
        temperature=0,
    )

    assert speculative.output_token_ids == baseline.output_token_ids
    assert speculative.target_processed_tokens <= 2 + 4
    assert speculative.draft_processed_tokens <= 2 + 4


def test_tiny_llama_static_cache_matches_greedy_baseline() -> None:
    torch.manual_seed(11)
    source = LlamaForCausalLM(
        LlamaConfig(
            vocab_size=8,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=2,
            max_position_embeddings=32,
        )
    )
    state_dict = source.state_dict()

    baseline = generate_baseline(
        _tiny_llama_bundle("target", state_dict),
        "prompt",
        max_new_tokens=4,
        temperature=0,
    )
    speculative = generate_speculative(
        _tiny_llama_bundle("target", state_dict),
        _tiny_llama_bundle("draft", state_dict),
        "prompt",
        max_new_tokens=4,
        speculation_length=2,
        temperature=0,
        draft_cache_implementation="static",
        target_cache_implementation="static",
        static_cache_max_length=8,
    )

    assert speculative.output_token_ids == baseline.output_token_ids
