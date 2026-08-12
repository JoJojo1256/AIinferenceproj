import pytest
import torch
from transformers import BatchEncoding, LlamaConfig, LlamaForCausalLM
from transformers.cache_utils import DynamicCache

from specdec.cache import cache_sequence_length, crop_cache, require_kv_cache
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
