from __future__ import annotations

import time

import pytest
import torch

from continuous_batching.cache import KVCacheManager
from continuous_batching.config import InferenceConfig
from continuous_batching.engine import ModelEngine, build_ragged_batch
from continuous_batching.models import GenerationRequest
from continuous_batching.scheduler import BatchScheduler
from tests.continuous_batching_fakes import FakeTokenizer, IncrementModel


def make_scheduler(
    mode: str,
    *,
    max_batch_size: int = 4,
    static_batch_size: int = 2,
    token_budget: int | None = None,
) -> tuple[BatchScheduler, IncrementModel]:
    config = InferenceConfig(
        model_name="fake",
        dtype="float32",
        max_batch_size=max_batch_size,
        max_seq_len=32,
        kv_cache_budget_tokens=token_budget,
        default_max_new_tokens=4,
        device="cpu",
        mode=mode,
        static_batch_size=static_batch_size,
    )
    model = IncrementModel()
    engine = ModelEngine(config, model=model, tokenizer=FakeTokenizer())
    return BatchScheduler(config, engine), model


def test_hand_built_ragged_masks_and_position_ids() -> None:
    batch = build_ragged_batch([[4, 5, 6], [8]], pad_token_id=0, device="cpu")

    assert torch.equal(batch.input_ids, torch.tensor([[4, 5, 6], [8, 0, 0]]))
    assert torch.equal(batch.attention_mask, torch.tensor([[1, 1, 1], [1, 0, 0]]))
    assert torch.equal(batch.position_ids, torch.tensor([[0, 1, 2], [0, 0, 0]]))
    assert torch.equal(batch.sequence_lengths, torch.tensor([3, 1]))


def test_slot_reuse_clears_all_cache_state() -> None:
    manager = KVCacheManager(max_slots=1, token_budget=10)
    slot = manager.allocate("first", 5)
    assert slot == 0
    marker = object()
    manager.update(slot, "first", [1, 2], marker)
    manager.free(slot, "first")
    reused = manager.allocate("second", 3)

    assert reused == slot
    snapshot = manager.snapshot(reused)
    assert snapshot.request_id == "second"
    assert snapshot.capacity_tokens == 3
    assert snapshot.token_ids == []
    assert snapshot.past_key_values is None


def test_low_budget_queues_without_overallocation() -> None:
    scheduler, _ = make_scheduler("continuous", token_budget=8)
    first = GenerationRequest("a", request_id="first", max_new_tokens=4)
    second = GenerationRequest("b", request_id="second", max_new_tokens=4)
    scheduler.submit(first)
    scheduler.submit(second)

    scheduler.step()

    assert [state.request_id for state in scheduler.active] == ["first"]
    assert [state.request_id for state in scheduler.pending] == ["second"]
    assert scheduler.engine.cache_manager.used_tokens == 5


def test_greedy_tokens_equal_for_all_modes_and_arrival_orders() -> None:
    requests = [
        GenerationRequest("a", request_id="one", max_new_tokens=2),
        GenerationRequest("bc", request_id="two", max_new_tokens=4),
        GenerationRequest("def", request_id="three", max_new_tokens=3),
    ]
    outputs: dict[str, dict[str, list[int]]] = {}
    for mode, order in (
        ("sequential", requests),
        ("static", list(reversed(requests))),
        ("continuous", [requests[1], requests[0], requests[2]]),
    ):
        scheduler, _ = make_scheduler(mode)
        states = scheduler.run_until_complete(order)
        outputs[mode] = {
            state.request_id: state.generated_token_ids for state in states
        }

    assert outputs["sequential"] == outputs["static"] == outputs["continuous"]


def test_continuous_replaces_short_request_before_long_finishes() -> None:
    scheduler, model = make_scheduler("continuous", max_batch_size=2)
    scheduler.submit(GenerationRequest("a", request_id="long", max_new_tokens=5))
    scheduler.submit(GenerationRequest("b", request_id="short", max_new_tokens=1))
    scheduler.step()
    scheduler.submit(GenerationRequest("c", request_id="replacement", max_new_tokens=2))
    scheduler.step()

    assert "replacement" in [state.request_id for state in scheduler.active]
    assert "long" in [state.request_id for state in scheduler.active]
    assert len(model.forward_batches[-1]) == 2


@pytest.mark.parametrize("mode", ["sequential", "static", "continuous"])
def test_n_inputs_produce_n_unique_outputs(mode: str) -> None:
    scheduler, _ = make_scheduler(mode)
    requests = [
        GenerationRequest(str(index + 1), request_id=f"request-{index}")
        for index in range(8)
    ]
    states = scheduler.run_until_complete(requests)

    assert len(states) == len(requests)
    assert len({state.request_id for state in states}) == len(requests)


def test_reproducible_per_request_sampling_independent_of_batch_order() -> None:
    first = GenerationRequest(
        "a", request_id="first", max_new_tokens=3, temperature=1.0, seed=7
    )
    second = GenerationRequest(
        "b", request_id="second", max_new_tokens=3, temperature=1.0, seed=9
    )
    scheduler_one, _ = make_scheduler("continuous")
    scheduler_two, _ = make_scheduler("continuous")

    forward = scheduler_one.run_until_complete([first, second])
    reverse = scheduler_two.run_until_complete([second, first])

    assert {
        state.request_id: state.generated_token_ids for state in forward
    } == {
        state.request_id: state.generated_token_ids for state in reverse
    }

