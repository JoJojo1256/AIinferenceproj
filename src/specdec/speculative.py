from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable

import torch
from transformers.cache_utils import StaticCache

from specdec.cache import (
    CacheImplementation,
    CacheState,
    advance_cache_state,
    cache_position,
    model_attention_mask,
    require_kv_cache,
    rollback_cache_state,
    validate_cache_state,
)
from specdec.metrics import SpeculativeGenerationMetrics
from specdec.models import ModelBundle, encode_prompt, validate_shared_tokenizer
from specdec.sampling import (
    modified_rejection_sample,
    probabilities_from_logits,
    sample_distribution,
)


@dataclass(frozen=True)
class CachedModelState:
    cache_state: CacheState
    next_logits: torch.Tensor


@dataclass(frozen=True)
class DraftProposal:
    token_ids: list[int]
    probabilities: list[torch.Tensor]
    state: CachedModelState
    processed_tokens: int


ModelForward = Callable[..., Any]


@dataclass
class _StageTimer:
    start_cpu: float
    start_event: torch.cuda.Event | None
    end_cpu: float | None = None
    end_event: torch.cuda.Event | None = None

    @classmethod
    def start(cls, device: torch.device) -> _StageTimer:
        event = None
        if device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
        return cls(start_cpu=time.perf_counter(), start_event=event)

    def stop(self) -> None:
        self.end_cpu = time.perf_counter()
        if self.start_event is not None:
            self.end_event = torch.cuda.Event(enable_timing=True)
            self.end_event.record()

    def elapsed_ms(self) -> float:
        if self.end_cpu is None:
            raise RuntimeError("Stage timer has not been stopped")
        if self.start_event is not None:
            if self.end_event is None:
                raise RuntimeError("CUDA stage timer has no end event")
            return self.start_event.elapsed_time(self.end_event)
        return (self.end_cpu - self.start_cpu) * 1_000


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_device(bundle: ModelBundle) -> torch.device:
    return next(bundle.model.parameters()).device


def _append_attention_tokens(attention_mask: torch.Tensor, count: int) -> torch.Tensor:
    if count < 0:
        raise ValueError("Attention token count must be non-negative")
    if count == 0:
        return attention_mask
    extension = torch.ones(
        (attention_mask.shape[0], count),
        device=attention_mask.device,
        dtype=attention_mask.dtype,
    )
    return torch.cat((attention_mask, extension), dim=1)


def _termination_token_ids(bundle: ModelBundle) -> set[int]:
    token_ids = {bundle.tokenizer.eos_token_id}
    end_of_turn_id = bundle.tokenizer.get_vocab().get("<|eot_id|>")
    if end_of_turn_id is not None:
        token_ids.add(end_of_turn_id)
    return {token_id for token_id in token_ids if token_id is not None}


def _select_token(
    logits: torch.Tensor,
    temperature: float,
    generator: torch.Generator | None,
) -> tuple[int, torch.Tensor]:
    if temperature == 0:
        token_id = int(torch.argmax(logits).item())
        probabilities = torch.nn.functional.one_hot(
            torch.tensor(token_id, device=logits.device),
            num_classes=logits.numel(),
        ).float()
        return token_id, probabilities
    probabilities = probabilities_from_logits(logits, temperature)
    return sample_distribution(probabilities, generator=generator), probabilities


def next_adaptive_speculation_length(
    current_length: int,
    *,
    all_tokens_accepted: bool,
    max_length: int | None = None,
) -> int:
    if current_length < 1:
        raise ValueError("Current speculation length must be at least 1")
    if max_length is not None and max_length < 1:
        raise ValueError("Maximum speculation length must be at least 1")
    if all_tokens_accepted:
        next_length = current_length + 2
        return min(next_length, max_length) if max_length is not None else next_length
    return max(1, current_length - 1)


def compile_draft_forward(draft: ModelBundle) -> ModelForward:
    cached = draft.model.__dict__.get("_specdec_compiled_forward")
    if cached is not None:
        return cached
    compiled = torch.compile(draft.model.forward, mode="reduce-overhead")
    draft.model.__dict__["_specdec_compiled_forward"] = compiled
    return compiled


def _reusable_static_cache(
    bundle: ModelBundle,
    *,
    max_cache_length: int,
    device: torch.device,
) -> StaticCache:
    if max_cache_length < 1:
        raise ValueError("Static cache length must be at least 1")
    config = getattr(bundle.model, "config", None)
    if config is None:
        raise TypeError("StaticCache requires a model with a Transformers config")
    dtype = next(bundle.model.parameters()).dtype
    cache_key = (max_cache_length, str(device), dtype)
    cache_pool = bundle.model.__dict__.setdefault("_specdec_static_caches", {})
    cache = cache_pool.get(cache_key)
    if cache is None:
        cache = StaticCache(
            config=config,
            max_batch_size=1,
            max_cache_len=max_cache_length,
            device=device,
            dtype=dtype,
        )
        cache_pool[cache_key] = cache
    else:
        cache.reset()
    return cache


def _prefill_model(
    bundle: ModelBundle,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    cache_implementation: CacheImplementation,
    max_cache_length: int,
) -> CachedModelState:
    if cache_implementation == "static":
        initial_cache: object | None = _reusable_static_cache(
            bundle,
            max_cache_length=max_cache_length,
            device=input_ids.device,
        )
        initial_state = CacheState(require_kv_cache(initial_cache), 0)
        forward_attention_mask = model_attention_mask(initial_state, attention_mask)
    elif cache_implementation == "dynamic":
        initial_cache = None
        forward_attention_mask = attention_mask
    else:
        raise ValueError(f"Unsupported cache implementation {cache_implementation!r}")

    outputs = bundle.model(
        input_ids=input_ids,
        attention_mask=forward_attention_mask,
        past_key_values=initial_cache,
        use_cache=True,
        cache_position=torch.arange(input_ids.shape[1], device=input_ids.device),
    )
    cache = require_kv_cache(getattr(outputs, "past_key_values", None))
    cache_state = CacheState(cache, input_ids.shape[1])
    validate_cache_state(cache_state, input_ids.shape[1])
    return CachedModelState(
        cache_state=cache_state,
        next_logits=outputs.logits[0, -1],
    )


def _advance_model(
    bundle: ModelBundle,
    state: CachedModelState,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    *,
    forward: ModelForward | None = None,
    clone_logits: bool = False,
) -> tuple[CachedModelState, torch.Tensor]:
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("Cached model input must contain at least one token for one sequence")
    previous_length = state.cache_state.sequence_length
    expected_length = previous_length + input_ids.shape[1]
    if attention_mask.shape != (1, expected_length):
        raise ValueError(
            f"Attention mask shape {tuple(attention_mask.shape)} does not match "
            f"cached sequence length {expected_length}"
        )

    model_forward = forward if forward is not None else bundle.model
    outputs = model_forward(
        input_ids=input_ids,
        attention_mask=model_attention_mask(state.cache_state, attention_mask),
        past_key_values=state.cache_state.cache,
        use_cache=True,
        cache_position=cache_position(
            state.cache_state,
            input_ids.shape[1],
            input_ids.device,
        ),
    )
    updated_cache_state = advance_cache_state(
        state.cache_state,
        getattr(outputs, "past_key_values", None),
        input_ids.shape[1],
    )
    validate_cache_state(updated_cache_state, expected_length)
    logits = outputs.logits[0]
    if clone_logits:
        # reduce-overhead/CUDA graphs reuse output buffers across replay.
        logits = logits.clone()
    if logits.shape[0] != input_ids.shape[1]:
        raise RuntimeError("Model must return one logits row per incremental input token")
    return (
        CachedModelState(
            cache_state=updated_cache_state,
            next_logits=logits[-1],
        ),
        logits,
    )


@torch.inference_mode()
def propose_tokens(
    draft: ModelBundle,
    state: CachedModelState,
    attention_mask: torch.Tensor,
    *,
    speculation_length: int,
    temperature: float,
    generator: torch.Generator | None,
    forward: ModelForward | None = None,
    clone_logits: bool = False,
) -> DraftProposal:
    if speculation_length < 1:
        raise ValueError("speculation_length must be at least 1")

    proposed_ids: list[int] = []
    proposal_probabilities: list[torch.Tensor] = []
    termination_token_ids = _termination_token_ids(draft)
    staged_state = state
    staged_attention_mask = attention_mask
    processed_tokens = 0

    for _ in range(speculation_length):
        token_id, probabilities = _select_token(
            staged_state.next_logits,
            temperature,
            generator,
        )
        proposed_ids.append(token_id)
        proposal_probabilities.append(probabilities)
        model_input_ids = torch.tensor(
            [[token_id]],
            device=attention_mask.device,
            dtype=torch.long,
        )
        staged_attention_mask = _append_attention_tokens(staged_attention_mask, 1)
        staged_state, _ = _advance_model(
            draft,
            staged_state,
            model_input_ids,
            staged_attention_mask,
            forward=forward,
            clone_logits=clone_logits,
        )
        processed_tokens += 1
        if token_id in termination_token_ids:
            break

    return DraftProposal(
        token_ids=proposed_ids,
        probabilities=proposal_probabilities,
        state=staged_state,
        processed_tokens=processed_tokens,
    )


@torch.inference_mode()
def generate_speculative(
    target: ModelBundle,
    draft: ModelBundle,
    prompt: str,
    *,
    max_new_tokens: int,
    speculation_length: int = 4,
    temperature: float = 0.0,
    seed: int = 0,
    compile_draft: bool = False,
    draft_cache_implementation: CacheImplementation = "dynamic",
    target_cache_implementation: CacheImplementation = "dynamic",
    adaptive_speculation: bool = False,
    max_speculation_length: int | None = None,
    static_cache_max_length: int | None = None,
) -> SpeculativeGenerationMetrics:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if speculation_length < 1:
        raise ValueError("speculation_length must be at least 1")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if draft_cache_implementation not in {"dynamic", "static"}:
        raise ValueError("draft_cache_implementation must be 'dynamic' or 'static'")
    if target_cache_implementation not in {"dynamic", "static"}:
        raise ValueError("target_cache_implementation must be 'dynamic' or 'static'")
    if compile_draft and draft_cache_implementation != "static":
        raise ValueError("Compiled draft decoding requires draft_cache_implementation='static'")
    if compile_draft and static_cache_max_length is None:
        raise ValueError(
            "Compiled draft decoding requires an explicit static_cache_max_length "
            "to keep decode shapes fixed across warmup and measured runs"
        )
    if max_speculation_length is not None:
        if max_speculation_length < 1:
            raise ValueError("max_speculation_length must be at least 1")
        if speculation_length > max_speculation_length:
            raise ValueError("speculation_length cannot exceed max_speculation_length")

    validate_shared_tokenizer(target, draft)
    target_device = _model_device(target)
    draft_device = _model_device(draft)
    if target_device != draft_device:
        raise ValueError("Target and draft models must be on the same device")

    encoded = encode_prompt(target.tokenizer, prompt, target_device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    required_cache_length = input_ids.shape[1] + max_new_tokens
    resolved_static_cache_length = static_cache_max_length or required_cache_length
    if resolved_static_cache_length < required_cache_length:
        raise ValueError(
            f"static_cache_max_length={resolved_static_cache_length} is smaller than "
            f"the required prompt + generation length {required_cache_length}"
        )
    generator = torch.Generator(device=target_device).manual_seed(seed)
    termination_token_ids = _termination_token_ids(target)
    output_ids: list[int] = []
    decode_latencies_ms: list[float] = []
    block_latencies_ms: list[float] = []
    proposed_tokens = 0
    accepted_tokens = 0
    target_forward_passes = 0
    target_processed_tokens = input_ids.shape[1]
    draft_processed_tokens = input_ids.shape[1]
    draft_proposal_time_ms = 0.0
    target_verification_time_ms = 0.0
    sampling_overhead_time_ms = 0.0
    pending_token_id: int | None = None
    current_speculation_length = speculation_length
    realized_speculation_lengths: list[int] = []
    draft_forward = compile_draft_forward(draft) if compile_draft else None

    _synchronize(target_device)
    total_start = time.perf_counter()
    prefill_timer = _StageTimer.start(target_device)
    target_state = _prefill_model(
        target,
        input_ids,
        attention_mask,
        cache_implementation=target_cache_implementation,
        max_cache_length=resolved_static_cache_length,
    )
    draft_state = _prefill_model(
        draft,
        input_ids,
        attention_mask,
        cache_implementation=draft_cache_implementation,
        max_cache_length=resolved_static_cache_length,
    )
    prefill_timer.stop()
    _synchronize(target_device)
    prefill_time_ms = prefill_timer.elapsed_ms()
    ttft_ms: float | None = None

    while len(output_ids) < max_new_tokens:
        _synchronize(target_device)
        block_start = time.perf_counter()
        remaining = max_new_tokens - len(output_ids)
        committed_length = attention_mask.shape[1]
        pending_count = 1 if pending_token_id is not None else 0
        validate_cache_state(
            target_state.cache_state,
            committed_length - pending_count,
        )
        validate_cache_state(
            draft_state.cache_state,
            committed_length - pending_count,
        )
        proposal_timer = _StageTimer.start(target_device)
        if pending_token_id is not None:
            pending_tensor = torch.tensor(
                [[pending_token_id]],
                device=target_device,
                dtype=input_ids.dtype,
            )
            draft_state, _ = _advance_model(
                draft,
                draft_state,
                pending_tensor,
                attention_mask,
                forward=draft_forward,
                clone_logits=compile_draft,
            )
            draft_processed_tokens += 1
        proposal = propose_tokens(
            draft,
            draft_state,
            attention_mask,
            speculation_length=min(current_speculation_length, remaining),
            temperature=temperature,
            generator=generator,
            forward=draft_forward,
            clone_logits=compile_draft,
        )
        proposal_timer.stop()
        realized_speculation_lengths.append(len(proposal.token_ids))
        proposed_tokens += len(proposal.token_ids)
        draft_processed_tokens += proposal.processed_tokens
        block_output_start = len(output_ids)
        proposed_tensor = torch.tensor(
            [proposal.token_ids],
            device=target_device,
            dtype=input_ids.dtype,
        )
        target_input_ids = proposed_tensor
        if pending_token_id is not None:
            target_input_ids = torch.cat((pending_tensor, proposed_tensor), dim=1)
        verification_attention_mask = _append_attention_tokens(
            attention_mask,
            len(proposal.token_ids),
        )
        verification_timer = _StageTimer.start(target_device)
        verified_target_state, target_logits = _advance_model(
            target,
            target_state,
            target_input_ids,
            verification_attention_mask,
        )
        verification_timer.stop()
        target_forward_passes += 1
        target_processed_tokens += target_input_ids.shape[1]

        rejection_index: int | None = None
        terminated = False
        for proposal_index, proposed_token_id in enumerate(proposal.token_ids):
            prediction_index = proposal_index + pending_count - 1
            logits = (
                target_state.next_logits
                if prediction_index < 0
                else target_logits[prediction_index]
            )

            if temperature == 0:
                target_token_id = int(torch.argmax(logits).item())
                if target_token_id == proposed_token_id:
                    accepted_tokens += 1
                    emitted_token_id = proposed_token_id
                else:
                    emitted_token_id = target_token_id
                    rejection_index = proposal_index
            else:
                target_probabilities = probabilities_from_logits(logits, temperature)
                accepted, emitted_token_id = modified_rejection_sample(
                    target_probabilities,
                    proposal.probabilities[proposal_index],
                    proposed_token_id,
                    generator=generator,
                )
                if accepted:
                    accepted_tokens += 1
                else:
                    rejection_index = proposal_index

            output_ids.append(emitted_token_id)
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - total_start) * 1_000
            if emitted_token_id in termination_token_ids or len(output_ids) >= max_new_tokens:
                terminated = emitted_token_id in termination_token_ids
                break
            if rejection_index is not None:
                break

        if rejection_index is not None:
            retained_length = committed_length + rejection_index
            target_cache_state = rollback_cache_state(
                verified_target_state.cache_state,
                retained_length,
            )
            draft_cache_state = rollback_cache_state(
                proposal.state.cache_state,
                retained_length,
            )
            attention_mask = verification_attention_mask[:, :retained_length]
            attention_mask = _append_attention_tokens(attention_mask, 1)
            target_state = CachedModelState(
                cache_state=target_cache_state,
                next_logits=target_state.next_logits,
            )
            draft_state = CachedModelState(
                cache_state=draft_cache_state,
                next_logits=draft_state.next_logits,
            )
            pending_token_id = output_ids[-1]
        else:
            target_state = verified_target_state
            draft_state = proposal.state
            attention_mask = verification_attention_mask
            pending_token_id = None

        if (
            rejection_index is None
            and not terminated
            and len(output_ids) < max_new_tokens
        ):
            bonus_logits = target_logits[
                len(proposal.token_ids) + pending_count - 1
            ]
            if temperature == 0:
                bonus_token_id = int(torch.argmax(bonus_logits).item())
            else:
                bonus_token_id = sample_distribution(
                    probabilities_from_logits(bonus_logits, temperature),
                    generator=generator,
                )
            output_ids.append(bonus_token_id)
            if ttft_ms is None:
                ttft_ms = (time.perf_counter() - total_start) * 1_000
            attention_mask = _append_attention_tokens(attention_mask, 1)
            pending_token_id = bonus_token_id
            if bonus_token_id in termination_token_ids:
                terminated = True

        expected_cache_length = attention_mask.shape[1] - (
            1 if pending_token_id is not None else 0
        )
        validate_cache_state(target_state.cache_state, expected_cache_length)
        validate_cache_state(draft_state.cache_state, expected_cache_length)

        if adaptive_speculation:
            current_speculation_length = next_adaptive_speculation_length(
                current_speculation_length,
                all_tokens_accepted=rejection_index is None,
                max_length=max_speculation_length,
            )

        _synchronize(target_device)
        block_latency_ms = (time.perf_counter() - block_start) * 1_000
        proposal_latency_ms = proposal_timer.elapsed_ms()
        verification_latency_ms = verification_timer.elapsed_ms()
        draft_proposal_time_ms += proposal_latency_ms
        target_verification_time_ms += verification_latency_ms
        sampling_overhead_time_ms += max(
            0.0,
            block_latency_ms - proposal_latency_ms - verification_latency_ms,
        )
        emitted_in_block = len(output_ids) - block_output_start
        block_latencies_ms.append(block_latency_ms)
        if block_output_start == 0:
            decode_latencies_ms.extend([0.0] * max(0, emitted_in_block - 1))
        else:
            decode_latencies_ms.append(block_latency_ms)
            decode_latencies_ms.extend([0.0] * max(0, emitted_in_block - 1))

        if terminated:
            break

    _synchronize(target_device)
    total_latency_ms = (time.perf_counter() - total_start) * 1_000
    output_text = target.tokenizer.decode(output_ids, skip_special_tokens=True)
    return SpeculativeGenerationMetrics(
        prompt=prompt,
        output_text=output_text,
        output_token_ids=output_ids,
        ttft_ms=ttft_ms if ttft_ms is not None else 0.0,
        decode_latencies_ms=decode_latencies_ms,
        total_latency_ms=total_latency_ms,
        proposed_tokens=proposed_tokens,
        accepted_tokens=accepted_tokens,
        target_forward_passes=target_forward_passes,
        block_latencies_ms=block_latencies_ms,
        prefill_time_ms=prefill_time_ms,
        draft_proposal_time_ms=draft_proposal_time_ms,
        target_verification_time_ms=target_verification_time_ms,
        sampling_overhead_time_ms=sampling_overhead_time_ms,
        target_processed_tokens=target_processed_tokens,
        draft_processed_tokens=draft_processed_tokens,
        draft_compiled=compile_draft,
        draft_cache_implementation=draft_cache_implementation,
        target_cache_implementation=target_cache_implementation,
        adaptive_speculation=adaptive_speculation,
        initial_speculation_length=speculation_length,
        realized_speculation_lengths=realized_speculation_lengths,
    )
