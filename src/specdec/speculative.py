from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from specdec.metrics import SpeculativeGenerationMetrics
from specdec.models import ModelBundle, encode_prompt, validate_shared_tokenizer
from specdec.sampling import (
    modified_rejection_sample,
    probabilities_from_logits,
    sample_distribution,
)


@dataclass(frozen=True)
class DraftProposal:
    token_ids: list[int]
    probabilities: list[torch.Tensor]


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _model_device(bundle: ModelBundle) -> torch.device:
    return next(bundle.model.parameters()).device


def _append_token(input_ids: torch.Tensor, token_id: int) -> torch.Tensor:
    token = torch.tensor([[token_id]], device=input_ids.device, dtype=input_ids.dtype)
    return torch.cat((input_ids, token), dim=1)


def _termination_token_ids(bundle: ModelBundle) -> set[int]:
    token_ids = {bundle.tokenizer.eos_token_id}
    end_of_turn_id = bundle.tokenizer.get_vocab().get("<|eot_id|>")
    if end_of_turn_id is not None:
        token_ids.add(end_of_turn_id)
    return {token_id for token_id in token_ids if token_id is not None}


@torch.inference_mode()
def propose_tokens(
    draft: ModelBundle,
    prefix_ids: torch.Tensor,
    *,
    speculation_length: int,
    temperature: float,
    generator: torch.Generator | None,
) -> DraftProposal:
    if speculation_length < 1:
        raise ValueError("speculation_length must be at least 1")

    draft_ids = prefix_ids
    model_input_ids = prefix_ids
    past_key_values = None
    proposed_ids: list[int] = []
    proposal_probabilities: list[torch.Tensor] = []
    termination_token_ids = _termination_token_ids(draft)

    for _ in range(speculation_length):
        attention_mask = torch.ones_like(draft_ids)
        outputs = draft.model(
            input_ids=model_input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        logits = outputs.logits[0, -1]
        if temperature == 0:
            token_id = int(torch.argmax(logits).item())
            probabilities = torch.nn.functional.one_hot(
                torch.tensor(token_id, device=logits.device),
                num_classes=logits.numel(),
            ).float()
        else:
            probabilities = probabilities_from_logits(logits, temperature)
            token_id = sample_distribution(probabilities, generator=generator)

        proposed_ids.append(token_id)
        proposal_probabilities.append(probabilities)
        draft_ids = _append_token(draft_ids, token_id)
        past_key_values = getattr(outputs, "past_key_values", None)
        if past_key_values is None:
            model_input_ids = draft_ids
        else:
            model_input_ids = draft_ids[:, -1:]
        if token_id in termination_token_ids:
            break

    return DraftProposal(token_ids=proposed_ids, probabilities=proposal_probabilities)


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
) -> SpeculativeGenerationMetrics:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")
    if speculation_length < 1:
        raise ValueError("speculation_length must be at least 1")
    if temperature < 0:
        raise ValueError("temperature must be non-negative")

    validate_shared_tokenizer(target, draft)
    target_device = _model_device(target)
    draft_device = _model_device(draft)
    if target_device != draft_device:
        raise ValueError("Target and draft models must be on the same device")

    encoded = encode_prompt(target.tokenizer, prompt, target_device)
    prefix_ids = encoded["input_ids"]
    generator = torch.Generator(device=target_device).manual_seed(seed)
    termination_token_ids = _termination_token_ids(target)
    output_ids: list[int] = []
    decode_latencies_ms: list[float] = []
    block_latencies_ms: list[float] = []
    proposed_tokens = 0
    accepted_tokens = 0
    target_forward_passes = 0

    _synchronize(target_device)
    total_start = time.perf_counter()
    ttft_ms: float | None = None

    while len(output_ids) < max_new_tokens:
        _synchronize(target_device)
        block_start = time.perf_counter()
        remaining = max_new_tokens - len(output_ids)
        proposal = propose_tokens(
            draft,
            prefix_ids,
            speculation_length=min(speculation_length, remaining),
            temperature=temperature,
            generator=generator,
        )
        proposed_tokens += len(proposal.token_ids)
        block_prefix_length = prefix_ids.shape[1]
        block_output_start = len(output_ids)
        proposed_tensor = torch.tensor(
            [proposal.token_ids],
            device=target_device,
            dtype=prefix_ids.dtype,
        )
        verification_ids = torch.cat((prefix_ids, proposed_tensor), dim=1)
        attention_mask = torch.ones_like(verification_ids)

        target_outputs = target.model(
            input_ids=verification_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        target_forward_passes += 1
        target_logits = target_outputs.logits[0]

        rejected = False
        for proposal_index, proposed_token_id in enumerate(proposal.token_ids):
            prediction_index = block_prefix_length - 1 + proposal_index
            logits = target_logits[prediction_index]

            if temperature == 0:
                target_token_id = int(torch.argmax(logits).item())
                if target_token_id == proposed_token_id:
                    accepted_tokens += 1
                    emitted_token_id = proposed_token_id
                else:
                    emitted_token_id = target_token_id
                    rejected = True
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
                    rejected = True

            output_ids.append(emitted_token_id)
            prefix_ids = _append_token(prefix_ids, emitted_token_id)
            if emitted_token_id in termination_token_ids or len(output_ids) >= max_new_tokens:
                rejected = True
                break
            if rejected:
                break

        if not rejected and len(output_ids) < max_new_tokens:
            bonus_logits = target_logits[block_prefix_length + len(proposal.token_ids) - 1]
            if temperature == 0:
                bonus_token_id = int(torch.argmax(bonus_logits).item())
            else:
                bonus_token_id = sample_distribution(
                    probabilities_from_logits(bonus_logits, temperature),
                    generator=generator,
                )
            output_ids.append(bonus_token_id)
            prefix_ids = _append_token(prefix_ids, bonus_token_id)

        _synchronize(target_device)
        block_latency_ms = (time.perf_counter() - block_start) * 1_000
        emitted_in_block = len(output_ids) - block_output_start
        block_latencies_ms.append(block_latency_ms)
        if ttft_ms is None:
            ttft_ms = block_latency_ms
            decode_latencies_ms.extend([0.0] * max(0, emitted_in_block - 1))
        else:
            decode_latencies_ms.append(block_latency_ms)
            decode_latencies_ms.extend([0.0] * max(0, emitted_in_block - 1))

        if output_ids[-1] in termination_token_ids:
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
    )
