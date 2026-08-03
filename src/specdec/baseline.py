from __future__ import annotations

import time

import torch

from specdec.metrics import GenerationMetrics
from specdec.models import ModelBundle, encode_prompt


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _select_token(logits: torch.Tensor, temperature: float, generator: torch.Generator | None) -> torch.Tensor:
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    probabilities = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probabilities, num_samples=1, generator=generator)


@torch.inference_mode()
def generate_baseline(
    bundle: ModelBundle,
    prompt: str,
    *,
    max_new_tokens: int,
    temperature: float = 0.0,
    seed: int = 0,
) -> GenerationMetrics:
    if max_new_tokens < 1:
        raise ValueError("max_new_tokens must be at least 1")

    model = bundle.model
    tokenizer = bundle.tokenizer
    device = next(model.parameters()).device
    encoded = encode_prompt(tokenizer, prompt, device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    generator = torch.Generator(device=device).manual_seed(seed)
    output_ids: list[int] = []
    decode_latencies_ms: list[float] = []

    _synchronize(device)
    total_start = time.perf_counter()
    prefill_start = time.perf_counter()
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
    next_token = _select_token(outputs.logits[:, -1, :], temperature, generator)
    _synchronize(device)
    ttft_ms = (time.perf_counter() - prefill_start) * 1_000

    output_ids.append(int(next_token.item()))
    past_key_values = outputs.past_key_values
    eos_token_id = tokenizer.eos_token_id

    while len(output_ids) < max_new_tokens and output_ids[-1] != eos_token_id:
        attention_mask = torch.cat(
            (attention_mask, torch.ones((attention_mask.shape[0], 1), device=device, dtype=attention_mask.dtype)),
            dim=1,
        )
        _synchronize(device)
        decode_start = time.perf_counter()
        outputs = model(
            input_ids=next_token,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        next_token = _select_token(outputs.logits[:, -1, :], temperature, generator)
        _synchronize(device)
        decode_latencies_ms.append((time.perf_counter() - decode_start) * 1_000)
        output_ids.append(int(next_token.item()))
        past_key_values = outputs.past_key_values

    _synchronize(device)
    total_latency_ms = (time.perf_counter() - total_start) * 1_000
    output_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    return GenerationMetrics(
        prompt=prompt,
        output_text=output_text,
        output_token_ids=output_ids,
        ttft_ms=ttft_ms,
        decode_latencies_ms=decode_latencies_ms,
        total_latency_ms=total_latency_ms,
    )
