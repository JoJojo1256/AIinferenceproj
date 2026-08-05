from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import httpx


@dataclass(frozen=True)
class ClientMetric:
    type: str
    request_id: str
    trial_index: int
    prompt_index: int
    max_new_tokens: int
    start_time: float
    first_token_time: float
    finish_time: float
    ttft_ms: float
    tpot_ms: float
    end_to_end_ms: float
    output_token_count: int
    status_code: int
    error: str | None = None


def load_prompts(path: str | Path) -> list[str]:
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not prompts:
        raise ValueError("Prompt file must contain at least one non-empty line")
    return prompts


def output_lengths(
    count: int,
    workload: str,
    *,
    seed: int,
    minimum: int,
    maximum: int,
) -> list[int]:
    if count < 1 or minimum < 1 or maximum < minimum:
        raise ValueError("Invalid output length parameters")
    rng = random.Random(seed)
    if workload == "uniform":
        return [rng.randint(minimum, maximum) for _ in range(count)]
    if workload == "bimodal":
        low_max = minimum + max(0, (maximum - minimum) // 4)
        high_min = maximum - max(0, (maximum - minimum) // 4)
        return [
            rng.randint(minimum, low_max)
            if rng.random() < 0.5
            else rng.randint(high_min, maximum)
            for _ in range(count)
        ]
    if workload == "sampled":
        choices = [minimum, max(minimum, (minimum + maximum) // 2), maximum]
        weights = [0.55, 0.30, 0.15]
        return rng.choices(choices, weights=weights, k=count)
    raise ValueError("workload must be uniform, bimodal, or sampled")


async def _request(
    client: httpx.AsyncClient,
    url: str,
    prompt: str,
    prompt_index: int,
    max_new_tokens: int,
    seed: int,
    trial_index: int,
) -> ClientMetric:
    request_id = uuid4().hex
    start = time.time()
    first_token: float | None = None
    output_tokens = 0
    token_times: list[float] = []
    terminal_event_seen = False
    status_code = 0
    error: str | None = None
    try:
        async with client.stream(
            "POST",
            f"{url.rstrip('/')}/generate",
            json={
                "request_id": request_id,
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
                "seed": seed,
            },
        ) as response:
            status_code = response.status_code
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data: "):
                    continue
                event = json.loads(line[6:])
                if terminal_event_seen:
                    raise RuntimeError("Received data after the terminal token event")
                if event.get("error"):
                    raise RuntimeError(str(event["error"]))
                if "token_id" not in event or "finished" not in event:
                    raise RuntimeError("Malformed token event")
                now = time.time()
                if first_token is None:
                    first_token = now
                token_times.append(now)
                output_tokens += 1
                if event["finished"]:
                    terminal_event_seen = True
            if not terminal_event_seen:
                raise RuntimeError("Token stream ended without a terminal event")
    except (httpx.HTTPError, json.JSONDecodeError, RuntimeError) as exc:
        error = str(exc)
    finish = time.time()
    first = first_token or finish
    return ClientMetric(
        type="client_request",
        request_id=request_id,
        trial_index=trial_index,
        prompt_index=prompt_index,
        max_new_tokens=max_new_tokens,
        start_time=start,
        first_token_time=first,
        finish_time=finish,
        ttft_ms=(first - start) * 1_000,
        tpot_ms=(
            (token_times[-1] - token_times[0]) * 1_000 / (len(token_times) - 1)
            if len(token_times) > 1
            else 0.0
        ),
        end_to_end_ms=(finish - start) * 1_000,
        output_token_count=output_tokens,
        status_code=status_code,
        error=error,
    )


async def run_closed_loop(
    url: str,
    prompts: list[str],
    lengths: list[int],
    *,
    concurrency: int,
    seed: int,
    timeout: float,
    trial_index: int = 0,
) -> list[ClientMetric]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    work: asyncio.Queue[int] = asyncio.Queue()
    for index in range(len(lengths)):
        work.put_nowait(index)
    results: list[ClientMetric] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        async def worker() -> None:
            while not work.empty():
                try:
                    index = work.get_nowait()
                except asyncio.QueueEmpty:
                    return
                results.append(
                    await _request(
                        client,
                        url,
                        prompts[index % len(prompts)],
                        index % len(prompts),
                        lengths[index],
                        seed + index,
                        trial_index,
                    )
                )

        await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


async def run_open_loop(
    url: str,
    prompts: list[str],
    lengths: list[int],
    *,
    rate: float,
    seed: int,
    timeout: float,
    trial_index: int = 0,
) -> list[ClientMetric]:
    if rate <= 0:
        raise ValueError("rate must be positive")
    rng = random.Random(seed)
    start = time.monotonic()
    scheduled = 0.0
    tasks: list[asyncio.Task[ClientMetric]] = []
    async with httpx.AsyncClient(timeout=timeout) as client:
        for index, length in enumerate(lengths):
            if index:
                scheduled += -math.log1p(-rng.random()) / rate
                await asyncio.sleep(max(0.0, start + scheduled - time.monotonic()))
            tasks.append(
                asyncio.create_task(
                    _request(
                        client,
                        url,
                        prompts[index % len(prompts)],
                        index % len(prompts),
                        length,
                        seed + index,
                        trial_index,
                    )
                )
            )
        return await asyncio.gather(*tasks)


def write_jsonl(
    path: str | Path,
    records: list[ClientMetric],
    configuration: dict[str, Any],
    server_metrics: dict[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps({"type": "loadgen_config", **configuration}) + "\n")
        stream.write(json.dumps({"type": "server_metrics", **server_metrics}) + "\n")
        for record in records:
            stream.write(json.dumps(asdict(record), sort_keys=True) + "\n")


async def run_trials(
    *,
    url: str,
    prompts: list[str],
    request_count: int,
    arrival: str,
    concurrency: int,
    rate: float,
    workload: str,
    minimum: int,
    maximum: int,
    seed: int,
    timeout: float,
    warmup_requests: int,
    trials: int,
) -> list[ClientMetric]:
    if warmup_requests < 0:
        raise ValueError("warmup_requests must be non-negative")
    if trials < 3:
        raise ValueError("At least three measured trials are required")
    if warmup_requests:
        warmup_lengths = output_lengths(
            warmup_requests,
            workload,
            seed=seed - 1,
            minimum=minimum,
            maximum=maximum,
        )
        await run_closed_loop(
            url,
            prompts,
            warmup_lengths,
            concurrency=max(1, min(concurrency, warmup_requests)),
            seed=seed - 1,
            timeout=timeout,
            trial_index=-1,
        )
    records: list[ClientMetric] = []
    for trial in range(trials):
        trial_seed = seed + trial * request_count
        lengths = output_lengths(
            request_count,
            workload,
            seed=trial_seed,
            minimum=minimum,
            maximum=maximum,
        )
        if arrival == "closed":
            trial_records = await run_closed_loop(
                url,
                prompts,
                lengths,
                concurrency=concurrency,
                seed=trial_seed,
                timeout=timeout,
                trial_index=trial,
            )
        else:
            trial_records = await run_open_loop(
                url,
                prompts,
                lengths,
                rate=rate,
                seed=trial_seed,
                timeout=timeout,
                trial_index=trial,
            )
        records.extend(trial_records)
    return records


async def fetch_server_metrics(url: str, timeout: float) -> dict[str, Any]:
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(f"{url.rstrip('/')}/metrics")
        response.raise_for_status()
        return response.json()


def main() -> None:
    parser = argparse.ArgumentParser(description="Load generator for the inference server")
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--arrival", choices=("closed", "poisson"), default="closed")
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--rate", type=float, default=4.0)
    parser.add_argument(
        "--length-workload",
        choices=("uniform", "bimodal", "sampled"),
        default="uniform",
    )
    parser.add_argument("--min-output-tokens", type=int, default=16)
    parser.add_argument("--max-output-tokens", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--warmup-requests", type=int, default=4)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()
    prompts = load_prompts(args.prompts)
    records = asyncio.run(
        run_trials(
            url=args.url,
            prompts=prompts,
            request_count=args.requests,
            arrival=args.arrival,
            concurrency=args.concurrency,
            rate=args.rate,
            workload=args.length_workload,
            minimum=args.min_output_tokens,
            maximum=args.max_output_tokens,
            seed=args.seed,
            timeout=args.timeout,
            warmup_requests=args.warmup_requests,
            trials=args.trials,
        )
    )
    configuration = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    server_metrics = asyncio.run(fetch_server_metrics(args.url, args.timeout))
    write_jsonl(args.output, records, configuration, server_metrics)


if __name__ == "__main__":
    main()
