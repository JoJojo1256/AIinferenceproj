from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class GenerationMetrics:
    prompt: str
    output_text: str
    output_token_ids: list[int]
    ttft_ms: float
    decode_latencies_ms: list[float]
    total_latency_ms: float

    @property
    def output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def tokens_per_second(self) -> float:
        if self.total_latency_ms <= 0:
            return 0.0
        return self.output_tokens / (self.total_latency_ms / 1_000)

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["output_tokens"] = self.output_tokens
        data["tokens_per_second"] = self.tokens_per_second
        return data


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def summarize_trials(trials: Sequence[GenerationMetrics]) -> dict[str, float | int]:
    if not trials:
        raise ValueError("At least one measured trial is required")

    throughput = [trial.tokens_per_second for trial in trials]
    ttft = [trial.ttft_ms for trial in trials]
    total_latency = [trial.total_latency_ms for trial in trials]
    decode_latency = [
        latency
        for trial in trials
        for latency in trial.decode_latencies_ms
    ]
    return {
        "trial_count": len(trials),
        "tokens_per_second_median": percentile(throughput, 50),
        "tokens_per_second_p99": percentile(throughput, 99),
        "ttft_ms_median": percentile(ttft, 50),
        "ttft_ms_p99": percentile(ttft, 99),
        "decode_latency_ms_median": percentile(decode_latency, 50),
        "decode_latency_ms_p99": percentile(decode_latency, 99),
        "total_latency_ms_median": percentile(total_latency, 50),
        "total_latency_ms_p99": percentile(total_latency, 99),
    }
