from __future__ import annotations

import json
import platform
import subprocess
import time
from collections import Counter
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from threading import Lock
from typing import Any, Sequence

import numpy as np
import torch
import transformers

from continuous_batching.config import InferenceConfig
from continuous_batching.models import SequenceState


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def collect_provenance(
    config: InferenceConfig,
    model_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    packages: dict[str, str] = {}
    for name in ("accelerate", "fastapi", "httpx", "numpy", "torch", "transformers", "uvicorn"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "not-installed"
    cuda_available = torch.cuda.is_available()
    return {
        "timestamp": time.time(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
        "transformers_version": transformers.__version__,
        "config": asdict(config),
        "model": model_metadata,
        "packages": packages,
    }


@dataclass(frozen=True)
class RequestMetric:
    type: str
    request_id: str
    arrival_time: float
    first_token_time: float
    finish_time: float
    ttft_ms: float
    tpot_ms: float
    end_to_end_ms: float
    output_token_count: int

    @classmethod
    def from_state(cls, state: SequenceState) -> RequestMetric:
        if state.first_token_time is None or state.finish_time is None:
            raise ValueError("Cannot record metrics for an unfinished sequence")
        output_count = state.output_tokens
        decode_seconds = max(0.0, state.finish_time - state.first_token_time)
        return cls(
            type="request",
            request_id=state.request_id,
            arrival_time=state.request.arrival_time,
            first_token_time=state.first_token_time,
            finish_time=state.finish_time,
            ttft_ms=(state.first_token_time - state.request.arrival_time) * 1_000,
            tpot_ms=(decode_seconds * 1_000 / (output_count - 1)) if output_count > 1 else 0.0,
            end_to_end_ms=(state.finish_time - state.request.arrival_time) * 1_000,
            output_token_count=output_count,
        )


class JSONLWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()

    def write(self, record: dict[str, Any]) -> None:
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")


class MetricsCollector:
    def __init__(
        self,
        config: InferenceConfig,
        *,
        jsonl_path: str | Path | None = None,
        slo_ttft_ms: float = 500.0,
        slo_end_to_end_ms: float = 5_000.0,
        model_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.slo_ttft_ms = slo_ttft_ms
        self.slo_end_to_end_ms = slo_end_to_end_ms
        self.requests: list[RequestMetric] = []
        self.active_batch_trace: list[tuple[float, int]] = []
        self.writer = JSONLWriter(jsonl_path) if jsonl_path else None
        self.provenance = collect_provenance(config, model_metadata)
        if self.writer:
            self.writer.write({"type": "provenance", **self.provenance})

    def record_batch_size(self, active: int, timestamp: float | None = None) -> None:
        now = timestamp if timestamp is not None else time.time()
        if self.active_batch_trace and self.active_batch_trace[-1][1] == active:
            return
        self.active_batch_trace.append((now, active))
        if self.writer:
            self.writer.write({"type": "batch_size", "timestamp": now, "active": active})

    def record_request(self, state: SequenceState) -> RequestMetric:
        metric = RequestMetric.from_state(state)
        self.requests.append(metric)
        if self.writer:
            self.writer.write(asdict(metric))
        return metric

    def snapshot(self) -> dict[str, Any]:
        ttft = [metric.ttft_ms for metric in self.requests]
        tpot = [metric.tpot_ms for metric in self.requests]
        latency = [metric.end_to_end_ms for metric in self.requests]
        if self.requests:
            duration = max(metric.finish_time for metric in self.requests) - min(
                metric.arrival_time for metric in self.requests
            )
        else:
            duration = 0.0
        total_tokens = sum(metric.output_token_count for metric in self.requests)
        good = sum(
            metric.ttft_ms <= self.slo_ttft_ms
            and metric.end_to_end_ms <= self.slo_end_to_end_ms
            for metric in self.requests
        )
        histogram = Counter(size for _, size in self.active_batch_trace)
        weighted_size = 0.0
        weighted_duration = 0.0
        intervals = list(zip(self.active_batch_trace, self.active_batch_trace[1:]))
        if self.active_batch_trace:
            intervals.append((self.active_batch_trace[-1], (time.time(), 0)))
        for (start, size), (end, _) in intervals:
            interval = max(0.0, end - start)
            weighted_size += size * interval
            weighted_duration += interval
        return {
            "provenance": self.provenance,
            "request_count": len(self.requests),
            "output_token_count": total_tokens,
            "throughput_tokens_per_second": total_tokens / duration if duration > 0 else 0.0,
            "goodput_requests_per_second": good / duration if duration > 0 else 0.0,
            "slo": {
                "ttft_ms": self.slo_ttft_ms,
                "end_to_end_ms": self.slo_end_to_end_ms,
                "meeting_requests": good,
            },
            "ttft_ms": {"p50": percentile(ttft, 50), "p99": percentile(ttft, 99)},
            "tpot_ms": {"p50": percentile(tpot, 50), "p99": percentile(tpot, 99)},
            "end_to_end_ms": {
                "p50": percentile(latency, 50),
                "p99": percentile(latency, 99),
            },
            "active_batch_size": {
                "time_average": (
                    weighted_size / weighted_duration if weighted_duration > 0 else 0.0
                ),
                "histogram": {str(key): value for key, value in sorted(histogram.items())},
                "trace": [
                    {"timestamp": timestamp, "active": active}
                    for timestamp, active in self.active_batch_trace
                ],
            },
        }
