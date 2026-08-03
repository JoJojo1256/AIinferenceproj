from __future__ import annotations

WORKLOADS: dict[str, list[str]] = {
    "qa": [
        "Explain why the sky appears blue during the day in three concise paragraphs.",
        "What is virtual memory, and why do operating systems use it?",
        "Describe the difference between throughput and latency in an inference system.",
    ],
    "code": [
        "Write a Python function that merges overlapping intervals and explain its complexity.",
        "Implement an LRU cache in Python without using functools.",
        "Write a unit-tested binary search function that returns the first matching index.",
    ],
    "reasoning": [
        "A server receives 120 requests per second and each request takes 25 ms of GPU time. Reason about the minimum parallelism needed to keep up.",
        "Compare two inference designs: one doubles throughput but adds 40 ms to TTFT, while the other reduces TTFT by 25 ms without changing throughput.",
        "Explain step by step how speculative decoding can become slower when draft acceptance is low.",
    ],
}


def get_workload(name: str) -> list[str]:
    try:
        return WORKLOADS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown workload {name!r}; choose one of {sorted(WORKLOADS)}") from exc
