from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def read_jsonl(paths: list[Path]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        with path.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid JSON in {path}:{line_number}") from exc
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, float | int]:
    requests = [
        record
        for record in records
        if record.get("type") in {"request", "client_request"}
        and not record.get("error")
    ]
    if not requests:
        raise ValueError("No successful request records found")
    ttft = np.asarray([record["ttft_ms"] for record in requests], dtype=np.float64)
    latency = np.asarray(
        [
            record.get("end_to_end_ms", record.get("total_latency_ms"))
            for record in requests
        ],
        dtype=np.float64,
    )
    tpot = np.asarray(
        [record.get("tpot_ms", 0.0) for record in requests], dtype=np.float64
    )
    tokens = int(sum(record["output_token_count"] for record in requests))
    starts = [
        record.get("arrival_time", record.get("start_time")) for record in requests
    ]
    finishes = [record["finish_time"] for record in requests]
    duration = max(finishes) - min(starts)
    return {
        "request_count": len(requests),
        "output_token_count": tokens,
        "throughput_tokens_per_second": tokens / duration if duration > 0 else 0.0,
        "ttft_ms_p50": float(np.percentile(ttft, 50)),
        "ttft_ms_p99": float(np.percentile(ttft, 99)),
        "tpot_ms_p50": float(np.percentile(tpot, 50)),
        "tpot_ms_p99": float(np.percentile(tpot, 99)),
        "end_to_end_ms_p50": float(np.percentile(latency, 50)),
        "end_to_end_ms_p99": float(np.percentile(latency, 99)),
    }


def write_outputs(
    summary: dict[str, float | int],
    records: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        writer.writeheader()
        writer.writerow(summary)
    requests = [
        record
        for record in records
        if record.get("type") in {"request", "client_request"}
        and not record.get("error")
    ]
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist([record["ttft_ms"] for record in requests], bins=20)
    axes[0].set(title="Time to first token", xlabel="TTFT (ms)", ylabel="Requests")
    axes[1].hist(
        [
            record.get("end_to_end_ms", record.get("total_latency_ms"))
            for record in requests
        ],
        bins=20,
    )
    axes[1].set(title="End-to-end latency", xlabel="Latency (ms)", ylabel="Requests")
    figure.tight_layout()
    figure.savefig(output_dir / "latency_histograms.png", dpi=160)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze benchmark JSONL")
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    records = read_jsonl(args.inputs)
    summary = summarize(records)
    write_outputs(summary, records, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
