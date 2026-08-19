from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


@dataclass(frozen=True)
class Measurement:
    draft_model: str
    workload: str
    speculation_length: int
    acceptance_rate: float
    speedup: float


@dataclass(frozen=True)
class SpeculationSeries:
    draft_model: str
    workload: str
    lengths: tuple[int, ...]
    speedups: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Phase 3 figures from sweep JSON.")
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Sweep JSON files (default: results/raw/phase3_sweep_*.json)",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/figures"))
    return parser.parse_args()


def _throughput(run: dict[str, Any]) -> float:
    try:
        value = float(run["summary"]["tokens_per_second_median"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Every run must contain median throughput") from exc
    if value <= 0:
        raise ValueError("Median throughput must be positive")
    return value


def measurements_from_document(document: dict[str, Any]) -> list[Measurement]:
    if document.get("experiment", {}).get("type") != "phase3_sweep":
        raise ValueError("Expected a phase3_sweep result document")
    runs = document.get("runs")
    if not isinstance(runs, list) or not runs:
        raise ValueError("Sweep result must contain at least one run")

    baselines: dict[str, float] = {}
    for run in runs:
        if run.get("experiment", {}).get("type") == "baseline":
            comparison_id = run.get("comparison_id")
            if not isinstance(comparison_id, str):
                raise ValueError("Baseline run is missing comparison_id")
            if comparison_id in baselines:
                raise ValueError(f"Duplicate baseline for {comparison_id}")
            baselines[comparison_id] = _throughput(run)

    measurements: list[Measurement] = []
    for run in runs:
        experiment = run.get("experiment", {})
        if experiment.get("type") != "speculative":
            continue
        comparison_id = run.get("comparison_id")
        if comparison_id not in baselines:
            raise ValueError(f"No matching baseline for {comparison_id!r}")
        summary = run.get("summary", {})
        try:
            acceptance_rate = float(summary["acceptance_rate"])
            speculative_throughput = _throughput(run)
            measurements.append(
                Measurement(
                    draft_model=str(experiment["draft_model"]),
                    workload=str(experiment["workload"]),
                    speculation_length=int(experiment["speculation_length"]),
                    acceptance_rate=acceptance_rate,
                    speedup=speculative_throughput / baselines[comparison_id],
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Malformed speculative run") from exc

    if not measurements:
        raise ValueError("Sweep contains no speculative runs")
    return measurements


def load_measurements(paths: Sequence[Path]) -> list[Measurement]:
    measurements: list[Measurement] = []
    for path in paths:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Could not read result file {path}") from exc
        measurements.extend(measurements_from_document(document))
    if not measurements:
        raise ValueError("No Phase 3 measurements were loaded")
    return measurements


def _short_model_name(model_id: str) -> str:
    return model_id.rsplit("/", maxsplit=1)[-1]


def _save(fig: plt.Figure, output_dir: Path, name: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_acceptance_vs_speedup(
    measurements: Sequence[Measurement],
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for draft_model in sorted({item.draft_model for item in measurements}):
        selected = [item for item in measurements if item.draft_model == draft_model]
        ax.scatter(
            [item.acceptance_rate for item in selected],
            [item.speedup for item in selected],
            label=_short_model_name(draft_model),
            alpha=0.8,
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Draft token acceptance rate", ylabel="Speedup vs. baseline")
    ax.set_title("Acceptance rate vs. speculative decoding speedup")
    ax.legend()
    ax.grid(alpha=0.25)
    return _save(fig, output_dir, "acceptance_vs_speedup.png")


def _speculation_length_series(
    measurements: Sequence[Measurement],
) -> list[SpeculationSeries]:
    grouped: dict[tuple[str, str, int], list[float]] = {}
    for item in measurements:
        key = (item.draft_model, item.workload, item.speculation_length)
        grouped.setdefault(key, []).append(item.speedup)

    pairs = sorted({(model, workload) for model, workload, _ in grouped})
    series = []
    for draft_model, workload in pairs:
        lengths = tuple(
            sorted(
                length
                for model, observed_workload, length in grouped
                if model == draft_model and observed_workload == workload
            )
        )
        speedups = tuple(
            float(np.median(grouped[(draft_model, workload, length)]))
            for length in lengths
        )
        series.append(
            SpeculationSeries(
                draft_model=draft_model,
                workload=workload,
                lengths=lengths,
                speedups=speedups,
            )
        )
    return series


def plot_speculation_length(
    measurements: Sequence[Measurement],
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(9, 6))
    best: tuple[float, int, str] | None = None
    for series in _speculation_length_series(measurements):
        model_name = _short_model_name(series.draft_model)
        label = f"{model_name} / {series.workload}"
        ax.plot(series.lengths, series.speedups, marker="o", label=label)
        best_index = int(np.argmax(series.speedups))
        candidate = (
            series.speedups[best_index],
            series.lengths[best_index],
            model_name,
        )
        if series.workload == "code" and (best is None or candidate[0] > best[0]):
            best = candidate
    if best is not None:
        ax.annotate(
            f"Best code: k={best[1]}, {best[0]:.3f}x\n({best[2]})",
            xy=(best[1], best[0]),
            xytext=(-130, -32),
            textcoords="offset points",
            arrowprops={"arrowstyle": "->", "color": "black"},
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set(xlabel="Speculation length (k)", ylabel="Median speedup vs. baseline")
    ax.set_title("Speculation length by workload")
    ax.legend(ncol=2, fontsize="small")
    ax.grid(alpha=0.25)
    return _save(fig, output_dir, "speculation_length_vs_speedup.png")


def plot_workload_speedup(
    measurements: Sequence[Measurement],
    output_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8, 5))
    draft_models = sorted({item.draft_model for item in measurements})
    workloads = sorted({item.workload for item in measurements})
    x = np.arange(len(workloads))
    width = 0.8 / len(draft_models)
    for index, draft_model in enumerate(draft_models):
        best_speedups = []
        for workload in workloads:
            observed = [
                item.speedup
                for item in measurements
                if item.draft_model == draft_model and item.workload == workload
            ]
            best_speedups.append(max(observed) if observed else np.nan)
        offset = (index - (len(draft_models) - 1) / 2) * width
        ax.bar(
            x + offset,
            best_speedups,
            width,
            label=_short_model_name(draft_model),
        )
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    ax.set_xticks(x, workloads)
    ax.set(ylabel="Best observed speedup vs. baseline")
    ax.set_title("Best speculative configuration by workload")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir, "speedup_by_workload.png")


def plot_negative_result(
    measurements: Sequence[Measurement],
    output_dir: Path,
) -> Path:
    slowest = min(measurements, key=lambda item: item.speedup)
    fig, ax = plt.subplots(figsize=(7, 4))
    label = (
        f"{_short_model_name(slowest.draft_model)}\n"
        f"{slowest.workload}, k={slowest.speculation_length}"
    )
    ax.bar([label], [slowest.speedup], color="#c44e52")
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1, label="Baseline")
    ax.set(ylabel="Speedup vs. baseline")
    if slowest.speedup < 1.0:
        slowdown = (1.0 - slowest.speedup) * 100
        ax.set_title(f"Negative result: {slowdown:.1f}% slower than baseline")
    else:
        ax.set_title("Lowest observed speedup (no slowdown measured)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    return _save(fig, output_dir, "negative_result.png")


def make_figures(
    measurements: Sequence[Measurement],
    output_dir: Path,
) -> list[Path]:
    return [
        plot_acceptance_vs_speedup(measurements, output_dir),
        plot_speculation_length(measurements, output_dir),
        plot_workload_speedup(measurements, output_dir),
        plot_negative_result(measurements, output_dir),
    ]


def main() -> None:
    args = parse_args()
    inputs = args.inputs or sorted(Path("results/raw").glob("phase3_sweep_*.json"))
    measurements = load_measurements(inputs)
    for path in make_figures(measurements, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
