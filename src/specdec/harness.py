from __future__ import annotations

import json
import platform
import subprocess
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from specdec.metrics import (
    GenerationMetrics,
    SpeculativeGenerationMetrics,
    percentile,
    summarize_trials,
)

GenerateFunction = Callable[[str, int], GenerationMetrics]


def _git_value(*args: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *args],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def collect_provenance() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    return {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_dirty": bool(_git_value("status", "--porcelain")),
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime_version": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0) if cuda_available else None,
        "gpu_count": torch.cuda.device_count() if cuda_available else 0,
    }


def run_benchmark(
    generate: GenerateFunction,
    prompts: Sequence[str],
    *,
    warmup_runs: int,
    trials: int,
    seed: int,
) -> dict[str, Any]:
    if not prompts:
        raise ValueError("At least one prompt is required")
    if warmup_runs < 0 or trials < 1:
        raise ValueError("warmup_runs must be non-negative and trials must be positive")

    for warmup_index in range(warmup_runs):
        generate(prompts[warmup_index % len(prompts)], seed + warmup_index)

    measured: list[GenerationMetrics] = []
    for trial_index in range(trials):
        prompt = prompts[trial_index % len(prompts)]
        measured.append(generate(prompt, seed + warmup_runs + trial_index))

    summary = summarize_trials(measured)
    speculative_trials = [
        trial for trial in measured if isinstance(trial, SpeculativeGenerationMetrics)
    ]
    if speculative_trials:
        proposed_tokens = sum(trial.proposed_tokens for trial in speculative_trials)
        accepted_tokens = sum(trial.accepted_tokens for trial in speculative_trials)
        realized_speculation_lengths = [
            length
            for trial in speculative_trials
            for length in (trial.realized_speculation_lengths or [])
        ]
        first_speculative = speculative_trials[0]
        summary.update(
            {
                "proposed_tokens": proposed_tokens,
                "accepted_tokens": accepted_tokens,
                "acceptance_rate": accepted_tokens / proposed_tokens if proposed_tokens else 0.0,
                "target_forward_passes": sum(
                    trial.target_forward_passes for trial in speculative_trials
                ),
                "target_processed_tokens": sum(
                    trial.target_processed_tokens for trial in speculative_trials
                ),
                "draft_processed_tokens": sum(
                    trial.draft_processed_tokens for trial in speculative_trials
                ),
                "prefill_time_ms": sum(
                    trial.prefill_time_ms for trial in speculative_trials
                ),
                "draft_proposal_time_ms": sum(
                    trial.draft_proposal_time_ms for trial in speculative_trials
                ),
                "target_verification_time_ms": sum(
                    trial.target_verification_time_ms for trial in speculative_trials
                ),
                "sampling_overhead_time_ms": sum(
                    trial.sampling_overhead_time_ms for trial in speculative_trials
                ),
                "draft_compiled": first_speculative.draft_compiled,
                "draft_cache_implementation": (
                    first_speculative.draft_cache_implementation
                ),
                "target_cache_implementation": (
                    first_speculative.target_cache_implementation
                ),
                "adaptive_speculation": first_speculative.adaptive_speculation,
                "initial_speculation_length": (
                    first_speculative.initial_speculation_length
                ),
                "realized_speculation_length_mean": (
                    sum(realized_speculation_lengths) / len(realized_speculation_lengths)
                    if realized_speculation_lengths
                    else 0.0
                ),
                "realized_speculation_length_median": percentile(
                    realized_speculation_lengths,
                    50,
                ),
            }
        )

    return {
        "provenance": collect_provenance(),
        "configuration": {
            "warmup_runs": warmup_runs,
            "trials": trials,
            "seed": seed,
            "prompt_count": len(prompts),
        },
        "summary": summary,
        "trials": [trial.to_dict() for trial in measured],
    }


def write_results(results: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(f"{path.suffix}.tmp")
    temporary_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    temporary_path.replace(path)
    return path
