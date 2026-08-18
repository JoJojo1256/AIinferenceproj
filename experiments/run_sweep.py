from __future__ import annotations

import argparse
import gc
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from specdec.baseline import generate_baseline
from specdec.harness import collect_provenance, run_benchmark, write_results
from specdec.models import load_model, validate_shared_tokenizer
from specdec.speculative import generate_speculative
from specdec.workloads import WORKLOADS, get_workload

DEFAULT_DRAFT_MODELS = (
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
)
DEFAULT_SPECULATION_LENGTHS = (2, 3, 4, 5, 7)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the complete Phase 3 benchmark sweep.")
    parser.add_argument("--target-model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--draft-model", action="append", dest="draft_models")
    parser.add_argument("--target-revision")
    parser.add_argument("--draft-revision")
    parser.add_argument(
        "--workload",
        action="append",
        choices=sorted(WORKLOADS),
        dest="workloads",
    )
    parser.add_argument(
        "--speculation-length",
        action="append",
        type=int,
        dest="speculation_lengths",
    )
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--compile-draft", action="store_true")
    parser.add_argument(
        "--draft-cache-implementation",
        choices=("dynamic", "static"),
        default="dynamic",
    )
    parser.add_argument(
        "--target-cache-implementation",
        choices=("dynamic", "static"),
        default="dynamic",
    )
    parser.add_argument("--static-cache-max-length", type=int)
    parser.add_argument("--adaptive-speculation", action="store_true")
    parser.add_argument("--max-speculation-length", type=int)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--output")
    return parser.parse_args()


def _benchmark_record(
    results: dict[str, Any],
    experiment: dict[str, Any],
    comparison_id: str,
) -> dict[str, Any]:
    results["experiment"] = experiment
    results["comparison_id"] = comparison_id
    return results


def main() -> None:
    args = parse_args()
    draft_models = args.draft_models or list(DEFAULT_DRAFT_MODELS)
    workloads = args.workloads or list(WORKLOADS)
    speculation_lengths = args.speculation_lengths or list(DEFAULT_SPECULATION_LENGTHS)
    if any(length < 1 for length in speculation_lengths):
        raise ValueError("speculation lengths must be at least 1")
    if args.compile_draft and args.draft_cache_implementation != "static":
        raise ValueError("--compile-draft requires --draft-cache-implementation static")
    if args.compile_draft and args.static_cache_max_length is None:
        raise ValueError("--compile-draft requires --static-cache-max-length")
    if args.compile_draft and args.warmup_runs < 1:
        raise ValueError("--compile-draft requires at least one warmup run")
    if args.static_cache_max_length is not None and args.static_cache_max_length < 1:
        raise ValueError("--static-cache-max-length must be at least 1")
    if args.max_speculation_length is not None:
        if args.max_speculation_length < 1:
            raise ValueError("--max-speculation-length must be at least 1")
        if any(length > args.max_speculation_length for length in speculation_lengths):
            raise ValueError(
                "speculation lengths cannot exceed --max-speculation-length"
            )

    timestamp = datetime.now(UTC)
    sweep_id = timestamp.strftime("%Y%m%dT%H%M%SZ")
    output = args.output
    if output is None:
        output = Path("results/raw") / f"phase3_sweep_{sweep_id}.json"
    token = os.environ.get("HF_TOKEN")
    target = load_model(
        args.target_model,
        revision=args.target_revision,
        dtype=args.dtype,
        device=args.device,
        cache_dir=args.cache_dir,
        token=token,
    )
    results: dict[str, Any] = {
        "schema_version": 2,
        "sweep_id": sweep_id,
        "provenance": collect_provenance(),
        "experiment": {
            "type": "phase3_sweep",
            "target_model": args.target_model,
            "target_revision": args.target_revision,
            "draft_models": draft_models,
            "draft_revision": args.draft_revision,
            "workloads": workloads,
            "speculation_lengths": speculation_lengths,
            "dtype": args.dtype,
            "device": args.device,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "warmup_runs": args.warmup_runs,
            "trials": args.trials,
            "seed": args.seed,
            "compile_draft": args.compile_draft,
            "draft_cache_implementation": args.draft_cache_implementation,
            "target_cache_implementation": args.target_cache_implementation,
            "static_cache_max_length": args.static_cache_max_length,
            "adaptive_speculation": args.adaptive_speculation,
            "max_speculation_length": args.max_speculation_length,
        },
        "runs": [],
    }
    runs: list[dict[str, Any]] = results["runs"]

    for workload in workloads:
        prompts = get_workload(workload)

        def baseline_generate(prompt: str, seed: int):
            return generate_baseline(
                target,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                seed=seed,
            )

        baseline_results = run_benchmark(
            baseline_generate,
            prompts,
            warmup_runs=args.warmup_runs,
            trials=args.trials,
            seed=args.seed,
        )
        runs.append(
            _benchmark_record(
                baseline_results,
                {
                    "type": "baseline",
                    "model": args.target_model,
                    "model_revision": args.target_revision,
                    "workload": workload,
                    "dtype": args.dtype,
                    "device": args.device,
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                },
                comparison_id=f"{sweep_id}:{workload}",
            )
        )
        write_results(results, output)

    for draft_model in draft_models:
        draft = load_model(
            draft_model,
            revision=args.draft_revision,
            dtype=args.dtype,
            device=args.device,
            cache_dir=args.cache_dir,
            token=token,
        )
        validate_shared_tokenizer(target, draft)

        for workload in workloads:
            prompts = get_workload(workload)
            for speculation_length in speculation_lengths:

                def speculative_generate(prompt: str, seed: int):
                    return generate_speculative(
                        target,
                        draft,
                        prompt,
                        max_new_tokens=args.max_new_tokens,
                        speculation_length=speculation_length,
                        temperature=args.temperature,
                        seed=seed,
                        compile_draft=args.compile_draft,
                        draft_cache_implementation=args.draft_cache_implementation,
                        target_cache_implementation=args.target_cache_implementation,
                        adaptive_speculation=args.adaptive_speculation,
                        max_speculation_length=args.max_speculation_length,
                        static_cache_max_length=args.static_cache_max_length,
                    )

                speculative_results = run_benchmark(
                    speculative_generate,
                    prompts,
                    warmup_runs=args.warmup_runs,
                    trials=args.trials,
                    seed=args.seed,
                )
                runs.append(
                    _benchmark_record(
                        speculative_results,
                        {
                            "type": "speculative",
                            "target_model": args.target_model,
                            "target_revision": args.target_revision,
                            "draft_model": draft_model,
                            "draft_revision": args.draft_revision,
                            "workload": workload,
                            "dtype": args.dtype,
                            "device": args.device,
                            "max_new_tokens": args.max_new_tokens,
                            "speculation_length": speculation_length,
                            "temperature": args.temperature,
                            "compile_draft": args.compile_draft,
                            "draft_cache_implementation": (
                                args.draft_cache_implementation
                            ),
                            "target_cache_implementation": (
                                args.target_cache_implementation
                            ),
                            "static_cache_max_length": (
                                args.static_cache_max_length
                            ),
                            "adaptive_speculation": args.adaptive_speculation,
                            "max_speculation_length": (
                                args.max_speculation_length
                            ),
                        },
                        comparison_id=f"{sweep_id}:{workload}",
                    )
                )
                write_results(results, output)

        del draft
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    path = write_results(results, output)
    print(path)


if __name__ == "__main__":
    main()
