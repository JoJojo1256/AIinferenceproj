from __future__ import annotations

import argparse
import os
from datetime import UTC, datetime
from pathlib import Path

from specdec.baseline import generate_baseline
from specdec.harness import run_benchmark, write_results
from specdec.models import load_model
from specdec.workloads import WORKLOADS, get_workload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark standard autoregressive decoding.")
    parser.add_argument("--model", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--workload", choices=sorted(WORKLOADS), default="qa")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--warmup-runs", type=int, default=3)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--output")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bundle = load_model(
        args.model,
        revision=args.revision,
        dtype=args.dtype,
        device=args.device,
        cache_dir=args.cache_dir,
        token=os.environ.get("HF_TOKEN"),
    )

    def generate(prompt: str, seed: int):
        return generate_baseline(
            bundle,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            seed=seed,
        )

    results = run_benchmark(
        generate,
        get_workload(args.workload),
        warmup_runs=args.warmup_runs,
        trials=args.trials,
        seed=args.seed,
    )
    results["experiment"] = {
        "type": "baseline",
        "model": args.model,
        "model_revision": args.revision,
        "workload": args.workload,
        "dtype": args.dtype,
        "device": args.device,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }

    output = args.output
    if output is None:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        output = Path("results/raw") / f"baseline_{args.workload}_{timestamp}.json"
    path = write_results(results, output)
    print(path)


if __name__ == "__main__":
    main()
