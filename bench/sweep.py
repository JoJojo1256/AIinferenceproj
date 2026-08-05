from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the standard batching sweep")
    parser.add_argument("--static-url", required=True)
    parser.add_argument("--continuous-url", required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for mode, url in (
        ("static", args.static_url),
        ("continuous", args.continuous_url),
    ):
        for concurrency in (1, 2, 4, 8, 16, 32):
            for workload in ("uniform", "bimodal", "sampled"):
                output = (
                    args.output_dir
                    / f"{mode}_c{concurrency}_{workload}_seed{args.seed}.jsonl"
                )
                subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "bench.loadgen",
                        "--url",
                        url,
                        "--prompts",
                        str(args.prompts),
                        "--output",
                        str(output),
                        "--requests",
                        str(args.requests),
                        "--concurrency",
                        str(concurrency),
                        "--length-workload",
                        workload,
                        "--seed",
                        str(args.seed),
                    ],
                    check=True,
                )


if __name__ == "__main__":
    main()

