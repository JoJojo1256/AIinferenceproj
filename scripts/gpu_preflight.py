from __future__ import annotations

import argparse
import json

import torch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a CUDA host before downloading model weights.")
    parser.add_argument(
        "--minimum-vram-gb",
        type=float,
        default=20.0,
        help="Minimum physical VRAM required on the selected device.",
    )
    parser.add_argument("--device-index", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.minimum_vram_gb <= 0:
        raise SystemExit("--minimum-vram-gb must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; this host cannot run the GPU benchmark.")
    if not 0 <= args.device_index < torch.cuda.device_count():
        raise SystemExit(
            f"CUDA device {args.device_index} is unavailable; found {torch.cuda.device_count()} device(s)."
        )

    properties = torch.cuda.get_device_properties(args.device_index)
    total_vram_gb = properties.total_memory / 1024**3
    report = {
        "cuda_runtime_version": torch.version.cuda,
        "device_count": torch.cuda.device_count(),
        "device_index": args.device_index,
        "device_name": properties.name,
        "total_vram_gb": round(total_vram_gb, 2),
        "minimum_vram_gb": args.minimum_vram_gb,
        "torch_version": torch.__version__,
    }
    print(json.dumps(report, indent=2))

    if total_vram_gb < args.minimum_vram_gb:
        raise SystemExit(
            f"GPU has {total_vram_gb:.2f} GiB VRAM; at least "
            f"{args.minimum_vram_gb:.2f} GiB is required."
        )


if __name__ == "__main__":
    main()
