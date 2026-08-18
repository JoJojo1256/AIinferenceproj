from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

import specdec.speculative as speculative_module
from specdec.baseline import generate_baseline
from specdec.harness import collect_provenance
from specdec.models import encode_prompt, load_model, validate_shared_tokenizer
from specdec.speculative import generate_speculative
from specdec.workloads import WORKLOADS

TARGET_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
DRAFT_MODEL = "meta-llama/Llama-3.2-1B-Instruct"
VERIFICATION_PROMPTS = {
    "code": [
        *WORKLOADS["code"],
        "Write a Python function that detects a cycle in a linked list and explain its complexity.",
    ],
    "qa": [
        *WORKLOADS["qa"],
        "Explain the difference between concurrency and parallelism with one concrete example.",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify speculative decoding equality under real CUDA graph replay."
    )
    parser.add_argument("--target-model", default=TARGET_MODEL)
    parser.add_argument("--draft-model", default=DRAFT_MODEL)
    parser.add_argument("--target-revision")
    parser.add_argument("--draft-revision")
    parser.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=os.environ.get("HF_HOME"))
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--static-cache-max-length", type=int, default=512)
    parser.add_argument("--compile-warmups", type=int, default=3)
    parser.add_argument("--replays-per-prompt", type=int, default=3)
    parser.add_argument("--stochastic-seeds", type=int, default=64)
    parser.add_argument("--stochastic-temperature", type=float, default=0.8)
    parser.add_argument("--max-first-token-tv", type=float, default=0.10)
    parser.add_argument("--max-first-token-mismatch-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--no-clone-logits",
        action="store_true",
        help=(
            "Negative control: disable compiled-logit cloning in this process. "
            "The alias probe must detect corruption and the script exits nonzero."
        ),
    )
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if not args.device.startswith("cuda"):
        raise ValueError("This verification requires a CUDA device")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.max_new_tokens < 1:
        raise ValueError("--max-new-tokens must be positive")
    if args.compile_warmups < 2:
        raise ValueError("--compile-warmups must be at least 2 to exercise replay")
    if args.replays_per_prompt < 2:
        raise ValueError("--replays-per-prompt must be at least 2")
    if args.stochastic_seeds < 16:
        raise ValueError("--stochastic-seeds must be at least 16")
    if args.stochastic_temperature <= 0:
        raise ValueError("--stochastic-temperature must be positive")
    for name in ("max_first_token_tv", "max_first_token_mismatch_rate"):
        value = getattr(args, name)
        if not 0 <= value <= 1:
            raise ValueError(f"--{name.replace('_', '-')} must be between 0 and 1")


def _disable_compiled_logit_clone() -> None:
    original_advance = speculative_module._advance_model

    def advance_without_clone(*args: Any, **kwargs: Any):
        if kwargs.get("forward") is not None:
            kwargs["clone_logits"] = False
        return original_advance(*args, **kwargs)

    speculative_module._advance_model = advance_without_clone


def _first_divergence(left: list[int], right: list[int]) -> int | None:
    for index, (left_token, right_token) in enumerate(zip(left, right, strict=False)):
        if left_token != right_token:
            return index
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def _mismatch_details(
    expected: list[int],
    actual: list[int],
    tokenizer: Any,
) -> dict[str, Any]:
    index = _first_divergence(expected, actual)
    if index is None:
        return {}
    start = max(0, index - 5)
    end = index + 6
    expected_window = expected[start:end]
    actual_window = actual[start:end]
    return {
        "first_divergent_index": index,
        "window_start": start,
        "expected_token_ids": expected_window,
        "actual_token_ids": actual_window,
        "expected_text": tokenizer.decode(expected_window, skip_special_tokens=False),
        "actual_text": tokenizer.decode(actual_window, skip_special_tokens=False),
    }


def _speculative_options(
    args: argparse.Namespace,
    *,
    compiled: bool,
    adaptive: bool,
) -> dict[str, Any]:
    options: dict[str, Any] = {
        "max_new_tokens": args.max_new_tokens,
        "speculation_length": 5,
        "temperature": 0.0,
        "compile_draft": compiled,
        "draft_cache_implementation": "static" if compiled else "dynamic",
        "target_cache_implementation": "dynamic",
        "adaptive_speculation": adaptive,
    }
    if compiled:
        options["static_cache_max_length"] = args.static_cache_max_length
    if adaptive:
        options["max_speculation_length"] = 15
    return options


def _warm_compiled_paths(target: Any, draft: Any, args: argparse.Namespace) -> None:
    prompt = VERIFICATION_PROMPTS["code"][0]
    for adaptive in (False, True):
        label = "compiled-static-adaptive" if adaptive else "compiled-static-fixed"
        for warmup_index in range(args.compile_warmups):
            generate_speculative(
                target,
                draft,
                prompt,
                seed=args.seed + warmup_index,
                **_speculative_options(args, compiled=True, adaptive=adaptive),
            )
            print(f"WARMUP {label} {warmup_index + 1}/{args.compile_warmups}")


@torch.inference_mode()
def _run_alias_probe(draft: Any, args: argparse.Namespace) -> dict[str, Any]:
    device = next(draft.model.parameters()).device
    encoded = encode_prompt(draft.tokenizer, VERIFICATION_PROMPTS["code"][0], device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    state = speculative_module._prefill_model(
        draft,
        input_ids,
        attention_mask,
        cache_implementation="static",
        max_cache_length=args.static_cache_max_length,
    )
    compiled_forward = speculative_module.compile_draft_forward(draft)

    first_attention_mask = torch.cat(
        (
            attention_mask,
            torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
        ),
        dim=1,
    )
    first_state, retained_logits = speculative_module._advance_model(
        draft,
        state,
        torch.tensor([[1]], device=device, dtype=input_ids.dtype),
        first_attention_mask,
        forward=compiled_forward,
        clone_logits=not args.no_clone_logits,
    )
    retained_snapshot = retained_logits.clone()

    second_attention_mask = torch.cat(
        (
            first_attention_mask,
            torch.ones((1, 1), device=device, dtype=attention_mask.dtype),
        ),
        dim=1,
    )
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        _, second_logits = speculative_module._advance_model(
            draft,
            first_state,
            torch.tensor([[2]], device=device, dtype=input_ids.dtype),
            second_attention_mask,
            forward=compiled_forward,
            clone_logits=not args.no_clone_logits,
        )
        torch.cuda.synchronize(device)

    event_names = {
        event.key for event in profiler.key_averages()
    } | {
        event.name for event in profiler.events()
    }
    graph_events = [
        name
        for name in sorted(event_names)
        if any(marker in name.lower() for marker in ("cudagraph", "cuda graph", "graphlaunch"))
    ]
    retained_unchanged = torch.equal(retained_logits, retained_snapshot)

    raw_state = speculative_module._prefill_model(
        draft,
        input_ids,
        attention_mask,
        cache_implementation="static",
        max_cache_length=args.static_cache_max_length,
    )
    raw_first_state, raw_first_logits = speculative_module._advance_model(
        draft,
        raw_state,
        torch.tensor([[3]], device=device, dtype=input_ids.dtype),
        first_attention_mask,
        forward=compiled_forward,
        clone_logits=False,
    )
    raw_first_snapshot = raw_first_logits.clone()
    _, raw_second_logits = speculative_module._advance_model(
        draft,
        raw_first_state,
        torch.tensor([[4]], device=device, dtype=input_ids.dtype),
        second_attention_mask,
        forward=compiled_forward,
        clone_logits=False,
    )
    torch.cuda.synchronize(device)
    raw_output_buffer_reused = (
        raw_first_logits.data_ptr() == raw_second_logits.data_ptr()
    )
    raw_retained_logits_changed = not torch.equal(
        raw_first_logits,
        raw_first_snapshot,
    )
    replay_observed = bool(graph_events) or raw_output_buffer_reused
    result = {
        "cuda_graph_events": graph_events,
        "cuda_graph_replay_observed": replay_observed,
        "raw_output_buffer_reused": raw_output_buffer_reused,
        "raw_retained_logits_changed": raw_retained_logits_changed,
        "retained_logits_unchanged": retained_unchanged,
        "first_second_data_ptr_equal": (
            retained_logits.data_ptr() == second_logits.data_ptr()
        ),
        "negative_control": args.no_clone_logits,
    }
    print(f"CUDA GRAPH EVENTS: {graph_events or 'NONE'}")
    print(
        "ALIAS PROBE: "
        f"retained_unchanged={retained_unchanged} "
        f"same_data_ptr={result['first_second_data_ptr_equal']} "
        f"raw_buffer_reused={raw_output_buffer_reused} "
        f"raw_retained_changed={raw_retained_logits_changed} "
        f"negative_control={args.no_clone_logits}"
    )
    return result


def _run_greedy_equality(
    target: Any,
    draft: Any,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[str]]:
    records: list[dict[str, Any]] = []
    failures: list[str] = []
    variants = (
        ("eager-dynamic-fixed", False, False, 1),
        ("compiled-static-fixed", True, False, args.replays_per_prompt),
        ("compiled-static-adaptive", True, True, args.replays_per_prompt),
    )

    for workload, prompts in VERIFICATION_PROMPTS.items():
        if len(prompts) < 4:
            raise RuntimeError(f"{workload} must provide at least four verification prompts")
        for prompt_index, prompt in enumerate(prompts):
            baseline = generate_baseline(
                target,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                seed=args.seed,
            )
            for label, compiled, adaptive, repeats in variants:
                for replay_index in range(repeats):
                    result = generate_speculative(
                        target,
                        draft,
                        prompt,
                        seed=args.seed,
                        **_speculative_options(
                            args,
                            compiled=compiled,
                            adaptive=adaptive,
                        ),
                    )
                    details = _mismatch_details(
                        baseline.output_token_ids,
                        result.output_token_ids,
                        target.tokenizer,
                    )
                    passed = not details
                    record = {
                        "workload": workload,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "variant": label,
                        "replay_index": replay_index,
                        "passed": passed,
                        "baseline_tokens": len(baseline.output_token_ids),
                        "speculative_tokens": len(result.output_token_ids),
                        **details,
                    }
                    records.append(record)
                    status = "PASS" if passed else "FAIL"
                    print(
                        f"{status} {workload}[{prompt_index}] {label} "
                        f"replay={replay_index + 1}/{repeats}"
                    )
                    if not passed:
                        failures.append(
                            f"{workload}[{prompt_index}] {label} replay {replay_index + 1}"
                        )
                        print(json.dumps(details, indent=2))
    return records, failures


def _total_variation(left: Counter[int], right: Counter[int], samples: int) -> float:
    token_ids = left.keys() | right.keys()
    return 0.5 * sum(
        abs(left[token_id] / samples - right[token_id] / samples)
        for token_id in token_ids
    )


def _run_stochastic_check(
    target: Any,
    draft: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    prompt = VERIFICATION_PROMPTS["qa"][0]
    eager_counts: Counter[int] = Counter()
    compiled_counts: Counter[int] = Counter()
    mismatches = 0

    for offset in range(args.stochastic_seeds):
        seed = args.seed + 10_000 + offset
        common = {
            "max_new_tokens": 1,
            "speculation_length": 1,
            "temperature": args.stochastic_temperature,
            "seed": seed,
        }
        eager = generate_speculative(
            target,
            draft,
            prompt,
            draft_cache_implementation="dynamic",
            **common,
        )
        compiled = generate_speculative(
            target,
            draft,
            prompt,
            compile_draft=True,
            draft_cache_implementation="static",
            static_cache_max_length=args.static_cache_max_length,
            **common,
        )
        eager_token = eager.output_token_ids[0]
        compiled_token = compiled.output_token_ids[0]
        eager_counts[eager_token] += 1
        compiled_counts[compiled_token] += 1
        mismatches += eager_token != compiled_token

    total_variation = _total_variation(
        eager_counts,
        compiled_counts,
        args.stochastic_seeds,
    )
    mismatch_rate = mismatches / args.stochastic_seeds
    passed = (
        total_variation <= args.max_first_token_tv
        and mismatch_rate <= args.max_first_token_mismatch_rate
    )
    result = {
        "passed": passed,
        "samples": args.stochastic_seeds,
        "temperature": args.stochastic_temperature,
        "paired_seed_mismatches": mismatches,
        "paired_seed_mismatch_rate": mismatch_rate,
        "total_variation": total_variation,
        "max_total_variation": args.max_first_token_tv,
        "max_mismatch_rate": args.max_first_token_mismatch_rate,
        "eager_counts": dict(sorted(eager_counts.items())),
        "compiled_counts": dict(sorted(compiled_counts.items())),
        "note": (
            "Exact seed-for-seed sequence identity is not a distributional invariant "
            "because DynamicCache and StaticCache can have small numerical differences. "
            "This compares paired first tokens and empirical first-token distributions."
        ),
    }
    print(
        f"{'PASS' if passed else 'FAIL'} stochastic first-token check: "
        f"TV={total_variation:.4f}, paired mismatch rate={mismatch_rate:.4f}, "
        f"samples={args.stochastic_seeds}"
    )
    print(result["note"])
    return result


def main() -> None:
    args = parse_args()
    _validate_args(args)
    if args.no_clone_logits:
        _disable_compiled_logit_clone()
        print("NEGATIVE CONTROL ENABLED: compiled logits will not be cloned")

    token = os.environ.get("HF_TOKEN")
    target = load_model(
        args.target_model,
        revision=args.target_revision,
        dtype=args.dtype,
        device=args.device,
        cache_dir=args.cache_dir,
        token=token,
    )
    draft = load_model(
        args.draft_model,
        revision=args.draft_revision,
        dtype=args.dtype,
        device=args.device,
        cache_dir=args.cache_dir,
        token=token,
    )
    validate_shared_tokenizer(target, draft)

    failures: list[str] = []
    _warm_compiled_paths(target, draft, args)
    alias_probe = _run_alias_probe(draft, args)
    if not alias_probe["cuda_graph_replay_observed"]:
        failures.append(
            "No CUDA graph event or compiled output-buffer reuse was observed"
        )
    if not alias_probe["raw_retained_logits_changed"]:
        failures.append(
            "The internal uncloned negative probe did not expose buffer overwrite"
        )
    if args.no_clone_logits:
        if alias_probe["retained_logits_unchanged"]:
            failures.append("Negative control did not expose output-buffer reuse")
        else:
            failures.append("Negative control exposed output-buffer alias corruption")
    elif not alias_probe["retained_logits_unchanged"]:
        failures.append("Cloned compiled logits were overwritten")

    greedy_records, greedy_failures = _run_greedy_equality(target, draft, args)
    failures.extend(greedy_failures)
    stochastic = _run_stochastic_check(target, draft, args)
    if not stochastic["passed"]:
        failures.append("Stochastic first-token distributions exceeded tolerance")

    document = {
        "schema_version": 1,
        "provenance": collect_provenance(),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "alias_probe": alias_probe,
        "greedy_equality": greedy_records,
        "stochastic_check": stochastic,
        "failures": failures,
        "passed": not failures,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
        print(args.output)

    if failures:
        print("GPU VERIFICATION FAILED")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("GPU VERIFICATION PASSED")


if __name__ == "__main__":
    main()
