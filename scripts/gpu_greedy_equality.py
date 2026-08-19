from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from torch.profiler import ProfilerActivity, profile
from transformers.cache_utils import DynamicCache

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
    parser.add_argument(
        "--dtype",
        choices=("float16", "bfloat16", "float32"),
        default="bfloat16",
    )
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
        "--clean-reference",
        type=Path,
        help="Clean-run JSON used to select and compare negative-control cases.",
    )
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
    if args.no_clone_logits and args.clean_reference is None:
        raise ValueError("--no-clone-logits requires --clean-reference")
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


def _top_two(logits: torch.Tensor) -> dict[str, Any]:
    values, token_ids = torch.topk(logits.float(), k=2)
    return {
        "token_ids": [int(token_id) for token_id in token_ids.tolist()],
        "logits": [float(value) for value in values.tolist()],
        "absolute_gap": float((values[0] - values[1]).abs().item()),
    }


@torch.inference_mode()
def _score_fixed_sequence(
    target: Any,
    prompt: str,
    token_ids: list[int],
    *,
    batched: bool,
) -> list[torch.Tensor]:
    device = next(target.model.parameters()).device
    encoded = encode_prompt(target.tokenizer, prompt, device)
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    outputs = target.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    scores = [outputs.logits[0, -1].float().cpu()]
    if len(token_ids) <= 1:
        return scores

    continuation = torch.tensor(
        [token_ids[:-1]],
        device=device,
        dtype=input_ids.dtype,
    )
    if batched:
        full_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (1, continuation.shape[1]),
                    device=device,
                    dtype=attention_mask.dtype,
                ),
            ),
            dim=1,
        )
        continuation_outputs = target.model(
            input_ids=continuation,
            attention_mask=full_mask,
            past_key_values=outputs.past_key_values,
            use_cache=True,
        )
        scores.extend(
            row.float().cpu() for row in continuation_outputs.logits[0]
        )
        return scores

    cache = outputs.past_key_values
    running_mask = attention_mask
    for token in continuation[0]:
        running_mask = torch.cat(
            (
                running_mask,
                torch.ones(
                    (1, 1),
                    device=device,
                    dtype=running_mask.dtype,
                ),
            ),
            dim=1,
        )
        step_outputs = target.model(
            input_ids=token.reshape(1, 1),
            attention_mask=running_mask,
            past_key_values=cache,
            use_cache=True,
        )
        cache = step_outputs.past_key_values
        scores.append(step_outputs.logits[0, -1].float().cpu())
    return scores


class _TargetLogitRecorder:
    def __init__(self, target: Any, prompt_length: int) -> None:
        self.target = target
        self.prompt_length = prompt_length
        self.original_forward = target.model.forward
        self.by_output_index: dict[int, list[dict[str, Any]]] = {}
        self.call_index = 0

    def forward(self, *args: Any, **kwargs: Any) -> Any:
        cache = kwargs.get("past_key_values")
        if cache is None:
            cache_length = 0
        elif isinstance(cache, DynamicCache):
            cache_length = int(cache.get_seq_length())
        else:
            cache_length = int(cache.get_seq_length())
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise RuntimeError("Diagnostic recorder requires input_ids")
        outputs = self.original_forward(*args, **kwargs)
        for row_index, row in enumerate(outputs.logits[0]):
            output_index = (
                cache_length + row_index + 1 - self.prompt_length
            )
            if output_index < 0:
                continue
            self.by_output_index.setdefault(output_index, []).append(
                {
                    "call_index": self.call_index,
                    "cache_length": cache_length,
                    "input_length": int(input_ids.shape[1]),
                    "logits": row.detach().float().cpu(),
                }
            )
        self.call_index += 1
        return outputs

    def logits_for(self, output_index: int, emitted_token_id: int) -> dict[str, Any] | None:
        candidates = self.by_output_index.get(output_index, [])
        for candidate in reversed(candidates):
            if int(torch.argmax(candidate["logits"]).item()) == emitted_token_id:
                return candidate
        return candidates[-1] if candidates else None


@contextmanager
def _record_target_logits(
    target: Any,
    prompt: str,
) -> Iterator[_TargetLogitRecorder]:
    device = next(target.model.parameters()).device
    prompt_length = int(
        encode_prompt(target.tokenizer, prompt, device)["input_ids"].shape[1]
    )
    recorder = _TargetLogitRecorder(target, prompt_length)
    target.model.forward = recorder.forward
    try:
        yield recorder
    finally:
        target.model.forward = recorder.original_forward


def _divergence_logit_details(
    divergence_index: int,
    actual_token_id: int,
    baseline_logits: list[torch.Tensor],
    recorder: _TargetLogitRecorder,
) -> dict[str, Any]:
    speculative = recorder.logits_for(divergence_index, actual_token_id)
    if speculative is None:
        return {"logit_diagnostics_error": "No speculative logits recorded"}
    baseline = baseline_logits[divergence_index]
    speculative_logits = speculative["logits"]
    return {
        "baseline_top2": _top_two(baseline),
        "speculative_top2": _top_two(speculative_logits),
        "max_absolute_logit_difference": float(
            torch.max(torch.abs(baseline - speculative_logits)).item()
        ),
        "speculative_target_call_index": speculative["call_index"],
        "speculative_target_cache_length": speculative["cache_length"],
        "speculative_target_input_length": speculative["input_length"],
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


def _run_shape_control(
    target: Any,
    prompt: str,
    token_ids: list[int],
    *,
    incremental: list[torch.Tensor] | None = None,
) -> dict[str, Any]:
    if incremental is None:
        incremental = _score_fixed_sequence(
            target,
            prompt,
            token_ids,
            batched=False,
        )
    batched = _score_fixed_sequence(
        target,
        prompt,
        token_ids,
        batched=True,
    )
    differences = [
        float(torch.max(torch.abs(left - right)).item())
        for left, right in zip(incremental, batched, strict=True)
    ]
    argmax_differences = [
        index
        for index, (left, right) in enumerate(
            zip(incremental, batched, strict=True)
        )
        if int(torch.argmax(left).item()) != int(torch.argmax(right).item())
    ]
    first_argmax_difference = (
        argmax_differences[0] if argmax_differences else None
    )
    result: dict[str, Any] = {
        "max_absolute_logit_difference": max(differences, default=0.0),
        "argmax_difference_count": len(argmax_differences),
        "argmax_difference_indices": argmax_differences,
        "first_argmax_difference": first_argmax_difference,
    }
    if first_argmax_difference is not None:
        index = first_argmax_difference
        result["first_difference_incremental_top2"] = _top_two(incremental[index])
        result["first_difference_batched_top2"] = _top_two(batched[index])
        result["first_difference_max_absolute_logit_difference"] = differences[index]
    print(
        "SHAPE CONTROL: "
        f"max_abs_diff={result['max_absolute_logit_difference']:.8f} "
        f"argmax_differences={len(argmax_differences)} "
        f"first={first_argmax_difference}"
    )
    return result


def _clean_reference_cases(
    path: Path | None,
) -> tuple[set[tuple[str, int, str]], dict[tuple[str, int, str], list[int]]]:
    if path is None:
        return set(), {}
    document = json.loads(path.read_text(encoding="utf-8"))
    grouped: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    for record in document.get("greedy_equality", []):
        key = (
            str(record["workload"]),
            int(record["prompt_index"]),
            str(record["variant"]),
        )
        grouped.setdefault(key, []).append(record)
    passing = {
        key
        for key, records in grouped.items()
        if records and all(bool(record["passed"]) for record in records)
    }
    outputs = {
        key: list(records[0]["actual_output_token_ids"])
        for key, records in grouped.items()
        if key in passing and "actual_output_token_ids" in records[0]
    }
    if passing and len(outputs) != len(passing):
        raise ValueError(
            "Clean reference lacks actual_output_token_ids; rerun the clean "
            "schema-v2 diagnostic before using --no-clone-logits"
        )
    return passing, outputs


def _run_greedy_equality(
    target: Any,
    draft: Any,
    args: argparse.Namespace,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
    dict[str, int],
]:
    records: list[dict[str, Any]] = []
    shape_controls: list[dict[str, Any]] = []
    failures: list[str] = []
    determinism = {
        "baseline_passes": 0,
        "baseline_failures": 0,
        "speculative_passes": 0,
        "speculative_failures": 0,
    }
    clean_cases, clean_outputs = _clean_reference_cases(args.clean_reference)
    negative_control_changes = 0
    variants = (
        ("eager-dynamic-fixed", False, False, 2),
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
            repeated_baseline = generate_baseline(
                target,
                prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=0.0,
                seed=args.seed,
            )
            baseline_deterministic = (
                baseline.output_token_ids == repeated_baseline.output_token_ids
            )
            determinism[
                "baseline_passes" if baseline_deterministic else "baseline_failures"
            ] += 1
            print(
                f"{'PASS' if baseline_deterministic else 'FAIL'} "
                f"{workload}[{prompt_index}] baseline determinism"
            )
            if not baseline_deterministic:
                failures.append(
                    f"{workload}[{prompt_index}] baseline was not deterministic"
                )
            baseline_logits = _score_fixed_sequence(
                target,
                prompt,
                baseline.output_token_ids,
                batched=False,
            )
            shape_control = _run_shape_control(
                target,
                prompt,
                baseline.output_token_ids,
                incremental=baseline_logits,
            )
            shape_controls.append(
                {
                    "workload": workload,
                    "prompt_index": prompt_index,
                    "prompt": prompt,
                    **shape_control,
                }
            )
            for label, compiled, adaptive, repeats in variants:
                case_key = (workload, prompt_index, label)
                if args.no_clone_logits and (
                    not compiled or case_key not in clean_cases
                ):
                    print(
                        f"SKIP {workload}[{prompt_index}] {label}: "
                        "not a clean-passing compiled case"
                    )
                    continue
                first_variant_output: list[int] | None = None
                for replay_index in range(repeats):
                    with _record_target_logits(target, prompt) as recorder:
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
                    divergence_index = details.get("first_divergent_index")
                    if divergence_index is not None:
                        index = int(divergence_index)
                        if (
                            index < len(result.output_token_ids)
                            and index < len(baseline_logits)
                        ):
                            details.update(
                                _divergence_logit_details(
                                    index,
                                    result.output_token_ids[index],
                                    baseline_logits,
                                    recorder,
                                )
                            )
                        else:
                            details["logit_diagnostics_error"] = (
                                "Length-only divergence: one output is a strict "
                                "prefix, so no paired logits exist at this index"
                            )
                    passed = not details
                    if first_variant_output is None:
                        first_variant_output = result.output_token_ids
                        same_path_deterministic = True
                    else:
                        same_path_deterministic = (
                            result.output_token_ids == first_variant_output
                        )
                        determinism[
                            "speculative_passes"
                            if same_path_deterministic
                            else "speculative_failures"
                        ] += 1
                        if not same_path_deterministic:
                            failures.append(
                                f"{workload}[{prompt_index}] {label} "
                                f"replay {replay_index + 1} was not deterministic"
                            )
                    reference_output = clean_outputs.get(case_key)
                    changed_from_clean = (
                        reference_output is not None
                        and result.output_token_ids != reference_output
                    )
                    negative_control_changes += changed_from_clean
                    record = {
                        "workload": workload,
                        "prompt_index": prompt_index,
                        "prompt": prompt,
                        "variant": label,
                        "replay_index": replay_index,
                        "passed": passed,
                        "same_path_deterministic": same_path_deterministic,
                        "changed_from_clean_reference": changed_from_clean,
                        "baseline_tokens": len(baseline.output_token_ids),
                        "speculative_tokens": len(result.output_token_ids),
                        "actual_output_token_ids": result.output_token_ids,
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
    if args.no_clone_logits and negative_control_changes == 0:
        failures.append(
            "No clean-passing generation changed when compiled-logit cloning was disabled"
        )
    determinism["negative_control_changed_generations"] = negative_control_changes
    return records, shape_controls, failures, determinism


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

    (
        greedy_records,
        shape_controls,
        greedy_failures,
        determinism,
    ) = _run_greedy_equality(target, draft, args)
    failures.extend(greedy_failures)
    stochastic = _run_stochastic_check(target, draft, args)
    if not stochastic["passed"]:
        failures.append("Stochastic first-token distributions exceeded tolerance")

    document = {
        "schema_version": 2,
        "provenance": collect_provenance(),
        "configuration": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "alias_probe": alias_probe,
        "same_path_determinism": determinism,
        "target_shape_controls": shape_controls,
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
