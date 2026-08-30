import json
from pathlib import Path

import pytest

from analysis.make_figures import (
    Measurement,
    _speculation_length_series,
    load_measurements,
    make_figures,
)


def _run(
    kind: str,
    throughput: float,
    *,
    workload: str,
    draft_model: str | None = None,
    speculation_length: int | None = None,
    acceptance_rate: float | None = None,
) -> dict[str, object]:
    experiment: dict[str, object] = {"type": kind, "workload": workload}
    summary = {"tokens_per_second_median": throughput}
    if kind == "baseline":
        experiment["model"] = "target"
    else:
        experiment.update(
            {
                "target_model": "target",
                "draft_model": draft_model,
                "speculation_length": speculation_length,
            }
        )
        summary["acceptance_rate"] = acceptance_rate
    return {
        "comparison_id": f"sweep:{workload}",
        "experiment": experiment,
        "summary": summary,
    }


def test_phase3_measurements_match_each_run_to_its_baseline(tmp_path: Path) -> None:
    document = {
        "experiment": {"type": "phase3_sweep"},
        "runs": [
            _run("baseline", 10.0, workload="qa"),
            _run(
                "speculative",
                15.0,
                workload="qa",
                draft_model="draft-1b",
                speculation_length=4,
                acceptance_rate=0.75,
            ),
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    measurements = load_measurements([path])

    assert len(measurements) == 1
    assert measurements[0].speedup == 1.5
    assert measurements[0].acceptance_rate == 0.75


def test_phase3_measurements_require_a_matching_baseline(tmp_path: Path) -> None:
    document = {
        "experiment": {"type": "phase3_sweep"},
        "runs": [
            _run(
                "speculative",
                15.0,
                workload="qa",
                draft_model="draft-1b",
                speculation_length=4,
                acceptance_rate=0.75,
            ),
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="matching baseline"):
        load_measurements([path])


def test_speculation_length_series_preserve_workload_boundaries() -> None:
    measurements = [
        Measurement("draft-1b", "code", 5, 0.8, 1.2),
        Measurement("draft-1b", "code", 9, 0.7, 1.6),
        Measurement("draft-1b", "qa", 5, 0.5, 0.9),
        Measurement("draft-1b", "qa", 9, 0.4, 0.7),
    ]

    series = {
        item.workload: item for item in _speculation_length_series(measurements)
    }

    assert series["code"].lengths == (5, 9)
    assert series["code"].speedups == (1.2, 1.6)
    assert series["qa"].lengths == (5, 9)
    assert series["qa"].speedups == (0.9, 0.7)


def test_make_figures_writes_all_phase3_plots(tmp_path: Path) -> None:
    document = {
        "experiment": {"type": "phase3_sweep"},
        "runs": [
            _run("baseline", 10.0, workload="qa"),
            _run("baseline", 8.0, workload="code"),
            _run(
                "speculative",
                15.0,
                workload="qa",
                draft_model="draft-1b",
                speculation_length=2,
                acceptance_rate=0.8,
            ),
            _run(
                "speculative",
                7.0,
                workload="code",
                draft_model="draft-1b",
                speculation_length=4,
                acceptance_rate=0.4,
            ),
        ],
    }
    path = tmp_path / "sweep.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    outputs = make_figures(load_measurements([path]), tmp_path / "figures")

    assert len(outputs) == 4
    assert all(output.is_file() and output.stat().st_size > 0 for output in outputs)


def test_make_figures_accepts_partial_model_workload_matrix(tmp_path: Path) -> None:
    document = {
        "experiment": {"type": "phase3_sweep"},
        "runs": [
            _run("baseline", 10.0, workload="qa"),
            _run("baseline", 8.0, workload="code"),
            _run(
                "speculative",
                15.0,
                workload="qa",
                draft_model="draft-1b",
                speculation_length=2,
                acceptance_rate=0.8,
            ),
            _run(
                "speculative",
                9.0,
                workload="code",
                draft_model="draft-3b",
                speculation_length=2,
                acceptance_rate=0.6,
            ),
        ],
    }
    path = tmp_path / "partial-sweep.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    outputs = make_figures(load_measurements([path]), tmp_path / "figures")

    assert all(output.is_file() for output in outputs)
