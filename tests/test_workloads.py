import pytest

from specdec.workloads import WORKLOADS, get_workload


def test_each_workload_has_multiple_prompts() -> None:
    assert all(len(prompts) >= 3 for prompts in WORKLOADS.values())


def test_unknown_workload_is_rejected() -> None:
    with pytest.raises(ValueError, match="Unknown workload"):
        get_workload("missing")
