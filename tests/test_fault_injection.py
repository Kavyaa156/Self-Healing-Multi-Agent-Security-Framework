import pytest

from schemas_mock.events import TaskSpec
from tests.fault_injection import inject_fault, simulate_execution, VALID_FAULT_TYPES


def make_task(task_id="t1"):
    return TaskSpec(task_id=task_id, description="test task", initial_input={})


@pytest.mark.parametrize("fault_type", sorted(VALID_FAULT_TYPES))
def test_inject_fault_valid_types(fault_type):
    task = make_task()
    injected = inject_fault(task, fault_type)
    assert injected.initial_input["fault_injection"]["fault_type"] == fault_type
    # original task must not be mutated
    assert "fault_injection" not in task.initial_input


def test_inject_fault_invalid_type_raises():
    task = make_task()
    with pytest.raises(ValueError):
        inject_fault(task, "F5")


def test_inject_fault_deterministic():
    task = make_task()
    a = inject_fault(task, "F1")
    b = inject_fault(task, "F1")
    assert a.initial_input == b.initial_input


def test_simulate_clean_task_high_scores():
    task = make_task()
    score, event, flags = simulate_execution(task, seed=1)
    assert score.C > 0.65 and score.S > 0.65 and score.E > 0.65
    assert event.success is True
    assert not flags["repeated_failure"] and not flags["abnormal_sequence"]


def test_simulate_f1_drops_semantic_accuracy():
    task = inject_fault(make_task(), "F1")
    score, _, _ = simulate_execution(task, seed=1)
    assert score.S < 0.3


def test_simulate_f2_drops_execution_rate_and_marks_failure():
    task = inject_fault(make_task(), "F2")
    score, event, _ = simulate_execution(task, seed=1)
    assert score.E < 0.4
    assert event.success is False


def test_simulate_f3_drops_consistency():
    task = inject_fault(make_task(), "F3")
    score, _, _ = simulate_execution(task, seed=1)
    assert score.C < 0.4


def test_simulate_f4_marks_execution_failure():
    task = inject_fault(make_task(), "F4")
    score, event, _ = simulate_execution(task, seed=1)
    assert event.success is False
    assert score.E < 0.4
