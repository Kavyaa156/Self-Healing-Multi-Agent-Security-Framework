import pytest

from schemas.events import TaskSpec
from tests.fault_injection import inject_fault, run_task_with_fault, VALID_FAULT_TYPES


def make_task(task_id="t1"):
    return TaskSpec(
        task_id=task_id,
        description="Audit access control policies for unauthorized privilege escalation risks",
        initial_input={"target_system": "Authentication Gateway"},
    )


# ---------------------------------------------------------------------------
# inject_fault() -- pure, no pipeline needed
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# run_task_with_fault() -- REAL pipeline (P1 -> P2 -> P3 -> P4), no faults
# ---------------------------------------------------------------------------
def test_clean_task_high_scores_and_not_detected(workflow, fake_validation_store):
    task = make_task("t_clean")
    result = run_task_with_fault(
        workflow, task, fault_type=None, k=3,
        weights=(0.2, 0.3, 0.5), validation_store=fake_validation_store, seed=1,
    )
    score = result["score"]
    assert score.C > 0.65 and score.S > 0.65 and score.E > 0.65
    assert result["detected"] is False
    assert result["diagnosis"] is None


# ---------------------------------------------------------------------------
# Each F-type actually degrades the component Section 9 says it should,
# and gets classified correctly once detected.
# ---------------------------------------------------------------------------
def test_f1_drops_semantic_accuracy(workflow, fake_validation_store):
    # multi_step_reasoning weights here (S weight=0.4) -- F1's whole signal
    # is a dropped S, so it needs a weight profile where S actually pulls
    # R down. Under api_orchestration weights (S weight=0.3, E weight=0.5)
    # a lone S crash doesn't reliably cross theta; see run_evaluation()'s
    # docstring in tests/evaluate.py for the same weight-sensitivity point.
    task = make_task("t_f1")
    result = run_task_with_fault(
        workflow, task, fault_type="F1", k=3,
        weights=(0.4, 0.4, 0.2), validation_store=fake_validation_store, seed=1,
    )
    assert result["score"].S < 0.3
    assert result["detected"] is True
    assert result["diagnosis"].failure_type == "F1"


def test_f2_drops_execution_rate_and_marks_failure(workflow, fake_validation_store):
    task = make_task("t_f2")
    result = run_task_with_fault(
        workflow, task, fault_type="F2", k=3,
        weights=(0.2, 0.3, 0.5), validation_store=fake_validation_store, seed=1,
    )
    assert result["score"].E == 0.0
    assert result["detected"] is True
    assert result["diagnosis"].failure_type == "F2"


def test_f3_drops_consistency(workflow, fake_validation_store):
    task = make_task("t_f3")
    result = run_task_with_fault(
        workflow, task, fault_type="F3", k=3,
        weights=(0.4, 0.4, 0.2), validation_store=fake_validation_store, seed=1,
    )
    assert result["score"].C < 0.7


def test_f4_marks_downstream_failure_and_classifies_as_propagation(workflow, fake_validation_store):
    task = make_task("t_f4")
    result = run_task_with_fault(
        workflow, task, fault_type="F4", k=3,
        weights=(0.2, 0.3, 0.5), validation_store=fake_validation_store, seed=1,
    )
    assert result["score"].E == 0.0
    assert result["detected"] is True
    assert result["diagnosis"].failure_type == "F4"