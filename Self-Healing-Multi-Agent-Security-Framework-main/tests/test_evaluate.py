from reliability.reliability_score import DEFAULT_WEIGHTS
from tests.evaluate import build_sample_task_set, run_evaluation


def test_run_evaluation_produces_report_with_expected_shape(workflow, fake_validation_store):
    tasks, fault_labels = build_sample_task_set(n_per_type=2)
    report = run_evaluation(
        tasks, fault_labels, workflow=workflow,
        weights=DEFAULT_WEIGHTS["api_orchestration"],
        validation_store=fake_validation_store,
    )

    assert report.n_tasks == len(tasks)
    assert 0.0 <= report.fda <= 1.0
    assert 0.0 <= report.false_positive_rate <= 1.0
    assert set(report.per_type_accuracy.keys()) <= {"F1", "F2", "F3", "F4"}
    assert report.execution_overhead_seconds >= 0.0
    assert len(report.raw_rows) == len(tasks)


def test_f1_and_f3_reliably_caught_under_multi_step_reasoning_weights(workflow, fake_validation_store):
    # F1 (hits S) and F3 (hits C) both have weight 0.4 under
    # multi_step_reasoning weights, so they reliably push R below theta.
    tasks, fault_labels = build_sample_task_set(n_per_type=5)
    report = run_evaluation(
        tasks, fault_labels, workflow=workflow,
        weights=DEFAULT_WEIGHTS["multi_step_reasoning"],
        validation_store=fake_validation_store,
    )
    assert report.per_type_accuracy["F1"] > 0.9
    assert report.per_type_accuracy["F3"] > 0.7


def test_f2_detection_improves_with_api_orchestration_weights(workflow, fake_validation_store):
    # KNOWN FINDING: F2/F4 only degrade E. Under multi_step_reasoning
    # weights (E weight=0.2), a cratered E alone often does NOT push R
    # below theta=0.65 -- this is real weight-driven behavior, not a bug,
    # and is exactly why Section 10 asks for per-type accuracy reporting.
    # Switching to api_orchestration weights (E weight=0.5) should
    # recover F2 detection.
    tasks, fault_labels = build_sample_task_set(n_per_type=5)
    weak_weights_report = run_evaluation(
        tasks, fault_labels, workflow=workflow,
        weights=DEFAULT_WEIGHTS["multi_step_reasoning"],
        validation_store=fake_validation_store,
    )
    strong_weights_report = run_evaluation(
        tasks, fault_labels, workflow=workflow,
        weights=DEFAULT_WEIGHTS["api_orchestration"],
        validation_store=fake_validation_store,
    )
    assert (
        strong_weights_report.per_type_accuracy["F2"]
        >= weak_weights_report.per_type_accuracy["F2"]
    )
    assert strong_weights_report.per_type_accuracy["F2"] > 0.9


def test_save_writes_valid_json(tmp_path, workflow, fake_validation_store):
    tasks, fault_labels = build_sample_task_set(n_per_type=2)
    report = run_evaluation(
        tasks, fault_labels, workflow=workflow,
        weights=DEFAULT_WEIGHTS["api_orchestration"],
        validation_store=fake_validation_store,
    )
    out_path = tmp_path / "results.json"
    report.save(str(out_path))
    assert out_path.exists()
    import json

    with open(out_path) as f:
        data = json.load(f)
    assert "summary" in data and "rows" in data