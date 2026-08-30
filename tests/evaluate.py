"""
tests/evaluate.py

P4 evaluation harness, per Section 10 ("Baseline Evaluation").

Metrics implemented, all named directly in Section 10:
  - Failure Detection Accuracy (FDA)
  - False Positive Rate
  - Per-type detection accuracy (F1/F2/F3/F4)
  - Diagnosis accuracy
  - Execution overhead
  - Wilcoxon signed-rank significance test (scipy.stats.wilcoxon)

IMPORTANT SCOPE NOTE: this harness currently runs against
tests.fault_injection.simulate_execution(), the MOCK simulator, because
P1's real run_task(), P2's real ExecutionGraph, and P3's real
compute_reliability() do not exist yet. Section 6 (Phase 6, Real
Integration) is: swap `simulate_execution(...)` for calls into the real
P1->P2->P3 pipeline. Nothing in `run_evaluation()`'s metric-computation
logic should need to change -- only where (score, event, graph_flags)
come from.

Wilcoxon usage: Section 10 says to test the framework's detection
against "a no-detection baseline." Concretely here that means: for each
task, record whether R alone (no pattern check) would have flagged it,
versus whether the full hybrid detector (R OR pattern) flagged it, then
run a paired Wilcoxon signed-rank test over the two per-task correctness
sequences (1 = correct vs ground truth, 0 = incorrect). This directly
mirrors Jeong's own use of Wilcoxon (§4.1) to compare a
before/after-detection condition. Do NOT use Wilcoxon on quantities
where pairing doesn't make sense (e.g. comparing across different task
IDs) -- only matched per-task pairs are valid here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import pandas as pd
from scipy.stats import wilcoxon

from detection.decision_controller import detect_failure
from diagnosis.classifier import build_diagnosis
from mocks.execution_graph import ExecutionGraph
from schemas_mock.events import TaskSpec
from tests.fault_injection import inject_fault, simulate_execution


@dataclass
class EvalReport:
    n_tasks: int
    fda: float  # Failure Detection Accuracy
    false_positive_rate: float
    per_type_accuracy: dict[str, float]
    diagnosis_accuracy: float
    execution_overhead_seconds: float
    wilcoxon_statistic: float | None
    wilcoxon_p_value: float | None
    wilcoxon_note: str
    raw_rows: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n_tasks": self.n_tasks,
            "fda": self.fda,
            "false_positive_rate": self.false_positive_rate,
            "per_type_accuracy": self.per_type_accuracy,
            "diagnosis_accuracy": self.diagnosis_accuracy,
            "execution_overhead_seconds": self.execution_overhead_seconds,
            "wilcoxon_statistic": self.wilcoxon_statistic,
            "wilcoxon_p_value": self.wilcoxon_p_value,
            "wilcoxon_note": self.wilcoxon_note,
        }

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump(
                {"summary": self.to_dict(), "rows": self.raw_rows}, f, indent=2
            )


def run_evaluation(
    tasks: list[TaskSpec],
    fault_labels: dict[str, str | None],
    theta: float = 0.65,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
) -> EvalReport:
    """Run the fault-injection + detection + diagnosis pipeline over
    `tasks` and compute Section 10's metrics.

    Args:
        tasks: list of TaskSpec, some already fault-injected (via
            inject_fault) and some clean.
        fault_labels: ground truth, task_id -> one of "F1"/"F2"/"F3"/"F4"
            for injected tasks, or None for clean (non-injected) tasks.
        theta: detection threshold passed through to detect_failure().
        weights: (w1, w2, w3) for C, S, E passed through to
            simulate_execution()/compute_reliability(). Defaults to
            Jeong's multi_step_reasoning weights (§4.1). NOTE: which
            weight triple is appropriate depends on task domain -- see
            "Known finding" in the README. Pass DEFAULT_WEIGHTS["api_orchestration"]
            (or whichever fits your actual task domain) explicitly rather
            than relying on this default when running a real evaluation.

    Returns:
        EvalReport with computed metrics and per-task raw rows for
        further analysis (e.g. loading into pandas for plots/tables).
    """
    rows = []
    graph = ExecutionGraph()
    no_detection_correct = []  # R-only baseline, per-task correctness (0/1)
    hybrid_correct = []  # full hybrid detector, per-task correctness (0/1)

    start = time.perf_counter()
    for task in tasks:
        ground_truth = fault_labels.get(task.task_id)  # "F1".."F4" or None
        score, event, graph_flags = simulate_execution(
            task, weights=weights, threshold=theta
        )
        graph.add_event(event)

        detected = detect_failure(score, graph_flags, theta=theta)
        r_only_detected = score.R < theta  # no-detection (pattern-blind) baseline

        expected_detected = ground_truth is not None

        no_detection_correct.append(int(r_only_detected == expected_detected))
        hybrid_correct.append(int(detected == expected_detected))

        predicted_type = None
        if detected:
            diagnosis = build_diagnosis(score, graph, event)
            predicted_type = diagnosis.failure_type

        rows.append(
            {
                "task_id": task.task_id,
                "ground_truth_fault": ground_truth,
                "expected_detected": expected_detected,
                "detected": detected,
                "r_only_detected": r_only_detected,
                "predicted_type": predicted_type,
                "C": score.C,
                "S": score.S,
                "E": score.E,
                "R": score.R,
            }
        )
    overhead = time.perf_counter() - start

    df = pd.DataFrame(rows)

    injected = df[df["ground_truth_fault"].notna()]
    clean = df[df["ground_truth_fault"].isna()]

    fda = (
        (injected["detected"] == injected["expected_detected"]).mean()
        if len(injected) > 0
        else float("nan")
    )
    false_positive_rate = (
        clean["detected"].mean() if len(clean) > 0 else float("nan")
    )

    per_type_accuracy: dict[str, float] = {}
    for f_type in ("F1", "F2", "F3", "F4"):
        subset = injected[injected["ground_truth_fault"] == f_type]
        if len(subset) > 0:
            per_type_accuracy[f_type] = (
                subset["detected"] == subset["expected_detected"]
            ).mean()

    diagnosed = injected[injected["detected"]]
    diagnosis_accuracy = (
        (diagnosed["predicted_type"] == diagnosed["ground_truth_fault"]).mean()
        if len(diagnosed) > 0
        else float("nan")
    )

    # Wilcoxon: paired per-task correctness, hybrid vs R-only baseline.
    # wilcoxon() requires at least one non-zero difference; guard for the
    # degenerate all-identical case rather than letting scipy raise.
    diffs = [h - r for h, r in zip(hybrid_correct, no_detection_correct)]
    if any(d != 0 for d in diffs):
        stat, p_value = wilcoxon(hybrid_correct, no_detection_correct)
        wilcoxon_note = (
            "Paired per-task correctness (1=correct vs ground truth, "
            "0=incorrect): full hybrid detector vs R-only (no pattern "
            "check) baseline, mirroring Jeong's use of Wilcoxon (§4.1) to "
            "compare a with/without-detection-component condition."
        )
    else:
        stat, p_value = None, None
        wilcoxon_note = (
            "Wilcoxon not run: hybrid and R-only baseline produced "
            "identical correctness on every task in this sample (no "
            "non-zero differences), so the test is undefined. This "
            "usually means no task in the sample triggered a pattern "
            "flag without also crossing the R threshold -- include a "
            "repeated-failure or abnormal-sequence scenario to test that."
        )

    return EvalReport(
        n_tasks=len(tasks),
        fda=fda,
        false_positive_rate=false_positive_rate,
        per_type_accuracy=per_type_accuracy,
        diagnosis_accuracy=diagnosis_accuracy,
        execution_overhead_seconds=overhead,
        wilcoxon_statistic=stat,
        wilcoxon_p_value=p_value,
        wilcoxon_note=wilcoxon_note,
        raw_rows=rows,
    )


def build_sample_task_set(n_per_type: int = 5) -> tuple[list[TaskSpec], dict]:
    """[TEST/DEV HELPER] Build a small mixed set of clean + injected tasks
    for exercising run_evaluation() before real P1 tasks exist. Scale
    numbers are illustrative for local testing only -- Section 10's actual
    course-project-scale recommendation is ~20-30 tasks total; Jeong's own
    experiment used 100/task-type, 300 total, 30% fault rate (§4.1) -- do
    not confuse this dev helper's counts with either of those real
    figures.
    """
    tasks: list[TaskSpec] = []
    fault_labels: dict[str, str | None] = {}
    tid = 0

    for f_type in ("F1", "F2", "F3", "F4"):
        for _ in range(n_per_type):
            tid += 1
            task_id = f"t{tid}"
            base = TaskSpec(task_id=task_id, description=f"sample task {task_id}")
            tasks.append(inject_fault(base, f_type))
            fault_labels[task_id] = f_type

    for _ in range(n_per_type):
        tid += 1
        task_id = f"t{tid}"
        tasks.append(TaskSpec(task_id=task_id, description=f"clean task {task_id}"))
        fault_labels[task_id] = None

    return tasks, fault_labels


if __name__ == "__main__":
    tasks, fault_labels = build_sample_task_set(n_per_type=5)
    report = run_evaluation(tasks, fault_labels)
    print(json.dumps(report.to_dict(), indent=2))
