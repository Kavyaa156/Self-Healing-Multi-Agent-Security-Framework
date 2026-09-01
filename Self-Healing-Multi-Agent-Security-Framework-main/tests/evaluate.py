"""
tests/evaluate.py

P4 evaluation harness, per Section 10 ("Baseline Evaluation").
[UPDATED for Phase 6 real integration]

Metrics implemented (all named directly in Section 10):
  - Failure Detection Accuracy (FDA)
  - False Positive Rate
  - Per-type detection accuracy (F1/F2/F3/F4)
  - Diagnosis accuracy
  - Execution overhead
  - Wilcoxon signed-rank significance test (scipy.stats.wilcoxon)

This now runs every task through tests.fault_injection.run_task_with_fault(),
which drives the REAL pipeline (P1 workflow -> P2 telemetry -> P3 C/S/E/R ->
P4 detect/diagnose) instead of the old simulate_execution() mock. Nothing
about the metric-computation logic below changed from the original design
-- only where (score, detected, diagnosis) come from.

Wilcoxon usage: Section 10 says to test the framework's detection against
"a no-detection baseline." Concretely: for each task, record whether R
alone (no pattern check) would have flagged it, versus whether the full
hybrid detector (R OR pattern) flagged it, then run a paired Wilcoxon
signed-rank test over the two per-task correctness sequences (1 = correct
vs ground truth, 0 = incorrect). This mirrors Jeong's own use of Wilcoxon
(§4.1). Only matched per-task pairs are valid here.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd
from scipy.stats import wilcoxon

from schemas.events import TaskSpec
from agents.workflow import MultiAgentWorkflow
from reliability.semantic_accuracy import ValidationStore
from tests.fault_injection import inject_fault, run_task_with_fault


@dataclass
class EvalReport:
    n_tasks: int
    fda: float  # Failure Detection Accuracy
    false_positive_rate: float
    per_type_accuracy: dict[str, float]
    diagnosis_accuracy: float
    execution_overhead_seconds: float
    wilcoxon_statistic: Optional[float]
    wilcoxon_p_value: Optional[float]
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
            json.dump({"summary": self.to_dict(), "rows": self.raw_rows}, f, indent=2)


def run_evaluation(
    tasks: list[TaskSpec],
    fault_labels: dict[str, Optional[str]],
    workflow: MultiAgentWorkflow,
    theta: float = 0.65,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    k: int = 3,
    validation_store: Optional[ValidationStore] = None,
) -> EvalReport:
    """
    Run the REAL fault-injection + detection + diagnosis pipeline over
    `tasks` and compute Section 10's metrics.

    Args:
        tasks: list of CLEAN TaskSpec (fault_labels tells this function
            which ones to inject and with what -- do NOT pre-inject them
            yourself, run_task_with_fault() calls inject_fault() internally).
        fault_labels: ground truth, task_id -> "F1"/"F2"/"F3"/"F4" for
            faulty tasks, or None for clean tasks.
        workflow: a real MultiAgentWorkflow instance (P1), shared across
            all tasks in this run.
        theta: detection threshold passed through to detect_failure().
        weights: (w1, w2, w3) for C, S, E. Pick the DEFAULT_WEIGHTS entry
            (see reliability/reliability_score.py) closest to your task
            domain -- api_orchestration weights (E weight=0.5) are needed
            to reliably catch F2/F4 in this project's minimal 3-node
            pipeline, since a single tool_call failure alone often won't
            drag R below theta under multi_step_reasoning weights (E
            weight=0.2). This is a real, documented weight-sensitivity
              finding, not a bug -- report per-type accuracy under more
            than one weight profile if you want to show it in the paper.
        k: number of resampled trajectories per task for C.
        validation_store: P3's seeded ValidationStore for real S scoring.
            Build one with reliability/seed_validation_store.seed_store()
            (needs internet on first run to download the embedding model).
            If None, S is stubbed to 1.0 for every task -- fine for
            smoke-testing detection/diagnosis wiring, but F1 will never
            be caught, since F1's whole signal IS a real S drop.

    Returns:
        EvalReport with computed metrics and per-task raw rows.
    """
    rows = []
    no_detection_correct = []  # R-only baseline, per-task correctness (0/1)
    hybrid_correct = []        # full hybrid detector, per-task correctness (0/1)

    start = time.perf_counter()
    for task in tasks:
        ground_truth = fault_labels.get(task.task_id)  # "F1".."F4" or None

        result = run_task_with_fault(
            workflow=workflow,
            task=task,
            fault_type=ground_truth,
            k=k,
            weights=weights,
            theta=theta,
            validation_store=validation_store,
        )
        score = result["score"]
        detected = result["detected"]
        diagnosis = result["diagnosis"]

        r_only_detected = score.R < theta  # pattern-blind baseline
        expected_detected = ground_truth is not None

        no_detection_correct.append(int(r_only_detected == expected_detected))
        hybrid_correct.append(int(detected == expected_detected))

        predicted_type = diagnosis.failure_type if diagnosis else None

        rows.append({
            "task_id": task.task_id,
            "ground_truth_fault": ground_truth,
            "expected_detected": expected_detected,
            "detected": detected,
            "r_only_detected": r_only_detected,
            "predicted_type": predicted_type,
            "root_cause_agent": diagnosis.root_cause_agent if diagnosis else None,
            "root_cause_step": diagnosis.root_cause_step if diagnosis else None,
            "attribution_confidence": diagnosis.attribution_confidence if diagnosis else None,
            "C": score.C, "S": score.S, "E": score.E, "R": score.R,
        })
    overhead = time.perf_counter() - start

    df = pd.DataFrame(rows)
    injected = df[df["ground_truth_fault"].notna()]
    clean = df[df["ground_truth_fault"].isna()]

    fda = (
        (injected["detected"] == injected["expected_detected"]).mean()
        if len(injected) > 0 else float("nan")
    )
    false_positive_rate = clean["detected"].mean() if len(clean) > 0 else float("nan")

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
        if len(diagnosed) > 0 else float("nan")
    )

    diffs = [h - r for h, r in zip(hybrid_correct, no_detection_correct)]
    if any(d != 0 for d in diffs):
        stat, p_value = wilcoxon(hybrid_correct, no_detection_correct)
        wilcoxon_note = (
            "Paired per-task correctness (1=correct vs ground truth, "
            "0=incorrect): full hybrid detector vs R-only (no pattern "
            "check) baseline, mirroring Jeong's use of Wilcoxon (§4.1)."
        )
    else:
        stat, p_value = None, None
        wilcoxon_note = (
            "Wilcoxon not run: hybrid and R-only baseline produced "
            "identical correctness on every task in this sample. Include "
            "a repeated-failure or abnormal-sequence scenario, or run "
            "more tasks, to get a non-degenerate comparison."
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


def build_sample_task_set(n_per_type: int = 3) -> tuple[list[TaskSpec], dict]:
    """
    Build a small mixed set of clean + to-be-injected tasks. Section 10's
    course-project-scale recommendation is ~20-30 tasks total (Jeong's own
    experiment used 100/task-type, 300 total -- do not confuse these dev
    counts with either of those real figures).

    NOTE: tasks returned here are CLEAN (fault_injection is applied later,
    inside run_evaluation -> run_task_with_fault). fault_labels is the
    ground-truth map you pass to run_evaluation.
    """
    tasks: list[TaskSpec] = []
    fault_labels: dict[str, Optional[str]] = {}
    tid = 0

    for f_type in ("F1", "F2", "F3", "F4"):
        for _ in range(n_per_type):
            tid += 1
            task_id = f"eval_t{tid}"
            tasks.append(TaskSpec(
                task_id=task_id,
                description="Audit access control policies for unauthorized privilege escalation risks",
                initial_input={"target_system": "Authentication Gateway"},
            ))
            fault_labels[task_id] = f_type

    for _ in range(n_per_type):
        tid += 1
        task_id = f"eval_t{tid}"
        tasks.append(TaskSpec(
            task_id=task_id,
            description="Audit access control policies for unauthorized privilege escalation risks",
            initial_input={"target_system": "Authentication Gateway"},
        ))
        fault_labels[task_id] = None

    return tasks, fault_labels


if __name__ == "__main__":
    from reliability.seed_validation_store import seed_store

    workflow = MultiAgentWorkflow()
    store = seed_store()  # needs internet the first time (downloads the embedding model)

    tasks, fault_labels = build_sample_task_set(n_per_type=2)
    report = run_evaluation(
        tasks, fault_labels, workflow=workflow,
        weights=(0.4, 0.4, 0.2),  # api_orchestration weights -- see run_evaluation docstring
        validation_store=store,
        k=2,
    )
    print(json.dumps(report.to_dict(), indent=2))
    report.save("evaluation_report_with_attribution.json")
    print("\nSaved full report (with per-task rows) to evaluation_report_with_attribution.json")