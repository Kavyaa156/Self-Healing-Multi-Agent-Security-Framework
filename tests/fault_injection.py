"""
tests/fault_injection.py

P4 Task 1 -- Fault Injection Harness.

Scope: implements inject_fault() for the four labeled fault TYPES the
diagnosis classifier must distinguish (F1-F4), following Section 9's
scenarios 1-4 exactly. Scenarios 5 and 6 in Section 9 ("repeated tool
failure pattern" and "abnormal execution sequence") are PATTERN triggers
for the *detector*, not F-type labels for the *classifier* -- they're
exercised directly in tests/evaluate.py against the mock ExecutionGraph
instead of through inject_fault(), since they describe telemetry patterns
rather than a property of the TaskSpec.

Design note [PROJECT DESIGN DECISION]: because P1's real run_task() does
not exist yet, inject_fault() cannot literally "make the tool-agent skip
its tool call" inside a live LLM pipeline. What it CAN do reproducibly is
annotate the TaskSpec with a machine-readable fault directive that:
  (a) a real run_task() can later read and honor (the intended long-term
      integration path), and
  (b) this harness's own deterministic simulator (`simulate_execution`)
      reads right now, to produce the ExecutionEvent/ReliabilityScore/
      graph-flag combination Section 9's table says that scenario should
      produce -- so detect_failure()/classify_failure() can be developed
      and unit-tested today without waiting on P1/P2/P3.

simulate_execution() is clearly a MOCK/SIMULATION, not a claim about what
Jeong's or P1's real system does. It must be deleted/replaced once real
integration (Phase 6) is possible.
"""

from __future__ import annotations

import random
from typing import Any

from schemas_mock.events import ExecutionEvent, ReliabilityScore, TaskSpec

VALID_FAULT_TYPES = {"F1", "F2", "F3", "F4"}

# Section 9's concrete injection method per fault type, reproduced for
# traceability from plan doc, verbatim in spirit:
FAULT_DESCRIPTIONS = {
    "F1": {
        "name": "Hallucination Error",
        "injected_where": "tool-using agent step",
        "how": "Force the tool-agent to answer without calling the tool, "
        "contradicting the validation DB.",
        "observable_symptom": "S score drops sharply (output doesn't match "
        "the validation store).",
        "p2_p3_signal": "P3's compute_semantic_accuracy() output (S).",
        "expected_label": "F1",
    },
    "F2": {
        "name": "Execution Error",
        "injected_where": "tool_call event",
        "how": "Mock a tool to raise a timeout/exception.",
        "observable_symptom": "success=False on the tool_call event; "
        "E score drops.",
        "p2_p3_signal": "P3's compute_execution_rate() output (E); "
        "P2's success flag on the ExecutionEvent.",
        "expected_label": "F2",
    },
    "F3": {
        "name": "Reasoning Inconsistency",
        "injected_where": "K resampled trajectories of the same task",
        "how": "Add a temperature/prompt perturbation so K resampled runs "
        "diverge in plan.",
        "observable_symptom": "C score drops (trajectories disagree).",
        "p2_p3_signal": "P3's compute_consistency() output (C).",
        "expected_label": "F3",
    },
    "F4": {
        "name": "Workflow Propagation Error",
        "injected_where": "an early step, observed at a later dependent step",
        "how": "Inject an F2 fault at an early step; downstream dependent "
        "steps then also fail.",
        "observable_symptom": "Execution graph shows failure at a node "
        "whose upstream dependency also failed.",
        "p2_p3_signal": "P2's ExecutionGraph ancestor-failure relationship "
        "(see mocks/execution_graph.py get_upstream_failed).",
        "expected_label": "F4",
    },
}


def inject_fault(task: TaskSpec, fault_type: str) -> TaskSpec:
    """Return a NEW TaskSpec annotated with a fault directive.

    Does not mutate `task`. Deterministic given the same task + fault_type
    (no randomness at this stage -- randomness, where needed for realistic
    variation, lives in simulate_execution(), seeded explicitly there).

    Raises:
        ValueError: if fault_type is not one of F1-F4.
    """
    if fault_type not in VALID_FAULT_TYPES:
        raise ValueError(
            f"Unknown fault_type {fault_type!r}; must be one of {sorted(VALID_FAULT_TYPES)}. "
            "Do not invent fault types outside Jeong's F1-F4 taxonomy."
        )

    new_input = dict(task.initial_input)
    new_input["fault_injection"] = {
        "fault_type": fault_type,
        **FAULT_DESCRIPTIONS[fault_type],
    }
    return TaskSpec(
        task_id=task.task_id,
        description=task.description,
        initial_input=new_input,
    )


# ---------------------------------------------------------------------------
# [MOCK / SIMULATION LAYER -- not part of the required P4 interface, but
# needed to exercise detect_failure()/classify_failure() before P1-P3 exist.]
# ---------------------------------------------------------------------------
def simulate_execution(
    task: TaskSpec,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    threshold: float = 0.65,
    seed: int | None = None,
) -> tuple[ReliabilityScore, ExecutionEvent, dict[str, bool]]:
    """Deterministically simulate what P2's ExecutionEvent + P3's
    ReliabilityScore + P2's pattern flags would plausibly look like for a
    task, honoring whatever fault_injection directive inject_fault() set
    (or none, for a clean/no-fault task).

    This is a STAND-IN, not a claim about real system behavior. Numeric
    values are chosen to be clearly on the "should trigger this label"
    side of the relevant threshold, per Section 9's stated symptoms --
    they are illustrative, not measured, and must not be reported as
    experimental results.
    """
    rng = random.Random(seed if seed is not None else hash(task.task_id) & 0xFFFF)
    fault = task.initial_input.get("fault_injection")
    w1, w2, w3 = weights

    C, S, E = 0.9, 0.9, 0.9  # clean-task baseline, all high
    success = True
    repeated_failure = False
    abnormal_sequence = False

    fault_type = fault["fault_type"] if fault else None

    if fault_type == "F1":
        S = round(rng.uniform(0.0, 0.2), 3)  # sharp S drop, per Section 9 #1
    elif fault_type == "F2":
        E = round(rng.uniform(0.0, 0.3), 3)  # E drop
        success = False
    elif fault_type == "F3":
        C = round(rng.uniform(0.0, 0.3), 3)  # C drop from divergent trajectories
    elif fault_type == "F4":
        E = round(rng.uniform(0.0, 0.3), 3)  # originating F2 at an earlier step
        success = False
        # F4's distinguishing signal is graph-level (upstream also failed),
        # not a component score by itself -- callers should also check
        # ExecutionGraph.get_upstream_failed() for this task.
    elif fault_type == "REPEATED_FAILURE_PATTERN":
        repeated_failure = True
    elif fault_type == "ABNORMAL_SEQUENCE_PATTERN":
        abnormal_sequence = True

    R = round(w1 * C + w2 * S + w3 * E, 4)

    score = ReliabilityScore(
        task_id=task.task_id,
        step_id=1,
        C=C,
        S=S,
        E=E,
        R=R,
        weights=(w1, w2, w3),
        threshold=threshold,
    )
    event = ExecutionEvent(
        task_id=task.task_id,
        agent_id="tool_agent",
        step_id=1,
        event_type="tool_call",
        content=task.description,
        tool_name="search",
        success=success,
        timestamp=0.0,
    )
    graph_flags = {
        "repeated_failure": repeated_failure,
        "abnormal_sequence": abnormal_sequence,
    }
    return score, event, graph_flags
