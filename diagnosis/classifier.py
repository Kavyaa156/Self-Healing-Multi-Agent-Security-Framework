"""
diagnosis/classifier.py

P4 Task 3 -- Failure Diagnosis.

RESEARCH FIDELITY NOTE (read before touching this file):
Jeong et al. define WHAT F1-F4 mean (§3.2) but do NOT give an algorithm
for mapping an observed event/score/graph state to one of the four
labels. The mapping implemented below is entirely a [PROJECT
IMPLEMENTATION -- OUR ADAPTATION], informed by:
  - the taxonomy's own definitions (which component each fault type
    would plausibly degrade), and
  - Section 9's worked examples, which are the plan's own (not Jeong's)
    illustrative mapping of "which score drops -> which label":
        F1 (Hallucination)          -> S drops sharply
        F2 (Execution Error)        -> E drops / success=False events
        F3 (Reasoning Inconsistency) -> C drops
        F4 (Workflow Propagation)    -> failure at a node whose upstream
                                         dependency also failed (graph-level,
                                         not a component-score signal)

Rule used here, exactly as it will be documented for the team:
    1. If the ExecutionGraph shows this event's step has a failed
       upstream ancestor -> F4 (propagation overrides other signals,
       since an upstream-caused failure isn't "about" this step's own
       C/S/E profile).
    2. Otherwise, look at whichever of C/S/E is LOWEST and treat that as
       the most-degraded component:
           lowest = S  -> F1
           lowest = E  -> F2
           lowest = C  -> F3
    3. Ties are broken in the fixed order S, E, C (i.e. F1 before F2
       before F3), documented here rather than left implicit.

This is a transparent, inspectable rule set -- not a learned classifier,
per the plan's explicit instruction not to introduce an ML model here.
"""

from __future__ import annotations

from schemas.events import ExecutionEvent, FailureDiagnosis, ReliabilityScore
from monitoring.execution_graph import ExecutionGraph

VALID_LABELS = ("F1", "F2", "F3", "F4")


def classify_failure(
    score: ReliabilityScore,
    graph: ExecutionGraph,
    event: ExecutionEvent,
) -> str:
    """Return one of "F1"/"F2"/"F3"/"F4" for an already-detected failure.

    Preconditions: caller has already established `detect_failure(...)`
    returned True for this (score, graph_flags) pair. This function does
    not re-check that -- classifying a non-failure is meaningless, and
    it's the caller's job (build_diagnosis / run_evaluation) to only
    invoke this after detection.

    Args:
        score: P3's ReliabilityScore (C, S, E, R) for this task/step.
        graph: P2's ExecutionGraph (or the mock in mocks/execution_graph.py)
            for the F4 upstream-ancestor check.
        event: the ExecutionEvent this diagnosis concerns -- used for
            task_id/step_id and, in a future extension, event content.

    Returns:
        One of "F1", "F2", "F3", "F4".
    """
    # Step 1: F4 override -- propagation takes precedence over component
    # scores, per Section 9 scenario 4's framing (graph-level signal).
    if graph.get_upstream_failed(event.task_id, event.step_id):
        return "F4"

    # Step 2: lowest-contributing component decides the type.
    # Fixed tie-break order: S, E, C  ->  F1, F2, F3
    components = [("F1", score.S), ("F2", score.E), ("F3", score.C)]
    label, _ = min(components, key=lambda pair: pair[1])
    return label


def build_diagnosis(
    score: ReliabilityScore,
    graph: ExecutionGraph,
    event: ExecutionEvent,
) -> FailureDiagnosis:
    """Produce the full FailureDiagnosis object (schema-conformant,
    Section 5) for an already-detected failure.

    Note on root-cause step: the frozen FailureDiagnosis schema has
    `root_cause_agent` and `description` but no separate `root_cause_step`
    field (flagged as a gap during Phase 1 review -- confirm with P1
    whether the schema should gain one). Until then, step-level detail is
    embedded in `description`, and the exact step is always recoverable
    via `reliability_score.step_id`.
    """
    failure_type = classify_failure(score, graph, event)

    reason_by_type = {
        "F1": f"Semantic accuracy S={score.S:.3f} is the most-degraded "
        f"component and no upstream propagation was found; output likely "
        f"ungrounded relative to the validation store.",
        "F2": f"Execution rate E={score.E:.3f} is the most-degraded "
        f"component (event success={event.success}); likely a tool/API "
        f"invocation failure.",
        "F3": f"Consistency C={score.C:.3f} is the most-degraded "
        f"component; resampled trajectories for this task diverge, "
        f"indicating unstable reasoning.",
        "F4": f"An upstream step for task {event.task_id} failed and this "
        f"step (id={event.step_id}) failed downstream of it -- cascading "
        f"propagation.",
    }

    return FailureDiagnosis(
        task_id=event.task_id,
        step_id=event.step_id,
        failure_type=failure_type,
        root_cause_agent=event.agent_id,
        description=reason_by_type[failure_type],
        reliability_score=score,
    )
