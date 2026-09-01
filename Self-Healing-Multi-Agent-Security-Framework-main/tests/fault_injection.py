"""
tests/fault_injection.py

P4 Task 1 -- Fault Injection Harness.
[UPDATED for Phase 6 real integration -- see main.py / Section 12]

This file now does two things:

1. inject_fault(task, fault_type)
   Unchanged in spirit from the original design: returns a NEW TaskSpec
   annotated with a machine-readable fault directive (Section 9,
   scenarios 1-4). Still deterministic, still non-mutating.

2. run_task_with_fault(...)
   REPLACES the old simulate_execution() mock. Instead of inventing
   random ReliabilityScore numbers, this runs the task through the REAL
   pipeline:

       P1 workflow.run_task()      -> RawEvents
       (fault applied here, on the RawEvents themselves)
       P2 telemetry.process_events() -> ExecutionEvents + ExecutionGraph
       P3 compute_consistency / compute_semantic_accuracy /
          compute_execution_rate / compute_reliability -> ReliabilityScore
       P4 detect_failure() + build_diagnosis()         -> FailureDiagnosis

   Because we don't have a way to literally force a live LLM to
   hallucinate/timeout on command, faults are injected at the RawEvent
   level -- BEFORE normalization -- by mutating the exact fields Section 9
   says each fault type should affect:

       F1 (Hallucination)        -> overwrite the final_answer event's
                                     content with an ungrounded statement
                                     (tool-agent "answers without calling
                                     the tool" -> final output disagrees
                                     with the validation store)
       F2 (Execution Error)      -> mark the tool_call event
                                     success=False + tool_output={"error":...}
       F3 (Reasoning Inconsistency) -> perturb the K resampled trajectories
                                     (used for C) so they diverge, simulating
                                     a temperature/prompt perturbation
       F4 (Workflow Propagation)  -> same as F2, but the downstream
                                     final_answer event is ALSO marked
                                     success=False, so P4's real
                                     get_upstream_failed() finds a genuine
                                     failed ancestor in the graph

   This is a controlled, documented adaptation (same category of
   [PROJECT DESIGN DECISION] the plan already uses elsewhere) -- it is
   real fault injection into the real pipeline's data, not a canned
   random-number simulator.
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from schemas.events import ExecutionEvent, ReliabilityScore, TaskSpec, RawEvent
from agents.workflow import MultiAgentWorkflow
from monitoring.telemetry_collector import TelemetryCollector
from reliability.consistency import compute_consistency
from reliability.execution_rate import compute_execution_rate
from reliability.reliability_score import compute_reliability
from reliability.semantic_accuracy import compute_semantic_accuracy, ValidationStore
from detection.decision_controller import detect_failure
from diagnosis.classifier import build_diagnosis

VALID_FAULT_TYPES = {"F1", "F2", "F3", "F4"}

# Section 9's concrete injection method per fault type, reproduced for
# traceability from the plan doc.
FAULT_DESCRIPTIONS = {
    "F1": {
        "name": "Hallucination Error",
        "injected_where": "tool-using agent / final answer step",
        "how": "Force the tool-agent to answer without calling the tool, "
        "contradicting the validation DB.",
        "observable_symptom": "S score drops sharply.",
        "expected_label": "F1",
    },
    "F2": {
        "name": "Execution Error",
        "injected_where": "tool_call event",
        "how": "Mock a tool to raise a timeout/exception.",
        "observable_symptom": "success=False on the tool_call event; E score drops.",
        "expected_label": "F2",
    },
    "F3": {
        "name": "Reasoning Inconsistency",
        "injected_where": "K resampled trajectories of the same task",
        "how": "Add a temperature/prompt perturbation so K resampled runs diverge in plan.",
        "observable_symptom": "C score drops (trajectories disagree).",
        "expected_label": "F3",
    },
    "F4": {
        "name": "Workflow Propagation Error",
        "injected_where": "an early step, observed at a later dependent step",
        "how": "Inject an F2 fault at an early step; downstream dependent steps then also fail.",
        "observable_symptom": "Execution graph shows failure at a node whose upstream dependency also failed.",
        "expected_label": "F4",
    },
}


def inject_fault(task: TaskSpec, fault_type: str) -> TaskSpec:
    """Return a NEW TaskSpec annotated with a fault directive. Does not mutate `task`."""
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
# Real-pipeline fault application (operates on RawEvents, pre-normalization)
# ---------------------------------------------------------------------------
_UNGROUNDED_STATEMENT = (
    "The weather today is sunny with a light breeze. "
    "(Injected F1: this output ignores the actual task and validation domain.)"
)


def _apply_fault_to_raw_events(raw_events: list[RawEvent], fault_type: Optional[str]) -> list[RawEvent]:
    """Return a NEW list of RawEvents with the requested fault applied. No-op if fault_type is None."""
    events = copy.deepcopy(raw_events)
    if fault_type is None:
        return events

    if fault_type == "F1":
        for evt in events:
            if evt.event_type == "final_answer":
                evt.content = _UNGROUNDED_STATEMENT

    elif fault_type in ("F2", "F4"):
        for evt in events:
            if evt.event_type == "tool_call":
                evt.success = False
                evt.tool_output = {**(evt.tool_output or {}), "error": "timeout"}
        if fault_type == "F4":
            # Propagation: the downstream final_answer step also fails,
            # so the graph shows a genuine failed ancestor (tool_call)
            # feeding a failed descendant (final_answer).
            for evt in events:
                if evt.event_type == "final_answer":
                    evt.success = False

    # F3 is NOT applied here -- it affects the K-trajectory set, not a
    # single run's raw events. See _maybe_perturb_trajectories() below.
    return events


def _maybe_perturb_trajectories(
    trajectories: list[list[RawEvent]], fault_type: Optional[str], rng: random.Random
) -> list[list[RawEvent]]:
    """
    For F3, artificially diverge the K trajectories' step content so
    compute_consistency() (real rapidfuzz edit-distance) sees genuine
    disagreement -- simulating what a temperature/prompt perturbation
    would produce on a live LLM.

    NOTE: a simple word-shuffle is NOT divergent enough -- rapidfuzz's
    normalized Levenshtein distance still scores shuffled-but-identical
    vocabulary as fairly similar (same character set, similar length),
    so C only drops to ~0.4-0.5, not the "runs disagree" level Section 9
    describes. Instead, each trajectory's content is replaced with an
    independently generated random string per trajectory -- this is what
    genuinely divergent reasoning across K resampled runs looks like at
    the character level, and reliably drives C toward 0.

    For all other fault types (or None), trajectories are returned
    unmodified -- a clean/other-fault task should stay internally
    consistent across resamples.
    """
    if fault_type != "F3":
        return trajectories

    import string

    perturbed = copy.deepcopy(trajectories)
    for traj in perturbed:
        for evt in traj:
            random_text = "".join(
                rng.choices(string.ascii_lowercase + " ", k=max(len(evt.content), 40))
            )
            evt.content = random_text
    return perturbed


def run_task_with_fault(
    workflow: MultiAgentWorkflow,
    task: TaskSpec,
    fault_type: Optional[str] = None,
    k: int = 3,
    weights: tuple[float, float, float] = (0.4, 0.4, 0.2),
    theta: float = 0.65,
    validation_store: Optional[ValidationStore] = None,
    seed: Optional[int] = None,
) -> dict:
    """
    Run ONE task end-to-end through the REAL pipeline, optionally
    fault-injected, and return everything P4/evaluate.py needs.

    Args:
        workflow: a real MultiAgentWorkflow instance (P1).
        task: the TaskSpec to run (clean, i.e. NOT yet passed through
            inject_fault -- this function calls inject_fault internally
            if fault_type is given, so the fault directive is also
            visible on task.initial_input for logging/debugging).
        fault_type: None for a clean run, or one of "F1".."F4".
        k: number of resampled trajectories for C (Jeong recommends 3-5).
        weights: (w1, w2, w3) for R = w1*C + w2*S + w3*E.
        theta: detection threshold passed to detect_failure().
        validation_store: P3's seeded ValidationStore for S. If None, S is
            skipped (set to 1.0) with a warning -- pass a real store
            (see reliability/seed_validation_store.py) for a real S score.
        seed: RNG seed for F3's trajectory perturbation (reproducibility).

    Returns:
        {
          "task": TaskSpec (fault-annotated if fault_type given),
          "events": list[ExecutionEvent]   (P2 real output),
          "telemetry": TelemetryCollector  (holds the real ExecutionGraph),
          "score": ReliabilityScore        (P3 real output),
          "detected": bool                 (P4 real output),
          "diagnosis": FailureDiagnosis | None,
        }
    """
    rng = random.Random(seed if seed is not None else hash(task.task_id) & 0xFFFF)

    injected_task = inject_fault(task, fault_type) if fault_type else task

    # --- P1: real single run (for E / detection / diagnosis) ---
    raw_events = workflow.run_task(injected_task)
    raw_events = _apply_fault_to_raw_events(raw_events, fault_type)

    # --- P2: real normalize + real ExecutionGraph ---
    telemetry = TelemetryCollector()
    events = telemetry.process_events(raw_events)

    final_event = next((e for e in reversed(events) if e.event_type == "final_answer"), events[-1])
    tool_events = [e for e in events if e.event_type == "tool_call"]

    # Which event to anchor diagnosis on: the actual failing event, not
    # always the final step. This matters for F4 vs F2 -- in this
    # pipeline's linear step order, EVERY later step is topologically
    # "downstream" of an earlier one, so anchoring on final_answer for a
    # plain F2 (only the tool_call itself failed) would make
    # get_upstream_failed() true for the wrong reason (structural order,
    # not real propagation) and misclassify F2 as F4. Anchoring on the
    # LAST event that actually failed (success=False) fixes this: a lone
    # F2 has only one failed event (the tool_call itself), so it anchors
    # there (no failed ancestor before it -> correctly falls through to
    # component-based classification). F4 has two failed events (tool_call
    # AND the downstream final_answer); anchoring on the LAST one
    # (final_answer) is what actually demonstrates propagation, since it
    # genuinely has a failed ancestor (the tool_call) -> correctly
    # classified F4.
    failed_events = [e for e in events if not e.success]
    diagnosis_event = failed_events[-1] if failed_events else final_event

    # --- P3a: E (real, from real success flags) ---
    E = compute_execution_rate(tool_events if tool_events else events)

    # --- P3b: C (real rapidfuzz, over real K trajectories) ---
    raw_trajectories = workflow.run_task_k_times(injected_task, k=k, temperature=0.7)
    raw_trajectories = _maybe_perturb_trajectories(raw_trajectories, fault_type, rng)
    normalized_trajectories = [
        [telemetry_scratch.process_event(r) for r in traj]
        for traj, telemetry_scratch in ((t, TelemetryCollector()) for t in raw_trajectories)
    ]
    C = compute_consistency(normalized_trajectories, K=k)

    # --- P3c: S (real Chroma + sentence-transformers, if a store is given) ---
    if validation_store is not None:
        S = compute_semantic_accuracy(final_event, validation_store)
    else:
        S = 1.0  # no validation store wired in -- see docstring

    score = compute_reliability(
        C=C, S=S, E=E, weights=weights,
        task_id=injected_task.task_id, step_id=final_event.step_id,
        threshold=theta,
    )

    # --- P4: real detection + diagnosis ---
    graph_flags = telemetry.get_flags(injected_task.task_id, agent_id="tool_agent")
    detected = detect_failure(score, graph_flags, theta=theta)

    diagnosis = None
    if detected:
        diagnosis = build_diagnosis(score, telemetry.graph, diagnosis_event)

    return {
        "task": injected_task,
        "events": events,
        "telemetry": telemetry,
        "score": score,
        "detected": detected,
        "diagnosis": diagnosis,
    }