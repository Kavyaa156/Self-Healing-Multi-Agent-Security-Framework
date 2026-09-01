"""
main.py -- Integration entry point (Step 7 / Section 7).

Wires the full pipeline end-to-end:

    P1  MultiAgentWorkflow.run_task()          -> RawEvents
    P2  TelemetryCollector.process_events()    -> ExecutionEvents + ExecutionGraph
    P3  compute_consistency / compute_semantic_accuracy /
        compute_execution_rate / compute_reliability -> ReliabilityScore
    P4  detect_failure() + build_diagnosis()   -> FailureDiagnosis

Two demo scenarios, per the plan:
    1. Clean run          -- no fault, should NOT be flagged.
    2. Forced-failure run -- an F2 (execution error) fault is injected,
       showing the full detect -> diagnose loop end-to-end.

Run with:  python main.py
"""

import json
import os
import uuid
from typing import List

from schemas.events import TaskSpec, RawEvent
from agents.workflow import MultiAgentWorkflow
from tests.fault_injection import run_task_with_fault

# S (semantic accuracy) needs a seeded ValidationStore (Chroma +
# sentence-transformers). This downloads an embedding model from
# Hugging Face on first run, so it needs real internet access once.
# If chromadb/sentence-transformers aren't installed yet, or you're
# offline, main() falls back to running WITHOUT a validation store
# (S is stubbed to 1.0) so the rest of the pipeline still runs.
try:
    from reliability.seed_validation_store import seed_store
    HAS_VALIDATION_STORE_DEPS = True
except ImportError:
    HAS_VALIDATION_STORE_DEPS = False


def save_events_to_jsonl(events: List[RawEvent], filepath: str = "telemetry_events.jsonl"):
    """Writes raw telemetry events to a JSONL file for debugging/inspection."""
    with open(filepath, "a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event.model_dump()) + "\n")
    print(f"Saved {len(events)} events to {filepath}")


def print_result(label: str, result: dict):
    score = result["score"]
    print(f"\n--- {label} ---")
    print(f"  C={score.C:.3f}  S={score.S:.3f}  E={score.E:.3f}  R={score.R:.3f}  (theta={score.threshold})")
    print(f"  Detected as failure: {result['detected']}")
    if result["diagnosis"]:
        d = result["diagnosis"]
        print(f"  Diagnosis: {d.failure_type} | root cause agent: {d.root_cause_agent}")
        print(f"  Description: {d.description}")
    else:
        print("  Diagnosis: N/A (no failure detected)")


def main():
    print("==========================================================")
    print(" Self-Healing Multi-Agent Security Framework -- Integration ")
    print("==========================================================")

    workflow = MultiAgentWorkflow(model_name="openai/gpt-oss-20b", temperature=0.0)

    # Weights: api_orchestration weights (E weight=0.5) are used here
    # because this project's minimal pipeline puts most of its failure
    # signal on E (tool_call success/failure). See
    # reliability/reliability_score.py's DEFAULT_WEIGHTS and
    # tests/evaluate.py's run_evaluation() docstring for the full
    # rationale -- swap to whichever weight profile fits your domain.
    WEIGHTS = (0.2, 0.3, 0.5)
    THETA = 0.65

    validation_store = None
    if HAS_VALIDATION_STORE_DEPS:
        try:
            validation_store = seed_store()
        except Exception as e:
            print(f"[!] Could not build/seed validation store ({e}); "
                  f"continuing with S stubbed to 1.0.")
    else:
        print("[!] chromadb / sentence-transformers not installed; "
              "continuing with S stubbed to 1.0. Run "
              "`pip install chromadb sentence-transformers` for real S scoring.")

    # -----------------------------------------------------------------
    # Scenario 1: Clean run
    # -----------------------------------------------------------------
    clean_task = TaskSpec(
        task_id=f"task_clean_{uuid.uuid4().hex[:8]}",
        description="Audit access control policies for unauthorized privilege escalation risks",
        initial_input={
            "target_system": "Authentication Gateway",
            "log_sample_id": "LOG_2026_08_28",
        },
    )
    clean_result = run_task_with_fault(
        workflow=workflow, task=clean_task, fault_type=None,
        k=3, weights=WEIGHTS, theta=THETA, validation_store=validation_store,
    )
    save_events_to_jsonl([RawEvent(**e.model_dump()) for e in clean_result["events"]])
    print_result("SCENARIO 1: Clean run", clean_result)

    # -----------------------------------------------------------------
    # Scenario 2: Forced failure (F2 -- execution error)
    # -----------------------------------------------------------------
    faulty_task = TaskSpec(
        task_id=f"task_faulty_{uuid.uuid4().hex[:8]}",
        description="Audit access control policies for unauthorized privilege escalation risks",
        initial_input={
            "target_system": "Authentication Gateway",
            "log_sample_id": "LOG_2026_08_28",
        },
    )
    faulty_result = run_task_with_fault(
        workflow=workflow, task=faulty_task, fault_type="F2",
        k=3, weights=WEIGHTS, theta=THETA, validation_store=validation_store,
    )
    save_events_to_jsonl([RawEvent(**e.model_dump()) for e in faulty_result["events"]])
    print_result("SCENARIO 2: Forced failure (F2 injected)", faulty_result)

    print("\n==========================================================")
    print(" Integration run complete.")
    print("==========================================================")


if __name__ == "__main__":
    main()