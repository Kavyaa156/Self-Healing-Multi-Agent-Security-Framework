"""
Person 3 — Reliability Evaluation Module
tests/test_consistency_integration.py — runs the REAL P1 pipeline's
run_task_k_times() and feeds the output directly into
compute_consistency(), instead of the synthetic trajectories used in
test_consistency.py.

Requires a real GROQ_API_KEY in .env (loaded via python-dotenv, same
as agents/workflow.py already does). Without it, MultiAgentWorkflow's
self.llm is None and every trajectory falls back to an identical
hardcoded string, which would trivially give C=1.0 every time — not a
real test.

Usage (run from the project root):
    python -m tests.test_consistency_integration
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from schemas.events import TaskSpec, ExecutionEvent
from agents.workflow import MultiAgentWorkflow
from reliability.consistency import compute_consistency


def raw_trajectories_to_execution_events(
    trajectories: list[list],
) -> list[list[ExecutionEvent]]:
    """
    P1's run_task_k_times() returns List[List[RawEvent]]. compute_consistency()
    expects List[List[ExecutionEvent]]. RawEvent and ExecutionEvent share the
    exact same fields (per schemas/events.py), so this just re-wraps each
    RawEvent's data into an ExecutionEvent -- standing in for the normalization
    step P2 would normally own.
    """
    converted = []
    for trajectory in trajectories:
        converted.append(
            [
                ExecutionEvent(**raw_event.model_dump())
                for raw_event in trajectory
            ]
        )
    return converted


def main():
    if not os.getenv("GROQ_API_KEY"):
        print(
            "WARNING: No GROQ_API_KEY found in environment. "
            "MultiAgentWorkflow will fall back to a hardcoded deterministic "
            "response for every run, so all K trajectories will be identical "
            "and C will trivially be 1.0. This test won't be meaningful "
            "until a real key is set in .env."
        )

    workflow = MultiAgentWorkflow()

    task = TaskSpec(
        task_id="consistency_check_1",
        description="Summarize the security posture of the Authentication "
        "Gateway based on recent audit findings and recommend top priorities.",
        initial_input={"log_sample": "LOG_2026_08_28"},
    )

    K = 3
    print(f"Running the real workflow {K} times at temperature=0.7 "
          f"via run_task_k_times()...\n")
    raw_trajectories = workflow.run_task_k_times(task, k=K, temperature=0.7)

    print(f"Got {len(raw_trajectories)} trajectories, "
          f"{len(raw_trajectories[0])} steps each.\n")

    for i, traj in enumerate(raw_trajectories):
        for event in traj:
            preview = event.content[:80].replace("\n", " ")
            print(f"  [traj {i}] step {event.step_id} ({event.agent_id}): {preview}...")
        print()

    execution_trajectories = raw_trajectories_to_execution_events(raw_trajectories)

    C = compute_consistency(execution_trajectories, K=K)
    print(f"=== Consistency (C) score across {K} real trajectories: {C:.4f} ===")

    if C > 0.999:
        print(
            "\nNote: C came back essentially 1.0 -- if GROQ_API_KEY isn't set "
            "or temperature sampling isn't producing variation, this likely "
            "means all K runs returned identical fallback text rather than "
            "genuinely diverse LLM outputs. Worth double-checking .env if "
            "this wasn't expected."
        )


if __name__ == "__main__":
    main()