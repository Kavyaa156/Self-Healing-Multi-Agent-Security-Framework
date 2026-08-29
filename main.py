import json
import os
import uuid
from typing import List
from schemas.events import TaskSpec, RawEvent
from agents.workflow import MultiAgentWorkflow


def save_events_to_jsonl(events: List[RawEvent], filepath: str = "telemetry_events.jsonl"):
    """
    Writes raw telemetry events to a JSONL file.
    Person 2, 3, and 4 can read this file or consume the raw dicts directly.
    """
    with open(filepath, "a", encoding="utf-8") as f:
        for event in events:
            f.write(json.dumps(event.model_dump()) + "\n")
    print(f"Saved {len(events)} events to {filepath}")


def main():
    print("==========================================================")
    print(" Self-Healing Multi-Agent Security Framework (Person 1)  ")
    print("==========================================================")

    # Initialize Person 1 Multi-Agent Workflow
    workflow = MultiAgentWorkflow(model_name="openai/gpt-oss-20b", temperature=0.0)

    # Define a sample security auditing task.
    # task_id gets a unique suffix per run so telemetry_events.jsonl can
    # safely accumulate multiple runs without P2's ExecutionGraph mistaking
    # "two separate runs" for "one task looping back to step 1" (abnormal_sequence).
    sample_task = TaskSpec(
        task_id=f"task_sec_{uuid.uuid4().hex[:8]}",
        description="Audit access control policies for unauthorized privilege escalation risks",
        initial_input={
            "target_system": "Authentication Gateway",
            "log_sample_id": "LOG_2026_08_28"
        }
    )

    # 1. Single Execution Run
    print("\n[+] Running single workflow execution...")
    events = workflow.run_task(sample_task)
    
    print("\n--- Event Stream Summary ---")
    for evt in events:
        print(f"  Step {evt.step_id} | Agent: {evt.agent_id:10s} | Type: {evt.event_type:10s} | Content Snippet: {evt.content[:60]}...")

    # Save telemetry output for downstream agents
    save_events_to_jsonl(events, "telemetry_events.jsonl")

    # 2. Trajectory Sampling Run (K=3) for Person 3 Consistency Calculation
    print("\n[+] Running trajectory sampling (K=3) for Person 3 reliability calculation...")
    trajectories = workflow.run_task_k_times(sample_task, k=3, temperature=0.7)
    
    print(f"--- Sampled {len(trajectories)} trajectories successfully ---")
    for idx, traj in enumerate(trajectories, start=1):
        print(f"  Trajectory {idx}: {len(traj)} steps emitted.")

    print("\n==========================================================")
    print(" Person 1 Execution Completed Successfully!               ")
    print("==========================================================")


if __name__ == "__main__":
    main()
