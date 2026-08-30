import json
import time
from typing import List

# P1 Imports (Workflow & Task Schemas)
from schemas.events import TaskSpec, RawEvent, ExecutionEvent, ReliabilityScore, FailureDiagnosis
from agents.workflow import MultiAgentWorkflow

# P2 Import (Monitoring / Telemetry Normalizer)
from monitoring.normalizer import normalize_events

# P3 Imports (Reliability Metrics & Scoring)
from reliability.consistency import compute_consistency
from reliability.reliability_score import compute_reliability

# P4 Imports (Failure Diagnosis & Patching)
from diagnosis.classifier import classify_failure, build_diagnosis


class SelfHealingOrchestrator:
    def __init__(self, max_retries: int = 2, trajectory_k: int = 2, model_name: str = "openai/gpt-oss-120b"):
        self.workflow = MultiAgentWorkflow(model_name=model_name, temperature=0.0)
        self.max_retries = max_retries
        self.k = trajectory_k

    def run(self, task: TaskSpec):
        print("\n==========================================================")
        print("   STARTING INTEGRATED SELF-HEALING MULTI-AGENT PIPELINE   ")
        print("==========================================================")

        current_task = task

        for attempt in range(1, self.max_retries + 1):
            print(f"\n[🔄] --- ATTEMPT {attempt}/{self.max_retries} ---")

            if attempt > 1:
                print("[⏳] Pausing 3 seconds before retry attempt to regulate API rate limits...")
                time.sleep(3)

            # -----------------------------------------------------------------
            # STEP 1 (P1): Run Trajectories (K sampling for reliability)
            # -----------------------------------------------------------------
            print(f"[+] P1: Sampling {self.k} execution trajectories...")
            trajectories_raw: List[List[RawEvent]] = self.workflow.run_task_k_times(current_task, k=self.k)

            # -----------------------------------------------------------------
            # STEP 2 (P2): Normalize Raw Telemetry
            # -----------------------------------------------------------------
            print("[+] P2 Telemetry: Normalizing raw execution events across trajectories...")
            trajectories_normalized: List[List[ExecutionEvent]] = [
                normalize_events(raw_traj) for raw_traj in trajectories_raw
            ]

            # -----------------------------------------------------------------
            # STEP 3 (P3): Calculate Consistency & Reliability Score
            # -----------------------------------------------------------------
            print("[+] P3 Reliability: Calculating consistency metric (C)...")
            
            if self.k < 2:
                print("    [!] K < 2: Bypassing pairwise consistency calculation; defaulting C = 1.0")
                consistency_c = 1.0
            else:
                consistency_c = compute_consistency(trajectories_normalized, K=self.k)
            
            semantic_s = 1.0  
            execution_e = 1.0 
            weights = (0.4, 0.4, 0.2)

            reliability_score: ReliabilityScore = compute_reliability(
                C=consistency_c,
                S=semantic_s,
                E=execution_e,
                weights=weights,
                task_id=current_task.task_id,
                step_id=1,
                threshold=0.90
            )

            is_passed = reliability_score.R >= reliability_score.threshold

            print(f"    Consistency Score (C): {reliability_score.C:.2f}")
            print(f"    Overall Reliability (R): {reliability_score.R:.2f} | Passed: {is_passed}")

            if is_passed:
                print("\n==========================================================")
                print(" ✅ SUCCESS: Task passed reliability check!")
                print("==========================================================")
                return {
                    "status": "SUCCESS",
                    "attempts": attempt,
                    "reliability_score": reliability_score,
                    "events": trajectories_normalized[0]
                }

            # -----------------------------------------------------------------
            # STEP 4 (P4): Failure Diagnosis & Self-Healing Patching
            # -----------------------------------------------------------------
            print("\n[⚠️] Task failed reliability check. Triggering P4 Diagnosis...")
            
            last_event = trajectories_normalized[0][-1] if trajectories_normalized[0] else None
            
            diagnosis: FailureDiagnosis = build_diagnosis(
                score=reliability_score,
                graph=None,  
                event=last_event
            )

            print(f"    Root Cause Identified: {diagnosis.root_cause}")
            print(f"    Suggested Fix: {diagnosis.suggested_fix}")

            # Apply self-healing patch to input context for next retry iteration
            current_task.initial_input["healing_context"] = diagnosis.suggested_fix
            current_task.description += f"\n[Self-Healing Patch Attempt {attempt}]: {diagnosis.suggested_fix}"

        print("\n==========================================================")
        print(" ❌ FAILURE: Reached max retries without meeting reliability target.")
        print("==========================================================")
        return {"status": "FAILED", "attempts": self.max_retries}


def main():
    task = TaskSpec(
        task_id="integrated_sec_001",
        description="Audit system access control logs and detect privilege escalation vulnerabilities.",
        initial_input={"target_system": "Auth Gateway"}
    )

    orchestrator = SelfHealingOrchestrator(
        max_retries=2, 
        trajectory_k=2, 
        model_name="openai/gpt-oss-120b"
    )
    results = orchestrator.run(task)
    print("\nFinal Pipeline Result:", results["status"])


if __name__ == "__main__":
    main()