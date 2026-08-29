"""
Person 3 — Reliability Evaluation Module
reliability_score.py — combines C, S, E into the final Reliability (R)
score, per Jeong's formula:

    R = w1*C + w2*S + w3*E

Weights are configurable per task type. The plan documents Jeong's own
reported example weights as a starting point (not a guess) for three
task-type categories — these are exposed here as DEFAULT_WEIGHTS so P4's
decision controller (or whoever wires this up) doesn't have to hardcode
them elsewhere.
"""

from schemas.events import ReliabilityScore

# Jeong's reported example weights per task type (Section 4, Person 3
# spec) — documented starting points, not tuned for this project's
# specific domain. (w1=C weight, w2=S weight, w3=E weight)
DEFAULT_WEIGHTS = {
    "multi_step_reasoning": (0.4, 0.4, 0.2),
    "api_orchestration": (0.2, 0.3, 0.5),
    "document_processing": (0.3, 0.4, 0.3),
}


def compute_reliability(
    C: float,
    S: float,
    E: float,
    weights: tuple[float, float, float],
    task_id: str,
    step_id: int,
    threshold: float = 0.65,
) -> ReliabilityScore:
    """
    Combine C, S, E into the final Reliability (R) score.

    Args:
        C: Consistency score, in [0, 1] (from compute_consistency).
        S: Semantic Accuracy score, in [0, 1] (from compute_semantic_accuracy).
        E: Execution Rate score, in [0, 1] (from compute_execution_rate).
        weights: (w1, w2, w3) tuple weighting C, S, E respectively. See
            DEFAULT_WEIGHTS for Jeong's documented starting points by
            task type — pick whichever is closest to this project's
            domain, or supply tuned weights once available.
        task_id: the task this score belongs to (carried into the
            output ReliabilityScore for P4's downstream use).
        step_id: the step this score belongs to.
        threshold: theta, the failure-detection cutoff. Defaults to
            Jeong's own calibrated value of 0.65 (the plan notes this
            needs domain-specific recalibration as a follow-up
            experiment — P4 owns that retuning, not this module).

    Returns:
        ReliabilityScore — the shared Pydantic model, with R computed
        as the weighted sum of C, S, E.

    Raises:
        ValueError: if any of C, S, E are outside [0, 1], if weights
            doesn't have exactly 3 values, or if the weights don't sum
            to (approximately) 1.0.
    """
    for name, value in (("C", C), ("S", S), ("E", E)):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}.")

    if len(weights) != 3:
        raise ValueError(f"weights must have exactly 3 values (w1, w2, w3), got {weights}.")

    w1, w2, w3 = weights
    weight_sum = w1 + w2 + w3
    if abs(weight_sum - 1.0) > 1e-6:
        raise ValueError(
            f"weights must sum to 1.0 (Jeong's R is a weighted average), "
            f"got {weights} summing to {weight_sum}."
        )

    R = w1 * C + w2 * S + w3 * E

    return ReliabilityScore(
        task_id=task_id,
        step_id=step_id,
        C=C,
        S=S,
        E=E,
        R=R,
        weights=weights,
        threshold=threshold,
    )
