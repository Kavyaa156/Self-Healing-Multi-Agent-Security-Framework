"""
Person 3 — Reliability Evaluation Module
consistency.py — implements Jeong et al.'s Consistency (C) score.

Formula (Jeong §3.3):
    C = 1 - (2 / (T*K*(K-1))) * sum_t sum_{i<j} d_norm(a_t^(i), a_t^(j))

Where:
    T = number of steps in the task
    K = number of independently resampled trajectories of the same task
    a_t^(i) = the "content" produced at step t in trajectory i
    d_norm  = normalized Levenshtein distance (rapidfuzz), in [0, 1]

Intuition: for each step t, compare that step's output across every pair
of K resampled runs. If the K runs agree closely at every step, the
average normalized distance is near 0 and C is near 1 (highly
consistent/deterministic reasoning). If runs diverge wildly, C -> 0.
"""

from itertools import combinations
from rapidfuzz.distance import Levenshtein

from schemas.events import ExecutionEvent


def compute_consistency(trajectories: list[list[ExecutionEvent]], K: int) -> float:
    """
    Compute the Consistency (C) score across K resampled trajectories of
    the same task.

    Args:
        trajectories: list of K trajectories, each trajectory being an
            ordered list of ExecutionEvent objects (one per step, aligned
            by step_id across trajectories — i.e. trajectories[i][t] and
            trajectories[j][t] both refer to step t).
        K: number of trajectories. Must equal len(trajectories) and be >= 2
            (Jeong's formula divides by K*(K-1), which is undefined for K<2).

    Returns:
        float in [0, 1] — 1.0 means perfectly consistent across all K runs
        at every step; 0.0 means maximally divergent.

    Raises:
        ValueError: if K < 2, trajectories is empty, K doesn't match the
            number of trajectories given, or trajectories have mismatched
            lengths (unequal T across runs).
    """
    if K < 2:
        raise ValueError(
            f"compute_consistency requires K >= 2 resampled trajectories "
            f"(Jeong's formula divides by K*(K-1)); got K={K}."
        )
    if len(trajectories) != K:
        raise ValueError(
            f"K={K} does not match len(trajectories)={len(trajectories)}."
        )
    if K == 0 or len(trajectories[0]) == 0:
        raise ValueError("trajectories must be non-empty.")

    T = len(trajectories[0])
    for idx, traj in enumerate(trajectories):
        if len(traj) != T:
            raise ValueError(
                f"All K trajectories must have the same number of steps T. "
                f"Trajectory 0 has {T} steps, trajectory {idx} has {len(traj)}."
            )

    total_distance = 0.0
    num_pairs = 0

    for t in range(T):
        # content produced at step t, one string per trajectory
        step_contents = [trajectories[i][t].content for i in range(K)]

        for i, j in combinations(range(K), 2):
            d_norm = Levenshtein.normalized_distance(
                step_contents[i], step_contents[j]
            )
            total_distance += d_norm
            num_pairs += 1

    # num_pairs should equal T * K*(K-1)/2 (Jeong's denominator, halved
    # because we only sum over i<j instead of double-counting i,j and j,i)
    avg_distance = total_distance / num_pairs

    C = 1.0 - avg_distance
    return C
