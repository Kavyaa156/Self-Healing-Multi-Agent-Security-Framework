"""
detection/decision_controller.py

P4 Task 2 -- Failure Detection.

Directly implements the hybrid detection concept from Jeong §3.4
(execution pattern analysis + threshold-based detection), operationalized
by the project plan as:

    is_failure = (R < theta) OR pattern_anomaly_flag

- The `R < theta` half is Jeong's own threshold mechanism (§3.4), direct.
- The `pattern_anomaly_flag` half corresponds to Jeong's "execution
  pattern analysis" (repeated tool failures, abnormal execution
  sequences), but the *exact* two checks and their trigger conditions are
  P2's/the project's own implementation (Jeong names the category, not
  the checks).
- theta=0.65 is Jeong's own reported value (§4.1), explicitly a
  recalibratable starting point per Jeong's own threats-to-validity
  discussion -- kept configurable here, not hardcoded as truth.
"""

from __future__ import annotations

from schemas.events import ReliabilityScore


def detect_failure(
    score: ReliabilityScore,
    graph_flags: dict[str, bool],
    theta: float = 0.65,
) -> bool:
    """Hybrid failure decision: threshold OR pattern anomaly.

    Args:
        score: P3's ReliabilityScore for this task/step. `score.R` is used;
            `score.threshold` is NOT silently substituted for `theta` --
            the caller's explicit `theta` argument wins, since the plan
            treats theta as a project-level configurable, not something
            baked permanently into the score object.
        graph_flags: dict with (at least) the keys "repeated_failure" and
            "abnormal_sequence", as produced by
            mocks.execution_graph.ExecutionGraph.get_flags() (or the real
            P2 module's equivalent adapter). Missing keys default to False
            rather than raising, so a partially-populated flags dict
            (e.g. from an older P2 version) fails safe instead of crashing
            the detector.
        theta: reliability threshold. Defaults to Jeong's reported 0.65
            (§4.1) but is explicitly configurable per the plan's
            instruction not to treat 0.65 as immutable.

    Returns:
        True if a failure is detected, False otherwise.

    Raises:
        TypeError: if graph_flags is not a dict-like mapping.
    """
    if not hasattr(graph_flags, "get"):
        raise TypeError(
            f"graph_flags must be a dict-like mapping with .get(), got {type(graph_flags)!r}. "
            "If you have a real ExecutionGraph object, call its .get_flags(...) "
            "adapter first (see mocks/execution_graph.py)."
        )

    threshold_breach = score.R < theta
    pattern_anomaly = bool(graph_flags.get("repeated_failure", False)) or bool(
        graph_flags.get("abnormal_sequence", False)
    )

    return threshold_breach or pattern_anomaly
