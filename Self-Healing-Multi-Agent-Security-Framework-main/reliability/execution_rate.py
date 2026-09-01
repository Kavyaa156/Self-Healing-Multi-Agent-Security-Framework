"""
E = successful tool/agent calls / total calls, over a window of ExecutionEvents.
Per Section 4 (Person 3): "Compute E as successful/total tool calls from
ExecutionEvents."
"""

from schemas.events import ExecutionEvent


def compute_execution_rate(events: list[ExecutionEvent]) -> float:
    """
    Purpose: E = successful calls / total calls in the window.

    Input:  list of ExecutionEvent (typically all events for one task, or a
            sliding window if you're scoring per-step rather than per-task).
    Output: float in [0, 1]. Returns 1.0 for an empty window (no evidence of
            failure is treated as fully reliable, matching E's "success ratio"
            definition — an empty denominator has no fair E<1.0 to assign).
    """
    if not events:
        return 1.0

    successes = sum(1 for e in events if e.success)
    return successes / len(events)
