"""
diagnosis/attribution.py

Phase 5 Novelty #1 -- Trajectory-based failure attribution.

classify_failure() (existing) answers a binary question: is this F4
(propagation) or not. This module goes one step further: for F4, it
walks the ExecutionGraph's ancestors to find the EARLIEST failed node
-- the true root cause -- rather than assuming the immediate upstream
step is responsible. It also returns the full evidence chain, which
Phase 6 (self-healing) will need to decide WHICH agent to retry/replan,
not just that "something upstream broke".

For F1/F2/F3 (non-propagating failures), there's nothing to trace --
the failure is local to the diagnosed event itself, so attribution is
trivial (confidence=1.0, chain length 1).
"""

from __future__ import annotations

from schemas.events import ExecutionEvent, ReliabilityScore, FailureAttribution
from monitoring.execution_graph import ExecutionGraph


def attribute_root_cause(
    score: ReliabilityScore,
    graph: ExecutionGraph,
    event: ExecutionEvent,
    failure_type: str,
) -> FailureAttribution:
    """
    Args:
        score: P3's ReliabilityScore (kept in the signature for future
            extensions, e.g. weighting confidence by how far R fell
            below theta -- not used yet).
        graph: the real ExecutionGraph for this run.
        event: the ExecutionEvent classify_failure() was run against
            (the step where the failure was DETECTED, not necessarily
            where it ORIGINATED).
        failure_type: output of classify_failure() -- "F1".."F4".

    Returns:
        FailureAttribution with root_cause_agent/root_cause_step set to
        the TRUE origin (for F4) or the event itself (for F1/F2/F3).
    """
    if failure_type == "F4":
        chain = graph.get_propagation_chain(event.task_id, event.step_id)
        root = graph.get_root_cause_node(event.task_id, event.step_id)
        if root is not None:
            # Confidence decays slightly with longer chains -- a 2-node
            # chain (root -> target) is a clean, direct attribution; a
            # longer chain means more intermediate steps could also be
            # implicated, so we're less certain THIS specific node is
            # solely responsible.
            chain_len = len(chain)
            confidence = 1.0 if chain_len <= 2 else max(0.5, 1.0 - 0.1 * (chain_len - 2))
            return FailureAttribution(
                root_cause_agent=root["agent_id"],
                root_cause_step=root["step_id"],
                propagation_chain=chain,
                confidence=confidence,
            )
        # Safety net: classify_failure said F4 but no failed ancestor was
        # found (shouldn't happen given classify_failure's own logic, but
        # fail safe rather than raise).
        # Falls through to the local-attribution branch below.

    # F1 / F2 / F3 (or F4 safety-net fallback): failure is local to this
    # event -- no upstream node to trace, so it IS its own root cause.
    return FailureAttribution(
        root_cause_agent=event.agent_id,
        root_cause_step=event.step_id,
        propagation_chain=[{
            "node_id": None,
            "agent_id": event.agent_id,
            "step_id": event.step_id,
            "event_type": event.event_type,
            "success": event.success,
        }],
        confidence=1.0,
    )