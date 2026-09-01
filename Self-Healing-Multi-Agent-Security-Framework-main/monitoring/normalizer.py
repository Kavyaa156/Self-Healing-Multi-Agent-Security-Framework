from typing import List
from schemas.events import RawEvent, ExecutionEvent


def normalize_event(raw: RawEvent) -> ExecutionEvent:
    """
    Convert a single RawEvent (emitted by P1's LangGraph pipeline) into the
    common ExecutionEvent format shared with P3 (Reliability) and P4
    (Detection & Diagnosis).
    """
    success = raw.success

    # Defensive check: if a tool_call's output dict contains an "error" key,
    # treat it as a failure even if P1 didn't explicitly set success=False.
    if raw.event_type == "tool_call" and isinstance(raw.tool_output, dict):
        if "error" in raw.tool_output:
            success = False

    return ExecutionEvent(
        task_id=raw.task_id,
        agent_id=raw.agent_id,
        step_id=raw.step_id,
        event_type=raw.event_type,
        content=raw.content,
        tool_name=raw.tool_name,
        tool_input=raw.tool_input,
        tool_output=raw.tool_output,
        success=success,
        timestamp=raw.timestamp,
    )


def normalize_events(raw_events: List[RawEvent]) -> List[ExecutionEvent]:
    """Batch version of normalize_event — normalizes a whole trajectory at once."""
    return [normalize_event(raw) for raw in raw_events]