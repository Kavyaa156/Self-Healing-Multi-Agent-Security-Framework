import json
from typing import List, Dict, Optional
from schemas.events import RawEvent, ExecutionEvent
from monitoring.normalizer import normalize_event
from monitoring.execution_graph import ExecutionGraph


class TelemetryCollector:
    """
    Ties normalization + graph-building into one pipeline:

        RawEvent (P1) --normalize_event()--> ExecutionEvent --> ExecutionGraph

    P3 and P4 use one TelemetryCollector instance as their entry point into
    P2's output, instead of calling normalize_event / ExecutionGraph directly.
    """

    def __init__(self):
        self.graph = ExecutionGraph()
        self.events: List[ExecutionEvent] = []

    def process_event(self, raw: RawEvent) -> ExecutionEvent:
        """Normalize one RawEvent, add it to the graph, and store it in the stream."""
        event = normalize_event(raw)
        self.graph.add_event(event)
        self.events.append(event)
        return event

    def process_events(self, raw_events: List[RawEvent]) -> List[ExecutionEvent]:
        """Batch version — normalizes and graphs a whole list of RawEvents."""
        return [self.process_event(r) for r in raw_events]

    def load_from_jsonl(self, filepath: str = "telemetry_events.jsonl") -> List[ExecutionEvent]:
        """
        Reads P1's saved raw event log (main.py writes to this file) and runs
        the full normalize + graph-build pipeline over it. Handy for testing
        P2 on its own, without re-running P1's LangGraph workflow each time.
        """
        raw_events = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    raw_events.append(RawEvent(**json.loads(line)))
        return self.process_events(raw_events)

    def get_events_for_task(self, task_id: str) -> List[ExecutionEvent]:
        """All normalized events for one task, in the order they were processed."""
        return [e for e in self.events if e.task_id == task_id]

    def get_flags(self, task_id: str, agent_id: Optional[str] = None) -> Dict[str, bool]:
        """Convenience pass-through to the graph's anomaly flags for P4."""
        return self.graph.get_flags(task_id, agent_id)