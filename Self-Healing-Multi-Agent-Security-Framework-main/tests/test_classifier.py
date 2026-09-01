from diagnosis.classifier import build_diagnosis, classify_failure
from monitoring.execution_graph import ExecutionGraph
from schemas.events import ExecutionEvent, ReliabilityScore


def make_event(task_id="t1", step_id=1, agent_id="tool_agent", success=True):
    return ExecutionEvent(
        task_id=task_id, agent_id=agent_id, step_id=step_id,
        event_type="tool_call", content="x", success=success, timestamp=0.0,
    )


def make_score(task_id="t1", step_id=1, C=0.9, S=0.9, E=0.9):
    R = 0.4 * C + 0.4 * S + 0.2 * E
    return ReliabilityScore(
        task_id=task_id, step_id=step_id, C=C, S=S, E=E, R=R,
        weights=(0.4, 0.4, 0.2), threshold=0.65,
    )


def test_low_s_classified_as_f1():
    graph = ExecutionGraph()
    event = make_event()
    graph.add_event(event)
    score = make_score(S=0.1, C=0.8, E=0.8)
    assert classify_failure(score, graph, event) == "F1"


def test_low_e_classified_as_f2():
    graph = ExecutionGraph()
    event = make_event(success=False)
    graph.add_event(event)
    score = make_score(S=0.8, C=0.8, E=0.1)
    assert classify_failure(score, graph, event) == "F2"


def test_low_c_classified_as_f3():
    graph = ExecutionGraph()
    event = make_event()
    graph.add_event(event)
    score = make_score(S=0.8, C=0.1, E=0.8)
    assert classify_failure(score, graph, event) == "F3"


def test_upstream_failure_overrides_to_f4():
    graph = ExecutionGraph()
    upstream = make_event(step_id=1, success=False)
    downstream = make_event(step_id=2, success=True)
    graph.add_event(upstream)
    graph.add_event(downstream)
    # Even though downstream's own C/S/E look like an F3 case, F4 must win.
    score = make_score(step_id=2, S=0.8, C=0.1, E=0.8)
    assert classify_failure(score, graph, downstream) == "F4"


def test_build_diagnosis_matches_schema_and_classification():
    graph = ExecutionGraph()
    event = make_event()
    graph.add_event(event)
    score = make_score(S=0.1, C=0.8, E=0.8)
    diagnosis = build_diagnosis(score, graph, event)
    assert diagnosis.failure_type == "F1"
    assert diagnosis.task_id == event.task_id
    assert diagnosis.step_id == event.step_id
    assert diagnosis.root_cause_agent == event.agent_id
    assert diagnosis.reliability_score == score