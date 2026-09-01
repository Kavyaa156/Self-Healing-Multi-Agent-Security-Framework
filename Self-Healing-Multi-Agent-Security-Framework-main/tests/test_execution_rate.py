import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.events import ExecutionEvent
from reliability.execution_rate import compute_execution_rate


def make_event(success: bool, step_id: int) -> ExecutionEvent:
    return ExecutionEvent(
        task_id="t1", agent_id="tool_agent", step_id=step_id,
        event_type="tool_call", content="search(...)", success=success,
    )


def test_all_success():
    events = [make_event(True, i) for i in range(10)]
    assert compute_execution_rate(events) == 1.0


def test_all_fail():
    events = [make_event(False, i) for i in range(10)]
    assert compute_execution_rate(events) == 0.0


def test_mixed_8_of_10():
    events = [make_event(True, i) for i in range(8)] + [make_event(False, i) for i in range(8, 10)]
    assert compute_execution_rate(events) == 0.8


def test_empty_window():
    assert compute_execution_rate([]) == 1.0


if __name__ == "__main__":
    test_all_success()
    test_all_fail()
    test_mixed_8_of_10()
    test_empty_window()
    print("All execution_rate tests passed.")
