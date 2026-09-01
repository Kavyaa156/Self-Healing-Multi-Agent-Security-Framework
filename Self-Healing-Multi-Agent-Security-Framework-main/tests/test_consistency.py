"""
Tests for reliability/consistency.py::compute_consistency

Covers the test cases from Section 4 (Person 3 spec):
1. Identical K trajectories -> C approximately 1.0
2. Completely divergent trajectories -> C approximately 0.0
Plus edge-case / error handling tests.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.events import ExecutionEvent
from reliability.consistency import compute_consistency


def make_event(task_id, step_id, content, agent_id="planner"):
    return ExecutionEvent(
        task_id=task_id,
        agent_id=agent_id,
        step_id=step_id,
        event_type="reasoning",
        content=content,
    )


def test_identical_trajectories_gives_c_near_one():
    # Same 4-step task, 3 identical trajectories (K=3)
    steps = ["plan the search", "call search tool", "read result", "write answer"]
    trajectories = [
        [make_event("t1", t, steps[t]) for t in range(4)] for _ in range(3)
    ]
    C = compute_consistency(trajectories, K=3)
    assert abs(C - 1.0) < 1e-9, f"Expected C approx 1.0 for identical trajectories, got {C}"


def test_completely_divergent_trajectories_gives_c_near_zero():
    # Each trajectory uses a totally different string per step -> normalized
    # Levenshtein distance approx 1.0 for very dissimilar strings
    trajectories = [
        [make_event("t1", t, f"xxxxxxxxxx{i}") for t in range(3)]
        for i in range(3)
    ]
    trajectories = [
        [make_event("t1", t, s) for t, s in enumerate(traj_contents)]
        for traj_contents in [
            ["aaaaaaaaaa", "bbbbbbbbbb", "cccccccccc"],
            ["1111111111", "2222222222", "3333333333"],
            ["!!!!!!!!!!", "@@@@@@@@@@", "##########"],
        ]
    ]
    C = compute_consistency(trajectories, K=3)
    assert C < 0.15, f"Expected C near 0.0 for maximally divergent trajectories, got {C}"


def test_partial_divergence_is_between_zero_and_one():
    trajectories = [
        [make_event("t1", 0, "search the document")],
        [make_event("t1", 0, "search the documents")],  # minor variation
    ]
    C = compute_consistency(trajectories, K=2)
    assert 0.8 < C < 1.0, f"Expected high but not perfect C for near-identical text, got {C}"


def test_k_less_than_two_raises():
    trajectories = [[make_event("t1", 0, "only one trajectory")]]
    try:
        compute_consistency(trajectories, K=1)
        assert False, "Expected ValueError for K < 2"
    except ValueError:
        pass


def test_k_mismatch_raises():
    trajectories = [
        [make_event("t1", 0, "a")],
        [make_event("t1", 0, "b")],
    ]
    try:
        compute_consistency(trajectories, K=3)  # says 3 but only 2 given
        assert False, "Expected ValueError when K != len(trajectories)"
    except ValueError:
        pass


def test_mismatched_trajectory_lengths_raises():
    trajectories = [
        [make_event("t1", 0, "a"), make_event("t1", 1, "b")],
        [make_event("t1", 0, "a")],  # only 1 step, other has 2
    ]
    try:
        compute_consistency(trajectories, K=2)
        assert False, "Expected ValueError for mismatched trajectory lengths"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_identical_trajectories_gives_c_near_one,
        test_completely_divergent_trajectories_gives_c_near_zero,
        test_partial_divergence_is_between_zero_and_one,
        test_k_less_than_two_raises,
        test_k_mismatch_raises,
        test_mismatched_trajectory_lengths_raises,
    ]
    passed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {t.__name__} -- {e}")
    print(f"\n{passed}/{len(tests)} passed")
