"""
Tests for reliability/semantic_accuracy.py::compute_semantic_accuracy

These tests avoid downloading the real sentence-transformers model
(all-MiniLM-L6-v2 requires network access to huggingface.co, which may
not be available in CI/sandboxed environments). Instead they use a
FakeValidationStore that returns pre-set Chroma-style distances, so we
can test the distance-to-similarity math and error handling in
isolation.

A separate manual smoke test using the real embedding model lives at
the bottom of reliability/semantic_accuracy.py — run that directly
(`python3 reliability/semantic_accuracy.py`) on a machine with normal
internet access to verify the real embeddings end-to-end.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from schemas.events import ExecutionEvent
from reliability.semantic_accuracy import compute_semantic_accuracy


class FakeValidationStore:
    """Stands in for a real ValidationStore, returning fixed distances
    as if Chroma had already run the embedding + nearest-neighbor query."""

    def __init__(self, distances):
        self._distances = distances

    def query(self, text, top_k=3):
        return {"distances": [self._distances]}


def make_event(content="some event content"):
    return ExecutionEvent(
        task_id="t1",
        agent_id="executor",
        step_id=1,
        event_type="final_answer",
        content=content,
    )


def test_zero_distance_gives_s_one():
    store = FakeValidationStore([0.0, 0.0, 0.0])
    S = compute_semantic_accuracy(make_event(), store)
    assert S == 1.0, f"Expected S=1.0 for perfect match, got {S}"


def test_max_distance_gives_s_zero():
    store = FakeValidationStore([2.0, 2.0, 2.0])
    S = compute_semantic_accuracy(make_event(), store)
    assert S == 0.0, f"Expected S=0.0 for max L2 distance, got {S}"


def test_mid_distance_gives_s_half():
    store = FakeValidationStore([1.0, 1.0, 1.0])
    S = compute_semantic_accuracy(make_event(), store)
    assert abs(S - 0.5) < 1e-9, f"Expected S=0.5 for mid distance, got {S}"


def test_averages_across_top_k():
    # top_k=3 with mixed distances -> S should be the average similarity
    store = FakeValidationStore([0.0, 1.0, 2.0])  # similarities: 1.0, 0.5, 0.0
    S = compute_semantic_accuracy(make_event(), store)
    assert abs(S - 0.5) < 1e-9, f"Expected S=0.5 (average of 1.0, 0.5, 0.0), got {S}"


def test_distance_beyond_two_is_clamped():
    # shouldn't happen with normalized embeddings, but guard anyway
    store = FakeValidationStore([3.0])
    S = compute_semantic_accuracy(make_event(), store)
    assert S == 0.0, f"Expected S clamped to 0.0 for out-of-range distance, got {S}"


def test_empty_content_raises():
    store = FakeValidationStore([0.0])
    try:
        compute_semantic_accuracy(make_event(content="   "), store)
        assert False, "Expected ValueError for empty/whitespace content"
    except ValueError:
        pass


def test_no_matches_raises():
    store = FakeValidationStore([])
    try:
        compute_semantic_accuracy(make_event(), store)
        assert False, "Expected ValueError when validation store has no matches"
    except ValueError:
        pass


if __name__ == "__main__":
    tests = [
        test_zero_distance_gives_s_one,
        test_max_distance_gives_s_zero,
        test_mid_distance_gives_s_half,
        test_averages_across_top_k,
        test_distance_beyond_two_is_clamped,
        test_empty_content_raises,
        test_no_matches_raises,
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
