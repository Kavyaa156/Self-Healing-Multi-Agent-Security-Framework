"""
Tests for reliability/reliability_score.py::compute_reliability

Covers weight validation, range validation, and the plan's sample
input/output example (Section 4, Person 3 spec):
    ReliabilityScore(C=0.82, S=0.71, E=0.80, R=0.78, weights=(0.4,0.4,0.2))
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reliability.reliability_score import compute_reliability, DEFAULT_WEIGHTS


def test_matches_plan_sample_example():
    # Section 4 sample inputs: C=0.82, S=0.71, E=0.80, weights=(0.4,0.4,0.2).
    # NOTE: the plan doc's sample output states R=0.78, but the correct
    # weighted sum is 0.4*0.82 + 0.4*0.71 + 0.2*0.80 = 0.772 -- the doc's
    # 0.78 appears to be a rounded/illustrative figure, not an exact worked
    # example. We assert against the mathematically correct value.
    score = compute_reliability(
        C=0.82, S=0.71, E=0.80,
        weights=(0.4, 0.4, 0.2),
        task_id="t1", step_id=1,
    )
    assert abs(score.R - 0.772) < 1e-9, f"Expected R=0.772, got {score.R}"
    assert score.C == 0.82 and score.S == 0.71 and score.E == 0.80


def test_perfect_scores_give_r_one():
    score = compute_reliability(
        C=1.0, S=1.0, E=1.0,
        weights=DEFAULT_WEIGHTS["multi_step_reasoning"],
        task_id="t2", step_id=0,
    )
    assert abs(score.R - 1.0) < 1e-9, f"Expected R=1.0, got {score.R}"


def test_zero_scores_give_r_zero():
    score = compute_reliability(
        C=0.0, S=0.0, E=0.0,
        weights=DEFAULT_WEIGHTS["api_orchestration"],
        task_id="t3", step_id=0,
    )
    assert score.R == 0.0


def test_out_of_range_component_raises():
    try:
        compute_reliability(C=1.5, S=0.5, E=0.5, weights=(0.4, 0.4, 0.2), task_id="t", step_id=0)
        assert False, "Expected ValueError for C > 1.0"
    except ValueError:
        pass


def test_weights_not_summing_to_one_raises():
    try:
        compute_reliability(C=0.5, S=0.5, E=0.5, weights=(0.5, 0.5, 0.5), task_id="t", step_id=0)
        assert False, "Expected ValueError for weights summing to 1.5"
    except ValueError:
        pass


def test_wrong_number_of_weights_raises():
    try:
        compute_reliability(C=0.5, S=0.5, E=0.5, weights=(0.5, 0.5), task_id="t", step_id=0)
        assert False, "Expected ValueError for only 2 weights"
    except ValueError:
        pass


def test_default_threshold_is_jeongs_value():
    score = compute_reliability(C=0.5, S=0.5, E=0.5, weights=(0.4, 0.4, 0.2), task_id="t", step_id=0)
    assert score.threshold == 0.65


if __name__ == "__main__":
    tests = [
        test_matches_plan_sample_example,
        test_perfect_scores_give_r_one,
        test_zero_scores_give_r_zero,
        test_out_of_range_component_raises,
        test_weights_not_summing_to_one_raises,
        test_wrong_number_of_weights_raises,
        test_default_threshold_is_jeongs_value,
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
