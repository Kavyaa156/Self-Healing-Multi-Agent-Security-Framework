"""
Tests for reliability/adaptive_weights.py::select_weights

Uses synthetic C/S/E combinations chosen to sit clearly inside each
band documented in adaptive_weights.py's module docstring (derived from
evaluation_report.json / evaluation_report_reasoning_weights.json),
plus the compute_reliability(weights=None) integration path.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reliability.adaptive_weights import (
    select_weights,
    E_HEAVY_WEIGHTS,
    S_HEAVY_WEIGHTS,
    C_HEAVY_WEIGHTS,
)
from reliability.reliability_score import compute_reliability


def _assert_weights_sum_to_one(weights):
    assert abs(sum(weights) - 1.0) < 1e-9, f"weights {weights} do not sum to 1.0"


# ---------------------------------------------------------------------------
# Branch 1: E < 1.0 (a tool_call actually failed) -> E-heavy, regardless
# of what C/S look like. Modeled on real F2/F4 rows (E=0.0).
# ---------------------------------------------------------------------------
def test_failed_tool_call_gives_e_heavy_weights():
    weights = select_weights(C=0.52, S=0.55, E=0.0, task_type="api_orchestration")
    assert weights == E_HEAVY_WEIGHTS
    _assert_weights_sum_to_one(weights)


def test_failed_tool_call_overrides_low_c_too():
    # Even if C also looks like an F3 signal, a genuine execution failure
    # (E < 1.0) is the least noisy signal available and takes priority.
    weights = select_weights(C=0.1, S=0.55, E=0.0)
    assert weights == E_HEAVY_WEIGHTS


# ---------------------------------------------------------------------------
# Branch 2: S far below the ~0.47-0.62 clean/other-fault band -> S-heavy.
# Modeled on real F1 rows (S~0.13, E=1.0, C~0.52-0.55).
# ---------------------------------------------------------------------------
def test_hallucination_signal_gives_s_heavy_weights():
    weights = select_weights(C=0.52, S=0.13, E=1.0)
    assert weights == S_HEAVY_WEIGHTS
    _assert_weights_sum_to_one(weights)


# ---------------------------------------------------------------------------
# Branch 3: C far below the ~0.49-0.56 clean/other-fault band -> C-heavy.
# Modeled on real F3 rows (C~0.11-0.16, S~0.49-0.62, E=1.0).
# ---------------------------------------------------------------------------
def test_reasoning_inconsistency_signal_gives_c_heavy_weights():
    weights = select_weights(C=0.12, S=0.6, E=1.0)
    assert weights == C_HEAVY_WEIGHTS
    _assert_weights_sum_to_one(weights)


# ---------------------------------------------------------------------------
# Branch 4: no strong single-metric signal (clean-run band) -> falls back
# to the E-heavy profile (Phase 4 showed this gives FPR=0.0 on clean runs).
# ---------------------------------------------------------------------------
def test_clean_run_falls_back_to_e_heavy_weights():
    weights = select_weights(C=0.52, S=0.55, E=1.0)
    assert weights == E_HEAVY_WEIGHTS


def test_out_of_range_component_raises():
    try:
        select_weights(C=1.5, S=0.5, E=0.5)
        assert False, "Expected ValueError for C > 1.0"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Integration: compute_reliability(weights=None) actually calls
# select_weights() and records the ACTUAL weights used on the returned
# ReliabilityScore (not a None/placeholder).
# ---------------------------------------------------------------------------
def test_compute_reliability_adaptive_mode_uses_selected_weights():
    score = compute_reliability(
        C=0.12, S=0.6, E=1.0,
        weights=None,
        task_id="t_adaptive", step_id=0,
    )
    assert score.weights == C_HEAVY_WEIGHTS
    expected_r = C_HEAVY_WEIGHTS[0] * 0.12 + C_HEAVY_WEIGHTS[1] * 0.6 + C_HEAVY_WEIGHTS[2] * 1.0
    assert abs(score.R - expected_r) < 1e-9


def test_compute_reliability_explicit_weights_still_works_unchanged():
    # Passing an explicit tuple must behave exactly as before -- adaptive
    # mode should never kick in unless weights=None.
    score = compute_reliability(
        C=0.12, S=0.6, E=1.0,
        weights=(0.4, 0.4, 0.2),
        task_id="t_explicit", step_id=0,
    )
    assert score.weights == (0.4, 0.4, 0.2)


if __name__ == "__main__":
    tests = [
        test_failed_tool_call_gives_e_heavy_weights,
        test_failed_tool_call_overrides_low_c_too,
        test_hallucination_signal_gives_s_heavy_weights,
        test_reasoning_inconsistency_signal_gives_c_heavy_weights,
        test_clean_run_falls_back_to_e_heavy_weights,
        test_out_of_range_component_raises,
        test_compute_reliability_adaptive_mode_uses_selected_weights,
        test_compute_reliability_explicit_weights_still_works_unchanged,
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