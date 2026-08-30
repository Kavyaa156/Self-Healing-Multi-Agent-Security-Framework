import pytest

from detection.decision_controller import detect_failure
from schemas_mock.events import ReliabilityScore


def score(R, task_id="t1", step_id=1, weights=(0.4, 0.4, 0.2), threshold=0.65):
    return ReliabilityScore(
        task_id=task_id, step_id=step_id, C=R, S=R, E=R, R=R,
        weights=weights, threshold=threshold,
    )


def test_r_above_theta_no_pattern_is_not_a_failure():
    # Test case 1 from the plan's spec for P4 (Section 4).
    s = score(0.8)
    assert detect_failure(s, {"repeated_failure": False, "abnormal_sequence": False}) is False


def test_r_below_theta_is_a_failure():
    s = score(0.5)
    assert detect_failure(s, {"repeated_failure": False, "abnormal_sequence": False}) is True


def test_repeated_failure_flag_triggers_even_if_r_above_theta():
    # Section 9, scenario 5: detected via pattern check even if R stays above theta.
    s = score(0.9)
    assert detect_failure(s, {"repeated_failure": True, "abnormal_sequence": False}) is True


def test_abnormal_sequence_flag_triggers_even_if_r_above_theta():
    # Section 9, scenario 6.
    s = score(0.9)
    assert detect_failure(s, {"repeated_failure": False, "abnormal_sequence": True}) is True


def test_missing_flag_keys_default_to_false():
    s = score(0.9)
    assert detect_failure(s, {}) is False


def test_theta_is_configurable():
    s = score(0.7)
    assert detect_failure(s, {}, theta=0.65) is False
    assert detect_failure(s, {}, theta=0.75) is True


def test_non_dict_graph_flags_raises_typeerror():
    s = score(0.9)
    with pytest.raises(TypeError):
        detect_failure(s, graph_flags=["not", "a", "dict"])  # type: ignore[arg-type]
