"""
Person 3/4 -- Phase 5, Novelty #2: Adaptive / Learned C/S/E Weighting.

Phase 4's limitations finding (see limitations.txt / the Phase 4 writeup)
was that NO single static (w1, w2, w3) weight tuple can simultaneously:
  - catch F3 (Reasoning Inconsistency), which needs a high C weight, AND
  - avoid false positives on clean runs, which needs a LOW C weight,
    because C has ~0.49-0.56 baseline noise even on clean/other-fault
    runs (it is not a clean 0/1 signal).

This module resolves that tradeoff WITHOUT a fixed tuple: it inspects the
already-computed C, S, E values for the CURRENT step and picks whichever
weight profile best fits the dominant failure signal actually present,
instead of committing to one profile for every task up front.

This is intentionally rule-based, not ML -- per the project's Phase 5
plan ("no ML model" instruction for this phase). All thresholds below
are calibrated directly from the two real evaluation reports already
generated in Phase 4:

  evaluation_report.json                 (E-heavy weights (0.2,0.3,0.5))
  evaluation_report_reasoning_weights.json (C-heavy weights (0.4,0.4,0.2))

Observed bands across both reports (10 real, live-LLM tasks):

    metric | clean tasks      | F1 (hallucination) | F2/F4 (exec fail) | F3 (reasoning)
    -------|------------------|---------------------|--------------------|----------------
    E      | 1.0              | 1.0                 | 0.0                | 1.0
    S      | ~0.47 - 0.61     | ~0.13               | ~0.48 - 0.61       | ~0.49 - 0.62
    C      | ~0.49 - 0.53     | ~0.52 - 0.55        | ~0.49 - 0.56       | ~0.11 - 0.16

Two things fall out of that table directly:
  - E is a clean binary signal for F2/F4 (a failed tool_call zeroes it) --
    no calibration needed, just branch on it first.
  - S ~0.13 for F1 sits far below the ~0.47-0.62 band every other
    condition (including clean) produces -- so S < 0.30 is a safe,
    non-noise cutoff for "this is a real hallucination, not baseline
    variance".
  - C ~0.11-0.16 for F3 sits far below the ~0.49-0.56 band clean/F1/F2/F4
    runs all produce -- so C < 0.25 is a safe, non-noise cutoff for
    "this is a real reasoning-inconsistency signal, not the ~0.5
    baseline noise Phase 4 documented".

select_weights() below picks a weight profile using exactly those three
observations, checked in order of how unambiguous the signal is (E is
binary and checked first; S and C thresholds are checked next, each
comfortably outside the noise band above; anything left over falls back
to the E-heavy profile, which Phase 4 showed gives FPR=0.0 on clean
tasks when no other-metric signal is present).
"""

from __future__ import annotations

from typing import Optional

# --- Calibration constants, derived from evaluation_report.json and
# evaluation_report_reasoning_weights.json (see module docstring table).
# Keep these named and documented -- do NOT inline magic numbers below,
# so a teammate re-running with more tasks (n_per_type=5+, per the
# left-to-do list) can re-derive/tune them from a larger sample without
# hunting through the function body.

# Clean-run / baseline-noise band for C sits at ~0.49-0.56 across every
# non-F3 condition observed. A real F3 signal (~0.11-0.16 observed) is
# well below this -- 0.25 leaves comfortable margin on both sides.
C_LOW_THRESHOLD = 0.25

# Clean/F2/F3/F4 S sits at ~0.47-0.62. F1's real hallucination signal
# (~0.13 observed) is far below this -- 0.30 leaves comfortable margin.
S_LOW_THRESHOLD = 0.30

# Weight profiles reused from reliability/reliability_score.DEFAULT_WEIGHTS
# (kept as local literals so this module has no import-order dependency
# on that file, and so the profile choice here is self-documenting).
E_HEAVY_WEIGHTS = (0.2, 0.3, 0.5)   # Jeong's "api_orchestration" profile
S_HEAVY_WEIGHTS = (0.2, 0.6, 0.2)   # push weight onto S for hallucination
C_HEAVY_WEIGHTS = (0.7, 0.15, 0.15)  # push weight onto C for reasoning drift


def select_weights(
    C: float,
    S: float,
    E: float,
    task_type: Optional[str] = None,
) -> tuple[float, float, float]:
    """
    Pick a (w1, w2, w3) weight tuple for THIS step's C/S/E, instead of
    using one fixed tuple for every task.

    Args:
        C: this step's Consistency score, in [0, 1].
        S: this step's Semantic Accuracy score, in [0, 1].
        E: this step's Execution Rate score, in [0, 1].
        task_type: optional, unused for now (accepted for forward
            compatibility / future per-task-type tuning, and so the
            call site doesn't need to change if this becomes
            task-type-aware later).

    Returns:
        (w1, w2, w3) summing to 1.0, suitable for
        reliability.reliability_score.compute_reliability().

    Decision order (most-to-least unambiguous signal):
        1. E < 1.0  -> a tool_call actually failed (E is a direct
           success-ratio measurement, not noisy like C/S) -> weight
           heavily toward E. Matches Phase 4's E-heavy profile, which
           already gave 100% per-type accuracy on F2/F4.
        2. S < S_LOW_THRESHOLD -> a real hallucination signal (far
           below the clean/other-fault noise band) -> weight toward S.
        3. C < C_LOW_THRESHOLD -> a real reasoning-inconsistency signal
           (far below the clean/other-fault noise band) -> weight
           toward C. This is the branch that resolves Phase 4's
           documented F3-vs-FPR tradeoff: C only gets heavy weight when
           it is UNAMBIGUOUSLY low, not on ordinary ~0.5 baseline noise.
        4. Otherwise (no strong single-metric signal) -> fall back to
           the E-heavy profile, which Phase 4 showed gives FPR=0.0 on
           clean tasks.
    """
    for name, value in (("C", C), ("S", S), ("E", E)):
        if not (0.0 <= value <= 1.0):
            raise ValueError(f"{name} must be in [0, 1], got {value}.")

    if E < 1.0:
        return E_HEAVY_WEIGHTS

    if S < S_LOW_THRESHOLD:
        return S_HEAVY_WEIGHTS

    if C < C_LOW_THRESHOLD:
        return C_HEAVY_WEIGHTS

    return E_HEAVY_WEIGHTS