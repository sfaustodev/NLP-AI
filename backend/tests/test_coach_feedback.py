"""Tests for Coach per-response feedback: deltas + cartesian + label + narrative."""

from __future__ import annotations

import math

import pytest

from app.coach import feedback as fb


BASELINE = {
    "jitter_local":         0.020,
    "mfcc_delta_var_mean":  0.050,
    "spectral_flux_mean":   0.120,
    "microtremor_envelope": 0.004,
}


# ----------------------------------------------------------- delta_pct

def test_delta_pct_zero_on_identical_features() -> None:
    deltas = fb.compute_delta_pct(BASELINE, BASELINE)
    for v in deltas.values():
        assert v == 0.0


def test_delta_pct_positive_when_current_higher() -> None:
    current = dict(BASELINE, jitter_local=0.030)  # +50% vs baseline 0.020
    deltas = fb.compute_delta_pct(current, BASELINE)
    assert math.isclose(deltas["jitter_local"], 50.0, abs_tol=0.1)


def test_delta_pct_clamped_to_500() -> None:
    current = dict(BASELINE, jitter_local=1.0)  # 4900% over baseline 0.020
    deltas = fb.compute_delta_pct(current, BASELINE)
    assert deltas["jitter_local"] == fb.DELTA_CLAMP_HIGH


def test_delta_pct_clamped_to_neg_100() -> None:
    current = dict(BASELINE, jitter_local=0.0)
    deltas = fb.compute_delta_pct(current, BASELINE)
    assert deltas["jitter_local"] == fb.DELTA_CLAMP_LOW


def test_delta_pct_handles_zero_baseline_safely() -> None:
    """Div-by-zero protection: baseline=0 → clamp to high on increase, 0 on equal."""
    base = dict(BASELINE, jitter_local=0.0)
    current_up = dict(BASELINE, jitter_local=0.5)
    current_zero = dict(BASELINE, jitter_local=0.0)
    assert fb.compute_delta_pct(current_up, base)["jitter_local"] == fb.DELTA_CLAMP_HIGH
    assert fb.compute_delta_pct(current_zero, base)["jitter_local"] == 0.0


# ----------------------------------------------------------- consistency_label

@pytest.mark.parametrize("max_abs,want_label,want_color", [
    (0.0,  "BASELINE",      "GREEN"),
    (9.5,  "BASELINE",      "GREEN"),
    (10.0, "SLIGHT_SHIFT",  "YELLOW"),
    (19.5, "SLIGHT_SHIFT",  "YELLOW"),
    (20.0, "NOTABLE_SHIFT", "ORANGE"),
    (34.5, "NOTABLE_SHIFT", "ORANGE"),
    (35.0, "MAJOR_SHIFT",   "RED"),
    (90.0, "MAJOR_SHIFT",   "RED"),
])
def test_consistency_label_boundary_table(max_abs, want_label, want_color) -> None:
    # Inject the target max_abs as jitter delta; other features stay at 0.
    deltas = {k: 0.0 for k in BASELINE}
    deltas["jitter_local"] = max_abs
    label, color = fb.consistency_label(deltas)
    assert (label, color) == (want_label, want_color)


def test_consistency_label_uses_max_abs_across_all_features() -> None:
    """Max-abs picks the biggest |delta| across all 4 features."""
    deltas = {"jitter_local": 5.0, "mfcc_delta_var_mean": 25.0,
              "spectral_flux_mean": -3.0, "microtremor_envelope": 18.0}
    label, color = fb.consistency_label(deltas)
    assert label == "NOTABLE_SHIFT"
    assert color == "ORANGE"


# ----------------------------------------------------------- cartesian

def test_cartesian_origin_when_no_change() -> None:
    deltas = {k: 0.0 for k in BASELINE}
    x, y = fb.cartesian_xy(deltas)
    assert x == 0.0 and y == 0.0


def test_cartesian_positive_x_on_elevated_voluntary_features() -> None:
    """Elevated jitter+mfcc+flux → positive naturalness X."""
    deltas = {"jitter_local": 30.0, "mfcc_delta_var_mean": 30.0,
              "spectral_flux_mean": 30.0, "microtremor_envelope": 0.0}
    x, y = fb.cartesian_xy(deltas)
    assert x > 0
    assert y == 0.0


def test_cartesian_negative_x_on_suppressed_voluntary_features() -> None:
    deltas = {"jitter_local": -40.0, "mfcc_delta_var_mean": -40.0,
              "spectral_flux_mean": -40.0, "microtremor_envelope": 0.0}
    x, _ = fb.cartesian_xy(deltas)
    assert x < 0


def test_cartesian_positive_y_on_elevated_microtremor() -> None:
    deltas = {"jitter_local": 0.0, "mfcc_delta_var_mean": 0.0,
              "spectral_flux_mean": 0.0, "microtremor_envelope": 40.0}
    _, y = fb.cartesian_xy(deltas)
    assert y > 0


def test_cartesian_bounded_to_unit_square() -> None:
    """tanh keeps every output in [-1, +1] even under saturation."""
    deltas = {k: fb.DELTA_CLAMP_HIGH for k in BASELINE}
    x, y = fb.cartesian_xy(deltas)
    assert -1.0 <= x <= 1.0
    assert -1.0 <= y <= 1.0


# ----------------------------------------------------------- narrative

def test_narrative_baseline_label_has_calm_message() -> None:
    deltas = {k: 0.0 for k in BASELINE}
    text = fb.short_narrative("BASELINE", deltas)
    assert "baseline" in text.lower()


def test_narrative_highlights_biggest_shifted_feature() -> None:
    deltas = {"jitter_local": 5.0, "mfcc_delta_var_mean": -42.0,
              "spectral_flux_mean": 10.0, "microtremor_envelope": 8.0}
    text = fb.short_narrative("MAJOR_SHIFT", deltas)
    assert "MFCC" in text or "espectral" in text.lower()
    assert "42" in text


# ----------------------------------------------------------- orchestrator

def test_compute_feedback_full_pipeline_baseline_path() -> None:
    """Identical features → BASELINE label, zero cartesian, calm narrative."""
    result = fb.compute_feedback(
        current_features=BASELINE, baseline_features=BASELINE,
    )
    assert result.consistency_label == "BASELINE"
    assert result.color == "GREEN"
    assert result.cartesian_x == 0.0
    assert result.cartesian_y == 0.0
    for v in result.delta_pct.values():
        assert v == 0.0


def test_compute_feedback_full_pipeline_major_shift() -> None:
    """Spec's canonical 'OVER_CONTROLLED_TENSE' pattern: jitter+mfcc+flux down,
    microtremor up. Expect negative X, positive Y, MAJOR_SHIFT label."""
    current = {
        "jitter_local":         0.012,  # -40%
        "mfcc_delta_var_mean":  0.025,  # -50%
        "spectral_flux_mean":   0.060,  # -50%
        "microtremor_envelope": 0.006,  # +50%
    }
    result = fb.compute_feedback(
        current_features=current, baseline_features=BASELINE,
    )
    assert result.consistency_label == "MAJOR_SHIFT"
    assert result.color == "RED"
    assert result.cartesian_x < 0    # over-controlled
    assert result.cartesian_y > 0    # tense
    assert "50" in result.narrative or "40" in result.narrative
