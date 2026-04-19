"""Tests for app.audio.projection (SPEC §8.4-§8.6, §14.1)."""

from __future__ import annotations

import pytest

from app.audio import projection as proj


def _deltas(jitter: float = 0.0, mfcc: float = 0.0,
            flux: float = 0.0, microtremor: float = 0.0) -> dict[str, float]:
    return {
        "jitter_local_pct":         jitter,
        "mfcc_delta_var_pct":       mfcc,
        "spectral_flux_pct":        flux,
        "microtremor_envelope_pct": microtremor,
    }


def test_origin_when_all_zero() -> None:
    p = proj.project(_deltas())
    assert p.quadrant == "ORIGIN"
    assert p.naturalness == pytest.approx(0.0)
    assert p.involuntary_stress == pytest.approx(0.0)


def test_natural_calm_quadrant() -> None:
    """Positive voluntary variation, suppressed microtremor."""
    p = proj.project(_deltas(jitter=50, mfcc=50, flux=50, microtremor=-50))
    assert p.naturalness > 0
    assert p.involuntary_stress < 0
    assert p.quadrant == "NATURAL_CALM"


def test_over_controlled_tense_quadrant() -> None:
    """The deception signature: suppressed voluntary, elevated tremor."""
    p = proj.project(_deltas(jitter=-50, mfcc=-50, flux=-50, microtremor=50))
    assert p.naturalness < 0
    assert p.involuntary_stress > 0
    assert p.quadrant == "OVER_CONTROLLED_TENSE"


def test_natural_stressed_quadrant() -> None:
    p = proj.project(_deltas(jitter=50, mfcc=50, flux=50, microtremor=50))
    assert p.naturalness > 0
    assert p.involuntary_stress > 0
    assert p.quadrant == "NATURAL_STRESSED"


def test_over_controlled_calm_quadrant() -> None:
    p = proj.project(_deltas(jitter=-50, mfcc=-50, flux=-50, microtremor=-50))
    assert p.naturalness < 0
    assert p.involuntary_stress < 0
    assert p.quadrant == "OVER_CONTROLLED_CALM"


def test_axes_saturate_in_unit_box() -> None:
    """Extreme deltas must stay within [-1, +1] via tanh saturation."""
    p = proj.project(_deltas(jitter=500, mfcc=500, flux=500, microtremor=500))
    assert -1.0 <= p.naturalness <= 1.0
    assert -1.0 <= p.involuntary_stress <= 1.0
    # With +500 on every channel we should be near +1 but not exceed it.
    assert p.naturalness > 0.9
    assert p.involuntary_stress > 0.9


def test_origin_dead_zone_boundary() -> None:
    """Both axes within ±0.05 of zero → ORIGIN, even if one is negative."""
    # With axis_sharpness=0.02, a pct of ~2.4 gives tanh(0.048) ≈ 0.0479.
    inside = proj.project(_deltas(jitter=2.0, mfcc=2.0, flux=2.0, microtremor=-2.0))
    assert inside.quadrant == "ORIGIN"

    # Nudge one axis past the dead-zone; quadrant classification kicks in.
    outside = proj.project(_deltas(jitter=20, mfcc=20, flux=20, microtremor=-2.0))
    assert outside.quadrant != "ORIGIN"
