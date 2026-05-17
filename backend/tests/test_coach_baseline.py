"""Tests for Coach baseline snapshot validation."""

from __future__ import annotations

import math

import pytest

from app.coach import baseline as bl
from app.errors import VoxError


_VALID = {
    "jitter_local":         0.0182,
    "mfcc_delta_var_mean":  0.04715,
    "spectral_flux_mean":   0.12841,
    "microtremor_envelope": 0.00342,
}


def test_snapshot_valid_returns_clean_copy() -> None:
    out = bl.snapshot_baseline(_VALID)
    assert out == _VALID
    assert out is not _VALID  # defensive copy


def test_snapshot_strips_extra_keys() -> None:
    """Only the 4 expected keys make it through."""
    src = dict(_VALID, extra_field=99.0)
    out = bl.snapshot_baseline(src)
    assert set(out.keys()) == set(bl.BASELINE_FEATURE_KEYS)
    assert "extra_field" not in out


@pytest.mark.parametrize("missing", list(bl.BASELINE_FEATURE_KEYS))
def test_missing_required_key_rejected(missing) -> None:
    bad = {k: v for k, v in _VALID.items() if k != missing}
    with pytest.raises(VoxError) as ei:
        bl.snapshot_baseline(bad)
    assert ei.value.code == bl.COACH_BASELINE_INVALID


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf"), None])
def test_non_finite_value_rejected(bad_value) -> None:
    src = dict(_VALID, jitter_local=bad_value)
    with pytest.raises(VoxError) as ei:
        bl.snapshot_baseline(src)
    assert ei.value.code == bl.COACH_BASELINE_INVALID
    assert "jitter_local" in ei.value.message


def test_int_value_coerced_to_float() -> None:
    src = dict(_VALID, jitter_local=0)
    out = bl.snapshot_baseline(src)
    assert isinstance(out["jitter_local"], float)
    assert out["jitter_local"] == 0.0
