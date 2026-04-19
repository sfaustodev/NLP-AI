"""Tests for app.audio.baseline (SPEC §8.1-§8.3, §14.1)."""

from __future__ import annotations

import math

import pytest

from app.audio import baseline as bl


def test_global_fallback_provenance() -> None:
    b = bl.global_fallback()
    assert b.source == "global"
    assert set(b.values) == {
        "jitter_local", "mfcc_delta_var_mean",
        "spectral_flux_mean", "microtremor_envelope",
    }
    # Paper values from SPEC §8.2 must not drift without a deliberate
    # code review — the defensive pin.
    assert b.values["jitter_local"] == pytest.approx(0.0182)
    assert b.values["microtremor_envelope"] == pytest.approx(0.0034)


def test_from_session_requires_all_columns() -> None:
    # Missing column → None (no baseline yet).
    partial = {
        "jitter_local": 0.01,
        "mfcc_delta_var_mean": 0.05,
        "spectral_flux_mean": 0.1,
        "microtremor_envelope": None,
    }
    assert bl.from_session(partial) is None

    full = {
        "jitter_local": 0.01,
        "mfcc_delta_var_mean": 0.05,
        "spectral_flux_mean": 0.1,
        "microtremor_envelope": 0.003,
    }
    resolved = bl.from_session(full)
    assert resolved is not None
    assert resolved.source == "session"
    assert resolved.values["jitter_local"] == pytest.approx(0.01)


def test_delta_pct_clamped() -> None:
    # +500 cap
    assert bl.delta_pct(sample=10.0, baseline=1.0) == 500.0
    # -100 floor
    assert bl.delta_pct(sample=0.0, baseline=1.0) == -100.0
    # Normal computation mid-range.
    assert bl.delta_pct(sample=1.5, baseline=1.0) == pytest.approx(50.0)
    # Zero baseline guard.
    assert bl.delta_pct(sample=5.0, baseline=0.0) == 0.0


def test_compute_deltas_keys_match_spec() -> None:
    """SPEC §6.2 uses mfcc_delta_var_pct (not mfcc_delta_var_mean_pct)."""
    sample = {
        "jitter_local": 0.0100,            # baseline 0.0182 → ~-45%
        "mfcc_delta_var_mean": 0.0200,
        "spectral_flux_mean": 0.1000,
        "microtremor_envelope": 0.0040,
    }
    baseline = bl.global_fallback()
    out = bl.compute_deltas(sample, baseline)
    assert set(out) == {
        "jitter_local_pct", "mfcc_delta_var_pct",
        "spectral_flux_pct", "microtremor_envelope_pct",
    }
    assert out["jitter_local_pct"] < 0
    assert all(math.isfinite(v) for v in out.values())


def test_compute_deltas_none_jitter_becomes_zero() -> None:
    sample = {
        "jitter_local": None,
        "mfcc_delta_var_mean": 0.05,
        "spectral_flux_mean": 0.13,
        "microtremor_envelope": 0.003,
    }
    out = bl.compute_deltas(sample, bl.global_fallback())
    assert out["jitter_local_pct"] == 0.0
