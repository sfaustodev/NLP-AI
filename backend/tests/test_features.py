"""Tests for app.audio.features (SPEC §14.1).

The four extractors have very different flavours, so we hit each with
a targeted invariant rather than a giant gold-file comparison:

- jitter: on a pure (non-jittered) sine, jitter should be *small*
- MFCC delta variance: non-zero and finite on a varying sine
- spectral flux: small on a stable sine, larger on a chirp
- microtremor envelope: non-zero and finite on any voiced signal
"""

from __future__ import annotations

import numpy as np
import pytest

from app.audio import features as feat


SR = 16_000


def _sine(freq_hz: float, duration_s: float = 3.0, *,
          amplitude: float = 0.3, jitter_pct: float = 0.0) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    if jitter_pct > 0:
        rng = np.random.default_rng(1337)
        drift = rng.normal(0, jitter_pct, n).cumsum() / SR
        phase = 2 * np.pi * freq_hz * (t + drift)
    else:
        phase = 2 * np.pi * freq_hz * t
    return (amplitude * np.sin(phase)).astype(np.float32)


def _chirp(f0: float, f1: float, duration_s: float = 3.0,
           amplitude: float = 0.3) -> np.ndarray:
    n = int(duration_s * SR)
    t = np.arange(n) / SR
    # Linear frequency sweep f0 → f1.
    phase = 2 * np.pi * (f0 * t + 0.5 * (f1 - f0) / duration_s * t ** 2)
    return (amplitude * np.sin(phase)).astype(np.float32)


def test_mfcc_delta_var_mean_finite() -> None:
    y = _sine(freq_hz=200.0, jitter_pct=0.02)
    val = feat.mfcc_delta_var_mean(y)
    assert np.isfinite(val)
    assert val >= 0.0


def test_spectral_flux_stable_vs_chirp() -> None:
    """A chirp changes spectrum frame-to-frame; a pure tone doesn't."""
    stable = feat.spectral_flux_mean(_sine(freq_hz=200.0))
    sweep = feat.spectral_flux_mean(_chirp(f0=100.0, f1=2000.0))
    assert 0.0 <= stable < sweep
    # Dixon flux on a clean sine should be very small.
    assert stable < 0.05


def test_microtremor_envelope_finite() -> None:
    val = feat.microtremor_envelope(_sine(freq_hz=160.0))
    assert np.isfinite(val)
    assert val >= 0.0


def test_jitter_small_on_pure_sine() -> None:
    """Skip silently if parselmouth isn't installed in this env."""
    if not feat._HAS_PARSELMOUTH:
        pytest.skip("parselmouth not available in this environment")
    y = _sine(freq_hz=160.0, duration_s=3.0)
    val = feat.jitter_local(y)
    # Pure sine should produce near-zero jitter (or None on degenerate
    # pitch tracking — either is acceptable behaviour per SPEC §7.2).
    if val is not None:
        assert val < 0.01


def test_extract_all_returns_features() -> None:
    y = _sine(freq_hz=180.0, jitter_pct=0.02)
    out = feat.extract_all(y)
    assert np.isfinite(out.mfcc_delta_var_mean)
    assert np.isfinite(out.spectral_flux_mean)
    assert np.isfinite(out.microtremor_envelope)
    # jitter may be None if parselmouth is missing — that's fine.
    if out.jitter_local is not None:
        assert np.isfinite(out.jitter_local)
