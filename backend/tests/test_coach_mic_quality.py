"""Tests for Coach mic quality (SPEC §6).

Three layers:
- ``mic_label`` pure decision table — every combination of (SNR band) ×
  (SR pass/fail) × (centroid in/out) produces the expected colour.
- ``compute_snr_db`` on synthetic numpy signals — silent, pure tone,
  noisy tone.
- ``compute_centroid_hz`` on pure sines at known frequencies — centroid
  should land near the sine frequency.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.coach import mic_quality as mq


# ----------------------------------------------------------- pure label

@pytest.mark.parametrize("snr,sr,centroid,want", [
    # GREEN baseline: high SNR + good SR + voice-band centroid
    (30.0, 44_100, 2000.0, "GREEN"),
    (26.0, 44_100, 1500.0, "GREEN"),
    # SNR boundary
    (25.0, 44_100, 2000.0, "YELLOW"),   # exactly at threshold → YELLOW
    (20.0, 44_100, 2000.0, "YELLOW"),
    (15.0, 44_100, 2000.0, "RED"),      # exactly at threshold → RED
    (10.0, 44_100, 2000.0, "RED"),
    # SR downgrade: <22k pushes GREEN→YELLOW, YELLOW→RED, RED→RED
    (30.0, 16_000, 2000.0, "YELLOW"),
    (20.0, 16_000, 2000.0, "RED"),
    (10.0, 16_000, 2000.0, "RED"),
    # Centroid downgrade: outside 800-4500 pushes one level down
    (30.0, 44_100, 100.0,  "YELLOW"),   # too low (hum)
    (30.0, 44_100, 7000.0, "YELLOW"),   # too high (hiss)
    (20.0, 44_100, 100.0,  "RED"),
    (10.0, 44_100, 100.0,  "RED"),
    # Both penalties stack: GREEN → YELLOW (sr) → RED (centroid)
    (30.0, 16_000, 100.0,  "RED"),
])
def test_mic_label_decision_table(snr, sr, centroid, want) -> None:
    assert mq.mic_label(snr, sr, centroid) == want


# ----------------------------------------------------------- SNR

def test_snr_zero_on_silence() -> None:
    silence = np.zeros(16_000 * 3, dtype=np.float32)
    assert mq.compute_snr_db(silence) == 0.0


def test_snr_zero_on_short_signal() -> None:
    short = np.random.default_rng(0).normal(0, 0.1, 100).astype(np.float32)
    assert mq.compute_snr_db(short) == 0.0


def test_snr_positive_on_pure_tone_with_noise_floor() -> None:
    """Speech-like: loud sine with quiet noise floor → SNR should be > 20 dB."""
    rng = np.random.default_rng(42)
    sr = 16_000
    n = sr * 3
    t = np.arange(n) / sr
    # Long sine 440 Hz amplitude 0.5 (loud) + low noise 0.005
    signal = 0.5 * np.sin(2 * np.pi * 440 * t)
    noise = rng.normal(0, 0.005, n)
    # Suppress signal in the first quarter to give the SNR estimator a quiet
    # window to call "noise floor".
    signal[: n // 4] = 0
    mix = (signal + noise).astype(np.float32)
    snr = mq.compute_snr_db(mix)
    assert snr > 20.0, f"expected SNR > 20 dB, got {snr:.1f}"


def test_snr_low_on_uniform_noise() -> None:
    """Uniform noise everywhere — top and bottom frames have similar energy →
    SNR close to 0 dB (i.e., low)."""
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 0.1, 16_000 * 3).astype(np.float32)
    snr = mq.compute_snr_db(noise)
    assert snr < 15.0, f"expected SNR < 15 dB on uniform noise, got {snr:.1f}"


# ----------------------------------------------------------- centroid

@pytest.mark.parametrize("freq_hz", [220.0, 1000.0, 2500.0, 5000.0])
def test_centroid_near_pure_sine_freq(freq_hz) -> None:
    sr = 16_000
    n = sr * 2
    t = np.arange(n) / sr
    sine = (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)
    centroid = mq.compute_centroid_hz(sine, sr_hz=sr)
    # Pure sine should put centroid within ±100 Hz of the tone (spectral leakage).
    assert abs(centroid - freq_hz) < 100, \
        f"centroid {centroid:.1f} Hz too far from sine {freq_hz} Hz"


def test_centroid_zero_on_silence() -> None:
    silence = np.zeros(16_000 * 2, dtype=np.float32)
    assert mq.compute_centroid_hz(silence, sr_hz=16_000) == 0.0


# ----------------------------------------------------------- orchestrator

def test_compute_mic_quality_pipeline() -> None:
    """End-to-end: build a clean voice-band sample, expect GREEN."""
    rng = np.random.default_rng(123)
    sr = 16_000
    n = sr * 3
    t = np.arange(n) / sr
    # Voice-band 1500 Hz sine + low noise floor, original recording 44.1k
    signal = 0.5 * np.sin(2 * np.pi * 1500 * t)
    signal[: n // 4] = 0      # quiet head for SNR floor measurement
    noise = rng.normal(0, 0.003, n)
    samples = (signal + noise).astype(np.float32)
    result = mq.compute_mic_quality(
        samples, sample_rate_hz=sr, original_sample_rate_hz=44_100,
    )
    assert result.label == "GREEN"
    assert result.sample_rate_hz == 44_100
    # 1500 Hz tone plus small white-noise floor — noise spreads energy upward
    # so the centroid drifts above the tone freq; 1000–2500 Hz is acceptable.
    assert 1000 < result.centroid_hz < 2500
    assert result.snr_db > 25
