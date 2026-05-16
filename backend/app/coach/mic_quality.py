"""Microphone quality check (SPEC_COACH §6).

Three signals are computed from the loaded audio:

- **SNR (dB)** — ratio of signal energy (top 10% high-energy frames) to
  noise floor (bottom 10% low-energy frames). 25 dB+ is studio-grade,
  15 dB is borderline intelligible, below 15 dB likely unusable.
- **Sample rate (Hz)** — read from the source file before v0.1's 16 kHz
  normalisation. < 22 000 Hz suggests telephony / phone-mic, which
  loses the 8-12 Hz microtremor band info we need (SPEC §7.5).
- **Spectral centroid (Hz)** — "center of mass" of the magnitude
  spectrum. Voice typically sits 800-4500 Hz; outside that range the
  recording is dominated by hum, hiss, or instrument noise.

Label rule (verbatim SPEC §6, frontend behaviour `warning only`):

    base = GREEN if snr_db > 25 else YELLOW if snr_db > 15 else RED
    if sr_hz < 22000:           base = downgrade(base)
    if not 800 ≤ centroid_hz ≤ 4500: base = downgrade(base)

GREEN → silent, YELLOW → "qualidade moderada" banner, RED → strong
warning banner. No state is hard-blocked — the lawyer is in charge.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# SPEC §6 thresholds.
SNR_GREEN_DB = 25.0
SNR_YELLOW_DB = 15.0
SR_MIN_HZ = 22_000
CENTROID_MIN_HZ = 800
CENTROID_MAX_HZ = 4500

# Frame size used by the SNR + centroid passes (samples at 16 kHz).
FRAME_SIZE = 1024
FRAME_HOP = 512


@dataclass(frozen=True, slots=True)
class MicQuality:
    label: str        # GREEN | YELLOW | RED
    snr_db: float
    sample_rate_hz: int
    centroid_hz: float


# ------------------------------------------------------------------ pure

def mic_label(snr_db: float, sr_hz: int, centroid_hz: float) -> str:
    """SPEC §6 — pure label decision from the three signals."""
    if snr_db > SNR_GREEN_DB:
        base = "GREEN"
    elif snr_db > SNR_YELLOW_DB:
        base = "YELLOW"
    else:
        base = "RED"

    if sr_hz < SR_MIN_HZ:
        base = "YELLOW" if base == "GREEN" else "RED"

    if not (CENTROID_MIN_HZ <= centroid_hz <= CENTROID_MAX_HZ):
        base = {"GREEN": "YELLOW", "YELLOW": "RED", "RED": "RED"}[base]

    return base


def compute_snr_db(samples: np.ndarray, *,
                   frame_size: int = FRAME_SIZE,
                   hop: int = FRAME_HOP) -> float:
    """Estimate SNR (dB) by ratio of top-10% to bottom-10% frame energies.

    Robust to short / silent inputs:
    - < 10 frames worth of audio → returns 0.0 (caller treats as RED)
    - all-zero / DC signal → returns 0.0
    """
    if samples.size < frame_size * 2:
        return 0.0

    energies = []
    for i in range(0, samples.size - frame_size, hop):
        chunk = samples[i:i + frame_size].astype(np.float64)
        energies.append(float(np.mean(chunk * chunk)))

    if len(energies) < 10:
        return 0.0

    e = np.sort(np.asarray(energies))
    top_n = max(1, len(e) // 10)
    noise = float(np.mean(e[:top_n]))
    signal = float(np.mean(e[-top_n:]))

    if noise <= 0 or signal <= 0:
        return 0.0
    # 10*log10 because both numerator and denominator are already power-domain
    # (squared amplitudes), not amplitude-domain.
    return float(10.0 * np.log10(signal / max(noise, 1e-12)))


def compute_centroid_hz(samples: np.ndarray, sr_hz: int) -> float:
    """Magnitude-weighted mean frequency (spectral centroid) via numpy FFT.

    Uses real FFT over the whole signal — fine for short Coach calibrations
    (8s × 16 kHz = 128k samples, single FFT < 5 ms). No librosa dependency
    keeps this testable in environments with broken numba/llvmlite.
    """
    if samples.size == 0:
        return 0.0
    spec = np.abs(np.fft.rfft(samples.astype(np.float64)))
    freqs = np.fft.rfftfreq(samples.size, 1.0 / sr_hz)
    total = float(np.sum(spec))
    if total <= 0:
        return 0.0
    return float(np.sum(freqs * spec) / total)


def compute_mic_quality(samples: np.ndarray, *, sample_rate_hz: int,
                        original_sample_rate_hz: int) -> MicQuality:
    """Orchestrator — compute all three signals + label.

    ``samples`` is the post-normalisation 16 kHz array (from ``audio.load.decode``).
    ``original_sample_rate_hz`` is the source-file rate before resample
    (also from ``LoadedAudio.original_sample_rate``).
    """
    snr = compute_snr_db(samples)
    centroid = compute_centroid_hz(samples, sr_hz=sample_rate_hz)
    label = mic_label(snr, original_sample_rate_hz, centroid)
    return MicQuality(
        label=label,
        snr_db=snr,
        sample_rate_hz=original_sample_rate_hz,
        centroid_hz=centroid,
    )
