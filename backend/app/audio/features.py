"""The four spectral features, extracted per SPEC §7.2–§7.5.

Input contract: every extractor takes a 1-D float32 numpy array sampled
at 16 kHz. Output contract: every extractor returns a float or None.
``None`` propagates as "insufficient data" to the caller, which then
sets ``confidence="unreliable"`` in the response (SPEC §6.2).

These functions are deliberately stateless and pure: the calibration
sample and the analysis sample are decoded by the same pipeline and
then run through the same extractor. The scientific claim (that the
quadrant projection is a deception signature) lives in this file.

References:
- Boersma (2001) for jitter (Praat's 'Get jitter (local)')
- Davis & Mermelstein (1980) for MFCC, Furui (1986) for deltas
- Dixon (2006) for the spectral-flux definition used here
- Lippold (1971) for the 8–12 Hz physiological tremor band
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, hilbert, resample_poly

log = logging.getLogger("vox.audio.features")

SR = 16_000

# STFT parameters (SPEC §7.3, §7.4): 25 ms window, 10 ms hop at 16 kHz.
N_FFT = 400
HOP_LENGTH = 160
WIN_LENGTH = 400
N_MFCC = 13

# Jitter: Praat 'To PointProcess (periodic, cc)' pitch floor/ceiling.
JITTER_PITCH_FLOOR_HZ = 75
JITTER_PITCH_CEILING_HZ = 500
JITTER_MIN_PERIODS = 10

# Microtremor envelope band (SPEC §7.5).
MICROTREMOR_BAND_HZ = (8.0, 12.0)
MICROTREMOR_ENVELOPE_SR = 1_000   # resample envelope from 16k -> 1k (16:1)

# Import parselmouth lazily: it's heavy, and we want the rest of the
# module to remain importable in environments without Praat (e.g. CI
# that only tests the non-Praat features). __init__-time import would
# force a hard dependency across the whole codebase.
try:
    import parselmouth
    from parselmouth.praat import call as praat_call
    _HAS_PARSELMOUTH = True
except ImportError:   # pragma: no cover
    parselmouth = None         # type: ignore[assignment]
    praat_call = None          # type: ignore[assignment]
    _HAS_PARSELMOUTH = False
    log.warning("parselmouth unavailable — jitter will always return None")


@dataclass(frozen=True, slots=True)
class Features:
    jitter_local: float | None
    mfcc_delta_var_mean: float
    spectral_flux_mean: float
    microtremor_envelope: float

    def as_dict(self) -> dict[str, float | None]:
        return {
            "jitter_local":         self.jitter_local,
            "mfcc_delta_var_mean":  self.mfcc_delta_var_mean,
            "spectral_flux_mean":   self.spectral_flux_mean,
            "microtremor_envelope": self.microtremor_envelope,
        }


# ---------------------------------------------------------------- jitter
def jitter_local(y: np.ndarray) -> float | None:
    """Praat 'Get jitter (local)'. Default parameters per SPEC §7.2.

    Returns None when the signal yields fewer than JITTER_MIN_PERIODS
    voiced periods — callers should then mark confidence=unreliable.
    """
    if not _HAS_PARSELMOUTH:
        return None
    try:
        snd = parselmouth.Sound(values=y, sampling_frequency=SR)
        point_process = praat_call(
            snd, "To PointProcess (periodic, cc)",
            JITTER_PITCH_FLOOR_HZ, JITTER_PITCH_CEILING_HZ,
        )
        n_points = int(praat_call(point_process, "Get number of points"))
        if n_points < JITTER_MIN_PERIODS:
            return None
        # Praat 'Get jitter (local)' defaults: time range 0–0 (whole),
        # period_floor=0.0001, period_ceiling=0.02, max factor=1.3.
        value = praat_call(
            point_process, "Get jitter (local)",
            0, 0, 0.0001, 0.02, 1.3,
        )
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        return float(value)
    except Exception as exc:   # Praat can raise on degenerate signals
        log.warning("jitter extraction failed: %s", exc)
        return None


# ------------------------------------------------------------ mfcc delta
def mfcc_delta_var_mean(y: np.ndarray) -> float:
    """Mean-of-variances across the 13 MFCC-delta coefficient streams.

    We use mean-of-variances (not variance-of-means) because the paper's
    proxy for 'voluntary variation' is how much *each* MFCC coefficient
    fluctuates frame-to-frame — aggregation at the end preserves that.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=SR,
        n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
    )
    delta = librosa.feature.delta(mfcc, order=1)
    return float(np.mean(np.var(delta, axis=1)))


# --------------------------------------------------------- spectral flux
def spectral_flux_mean(y: np.ndarray) -> float:
    """L2 distance between successive normalised magnitude spectra (Dixon 2006)."""
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP_LENGTH))
    # Per-frame sum-normalisation (add tiny epsilon to avoid div-by-zero
    # on silent frames that slipped past the voiced-frame gate).
    S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-9)
    flux = np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))
    return float(np.mean(flux)) if flux.size else 0.0


# ---------------------------------------------------------- microtremor
def microtremor_envelope(y: np.ndarray) -> float:
    """RMS of the envelope band-passed through 8–12 Hz (SPEC §7.5).

    Classical Lippold microtremor observation adapted for voice:
    physiological tremor modulates the amplitude envelope in the
    8–12 Hz band. Steps:

        Hilbert  -> analytic signal -> envelope = |analytic|
        resample to 1 kHz  (from 16 kHz, 16:1 — plenty of margin for 8–12 Hz)
        Butterworth 4th-order bandpass 8–12 Hz
        RMS of the filtered envelope

    The v0.1 coefficient may be refined in v0.2 based on real usage data.
    """
    analytic = hilbert(y)
    envelope = np.abs(analytic)
    env_1k = resample_poly(envelope, up=1, down=SR // MICROTREMOR_ENVELOPE_SR)
    nyquist = MICROTREMOR_ENVELOPE_SR / 2
    low, high = MICROTREMOR_BAND_HZ
    b, a = butter(N=4, Wn=[low / nyquist, high / nyquist], btype="bandpass")
    tremor_band = filtfilt(b, a, env_1k)
    return float(np.sqrt(np.mean(tremor_band ** 2)))


# ------------------------------------------------------------- orchestrator
def extract_all(y: np.ndarray) -> Features:
    """Compute all four features. Each extractor is independent."""
    return Features(
        jitter_local=jitter_local(y),
        mfcc_delta_var_mean=mfcc_delta_var_mean(y),
        spectral_flux_mean=spectral_flux_mean(y),
        microtremor_envelope=microtremor_envelope(y),
    )
