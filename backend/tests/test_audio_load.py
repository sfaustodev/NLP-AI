"""Tests for app.audio.load (SPEC §7.1, §14.1).

We cover the three "reject" paths (too short, silence, unsupported
format) and the one "accept" path (synthesised voiced WAV) because
that's what the error taxonomy and the v0.1 API contract hinge on.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import soundfile as sf

from app.audio.load import TARGET_SR, decode, sniff_format
from app.errors import VoxError


def test_decode_accepts_voiced_wav(wav_voiced: bytes) -> None:
    loaded = decode(wav_voiced)
    assert loaded.sample_rate == TARGET_SR
    assert loaded.duration_s == pytest.approx(5.0, abs=0.05)
    assert loaded.voiced_frame_ratio > 0.5       # near-full voicing
    assert loaded.samples.dtype == np.float32
    # Normalised to [-1, 1] — a 0.3-amplitude sine should stay in bounds.
    assert float(np.max(np.abs(loaded.samples))) <= 1.0


def test_decode_rejects_too_short(wav_too_short: bytes) -> None:
    with pytest.raises(VoxError) as excinfo:
        decode(wav_too_short)
    assert excinfo.value.code == "AUDIO_TOO_SHORT"


def test_decode_rejects_silence(wav_silence: bytes) -> None:
    with pytest.raises(VoxError) as excinfo:
        decode(wav_silence)
    assert excinfo.value.code == "NO_VOICE_DETECTED"


def test_decode_rejects_empty_bytes() -> None:
    with pytest.raises(VoxError) as excinfo:
        decode(b"")
    assert excinfo.value.code == "AUDIO_MISSING"


def test_decode_rejects_text_renamed_wav() -> None:
    """Text file with a .wav extension should fail magic-byte sniff."""
    bogus = b"this is definitely not audio, it's 64 bytes of pure text lol!!"
    with pytest.raises(VoxError) as excinfo:
        decode(bogus)
    assert excinfo.value.code == "AUDIO_UNSUPPORTED_FORMAT"


def test_sniff_format_wav() -> None:
    """A real WAV header should sniff as wav."""
    buf = io.BytesIO()
    sf.write(buf, np.zeros(1600, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    assert sniff_format(buf.getvalue()[:64]) == "wav"


def test_sniff_format_rejects_short() -> None:
    with pytest.raises(VoxError) as excinfo:
        sniff_format(b"abc")
    assert excinfo.value.code == "AUDIO_UNSUPPORTED_FORMAT"
