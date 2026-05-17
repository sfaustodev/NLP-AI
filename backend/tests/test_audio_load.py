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


def test_sniff_format_webm() -> None:
    """WEBM/EBML header — requires EBML magic AND a 'webm' DocType marker
    in the first 64 bytes (rules out other Matroska variants like .mkv)."""
    head = b"\x1a\x45\xdf\xa3" + b"\x00" * 8 + b"webm" + b"\x00" * 48
    assert sniff_format(head) == "webm"


def test_sniff_format_ogg_opus() -> None:
    """Sanity: OggS header still sniffs as ogg (Opus inside OGG container).
    Some non-Chromium browsers may emit audio/ogg;codecs=opus from MediaRecorder."""
    head = b"OggS" + b"\x00" * 12
    assert sniff_format(head) == "ogg"


def test_sniff_format_rejects_riff_non_wave() -> None:
    """RIFF container without WAVE word (e.g. AVI/WebP/ANI) must NOT be
    accepted as wav. Closes a pre-existing classification gap from when
    _MAGIC['wav'] = (b'RIFF', 0) — any RIFF was falling through to wav."""
    avi_like = b"RIFF" + b"\x00\x00\x00\x00" + b"AVI " + b"\x00" * 8
    with pytest.raises(VoxError) as ei:
        sniff_format(avi_like)
    assert ei.value.code == "AUDIO_UNSUPPORTED_FORMAT"


def test_sniff_format_rejects_ebml_without_webm_doctype() -> None:
    """EBML magic + non-webm DocType (e.g. plain .mkv with video) must NOT
    be classified as webm — defense against feeding arbitrary Matroska
    payload to ffmpeg from a Coach upload."""
    mkv_like = b"\x1a\x45\xdf\xa3" + b"\x00" * 4 + b"matroska" + b"\x00" * 48
    with pytest.raises(VoxError) as ei:
        sniff_format(mkv_like)
    assert ei.value.code == "AUDIO_UNSUPPORTED_FORMAT"


def test_sniff_format_truncated_rejected() -> None:
    """Any input shorter than 12 bytes (incl. truncated EBML mid-header)
    must reject with UNSUPPORTED_FORMAT before any further inspection."""
    for short in (b"", b"\x1a\x45", b"\x1a\x45\xdf\xa3", b"RIFF"):
        with pytest.raises(VoxError) as ei:
            sniff_format(short)
        assert ei.value.code == "AUDIO_UNSUPPORTED_FORMAT"


# ----------------------------------------------------------- AUDIO-BOMB defense

def test_probe_duration_returns_none_for_missing_file() -> None:
    """ffprobe on a nonexistent path → non-zero exit → returns None.
    Caller treats None as 'unknown', proceeds to pydub decode."""
    from app.audio.load import _probe_duration_seconds
    assert _probe_duration_seconds("/tmp/definitely_does_not_exist_42") is None


def test_decode_rejects_long_audio_pre_decode(wav_voiced: bytes, monkeypatch) -> None:
    """VOX-COACH-AUDIO-BOMB: if ffprobe reports duration > MAX_DURATION_S
    (with 5% tolerance), reject before pydub decodes. Without this, a small
    compressed payload (Opus/AAC) could expand to >1GB PCM in memory."""
    from app.audio import load as audio_load
    monkeypatch.setattr(audio_load, "_probe_duration_seconds",
                          lambda path, **_: 3600.0)  # 1h
    with pytest.raises(VoxError) as ei:
        audio_load.decode(wav_voiced)
    assert ei.value.code == "AUDIO_TOO_LONG"
    assert "3600" in ei.value.message
    assert "60" in ei.value.message


def test_decode_passes_when_probe_within_tolerance(wav_voiced: bytes, monkeypatch) -> None:
    """A duration slightly over MAX (e.g. 62s due to encoder rounding) stays
    within PROBE_TOLERANCE 5% and proceeds to decode. Decode itself then
    truncates to MAX_DURATION_S per the existing post-decode path."""
    from app.audio import load as audio_load
    monkeypatch.setattr(audio_load, "_probe_duration_seconds",
                          lambda path, **_: 62.0)
    # 5s wav fixture decodes fine; the monkeypatched probe doesn't actually
    # match the real duration, it just simulates the gate behaviour.
    loaded = audio_load.decode(wav_voiced)
    assert loaded.duration_s == pytest.approx(5.0, abs=0.1)


def test_decode_proceeds_when_probe_returns_none(wav_voiced: bytes, monkeypatch) -> None:
    """If ffprobe failed (None), caller should NOT reject — fall through to
    pydub which is the existing path. Preserves backward-compat for formats
    ffprobe doesn't introspect (e.g. raw PCM, exotic codecs)."""
    from app.audio import load as audio_load
    monkeypatch.setattr(audio_load, "_probe_duration_seconds",
                          lambda path, **_: None)
    loaded = audio_load.decode(wav_voiced)
    assert loaded.duration_s == pytest.approx(5.0, abs=0.1)
