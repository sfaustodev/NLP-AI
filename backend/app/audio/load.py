"""Decode an upload into a mono 16 kHz float32 numpy array.

Pipeline (SPEC §7.1):

    bytes  -> MIME/magic check
           -> NamedTemporaryFile on /tmp (UUID name, auto-deleted)
           -> pydub.AudioSegment.from_file  [via ffmpeg for MP3/M4A/OGG]
           -> set_channels(1).set_frame_rate(16000)
           -> export to an in-memory WAV
           -> soundfile.read -> float32 in [-1, 1]
           -> return array, along with voiced_frame_ratio and duration_s

Privacy: the /tmp file is opened with ``delete=True`` so Python unlinks
it the moment the ``with`` block exits — and the unlink happens even on
exceptions. Raw audio lives on disk only for the ~50 ms that ffmpeg
needs to convert it.

Rejections:
- > MAX_UPLOAD_MB  (caller enforces via Content-Length; we re-check)
- duration < 3s    -> AUDIO_TOO_SHORT
- duration > 60s   -> truncate silently, log a warning (SPEC §7.1)
- voiced ratio < 0.1 -> NO_VOICE_DETECTED
- any decode raise -> AUDIO_CORRUPT
"""

from __future__ import annotations

import io
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from ..config import settings
from ..errors import raise_vox
from ..errors import (
    AUDIO_CORRUPT,
    AUDIO_MISSING,
    AUDIO_TOO_LARGE,
    AUDIO_TOO_LONG,
    AUDIO_TOO_SHORT,
    AUDIO_UNSUPPORTED_FORMAT,
    NO_VOICE_DETECTED,
)

log = logging.getLogger("vox.audio.load")

TARGET_SR = 16_000       # Hz — SPEC §7.1
MIN_DURATION_S = 3.0
MAX_DURATION_S = 60.0
MIN_VOICED_RATIO = 0.10  # reject silence / pure noise
VOICE_SPLIT_TOP_DB = 30  # librosa.effects.split threshold (SPEC §7.1)

# Magic-byte signatures for the five accepted formats.
# Order matters: MP4/M4A's 'ftyp' box lives at offset 4 so we test position.
_MAGIC: dict[str, tuple[bytes, int]] = {
    "wav":  (b"RIFF", 0),   # then b"WAVE" at offset 8 — checked in sniff()
    "mp3":  (b"ID3",  0),   # ID3v2-tagged MP3
    "ogg":  (b"OggS", 0),
    "flac": (b"fLaC", 0),
    "m4a":  (b"ftyp", 4),   # MP4/M4A box
}


@dataclass(frozen=True, slots=True)
class LoadedAudio:
    samples: np.ndarray        # mono float32, range [-1, 1]
    sample_rate: int           # 16000
    duration_s: float
    voiced_frame_ratio: float


def sniff_format(head: bytes) -> str:
    """Return a lowercase format tag or raise AUDIO_UNSUPPORTED_FORMAT."""
    if len(head) < 12:
        raise_vox(AUDIO_UNSUPPORTED_FORMAT, "File is too small to identify.")

    # WAV needs the extra 'WAVE' word to rule out other RIFF containers.
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "wav"
    # Plain MP3 without ID3 starts with an MPEG frame sync 0xFF 0xE0..FF.
    if head[0] == 0xFF and (head[1] & 0xE0) == 0xE0:
        return "mp3"
    for fmt, (sig, offset) in _MAGIC.items():
        if head[offset:offset + len(sig)] == sig:
            return fmt
    raise_vox(AUDIO_UNSUPPORTED_FORMAT)


def _detect_voiced_ratio(samples: np.ndarray) -> float:
    """Proportion of the signal that contains voiced activity (SPEC §7.1).

    Guard against a librosa quirk: ``effects.split`` computes top_db
    relative to the peak STFT power. On a pure-silence signal (all
    zeros or uniform DC) the reference power collapses and every
    frame lands at "the peak", so the whole array is reported as
    voiced. We short-circuit to 0.0 when the RMS energy is below a
    sane floor (~-70 dBFS) before letting librosa see the signal.
    """
    if samples.size == 0:
        return 0.0
    rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
    if rms < 1e-4:                # -80 dBFS; below any real mic floor
        return 0.0
    intervals = librosa.effects.split(samples, top_db=VOICE_SPLIT_TOP_DB)
    voiced = int(np.sum(intervals[:, 1] - intervals[:, 0])) if len(intervals) else 0
    return voiced / samples.size


def decode(raw: bytes) -> LoadedAudio:
    """Decode any supported format to a mono 16 kHz float32 array.

    Caller is responsible for enforcing an outer byte-length cap; we
    also re-check here against ``settings.max_upload_mb`` as defense in
    depth (a streaming request with no Content-Length could otherwise
    sneak past the handler).
    """
    if not raw:
        raise_vox(AUDIO_MISSING)

    if len(raw) > settings.max_upload_mb * 1024 * 1024:
        raise_vox(AUDIO_TOO_LARGE,
                  f"File is {len(raw) / 1_048_576:.1f} MB, limit is {settings.max_upload_mb} MB.")

    fmt = sniff_format(raw[:64])

    # pydub reads from a path, so the upload has to hit /tmp briefly.
    # NamedTemporaryFile(delete=True) unlinks the file on exit, even on raise.
    try:
        with tempfile.NamedTemporaryFile(
            dir="/tmp", prefix="vox_", suffix=f".{fmt}", delete=True,
        ) as tmp:
            tmp.write(raw)
            tmp.flush()
            segment = AudioSegment.from_file(tmp.name, format=fmt)
    except Exception as exc:
        log.warning("pydub decode failed: %s", exc)
        raise_vox(AUDIO_CORRUPT)

    # Truncate silently per SPEC §7.1.
    duration_s = segment.duration_seconds
    if duration_s > MAX_DURATION_S:
        log.warning("audio length %.1fs exceeded cap; truncating to %.0fs",
                    duration_s, MAX_DURATION_S)
        segment = segment[: int(MAX_DURATION_S * 1000)]
        duration_s = segment.duration_seconds
        if duration_s > MAX_DURATION_S + 0.1:   # defensive; shouldn't happen
            raise_vox(AUDIO_TOO_LONG)

    # Mono, 16 kHz, 16-bit PCM — then re-decode via soundfile for a
    # clean float32 numpy array in [-1, 1].
    segment = segment.set_channels(1).set_frame_rate(TARGET_SR).set_sample_width(2)
    buf = io.BytesIO()
    segment.export(buf, format="wav")
    buf.seek(0)
    samples, sr = sf.read(buf, dtype="float32", always_2d=False)
    if sr != TARGET_SR:
        # soundfile preserves whatever pydub wrote; this should never trip.
        samples = librosa.resample(samples, orig_sr=sr, target_sr=TARGET_SR)
        sr = TARGET_SR

    duration_s = float(samples.size) / TARGET_SR
    if duration_s < MIN_DURATION_S:
        raise_vox(
            AUDIO_TOO_SHORT,
            f"Audio must be at least {MIN_DURATION_S:.0f} seconds long. "
            f"Received {duration_s:.1f}s.",
        )

    voiced_ratio = _detect_voiced_ratio(samples)
    if voiced_ratio < MIN_VOICED_RATIO:
        raise_vox(NO_VOICE_DETECTED,
                  f"Only {voiced_ratio * 100:.0f}% of the sample contains voiced activity.")

    return LoadedAudio(
        samples=samples.astype(np.float32, copy=False),
        sample_rate=TARGET_SR,
        duration_s=duration_s,
        voiced_frame_ratio=float(voiced_ratio),
    )
