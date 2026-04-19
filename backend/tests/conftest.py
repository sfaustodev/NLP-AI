"""Shared pytest fixtures for the Vox Probabilis test suite.

The suite is deliberately hermetic:

- Every test runs against a fresh ``:memory:``-like sqlite file under
  ``tmp_path`` and a throwaway ``VOX_SECRET_SALT``.
- Audio fixtures are **synthesised in-process** with numpy rather than
  checked into the repo, so the fixtures folder stays empty and the
  suite works on any machine with numpy + soundfile + ffmpeg.
- The FastAPI app is rebuilt per test (via the ``client`` fixture) so
  no global state leaks across tests.
"""

from __future__ import annotations

import io
import os
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

# -------- env var priming -----------------------------------------------
# These MUST be set before ``app.config`` is first imported, because that
# module reads os.environ at import time and exits the process on missing
# VOX_SECRET_SALT. Environment is per-pytest-process, so setting here in
# conftest.py (loaded before test modules) is enough.
os.environ.setdefault("VOX_SECRET_SALT", "test_salt_for_hermetic_suite_0123456789abcdef")
os.environ.setdefault("VOX_COOKIE_SECURE", "false")


@pytest.fixture()
def tmp_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway sqlite file, wired in via VOX_DB_PATH.

    Reloading ``app.config`` is the only way to force it to pick up a
    new env var mid-suite since it freezes ``settings`` at import time.
    """
    db = tmp_path / "vox_test.db"
    monkeypatch.setenv("VOX_DB_PATH", str(db))
    # The test code imports ``settings`` at its own module-load, so we
    # rebuild it here after the env tweak.
    import importlib
    from app import config as _config
    importlib.reload(_config)
    return db


@pytest.fixture()
def client(tmp_db: Path):
    """A fresh FastAPI TestClient with migrations applied against ``tmp_db``."""
    import importlib
    from fastapi.testclient import TestClient

    # Reload every app module that captured ``settings`` at import time.
    from app import db as _db, sessions as _sessions, rate_limit as _rate
    from app.api import analyze as _analyze
    from app.api import calibrate as _calibrate
    from app.api import session as _session_api
    from app.api import health as _health
    from app import main as _main
    for mod in (_db, _sessions, _rate, _analyze, _calibrate, _session_api, _health, _main):
        importlib.reload(mod)

    with TestClient(_main.app) as c:
        yield c


# -------- synthetic audio -----------------------------------------------
def _sine_wav(freq_hz: float, duration_s: float, sr: int = 16_000,
              amplitude: float = 0.3, jitter_pct: float = 0.0) -> bytes:
    """Return in-memory WAV bytes of a (possibly jittered) sine wave.

    ``jitter_pct`` modulates the instantaneous frequency by a small
    random walk so the jitter extractor has something non-zero to find.
    """
    n = int(duration_s * sr)
    t = np.arange(n) / sr
    if jitter_pct > 0:
        # Small random walk on the frequency to simulate natural voice.
        rng = np.random.default_rng(42)
        drift = rng.normal(0, jitter_pct, n).cumsum() / sr
        phase = 2 * np.pi * freq_hz * (t + drift)
    else:
        phase = 2 * np.pi * freq_hz * t
    y = (amplitude * np.sin(phase)).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture()
def wav_voiced() -> bytes:
    """A 5-second 160 Hz sine with light jitter — passes voiced-ratio gate."""
    return _sine_wav(freq_hz=160.0, duration_s=5.0, jitter_pct=0.02)


@pytest.fixture()
def wav_too_short() -> bytes:
    """A 1-second clip — should trip AUDIO_TOO_SHORT."""
    return _sine_wav(freq_hz=160.0, duration_s=1.0)


@pytest.fixture()
def wav_silence() -> bytes:
    """A 5-second near-silent clip — should trip NO_VOICE_DETECTED."""
    sr = 16_000
    y = (1e-5 * np.ones(5 * sr, dtype=np.float32))
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@pytest.fixture()
def wav_second_sample() -> bytes:
    """A different-timbre voiced sample for analyze-after-calibrate tests."""
    return _sine_wav(freq_hz=220.0, duration_s=5.0, jitter_pct=0.03, amplitude=0.25)
