"""Integration tests for the HTTP API (SPEC §14.2).

Covers the five bullets in the spec:

- Full ritual flow (calibrate → uncertain → lie → extra counted)
- 4th normal analyze in one day → RATE_LIMITED
- Calibration does not count against quota
- Invalid audio → correct error code + HTTP status
- Session cookie set on first visit, reused after
"""

from __future__ import annotations

from typing import Any


def _post_audio(client, path: str, wav: bytes, **extra) -> Any:
    files = {"audio": ("sample.wav", wav, "audio/wav")}
    return client.post(path, files=files, data=extra)


def test_session_cookie_set_on_first_visit(client) -> None:
    r = client.get("/api/session")
    assert r.status_code == 200
    assert "vox_session" in r.cookies
    body = r.json()
    assert len(body["session_id"]) == 43           # token_urlsafe(32) → 43 chars
    assert body["has_baseline"] is False
    assert body["quota"]["remaining_today"] == 3


def test_session_cookie_reused(client) -> None:
    r1 = client.get("/api/session")
    r2 = client.get("/api/session")
    assert r1.json()["session_id"] == r2.json()["session_id"]


def test_invalid_audio_returns_spec_error_shape(client) -> None:
    r = _post_audio(client, "/api/calibrate", b"not-audio-at-all-just-some-text-bytes-here-ok-xx")
    assert r.status_code == 400
    body = r.json()
    assert "error" in body
    assert body["error"]["code"] == "AUDIO_UNSUPPORTED_FORMAT"
    assert "hint" in body["error"]


def test_too_short_audio_rejected(client, wav_too_short: bytes) -> None:
    r = _post_audio(client, "/api/calibrate", wav_too_short)
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "AUDIO_TOO_SHORT"


def test_calibrate_then_analyze_flow(client, wav_voiced: bytes,
                                      wav_second_sample: bytes) -> None:
    """Happy path: calibrate establishes baseline, then analyze uses it."""
    cal = _post_audio(client, "/api/calibrate", wav_voiced)
    assert cal.status_code == 200
    cal_body = cal.json()
    assert cal_body["baseline_established"] is True
    assert set(cal_body["baseline"]) >= {
        "jitter_local", "mfcc_delta_var_mean",
        "spectral_flux_mean", "microtremor_envelope",
    }

    # Now a normal analyze should use the session baseline.
    an = _post_audio(client, "/api/analyze", wav_second_sample)
    assert an.status_code == 200
    an_body = an.json()
    assert an_body["baseline_source"] == "session"
    assert an_body["projection"]["quadrant"] in {
        "ORIGIN", "NATURAL_CALM", "NATURAL_STRESSED",
        "OVER_CONTROLLED_CALM", "OVER_CONTROLLED_TENSE",
    }
    # The response must carry the SPEC §10.4 methodology note.
    assert "methodology_note" in an_body


def test_calibrate_never_counts_against_quota(client, wav_voiced: bytes) -> None:
    for _ in range(5):
        r = _post_audio(client, "/api/calibrate", wav_voiced)
        assert r.status_code == 200
    # Quota still full after 5 calibrations.
    sess = client.get("/api/session").json()
    assert sess["quota"]["remaining_today"] == 3


def test_rate_limit_after_three_normal_analyses(client, wav_voiced: bytes,
                                                 wav_second_sample: bytes) -> None:
    # Calibrate once so analyze has a baseline (any analyze without one
    # uses global baseline; doesn't matter for quota mechanics).
    assert _post_audio(client, "/api/calibrate", wav_voiced).status_code == 200

    # Three normal analyses succeed.
    for _ in range(3):
        r = _post_audio(client, "/api/analyze", wav_second_sample)
        assert r.status_code == 200

    # Fourth is rate-limited.
    r4 = _post_audio(client, "/api/analyze", wav_second_sample)
    assert r4.status_code == 429
    assert r4.json()["error"]["code"] == "RATE_LIMITED"


def test_ritual_freebie_does_not_count(client, wav_voiced: bytes,
                                        wav_second_sample: bytes) -> None:
    """uncertain + lie ritual steps shouldn't eat the 3/day quota."""
    assert _post_audio(client, "/api/calibrate", wav_voiced).status_code == 200

    r_unc = _post_audio(client, "/api/analyze", wav_second_sample,
                        ritual_step="uncertain")
    assert r_unc.status_code == 200
    r_lie = _post_audio(client, "/api/analyze", wav_second_sample,
                        ritual_step="lie")
    assert r_lie.status_code == 200

    # Still 3 quota-counting analyses available.
    sess = client.get("/api/session").json()
    assert sess["quota"]["remaining_today"] == 3
    assert set(sess["ritual_steps_done"]) >= {"truth", "uncertain", "lie"}


def test_ritual_requires_baseline(client, wav_second_sample: bytes) -> None:
    """ritual_step=lie before calibration → BASELINE_REQUIRED."""
    r = _post_audio(client, "/api/analyze", wav_second_sample,
                    ritual_step="lie")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "BASELINE_REQUIRED"


def test_ritual_already_used(client, wav_voiced: bytes,
                              wav_second_sample: bytes) -> None:
    assert _post_audio(client, "/api/calibrate", wav_voiced).status_code == 200
    r1 = _post_audio(client, "/api/analyze", wav_second_sample,
                     ritual_step="uncertain")
    assert r1.status_code == 200
    r2 = _post_audio(client, "/api/analyze", wav_second_sample,
                     ritual_step="uncertain")
    assert r2.status_code == 400
    assert r2.json()["error"]["code"] == "RITUAL_ALREADY_USED"


def test_health_endpoint(client) -> None:
    r = client.get("/api/health")
    # On a dev box without ffmpeg installed, we accept 503 — the shape
    # is what matters. CI/VPS should both have ffmpeg.
    assert r.status_code in (200, 503)
    body = r.json()
    assert "status" in body


def test_version_header_present(client) -> None:
    r = client.get("/api/session")
    assert r.headers.get("X-Vox-Version")
