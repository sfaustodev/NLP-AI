"""Integration tests for Coach HTTP routes (FastAPI TestClient).

Audio-dependent endpoints (``calibrate``, ``response``) require librosa /
parselmouth / numba and will fail under the locally-broken Python 3.13
llvmlite ABI. They're auto-skipped when ``app.audio.features.extract_all``
can't be imported; CI / VPS prod (Python 3.12) runs them green.
"""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from app.coach import auth as coach_auth
from app.coach import users as coach_users
from app.coach.middleware import COACH_COOKIE_NAME


# ----------------------------------------------------------- audio fixtures

_SAMPLES_DIR = (
    Path(__file__).resolve().parents[2]
    / "landing_page" / "samples" / "audios_claude"
)


def _can_decode_audio() -> bool:
    """Check whether the audio stack imports cleanly (llvmlite/numba)."""
    try:
        from app.audio import features as _f
        from app.audio import load as _l
        # Exercise the numba lazy-load path on a trivial array.
        import numpy as np
        _f.extract_all(np.zeros(16_000 * 3, dtype=np.float32))
        return True
    except Exception:
        return False


# Used as test-skip predicate; computed once per session by pytest.
_AUDIO_OK = _can_decode_audio()
_audio_skip = pytest.mark.skipif(
    not _AUDIO_OK,
    reason="audio stack (librosa/numba/parselmouth) unavailable in this env",
)


def _read_wav(name: str) -> bytes:
    p = _SAMPLES_DIR / name
    if not p.is_file():
        pytest.skip(f"audio fixture missing: {p}")
    return p.read_bytes()


# ----------------------------------------------------------- helpers

def _create_user_with_cookie(client, *, email: str = "adv@x.com",
                              tier: str = "TIER_1_MONTHLY") -> tuple[str, str]:
    """Set up a lawyer user + sign + install the coach_session cookie on the
    test client. Returns (user_id, cookie_token)."""
    user = coach_users.create_or_upgrade(email=email, tier_key=tier)
    cookie_token = coach_auth.gen_lawyer_cookie_token(user.id)
    client.cookies.set(COACH_COOKIE_NAME, cookie_token)
    return user.id, cookie_token


# ----------------------------------------------------------- /api/coach/quota

def test_quota_requires_cookie(client) -> None:
    r = client.get("/api/coach/quota")
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "COACH_NO_COOKIE"


def test_quota_with_cookie_returns_tier_info(client) -> None:
    _create_user_with_cookie(client, email="quota@test.com", tier="TIER_1_MONTHLY")
    r = client.get("/api/coach/quota")
    assert r.status_code == 200
    body = r.json()
    assert body["email"] == "quota@test.com"
    assert body["tier"]["key"] == "TIER_1_MONTHLY"
    assert body["tier"]["sessions_per_period"] == 45
    assert body["sessions_used"] == 0


# ----------------------------------------------------------- /api/coach/session/create

def test_create_session_requires_cookie(client) -> None:
    r = client.post("/api/coach/session/create", json={"session_name": "x"})
    assert r.status_code == 401


def test_create_session_with_cookie_returns_token(client) -> None:
    _create_user_with_cookie(client, email="c@c.com", tier="TIER_1_MONTHLY")
    r = client.post(
        "/api/coach/session/create",
        json={"session_name": "João prep", "planned_questions": ["Q1", "Q2"]},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"].startswith("ses_")
    assert "." in body["session_token"]  # HMAC token = body.signature
    assert body["state"] == "CREATED"
    assert body["session_name"] == "João prep"


def test_create_session_free_trial_quota_exhausted_on_second(client) -> None:
    _create_user_with_cookie(client, email="ft@ft.com", tier="FREE_TRIAL")
    r1 = client.post("/api/coach/session/create", json={"session_name": "s1"})
    assert r1.status_code == 200
    r2 = client.post("/api/coach/session/create", json={"session_name": "s2"})
    assert r2.status_code == 402
    assert r2.json()["error"]["code"] == "COACH_QUOTA_EXCEEDED"


# ----------------------------------------------------------- /api/coach/session/{token}

def test_get_session_invalid_token_401(client) -> None:
    r = client.get("/api/coach/session/garbage")
    assert r.status_code == 401
    assert r.json()["error"]["code"] in ("COACH_TOKEN_MALFORMED", "COACH_TOKEN_BAD_SIG")


def test_get_session_valid_token_returns_state(client) -> None:
    _create_user_with_cookie(client, email="g@g.com", tier="TIER_1_MONTHLY")
    create = client.post("/api/coach/session/create",
                          json={"session_name": "Test session"})
    token = create.json()["session_token"]
    r = client.get(f"/api/coach/session/{token}")
    assert r.status_code == 200
    body = r.json()
    assert body["state"] == "CREATED"
    assert body["session_name"] == "Test session"
    assert body["baseline_established"] is False


# ----------------------------------------------------------- end (no LLM, FREE_TRIAL)

def test_end_session_without_baseline_rejected(client) -> None:
    """SPEC: end allowed only from READY or IN_PRACTICE."""
    _create_user_with_cookie(client, email="e@e.com", tier="TIER_1_MONTHLY")
    create = client.post("/api/coach/session/create",
                          json={"session_name": "EndTest"})
    token = create.json()["session_token"]
    r = client.post(f"/api/coach/session/{token}/end")
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "COACH_INVALID_STATE_FOR_ACTION"


# ----------------------------------------------------------- aux: /coach/activate

def test_activate_consumes_token_sets_cookie_redirects(client) -> None:
    user = coach_users.create_or_upgrade(email="act@a.com", tier_key="FREE_TRIAL")
    tok = user.activation_token
    # Disable redirect-following so we can inspect the 303 + Set-Cookie.
    r = client.get(f"/coach/activate?token={tok}", follow_redirects=False)
    assert r.status_code == 303
    assert r.headers["location"] == "/coach"
    assert COACH_COOKIE_NAME in r.cookies
    # Subsequent /quota call with the new cookie works.
    r2 = client.get("/api/coach/quota")
    assert r2.status_code == 200
    assert r2.json()["email"] == "act@a.com"


def test_activate_invalid_token_401(client) -> None:
    r = client.get("/coach/activate?token=nonexistent", follow_redirects=False)
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "COACH_ACTIVATION_INVALID"


# ----------------------------------------------------------- aux: PDFs

def test_coach_terms_pdf_endpoint(client) -> None:
    r = client.get("/coach/terms.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


def test_coach_consent_template_pdf_endpoint(client) -> None:
    r = client.get("/coach/consent-template.pdf")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/pdf"
    assert r.content[:5] == b"%PDF-"


# ----------------------------------------------------------- v0.1 regression

def test_v1_root_still_serves_marketing(client) -> None:
    r = client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "VOX PROBABILIS" in body or "VOX&nbsp;PROBABILIS" in body


def test_v1_app_still_serves_v01_tool(client) -> None:
    r = client.get("/app")
    assert r.status_code == 200


def test_v1_api_health_intact(client) -> None:
    r = client.get("/api/health")
    assert r.status_code in (200, 503)


def test_v1_assets_mount_intact(client) -> None:
    r = client.get("/assets/index.html")
    assert r.status_code == 200


# ----------------------------------------------------------- audio-gated tests

@_audio_skip
def test_calibrate_happy_path(client) -> None:
    _create_user_with_cookie(client, email="cal@x.com", tier="TIER_1_MONTHLY")
    create = client.post("/api/coach/session/create",
                          json={"session_name": "CalTest"})
    token = create.json()["session_token"]
    wav = _read_wav("ai_truth.wav")
    r = client.post(
        f"/api/coach/session/{token}/calibrate",
        files={"audio": ("ai_truth.wav", wav, "audio/wav")},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["session_state"] == "READY"
    assert body["mic_quality"]["label"] in ("GREEN", "YELLOW", "RED")
    assert set(body["baseline"]) == {
        "jitter_local", "mfcc_delta_var_mean",
        "spectral_flux_mean", "microtremor_envelope",
    }


@_audio_skip
def test_full_flow_calibrate_response_end(client) -> None:
    """End-to-end happy path with real audio. Skipped if librosa env broken."""
    _create_user_with_cookie(client, email="full@x.com", tier="FREE_TRIAL")
    create = client.post("/api/coach/session/create",
                          json={"session_name": "Full"})
    token = create.json()["session_token"]

    cal = client.post(
        f"/api/coach/session/{token}/calibrate",
        files={"audio": ("t.wav", _read_wav("ai_truth.wav"), "audio/wav")},
    )
    assert cal.status_code == 200

    resp = client.post(
        f"/api/coach/session/{token}/response",
        files={"audio": ("l.wav", _read_wav("ai_lie.wav"), "audio/wav")},
        data={"question_text": "Onde você estava?"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["consistency_label"] in (
        "BASELINE", "SLIGHT_SHIFT", "NOTABLE_SHIFT", "MAJOR_SHIFT",
    )
    assert body["color"] in ("GREEN", "YELLOW", "ORANGE", "RED")

    end = client.post(f"/api/coach/session/{token}/end")
    assert end.status_code == 200
    assert end.json()["state"] == "ENDED"

    report = client.get(f"/api/coach/session/{token}/report.html")
    assert report.status_code == 200
    assert "Visão geral" in report.text  # template fallback for FREE_TRIAL

    pdf = client.get(f"/api/coach/session/{token}/report.pdf")
    assert pdf.status_code == 200
    assert pdf.content[:5] == b"%PDF-"
