"""Tests for Coach HMAC-signed tokens (session + lawyer cookie + activation).

All token verification is constant-time and tamper-resistant. Tests cover:
- round-trip sign + verify
- expiry rejection
- signature tamper rejection (1-byte mutation)
- kind mismatch rejection (using session token on lawyer cookie path)
- malformed token rejection
"""

from __future__ import annotations

import pytest

from app.coach import auth
from app.errors import VoxError


SECRET = "test_hmac_secret_only_for_unit_tests"


# ----------------------------------------------------------- session token

def test_session_token_roundtrip() -> None:
    tok = auth.gen_session_token("ses_abc", secret=SECRET, now=1000, ttl_seconds=3600)
    claims = auth.verify_session_token(tok, secret=SECRET, now=2000)
    assert claims.sub == "ses_abc"
    assert claims.kind == "session"
    assert claims.exp == 1000 + 3600


def test_session_token_expired() -> None:
    tok = auth.gen_session_token("ses_x", secret=SECRET, now=1000, ttl_seconds=60)
    with pytest.raises(VoxError) as ei:
        auth.verify_session_token(tok, secret=SECRET, now=2000)
    assert ei.value.code == auth.COACH_TOKEN_EXPIRED
    assert ei.value.http_status == 401


def test_session_token_tampered_signature_rejected() -> None:
    tok = auth.gen_session_token("ses_y", secret=SECRET, now=1000)
    body, sig = tok.split(".")
    # Flip last char of signature.
    last = sig[-1]
    flipped = "A" if last != "A" else "B"
    bad = f"{body}.{sig[:-1]}{flipped}"
    with pytest.raises(VoxError) as ei:
        auth.verify_session_token(bad, secret=SECRET, now=1100)
    assert ei.value.code == auth.COACH_TOKEN_BAD_SIG
    assert ei.value.http_status == 401


def test_session_token_tampered_body_rejected() -> None:
    """Mutating body invalidates HMAC even if body is well-formed JSON."""
    tok_a = auth.gen_session_token("ses_real", secret=SECRET, now=1000)
    tok_b = auth.gen_session_token("ses_FAKE", secret=SECRET, now=1000)
    body_b, _ = tok_b.split(".")
    _, sig_a = tok_a.split(".")
    swapped = f"{body_b}.{sig_a}"
    with pytest.raises(VoxError) as ei:
        auth.verify_session_token(swapped, secret=SECRET, now=1100)
    assert ei.value.code == auth.COACH_TOKEN_BAD_SIG


def test_session_token_different_secret_rejected() -> None:
    tok = auth.gen_session_token("ses_z", secret=SECRET, now=1000)
    with pytest.raises(VoxError) as ei:
        auth.verify_session_token(tok, secret="other_secret", now=1100)
    assert ei.value.code == auth.COACH_TOKEN_BAD_SIG


def test_session_token_malformed_rejected() -> None:
    with pytest.raises(VoxError) as ei:
        auth.verify_session_token("not-a-dot-token", secret=SECRET, now=1000)
    assert ei.value.code == auth.COACH_TOKEN_MALFORMED
    assert ei.value.http_status == 401


# ----------------------------------------------------------- lawyer cookie

def test_lawyer_cookie_roundtrip() -> None:
    tok = auth.gen_lawyer_cookie_token("usr_x", secret=SECRET, now=1000,
                                        ttl_seconds=60 * 60 * 24 * 30)
    claims = auth.verify_lawyer_cookie(tok, secret=SECRET, now=2000)
    assert claims.sub == "usr_x"
    assert claims.kind == "lawyer"


def test_lawyer_cookie_expired() -> None:
    tok = auth.gen_lawyer_cookie_token("usr_y", secret=SECRET, now=1000, ttl_seconds=10)
    with pytest.raises(VoxError) as ei:
        auth.verify_lawyer_cookie(tok, secret=SECRET, now=2000)
    assert ei.value.code == auth.COACH_TOKEN_EXPIRED


# ----------------------------------------------------------- kind mismatch

def test_session_token_used_as_lawyer_cookie_rejected() -> None:
    """A session_token is NOT a lawyer cookie even with correct signature."""
    sess_tok = auth.gen_session_token("ses_a", secret=SECRET, now=1000)
    with pytest.raises(VoxError) as ei:
        auth.verify_lawyer_cookie(sess_tok, secret=SECRET, now=1100)
    assert ei.value.code == auth.COACH_TOKEN_WRONG_KIND


def test_lawyer_cookie_used_as_session_token_rejected() -> None:
    cookie = auth.gen_lawyer_cookie_token("usr_a", secret=SECRET, now=1000)
    with pytest.raises(VoxError) as ei:
        auth.verify_session_token(cookie, secret=SECRET, now=1100)
    assert ei.value.code == auth.COACH_TOKEN_WRONG_KIND


# ----------------------------------------------------------- activation

def test_activation_token_unique_and_url_safe() -> None:
    tokens = {auth.gen_activation_token() for _ in range(50)}
    assert len(tokens) == 50
    # url-safe base64: alnum + -_
    import re
    for t in tokens:
        assert re.fullmatch(r"[A-Za-z0-9_-]+", t)
        assert len(t) >= 30  # token_urlsafe(24) ~32 chars


# ----------------------------------------------------------- defaults

def test_session_token_uses_settings_secret_when_unspecified() -> None:
    """Default path: gen + verify both fall back to settings.coach_hmac_secret."""
    tok = auth.gen_session_token("ses_default", now=1000)
    claims = auth.verify_session_token(tok, now=1100)
    assert claims.sub == "ses_default"
