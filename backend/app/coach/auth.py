"""HMAC-signed tokens for Coach.

Two token kinds, same underlying HMAC scheme:

- **session_token** binds a session id to an expiry timestamp. Carried in
  the URL path of every ``/api/coach/session/{token}/*`` endpoint. 1h TTL
  default. Tamper-resistant (HMAC SHA-256 over body) but not encrypted —
  body fields (session_id, expiry) are visible in the token, which is
  acceptable since session_id alone is useless without a valid signature.
- **lawyer_cookie_token** binds a user id to an expiry timestamp. Carried
  in the ``coach_session`` cookie. 30-day rolling TTL.

Encoding: ``base64url(body) + "." + base64url(sig)`` where body is JSON
``{"sub": "...", "kind": "session"|"lawyer", "exp": <unix_ts>}``.

The HMAC secret lives in ``settings.coach_hmac_secret`` (env
``VOX_COACH_HMAC_SECRET``). It is never logged.

Constant-time comparison is used everywhere a comparison touches the
secret-derived signature (``hmac.compare_digest``).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import secrets
import time
from dataclasses import dataclass
from typing import Literal

from ..config import settings
from ..errors import VoxError


# Error codes — VoxError code constants used by Coach auth.
COACH_TOKEN_MALFORMED = "COACH_TOKEN_MALFORMED"
COACH_TOKEN_BAD_SIG   = "COACH_TOKEN_BAD_SIG"
COACH_TOKEN_EXPIRED   = "COACH_TOKEN_EXPIRED"
COACH_TOKEN_WRONG_KIND = "COACH_TOKEN_WRONG_KIND"


# Token TTLs (seconds). Session = SPEC §3 ~1h; lawyer cookie = SPEC §9.1 30d.
SESSION_TOKEN_TTL_SECONDS = 60 * 60
LAWYER_COOKIE_TTL_SECONDS = 60 * 60 * 24 * 30
ACTIVATION_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 7   # magic-link lives 7d


TokenKind = Literal["session", "lawyer"]


@dataclass(frozen=True, slots=True)
class TokenClaims:
    """Decoded token body."""
    sub: str        # session_id or user_id
    kind: TokenKind
    exp: int        # unix ts


# ------------------------------------------------------------------ encoding

def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(s: str) -> bytes:
    # urlsafe_b64decode requires padding restored.
    pad = "=" * (-len(s) % 4)
    return base64.urlsafe_b64decode(s + pad)


def _sign(body_b64: str, secret: str) -> str:
    digest = hmac.new(secret.encode("utf-8"), body_b64.encode("ascii"),
                      hashlib.sha256).digest()
    return _b64url_encode(digest)


def _build_token(sub: str, kind: TokenKind, exp: int, secret: str) -> str:
    body = json.dumps({"sub": sub, "kind": kind, "exp": exp},
                      separators=(",", ":"), sort_keys=True).encode("utf-8")
    body_b64 = _b64url_encode(body)
    sig = _sign(body_b64, secret)
    return f"{body_b64}.{sig}"


def _parse_token(token: str, *, expected_kind: TokenKind, secret: str,
                 now: int) -> TokenClaims:
    """Verify token signature + expiry + kind. Raises VoxError on any failure."""
    parts = token.split(".")
    if len(parts) != 2:
        raise VoxError(
            code=COACH_TOKEN_MALFORMED,
            message="Token format invalid.",
            http_status=401,
            hint="Token must be 'body.signature' (two base64url parts joined by '.').",
        )
    body_b64, sig_received = parts
    sig_expected = _sign(body_b64, secret)
    if not hmac.compare_digest(sig_expected, sig_received):
        raise VoxError(
            code=COACH_TOKEN_BAD_SIG,
            message="Token signature invalid.",
            http_status=401,
            hint="Token was tampered with or signed by a different secret.",
        )
    try:
        body = json.loads(_b64url_decode(body_b64))
        claims = TokenClaims(sub=body["sub"], kind=body["kind"], exp=int(body["exp"]))
    except (ValueError, KeyError, TypeError) as e:
        raise VoxError(
            code=COACH_TOKEN_MALFORMED,
            message="Token body could not be decoded.",
            http_status=401,
            hint=f"Decode error: {type(e).__name__}",
        )
    if claims.kind != expected_kind:
        raise VoxError(
            code=COACH_TOKEN_WRONG_KIND,
            message=f"Token kind mismatch — expected '{expected_kind}', got '{claims.kind}'.",
            http_status=401,
            hint="Use a session_token for /api/coach/session/* and a lawyer cookie for /api/coach/quota.",
        )
    if now > claims.exp:
        raise VoxError(
            code=COACH_TOKEN_EXPIRED,
            message="Token expired.",
            http_status=401,
            hint="Re-authenticate (new session, or sign in again).",
        )
    return claims


# ------------------------------------------------------------------ public API

def gen_session_token(session_id: str, *, ttl_seconds: int | None = None,
                      now: int | None = None,
                      secret: str | None = None) -> str:
    """Sign a session_token binding ``session_id`` + expiry."""
    now = now if now is not None else int(time.time())
    ttl = ttl_seconds if ttl_seconds is not None else SESSION_TOKEN_TTL_SECONDS
    secret = secret if secret is not None else settings.coach_hmac_secret
    return _build_token(session_id, "session", now + ttl, secret)


def verify_session_token(token: str, *, now: int | None = None,
                         secret: str | None = None) -> TokenClaims:
    """Validate session_token; returns claims.sub == session_id."""
    now = now if now is not None else int(time.time())
    secret = secret if secret is not None else settings.coach_hmac_secret
    return _parse_token(token, expected_kind="session", secret=secret, now=now)


def gen_lawyer_cookie_token(user_id: str, *, ttl_seconds: int | None = None,
                            now: int | None = None,
                            secret: str | None = None) -> str:
    """Sign a lawyer cookie token binding ``user_id`` + expiry."""
    now = now if now is not None else int(time.time())
    ttl = ttl_seconds if ttl_seconds is not None else LAWYER_COOKIE_TTL_SECONDS
    secret = secret if secret is not None else settings.coach_hmac_secret
    return _build_token(user_id, "lawyer", now + ttl, secret)


def verify_lawyer_cookie(token: str, *, now: int | None = None,
                         secret: str | None = None) -> TokenClaims:
    """Validate lawyer cookie token; returns claims.sub == user_id."""
    now = now if now is not None else int(time.time())
    secret = secret if secret is not None else settings.coach_hmac_secret
    return _parse_token(token, expected_kind="lawyer", secret=secret, now=now)


def gen_activation_token() -> str:
    """Magic-link activation token. Opaque, random 24 url-safe bytes (~32 char).
    Stored in coach_users_tiers.activation_token; consumed on first hit."""
    return secrets.token_urlsafe(24)
