"""Session cookie lifecycle and DB row management (SPEC §9.1, §9.2).

Every incoming API request flows through :func:`get_session`, which is
used as a FastAPI dependency. It:

1. Reads the ``vox_session`` cookie (or mints a new 32-byte URL-safe
   base64 ID if absent or malformed).
2. Hashes the client IP with ``VOX_SECRET_SALT`` to a 16-char SHA-256
   prefix (SPEC §10.2) — enough entropy to dedup rate-limit buckets
   but not enough to reverse back to the IP.
3. Upserts the ``sessions`` row: creates on first sight, bumps
   ``last_seen_at`` on every subsequent call.
4. Schedules a ``Set-Cookie`` header on the outgoing response with a
   30-day rolling expiry.

The cookie is always refreshed on every successful request (rolling
30 days) per SPEC §9.1 — we just re-set it with the same ID and a
fresh Max-Age. Browsers merge the cookie silently.

A note on trust: ``X-Forwarded-For`` is honoured when
``settings.cookie_secure`` is true (== we're behind a trusted nginx).
On local dev (``cookie_secure=False``) we use the raw
``request.client.host`` to avoid header-spoofing a fake IP and
escaping rate limits. SPEC §13.2 requires the nginx layer to strip
incoming ``X-Forwarded-For`` headers before forwarding, so once that
is in place, trusting the header is safe.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import time
from dataclasses import dataclass

from fastapi import Request, Response

from .config import settings
from .db import connect

log = logging.getLogger("vox.sessions")

COOKIE_NAME = "vox_session"
COOKIE_MAX_AGE_S = 30 * 24 * 60 * 60   # 30-day rolling expiry

# A valid session ID is exactly 43 chars: secrets.token_urlsafe(32)
# yields 43 URL-safe base64 characters (no padding). Anything else is
# either a mangled cookie or a forgery attempt, and we mint fresh.
_SESSION_ID_LEN = 43
_URLSAFE_CHARS = set(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789-_"
)


@dataclass(frozen=True, slots=True)
class Session:
    """The resolved session for an in-flight request."""

    session_id: str
    ip_hash: str
    is_new: bool           # True when this request minted the ID


# ---------------------------------------------------------------- helpers
def _mint_session_id() -> str:
    """Cryptographically-random 32-byte URL-safe base64 (SPEC §9.1)."""
    return secrets.token_urlsafe(32)


def _looks_like_session_id(value: str | None) -> bool:
    if not value or len(value) != _SESSION_ID_LEN:
        return False
    return all(c in _URLSAFE_CHARS for c in value)


def _hash_ip(ip: str) -> str:
    """SHA-256 of ``ip + SALT``, first 16 hex chars (SPEC §10.2)."""
    digest = hashlib.sha256(f"{ip}{settings.secret_salt}".encode("utf-8")).hexdigest()
    return digest[:16]


def _client_ip(request: Request) -> str:
    """Resolve the client IP, honouring XFF only when we're behind nginx."""
    if settings.cookie_secure:
        xff = request.headers.get("x-forwarded-for", "").strip()
        if xff:
            # The left-most entry is the original client per RFC 7239.
            return xff.split(",", 1)[0].strip()
    if request.client is not None:
        return request.client.host
    return "unknown"


def _set_cookie(response: Response, session_id: str) -> None:
    """Write the rolling session cookie on the outgoing response."""
    response.set_cookie(
        key=COOKIE_NAME,
        value=session_id,
        max_age=COOKIE_MAX_AGE_S,
        httponly=True,
        samesite="lax",
        secure=settings.cookie_secure,
        path="/",
    )


def _upsert_row(session_id: str, ip_hash: str, is_new: bool) -> None:
    """INSERT on first sight; UPDATE last_seen_at otherwise. Autocommit."""
    now = int(time.time())
    conn = connect()
    try:
        if is_new:
            conn.execute(
                "INSERT INTO sessions (session_id, created_at, last_seen_at, ip_hash) "
                "VALUES (?, ?, ?, ?)",
                (session_id, now, now, ip_hash),
            )
        else:
            cursor = conn.execute(
                "UPDATE sessions SET last_seen_at = ?, ip_hash = ? "
                "WHERE session_id = ?",
                (now, ip_hash, session_id),
            )
            # If the cookie points to a row we've pruned (SPEC §10.3),
            # UPDATE affects 0 rows — resurrect it so the client isn't
            # silently unidentified on the next request.
            if cursor.rowcount == 0:
                conn.execute(
                    "INSERT INTO sessions (session_id, created_at, last_seen_at, ip_hash) "
                    "VALUES (?, ?, ?, ?)",
                    (session_id, now, now, ip_hash),
                )
    except sqlite3.Error as exc:
        log.error("session upsert failed: %s", exc)
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------- dependency
def get_session(request: Request, response: Response) -> Session:
    """FastAPI dependency that resolves (or mints) the session for this request.

    Usage in a handler::

        @router.post("/api/analyze")
        def analyze(session: Session = Depends(get_session)): ...
    """
    raw = request.cookies.get(COOKIE_NAME)
    if _looks_like_session_id(raw):
        session_id = raw                                    # type: ignore[assignment]
        is_new = False
    else:
        session_id = _mint_session_id()
        is_new = True

    ip_hash = _hash_ip(_client_ip(request))
    _upsert_row(session_id, ip_hash, is_new)
    _set_cookie(response, session_id)    # roll the 30-day window forward

    return Session(session_id=session_id, ip_hash=ip_hash, is_new=is_new)


# ---------------------------------------------------------------- lookups
def fetch_row(session_id: str) -> sqlite3.Row | None:
    """Return the raw ``sessions`` row or ``None`` if it vanished."""
    conn = connect()
    try:
        return conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ).fetchone()
    finally:
        conn.close()


def write_baseline(session_id: str, features: dict[str, float]) -> None:
    """Persist the four calibration features on the session row (SPEC §8.1)."""
    now = int(time.time())
    conn = connect()
    try:
        conn.execute(
            "UPDATE sessions SET "
            "  baseline_jitter = ?, "
            "  baseline_mfcc_delta_var = ?, "
            "  baseline_spectral_flux = ?, "
            "  baseline_microtremor = ?, "
            "  baseline_established_at = ? "
            "WHERE session_id = ?",
            (
                features["jitter_local"],
                features["mfcc_delta_var_mean"],
                features["spectral_flux_mean"],
                features["microtremor_envelope"],
                now,
                session_id,
            ),
        )
    finally:
        conn.close()


def baseline_from_row(row: sqlite3.Row) -> dict[str, float | None]:
    """Unpack the four baseline_* columns back to feature-keyed dict."""
    return {
        "jitter_local":         row["baseline_jitter"],
        "mfcc_delta_var_mean":  row["baseline_mfcc_delta_var"],
        "spectral_flux_mean":   row["baseline_spectral_flux"],
        "microtremor_envelope": row["baseline_microtremor"],
    }
