"""FastAPI dependency helpers for Coach authentication.

Two flavours used across routes:

- ``get_lawyer_user_id`` — for endpoints that require a logged-in lawyer
  (``/api/coach/session/create``, ``/api/coach/quota``). Reads the
  ``coach_session`` cookie, validates the HMAC, returns the user id.
- ``get_session_from_path_token`` — for per-session endpoints. Validates
  the URL-path ``session_token`` (HMAC), loads the row, returns the
  session struct. The token already encodes session ownership cryptographically.
"""

from __future__ import annotations

from fastapi import Request

from ..errors import VoxError
from . import auth, session as cs


COACH_COOKIE_NAME = "coach_session"
COACH_NO_COOKIE   = "COACH_NO_COOKIE"


def get_lawyer_user_id(request: Request) -> str:
    """Resolve user_id from the ``coach_session`` cookie. Raises 401 if missing
    or invalid. Use as ``Depends(get_lawyer_user_id)`` in FastAPI routes.
    """
    token = request.cookies.get(COACH_COOKIE_NAME)
    if not token:
        raise VoxError(
            code=COACH_NO_COOKIE,
            message="Lawyer authentication required.",
            http_status=401,
            hint="Activate your account first via /coach/activate?token=...",
        )
    claims = auth.verify_lawyer_cookie(token)
    return claims.sub


def get_session_from_path_token(session_token: str) -> cs.CoachSession:
    """Resolve a CoachSession from a path ``{session_token}`` parameter.

    Validates token signature/expiry first (auth), then loads the row.
    Both layers raise VoxError on failure with appropriate HTTP status.
    """
    # auth.verify_session_token will raise on bad/expired token.
    auth.verify_session_token(session_token)
    return cs.get_session_by_token(session_token)
