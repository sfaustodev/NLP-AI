"""Coach session lifecycle + state machine.

State machine:

    CREATED
        │  set_baseline()
        ▼
    AWAITING_CALIBRATION  (transient — set during calibrate processing)
        │  on success
        ▼
    READY
        │  first response submitted
        ▼
    IN_PRACTICE  ⟲  (subsequent responses keep state IN_PRACTICE)
        │  end_session()
        ▼
    ENDED  (terminal)

Transitions are enforced server-side: every state-mutating function validates
the current state against ``ALLOWED_TRANSITIONS`` and raises ``VoxError`` if
the request would violate the machine. Frontend state is purely a mirror.

Pure helpers live at module top; DB CRUD lives at the bottom. Token
generation is intentionally NOT here — see ``coach/auth.py`` — so session
operations stay testable without a secret key.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from ..errors import VoxError
from ..db import connect, transaction


class SessionState(str, Enum):
    CREATED = "CREATED"
    AWAITING_CALIBRATION = "AWAITING_CALIBRATION"
    READY = "READY"
    IN_PRACTICE = "IN_PRACTICE"
    ENDED = "ENDED"


# DAG of legal transitions. Self-loops (IN_PRACTICE → IN_PRACTICE) are
# allowed so successive responses don't trigger a transition error.
ALLOWED_TRANSITIONS: dict[SessionState, set[SessionState]] = {
    SessionState.CREATED:              {SessionState.AWAITING_CALIBRATION},
    SessionState.AWAITING_CALIBRATION: {SessionState.READY, SessionState.CREATED},
    SessionState.READY:                {SessionState.IN_PRACTICE, SessionState.ENDED},
    SessionState.IN_PRACTICE:          {SessionState.IN_PRACTICE, SessionState.ENDED},
    SessionState.ENDED:                set(),
}


# Error codes used by Coach routes — VoxError code constants.
COACH_SESSION_NOT_FOUND       = "COACH_SESSION_NOT_FOUND"
COACH_SESSION_EXPIRED         = "COACH_SESSION_EXPIRED"
COACH_INVALID_STATE_FOR_ACTION = "COACH_INVALID_STATE_FOR_ACTION"
COACH_BASELINE_ALREADY_SET    = "COACH_BASELINE_ALREADY_SET"
COACH_BASELINE_REQUIRED       = "COACH_BASELINE_REQUIRED"
COACH_SESSION_ALREADY_ENDED   = "COACH_SESSION_ALREADY_ENDED"


# Session row TTL — 1h from creation, refreshed when state advances.
SESSION_TTL_SECONDS = 60 * 60


@dataclass(frozen=True, slots=True)
class CoachSession:
    """Read model — mirrors the coach_sessions row."""
    id: str
    session_token: str
    owner_user_id: str
    session_name: str
    state: SessionState
    baseline_features: Optional[dict]
    mic_quality_label: Optional[str]
    mic_quality_snr_db: Optional[float]
    planned_questions: list[str]
    report_html: Optional[str]
    report_generated_at: Optional[int]
    created_at: int
    expires_at: int
    ended_at: Optional[int]


# ------------------------------------------------------------------ pure helpers

def gen_session_id() -> str:
    """Random session id. ulid-shaped (``ses_`` + 16 url-safe bytes) is plenty
    for our scale; we don't need monotonic ordering."""
    return f"ses_{secrets.token_urlsafe(16)}"


def validate_transition(current: SessionState, target: SessionState) -> None:
    """Raise VoxError if ``current → target`` is not in ALLOWED_TRANSITIONS."""
    allowed = ALLOWED_TRANSITIONS.get(current, set())
    if target not in allowed:
        raise VoxError(
            code=COACH_INVALID_STATE_FOR_ACTION,
            message=f"Cannot transition from {current.value} to {target.value}.",
            http_status=400,
            hint=f"Allowed from {current.value}: {sorted(s.value for s in allowed) or 'none (terminal)'}.",
        )


def _row_to_session(row) -> CoachSession:
    """Map sqlite Row → CoachSession dataclass. Decodes JSON blobs."""
    baseline = json.loads(row["baseline_features"]) if row["baseline_features"] else None
    planned = json.loads(row["planned_questions_json"]) if row["planned_questions_json"] else []
    return CoachSession(
        id=row["id"],
        session_token=row["session_token"],
        owner_user_id=row["owner_user_id"],
        session_name=row["session_name"],
        state=SessionState(row["state"]),
        baseline_features=baseline,
        mic_quality_label=row["mic_quality_label"],
        mic_quality_snr_db=row["mic_quality_snr_db"],
        planned_questions=planned,
        report_html=row["report_html"],
        report_generated_at=row["report_generated_at"],
        created_at=row["created_at"],
        expires_at=row["expires_at"],
        ended_at=row["ended_at"],
    )


# ------------------------------------------------------------------ DB CRUD

def create_session(
    *,
    owner_user_id: str,
    session_name: str,
    session_token: str,
    planned_questions: list[str] | None = None,
    ttl_seconds: int = SESSION_TTL_SECONDS,
    now: int | None = None,
) -> CoachSession:
    """Insert a fresh coach_sessions row in state CREATED.

    Caller must have already generated ``session_token`` via ``coach.auth``.
    """
    now = now if now is not None else int(time.time())
    sid = gen_session_id()
    conn = connect()
    try:
        with transaction(conn):
            conn.execute(
                """
                INSERT INTO coach_sessions (
                    id, session_token, owner_user_id, session_name, state,
                    baseline_features, mic_quality_label, mic_quality_snr_db,
                    planned_questions_json, report_html, report_generated_at,
                    created_at, expires_at, ended_at, deleted_at
                ) VALUES (?, ?, ?, ?, ?, NULL, NULL, NULL, ?, NULL, NULL, ?, ?, NULL, NULL)
                """,
                (
                    sid, session_token, owner_user_id, session_name,
                    SessionState.CREATED.value,
                    json.dumps(planned_questions or []),
                    now, now + ttl_seconds,
                ),
            )
        row = conn.execute(
            "SELECT * FROM coach_sessions WHERE id = ?", (sid,),
        ).fetchone()
        return _row_to_session(row)
    finally:
        conn.close()


def get_session_by_token(token: str, *, now: int | None = None) -> CoachSession:
    """Look up by HMAC token. Raises COACH_SESSION_NOT_FOUND / _EXPIRED."""
    now = now if now is not None else int(time.time())
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM coach_sessions WHERE session_token = ? AND deleted_at IS NULL",
            (token,),
        ).fetchone()
        if row is None:
            raise VoxError(
                code=COACH_SESSION_NOT_FOUND,
                message="Session not found.",
                http_status=404,
                hint="The session_token may have been mistyped or the session was deleted.",
            )
        sess = _row_to_session(row)
        # Expired sessions stay in DB for audit but become inaccessible.
        if now > sess.expires_at and sess.state != SessionState.ENDED:
            raise VoxError(
                code=COACH_SESSION_EXPIRED,
                message="Session token expired.",
                http_status=410,
                hint=f"Sessions live for {SESSION_TTL_SECONDS // 60} minutes. Create a new one.",
            )
        return sess
    finally:
        conn.close()


def get_session_by_id(session_id: str) -> CoachSession:
    """Internal lookup by primary key (used in tests + report generation)."""
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM coach_sessions WHERE id = ? AND deleted_at IS NULL",
            (session_id,),
        ).fetchone()
        if row is None:
            raise VoxError(
                code=COACH_SESSION_NOT_FOUND,
                message="Session not found.",
                http_status=404,
            )
        return _row_to_session(row)
    finally:
        conn.close()


def set_baseline(
    *,
    session_id: str,
    baseline_features: dict,
    mic_quality_label: str,
    mic_quality_snr_db: float,
) -> CoachSession:
    """Stores calibration result, transitions to READY (one-shot).

    Re-calibration is rejected: SessionState.READY → AWAITING_CALIBRATION is
    not in ALLOWED_TRANSITIONS. SPEC §3 §7.1 — baseline is immutable.
    """
    current = get_session_by_id(session_id)
    if current.baseline_features is not None:
        raise VoxError(
            code=COACH_BASELINE_ALREADY_SET,
            message="Baseline already established for this session.",
            http_status=409,
            hint="Per-session baseline is immutable after calibration (SPEC §7.1). Create a new session to recalibrate.",
        )
    # CREATED → AWAITING_CALIBRATION → READY is the canonical path; we collapse
    # both transitions into one write because the AWAITING_CALIBRATION state
    # is only meaningful while the request is in flight.
    validate_transition(current.state, SessionState.AWAITING_CALIBRATION)
    validate_transition(SessionState.AWAITING_CALIBRATION, SessionState.READY)

    conn = connect()
    try:
        with transaction(conn):
            conn.execute(
                """
                UPDATE coach_sessions
                SET baseline_features = ?,
                    mic_quality_label = ?,
                    mic_quality_snr_db = ?,
                    state = ?
                WHERE id = ?
                """,
                (
                    json.dumps(baseline_features),
                    mic_quality_label,
                    mic_quality_snr_db,
                    SessionState.READY.value,
                    session_id,
                ),
            )
        return get_session_by_id(session_id)
    finally:
        conn.close()


def mark_in_practice(session_id: str) -> CoachSession:
    """Advance READY → IN_PRACTICE on first response submission. Idempotent
    in IN_PRACTICE (self-loop allowed)."""
    current = get_session_by_id(session_id)
    if current.baseline_features is None:
        raise VoxError(
            code=COACH_BASELINE_REQUIRED,
            message="Baseline required before submitting responses.",
            http_status=400,
            hint="Run POST /api/coach/session/{token}/calibrate first.",
        )
    if current.state == SessionState.IN_PRACTICE:
        return current
    validate_transition(current.state, SessionState.IN_PRACTICE)

    conn = connect()
    try:
        with transaction(conn):
            conn.execute(
                "UPDATE coach_sessions SET state = ? WHERE id = ?",
                (SessionState.IN_PRACTICE.value, session_id),
            )
        return get_session_by_id(session_id)
    finally:
        conn.close()


def end_session(session_id: str, *, report_html: str | None = None,
                now: int | None = None) -> CoachSession:
    """Terminal transition to ENDED. Stores report_html if provided."""
    now = now if now is not None else int(time.time())
    current = get_session_by_id(session_id)
    if current.state == SessionState.ENDED:
        raise VoxError(
            code=COACH_SESSION_ALREADY_ENDED,
            message="Session already ended.",
            http_status=409,
        )
    validate_transition(current.state, SessionState.ENDED)

    conn = connect()
    try:
        with transaction(conn):
            conn.execute(
                """
                UPDATE coach_sessions
                SET state = ?, ended_at = ?, report_html = ?, report_generated_at = ?
                WHERE id = ?
                """,
                (
                    SessionState.ENDED.value,
                    now,
                    report_html,
                    now if report_html else None,
                    session_id,
                ),
            )
        return get_session_by_id(session_id)
    finally:
        conn.close()


def soft_delete(session_id: str, *, now: int | None = None) -> None:
    """LGPD right-to-erasure: marks deleted_at; row stays for audit."""
    now = now if now is not None else int(time.time())
    conn = connect()
    try:
        with transaction(conn):
            conn.execute(
                "UPDATE coach_sessions SET deleted_at = ? WHERE id = ?",
                (now, session_id),
            )
    finally:
        conn.close()
