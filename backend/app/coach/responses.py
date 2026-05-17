"""CRUD for coach_responses (per-response feature snapshot + delta).

Thin DB layer mirroring users.py / session.py pattern. The actual feature
extraction + delta computation happens in routes.py (calls audio.features
and coach.feedback); this module just persists the result.
"""

from __future__ import annotations

import json
import secrets
import time
from dataclasses import dataclass
from typing import Optional

from ..db import connect, transaction


def _gen_response_id() -> str:
    return f"rsp_{secrets.token_urlsafe(12)}"


@dataclass(frozen=True, slots=True)
class CoachResponse:
    id: str
    session_id: str
    response_index: int
    question_text: Optional[str]
    duration_s: float
    features: dict
    delta_pct: dict
    cartesian_x: float
    cartesian_y: float
    consistency_label: str
    color: str
    narrative: Optional[str]
    created_at: int


def _row_to_response(row) -> CoachResponse:
    return CoachResponse(
        id=row["id"],
        session_id=row["session_id"],
        response_index=row["response_index"],
        question_text=row["question_text"],
        duration_s=row["duration_s"],
        features=json.loads(row["features_json"]),
        delta_pct=json.loads(row["delta_pct_json"]),
        cartesian_x=row["cartesian_x"],
        cartesian_y=row["cartesian_y"],
        consistency_label=row["consistency_label"],
        color=row["color"],
        narrative=row["narrative"],
        created_at=row["created_at"],
    )


def insert_response(
    *,
    session_id: str,
    question_text: Optional[str],
    duration_s: float,
    features: dict,
    delta_pct: dict,
    cartesian_x: float,
    cartesian_y: float,
    consistency_label: str,
    color: str,
    narrative: Optional[str],
    now: int | None = None,
) -> CoachResponse:
    """Insert a new response row; auto-computes ``response_index`` as
    (max existing + 1) for the session."""
    now = now if now is not None else int(time.time())
    rid = _gen_response_id()
    conn = connect()
    try:
        with transaction(conn):
            # Compute next index — SELECT MAX then INSERT in same tx prevents races
            # given SQLite's single-writer model (we hold the lock for the BEGIN block).
            row = conn.execute(
                "SELECT COALESCE(MAX(response_index), 0) AS m FROM coach_responses "
                "WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            next_idx = int(row["m"]) + 1
            conn.execute(
                """
                INSERT INTO coach_responses (
                    id, session_id, response_index, question_text, duration_s,
                    features_json, delta_pct_json, cartesian_x, cartesian_y,
                    consistency_label, color, narrative, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rid, session_id, next_idx, question_text, duration_s,
                    json.dumps(features), json.dumps(delta_pct),
                    cartesian_x, cartesian_y, consistency_label, color, narrative, now,
                ),
            )
        return get_response_by_id(rid)
    finally:
        conn.close()


def get_response_by_id(response_id: str) -> CoachResponse:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM coach_responses WHERE id = ?", (response_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"coach_responses row {response_id} not found")
        return _row_to_response(row)
    finally:
        conn.close()


def list_session_responses(session_id: str) -> list[CoachResponse]:
    """Return all responses for a session, ordered by response_index ASC."""
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT * FROM coach_responses WHERE session_id = ? "
            "ORDER BY response_index ASC",
            (session_id,),
        ).fetchall()
        return [_row_to_response(r) for r in rows]
    finally:
        conn.close()
