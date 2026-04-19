"""GET /api/session — inspect current session state (SPEC §6.3).

Serves the frontend enough state to render the ritual correctly:
does the user have a baseline? which ritual steps already happened
today? how many analyses left on the quota? The response shape is
the SPEC §6.3 contract, and the frontend uses it to decide which
ritual steps are still clickable.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends

from ..rate_limit import (
    RITUAL_STEP_LIE,
    RITUAL_STEP_UNCERTAIN,
    check_quota,
    utc_day_bucket,
)
from ..sessions import Session, fetch_row, get_session
from ..db import connect

router = APIRouter()


def _ritual_steps_done_today(session_id: str, has_baseline: bool) -> list[str]:
    """Which ritual steps already fired for this session today.

    Driven by the ``analyses`` table: we look for any row with this
    session + today's UTC bucket + a matching ritual_step. The "truth"
    step comes from the ``baseline_established_at`` column on the
    sessions row (calibrate has no row in analyses because it doesn't
    count against quota).
    """
    day = utc_day_bucket()
    done: list[str] = []
    if has_baseline:
        done.append("truth")
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT DISTINCT ritual_step FROM analyses "
            "WHERE session_id = ? AND day_bucket = ? AND ritual_step IS NOT NULL",
            (session_id, day),
        ).fetchall()
    finally:
        conn.close()
    step_set = {row["ritual_step"] for row in rows}
    for step in (RITUAL_STEP_UNCERTAIN, RITUAL_STEP_LIE):
        if step in step_set:
            done.append(step)
    return done


def _iso_utc(ts: int) -> str:
    """Unix ts → ISO-8601 with the 'Z' Zulu marker (matches SPEC §6.3 example)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@router.get("/api/session")
def get_session_state(session: Session = Depends(get_session)) -> dict:
    """SPEC §6.3 response shape."""
    row = fetch_row(session.session_id)
    has_baseline = bool(row and row["baseline_established_at"])
    ritual_done = _ritual_steps_done_today(session.session_id, has_baseline)
    quota = check_quota(session.session_id)

    created_at = int(row["created_at"]) if row else 0

    return {
        "session_id":        session.session_id,
        "has_baseline":      has_baseline,
        "ritual_complete":   len(ritual_done) >= 3,   # truth + uncertain + lie
        "ritual_steps_done": ritual_done,
        "quota": {
            "remaining_today": quota.remaining_today,
            "resets_at":       quota.resets_at_iso,
        },
        "created_at":        _iso_utc(created_at) if created_at else None,
    }
