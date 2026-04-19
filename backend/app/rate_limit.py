"""Rate limiting for /api/analyze (SPEC §9.3).

Rules in priority order (SPEC §9.3):

1. ``/api/calibrate`` never counts — it's the baseline setup step.
2. ``ritual_step="uncertain"`` AND session has a baseline AND the
   uncertain freebie hasn't been spent today → free. Otherwise the
   caller gets a ``RITUAL_ALREADY_USED`` (when already spent) or
   ``BASELINE_REQUIRED`` (when no baseline) up-front from the
   /api/analyze handler — this module deals only with *quota*.
3. Same rule for ``ritual_step="lie"``.
4. ``ritual_step="ai_bonus"`` → counts like a normal analysis.
5. No ritual_step → counts against the 3/day quota.

The rate limiter is the last thing we check before the expensive
decode+FFT path (SPEC §11.2 step 5). It writes a row only after a
successful analysis, so a rejected upload does not consume quota.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from datetime import datetime, timezone

from .config import settings
from .db import connect

RITUAL_STEP_UNCERTAIN = "uncertain"
RITUAL_STEP_LIE = "lie"
RITUAL_STEP_AI_BONUS = "ai_bonus"
_RITUAL_FREEBIE_STEPS = frozenset({RITUAL_STEP_UNCERTAIN, RITUAL_STEP_LIE})


@dataclass(frozen=True, slots=True)
class QuotaState:
    """Snapshot of a session's quota for the current UTC day."""

    remaining_today: int
    resets_at_iso: str      # ISO-8601 midnight UTC tomorrow


def utc_day_bucket(now_ts: int | None = None) -> str:
    """The ``YYYY-MM-DD`` UTC bucket used in the analyses index (SPEC §9.2)."""
    ts = now_ts if now_ts is not None else int(time.time())
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def _next_utc_midnight_iso(now_ts: int | None = None) -> str:
    """ISO-8601 timestamp for 00:00 UTC of the day *after* ``now_ts``."""
    ts = now_ts if now_ts is not None else int(time.time())
    now = datetime.fromtimestamp(ts, tz=timezone.utc)
    # Midnight tomorrow: strip to date + 1 day.
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    tomorrow_ts = int(midnight.timestamp()) + 24 * 3600
    return datetime.fromtimestamp(tomorrow_ts, tz=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def _count_quota_used(conn: sqlite3.Connection, session_id: str, day: str) -> int:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM analyses "
        "WHERE session_id = ? AND day_bucket = ? AND counted_against_quota = 1",
        (session_id, day),
    ).fetchone()
    return int(row["n"]) if row is not None else 0


def _ritual_step_spent_today(
    conn: sqlite3.Connection,
    session_id: str,
    day: str,
    step: str,
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM analyses "
        "WHERE session_id = ? AND day_bucket = ? AND ritual_step = ? "
        "LIMIT 1",
        (session_id, day, step),
    ).fetchone()
    return row is not None


def is_ritual_freebie(
    session_id: str,
    ritual_step: str | None,
    has_baseline: bool,
) -> bool:
    """Decide whether this analyze call should bypass the 3/day cap.

    Returns True only when all three conditions hold (SPEC §9.3.2-3):

    - ``ritual_step`` is ``uncertain`` or ``lie``
    - the session has an established baseline (else BASELINE_REQUIRED)
    - the freebie for that step hasn't been spent today
    """
    if ritual_step not in _RITUAL_FREEBIE_STEPS:
        return False
    if not has_baseline:
        return False
    conn = connect()
    try:
        return not _ritual_step_spent_today(
            conn, session_id, utc_day_bucket(), ritual_step
        )
    finally:
        conn.close()


def check_quota(session_id: str) -> QuotaState:
    """Return remaining 3/day quota for this session (SPEC §9.3.5).

    Does NOT raise on empty quota — callers use ``remaining_today`` to
    make the reject/allow decision so the ``/api/session`` endpoint can
    surface the state without triggering a 429.
    """
    day = utc_day_bucket()
    conn = connect()
    try:
        used = _count_quota_used(conn, session_id, day)
    finally:
        conn.close()
    remaining = max(0, settings.free_daily_quota - used)
    return QuotaState(
        remaining_today=remaining,
        resets_at_iso=_next_utc_midnight_iso(),
    )


def record_analysis(
    session_id: str,
    ritual_step: str | None,
    counted_against_quota: bool,
    quadrant: str,
) -> None:
    """Insert an analyses row — called *after* a successful /api/analyze.

    SPEC §9.3: rows capture enough to reproduce the quota decision
    without storing any audio or feature data. The quadrant is kept
    purely for v0.2 funnel analysis (which outcome was most common?)
    and is pre-computed so this write is fast.
    """
    now = int(time.time())
    day = utc_day_bucket(now)
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO analyses "
            "  (session_id, created_at, day_bucket, ritual_step, "
            "   counted_against_quota, quadrant) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                session_id,
                now,
                day,
                ritual_step,
                1 if counted_against_quota else 0,
                quadrant,
            ),
        )
    finally:
        conn.close()
