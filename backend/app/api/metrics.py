"""GET /api/metrics — private operator dashboard (DEPLOY.md §12.2).

Authenticated by ``VOX_METRICS_KEY`` (constant-time compare). Empty key
in env disables the endpoint entirely (404), so a stock deploy never
exposes counts by accident.
"""

from __future__ import annotations

import hmac
import logging
import time

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse

from ..config import settings
from ..db import connect

log = logging.getLogger("vox.api.metrics")
router = APIRouter()

# Process start time, captured at module load. Reload in tests via
# ``importlib.reload`` resets this — which is the desired behaviour.
_BOOT_TS = time.time()


def _unauthorized() -> JSONResponse:
    return JSONResponse(
        status_code=404,
        content={"error": {"code": "NOT_FOUND", "message": "Not Found", "hint": None}},
    )


@router.get("/api/metrics")
def metrics(request: Request, key: str = Query(default="")) -> JSONResponse:
    expected = settings.metrics_key
    if not expected:
        return _unauthorized()
    if not hmac.compare_digest(key, expected):
        return _unauthorized()

    now = int(time.time())
    cutoff_24h = now - 86_400

    conn = connect()
    try:
        total_calibrations = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE baseline_established_at IS NOT NULL"
        ).fetchone()[0]
        total_analyses = conn.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
        quadrant_rows = conn.execute(
            "SELECT quadrant, COUNT(*) FROM analyses GROUP BY quadrant"
        ).fetchall()
        active_sessions_24h = conn.execute(
            "SELECT COUNT(*) FROM sessions WHERE last_seen_at >= ?",
            (cutoff_24h,),
        ).fetchone()[0]
        dataset_optin_count = conn.execute(
            "SELECT COUNT(*) FROM dataset_optins"
        ).fetchone()[0]
    finally:
        conn.close()

    quadrant_counts = {row[0]: row[1] for row in quadrant_rows}

    return JSONResponse(
        status_code=200,
        content={
            "uptime_s": int(time.time() - _BOOT_TS),
            "total_calibrations": total_calibrations,
            "total_analyses": total_analyses,
            "quadrant_counts": quadrant_counts,
            "active_sessions_24h": active_sessions_24h,
            "dataset_optin_count": dataset_optin_count,
        },
    )
