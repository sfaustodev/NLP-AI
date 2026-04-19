"""GET /api/health — liveness for nginx and uptime monitors (SPEC §6.4).

Returns 200 when:

- the SQLite DB is writable (temp table create/drop probe)
- ``ffmpeg`` is callable (pydub delegates to it for MP3/M4A/OGG)
- parselmouth imported successfully at module load (jitter extractor)

Returns 503 with a machine-readable reason on any of the above
failing. This is the endpoint nginx can wire to its own ``location
/health`` probe, and that uptime monitors should hit on a ~30s cadence.
"""

from __future__ import annotations

import logging
import shutil
import subprocess

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from .. import __version__
from ..audio import features as _features   # triggers parselmouth import attempt
from ..db import healthcheck as db_healthcheck

log = logging.getLogger("vox.api.health")
router = APIRouter()


def _ffmpeg_ok() -> bool:
    """True iff ``ffmpeg`` is on PATH and returns rc=0 on ``-version``."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        # ``check=False`` + inspect rc: we never want a CalledProcessError
        # to propagate out of a healthcheck.
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, timeout=5, check=False,
        )
        return result.returncode == 0
    except (OSError, subprocess.TimeoutExpired) as exc:
        log.warning("ffmpeg probe failed: %s", exc)
        return False


@router.get("/api/health")
def health() -> JSONResponse:
    """SPEC §6.4: 200 ok / 503 with reason."""
    if not _ffmpeg_ok():
        return JSONResponse(
            status_code=503,
            content={"status": "down", "reason": "ffmpeg not available"},
        )
    try:
        db_healthcheck()
    except Exception as exc:                  # pragma: no cover — SQLite is rarely down
        log.error("db healthcheck failed: %s", exc)
        return JSONResponse(
            status_code=503,
            content={"status": "down", "reason": "database not writable"},
        )
    if not _features._HAS_PARSELMOUTH:
        # Parselmouth unavailable means jitter is permanently None —
        # the rest of the service works, but we're degraded.
        return JSONResponse(
            status_code=503,
            content={"status": "degraded", "reason": "parselmouth missing"},
        )
    return JSONResponse(
        status_code=200,
        content={"status": "ok", "version": __version__},
    )
