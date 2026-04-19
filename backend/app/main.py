"""FastAPI app assembly — the one place that wires everything together.

Responsibilities:

- Configure stdlib logging (JSON-ish lines, journald-friendly).
- Apply migrations idempotently at startup (``apply_migrations``).
- Install a single exception handler that turns ``VoxError`` into the
  SPEC §6.5 error-response shape.
- Add a middleware that stamps ``X-Vox-Version`` on every response
  (SPEC §6 preamble).
- Register the four API routers under ``/api/*``.
- Mount the landing page:
  - ``GET /`` → the ``index.html`` of ``settings.static_dir``
  - ``GET /assets/...`` → the rest of the static tree

This module is deliberately small. Business logic stays in ``api/``
and ``audio/``; this file is pure wiring so that reading it gives
you a one-page tour of the service.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import analyze as analyze_api
from .api import calibrate as calibrate_api
from .api import health as health_api
from .api import session as session_api
from .config import settings
from .db import apply_migrations
from .errors import VoxError


def _configure_logging() -> None:
    """stdlib logging → stderr, INFO by default, honours VOX_LOG_LEVEL."""
    level = getattr(logging, settings.log_level, logging.INFO)
    logging.basicConfig(
        level=level,
        stream=sys.stderr,
        format='{"ts":"%(asctime)s","level":"%(levelname)s","logger":"%(name)s","msg":%(message)r}',
        datefmt="%Y-%m-%dT%H:%M:%S",
        force=True,
    )


def _register_exception_handlers(app: FastAPI) -> None:
    @app.exception_handler(VoxError)
    async def _vox_error_handler(_request: Request, exc: VoxError) -> JSONResponse:
        """SPEC §6.5 — typed error shape with code/message/hint."""
        return JSONResponse(
            status_code=exc.http_status,
            content={
                "error": {
                    "code":    exc.code,
                    "message": exc.message,
                    "hint":    exc.hint,
                },
            },
        )


def _register_middleware(app: FastAPI) -> None:
    """SPEC §6: every response carries ``X-Vox-Version``."""

    @app.middleware("http")
    async def _version_header(request: Request, call_next):
        response = await call_next(request)
        response.headers["X-Vox-Version"] = __version__
        return response

    # CORS: the frontend is served same-origin in production, so the
    # default empty allowlist is correct. Non-empty is useful for
    # local dev (e.g. Vite on :5173 hitting :8000).
    if settings.cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(settings.cors_origins),
            allow_credentials=True,
            allow_methods=["GET", "POST"],
            allow_headers=["*"],
        )


def _mount_static(app: FastAPI) -> None:
    """Serve the existing landing page as the site root.

    We don't use ``StaticFiles(..., html=True)`` at ``/`` because that
    would swallow the ``/api/*`` routes. Instead we expose ``/`` as an
    explicit route that returns ``index.html``, and mount the rest of
    the static tree under ``/assets``. If the static dir vanishes (e.g.
    VOX_STATIC_DIR points nowhere), we still boot — the API keeps
    working, and a minimal placeholder serves at ``/``.
    """
    static_root: Path = settings.static_dir
    index_path = static_root / "index.html"

    @app.get("/", include_in_schema=False)
    def _index():
        if index_path.is_file():
            return FileResponse(index_path)
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code":    "STATIC_MISSING",
                    "message": "Landing page not found at configured static dir.",
                    "hint":    f"Set VOX_STATIC_DIR to the directory containing index.html (currently {static_root}).",
                },
            },
        )

    if static_root.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=static_root, check_dir=False),
            name="assets",
        )


def create_app() -> FastAPI:
    """Build and return the FastAPI app. Called by uvicorn as ``app.main:app``."""
    _configure_logging()
    log = logging.getLogger("vox.main")

    apply_migrations()
    log.info('"migrations applied"')

    app = FastAPI(
        title="Vox Probabilis",
        version=__version__,
        docs_url=None,          # hide /docs in production; re-enable in dev if needed
        redoc_url=None,
    )

    _register_exception_handlers(app)
    _register_middleware(app)

    app.include_router(health_api.router)
    app.include_router(session_api.router)
    app.include_router(calibrate_api.router)
    app.include_router(analyze_api.router)

    _mount_static(app)

    log.info('"app ready"')
    return app


app = create_app()
