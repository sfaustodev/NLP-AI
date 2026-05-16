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
from .api import metrics as metrics_api
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
    """Serve the marketing landing at ``/`` and the v0.1 tool at ``/app``.

    Two static trees coexist:

    - ``settings.marketing_dir`` — 3-tab marketing landing (Explorer/Academic/
      Coach) with pricing and Terms. Assets mounted at ``/m/*``.
    - ``settings.static_dir`` — v0.1 functional tool (calibrate + analyze).
      Assets mounted at ``/assets/*``.

    ``/api/*`` routes are registered before this, so they win over any
    ``/`` collision.

    If a directory or file vanishes (e.g. VOX_MARKETING_DIR pointing
    nowhere) we still boot — the API keeps working and the missing route
    returns a typed STATIC_MISSING error so ops can diagnose quickly.
    """
    static_root: Path = settings.static_dir
    marketing_root: Path = settings.marketing_dir

    def _serve(path: Path, label: str) -> FileResponse | JSONResponse:
        if path.is_file():
            return FileResponse(path)
        return JSONResponse(
            status_code=404,
            content={
                "error": {
                    "code":    "STATIC_MISSING",
                    "message": f"{label} not found at configured path.",
                    "hint":    f"Expected at {path}.",
                },
            },
        )

    @app.api_route("/", methods=["GET", "HEAD"], include_in_schema=False)
    def _root():
        return _serve(marketing_root / "index.html", "Marketing landing")

    @app.api_route("/app", methods=["GET", "HEAD"], include_in_schema=False)
    def _app_tool():
        return _serve(static_root / "index.html", "v0.1 tool landing")

    @app.api_route("/privacy", methods=["GET", "HEAD"], include_in_schema=False)
    def _privacy():
        return _serve(static_root / "privacy.html", "privacy.html")

    @app.api_route("/terms", methods=["GET", "HEAD"], include_in_schema=False)
    def _terms_hub():
        return _serve(marketing_root / "terms-hub.html", "terms-hub.html")

    @app.api_route("/coach/terms", methods=["GET", "HEAD"], include_in_schema=False)
    def _coach_terms():
        return _serve(marketing_root / "coach-terms.html", "coach-terms.html")

    @app.api_route("/academic/terms", methods=["GET", "HEAD"], include_in_schema=False)
    def _academic_terms():
        return _serve(marketing_root / "academic-terms.html", "academic-terms.html")

    if static_root.is_dir():
        app.mount(
            "/assets",
            StaticFiles(directory=static_root, check_dir=False),
            name="assets",
        )

    if marketing_root.is_dir():
        app.mount(
            "/m",
            StaticFiles(directory=marketing_root, check_dir=False),
            name="marketing",
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
    app.include_router(metrics_api.router)

    _mount_static(app)

    log.info('"app ready"')
    return app


app = create_app()
