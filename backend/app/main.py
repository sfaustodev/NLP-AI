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
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles

from . import __version__
from .api import analyze as analyze_api
from .api import calibrate as calibrate_api
from .api import health as health_api
from .api import metrics as metrics_api
from .api import session as session_api
from .coach import auth as coach_auth
from .coach import users as coach_users
from .coach.middleware import COACH_COOKIE_NAME
from .coach.pdf import consent as coach_pdf_consent
from .coach.pdf import terms as coach_pdf_terms
from .coach.routes import router as coach_router
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

    # ---- Coach (VOX-COACH-B): dashboard + session view + PDFs + activation
    @app.api_route("/coach", methods=["GET", "HEAD"], include_in_schema=False)
    def _coach_dashboard():
        return _serve(marketing_root / "coach" / "index.html", "coach/index.html")

    @app.api_route("/coach/session/{session_token}", methods=["GET", "HEAD"],
                   include_in_schema=False)
    def _coach_session_view(session_token: str):
        # session_token validated client-side via the /api/coach/session/{token}
        # polling call; this route just serves the static SPA shell.
        return _serve(marketing_root / "coach" / "session.html",
                      "coach/session.html")

    @app.api_route("/coach/session/{session_token}/response/{response_id}",
                   methods=["GET", "HEAD"], include_in_schema=False)
    def _coach_response_view(session_token: str, response_id: str):
        # Static page; JS fetches /api/coach/session/<token>/response/<id> for
        # the actual payload. Token + id validation lives server-side at the
        # JSON endpoint, not in this static shell.
        return _serve(marketing_root / "coach" / "response.html",
                      "coach/response.html")

    @app.api_route("/coach/terms.pdf", methods=["GET", "HEAD"],
                   include_in_schema=False)
    def _coach_terms_pdf():
        pdf_bytes = coach_pdf_terms.generate_terms_pdf()
        return Response(
            content=pdf_bytes, media_type="application/pdf",
            headers={"Content-Disposition": 'inline; filename="coach-terms.pdf"'},
        )

    @app.api_route("/coach/consent-template.pdf", methods=["GET", "HEAD"],
                   include_in_schema=False)
    def _coach_consent_pdf():
        pdf_bytes = coach_pdf_consent.generate_consent_pdf()
        return Response(
            content=pdf_bytes, media_type="application/pdf",
            headers={"Content-Disposition":
                     'inline; filename="coach-consent-template.pdf"'},
        )

    @app.get("/coach/activate", include_in_schema=False)
    def _coach_activate(token: str):
        """Consume activation_token → set HMAC lawyer cookie → redirect /coach."""
        user = coach_users.consume_activation_token(token)
        cookie_token = coach_auth.gen_lawyer_cookie_token(user.id)
        response = RedirectResponse(url="/coach", status_code=303)
        response.set_cookie(
            key=COACH_COOKIE_NAME, value=cookie_token,
            max_age=coach_auth.LAWYER_COOKIE_TTL_SECONDS,
            httponly=True, secure=settings.cookie_secure, samesite="lax",
            path="/",
        )
        return response

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

    coach_static_root = marketing_root / "coach" / "static"
    if coach_static_root.is_dir():
        app.mount(
            "/coach/static",
            StaticFiles(directory=coach_static_root, check_dir=False),
            name="coach_static",
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
    app.include_router(coach_router)

    _mount_static(app)

    log.info('"app ready"')
    return app


app = create_app()
