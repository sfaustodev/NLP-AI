"""FastAPI router for Coach — 8 endpoints under /api/coach/*.

Auth split:
- ``/session/create`` and ``/quota`` use lawyer cookie (via ``get_lawyer_user_id``).
- All other ``/session/{token}/*`` use the URL-path session_token (HMAC).
"""

from __future__ import annotations

import json
import logging
import re
import secrets
import time

from fastapi import APIRouter, Depends, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response

from ..audio import features as audio_features
from ..audio import load as audio_load
from ..db import connect, transaction
from ..errors import VoxError
from . import (
    auth as coach_auth,
    baseline as coach_baseline,
    feedback as coach_feedback,
    mic_quality,
    pricing,
    responses as coach_responses,
    session as cs,
    users as coach_users,
)
from .middleware import get_lawyer_user_id, get_session_from_path_token
from .pdf import session_report as pdf_session_report
from .reports.sonnet_standard import ReportInputs, generate_report


log = logging.getLogger("vox.coach.routes")
router = APIRouter(prefix="/api/coach", tags=["coach"])


# ------------------------------------------------------------------ helpers

_HTML_TAG_RX = re.compile(r"<[^>]+>")
_WHITESPACE_RX = re.compile(r"\s+")


def _html_to_plain(html: str) -> str:
    """Strip HTML tags + collapse whitespace — used to embed Sonnet narrative
    into the PDF (which can't render HTML directly)."""
    no_tags = _HTML_TAG_RX.sub(" ", html)
    return _WHITESPACE_RX.sub(" ", no_tags).strip()


def _ensure_features_dict(features) -> dict[str, float]:
    """Convert Features dataclass → dict + validate via baseline.snapshot.

    Raises COACH_BASELINE_INVALID if Praat returned None for jitter
    (parselmouth gives up on degenerate signals)."""
    return coach_baseline.snapshot_baseline(features.as_dict())


# ------------------------------------------------------------------ /session/create

@router.post("/session/create")
async def create_session(
    request: Request,
    user_id: str = Depends(get_lawyer_user_id),
) -> JSONResponse:
    """Create a new Coach session. Returns session_token to share with client.
    """
    user = coach_users.maybe_reset_period(user_id)

    # Body parsing — small JSON, accept either application/json or form.
    body = {}
    try:
        body = await request.json()
    except Exception:
        body = {}
    session_name = (body.get("session_name") or "").strip() or "Sessão Coach"
    planned_questions = body.get("planned_questions") or []
    if not isinstance(planned_questions, list):
        planned_questions = []
    planned_questions = [str(q).strip() for q in planned_questions if str(q).strip()]

    # Quota check before any side effect.
    pricing.check_can_start_session(
        tier_key=user.tier_key, sessions_used=user.sessions_used_this_period,
    )

    # We need session_id baked into the HMAC token, but cs.create_session
    # auto-generates an id. Generate locally, sign, then insert with that id.
    sid = f"ses_{secrets.token_urlsafe(16)}"
    session_token = coach_auth.gen_session_token(sid)
    now = int(time.time())
    expires_at = now + cs.SESSION_TTL_SECONDS

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
                (sid, session_token, user_id, session_name,
                 cs.SessionState.CREATED.value,
                 json.dumps(planned_questions), now, expires_at),
            )
    finally:
        conn.close()

    coach_users.increment_session_counter(user_id)

    return JSONResponse({
        "session_id": sid,
        "session_token": session_token,
        "session_name": session_name,
        "state": cs.SessionState.CREATED.value,
        "expires_at": expires_at,
    })


# ------------------------------------------------------------------ /session/{token}

@router.get("/session/{session_token}")
def get_session(session_token: str) -> JSONResponse:
    """Public state for the polling FE — frontend reads every ~1s."""
    sess = get_session_from_path_token(session_token)
    return JSONResponse(_session_public_payload(sess))


def _session_public_payload(sess: cs.CoachSession) -> dict:
    return {
        "session_id": sess.id,
        "session_name": sess.session_name,
        "state": sess.state.value,
        "baseline_established": sess.baseline_features is not None,
        "mic_quality_label": sess.mic_quality_label,
        "mic_quality_snr_db": sess.mic_quality_snr_db,
        "planned_questions": sess.planned_questions,
        "ended_at": sess.ended_at,
        "expires_at": sess.expires_at,
    }


# ------------------------------------------------------------------ /calibrate

@router.post("/session/{session_token}/calibrate")
async def calibrate_session(
    session_token: str,
    audio: UploadFile = File(...),
) -> JSONResponse:
    sess = get_session_from_path_token(session_token)
    if sess.state != cs.SessionState.CREATED:
        raise VoxError(
            code=cs.COACH_INVALID_STATE_FOR_ACTION,
            message=f"Cannot calibrate in state {sess.state.value}.",
            http_status=400,
            hint="Calibration is only allowed once, at session start.",
        )

    raw = await audio.read()
    loaded = audio_load.decode(raw)
    mic = mic_quality.compute_mic_quality(
        loaded.samples,
        sample_rate_hz=loaded.sample_rate,
        original_sample_rate_hz=loaded.original_sample_rate,
    )
    feats = audio_features.extract_all(loaded.samples)
    baseline = _ensure_features_dict(feats)
    cs.set_baseline(
        session_id=sess.id,
        baseline_features=baseline,
        mic_quality_label=mic.label,
        mic_quality_snr_db=mic.snr_db,
    )
    return JSONResponse({
        "session_state": cs.SessionState.READY.value,
        "mic_quality": {
            "label": mic.label,
            "snr_db": round(mic.snr_db, 2),
            "sample_rate_hz": mic.sample_rate_hz,
            "centroid_hz": round(mic.centroid_hz, 1),
        },
        "baseline": baseline,
        "duration_s": round(loaded.duration_s, 2),
    })


# ------------------------------------------------------------------ /response

@router.post("/session/{session_token}/response")
async def submit_response(
    session_token: str,
    audio: UploadFile = File(...),
    question_text: str | None = Form(None),
) -> JSONResponse:
    sess = get_session_from_path_token(session_token)
    if sess.state not in (cs.SessionState.READY, cs.SessionState.IN_PRACTICE):
        raise VoxError(
            code=cs.COACH_INVALID_STATE_FOR_ACTION,
            message=f"Cannot submit response in state {sess.state.value}.",
            http_status=400,
            hint="Calibrate the session first; or the session has already ended.",
        )

    raw = await audio.read()
    loaded = audio_load.decode(raw)
    feats = audio_features.extract_all(loaded.samples)
    current = _ensure_features_dict(feats)

    fb = coach_feedback.compute_feedback(
        current_features=current,
        baseline_features=sess.baseline_features,
    )

    # Transition to IN_PRACTICE (idempotent on subsequent responses).
    cs.mark_in_practice(sess.id)

    inserted = coach_responses.insert_response(
        session_id=sess.id,
        question_text=(question_text or "").strip()[:500] or None,
        duration_s=float(loaded.duration_s),
        features=current,
        delta_pct=fb.delta_pct,
        cartesian_x=fb.cartesian_x,
        cartesian_y=fb.cartesian_y,
        consistency_label=fb.consistency_label,
        color=fb.color,
        narrative=fb.narrative,
    )

    return JSONResponse({
        "response_id": inserted.id,
        "response_index": inserted.response_index,
        "delta_pct": {k: round(v, 2) for k, v in fb.delta_pct.items()},
        "cartesian": {
            "x": round(fb.cartesian_x, 3),
            "y": round(fb.cartesian_y, 3),
        },
        "consistency_label": fb.consistency_label,
        "color": fb.color,
        "narrative": fb.narrative,
        "duration_s": round(loaded.duration_s, 2),
    })


# ------------------------------------------------------------------ /end

@router.post("/session/{session_token}/end")
def end_session(session_token: str) -> JSONResponse:
    sess = get_session_from_path_token(session_token)
    if sess.state == cs.SessionState.ENDED:
        return JSONResponse(_session_public_payload(sess))  # idempotent
    if sess.state not in (cs.SessionState.READY, cs.SessionState.IN_PRACTICE):
        raise VoxError(
            code=cs.COACH_INVALID_STATE_FOR_ACTION,
            message=f"Cannot end session in state {sess.state.value}.",
            http_status=400,
        )

    owner = coach_users.get_user_by_id(sess.owner_user_id)
    resps = coach_responses.list_session_responses(sess.id)

    # LLM narrative if tier supports + quota OK; otherwise template fallback.
    narrative_html: str | None = None
    try:
        pricing.check_can_generate_report(
            tier_key=owner.tier_key, reports_used=owner.reports_used_this_period,
        )
        if pricing.supports_llm_report(owner.tier_key):
            narrative_html = generate_report(ReportInputs(
                session_name=sess.session_name,
                session_created_at=sess.created_at,
                session_ended_at=int(time.time()),
                mic_quality_label=sess.mic_quality_label or "UNKNOWN",
                mic_quality_snr_db=float(sess.mic_quality_snr_db or 0.0),
                baseline_features=sess.baseline_features or {},
                responses=[{
                    "question_text": r.question_text,
                    "duration_s": r.duration_s,
                    "delta_pct": r.delta_pct,
                    "consistency_label": r.consistency_label,
                    "color": r.color,
                    "narrative": r.narrative,
                } for r in resps],
            ))
            coach_users.increment_report_counter(owner.id)
    except VoxError as exc:
        # Quota exceeded → still end session, just no LLM report.
        log.warning('"end_session_skip_report code=%s reason=%s"', exc.code, exc.message)
        narrative_html = None

    ended = cs.end_session(sess.id, report_html=narrative_html)
    return JSONResponse({
        **_session_public_payload(ended),
        "report_available": narrative_html is not None,
    })


# ------------------------------------------------------------------ /report.html

# Tight CSP for the LLM-generated report endpoint. The Coach UI renders this
# inside an iframe sandbox="" — but if a user opens the URL directly the
# browser would otherwise honour the global nginx CSP ('self' 'unsafe-inline'),
# which permits inline scripts. A prompt-injected Sonnet narrative could then
# emit <script> and execute same-origin. We instead force a per-response CSP
# that disables every active capability:
#
#   - ``sandbox`` puts the response in a unique origin (no cookies, no DOM
#     access to the parent) — equivalent to ``<iframe sandbox="">``.
#   - ``default-src 'none'`` blocks every fetch unless explicitly allowed.
#   - ``script-src 'none'`` and ``style-src 'unsafe-inline'`` permit only the
#     inline styles needed by the report templates; no JS can run.
#   - ``base-uri`` / ``form-action`` ``'none'`` close the few CSP knobs that
#     can still cause side-effects under sandbox.
#
# Complementary headers (``X-Content-Type-Options``, ``X-Frame-Options``,
# ``Referrer-Policy``) make the endpoint hostile to clickjacking, MIME
# sniffing, and referrer leakage even if the user navigates directly.
_REPORT_CSP = (
    "sandbox; default-src 'none'; script-src 'none'; "
    "style-src 'unsafe-inline'; img-src data:; "
    "base-uri 'none'; form-action 'none'; frame-ancestors 'self'"
)
_REPORT_SECURITY_HEADERS = {
    "Content-Security-Policy": _REPORT_CSP,
    "X-Content-Type-Options":  "nosniff",
    "X-Frame-Options":         "SAMEORIGIN",
    "Referrer-Policy":         "no-referrer",
    "Cache-Control":           "private, no-store",
}


@router.get("/session/{session_token}/report.html")
def get_report_html(session_token: str) -> HTMLResponse:
    sess = get_session_from_path_token(session_token)
    if sess.state != cs.SessionState.ENDED:
        raise VoxError(
            code=cs.COACH_INVALID_STATE_FOR_ACTION,
            message="Report only available after session ends.",
            http_status=400,
        )
    html = sess.report_html or "<p>Relatório não gerado para esta sessão.</p>"
    return HTMLResponse(content=html, headers=_REPORT_SECURITY_HEADERS)


# ------------------------------------------------------------------ /report.pdf

@router.get("/session/{session_token}/report.pdf")
def get_report_pdf(session_token: str) -> Response:
    sess = get_session_from_path_token(session_token)
    if sess.state != cs.SessionState.ENDED:
        raise VoxError(
            code=cs.COACH_INVALID_STATE_FOR_ACTION,
            message="Report only available after session ends.",
            http_status=400,
        )
    owner = coach_users.get_user_by_id(sess.owner_user_id)
    tier = pricing.get_tier(owner.tier_key)
    resps = coach_responses.list_session_responses(sess.id)

    narrative_text = _html_to_plain(sess.report_html) if sess.report_html else None
    pdf_bytes = pdf_session_report.generate_session_report_pdf(
        session_name=sess.session_name,
        session_created_at=sess.created_at,
        session_ended_at=sess.ended_at or sess.created_at,
        tier_label=tier.label,
        mic_quality_label=sess.mic_quality_label or "UNKNOWN",
        mic_quality_snr_db=float(sess.mic_quality_snr_db or 0.0),
        responses=[{
            "duration_s": r.duration_s,
            "cartesian_x": r.cartesian_x,
            "cartesian_y": r.cartesian_y,
            "consistency_label": r.consistency_label,
            "color": r.color,
            "narrative": r.narrative,
        } for r in resps],
        narrative_text=narrative_text,
    )
    safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", sess.session_name)[:64] or "session"
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={"Content-Disposition": f'inline; filename="coach-{safe_name}.pdf"'},
    )


# ------------------------------------------------------------------ /quota

@router.get("/quota")
def get_quota(user_id: str = Depends(get_lawyer_user_id)) -> JSONResponse:
    user = coach_users.maybe_reset_period(user_id)
    tier = pricing.get_tier(user.tier_key)
    return JSONResponse({
        "email": user.email,
        "tier": {
            "key": tier.key,
            "label": tier.label,
            "price_monthly_usd": tier.price_monthly_usd,
            "sessions_per_period": tier.sessions_per_period,
            "reports_per_period": tier.reports_per_period,
            "report_model": tier.report_model,
            "retention_enabled": tier.retention_enabled,
        },
        "sessions_used": user.sessions_used_this_period,
        "reports_used": user.reports_used_this_period,
        "period_start": user.period_start,
        "tier_expires_at": user.tier_expires_at,
    })
