"""POST /api/analyze — the core analysis endpoint (SPEC §6.2).

Flow (aligned with SPEC §11.2 fail-fast validation order):

    1. Session middleware has already resolved/minted the cookie.
    2. Check quota / ritual-freebie eligibility BEFORE decoding.
    3. Decode audio (raises on bad format / bad duration / silence).
    4. Extract features.
    5. Resolve baseline (per-session if calibrated, else global).
    6. Compute deltas + cartesian projection.
    7. Decide confidence level from baseline source + sample quality.
    8. Record the analysis row (so quota ticks down).
    9. Optionally stash the 4 features in the anonymous dataset.
    10. Return the SPEC §6.2 response shape.

Step 2 happens before step 3 because decoding is the expensive part
(~100-300ms for a 60s file); a rate-limited user should get a 429
in single-digit milliseconds.
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..audio import baseline as _baseline
from ..audio import features as _features
from ..audio import projection as _projection
from ..audio.load import decode
from ..db import connect
from ..errors import (
    AUDIO_CORRUPT,
    AUDIO_MISSING,
    BASELINE_REQUIRED,
    RATE_LIMITED,
    RITUAL_ALREADY_USED,
    raise_vox,
)
from ..rate_limit import (
    RITUAL_STEP_AI_BONUS,
    RITUAL_STEP_LIE,
    RITUAL_STEP_UNCERTAIN,
    check_quota,
    is_ritual_freebie,
    record_analysis,
)
from ..sessions import Session, baseline_from_row, fetch_row, get_session

log = logging.getLogger("vox.api.analyze")
router = APIRouter()

_VALID_RITUAL_STEPS = frozenset({
    RITUAL_STEP_UNCERTAIN, RITUAL_STEP_LIE, RITUAL_STEP_AI_BONUS,
})

METHODOLOGY_NOTE = (
    "Features are computed per the methods described in "
    "DOI 10.5281/zenodo.19396809. The deception signature was validated "
    "on n=3 recordings from a single speaker. Generalization to your "
    "voice is hypothesized but not proven."
)


def _pick_confidence(
    baseline_source: str,
    voiced_ratio: float,
    duration_s: float,
    jitter_missing: bool,
) -> tuple[str, str]:
    """Map quality signals to (confidence, human-readable reason).

    SPEC §6.2 enum: high | medium | low | unreliable.
    """
    if jitter_missing or voiced_ratio < 0.2 or duration_s < 3.0:
        return (
            "unreliable",
            "Could not extract a stable pitch contour or voiced content. "
            "Treat this result as indicative only.",
        )
    if baseline_source == "global" or voiced_ratio < 0.4:
        return (
            "low",
            "Global fallback baseline used — calibrate with a truth sample "
            "first for a higher-confidence reading."
            if baseline_source == "global"
            else "Voiced frame ratio was low; result is directional only.",
        )
    if voiced_ratio >= 0.75 and duration_s >= 5.0:
        return (
            "high",
            "Per-session baseline and high-quality sample.",
        )
    return (
        "medium",
        "Per-session baseline from calibration used.",
    )


def _store_optin(features: dict[str, float | None], ritual_step: str | None,
                 quadrant: str) -> None:
    """Anonymous research dataset write (SPEC §10.2). No session/IP link."""
    # If any feature is missing we can't store a useful row.
    if any(features[k] is None for k in features):
        return
    now = int(time.time())
    conn = connect()
    try:
        conn.execute(
            "INSERT INTO dataset_optins "
            "  (created_at, jitter_local, mfcc_delta_var_mean, "
            "   spectral_flux_mean, microtremor_envelope, "
            "   ritual_step, quadrant) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                now,
                float(features["jitter_local"]),                # type: ignore[arg-type]
                float(features["mfcc_delta_var_mean"]),         # type: ignore[arg-type]
                float(features["spectral_flux_mean"]),          # type: ignore[arg-type]
                float(features["microtremor_envelope"]),        # type: ignore[arg-type]
                ritual_step,
                quadrant,
            ),
        )
    finally:
        conn.close()


@router.post("/api/analyze")
async def analyze(
    audio: UploadFile = File(...),
    ritual_step: str | None = Form(None),
    opt_in_dataset: bool = Form(False),
    session: Session = Depends(get_session),
) -> dict:
    """SPEC §6.2 full response, including projection, quota, and confidence."""
    if audio is None or audio.filename is None:
        raise_vox(AUDIO_MISSING)

    # Normalise ritual_step: empty-string from a form checkbox should
    # be treated as "no ritual step", not as an unknown step name.
    if ritual_step is not None:
        ritual_step = ritual_step.strip() or None
    if ritual_step is not None and ritual_step not in _VALID_RITUAL_STEPS:
        # Unknown values are silently demoted to "no ritual step" —
        # easier on a misbehaving frontend than a hard 400.
        log.info("unknown ritual_step=%r — treating as none", ritual_step)
        ritual_step = None

    # ---- step 2: cheap quota/eligibility checks before expensive decode
    row = fetch_row(session.session_id)
    has_baseline = bool(row and row["baseline_established_at"])

    # Freebie ritual steps require a baseline (SPEC §6.2, §11.1).
    if ritual_step in (RITUAL_STEP_UNCERTAIN, RITUAL_STEP_LIE) and not has_baseline:
        raise_vox(BASELINE_REQUIRED)

    freebie = is_ritual_freebie(session.session_id, ritual_step, has_baseline)

    # If they're trying to use a ritual-freebie step that's already
    # been spent today, tell them explicitly rather than burning quota.
    if (
        ritual_step in (RITUAL_STEP_UNCERTAIN, RITUAL_STEP_LIE)
        and has_baseline
        and not freebie
    ):
        raise_vox(RITUAL_ALREADY_USED)

    quota = check_quota(session.session_id)
    counted = not freebie
    if counted and quota.remaining_today <= 0:
        raise_vox(RATE_LIMITED)

    # ---- step 3-4: decode + extract
    raw = await audio.read()
    loaded = decode(raw)
    try:
        feats = _features.extract_all(loaded.samples)
    except Exception as exc:
        log.warning("feature extraction failed on analyze: %s", exc)
        raise_vox(AUDIO_CORRUPT)
        return {}   # unreachable

    features_dict = feats.as_dict()   # jitter may be None here

    # ---- step 5: resolve baseline
    if row is not None:
        maybe_session_baseline = _baseline.from_session(
            dict(baseline_from_row(row))
        )
        baseline = maybe_session_baseline or _baseline.global_fallback()
    else:
        baseline = _baseline.global_fallback()

    # ---- step 6: deltas + projection
    deltas = _baseline.compute_deltas(features_dict, baseline)
    proj = _projection.project(deltas)

    # ---- step 7: confidence
    conf, conf_reason = _pick_confidence(
        baseline_source=baseline.source,
        voiced_ratio=loaded.voiced_frame_ratio,
        duration_s=loaded.duration_s,
        jitter_missing=feats.jitter_local is None,
    )

    # ---- step 8: record analysis (updates quota)
    record_analysis(
        session_id=session.session_id,
        ritual_step=ritual_step,
        counted_against_quota=counted,
        quadrant=proj.quadrant,
    )

    # Re-read quota so the response reflects *after* this call.
    post_quota = check_quota(session.session_id)

    # ---- step 9: anonymous dataset (if opted in)
    if opt_in_dataset:
        _store_optin(features_dict, ritual_step, proj.quadrant)

    # ---- step 10: SPEC §6.2 response
    return {
        "features":           features_dict,
        "deltas":             deltas,
        "projection":         proj.as_dict(),
        "confidence":         conf,
        "confidence_reason":  conf_reason,
        "baseline_source":    baseline.source,
        "sample_duration_s":  loaded.duration_s,
        "voiced_frame_ratio": loaded.voiced_frame_ratio,
        "quota": {
            "remaining_today": post_quota.remaining_today,
            "resets_at":       post_quota.resets_at_iso,
        },
        "methodology_note":   METHODOLOGY_NOTE,
    }
