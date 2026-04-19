"""POST /api/calibrate — establish the per-session baseline (SPEC §6.1).

This is the first step of the 3-sample ritual. The user uploads a
voice sample of themselves saying something true; we extract the
four features and store them on the sessions row. Later analyses
compute deltas against this personal baseline instead of the global
n=3 fallback (SPEC §8.1 vs §8.2).

Does not count against the daily quota — it *enables* the quota-free
ritual steps, so gating it behind quota would be a dead-end funnel.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, File, Form, UploadFile

from ..audio import features as _features
from ..audio.load import decode
from ..errors import raise_vox
from ..errors import AUDIO_CORRUPT, AUDIO_MISSING
from ..sessions import Session, get_session, write_baseline

log = logging.getLogger("vox.api.calibrate")
router = APIRouter()


@router.post("/api/calibrate")
async def calibrate(
    audio: UploadFile = File(..., description="Truth-condition voice sample"),
    label: str = Form("truth"),   # noqa: ARG001 — accepted for future multi-label calibrations
    session: Session = Depends(get_session),
) -> dict:
    """SPEC §6.1 response: ``baseline_established=true`` on success."""
    if audio is None or audio.filename is None:
        raise_vox(AUDIO_MISSING)

    raw = await audio.read()
    loaded = decode(raw)                      # raises VoxError on any decode issue

    try:
        feats = _features.extract_all(loaded.samples)
    except Exception as exc:
        # Feature extractors are supposed to swallow their own failures
        # and return None — if one raises, treat it as AUDIO_CORRUPT so
        # the frontend gets a clean error instead of a 500.
        log.warning("feature extraction failed on calibrate: %s", exc)
        raise_vox(AUDIO_CORRUPT)
        return {}   # unreachable, satisfies type checker

    # Jitter can come back None on degenerate signals (SPEC §7.2). For
    # calibration we don't want that — without jitter, downstream
    # analyses would be computing a meaningless delta. Reject cleanly.
    if feats.jitter_local is None:
        raise_vox(
            AUDIO_CORRUPT,
            "Could not extract a stable pitch contour from this sample. "
            "Record somewhere quieter and try again.",
        )

    feat_dict = {
        "jitter_local":         feats.jitter_local,
        "mfcc_delta_var_mean":  feats.mfcc_delta_var_mean,
        "spectral_flux_mean":   feats.spectral_flux_mean,
        "microtremor_envelope": feats.microtremor_envelope,
    }
    write_baseline(session.session_id, feat_dict)

    return {
        "session_id":          session.session_id,
        "baseline_established": True,
        "baseline":             feat_dict,
        "sample_duration_s":    loaded.duration_s,
        "voiced_frame_ratio":   loaded.voiced_frame_ratio,
    }
