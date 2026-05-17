"""Per-session immutable baseline (SPEC_COACH §7.1).

This module is thin: feature extraction lives in ``app.audio.features``
(shared with v0.1), and the immutability constraint is enforced inside
``coach.session.set_baseline`` via the state machine + DB precondition.

What lives here:

- ``BASELINE_FEATURE_KEYS`` — the 4-feature contract Coach uses everywhere.
- ``snapshot_baseline()`` — validates a feature dict, returns clean copy.

By keeping audio extraction outside this module we get two benefits:

1. The route handler can call ``audio.features.extract_all`` once and
   pass the dict here, so Coach doesn't re-decode audio.
2. Tests of baseline + feedback can use synthetic feature dicts without
   spinning up the librosa/numba/parselmouth stack — important locally
   given the Python 3.13 llvmlite ABI issue we've been hitting.
"""

from __future__ import annotations

from ..errors import VoxError


# The 4-feature vector Coach baselines against. Order is informative;
# downstream code (feedback, reports) iterates these keys.
BASELINE_FEATURE_KEYS: tuple[str, ...] = (
    "jitter_local",
    "mfcc_delta_var_mean",
    "spectral_flux_mean",
    "microtremor_envelope",
)


COACH_BASELINE_INVALID = "COACH_BASELINE_INVALID"


def snapshot_baseline(features: dict[str, float]) -> dict[str, float]:
    """Return a clean baseline dict (only the 4 expected keys, all floats).

    Raises ``VoxError`` if any required key is missing or non-finite. This
    is a contract test on the upstream feature extractor — if it ever
    starts returning ``None`` or ``NaN``, we fail loudly rather than
    silently corrupt every subsequent delta computation.
    """
    import math
    missing = [k for k in BASELINE_FEATURE_KEYS if k not in features]
    if missing:
        raise VoxError(
            code=COACH_BASELINE_INVALID,
            message=f"Baseline missing keys: {missing}.",
            http_status=500,
            hint="audio.features.extract_all should always return all 4 keys; check the extractor.",
        )
    clean: dict[str, float] = {}
    for k in BASELINE_FEATURE_KEYS:
        v = features[k]
        if v is None or (isinstance(v, float) and not math.isfinite(v)):
            raise VoxError(
                code=COACH_BASELINE_INVALID,
                message=f"Baseline feature '{k}' is not a finite number (got {v!r}).",
                http_status=500,
                hint="Praat may have failed on this clip; reject calibrate and ask user to re-record.",
            )
        clean[k] = float(v)
    return clean
