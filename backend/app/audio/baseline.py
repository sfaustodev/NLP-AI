"""Baseline management and delta computation (SPEC §8.1–§8.3).

Two baseline sources:

1. Per-session baseline (primary): established by ``/api/calibrate``,
   stored in the ``sessions`` row. Confidence can reach 'medium' or
   'high' when this path is used.

2. Global fallback baseline (secondary): the hardcoded values below,
   derived from n=3 recordings in the companion paper. Analyses that
   use this source are always labelled ``baseline_source="global"``
   with at best ``confidence="low"``.

The clamp on per-feature delta percentages ([-100, +500]) exists so
that an outlier sample cannot break the Cartesian plot's visual range
in the frontend — tanh saturation would still tame the projection, but
we like the honesty of a bounded raw number in the API response.
"""

from __future__ import annotations

from dataclasses import dataclass

# ----------------------------------------------------------------
# Global fallback baseline — SPEC §8.2.
#
# Derived from Juan's paper: n=3 'truth' condition means, single
# speaker. These are placeholders until a properly-sized population
# sample is collected in v0.2. DO NOT tune these opportunistically;
# any change here must be paired with a paper revision and logged
# in the companion zenodo record.
# ----------------------------------------------------------------
GLOBAL_BASELINE: dict[str, float] = {
    "jitter_local":         0.0182,
    "mfcc_delta_var_mean":  0.0471,
    "spectral_flux_mean":   0.1284,
    "microtremor_envelope": 0.0034,
}

# Outlier clamp for delta percentages (SPEC §8.3).
DELTA_MIN_PCT = -100.0
DELTA_MAX_PCT = 500.0


@dataclass(frozen=True, slots=True)
class Baseline:
    """A resolved baseline with source provenance.

    ``source`` is either ``"session"`` or ``"global"`` and is reported
    verbatim in the API response so the frontend can label confidence.
    """

    values: dict[str, float]
    source: str


def from_session(row_values: dict[str, float | None]) -> Baseline | None:
    """Build a Baseline from a sessions-table row. Returns None if the
    row has no baseline columns populated (calibrate was never run)."""
    keys = ("jitter_local", "mfcc_delta_var_mean",
            "spectral_flux_mean", "microtremor_envelope")
    if any(row_values.get(k) is None for k in keys):
        return None
    return Baseline(
        values={k: float(row_values[k]) for k in keys},   # type: ignore[arg-type]
        source="session",
    )


def global_fallback() -> Baseline:
    """The SPEC §8.2 hardcoded baseline — returned when no per-session
    baseline has been established for the current request."""
    return Baseline(values=dict(GLOBAL_BASELINE), source="global")


def delta_pct(sample: float, baseline: float) -> float:
    """Percentage change clamped to [DELTA_MIN_PCT, DELTA_MAX_PCT].

    Baseline is guaranteed non-zero for our four features (all are
    positive-definite by construction), but we defend against division
    by zero anyway.
    """
    if baseline == 0:
        return 0.0
    raw = ((sample - baseline) / baseline) * 100.0
    return max(DELTA_MIN_PCT, min(DELTA_MAX_PCT, raw))


def compute_deltas(
    sample: dict[str, float | None],
    baseline: Baseline,
) -> dict[str, float]:
    """Compute the four delta percentages for an analysis sample.

    Jitter may be ``None`` when the extractor found fewer than
    JITTER_MIN_PERIODS voiced periods; in that case we report a 0%
    delta and the caller should downgrade confidence.
    """
    out = {}
    for key in ("jitter_local", "mfcc_delta_var_mean",
                "spectral_flux_mean", "microtremor_envelope"):
        sample_val = sample.get(key)
        if sample_val is None:
            out[f"{key}_pct" if key != "microtremor_envelope" else "microtremor_envelope_pct"] = 0.0
            continue
        out[_delta_key(key)] = delta_pct(float(sample_val), baseline.values[key])
    return out


def _delta_key(feature_key: str) -> str:
    """Map a feature key to the matching _pct key used in the API response.

    Per SPEC §6.2 the API uses 'mfcc_delta_var_pct' (not
    'mfcc_delta_var_mean_pct') and 'spectral_flux_pct' (not
    'spectral_flux_mean_pct'). Encode that mapping once here.
    """
    return {
        "jitter_local":         "jitter_local_pct",
        "mfcc_delta_var_mean":  "mfcc_delta_var_pct",
        "spectral_flux_mean":   "spectral_flux_pct",
        "microtremor_envelope": "microtremor_envelope_pct",
    }[feature_key]
