"""Cartesian projection of deltas into (naturalness, involuntary_stress).

SPEC §8.4-§8.6. The two axes are the product of the project:

    naturalness         in [-1, +1]   (voluntary-control signal)
    involuntary_stress  in [-1, +1]   (microtremor signal)

Both are tanh-saturated so an outlier sample cannot blow past the
[-1, +1] box that the frontend renders as a 2x2 Cartesian plane.

Quadrant meaning (SPEC §8.6):

    naturalness<0, stress>0   OVER_CONTROLLED_TENSE   — deception signature
    naturalness>0, stress>0   NATURAL_STRESSED        — aroused but unguarded
    naturalness<0, stress<0   OVER_CONTROLLED_CALM    — rehearsed, flat delivery
    naturalness>0, stress<0   NATURAL_CALM            — the truth-condition corner
    |both| < EPSILON          ORIGIN                  — too close to baseline to classify

The ORIGIN epsilon (±0.05) is intentionally generous: a sample within
5% of baseline on *both* axes is indistinguishable from "no signal" at
the sample sizes we work with, and reporting any of the four dramatic
labels there would be overclaiming.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# SPEC §8.4/§8.5 sharpness factor. The math is identical to the spec's
# ``tanh((mean_pct / 100) * 2)``: dividing by 100 converts percent to
# fraction, and the *2 is the "sharpness" that maps ±50% onto roughly
# ±0.76 on the axis — leaving visible headroom before the tanh asymptote.
AXIS_SHARPNESS = 2.0 / 100.0

# Dead-zone half-width around the origin (SPEC §8.6). Both axes must
# be within ±EPSILON for the ORIGIN label.
ORIGIN_EPSILON = 0.05

# Quadrant labels — these strings appear verbatim in the API response
# and the frontend switches on them, so do not rename without a
# coordinated frontend change.
QUADRANT_OVER_CONTROLLED_TENSE = "OVER_CONTROLLED_TENSE"
QUADRANT_NATURAL_STRESSED      = "NATURAL_STRESSED"
QUADRANT_OVER_CONTROLLED_CALM  = "OVER_CONTROLLED_CALM"
QUADRANT_NATURAL_CALM          = "NATURAL_CALM"
QUADRANT_ORIGIN                = "ORIGIN"


@dataclass(frozen=True, slots=True)
class Projection:
    """The SPEC §6.2 ``projection`` object, ready to be serialised."""

    naturalness: float          # voluntary axis, tanh-saturated in [-1, +1]
    involuntary_stress: float   # microtremor axis, tanh-saturated in [-1, +1]
    quadrant: str               # one of the QUADRANT_* constants above

    def as_dict(self) -> dict[str, float | str]:
        return {
            "naturalness":        self.naturalness,
            "involuntary_stress": self.involuntary_stress,
            "quadrant":           self.quadrant,
        }


def _naturalness(deltas: dict[str, float]) -> float:
    """Mean of the three *voluntary* feature deltas, tanh-scaled.

    Per SPEC §8.4 the voluntary-control bundle is jitter, MFCC-delta
    variance, and spectral flux — features a speaker can plausibly
    modulate by changing how they articulate. Positive means the
    delivery is drifting toward natural, fluent variation; strongly
    negative means the speaker is flattening it out (over-controlled).
    """
    mean_pct = (
        deltas["jitter_local_pct"]
        + deltas["mfcc_delta_var_pct"]
        + deltas["spectral_flux_pct"]
    ) / 3.0
    return math.tanh(mean_pct * AXIS_SHARPNESS)


def _involuntary_stress(deltas: dict[str, float]) -> float:
    """Microtremor delta alone, tanh-scaled.

    SPEC §8.5 — the 8-12 Hz envelope tremor is the one feature a
    speaker cannot suppress consciously; isolating it on its own axis
    keeps the involuntary signal legible even when the voluntary
    features are deliberately flattened.
    """
    return math.tanh(deltas["microtremor_envelope_pct"] * AXIS_SHARPNESS)


def _classify(naturalness: float, stress: float) -> str:
    """Assign a quadrant label, with an ORIGIN dead zone per SPEC §8.6."""
    if abs(naturalness) < ORIGIN_EPSILON and abs(stress) < ORIGIN_EPSILON:
        return QUADRANT_ORIGIN
    over_ctrl = naturalness < 0
    tense = stress > 0
    if over_ctrl and tense:
        return QUADRANT_OVER_CONTROLLED_TENSE
    if not over_ctrl and tense:
        return QUADRANT_NATURAL_STRESSED
    if over_ctrl and not tense:
        return QUADRANT_OVER_CONTROLLED_CALM
    return QUADRANT_NATURAL_CALM


def project(deltas: dict[str, float]) -> Projection:
    """Build a Projection from the four _pct deltas in SPEC §6.2."""
    nat = _naturalness(deltas)
    stress = _involuntary_stress(deltas)
    return Projection(
        naturalness=nat,
        involuntary_stress=stress,
        quadrant=_classify(nat, stress),
    )
