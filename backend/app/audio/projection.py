"""Cartesian projection of deltas into (naturalness, involuntary_stress).

SPEC §8.4-§8.6. The two axes are the product of the project:

    x = naturalness         in [-1, +1]
    y = involuntary_stress  in [-1, +1]

Both are tanh-saturated so an outlier sample cannot blow past the
[-1, +1] box that the frontend renders as the 2x2 Cartesian plane.

Quadrant meaning (SPEC §8.6):

    Q1  (+x, +y)  LIE                — natural-sounding *and* stressed
    Q2  (-x, +y)  DISCOMFORT         — guarded delivery, still stressed
    Q3  (-x, -y)  ROBOTIC            — rehearsed, low stress
    Q4  (+x, -y)  TRUTH              — natural, low stress
    Q0  (|x|<eps, |y|<eps) ORIGIN    — too close to baseline to classify

The ORIGIN epsilon (±0.05) is intentionally generous: a sample within
5% of baseline on *both* axes is indistinguishable from "no signal" at
the sample sizes we work with, and reporting any of the four dramatic
labels there would be overclaiming. The frontend renders ORIGIN as
"baseline confirmed — no deviation".
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# Scale factors applied inside tanh (SPEC §8.4, §8.5). Each was tuned
# against Juan's paper so that a ±50% delta lands near ±0.46 on the
# axis — leaving visible headroom before the tanh asymptote.
NATURALNESS_GAIN = 0.02           # x = tanh(mean_voluntary_pct * 0.02)
INVOLUNTARY_STRESS_GAIN = 0.02    # y = tanh(microtremor_pct     * 0.02)

# Dead-zone half-width around the origin. Samples with |x|<eps AND
# |y|<eps get the ORIGIN label regardless of their sign (SPEC §8.6).
ORIGIN_EPSILON = 0.05

# Quadrant labels — these strings appear verbatim in the API response
# and the frontend switches on them, so do not rename without a
# coordinated frontend change.
QUADRANT_LIE        = "LIE"
QUADRANT_DISCOMFORT = "DISCOMFORT"
QUADRANT_ROBOTIC    = "ROBOTIC"
QUADRANT_TRUTH      = "TRUTH"
QUADRANT_ORIGIN     = "ORIGIN"


@dataclass(frozen=True, slots=True)
class Projection:
    """The SPEC §6.2 ``projection`` object, ready to be serialised."""

    x: float          # naturalness axis, tanh-saturated in [-1, +1]
    y: float          # involuntary-stress axis, tanh-saturated in [-1, +1]
    quadrant: str     # one of the QUADRANT_* constants above

    def as_dict(self) -> dict[str, float | str]:
        return {"x": self.x, "y": self.y, "quadrant": self.quadrant}


def _naturalness_axis(deltas: dict[str, float]) -> float:
    """x axis: mean of the three *voluntary* feature deltas, tanh-scaled.

    Per SPEC §8.4 the voluntary-control bundle is jitter, MFCC-delta
    variance, and spectral flux — features a speaker can plausibly
    modulate by changing how they articulate. A high positive x means
    the delivery is drifting *away* from the baseline in the direction
    of natural, fluent variation; a strongly negative x means the
    speaker is flattening it out (the "robotic" signature).
    """
    voluntary = (
        deltas["jitter_local_pct"]
        + deltas["mfcc_delta_var_pct"]
        + deltas["spectral_flux_pct"]
    ) / 3.0
    return math.tanh(voluntary * NATURALNESS_GAIN)


def _involuntary_stress_axis(deltas: dict[str, float]) -> float:
    """y axis: microtremor delta alone, tanh-scaled.

    SPEC §8.5 — the 8-12 Hz envelope tremor is the one feature a
    speaker cannot suppress consciously; we project it on its own
    axis so the *involuntary* signal stays legible even when the
    voluntary features are deliberately flattened.
    """
    return math.tanh(
        deltas["microtremor_envelope_pct"] * INVOLUNTARY_STRESS_GAIN
    )


def _classify(x: float, y: float) -> str:
    """Assign a quadrant label, with an ORIGIN dead zone per SPEC §8.6."""
    if abs(x) < ORIGIN_EPSILON and abs(y) < ORIGIN_EPSILON:
        return QUADRANT_ORIGIN
    if x >= 0 and y >= 0:
        return QUADRANT_LIE
    if x < 0 and y >= 0:
        return QUADRANT_DISCOMFORT
    if x < 0 and y < 0:
        return QUADRANT_ROBOTIC
    return QUADRANT_TRUTH           # x >= 0 and y < 0


def project(deltas: dict[str, float]) -> Projection:
    """Build a Projection from the four _pct deltas in SPEC §6.2."""
    x = _naturalness_axis(deltas)
    y = _involuntary_stress_axis(deltas)
    return Projection(x=x, y=y, quadrant=_classify(x, y))
