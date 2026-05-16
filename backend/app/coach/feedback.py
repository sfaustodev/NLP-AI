"""Per-response feedback: deltas + consistency label + cartesian projection.

Inputs (always pure feature dicts — no audio decoding here):

- ``current_features``: 4-feature dict from the response audio.
- ``baseline_features``: 4-feature dict frozen at calibration time.

Outputs:

- ``delta_pct``: dict mapping each feature → percentage change vs baseline,
  clamped to ``[-100, +500]`` per v0.1 SPEC §8.3 (prevents outliers
  blowing up the UI).
- ``consistency_label``: one of ``BASELINE / SLIGHT_SHIFT / NOTABLE_SHIFT /
  MAJOR_SHIFT``, derived from the max absolute delta (SPEC_COACH §7.2).
- ``color``: one of ``GREEN / YELLOW / ORANGE / RED``.
- ``cartesian (x, y)``: SPEC_COACH §10.2 — reuses v0.1 §8.4/§8.5 mapping:
  X = mean of (jitter, mfcc_delta_var, spectral_flux) deltas, tanh-mapped;
  Y = microtremor delta, tanh-mapped. Both ∈ [-1, +1].

A short server-side ``narrative`` line is also emitted — template-based,
NOT LLM. The LLM narrative happens once at session end (Sonnet report).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .baseline import BASELINE_FEATURE_KEYS


# SPEC_COACH §7.2 thresholds.
LABEL_THRESHOLDS: tuple[tuple[float, str, str], ...] = (
    (10.0, "BASELINE",      "GREEN"),
    (20.0, "SLIGHT_SHIFT",  "YELLOW"),
    (35.0, "NOTABLE_SHIFT", "ORANGE"),
    # else MAJOR_SHIFT / RED
)


# v0.1 SPEC §8.3 — clamp delta_pct to prevent UI breakage from outliers.
DELTA_CLAMP_LOW  = -100.0
DELTA_CLAMP_HIGH = 500.0

# v0.1 SPEC §8.4 — tanh sharpness factor.
TANH_SHARPNESS = 2.0


@dataclass(frozen=True, slots=True)
class ResponseFeedback:
    delta_pct: dict[str, float]
    cartesian_x: float
    cartesian_y: float
    consistency_label: str
    color: str
    narrative: str


# ------------------------------------------------------------------ pure helpers

def compute_delta_pct(current: dict[str, float],
                       baseline: dict[str, float]) -> dict[str, float]:
    """Per-feature percentage change vs baseline, clamped to [-100, +500]."""
    out: dict[str, float] = {}
    for key in BASELINE_FEATURE_KEYS:
        b = baseline[key]
        c = current[key]
        if b == 0:
            # Avoid div-by-zero; treat as no-baseline → max positive shift.
            pct = DELTA_CLAMP_HIGH if c > 0 else 0.0
        else:
            pct = ((c - b) / abs(b)) * 100.0
        out[key] = float(max(DELTA_CLAMP_LOW, min(DELTA_CLAMP_HIGH, pct)))
    return out


def consistency_label(delta_pct: dict[str, float]) -> tuple[str, str]:
    """SPEC_COACH §7.2 — label + color from max-abs delta across features."""
    max_abs = max(abs(v) for v in delta_pct.values())
    for threshold, label, color in LABEL_THRESHOLDS:
        if max_abs < threshold:
            return label, color
    return "MAJOR_SHIFT", "RED"


def cartesian_xy(delta_pct: dict[str, float]) -> tuple[float, float]:
    """v0.1 §8.4/§8.5 projection adapted for Coach.

    X = mean of voluntary-variation deltas (jitter + mfcc_delta_var +
    spectral_flux) / 100, tanh-mapped × sharpness factor 2.
    Y = microtremor delta / 100, tanh-mapped × sharpness factor 2.

    Positive X means more "natural" (voluntary variation elevated);
    negative X means over-control. Positive Y means involuntary stress
    elevated; negative Y means calmer than baseline.
    """
    voluntary_keys = ("jitter_local", "mfcc_delta_var_mean", "spectral_flux_mean")
    voluntary_mean = float(np.mean([delta_pct[k] for k in voluntary_keys])) / 100.0
    microtremor = float(delta_pct["microtremor_envelope"]) / 100.0
    x = float(math.tanh(voluntary_mean * TANH_SHARPNESS))
    y = float(math.tanh(microtremor * TANH_SHARPNESS))
    return x, y


def short_narrative(label: str, delta_pct: dict[str, float]) -> str:
    """Single-line pt-BR template — no LLM. Highlight the most-shifted feature."""
    if label == "BASELINE":
        return "Resposta dentro do baseline — voz consistente com a calibração."
    biggest_key = max(delta_pct.keys(), key=lambda k: abs(delta_pct[k]))
    biggest_val = delta_pct[biggest_key]
    pretty = {
        "jitter_local":         "jitter",
        "mfcc_delta_var_mean":  "variação espectral (MFCC delta)",
        "spectral_flux_mean":   "fluxo espectral",
        "microtremor_envelope": "microtremor (8-12 Hz)",
    }[biggest_key]
    direction = "elevado" if biggest_val > 0 else "reduzido"
    return (f"Deslocamento {label.lower().replace('_', ' ')} — {pretty} "
            f"{direction} em {abs(biggest_val):.0f}% versus baseline.")


# ------------------------------------------------------------------ orchestrator

def compute_feedback(*, current_features: dict[str, float],
                      baseline_features: dict[str, float]) -> ResponseFeedback:
    """All-in-one: deltas + cartesian + label + narrative."""
    deltas = compute_delta_pct(current_features, baseline_features)
    x, y = cartesian_xy(deltas)
    label, color = consistency_label(deltas)
    narrative = short_narrative(label, deltas)
    return ResponseFeedback(
        delta_pct=deltas,
        cartesian_x=x,
        cartesian_y=y,
        consistency_label=label,
        color=color,
        narrative=narrative,
    )
