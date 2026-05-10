"""Vox Probabilis backend — FastAPI app package."""

import os

__version__ = "0.1.0"

# DEPLOY.md §4 — flip via VOX_LIVENESS_MODE without code surgery.
# Unknown values fall back to "off" so a typo never silently exposes a
# half-baked signal in the API contract.
LIVENESS_MODE = os.environ.get("VOX_LIVENESS_MODE", "off").strip().lower()
if LIVENESS_MODE not in ("off", "boolean", "full"):
    LIVENESS_MODE = "off"
