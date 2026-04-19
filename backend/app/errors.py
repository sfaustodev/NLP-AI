"""Typed errors for Vox Probabilis.

Every user-facing failure raises ``VoxError(code=..., message=..., hint=...)``.
A FastAPI exception handler (wired in ``main.py``) converts these into the
JSON shape specified by SPEC §6.5:

    {"error": {"code": "AUDIO_TOO_SHORT",
               "message": "Audio must be at least 3 seconds long...",
               "hint": "Try recording for 5 seconds or longer."}}

Keeping the code table centralised here means every layer (audio loader,
rate limiter, session middleware, API handlers) pulls from the same
vocabulary — no drift, and the frontend can pattern-match on ``error.code``
with confidence.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class VoxError(Exception):
    """Application-level error with a stable machine-readable code."""

    code: str
    message: str
    http_status: int = 400
    hint: str | None = None

    def __str__(self) -> str:  # for logs
        return f"{self.code}: {self.message}"


# ------------------------------------------------------------------ audio
AUDIO_MISSING = dict(code="AUDIO_MISSING",
                     message="No audio file was provided in the request.",
                     http_status=400,
                     hint="Attach a file under the 'audio' form field.")

AUDIO_TOO_LARGE = dict(code="AUDIO_TOO_LARGE",
                       http_status=413,
                       hint="Upload a smaller file; the limit is 10 MB.")

AUDIO_TOO_SHORT = dict(code="AUDIO_TOO_SHORT",
                       http_status=400,
                       hint="Try recording for 5 seconds or longer.")

AUDIO_TOO_LONG = dict(code="AUDIO_TOO_LONG",
                      http_status=400,
                      hint="The signal exceeded 60 seconds after truncation attempts.")

AUDIO_UNSUPPORTED_FORMAT = dict(code="AUDIO_UNSUPPORTED_FORMAT",
                                http_status=400,
                                hint="Allowed formats: WAV, MP3, M4A, OGG, FLAC.")

AUDIO_CORRUPT = dict(code="AUDIO_CORRUPT",
                     http_status=400,
                     hint="The file could not be decoded. Try re-exporting it.")

NO_VOICE_DETECTED = dict(code="NO_VOICE_DETECTED",
                         http_status=400,
                         hint="Record in a quieter environment, closer to the microphone.")

# ------------------------------------------------------------------ flow
RATE_LIMITED = dict(code="RATE_LIMITED",
                    http_status=429,
                    hint="Free tier allows three analyses per 24 hours. Wait or upgrade.")

BASELINE_REQUIRED = dict(code="BASELINE_REQUIRED",
                         http_status=400,
                         hint="Run /api/calibrate with a truth sample before this ritual step.")

RITUAL_ALREADY_USED = dict(code="RITUAL_ALREADY_USED",
                           http_status=400,
                           hint="This ritual freebie was already spent today; try again tomorrow.")

INTERNAL_ERROR = dict(code="INTERNAL_ERROR",
                      message="An internal error occurred. Please try again.",
                      http_status=500,
                      hint=None)


def raise_vox(template: dict, message: str | None = None) -> None:
    """Raise a VoxError from a template above; optionally override message."""
    raise VoxError(
        code=template["code"],
        message=message or template.get("message", template["code"]),
        http_status=template["http_status"],
        hint=template.get("hint"),
    )
