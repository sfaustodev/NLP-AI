"""Environment loading for Vox Probabilis.

One-shot: read ``os.environ`` at import time, freeze into ``settings``,
never read the environment again. Every other module imports ``settings``
rather than re-reading ``os.environ`` to keep config lookup centralised
and testable.

Missing ``VOX_SECRET_SALT`` raises at startup rather than silently using
a default — a salt is required to make the IP hash non-reversible. No
default can ever be safe here, so we refuse to start.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path


def _get(name: str, default: str | None = None) -> str:
    """Read env var; fall back to ``default`` if unset or empty."""
    value = os.environ.get(name, "").strip()
    if value:
        return value
    if default is None:
        print(f"[vox] FATAL: environment variable {name} is required", file=sys.stderr)
        raise SystemExit(2)
    return default


def _get_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name, "").strip().lower()
    if not raw:
        return default
    return raw in ("1", "true", "yes", "y", "on")


def _get_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    return int(raw)


@dataclass(frozen=True, slots=True)
class Settings:
    # Security
    secret_salt: str

    # Storage
    db_path: Path
    static_dir: Path

    # Behaviour
    log_level: str
    cookie_secure: bool
    max_upload_mb: int
    free_daily_quota: int
    cors_origins: tuple[str, ...]


def _load() -> Settings:
    salt = _get("VOX_SECRET_SALT")
    if salt == "CHANGE_ME_TO_32_RANDOM_BYTES_BASE64":
        print(
            "[vox] FATAL: VOX_SECRET_SALT is still the placeholder. "
            "Generate one with `python -c 'import secrets; print(secrets.token_urlsafe(32))'`",
            file=sys.stderr,
        )
        raise SystemExit(2)

    origins_raw = os.environ.get("VOX_CORS_ORIGINS", "").strip()
    origins = tuple(o.strip() for o in origins_raw.split(",") if o.strip()) if origins_raw else ()

    return Settings(
        secret_salt=salt,
        db_path=Path(_get("VOX_DB_PATH", "./vox.db")).expanduser().resolve(),
        static_dir=Path(_get("VOX_STATIC_DIR", "../landing_page")).expanduser().resolve(),
        log_level=_get("VOX_LOG_LEVEL", "INFO").upper(),
        cookie_secure=_get_bool("VOX_COOKIE_SECURE", False),
        max_upload_mb=_get_int("VOX_MAX_UPLOAD_MB", 10),
        free_daily_quota=_get_int("VOX_FREE_DAILY_QUOTA", 3),
        cors_origins=origins,
    )


settings: Settings = _load()
