"""SQLite connection + migrations for Vox Probabilis.

Design choices (SPEC §3, §9.2):

- WAL journal mode: concurrent reads during writes, which matters when
  uvicorn runs with --workers 2. ``synchronous=NORMAL`` is the
  recommended pairing — it trades a tiny durability window for a large
  throughput gain, acceptable for a session/analytics workload.
- ``check_same_thread=False`` is safe because every connection we hand
  out is used on exactly one request's thread (FastAPI creates a
  dependency-injected connection per request).
- Migrations are applied idempotently at process start: every file
  under ``migrations/`` matching ``NNN_*.sql`` is executed in order.
  There's no migration table yet — ``CREATE ... IF NOT EXISTS`` carries
  the idempotency. v0.2 may want a proper schema-version table.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .config import settings

_MIGRATIONS_DIR = Path(__file__).resolve().parent.parent / "migrations"


def connect() -> sqlite3.Connection:
    """Return a new SQLite connection with production pragmas applied."""
    conn = sqlite3.connect(
        settings.db_path,
        check_same_thread=False,
        isolation_level=None,            # autocommit; we manage transactions explicitly
        timeout=5.0,                     # seconds to wait on a busy lock
    )
    conn.row_factory = sqlite3.Row
    # Pragmas applied per connection (WAL is DB-wide once set but harmless to repeat).
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


@contextmanager
def transaction(conn: sqlite3.Connection) -> Iterator[sqlite3.Connection]:
    """Wrap a block of writes in a transaction — commit on success, rollback on any raise."""
    conn.execute("BEGIN")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def apply_migrations() -> None:
    """Apply every NNN_*.sql migration file in lexical order. Idempotent."""
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)
    migrations = sorted(_MIGRATIONS_DIR.glob("[0-9][0-9][0-9]_*.sql"))
    if not migrations:
        return
    conn = connect()
    try:
        for mig in migrations:
            conn.executescript(mig.read_text(encoding="utf-8"))
    finally:
        conn.close()


def healthcheck() -> None:
    """Raise if the DB is not writable. Used by GET /api/health."""
    conn = connect()
    try:
        conn.execute("CREATE TABLE IF NOT EXISTS _vox_health_probe (id INTEGER)")
        conn.execute("DROP TABLE _vox_health_probe")
    finally:
        conn.close()
