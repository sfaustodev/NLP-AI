-- Vox Probabilis · initial schema (SPEC §9.2)
-- Applied idempotently on startup; safe to run multiple times.

CREATE TABLE IF NOT EXISTS sessions (
    session_id              TEXT PRIMARY KEY,
    created_at              INTEGER NOT NULL,   -- unix timestamp (seconds)
    last_seen_at            INTEGER NOT NULL,
    ip_hash                 TEXT    NOT NULL,   -- sha256(ip + SALT)[:16]
    baseline_jitter         REAL,
    baseline_mfcc_delta_var REAL,
    baseline_spectral_flux  REAL,
    baseline_microtremor    REAL,
    baseline_established_at INTEGER,
    ritual_uncertain_done_at INTEGER,           -- one freebie per UTC day
    ritual_lie_done_at      INTEGER
);

CREATE TABLE IF NOT EXISTS analyses (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id              TEXT    NOT NULL,
    created_at              INTEGER NOT NULL,
    day_bucket              TEXT    NOT NULL,   -- 'YYYY-MM-DD' in UTC
    ritual_step             TEXT,               -- uncertain / lie / ai_bonus / NULL
    counted_against_quota   INTEGER NOT NULL,   -- 0 or 1
    quadrant                TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_session_day
    ON analyses(session_id, day_bucket);

-- Opt-in anonymous research dataset: four feature values only, no link
-- back to session/IP. Kept indefinitely per SPEC §10.3.
CREATE TABLE IF NOT EXISTS dataset_optins (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at              INTEGER NOT NULL,
    jitter_local            REAL    NOT NULL,
    mfcc_delta_var_mean     REAL    NOT NULL,
    spectral_flux_mean      REAL    NOT NULL,
    microtremor_envelope    REAL    NOT NULL,
    ritual_step             TEXT,
    quadrant                TEXT    NOT NULL
);
