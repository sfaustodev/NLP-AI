-- VOX-COACH-B · per-response features + cartesian + consistency label.
-- Never stores raw audio (LGPD §10 + SPEC §8.3 — audio deleted <60s after extraction).
-- delta_pct_json is the 4-feature delta vs session baseline.
CREATE TABLE IF NOT EXISTS coach_responses (
    id TEXT PRIMARY KEY,                  -- 'rsp_' + ulid
    session_id TEXT NOT NULL,             -- FK coach_sessions.id
    response_index INTEGER NOT NULL,      -- 1-based within session
    question_text TEXT,                   -- lawyer's typed question (optional, plain text)
    duration_s REAL NOT NULL,
    features_json TEXT NOT NULL,          -- JSON {jitter_local, mfcc_delta_var_mean, spectral_flux_mean, microtremor_envelope}
    delta_pct_json TEXT NOT NULL,         -- JSON {<feature>: pct_change}
    cartesian_x REAL,
    cartesian_y REAL,
    consistency_label TEXT NOT NULL,      -- BASELINE|SLIGHT_SHIFT|NOTABLE_SHIFT|MAJOR_SHIFT
    color TEXT NOT NULL,                  -- GREEN|YELLOW|ORANGE|RED
    narrative TEXT,                       -- short server-side template line, not LLM
    created_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES coach_sessions(id)
);
CREATE INDEX IF NOT EXISTS idx_coach_resp_session ON coach_responses(session_id);
CREATE INDEX IF NOT EXISTS idx_coach_resp_session_idx ON coach_responses(session_id, response_index);
