-- VOX-COACH-B · session lifecycle table.
-- state machine: CREATED → AWAITING_CALIBRATION → READY → IN_PRACTICE ⟲ → ENDED
-- session_token is HMAC-signed opaque (see app/coach/auth.py), 1h TTL,
-- bound to owner_user_id so a leaked token can't escalate to another lawyer.
CREATE TABLE IF NOT EXISTS coach_sessions (
    id TEXT PRIMARY KEY,                  -- 'ses_' + ulid
    session_token TEXT UNIQUE NOT NULL,   -- HMAC-signed opaque
    owner_user_id TEXT NOT NULL,          -- FK coach_users_tiers.id
    session_name TEXT NOT NULL,
    state TEXT NOT NULL,                  -- CREATED|AWAITING_CALIBRATION|READY|IN_PRACTICE|ENDED
    baseline_features TEXT,               -- JSON {jitter_local, mfcc_delta_var_mean, spectral_flux_mean, microtremor_envelope}
    mic_quality_label TEXT,               -- GREEN|YELLOW|RED
    mic_quality_snr_db REAL,
    planned_questions_json TEXT,
    report_html TEXT,                     -- cached Sonnet narrative after END
    report_generated_at INTEGER,
    created_at INTEGER NOT NULL,          -- unix ts
    expires_at INTEGER NOT NULL,          -- session token TTL
    ended_at INTEGER,
    deleted_at INTEGER                    -- soft delete for LGPD right-to-erasure
);
CREATE INDEX IF NOT EXISTS idx_coach_sess_owner ON coach_sessions(owner_user_id);
CREATE INDEX IF NOT EXISTS idx_coach_sess_token ON coach_sessions(session_token);
CREATE INDEX IF NOT EXISTS idx_coach_sess_state ON coach_sessions(state);
