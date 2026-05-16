-- VOX-COACH-B · lawyer accounts + tier activation.
-- Minimal PII: email only (LGPD justified — needed to deliver service).
-- Tier activation is manual via app/coach/cli.py (no checkout in VOX-COACH-B; Lemon Squeezy/Stripe in VOX-COACH-D).
-- period_start advances every period_days (30 by default) on first request of new period.
CREATE TABLE IF NOT EXISTS coach_users_tiers (
    id TEXT PRIMARY KEY,                  -- 'usr_' + ulid
    email TEXT UNIQUE NOT NULL,
    activation_token TEXT UNIQUE,         -- magic-link token, NULL after consumed
    activation_token_expires_at INTEGER,  -- 7d TTL
    tier_key TEXT NOT NULL,               -- FREE_TRIAL|TIER_1_MONTHLY|TIER_2_MONTHLY|TIER_3_MONTHLY
    tier_activated_at INTEGER NOT NULL,
    tier_expires_at INTEGER NOT NULL,
    sessions_used_this_period INTEGER NOT NULL DEFAULT 0,
    reports_used_this_period INTEGER NOT NULL DEFAULT 0,
    period_start INTEGER NOT NULL,
    created_at INTEGER NOT NULL,
    last_seen_at INTEGER NOT NULL,
    deleted_at INTEGER                    -- soft delete for LGPD right-to-erasure
);
CREATE INDEX IF NOT EXISTS idx_coach_users_email ON coach_users_tiers(email);
CREATE INDEX IF NOT EXISTS idx_coach_users_activation ON coach_users_tiers(activation_token);
