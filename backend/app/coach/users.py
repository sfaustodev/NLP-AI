"""Lawyer accounts + tier activation (coach_users_tiers CRUD).

The flow (sem checkout em VOX-COACH-B):

1. Faustão roda ``cli upgrade --email adv@example.com --tier FREE_TRIAL``.
2. ``create_or_upgrade()`` insere/atualiza a row, gera activation_token.
3. Faustão envia link ``https://voxprobabilis.com/coach/activate?token=...``.
4. Adv clica → ``consume_activation_token()`` retorna user_id, agente seta
   cookie ``coach_session`` HMAC-signed (auth.gen_lawyer_cookie_token).
5. Subsequent calls → ``get_user_by_id()`` via cookie.

Quota counters live aqui e são atualizados em transação no mesmo lugar onde
a ação acontece (route handler de session create / report generate).
``maybe_reset_period()`` deve ser chamado no início de cada request pra
zerar contadores quando period_start + period_days <= now.

LGPD: ``soft_delete_user`` apaga email + tokens mas mantém row pra auditoria.
"""

from __future__ import annotations

import secrets
import time
from dataclasses import dataclass
from typing import Optional

from ..db import connect, transaction
from ..errors import VoxError
from . import auth
from .pricing import get_tier, PRICING


COACH_USER_NOT_FOUND      = "COACH_USER_NOT_FOUND"
COACH_USER_DELETED        = "COACH_USER_DELETED"
COACH_USER_EMAIL_TAKEN    = "COACH_USER_EMAIL_TAKEN"
COACH_ACTIVATION_INVALID  = "COACH_ACTIVATION_INVALID"
COACH_ACTIVATION_EXPIRED  = "COACH_ACTIVATION_EXPIRED"


@dataclass(frozen=True, slots=True)
class CoachUser:
    id: str
    email: str
    activation_token: Optional[str]
    activation_token_expires_at: Optional[int]
    tier_key: str
    tier_activated_at: int
    tier_expires_at: int
    sessions_used_this_period: int
    reports_used_this_period: int
    period_start: int
    created_at: int
    last_seen_at: int
    deleted_at: Optional[int]


def _gen_user_id() -> str:
    return f"usr_{secrets.token_urlsafe(16)}"


def _row_to_user(row) -> CoachUser:
    return CoachUser(
        id=row["id"],
        email=row["email"],
        activation_token=row["activation_token"],
        activation_token_expires_at=row["activation_token_expires_at"],
        tier_key=row["tier_key"],
        tier_activated_at=row["tier_activated_at"],
        tier_expires_at=row["tier_expires_at"],
        sessions_used_this_period=row["sessions_used_this_period"],
        reports_used_this_period=row["reports_used_this_period"],
        period_start=row["period_start"],
        created_at=row["created_at"],
        last_seen_at=row["last_seen_at"],
        deleted_at=row["deleted_at"],
    )


# ------------------------------------------------------------------ CRUD

def create_or_upgrade(*, email: str, tier_key: str,
                      now: int | None = None) -> CoachUser:
    """Upsert by email. If user exists, upgrade tier + reset period + new
    activation token. Otherwise insert fresh.
    """
    now = now if now is not None else int(time.time())
    tier = get_tier(tier_key)
    activation_tok = auth.gen_activation_token()
    activation_expires = now + auth.ACTIVATION_TOKEN_TTL_SECONDS
    tier_expires = now + tier.period_days * 86_400

    conn = connect()
    try:
        existing = conn.execute(
            "SELECT id FROM coach_users_tiers WHERE email = ? AND deleted_at IS NULL",
            (email,),
        ).fetchone()
        with transaction(conn):
            if existing:
                conn.execute(
                    """
                    UPDATE coach_users_tiers
                    SET activation_token = ?, activation_token_expires_at = ?,
                        tier_key = ?, tier_activated_at = ?, tier_expires_at = ?,
                        sessions_used_this_period = 0, reports_used_this_period = 0,
                        period_start = ?, last_seen_at = ?
                    WHERE id = ?
                    """,
                    (activation_tok, activation_expires, tier.key, now,
                     tier_expires, now, now, existing["id"]),
                )
                uid = existing["id"]
            else:
                uid = _gen_user_id()
                conn.execute(
                    """
                    INSERT INTO coach_users_tiers (
                        id, email, activation_token, activation_token_expires_at,
                        tier_key, tier_activated_at, tier_expires_at,
                        sessions_used_this_period, reports_used_this_period,
                        period_start, created_at, last_seen_at, deleted_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, 0, 0, ?, ?, ?, NULL)
                    """,
                    (uid, email, activation_tok, activation_expires,
                     tier.key, now, tier_expires, now, now, now),
                )
        row = conn.execute(
            "SELECT * FROM coach_users_tiers WHERE id = ?", (uid,),
        ).fetchone()
        return _row_to_user(row)
    finally:
        conn.close()


def get_user_by_id(user_id: str) -> CoachUser:
    """Internal lookup by primary key (used after cookie verification)."""
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM coach_users_tiers WHERE id = ?", (user_id,),
        ).fetchone()
        if row is None:
            raise VoxError(code=COACH_USER_NOT_FOUND, message="User not found.",
                           http_status=404)
        if row["deleted_at"] is not None:
            raise VoxError(code=COACH_USER_DELETED, message="User account deleted.",
                           http_status=410)
        return _row_to_user(row)
    finally:
        conn.close()


def get_user_by_email(email: str) -> CoachUser:
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM coach_users_tiers WHERE email = ? AND deleted_at IS NULL",
            (email,),
        ).fetchone()
        if row is None:
            raise VoxError(code=COACH_USER_NOT_FOUND,
                           message=f"No user with email {email}.",
                           http_status=404)
        return _row_to_user(row)
    finally:
        conn.close()


def consume_activation_token(token: str, *, now: int | None = None) -> CoachUser:
    """One-shot consume of activation_token — clears it after success.

    Raises COACH_ACTIVATION_INVALID if unknown / consumed, or
    COACH_ACTIVATION_EXPIRED if past expiry.
    """
    now = now if now is not None else int(time.time())
    conn = connect()
    try:
        row = conn.execute(
            "SELECT * FROM coach_users_tiers WHERE activation_token = ? AND deleted_at IS NULL",
            (token,),
        ).fetchone()
        if row is None:
            raise VoxError(code=COACH_ACTIVATION_INVALID,
                           message="Activation token invalid or already used.",
                           http_status=401)
        if row["activation_token_expires_at"] and now > row["activation_token_expires_at"]:
            raise VoxError(code=COACH_ACTIVATION_EXPIRED,
                           message="Activation token expired.",
                           http_status=410,
                           hint="Ask Faustão to generate a new activation link.")
        with transaction(conn):
            conn.execute(
                """
                UPDATE coach_users_tiers
                SET activation_token = NULL, activation_token_expires_at = NULL,
                    last_seen_at = ?
                WHERE id = ?
                """,
                (now, row["id"]),
            )
        return _row_to_user(
            conn.execute(
                "SELECT * FROM coach_users_tiers WHERE id = ?", (row["id"],),
            ).fetchone()
        )
    finally:
        conn.close()


# ------------------------------------------------------------------ quota

def increment_session_counter(user_id: str) -> CoachUser:
    """Bump sessions_used_this_period by 1. Caller must have already
    checked the quota via ``pricing.check_can_start_session``."""
    conn = connect()
    try:
        with transaction(conn):
            conn.execute(
                "UPDATE coach_users_tiers SET sessions_used_this_period = sessions_used_this_period + 1 "
                "WHERE id = ?",
                (user_id,),
            )
        return get_user_by_id(user_id)
    finally:
        conn.close()


def increment_report_counter(user_id: str) -> CoachUser:
    conn = connect()
    try:
        with transaction(conn):
            conn.execute(
                "UPDATE coach_users_tiers SET reports_used_this_period = reports_used_this_period + 1 "
                "WHERE id = ?",
                (user_id,),
            )
        return get_user_by_id(user_id)
    finally:
        conn.close()


def maybe_reset_period(user_id: str, *, now: int | None = None) -> CoachUser:
    """If now > period_start + period_days, zero the counters and start a new
    period. Idempotent: calling this every request is cheap (single read +
    optional write)."""
    now = now if now is not None else int(time.time())
    user = get_user_by_id(user_id)
    tier = get_tier(user.tier_key)
    period_end = user.period_start + tier.period_days * 86_400
    if now < period_end:
        return user
    conn = connect()
    try:
        with transaction(conn):
            conn.execute(
                """
                UPDATE coach_users_tiers
                SET sessions_used_this_period = 0, reports_used_this_period = 0,
                    period_start = ?, last_seen_at = ?
                WHERE id = ?
                """,
                (now, now, user_id),
            )
        return get_user_by_id(user_id)
    finally:
        conn.close()


def soft_delete_user(user_id: str, *, now: int | None = None) -> None:
    """LGPD right-to-erasure: clear PII (email), keep id for audit."""
    now = now if now is not None else int(time.time())
    conn = connect()
    try:
        with transaction(conn):
            # Use a unique tombstone email to free the original for re-creation.
            tombstone = f"deleted_{secrets.token_urlsafe(8)}@deleted.invalid"
            conn.execute(
                """
                UPDATE coach_users_tiers
                SET deleted_at = ?, email = ?,
                    activation_token = NULL, activation_token_expires_at = NULL
                WHERE id = ?
                """,
                (now, tombstone, user_id),
            )
    finally:
        conn.close()
