"""Tests for Coach pricing tier definitions + quota checks + users CRUD."""

from __future__ import annotations

import pytest

from app.coach import pricing as pr
from app.coach import users as cu
from app.coach.auth import ACTIVATION_TOKEN_TTL_SECONDS
from app.errors import VoxError


# ----------------------------------------------------------- pricing pure

def test_all_four_tiers_defined() -> None:
    assert set(pr.PRICING.keys()) == {
        "FREE_TRIAL", "TIER_1_MONTHLY", "TIER_2_MONTHLY", "TIER_3_MONTHLY",
    }


def test_free_trial_no_llm_reports() -> None:
    free = pr.get_tier("FREE_TRIAL")
    assert free.report_model is None
    assert free.reports_per_period == 0
    assert not pr.supports_llm_report("FREE_TRIAL")


def test_tier_1_uses_sonnet() -> None:
    t1 = pr.get_tier("TIER_1_MONTHLY")
    assert t1.report_model == "claude-sonnet-4-6"
    assert t1.reports_per_period == 15
    assert t1.sessions_per_period == 45
    assert pr.supports_llm_report("TIER_1_MONTHLY")


def test_tier_2_and_3_use_opus() -> None:
    for k in ("TIER_2_MONTHLY", "TIER_3_MONTHLY"):
        assert pr.get_tier(k).report_model == "claude-opus-4-7"
        assert pr.get_tier(k).sessions_per_period == pr.UNLIMITED


def test_annual_price_36_discount() -> None:
    # T1 monthly $36 * 12 * 0.64 = $276.48
    assert pr.annual_price_usd("TIER_1_MONTHLY") == 276.48
    # T2 monthly $77 * 12 * 0.64 = $591.36
    assert pr.annual_price_usd("TIER_2_MONTHLY") == 591.36


def test_unknown_tier_raises() -> None:
    with pytest.raises(VoxError) as ei:
        pr.get_tier("ENTERPRISE_PLATINUM")
    assert ei.value.code == pr.COACH_INVALID_TIER


# ----------------------------------------------------------- quota checks

def test_quota_session_under_limit_passes() -> None:
    pr.check_can_start_session(tier_key="TIER_1_MONTHLY", sessions_used=44)
    pr.check_can_start_session(tier_key="FREE_TRIAL", sessions_used=0)


def test_quota_session_exceeded_raises() -> None:
    with pytest.raises(VoxError) as ei:
        pr.check_can_start_session(tier_key="TIER_1_MONTHLY", sessions_used=45)
    assert ei.value.code == pr.COACH_QUOTA_EXCEEDED
    assert ei.value.http_status == 402

    with pytest.raises(VoxError):
        pr.check_can_start_session(tier_key="FREE_TRIAL", sessions_used=1)


def test_quota_session_unlimited_never_raises() -> None:
    pr.check_can_start_session(tier_key="TIER_2_MONTHLY", sessions_used=10_000)
    pr.check_can_start_session(tier_key="TIER_3_MONTHLY", sessions_used=10_000)


def test_quota_report_unlimited_t3() -> None:
    pr.check_can_generate_report(tier_key="TIER_3_MONTHLY", reports_used=999)


def test_quota_report_exceeded_t2() -> None:
    with pytest.raises(VoxError) as ei:
        pr.check_can_generate_report(tier_key="TIER_2_MONTHLY", reports_used=50)
    assert ei.value.code == pr.COACH_QUOTA_EXCEEDED


def test_quota_report_free_trial_always_exceeded() -> None:
    """FREE_TRIAL has 0 reports — first attempt is already over the limit."""
    with pytest.raises(VoxError):
        pr.check_can_generate_report(tier_key="FREE_TRIAL", reports_used=0)


# ----------------------------------------------------------- users CRUD

def test_create_user_returns_activation_token(tmp_db) -> None:
    user = cu.create_or_upgrade(email="adv@example.com", tier_key="FREE_TRIAL")
    assert user.id.startswith("usr_")
    assert user.email == "adv@example.com"
    assert user.tier_key == "FREE_TRIAL"
    assert user.activation_token is not None
    assert user.activation_token_expires_at is not None
    assert user.sessions_used_this_period == 0
    assert user.reports_used_this_period == 0


def test_create_user_idempotent_upgrades_existing(tmp_db) -> None:
    """Calling create_or_upgrade twice for same email upgrades + new token."""
    u1 = cu.create_or_upgrade(email="adv@x.com", tier_key="FREE_TRIAL")
    u2 = cu.create_or_upgrade(email="adv@x.com", tier_key="TIER_1_MONTHLY")
    assert u1.id == u2.id              # same row
    assert u2.tier_key == "TIER_1_MONTHLY"
    assert u2.activation_token != u1.activation_token
    assert u2.sessions_used_this_period == 0  # reset


def test_get_user_by_email_roundtrip(tmp_db) -> None:
    created = cu.create_or_upgrade(email="e@e.com", tier_key="FREE_TRIAL")
    fetched = cu.get_user_by_email("e@e.com")
    assert fetched.id == created.id


def test_get_user_by_email_not_found(tmp_db) -> None:
    with pytest.raises(VoxError) as ei:
        cu.get_user_by_email("nope@nope.com")
    assert ei.value.code == cu.COACH_USER_NOT_FOUND


def test_consume_activation_token_clears_it(tmp_db) -> None:
    user = cu.create_or_upgrade(email="x@x.com", tier_key="FREE_TRIAL")
    tok = user.activation_token
    consumed = cu.consume_activation_token(tok)
    assert consumed.id == user.id
    assert consumed.activation_token is None  # cleared
    # Second consume fails (token already cleared).
    with pytest.raises(VoxError) as ei:
        cu.consume_activation_token(tok)
    assert ei.value.code == cu.COACH_ACTIVATION_INVALID


def test_consume_activation_token_expired(tmp_db) -> None:
    user = cu.create_or_upgrade(email="t@t.com", tier_key="FREE_TRIAL", now=1000)
    with pytest.raises(VoxError) as ei:
        cu.consume_activation_token(
            user.activation_token,
            now=1000 + ACTIVATION_TOKEN_TTL_SECONDS + 1,
        )
    assert ei.value.code == cu.COACH_ACTIVATION_EXPIRED


def test_increment_session_counter(tmp_db) -> None:
    user = cu.create_or_upgrade(email="i@i.com", tier_key="TIER_1_MONTHLY")
    updated = cu.increment_session_counter(user.id)
    assert updated.sessions_used_this_period == 1
    updated2 = cu.increment_session_counter(user.id)
    assert updated2.sessions_used_this_period == 2


def test_increment_report_counter(tmp_db) -> None:
    user = cu.create_or_upgrade(email="r@r.com", tier_key="TIER_1_MONTHLY")
    updated = cu.increment_report_counter(user.id)
    assert updated.reports_used_this_period == 1


def test_maybe_reset_period_within_period_no_op(tmp_db) -> None:
    user = cu.create_or_upgrade(email="p@p.com", tier_key="TIER_1_MONTHLY",
                                  now=1000)
    cu.increment_session_counter(user.id)
    # Still within 30-day period.
    result = cu.maybe_reset_period(user.id, now=1000 + 86400)
    assert result.sessions_used_this_period == 1


def test_maybe_reset_period_after_period_zeroes_counters(tmp_db) -> None:
    user = cu.create_or_upgrade(email="z@z.com", tier_key="TIER_1_MONTHLY",
                                  now=1000)
    cu.increment_session_counter(user.id)
    cu.increment_report_counter(user.id)
    # Past 30 days.
    later = 1000 + 31 * 86400
    result = cu.maybe_reset_period(user.id, now=later)
    assert result.sessions_used_this_period == 0
    assert result.reports_used_this_period == 0
    assert result.period_start == later


def test_soft_delete_clears_pii(tmp_db) -> None:
    user = cu.create_or_upgrade(email="d@d.com", tier_key="FREE_TRIAL")
    cu.soft_delete_user(user.id)
    # email replaced by tombstone, get_user_by_id raises USER_DELETED.
    with pytest.raises(VoxError) as ei:
        cu.get_user_by_id(user.id)
    assert ei.value.code == cu.COACH_USER_DELETED
    # Original email is freed for re-creation.
    new_user = cu.create_or_upgrade(email="d@d.com", tier_key="FREE_TRIAL")
    assert new_user.id != user.id
