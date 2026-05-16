"""Tier definitions + quota enforcement (SPEC_COACH §9).

Tier features (frozen v0.1.2 — Tier 2/3 are defined but the Cofre
features they unlock live in VOX-COACH-C):

| Tier           | Monthly | Sessions/mo | Reports/mo | Model            |
|----------------|---------|-------------|------------|------------------|
| FREE_TRIAL     | $0      | 1 (lifetime)| 0 (template only) | None       |
| TIER_1_MONTHLY | $36     | 45          | 15         | sonnet-4-6       |
| TIER_2_MONTHLY | $77     | unlimited   | 50         | opus (T2+ feature)|
| TIER_3_MONTHLY | $141.90 | unlimited   | unlimited  | opus             |

Annual discount: 36% on all paid tiers (handled at checkout — VOX-COACH-D).

Quota enforcement is read-only here: the caller (route handler) fetches
the user row, then calls ``check_can_start_session`` / ``check_can_generate_report``.
Counters live in ``coach_users_tiers`` and are incremented in ``users.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import VoxError


COACH_QUOTA_EXCEEDED = "COACH_QUOTA_EXCEEDED"
COACH_INVALID_TIER   = "COACH_INVALID_TIER"


# -1 = unlimited
UNLIMITED = -1


@dataclass(frozen=True, slots=True)
class TierDef:
    key: str
    price_monthly_usd: float
    sessions_per_period: int        # -1 = unlimited
    reports_per_period: int         # -1 = unlimited
    report_model: str | None        # None = template-only (no LLM)
    period_days: int
    retention_enabled: bool         # Cofre (VOX-COACH-C)
    personalized_reports: bool      # Opus + history (VOX-COACH-C)
    history_features: bool          # Cofre features (VOX-COACH-C)
    label: str


PRICING: dict[str, TierDef] = {
    "FREE_TRIAL": TierDef(
        key="FREE_TRIAL", price_monthly_usd=0.00,
        sessions_per_period=1, reports_per_period=0,
        report_model=None, period_days=30,
        retention_enabled=False, personalized_reports=False,
        history_features=False,
        label="Trial — uma vez por conta",
    ),
    "TIER_1_MONTHLY": TierDef(
        key="TIER_1_MONTHLY", price_monthly_usd=36.00,
        sessions_per_period=45, reports_per_period=15,
        report_model="claude-sonnet-4-6", period_days=30,
        retention_enabled=False, personalized_reports=False,
        history_features=False,
        label="Premium",
    ),
    "TIER_2_MONTHLY": TierDef(
        key="TIER_2_MONTHLY", price_monthly_usd=77.00,
        sessions_per_period=UNLIMITED, reports_per_period=50,
        report_model="claude-opus-4-7", period_days=30,
        retention_enabled=True, personalized_reports=True,
        history_features=True,
        label="Profissional",
    ),
    "TIER_3_MONTHLY": TierDef(
        key="TIER_3_MONTHLY", price_monthly_usd=141.90,
        sessions_per_period=UNLIMITED, reports_per_period=UNLIMITED,
        report_model="claude-opus-4-7", period_days=30,
        retention_enabled=True, personalized_reports=True,
        history_features=True,
        label="Escritório",
    ),
}


ANNUAL_DISCOUNT = 0.36


def annual_price_usd(tier_key: str) -> float:
    """Compute annual price after 36% discount, rounded to 2 decimals."""
    tier = get_tier(tier_key)
    return round(tier.price_monthly_usd * 12 * (1 - ANNUAL_DISCOUNT), 2)


def get_tier(tier_key: str) -> TierDef:
    """Lookup a tier or raise COACH_INVALID_TIER."""
    tier = PRICING.get(tier_key)
    if tier is None:
        raise VoxError(
            code=COACH_INVALID_TIER,
            message=f"Tier '{tier_key}' not recognized.",
            http_status=400,
            hint=f"Valid tiers: {', '.join(sorted(PRICING.keys()))}.",
        )
    return tier


# ------------------------------------------------------------------ quota checks

def check_can_start_session(*, tier_key: str, sessions_used: int) -> None:
    """Raise COACH_QUOTA_EXCEEDED if next session would exceed the limit."""
    tier = get_tier(tier_key)
    if tier.sessions_per_period == UNLIMITED:
        return
    if sessions_used >= tier.sessions_per_period:
        raise VoxError(
            code=COACH_QUOTA_EXCEEDED,
            message=(f"Session quota exhausted ({sessions_used}/{tier.sessions_per_period} "
                     f"used on {tier.label})."),
            http_status=402,
            hint="Wait for period reset or upgrade tier.",
        )


def check_can_generate_report(*, tier_key: str, reports_used: int) -> None:
    """Raise COACH_QUOTA_EXCEEDED if next report would exceed the limit.

    FREE_TRIAL has 0 reports — caller should fall back to template (no LLM).
    """
    tier = get_tier(tier_key)
    if tier.reports_per_period == UNLIMITED:
        return
    if reports_used >= tier.reports_per_period:
        raise VoxError(
            code=COACH_QUOTA_EXCEEDED,
            message=(f"Report quota exhausted ({reports_used}/{tier.reports_per_period} "
                     f"used on {tier.label})."),
            http_status=402,
            hint="Wait for period reset or upgrade tier. Template-only report still available.",
        )


def supports_llm_report(tier_key: str) -> bool:
    """True if this tier gets LLM narrative reports (False = template-only)."""
    return get_tier(tier_key).report_model is not None
