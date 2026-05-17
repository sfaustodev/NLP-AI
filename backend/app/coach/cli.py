"""Coach tier-activation CLI — Faustão's escape hatch until checkout lands.

Usage::

    python -m app.coach.cli upgrade --email adv@example.com --tier FREE_TRIAL
    python -m app.coach.cli upgrade --email firm@x.com --tier TIER_1_MONTHLY
    python -m app.coach.cli list-users
    python -m app.coach.cli revoke --email adv@example.com

After ``upgrade`` succeeds the CLI prints a magic-link activation URL.
Send it to the user — clicking it once consumes the token and sets the
``coach_session`` cookie, redirecting to ``/coach``.

No checkout / Stripe / Lemon Squeezy: VOX-COACH-D scope.
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Optional

from ..config import settings
from ..db import connect
from ..errors import VoxError
from .pricing import PRICING
from . import users as coach_users


TIER_CHOICES = sorted(PRICING.keys())


def cmd_upgrade(args: argparse.Namespace) -> int:
    user = coach_users.create_or_upgrade(email=args.email, tier_key=args.tier)
    activation_url = (
        f"https://{settings.hostname}/coach/activate"
        f"?token={user.activation_token}"
    )
    print(f"User ID         : {user.id}")
    print(f"Email           : {user.email}")
    print(f"Tier            : {user.tier_key}")
    print(f"Tier activated  : {_fmt_ts(user.tier_activated_at)}")
    print(f"Tier expires    : {_fmt_ts(user.tier_expires_at)}")
    print(f"Sessions used   : {user.sessions_used_this_period}")
    print(f"Reports used    : {user.reports_used_this_period}")
    print(f"Activation URL  : {activation_url}")
    print(f"Token expires   : {_fmt_ts(user.activation_token_expires_at)}")
    print()
    print("Send the Activation URL to the user. It is single-use and expires "
          "in 7 days. Clicking it sets the coach_session cookie and "
          "redirects to /coach.")
    return 0


def cmd_list_users(args: argparse.Namespace) -> int:
    conn = connect()
    try:
        rows = conn.execute(
            "SELECT id, email, tier_key, tier_expires_at, sessions_used_this_period, "
            "reports_used_this_period, created_at, deleted_at "
            "FROM coach_users_tiers ORDER BY created_at DESC",
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        print("(no Coach users yet)")
        return 0

    fmt = "{:<32}  {:<28}  {:<18}  {:<19}  {:>4}  {:>4}  {:<8}"
    print(fmt.format(
        "ID", "Email", "Tier", "Expires", "Sess", "Rep", "Status",
    ))
    print("-" * 130)
    for r in rows:
        status = "DELETED" if r["deleted_at"] is not None else "active"
        print(fmt.format(
            r["id"], r["email"][:28], r["tier_key"],
            _fmt_ts(r["tier_expires_at"]),
            r["sessions_used_this_period"], r["reports_used_this_period"],
            status,
        ))
    return 0


def cmd_revoke(args: argparse.Namespace) -> int:
    try:
        user = coach_users.get_user_by_email(args.email)
    except VoxError as e:
        print(f"error: {e.code} — {e.message}", file=sys.stderr)
        return 1
    coach_users.soft_delete_user(user.id)
    print(f"Soft-deleted user {user.id} (email PII tombstoned).")
    return 0


def _fmt_ts(ts: Optional[int]) -> str:
    if ts is None:
        return "(none)"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(ts)) + " UTC"


def main(argv: Optional[list[str]] = None) -> int:
    p = argparse.ArgumentParser(
        prog="python -m app.coach.cli",
        description="Coach tier activation + user management (Faustão escape hatch).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    up = sub.add_parser("upgrade", help="Activate or upgrade a Coach user tier.")
    up.add_argument("--email", required=True, help="Lawyer email address.")
    up.add_argument("--tier", required=True, choices=TIER_CHOICES,
                     help="Tier to activate.")
    up.set_defaults(func=cmd_upgrade)

    ls = sub.add_parser("list-users", help="List all Coach users.")
    ls.set_defaults(func=cmd_list_users)

    rv = sub.add_parser("revoke", help="Soft-delete a Coach user (LGPD erasure).")
    rv.add_argument("--email", required=True)
    rv.set_defaults(func=cmd_revoke)

    args = p.parse_args(argv)
    try:
        return args.func(args)
    except VoxError as e:
        print(f"error: {e.code} — {e.message}", file=sys.stderr)
        if e.hint:
            print(f"hint:  {e.hint}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
