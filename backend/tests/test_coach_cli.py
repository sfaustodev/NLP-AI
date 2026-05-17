"""Tests for Coach tier-activation CLI (`python -m app.coach.cli`)."""

from __future__ import annotations

import pytest

from app.coach import cli as coach_cli
from app.coach import users as coach_users


def test_upgrade_creates_user_and_prints_activation_url(tmp_db, capsys) -> None:
    rc = coach_cli.main(["upgrade", "--email", "adv@x.com", "--tier", "FREE_TRIAL"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "adv@x.com" in out
    assert "FREE_TRIAL" in out
    assert "/coach/activate?token=" in out
    # User actually committed to DB.
    user = coach_users.get_user_by_email("adv@x.com")
    assert user.tier_key == "FREE_TRIAL"
    assert user.activation_token is not None


def test_upgrade_existing_user_rotates_token(tmp_db, capsys) -> None:
    coach_cli.main(["upgrade", "--email", "a@a.com", "--tier", "FREE_TRIAL"])
    first_token = coach_users.get_user_by_email("a@a.com").activation_token

    coach_cli.main(["upgrade", "--email", "a@a.com", "--tier", "TIER_1_MONTHLY"])
    second = coach_users.get_user_by_email("a@a.com")
    assert second.tier_key == "TIER_1_MONTHLY"
    assert second.activation_token != first_token


def test_upgrade_rejects_unknown_tier_via_argparse(tmp_db, capsys) -> None:
    with pytest.raises(SystemExit) as ei:
        coach_cli.main(["upgrade", "--email", "x@x.com", "--tier", "PLATINUM_VIP"])
    # argparse prints to stderr + exits 2.
    assert ei.value.code == 2


def test_list_users_empty(tmp_db, capsys) -> None:
    rc = coach_cli.main(["list-users"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "no Coach users yet" in out


def test_list_users_shows_active_rows(tmp_db, capsys) -> None:
    coach_cli.main(["upgrade", "--email", "a@a.com", "--tier", "FREE_TRIAL"])
    coach_cli.main(["upgrade", "--email", "b@b.com", "--tier", "TIER_1_MONTHLY"])
    capsys.readouterr()  # drain previous output
    coach_cli.main(["list-users"])
    out = capsys.readouterr().out
    assert "a@a.com" in out
    assert "b@b.com" in out
    assert "FREE_TRIAL" in out
    assert "TIER_1_MONTHLY" in out
    assert "active" in out


def test_revoke_soft_deletes_existing_user(tmp_db, capsys) -> None:
    coach_cli.main(["upgrade", "--email", "rm@rm.com", "--tier", "FREE_TRIAL"])
    capsys.readouterr()
    rc = coach_cli.main(["revoke", "--email", "rm@rm.com"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Soft-deleted" in out
    # get_user_by_email skips deleted rows — should now miss.
    from app.errors import VoxError
    with pytest.raises(VoxError):
        coach_users.get_user_by_email("rm@rm.com")


def test_revoke_unknown_user_returns_1(tmp_db, capsys) -> None:
    rc = coach_cli.main(["revoke", "--email", "ghost@ghost.com"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "COACH_USER_NOT_FOUND" in err
