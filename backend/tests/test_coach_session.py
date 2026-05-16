"""Pure + DB-integration tests for Coach session state machine.

The pure transition tests don't touch DB; the CRUD tests use the existing
``tmp_db`` fixture from ``conftest.py`` so each test gets a fresh sqlite.
"""

from __future__ import annotations

import pytest

from app.coach import session as cs
from app.errors import VoxError


# ----------------------------------------------------------- pure transitions

def test_allowed_transitions_table_complete() -> None:
    """Every SessionState must have an entry — keeps state machine total."""
    for state in cs.SessionState:
        assert state in cs.ALLOWED_TRANSITIONS


def test_terminal_state_has_no_exits() -> None:
    assert cs.ALLOWED_TRANSITIONS[cs.SessionState.ENDED] == set()


@pytest.mark.parametrize("current,target", [
    (cs.SessionState.CREATED, cs.SessionState.AWAITING_CALIBRATION),
    (cs.SessionState.AWAITING_CALIBRATION, cs.SessionState.READY),
    (cs.SessionState.READY, cs.SessionState.IN_PRACTICE),
    (cs.SessionState.READY, cs.SessionState.ENDED),
    (cs.SessionState.IN_PRACTICE, cs.SessionState.IN_PRACTICE),
    (cs.SessionState.IN_PRACTICE, cs.SessionState.ENDED),
])
def test_valid_transitions_accepted(current, target) -> None:
    cs.validate_transition(current, target)  # must not raise


@pytest.mark.parametrize("current,target", [
    (cs.SessionState.CREATED, cs.SessionState.READY),           # skip calibration
    (cs.SessionState.CREATED, cs.SessionState.IN_PRACTICE),     # skip ahead
    (cs.SessionState.READY, cs.SessionState.AWAITING_CALIBRATION),  # re-calibrate
    (cs.SessionState.ENDED, cs.SessionState.IN_PRACTICE),       # zombie session
    (cs.SessionState.ENDED, cs.SessionState.READY),
])
def test_invalid_transitions_rejected(current, target) -> None:
    with pytest.raises(VoxError) as exc_info:
        cs.validate_transition(current, target)
    assert exc_info.value.code == cs.COACH_INVALID_STATE_FOR_ACTION
    assert exc_info.value.http_status == 400


def test_gen_session_id_unique() -> None:
    ids = {cs.gen_session_id() for _ in range(100)}
    assert len(ids) == 100
    assert all(i.startswith("ses_") for i in ids)


# ----------------------------------------------------------- DB CRUD

def test_create_session_returns_created_state(tmp_db) -> None:
    sess = cs.create_session(
        owner_user_id="usr_abc",
        session_name="João prep",
        session_token="tok_test_123",
    )
    assert sess.state == cs.SessionState.CREATED
    assert sess.owner_user_id == "usr_abc"
    assert sess.session_name == "João prep"
    assert sess.baseline_features is None
    assert sess.id.startswith("ses_")
    assert sess.expires_at > sess.created_at


def test_get_session_by_token_roundtrip(tmp_db) -> None:
    created = cs.create_session(
        owner_user_id="usr_a", session_name="t", session_token="tok_xyz",
    )
    fetched = cs.get_session_by_token("tok_xyz")
    assert fetched.id == created.id
    assert fetched.state == cs.SessionState.CREATED


def test_get_session_by_token_not_found(tmp_db) -> None:
    with pytest.raises(VoxError) as ei:
        cs.get_session_by_token("tok_nonexistent")
    assert ei.value.code == cs.COACH_SESSION_NOT_FOUND
    assert ei.value.http_status == 404


def test_get_session_by_token_expired(tmp_db) -> None:
    sess = cs.create_session(
        owner_user_id="u", session_name="s", session_token="tok_old",
        ttl_seconds=10, now=1000,
    )
    with pytest.raises(VoxError) as ei:
        cs.get_session_by_token("tok_old", now=10_000)
    assert ei.value.code == cs.COACH_SESSION_EXPIRED
    assert ei.value.http_status == 410


def test_set_baseline_transitions_to_ready(tmp_db) -> None:
    cs.create_session(owner_user_id="u", session_name="s", session_token="t1")
    sess = cs.get_session_by_token("t1")
    features = {
        "jitter_local": 0.018,
        "mfcc_delta_var_mean": 0.04,
        "spectral_flux_mean": 0.12,
        "microtremor_envelope": 0.003,
    }
    updated = cs.set_baseline(
        session_id=sess.id,
        baseline_features=features,
        mic_quality_label="GREEN",
        mic_quality_snr_db=28.5,
    )
    assert updated.state == cs.SessionState.READY
    assert updated.baseline_features == features
    assert updated.mic_quality_label == "GREEN"
    assert updated.mic_quality_snr_db == 28.5


def test_set_baseline_rejects_second_calibration(tmp_db) -> None:
    """SPEC §7.1 — per-session baseline immutable."""
    cs.create_session(owner_user_id="u", session_name="s", session_token="t2")
    sess = cs.get_session_by_token("t2")
    features = {"jitter_local": 0.01, "mfcc_delta_var_mean": 0.02,
                "spectral_flux_mean": 0.1, "microtremor_envelope": 0.003}
    cs.set_baseline(session_id=sess.id, baseline_features=features,
                    mic_quality_label="GREEN", mic_quality_snr_db=30.0)
    with pytest.raises(VoxError) as ei:
        cs.set_baseline(session_id=sess.id, baseline_features=features,
                        mic_quality_label="GREEN", mic_quality_snr_db=30.0)
    assert ei.value.code == cs.COACH_BASELINE_ALREADY_SET
    assert ei.value.http_status == 409


def test_mark_in_practice_requires_baseline(tmp_db) -> None:
    cs.create_session(owner_user_id="u", session_name="s", session_token="t3")
    sess = cs.get_session_by_token("t3")
    with pytest.raises(VoxError) as ei:
        cs.mark_in_practice(sess.id)
    assert ei.value.code == cs.COACH_BASELINE_REQUIRED
    assert ei.value.http_status == 400


def test_mark_in_practice_transitions_from_ready(tmp_db) -> None:
    cs.create_session(owner_user_id="u", session_name="s", session_token="t4")
    sess = cs.get_session_by_token("t4")
    cs.set_baseline(session_id=sess.id, baseline_features={"x": 1.0},
                    mic_quality_label="GREEN", mic_quality_snr_db=30.0)
    updated = cs.mark_in_practice(sess.id)
    assert updated.state == cs.SessionState.IN_PRACTICE


def test_mark_in_practice_idempotent_self_loop(tmp_db) -> None:
    cs.create_session(owner_user_id="u", session_name="s", session_token="t5")
    sess = cs.get_session_by_token("t5")
    cs.set_baseline(session_id=sess.id, baseline_features={"x": 1.0},
                    mic_quality_label="GREEN", mic_quality_snr_db=30.0)
    cs.mark_in_practice(sess.id)
    again = cs.mark_in_practice(sess.id)
    assert again.state == cs.SessionState.IN_PRACTICE


def test_end_session_from_ready(tmp_db) -> None:
    cs.create_session(owner_user_id="u", session_name="s", session_token="t6")
    sess = cs.get_session_by_token("t6")
    cs.set_baseline(session_id=sess.id, baseline_features={"x": 1.0},
                    mic_quality_label="GREEN", mic_quality_snr_db=30.0)
    ended = cs.end_session(sess.id, report_html="<p>report</p>", now=2000)
    assert ended.state == cs.SessionState.ENDED
    assert ended.ended_at == 2000
    assert ended.report_html == "<p>report</p>"
    assert ended.report_generated_at == 2000


def test_end_session_from_in_practice(tmp_db) -> None:
    cs.create_session(owner_user_id="u", session_name="s", session_token="t7")
    sess = cs.get_session_by_token("t7")
    cs.set_baseline(session_id=sess.id, baseline_features={"x": 1.0},
                    mic_quality_label="GREEN", mic_quality_snr_db=30.0)
    cs.mark_in_practice(sess.id)
    ended = cs.end_session(sess.id)
    assert ended.state == cs.SessionState.ENDED


def test_end_session_twice_rejected(tmp_db) -> None:
    cs.create_session(owner_user_id="u", session_name="s", session_token="t8")
    sess = cs.get_session_by_token("t8")
    cs.set_baseline(session_id=sess.id, baseline_features={"x": 1.0},
                    mic_quality_label="GREEN", mic_quality_snr_db=30.0)
    cs.end_session(sess.id)
    with pytest.raises(VoxError) as ei:
        cs.end_session(sess.id)
    assert ei.value.code == cs.COACH_SESSION_ALREADY_ENDED


def test_soft_delete_hides_from_lookup(tmp_db) -> None:
    cs.create_session(owner_user_id="u", session_name="s", session_token="t9")
    sess = cs.get_session_by_token("t9")
    cs.soft_delete(sess.id)
    with pytest.raises(VoxError) as ei:
        cs.get_session_by_token("t9")
    assert ei.value.code == cs.COACH_SESSION_NOT_FOUND
