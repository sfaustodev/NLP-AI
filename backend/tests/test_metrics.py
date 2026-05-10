"""GET /api/metrics — DEPLOY.md §12.2 operator endpoint.

Covers:
- 404 when VOX_METRICS_KEY is unset (endpoint disabled by default)
- 404 when key is wrong (do not leak existence)
- 200 with the expected shape when key matches
- Counters move when traffic flows through calibrate/analyze
"""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def metrics_client(tmp_db, monkeypatch):
    """Client whose Settings reload sees a real VOX_METRICS_KEY."""
    monkeypatch.setenv("VOX_METRICS_KEY", "test-metrics-key-secret")
    from app import config as _config
    importlib.reload(_config)

    from fastapi.testclient import TestClient
    from app import db as _db, sessions as _sessions, rate_limit as _rate
    from app.api import analyze as _analyze
    from app.api import calibrate as _calibrate
    from app.api import session as _session_api
    from app.api import health as _health
    from app.api import metrics as _metrics
    from app import main as _main
    for mod in (_db, _sessions, _rate, _analyze, _calibrate, _session_api, _health, _metrics, _main):
        importlib.reload(mod)

    with TestClient(_main.app) as c:
        yield c


def test_metrics_disabled_when_key_unset(client):
    """No VOX_METRICS_KEY → endpoint pretends not to exist."""
    r = client.get("/api/metrics?key=anything")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "NOT_FOUND"


def test_metrics_wrong_key_is_404_not_401(metrics_client):
    """Wrong key returns 404 to avoid leaking endpoint existence."""
    r = metrics_client.get("/api/metrics?key=wrong-key")
    assert r.status_code == 404


def test_metrics_missing_key_is_404(metrics_client):
    r = metrics_client.get("/api/metrics")
    assert r.status_code == 404


def test_metrics_correct_key_returns_expected_shape(metrics_client):
    r = metrics_client.get("/api/metrics?key=test-metrics-key-secret")
    assert r.status_code == 200
    body = r.json()
    assert set(body.keys()) == {
        "uptime_s",
        "total_calibrations",
        "total_analyses",
        "quadrant_counts",
        "active_sessions_24h",
        "dataset_optin_count",
    }
    assert isinstance(body["uptime_s"], int) and body["uptime_s"] >= 0
    assert body["total_calibrations"] == 0
    assert body["total_analyses"] == 0
    assert body["quadrant_counts"] == {}
    assert body["active_sessions_24h"] == 0
    assert body["dataset_optin_count"] == 0


def test_metrics_counts_after_calibrate(metrics_client, wav_voiced):
    """Calibrate once → calibrations=1, sessions tracked."""
    r = metrics_client.post(
        "/api/calibrate",
        files={"audio": ("truth.wav", wav_voiced, "audio/wav")},
    )
    assert r.status_code == 200, r.text

    m = metrics_client.get("/api/metrics?key=test-metrics-key-secret").json()
    assert m["total_calibrations"] == 1
    assert m["active_sessions_24h"] == 1
