"""Integration tests for the VOX-LANDING-A marketing routes.

Covers the new marketing landing at ``/`` (3-tab Explorer/Academic/Coach),
the v0.1 tool relocated to ``/app``, the per-product Terms HTML pages at
``/coach/terms`` and ``/academic/terms``, the hub at ``/terms``, and the
``/m/*`` asset mount. Also includes regression checks that the v0.1
backend (``/api/*``, ``/privacy``, ``/assets/*``) keeps working.
"""

from __future__ import annotations


def test_root_serves_marketing_landing(client) -> None:
    """GET / serves the 3-tab marketing landing (was v0.1 tool before)."""
    r = client.get("/")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "VOX&nbsp;PROBABILIS" in body or "VOX PROBABILIS" in body
    assert 'data-tab="v1"' in body
    assert 'data-tab="academic"' in body
    assert 'data-tab="coach"' in body
    assert "/m/static/style.css" in body
    assert "/m/static/script.js" in body


def test_head_root_returns_200(client) -> None:
    """HEAD / must not 405 (Cloudflare/uptime monitors use HEAD)."""
    r = client.head("/")
    assert r.status_code == 200


def test_app_serves_v1_tool(client) -> None:
    """GET /app serves the v0.1 functional tool that used to live at /."""
    r = client.get("/app")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "calibrate" in body.lower() or "calibração" in body.lower()


def test_coach_terms_returns_articles(client) -> None:
    """GET /coach/terms serves SPEC_COACH §8.1 Art. 1º-8º."""
    r = client.get("/coach/terms")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "Art. 1º" in body
    assert "Art. 8º" in body
    assert "R$ 1.000,00" in body
    assert "Porto Seguro" in body
    assert "Coach" in body


def test_academic_terms_returns_articles(client) -> None:
    """GET /academic/terms serves SPEC_ACADEMIC §8.2 Art. 1º-8º."""
    r = client.get("/academic/terms")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert "Art. 1º" in body
    assert "Art. 8º" in body
    assert "R$ 500,00" in body
    assert "LGPD" in body
    assert "Academic" in body


def test_terms_hub_lists_both_products(client) -> None:
    """GET /terms serves the hub linking to Coach and Academic terms."""
    r = client.get("/terms")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    body = r.text
    assert 'href="/coach/terms"' in body
    assert 'href="/academic/terms"' in body
    assert "Coach" in body
    assert "Academic" in body


def test_marketing_assets_mount(client) -> None:
    """GET /m/static/style.css serves the marketing CSS at the new mount."""
    r = client.get("/m/static/style.css")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/css")
    assert len(r.content) > 1000


def test_marketing_image_served(client) -> None:
    """GET /m/audiencia_cartesian.png serves the Academic demo plot."""
    r = client.get("/m/audiencia_cartesian.png")
    assert r.status_code == 200
    assert r.headers["content-type"] == "image/png"


def test_v1_privacy_still_served(client) -> None:
    """Regression: /privacy keeps serving v0.1 LGPD policy unchanged."""
    r = client.get("/privacy")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/html")
    assert "LGPD" in r.text or "Privacy" in r.text


def test_v1_api_health_unaffected(client) -> None:
    """Regression: marketing routing did not break the v0.1 API."""
    r = client.get("/api/health")
    assert r.status_code in (200, 503)
    assert "status" in r.json()


def test_v1_assets_mount_unaffected(client) -> None:
    """Regression: /assets/ still serves the v0.1 static tree."""
    r = client.get("/assets/index.html")
    assert r.status_code == 200
