"""Tests for Coach Sonnet narrative report generator.

Covers:
- Template fallback when no API key (FREE_TRIAL path)
- Prompt builder structure (deterministic, includes session metadata)
- Anthropic SDK call path with mock (captures model + prompt)
- Retry-then-template fallback on persistent SDK errors
"""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock, patch

import pytest

from app.coach.reports import sonnet_standard as ss


def _inputs(**overrides):
    base = ss.ReportInputs(
        session_name="João prep",
        session_created_at=1_000_000,
        session_ended_at=1_000_900,
        mic_quality_label="GREEN",
        mic_quality_snr_db=28.5,
        baseline_features={
            "jitter_local":         0.018,
            "mfcc_delta_var_mean":  0.047,
            "spectral_flux_mean":   0.128,
            "microtremor_envelope": 0.003,
        },
        responses=[
            {
                "question_text": "Onde estava no dia 14?",
                "duration_s": 12.4,
                "delta_pct": {"jitter_local": 5.0, "mfcc_delta_var_mean": -8.0,
                              "spectral_flux_mean": 3.0, "microtremor_envelope": 4.0},
                "consistency_label": "BASELINE",
                "color": "GREEN",
                "narrative": "Resposta dentro do baseline.",
            },
            {
                "question_text": "E quem mais estava lá?",
                "duration_s": 18.1,
                "delta_pct": {"jitter_local": -42.0, "mfcc_delta_var_mean": -28.0,
                              "spectral_flux_mean": -35.0, "microtremor_envelope": 22.0},
                "consistency_label": "MAJOR_SHIFT",
                "color": "RED",
                "narrative": "Deslocamento maior — jitter reduzido em 42%.",
            },
        ],
    )
    # Recreate with overrides if needed (dataclass slots → no __dict__).
    if overrides:
        return dataclasses.replace(base, **overrides)
    return base


# ----------------------------------------------------------- template fallback

def test_template_fallback_when_no_api_key() -> None:
    """No api_key → no SDK call, returns template HTML."""
    out = ss.generate_report(_inputs(), api_key="", model="claude-sonnet-4-6")
    assert "<h2>Visão geral</h2>" in out
    assert "Distribuição de consistência" in out
    assert "Sessão &quot;João prep&quot;" in out
    assert "GREEN" in out


def test_template_fallback_counts_consistency_labels() -> None:
    out = ss._template_fallback(_inputs())
    assert "Respostas em baseline: 1" in out
    assert "Deslocamentos maiores: 1" in out


def test_template_fallback_escapes_html_in_session_name() -> None:
    inp = _inputs(session_name='<script>alert("xss")</script>')
    out = ss._template_fallback(inp)
    assert "<script>" not in out
    assert "&lt;script&gt;" in out


# ----------------------------------------------------------- prompt builder

def test_prompt_contains_session_metadata() -> None:
    prompt = ss._build_prompt(_inputs())
    assert "João prep" in prompt
    assert "GREEN" in prompt
    assert "MAJOR_SHIFT" in prompt
    assert "Onde estava no dia 14" in prompt
    # Baseline features rounded to 5 decimals.
    assert "0.018" in prompt or "0.01800" in prompt


def test_prompt_includes_anti_truth_disclaimer() -> None:
    """SPEC §7.2 — language must be comparative, never truth/lie binary."""
    prompt = ss._build_prompt(_inputs())
    assert "verdade/mentira" in prompt.lower() or "comparativ" in prompt.lower()


# ----------------------------------------------------------- SDK call path

def test_sdk_call_invoked_with_model_and_prompt() -> None:
    """When api_key set, _call_anthropic runs and we capture model + prompt."""
    captured = {}

    def fake_call(prompt, *, api_key, model):
        captured["prompt"] = prompt
        captured["api_key"] = api_key
        captured["model"] = model
        return "<h2>Visão geral</h2><p>Mocked LLM HTML.</p>"

    with patch.object(ss, "_call_anthropic", side_effect=fake_call):
        html = ss.generate_report(_inputs(), api_key="sk-test", model="claude-sonnet-4-6")

    assert "Mocked LLM HTML" in html
    assert captured["model"] == "claude-sonnet-4-6"
    assert captured["api_key"] == "sk-test"
    assert "João prep" in captured["prompt"]


def test_sdk_retry_then_fallback_after_all_attempts_fail() -> None:
    """Persistent SDK error → 3 attempts → template fallback (not crash)."""
    call_count = {"n": 0}

    def always_fail(prompt, *, api_key, model):
        call_count["n"] += 1
        raise RuntimeError("simulated 503 from upstream")

    # Skip the sleep delays so the test stays fast.
    with patch.object(ss, "_call_anthropic", side_effect=always_fail), \
         patch.object(ss.time, "sleep", return_value=None):
        out = ss.generate_report(_inputs(), api_key="sk-test", model="m")

    assert call_count["n"] == ss.RETRY_MAX_ATTEMPTS
    assert "<h2>Visão geral</h2>" in out  # fallback template


def test_sdk_success_on_second_attempt_returns_llm_html() -> None:
    """Transient error then success → return SDK HTML, no fallback."""
    call_count = {"n": 0}

    def flaky(prompt, *, api_key, model):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("transient 502")
        return "<h2>Visão geral</h2><p>OK on retry.</p>"

    with patch.object(ss, "_call_anthropic", side_effect=flaky), \
         patch.object(ss.time, "sleep", return_value=None):
        out = ss.generate_report(_inputs(), api_key="sk-test", model="m")

    assert "OK on retry" in out
    assert call_count["n"] == 2


def test_sdk_empty_content_treated_as_failure() -> None:
    """If Anthropic returns no text blocks, _call_anthropic raises and we
    fall back. This guards against silently storing an empty report."""
    with patch.object(ss, "_call_anthropic",
                       side_effect=RuntimeError("Anthropic returned empty content")), \
         patch.object(ss.time, "sleep", return_value=None):
        out = ss.generate_report(_inputs(), api_key="sk-test", model="m")
    assert "<h2>Visão geral</h2>" in out
