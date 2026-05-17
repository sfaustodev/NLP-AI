"""Sonnet 4.6 narrative report for Tier 1 Coach sessions.

Flow:

1. Caller passes a CoachSession + list of CoachResponse rows + baseline_features.
2. We build a prompt that summarises the session (mic quality, baseline, every
   response's deltas + label + narrative).
3. If ``settings.coach_sonnet_api_key`` is set, we call the Anthropic SDK.
   Otherwise we return a template HTML report — useful for FREE_TRIAL (which
   per SPEC §9 gets no LLM) and for dev/test envs without a key.
4. Retries 3× with exponential backoff on transient Anthropic errors.

The output is always HTML (server stores it in ``coach_sessions.report_html``
and serves it via ``/api/coach/session/{token}/report.html``).
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Iterable, Optional

from ...config import settings


log = logging.getLogger("vox.coach.reports.sonnet")


RETRY_MAX_ATTEMPTS = 3
RETRY_BASE_DELAY_S = 1.0


@dataclass(frozen=True, slots=True)
class ReportInputs:
    """All data the Sonnet prompt needs — passed as a single struct so the
    caller (routes.py) does the DB reads in one place."""
    session_name: str
    session_created_at: int
    session_ended_at: int
    mic_quality_label: str
    mic_quality_snr_db: float
    baseline_features: dict
    responses: list[dict]   # each: {question_text, duration_s, delta_pct, consistency_label, color, narrative}


def _build_prompt(inp: ReportInputs) -> str:
    """Construct the user-message prompt sent to Sonnet.

    Keeps the prompt deterministic (sorted keys, no timestamps in the body)
    so the same session produces the same input twice in a row — easier to
    debug and lets us cache responses if we ever want to.
    """
    summary = {
        "session_name": inp.session_name,
        "session_minutes": round((inp.session_ended_at - inp.session_created_at) / 60.0, 1),
        "mic_quality": inp.mic_quality_label,
        "mic_snr_db": round(inp.mic_quality_snr_db, 1),
        "baseline": {k: round(v, 5) for k, v in sorted(inp.baseline_features.items())},
        "responses_count": len(inp.responses),
        "consistency_distribution": _consistency_distribution(inp.responses),
        "responses": [
            {
                "index": i + 1,
                "question": r.get("question_text") or "(sem texto)",
                "duration_s": round(r.get("duration_s", 0), 1),
                "label": r.get("consistency_label"),
                "color": r.get("color"),
                "deltas": {k: round(v, 1) for k, v in sorted(r.get("delta_pct", {}).items())},
            }
            for i, r in enumerate(inp.responses)
        ],
    }
    return (
        "Você é um assistente que produz relatórios em pt-BR para advogados brasileiros\n"
        "após uma sessão de preparação prosódica com cliente ou testemunha.\n\n"
        "REGRAS:\n"
        "- Não classifique respostas como verdade/mentira. Use linguagem comparativa\n"
        "  ('houve deslocamento prosódico em relação ao baseline').\n"
        "- Trate o output como ferramenta de preparação privada do escritório, não\n"
        "  como evidência judicial.\n"
        "- Estrutura: 3 seções HTML — <h2>Visão geral</h2>, <h2>Padrões observados</h2>,\n"
        "  <h2>Áreas de atenção</h2>. Use <p>, <ul>, <li>. Sem JS, sem CSS embutido.\n"
        "- 250-450 palavras totais. Português profissional, direto.\n\n"
        "DADOS DA SESSÃO (JSON):\n"
        f"{json.dumps(summary, indent=2, ensure_ascii=False)}\n\n"
        "Produza o relatório HTML agora."
    )


def _consistency_distribution(responses: Iterable[dict]) -> dict[str, int]:
    out = {"BASELINE": 0, "SLIGHT_SHIFT": 0, "NOTABLE_SHIFT": 0, "MAJOR_SHIFT": 0}
    for r in responses:
        label = r.get("consistency_label", "BASELINE")
        if label in out:
            out[label] += 1
    return out


def _template_fallback(inp: ReportInputs) -> str:
    """No-LLM HTML report — used by FREE_TRIAL and when no API key is set.

    Mirrors the structure of the LLM report so the UI doesn't have to branch.
    """
    dist = _consistency_distribution(inp.responses)
    rows_html = "\n".join(
        f"<li>Resposta {i+1} ({r.get('consistency_label', 'BASELINE')}): "
        f"{r.get('narrative') or 'sem narrativa'}</li>"
        for i, r in enumerate(inp.responses)
    )
    return (
        '<div class="coach-report coach-report--template">'
        '<h2>Visão geral</h2>'
        f'<p>Sessão &quot;{_escape(inp.session_name)}&quot; — '
        f'{len(inp.responses)} respostas analisadas, '
        f'qualidade de microfone <strong>{inp.mic_quality_label}</strong> '
        f'(SNR {inp.mic_quality_snr_db:.1f} dB).</p>'
        '<h2>Distribuição de consistência</h2>'
        '<ul>'
        f'<li>Respostas em baseline: {dist["BASELINE"]}</li>'
        f'<li>Deslocamentos leves: {dist["SLIGHT_SHIFT"]}</li>'
        f'<li>Deslocamentos notáveis: {dist["NOTABLE_SHIFT"]}</li>'
        f'<li>Deslocamentos maiores: {dist["MAJOR_SHIFT"]}</li>'
        '</ul>'
        '<h2>Respostas</h2>'
        f'<ul>{rows_html}</ul>'
        '<p><em>Relatório template (sem narrativa LLM). Upgrade para Tier 1 desbloqueia '
        'relatórios Sonnet personalizados.</em></p>'
        '</div>'
    )


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
              .replace(">", "&gt;").replace('"', "&quot;"))


def _call_anthropic(prompt: str, *, api_key: str, model: str) -> str:
    """Single Anthropic SDK call. Returns assistant text content (HTML)."""
    # Imported lazily so test envs without the SDK can still import this
    # module and exercise the template fallback path.
    import anthropic
    client = anthropic.Anthropic(api_key=api_key)
    message = client.messages.create(
        model=model,
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}],
    )
    # SDK returns a list of content blocks; concatenate text blocks.
    out_parts: list[str] = []
    for block in message.content:
        text = getattr(block, "text", None)
        if text:
            out_parts.append(text)
    html = "".join(out_parts).strip()
    if not html:
        raise RuntimeError("Anthropic returned empty content")
    return html


def generate_report(inputs: ReportInputs, *,
                    api_key: Optional[str] = None,
                    model: Optional[str] = None) -> str:
    """Top-level entry point. Returns HTML string.

    - ``api_key`` defaults to ``settings.coach_sonnet_api_key`` (None →
      template fallback).
    - ``model`` defaults to ``settings.coach_sonnet_model``.
    """
    api_key = api_key if api_key is not None else settings.coach_sonnet_api_key
    model = model if model is not None else settings.coach_sonnet_model

    prompt = _build_prompt(inputs)

    if not api_key:
        log.info('"sonnet_skipped reason=no_api_key — returning template fallback"')
        return _template_fallback(inputs)

    # Retry loop for transient errors. We don't try to introspect Anthropic
    # exception types here (SDK can change) — broad except, log, sleep, retry.
    last_exc: Optional[BaseException] = None
    for attempt in range(1, RETRY_MAX_ATTEMPTS + 1):
        try:
            return _call_anthropic(prompt, api_key=api_key, model=model)
        except Exception as exc:
            last_exc = exc
            delay = RETRY_BASE_DELAY_S * (2 ** (attempt - 1))
            log.warning(
                '"sonnet_attempt_failed attempt=%d max=%d delay_s=%.1f err=%r"',
                attempt, RETRY_MAX_ATTEMPTS, delay, exc,
            )
            if attempt < RETRY_MAX_ATTEMPTS:
                time.sleep(delay)

    log.error('"sonnet_all_attempts_failed err=%r — falling back to template"', last_exc)
    return _template_fallback(inputs)
