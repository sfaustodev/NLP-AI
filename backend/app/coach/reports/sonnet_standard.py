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
        "Você é um coach jurídico experiente escrevendo um briefing tático\n"
        "em pt-BR para o(a) advogado(a) usar entre uma sessão de preparação\n"
        "e a audiência real. Escreva COMO SE estivesse ao lado dele(a)\n"
        "dizendo o que fazer — não como um relatório técnico.\n\n"
        "REGRAS DE LINGUAGEM (CRÍTICO):\n"
        "- Use verbos IMPERATIVOS: 'use', 'comece por', 'evite', 'considere',\n"
        "  'reforce', 'pratique de novo', 'mude o ritmo aqui'.\n"
        "- ZERO jargão técnico (jitter, MFCC, spectral flux, microtremor,\n"
        "  'deslocamento prosódico'). Traduza tudo pra linguagem comum:\n"
        "    'voz mais firme' / 'voz mais hesitante' / 'tom muito controlado'\n"
        "    / 'pareceu ensaiada' / 'fluida e natural' / 'tensão no fim da frase'\n"
        "- Mostre ao advogado COMO ATUAR, não só O QUE OBSERVAR. Cada\n"
        "  ponto deve ter ação clara ('faça assim', 'evite isso').\n"
        "- NÃO classifique respostas como verdade/mentira nem dê percentuais.\n"
        "  Linguagem comparativa subjetiva ('parece mais natural quando...').\n"
        "- Trate como ferramenta de preparação do escritório, NÃO evidência.\n"
        "- Tom: humano, próximo, direto. Como um mentor sussurrando no ouvido,\n"
        "  não um perito escrevendo laudo.\n\n"
        "ESTRUTURA HTML (use <h2> + <p> + <ul><li>, sem CSS/JS embutido):\n"
        "  <h2>Como foi a sessão</h2>\n"
        "  <p>2-3 frases de visão geral em linguagem leiga. Diga ao advogado\n"
        "  o quê esperar do cliente na audiência baseado no que viu hoje.</p>\n\n"
        "  <h2>Pontos fortes — use a favor</h2>\n"
        "  <ul>\n"
        "    <li>Pergunta X: cliente respondeu firme, natural. <strong>Faça</strong>\n"
        "    isso virar uma âncora — peça a versão completa logo no início,\n"
        "    cria credibilidade.</li>\n"
        "    <li>(repete por resposta sólida)</li>\n"
        "  </ul>\n\n"
        "  <h2>Pontos a ajustar antes da audiência</h2>\n"
        "  <ul>\n"
        "    <li>Pergunta Y: voz travou aqui — provavelmente decorou. <strong>Pratique\n"
        "    de novo</strong> sem o texto, deixe ele(a) reformular com palavras próprias.</li>\n"
        "    <li>(repete por resposta que pediu atenção)</li>\n"
        "  </ul>\n\n"
        "  <h2>Como conduzir a inquirição</h2>\n"
        "  <p>2-3 frases táticas concretas: por onde começar, o que evitar,\n"
        "  como reagir se o advogado contrário pressionar áreas frágeis.</p>\n\n"
        "TAMANHO: 350-550 palavras. Português brasileiro. Sem números\n"
        "percentuais, sem nomes de features prosódicas, sem 'baseline'.\n\n"
        "DADOS DA SESSÃO (use pra basear o briefing, NÃO repasse os números):\n"
        f"{json.dumps(summary, indent=2, ensure_ascii=False)}\n\n"
        "Escreva o briefing tático agora."
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
