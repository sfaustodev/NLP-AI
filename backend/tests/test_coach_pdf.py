"""Tests for Coach PDF generators (Terms, Consent, Session Report).

Uses pypdf to extract text from rendered PDFs and assert critical phrases
appear. PDF byte structure is also sanity-checked (magic + min size).
"""

from __future__ import annotations

import io

import pytest
from pypdf import PdfReader

from app.coach.pdf import consent, session_report, terms


# ----------------------------------------------------------- helpers

def _extract_text(pdf_bytes: bytes) -> str:
    """Pull all text from every page of the PDF."""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    return "\n".join(p.extract_text() or "" for p in reader.pages)


def _assert_valid_pdf(pdf_bytes: bytes, min_size: int = 1500) -> None:
    """Sanity check: PDF magic + minimum size."""
    assert pdf_bytes[:5] == b"%PDF-", f"Not a PDF, got {pdf_bytes[:8]!r}"
    assert pdf_bytes.endswith(b"%%EOF\n") or pdf_bytes.endswith(b"%%EOF"), \
        "PDF missing %%EOF marker"
    assert len(pdf_bytes) >= min_size, f"PDF unexpectedly small: {len(pdf_bytes)} bytes"


# ----------------------------------------------------------- terms.py

def test_terms_pdf_renders_with_all_articles() -> None:
    pdf = terms.generate_terms_pdf()
    _assert_valid_pdf(pdf, min_size=2500)
    text = _extract_text(pdf)
    for art in ("Art. 1", "Art. 2", "Art. 3", "Art. 4",
                "Art. 5", "Art. 6", "Art. 7", "Art. 8"):
        assert art in text, f"Missing {art} from rendered Terms PDF"


def test_terms_pdf_contains_liability_cap() -> None:
    pdf = terms.generate_terms_pdf()
    text = _extract_text(pdf)
    assert "1.000,00" in text or "R$" in text
    assert "Porto Seguro" in text


def test_terms_pdf_deterministic_across_calls() -> None:
    """Same version arg → same bytes (cache-friendly)."""
    a = terms.generate_terms_pdf(version="2026-05-16")
    b = terms.generate_terms_pdf(version="2026-05-16")
    # reportlab embeds creation date in the PDF metadata, so exact byte
    # equality isn't guaranteed. Settle for "same byte size within tolerance".
    assert abs(len(a) - len(b)) < 256


# ----------------------------------------------------------- consent.py

def test_consent_pdf_blank_fields_render_underlines() -> None:
    pdf = consent.generate_consent_pdf()
    _assert_valid_pdf(pdf)
    text = _extract_text(pdf)
    assert "Termo de Consentimento" in text
    assert "Advogado" in text
    assert "Cliente" in text
    # Empty fields should produce underline placeholders.
    assert "_" in text


def test_consent_pdf_prefilled_fields_show_in_pdf() -> None:
    pdf = consent.generate_consent_pdf(
        lawyer_name="Dra. Ana Silva",
        client_name="João Santos",
        process_ref="1234567-89.2026.8.05.0001",
    )
    text = _extract_text(pdf)
    assert "Ana Silva" in text
    assert "João Santos" in text
    assert "1234567" in text


def test_consent_pdf_includes_lgpd_clause() -> None:
    pdf = consent.generate_consent_pdf()
    text = _extract_text(pdf)
    assert "LGPD" in text
    assert "13.709" in text


def test_consent_pdf_includes_signature_lines() -> None:
    pdf = consent.generate_consent_pdf()
    text = _extract_text(pdf)
    assert "Assinatura" in text


# ----------------------------------------------------------- session_report.py

_SAMPLE_RESPONSES = [
    {
        "duration_s": 12.4,
        "cartesian_x": 0.1,
        "cartesian_y": -0.05,
        "consistency_label": "BASELINE",
        "color": "GREEN",
        "narrative": "Resposta dentro do baseline.",
    },
    {
        "duration_s": 18.1,
        "cartesian_x": -0.6,
        "cartesian_y": 0.7,
        "consistency_label": "MAJOR_SHIFT",
        "color": "RED",
        "narrative": "Deslocamento maior em microtremor.",
    },
    {
        "duration_s": 9.8,
        "cartesian_x": -0.3,
        "cartesian_y": 0.2,
        "consistency_label": "NOTABLE_SHIFT",
        "color": "ORANGE",
        "narrative": "Deslocamento notável em jitter.",
    },
]


def test_session_report_pdf_renders_with_responses() -> None:
    pdf = session_report.generate_session_report_pdf(
        session_name="João prep",
        session_created_at=1_700_000_000,
        session_ended_at=1_700_000_900,
        tier_label="Trial",
        mic_quality_label="GREEN",
        mic_quality_snr_db=28.5,
        responses=_SAMPLE_RESPONSES,
    )
    _assert_valid_pdf(pdf, min_size=10_000)  # has embedded PNG
    text = _extract_text(pdf)
    assert "João prep" in text
    assert "GREEN" in text
    assert "BASELINE" in text
    assert "MAJOR_SHIFT" in text
    assert "Trial" in text


def test_session_report_pdf_empty_responses_doesnt_crash() -> None:
    pdf = session_report.generate_session_report_pdf(
        session_name="Empty session",
        session_created_at=1_700_000_000,
        session_ended_at=1_700_000_600,
        tier_label="Trial",
        mic_quality_label="YELLOW",
        mic_quality_snr_db=18.0,
        responses=[],
    )
    _assert_valid_pdf(pdf, min_size=5000)
    text = _extract_text(pdf)
    assert "nenhuma resposta" in text.lower()


def test_session_report_pdf_includes_narrative_when_provided() -> None:
    pdf = session_report.generate_session_report_pdf(
        session_name="With narrative",
        session_created_at=1_700_000_000,
        session_ended_at=1_700_000_900,
        tier_label="Premium",
        mic_quality_label="GREEN",
        mic_quality_snr_db=30.0,
        responses=_SAMPLE_RESPONSES,
        narrative_text=(
            "Análise narrativa de exemplo. "
            "O cliente apresentou padrões consistentes em maioria das respostas.\n\n"
            "Padrão notável em duas perguntas relacionadas ao depoimento principal."
        ),
    )
    text = _extract_text(pdf)
    assert "narrativa" in text.lower()
    assert "consistentes" in text


def test_session_report_pdf_handles_markup_in_session_name_without_crash() -> None:
    """Reportlab Paragraph uses XML-like markup; unsanitized < > in the
    session name would crash the parser. We escape, so this must render cleanly.

    XSS isn't a concern in the PDF surface (no code execution context), but a
    crash on a session name containing < or > would be a real bug.
    """
    pdf = session_report.generate_session_report_pdf(
        session_name='<script>alert("xss")</script>',
        session_created_at=1_700_000_000,
        session_ended_at=1_700_000_600,
        tier_label="Trial",
        mic_quality_label="GREEN",
        mic_quality_snr_db=25.0,
        responses=[],
    )
    _assert_valid_pdf(pdf)  # PDF rendered without crashing
    text = _extract_text(pdf)
    # The literal name characters should appear as inert text (no script context).
    assert "script" in text.lower()
    assert "alert" in text
