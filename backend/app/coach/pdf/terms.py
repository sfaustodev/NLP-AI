"""Vox Probabilis Coach — Terms of Use PDF (SPEC §8.1, Art. 1º-8º verbatim).

Body text mirrors ``landing_page/marketing/coach-terms.html`` so the HTML and
PDF stay aligned. Update both when the spec changes.
"""

from __future__ import annotations

from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

from ._style import (
    MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, PAGE_SIZE,
    vox_styles,
)


TERMS_VERSION = "2026-05-16"


ARTICLES: tuple[tuple[str, str, str], ...] = (
    (
        "Art. 1º", "Definição",
        "Coach é ferramenta interativa de preparação. Não é dispositivo forense, "
        "não é polígrafo, não é evidência judicial.",
    ),
    (
        "Art. 2º", "Finalidade exclusiva",
        "Uso restrito a sessões 1-on-1 entre advogado(a) e cliente(s) ou "
        "testemunha(s) da causa em que atua. Vedado o uso em sessões com partes "
        "contrárias, com testemunhas arroladas por outras partes sem o conhecimento "
        "delas, ou em qualquer contexto que viole o sigilo profissional do advogado.",
    ),
    (
        "Art. 3º", "Vedação ao uso processual",
        "É absolutamente vedado anexar relatórios produzidos pelo Coach a peças "
        "processuais, ou submetê-los como evidência em qualquer foro. Coach é "
        "ferramenta de preparação privada; seus outputs são para uso interno do "
        "escritório.",
    ),
    (
        "Art. 4º", "Consentimento do cliente",
        "O advogado declara, ao iniciar uma sessão, que obteve consentimento "
        "informado do cliente para a análise prosódica. Modelo de consentimento "
        "fornecido em /coach/consent-template.pdf.",
    ),
    (
        "Art. 5º", "LGPD",
        "Voz é dado pessoal sensível. Coach processa o áudio em memória, armazena "
        "apenas features extraídas (números) nos planos Tier 2+ que oferecem "
        "retenção, descarta o áudio bruto em todos os planos dentro de 60 segundos "
        "do processamento, e permite ao advogado solicitar exclusão da sessão a "
        "qualquer momento. Base legal: consentimento explícito do titular do dado "
        "(cliente/testemunha) intermediado pelo advogado.",
    ),
    (
        "Art. 6º", "Sigilo profissional",
        "O Coach não armazena conteúdo semântico das respostas. Apenas features "
        "prosódicas (4 números por resposta) e timestamps. As perguntas do advogado, "
        "quando fornecidas como texto, são armazenadas para o relatório — o advogado "
        "é responsável por não incluir matéria coberta por sigilo nesse campo.",
    ),
    (
        "Art. 7º", "Limitação de responsabilidade",
        "A responsabilidade civil máxima agregada do Operador, decorrente de "
        "qualquer reclamação relacionada ao serviço, fica limitada ao <b>maior</b> "
        "dos seguintes valores: <b>(i)</b> o total efetivamente pago pelo Usuário "
        "ao Operador nos 12 (doze) meses anteriores ao evento que ensejar a "
        "reclamação; ou <b>(ii)</b> R$ 1.000,00 (mil reais). Esta limitação não se "
        "aplica em caso de dolo ou culpa grave comprovada do Operador.",
    ),
    (
        "Art. 8º", "Foro",
        "Foro da Comarca de Porto Seguro, Bahia, com renúncia expressa a qualquer "
        "outro, por mais privilegiado que seja.",
    ),
)


def generate_terms_pdf(*, version: str = TERMS_VERSION) -> bytes:
    """Render the Coach Terms of Use as a single-document PDF.

    The output is deterministic given the same ``version`` string — no
    timestamps, no random IDs — which means the PDF endpoint is safe to
    cache aggressively at the CDN layer.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=PAGE_SIZE,
        leftMargin=MARGIN_LEFT, rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
        title="Vox Probabilis Coach — Termos de Uso",
        author="Vox Probabilis",
    )
    styles = vox_styles()

    flow = [
        Paragraph("VOX PROBABILIS · COACH", styles["crumb"]),
        Paragraph("Termos de Uso", styles["title"]),
        Paragraph(
            "Ferramenta interativa de preparação pré-audiência para advogados em exercício.",
            styles["subtitle"]),
        Paragraph(f"Versão {version} · jurisdição: Brasil", styles["meta"]),
    ]

    for num, heading, body in ARTICLES:
        flow.append(Paragraph(num, styles["art"]))
        flow.append(Paragraph(heading, styles["h2"]))
        # Article 7 is the liability cap — use highlighted style to mirror HTML.
        para_style = styles["liability"] if num == "Art. 7º" else styles["body"]
        flow.append(Paragraph(body, para_style))

    flow.append(Spacer(1, 1 * cm))
    flow.append(Paragraph(
        "Documento gerado automaticamente. Versão canônica em "
        "<font color='#B2231F'>https://voxprobabilis.com/coach/terms</font>.",
        styles["meta"]))

    doc.build(flow)
    return buf.getvalue()
