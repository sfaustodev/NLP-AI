"""Printable client consent template (SPEC §8.2).

The lawyer prints this and obtains client signature before recording. Fields
can be left blank and filled by hand, or pre-populated when the lawyer has
already typed the client info into the dashboard.
"""

from __future__ import annotations

from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import cm

from ._style import (
    MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT, MARGIN_TOP, PAGE_SIZE,
    vox_styles,
)


CONSENT_VERSION = "2026-05-16"


def _line(label: str, value: str | None) -> str:
    """Render a labelled line as '<b>Label:</b> _____' or '<b>Label:</b> value'."""
    filled = value.strip() if value else ""
    if filled:
        return f"<b>{label}:</b> {filled}"
    underline = "_" * max(40, 60 - len(label))
    return f"<b>{label}:</b> {underline}"


def generate_consent_pdf(*, lawyer_name: str = "", client_name: str = "",
                          process_ref: str = "",
                          version: str = CONSENT_VERSION) -> bytes:
    """Generate a 1-page consent template; fields can be empty for hand-fill."""
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=PAGE_SIZE,
        leftMargin=MARGIN_LEFT, rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
        title="Termo de Consentimento — Análise Prosódica Coach",
        author="Vox Probabilis",
    )
    styles = vox_styles()

    flow = [
        Paragraph("VOX PROBABILIS · COACH", styles["crumb"]),
        Paragraph("Termo de Consentimento Informado", styles["title"]),
        Paragraph(
            "Análise prosódica com finalidade exclusiva de preparação privada.",
            styles["subtitle"]),
        Paragraph(f"Versão {version} · jurisdição: Brasil", styles["meta"]),

        Paragraph("Identificação", styles["h2"]),
        Paragraph(_line("Advogado(a)", lawyer_name), styles["body"]),
        Paragraph(_line("Cliente / Testemunha", client_name), styles["body"]),
        Paragraph(_line("Processo / Referência", process_ref), styles["body"]),

        Paragraph("Descrição da análise", styles["h2"]),
        Paragraph(
            "O Coach é uma ferramenta interativa de preparação pré-audiência. "
            "Grava trechos curtos de áudio do cliente, extrai 4 características "
            "espectrais (jitter, variação MFCC, fluxo espectral, microtremor) e "
            "compara-as com uma calibração feita no início da sessão. Não "
            "classifica respostas como verdadeiras ou falsas; identifica apenas "
            "deslocamentos prosódicos em relação ao baseline.",
            styles["body"]),

        Paragraph("Cláusula de consentimento", styles["h2"]),
        Paragraph(
            "Declaro que fui informado(a) sobre a natureza da análise descrita "
            "acima e que autorizo, livremente e por escrito, o(a) advogado(a) "
            "supraidentificado(a) a realizá-la no contexto de preparação privada "
            "para audiência judicial. Estou ciente de que: (i) o áudio bruto é "
            "descartado em até 60 segundos após processamento; (ii) apenas "
            "features numéricas são retidas, e podem ser excluídas a qualquer "
            "momento mediante solicitação; (iii) os resultados são de uso "
            "interno do escritório e não serão anexados a peças processuais ou "
            "submetidos como evidência em qualquer foro; (iv) posso revogar este "
            "consentimento a qualquer tempo, com efeitos prospectivos.",
            styles["body"]),

        Paragraph("Direitos LGPD", styles["h2"]),
        Paragraph(
            "Nos termos da Lei Geral de Proteção de Dados (Lei nº 13.709/2018), "
            "tenho direito a confirmação de tratamento, acesso, retificação, "
            "anonimização, portabilidade e eliminação dos dados tratados, "
            "conforme Art. 18.",
            styles["body"]),

        Spacer(1, 0.8 * cm),
        Paragraph("Assinaturas", styles["h2"]),
        Paragraph(
            "Local e data: ________________________________________________________________",
            styles["sig"]),
        Spacer(1, 0.6 * cm),
        Paragraph(
            "Assinatura do(a) cliente/testemunha: ____________________________________________",
            styles["sig"]),
        Spacer(1, 0.6 * cm),
        Paragraph(
            "Assinatura do(a) advogado(a): _________________________________________________",
            styles["sig"]),
        Spacer(1, 0.6 * cm),
        Paragraph(
            "Testemunha (opcional): ____________________________________________________________",
            styles["sig"]),
    ]

    doc.build(flow)
    return buf.getvalue()
