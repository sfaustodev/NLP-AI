"""Session report PDF — header + cartesian plot + responses table + LLM narrative.

Combines reportlab (page layout) with matplotlib (Agg backend, no display) to
embed the cartesian projection PNG inside the PDF. The PNG is generated in
memory — never touches disk.
"""

from __future__ import annotations

from datetime import datetime, timezone
from io import BytesIO
from typing import Iterable, Mapping, Optional

# matplotlib Agg backend BEFORE pyplot import — no $DISPLAY required.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reportlab.lib.units import cm
from reportlab.platypus import (
    Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)
from reportlab.lib import colors as rl_colors

from ._style import (
    BONE_DIM, CRIMSON, INK, LINE, MARGIN_BOTTOM, MARGIN_LEFT, MARGIN_RIGHT,
    MARGIN_TOP, MUTED, PAGE_SIZE, vox_styles,
)


# Map consistency_label.color → matplotlib color for the cartesian plot.
COLOR_MAP = {
    "GREEN":  "#3a7a3a",
    "YELLOW": "#a89878",
    "ORANGE": "#b87a3a",
    "RED":    "#c33a3a",
}


def _render_cartesian_png(responses: list[Mapping]) -> bytes:
    """Render a 400×400 cartesian plot PNG in memory."""
    fig, ax = plt.subplots(figsize=(4, 4), dpi=120)

    # Quadrant guides.
    ax.axhline(0, color="#3a1a1a", linewidth=0.6)
    ax.axvline(0, color="#3a1a1a", linewidth=0.6)

    if responses:
        xs = [float(r.get("cartesian_x") or 0.0) for r in responses]
        ys = [float(r.get("cartesian_y") or 0.0) for r in responses]
        cs = [COLOR_MAP.get(r.get("color"), "#888888") for r in responses]
        ax.scatter(xs, ys, c=cs, s=80, edgecolors="#0a0a0a", linewidths=0.6, zorder=3)

        # Connect points by index to show trajectory (helps reader follow flow).
        ax.plot(xs, ys, color="#2A211F", linewidth=0.8, alpha=0.4, zorder=2)

        # Annotate each point with response number.
        for i, (x, y) in enumerate(zip(xs, ys), start=1):
            ax.annotate(str(i), (x, y),
                        textcoords="offset points", xytext=(8, 4),
                        fontsize=8, color="#0a0a0a")

    ax.set_xlim(-1.1, 1.1)
    ax.set_ylim(-1.1, 1.1)
    ax.set_xlabel("← Over-controlled    Naturalness    Elevated →", fontsize=8, color="#7A6F6A")
    ax.set_ylabel("← Calm    Involuntary Stress    Tense →", fontsize=8, color="#7A6F6A")
    ax.set_facecolor("#fafafa")
    ax.grid(True, color="#e8dfd0", linewidth=0.4)
    ax.tick_params(labelsize=7, colors="#7A6F6A")
    for spine in ax.spines.values():
        spine.set_color("#2A211F")
        spine.set_linewidth(0.6)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return buf.getvalue()


def _fmt_ts(unix_ts: int) -> str:
    """Format unix ts as 'YYYY-MM-DD HH:MM UTC'."""
    return datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _responses_table(responses: list[Mapping], styles) -> Table:
    """Build a reportlab Table with one row per response."""
    header = ["#", "Label", "Cor", "Duração", "Narrativa"]
    rows = [header]
    for i, r in enumerate(responses, start=1):
        rows.append([
            str(i),
            r.get("consistency_label", "—"),
            r.get("color", "—"),
            f"{float(r.get('duration_s', 0)):.1f}s",
            Paragraph((r.get("narrative") or "").replace("\n", "<br/>"),
                      styles["body"]),
        ])
    tbl = Table(rows, colWidths=[0.8 * cm, 3.2 * cm, 1.8 * cm, 1.8 * cm, 8.4 * cm])
    tbl.setStyle(TableStyle([
        ("FONT", (0, 0), (-1, 0), "Courier-Bold", 8),
        ("TEXTCOLOR", (0, 0), (-1, 0), CRIMSON),
        ("FONT", (0, 1), (-1, -1), "Times-Roman", 9),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, LINE),
        ("LINEBELOW", (0, 1), (-1, -1), 0.2, BONE_DIM),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return tbl


def generate_session_report_pdf(
    *,
    session_name: str,
    session_created_at: int,
    session_ended_at: int,
    tier_label: str,
    mic_quality_label: str,
    mic_quality_snr_db: float,
    responses: list[Mapping],
    narrative_text: Optional[str] = None,
) -> bytes:
    """Render the post-session PDF report.

    ``narrative_text`` is the plain-text Sonnet narrative (the route handler
    has already stripped HTML tags for the PDF embedding). If None, the PDF
    shows only the table + plot — appropriate for FREE_TRIAL.
    """
    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=PAGE_SIZE,
        leftMargin=MARGIN_LEFT, rightMargin=MARGIN_RIGHT,
        topMargin=MARGIN_TOP, bottomMargin=MARGIN_BOTTOM,
        title=f"Coach — {session_name} — {_fmt_ts(session_ended_at)}",
        author="Vox Probabilis",
    )
    styles = vox_styles()

    flow = [
        Paragraph("VOX PROBABILIS · COACH", styles["crumb"]),
        Paragraph(_escape(session_name), styles["title"]),
        Paragraph(
            f"Plano: <b>{_escape(tier_label)}</b> · {_fmt_ts(session_created_at)} → {_fmt_ts(session_ended_at)}",
            styles["subtitle"]),
        Paragraph(
            f"Microfone: <b>{mic_quality_label}</b> (SNR {mic_quality_snr_db:.1f} dB) · "
            f"{len(responses)} respostas analisadas",
            styles["meta"]),
    ]

    # Cartesian plot.
    flow.append(Paragraph("Projeção cartesiana", styles["h2"]))
    png = _render_cartesian_png(responses)
    img = Image(BytesIO(png), width=10 * cm, height=10 * cm)
    flow.append(img)
    flow.append(Spacer(1, 0.3 * cm))

    # Responses table.
    flow.append(Paragraph("Respostas", styles["h2"]))
    if responses:
        flow.append(_responses_table(responses, styles))
    else:
        flow.append(Paragraph("(nenhuma resposta gravada nesta sessão)", styles["body"]))

    # Narrative footer.
    if narrative_text:
        flow.append(Spacer(1, 0.6 * cm))
        flow.append(Paragraph("Análise narrativa", styles["h2"]))
        for paragraph in narrative_text.split("\n\n"):
            cleaned = paragraph.strip()
            if cleaned:
                flow.append(Paragraph(_escape(cleaned), styles["body"]))

    # Methodology footer.
    flow.append(Spacer(1, 0.8 * cm))
    flow.append(Paragraph(
        "Análise prosódica comparativa baseada em jitter, MFCC delta variance, "
        "spectral flux e microtremor 8-12 Hz. Não classifica veracidade. "
        "Documento de uso interno do escritório — vedado anexar a peças "
        "processuais (Coach Termos de Uso, Art. 3º).",
        styles["meta"]))

    doc.build(flow)
    return buf.getvalue()


def _escape(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
              .replace(">", "&gt;"))
