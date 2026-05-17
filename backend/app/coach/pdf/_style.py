"""Shared reportlab style helpers — Vox Probabilis paleta.

Times Roman (reportlab built-in) is used instead of Cormorant Garamond /
Inter to avoid bundling external font files. Visual mismatch with the HTML
Coach pages is acceptable for the PDF artifacts (Terms / Consent / Session
Report) — they're rendered server-side and consumed in PDF readers, not
viewed alongside the HTML.
"""

from __future__ import annotations

from reportlab.lib.colors import HexColor
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm


# Vox Probabilis palette (matches CSS tokens in coach-terms.html / terms-hub.html).
INK      = HexColor("#0A0808")
INK_2    = HexColor("#120E0D")
BONE     = HexColor("#E8DFD0")
BONE_DIM = HexColor("#BFB4A4")
MUTED    = HexColor("#7A6F6A")
CRIMSON  = HexColor("#B2231F")
LINE     = HexColor("#2A211F")

# Page dimensions.
PAGE_SIZE = A4
MARGIN_LEFT = MARGIN_RIGHT = 2.2 * cm
MARGIN_TOP = MARGIN_BOTTOM = 2.0 * cm


def vox_styles():
    """Build a Vox Probabilis ParagraphStyle sheet — title, h2, body, art,
    meta, liability, signature."""
    base = getSampleStyleSheet()

    title = ParagraphStyle(
        "VoxTitle", parent=base["Heading1"],
        fontName="Times-Roman", fontSize=26, leading=30,
        textColor=INK, spaceAfter=4,
    )
    subtitle = ParagraphStyle(
        "VoxSubtitle", parent=base["Italic"],
        fontName="Times-Italic", fontSize=12, leading=16,
        textColor=MUTED, spaceAfter=18,
    )
    meta = ParagraphStyle(
        "VoxMeta", parent=base["Normal"],
        fontName="Courier", fontSize=9, leading=12,
        textColor=MUTED, spaceAfter=24,
    )
    h2 = ParagraphStyle(
        "VoxH2", parent=base["Heading2"],
        fontName="Times-Roman", fontSize=14, leading=18,
        textColor=INK, spaceBefore=14, spaceAfter=2,
    )
    art = ParagraphStyle(
        "VoxArt", parent=base["Normal"],
        fontName="Courier-Bold", fontSize=9, leading=11,
        textColor=CRIMSON, spaceAfter=2, spaceBefore=14,
    )
    body = ParagraphStyle(
        "VoxBody", parent=base["Normal"],
        fontName="Times-Roman", fontSize=10.5, leading=15,
        textColor=INK, spaceAfter=8, alignment=4,  # 4 = TA_JUSTIFY
    )
    liability = ParagraphStyle(
        "VoxLiability", parent=body,
        fontName="Times-Italic", fontSize=10.5, leading=15,
        leftIndent=12, borderColor=CRIMSON, borderWidth=0,
        backColor=HexColor("#FAF2F1"),
        spaceAfter=10, spaceBefore=6,
    )
    sig = ParagraphStyle(
        "VoxSig", parent=base["Normal"],
        fontName="Times-Roman", fontSize=10, leading=14,
        spaceBefore=24, spaceAfter=6, textColor=INK,
    )
    crumb = ParagraphStyle(
        "VoxCrumb", parent=base["Normal"],
        fontName="Courier-Bold", fontSize=8, leading=12,
        textColor=MUTED, spaceAfter=4,
    )

    return {
        "crumb": crumb,
        "title": title,
        "subtitle": subtitle,
        "meta": meta,
        "h2": h2,
        "art": art,
        "body": body,
        "liability": liability,
        "sig": sig,
    }
