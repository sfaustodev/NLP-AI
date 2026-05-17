"""PDF generators for Coach (reportlab + matplotlib).

Three documents:
- ``terms`` — Vox Probabilis Coach Term of Use (Art. 1º-8º, verbatim SPEC §8.1)
- ``consent`` — printable client consent template
- ``session_report`` — post-session report with cartesian plot + responses table

All exposed as ``generate_*_pdf(...) -> bytes``. Caller is a FastAPI route
that returns ``Response(content=bytes, media_type='application/pdf')``.
"""
