"""Vox Probabilis Coach — live interactive practice tool for lawyers.

Module layout:
- session.py — state machine (CREATED → AWAITING_CALIBRATION → READY → IN_PRACTICE → ENDED)
- auth.py — HMAC session_token + lawyer activation_token
- mic_quality.py — SNR/sample rate/centroid → GREEN/YELLOW/RED
- baseline.py — per-session immutable 4-feature baseline
- feedback.py — delta vs baseline + consistency_label + cartesian projection
- pricing.py — tier definitions + quota enforcement
- routes.py — 7 FastAPI endpoints
- reports/sonnet_standard.py — Anthropic Sonnet narrative report
- pdf/{terms,consent,session_report}.py — reportlab generators
- cli.py — manual tier activation (Faustão's escape hatch until VOX-COACH-D)

Spec: landing_page/SPECS/SPEC_COACH.md v0.1.1
Ticket: VOX-COACH-B (scope: bare T1, no Cofre, no checkout).
"""
