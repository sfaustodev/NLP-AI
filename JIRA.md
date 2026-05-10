# JIRA.md — ticket index

> Cross-link tickets ↔ commits ↔ PRs.
> No Jira project opened yet for Vox Probabilis. Tracking by phase label until Juan creates one.

---

## Active

### VOX-DEPLOY-A · v0.1 production deploy
- **Status:** In progress
- **Branch:** master (no ticket branches yet — solo deploy)
- **Spec:** `landing_page/DEPLOY.md`
- **Sub-phases:** A0 (discipline) · A1 (LIVENESS_MODE) · A2 (/api/metrics) · A3 (wire.js) · A4 (LGPD pages) · A5 (nginx hardening) · A6 (systemd port) · A7 (backup) · A8 (README) · C (VPS deploy) · D (smoke + DoD)
- **Done:** —
- **Open:** all of A
- **Confirmed by human:** pending — see global rule #13

---

## Done

_None yet._

---

## Notes

- If Juan opens a Jira project, migrate this index to use `SCRUM-XX` IDs and link the corresponding tickets.
- Until then, commits reference `VOX-DEPLOY-A:<phase>` in the scope (e.g. `feat(VOX-DEPLOY-A:A2): add /api/metrics`).
