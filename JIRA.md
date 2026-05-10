# JIRA.md — ticket index

> Cross-link tickets ↔ commits ↔ PRs.
> No Jira project opened yet for Vox Probabilis. Tracking by phase label until Juan creates one.

---

## Active

### VOX-DEPLOY-A · v0.1 production deploy
- **Status:** Phase A done. Phase B blocked on Juan. Phase C pending Phase B.
- **Branch:** master (no ticket branches yet — solo deploy)
- **Spec:** `landing_page/DEPLOY.md`
- **Sub-phases:** A0 (discipline) · A1 (LIVENESS_MODE) · A2 (/api/metrics) · A3 (wire.js) · A4 (LGPD pages) · A5 (nginx hardening) · A6 (systemd port) · A7 (backup) · A8 (README) · C (VPS deploy) · D (smoke + DoD)
- **Done:** A0 (`8b24839`) · A1 (`08280f0`) · A2 (`337dcb4`) · A3 (`7448297`) · A4 (`e49368d`) · A5 (`5d6cde4`) · A6 (`738e9b7`) · A7 (`6e88ce4`) · A8 (`eef165a`)
- **Open:** Phase B (Juan-side: Porkbun → Cloudflare → DNS → Origin Cert → VPS apt) · Phase C (VPS deploy) · Phase D (smoke + DoD)
- **Confirmed by human:** pending — see global rule #13

---

## Done

_None yet._

---

## Notes

- If Juan opens a Jira project, migrate this index to use `SCRUM-XX` IDs and link the corresponding tickets.
- Until then, commits reference `VOX-DEPLOY-A:<phase>` in the scope (e.g. `feat(VOX-DEPLOY-A:A2): add /api/metrics`).
