# JIRA.md — ticket index

> Cross-link tickets ↔ commits ↔ PRs.
> No Jira project opened yet for Vox Probabilis. Tracking by phase label until Juan creates one.

---

## Active

### VOX-DEPLOY-A · v0.1 production deploy
- **Status:** Phase A/B/C/D done. Phase E open — awaiting Juan's "testei tudo, passou sem bugs".
- **Branch:** master
- **Spec:** `landing_page/DEPLOY.md`
- **Live URL:** https://voxprobabilis.com
- **Sub-phases:** A0–A8 (local) · B (Cloudflare/VPS prep) · C (VPS deploy) · D (smoke + DoD)
- **Done:** A0 (`8b24839`) · A1 (`08280f0`) · A2 (`337dcb4`) · A3 (`7448297`) · A4 (`e49368d`) · A5 (`5d6cde4`) · A6 (`738e9b7`) · A7 (`6e88ce4`) · A8 (`eef165a`) · close-A (`c42e36a`) · nginx :80 + py3.12 (`e9c8ea8`) · systemd StartLimit fix (`d35c9c0`) · HEAD method (`3d1c45d`) · numba cache (`8341769`)
- **Phase D smoke results:** §10 #1–#8 all 200/expected. Lie sample → `OVER_CONTROLLED_TENSE`, confidence high. Rate limit triggers on 4th call.
- **Open:** Juan's browser §10.1 incognito + DoD §15 walkthrough
- **Confirmed by human:** pending — see global rule #13

---

## Done

_None yet._

---

## Notes

- If Juan opens a Jira project, migrate this index to use `SCRUM-XX` IDs and link the corresponding tickets.
- Until then, commits reference `VOX-DEPLOY-A:<phase>` in the scope (e.g. `feat(VOX-DEPLOY-A:A2): add /api/metrics`).
