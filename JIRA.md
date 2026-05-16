# JIRA.md — ticket index

> Cross-link tickets ↔ commits ↔ PRs.
> No Jira project opened yet for Vox Probabilis. Tracking by phase label until Juan creates one.

---

## Active

### VOX-LANDING-A · marketing landing 3-tab + per-product Terms
- **Status:** 🟡 Em desenvolvimento · Phase A local complete · aguardando autorização Phase B prod
- **Branch:** `c/modest-napier-805905`
- **Specs:** `landing_page/SPECS/SPEC_COACH.md` v0.1.1, `landing_page/SPECS/SPEC_ACADEMIC.md` v0.1.0
- **Goal:** advogado amigo abre `https://voxprobabilis.com`, vê 3 produtos (Explorer/Academic/Coach) com pricing transparente + Terms sérios, sente que é produto de verdade
- **Phase A commits:**
  - `19062b8` feat: import marketing site assets (4 files · 70KB)
  - `c34143b` docs: import Coach + Academic v0.1 specs
  - `355159a` fix: rewrite marketing asset paths to /m/* mount
  - `f3ef69d` feat: add Coach + Academic Terms HTML (Art. 1º-8º verbatim)
  - `2cb04f1` fix: wire footer + banner links to Terms HTML
  - `38589fa` feat: add VOX_MARKETING_DIR env var
  - `dd89edf` feat: add marketing routes and /m static mount
  - `f463b95` test: cover new marketing routes and v1 regression
- **Próximas tasks (checklist):**
  - [x] A.1 import assets + SPECs
  - [x] A.2 rewrite asset paths to /m/*
  - [x] A.3 generate 3 Terms HTML (coach + academic + hub)
  - [x] A.4 wire footer + banner links
  - [x] A.5 FastAPI routes (/, /app, /terms, /coach/terms, /academic/terms) + /m mount + VOX_MARKETING_DIR
  - [x] A.6 11 testes novos (test_landing.py) + ajuste test_terms_page_served (test_api.py)
  - [ ] A.8 codex-cross-review do diff Phase A
  - [ ] B production deploy (aguarda autorização escrita rule #16D)
- **Pricing tier:** CTA-only (`href="#"`) per SPRINT.md §0 #6 + resposta humana 2026-05-16 q2 — checkout real fica pra sprint posterior (`VOX-COACH-B` candidato)
- **Pre-merge tests local:** 16/16 verdes em test_landing.py + non-audio test_api.py. 10 falhas pré-existentes em audio tests (llvmlite ABI mismatch venv local Python 3.13) — não relacionadas a esta mudança.

---

### VOX-DEPLOY-A · v0.1 production deploy
- **Status:** Done · confirmado por humano 2026-05-10
- **Branch:** master
- **Spec:** `landing_page/DEPLOY.md`
- **Live URL:** https://voxprobabilis.com
- **Sub-phases:** A0–A8 (local) · B (Cloudflare/VPS prep) · C (VPS deploy) · D (smoke + DoD) · E (close)
- **Done:** A0 (`8b24839`) · A1 (`08280f0`) · A2 (`337dcb4`) · A3 (`7448297`) · A4 (`e49368d`) · A5 (`5d6cde4`) · A6 (`738e9b7`) · A7 (`6e88ce4`) · A8 (`eef165a`) · close-A (`c42e36a`) · nginx :80 + py3.12 (`e9c8ea8`) · systemd StartLimit fix (`d35c9c0`) · HEAD method (`3d1c45d`) · numba cache (`8341769`) · close-B/C/D (`0c7ff71`)
- **Phase D smoke results:** §10 #1–#8 all 200/expected. Lie sample → `OVER_CONTROLLED_TENSE`, confidence high. Rate limit triggers on 4th call.
- **Confirmed by human:** "testei tudo passou" — Juan, 2026-05-10. Web Analytics enabled. Origin cert backed up in `.env` / `backend/secrets/`.

---

## Done

### VOX-DEPLOY-A · v0.1 production deploy · closed 2026-05-10
- Live at https://voxprobabilis.com per DEPLOY.md §15 DoD.
- Smoke §10 #1–#8 green. Browser §10.1 confirmed by Juan.
- Cloudflare Web Analytics on. Origin cert backed up.
- 14 commits across Phase A/B/C/D + close-out. 43/43 local tests green.

---

## Notes

- If Juan opens a Jira project, migrate this index to use `SCRUM-XX` IDs and link the corresponding tickets.
- Until then, commits reference `VOX-DEPLOY-A:<phase>` in the scope (e.g. `feat(VOX-DEPLOY-A:A2): add /api/metrics`).
