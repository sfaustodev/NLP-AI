# JIRA.md — ticket index

> Cross-link tickets ↔ commits ↔ PRs.
> No Jira project opened yet for Vox Probabilis. Tracking by phase label until Juan creates one.

---

## Active

### VOX-LANDING-A · marketing landing 3-tab + per-product Terms
- **Status:** 🟡 In Review · live em prod 2026-05-16 22:12 UTC · aguardando confirmação browser test Faustão (rule #13) pra fechar
- **Branch:** `c/modest-napier-805905` → merged via PR #1 em master (`f207038`)
- **Specs:** `landing_page/SPECS/SPEC_COACH.md` v0.1.1, `landing_page/SPECS/SPEC_ACADEMIC.md` v0.1.0
- **Live URL:** https://voxprobabilis.com (homepage agora marketing 3-tab; v0.1 ferramenta em /app)
- **PR:** [NLP-AI#1](https://github.com/sfaustodev/NLP-AI/pull/1) · merged 2026-05-16 22:11 UTC
- **Goal:** advogado amigo abre `https://voxprobabilis.com`, vê 3 produtos (Explorer/Academic/Coach) com pricing transparente + Terms sérios, sente que é produto de verdade
- **Phase A commits (9):**
  - `19062b8` feat: import marketing site assets (4 files · 70KB)
  - `c34143b` docs: import Coach + Academic v0.1 specs
  - `355159a` fix: rewrite marketing asset paths to /m/* mount
  - `f3ef69d` feat: add Coach + Academic Terms HTML (Art. 1º-8º verbatim)
  - `2cb04f1` fix: wire footer + banner links to Terms HTML
  - `38589fa` feat: add VOX_MARKETING_DIR env var
  - `dd89edf` feat: add marketing routes and /m static mount
  - `f463b95` test: cover new marketing routes and v1 regression
  - `c647eb4` docs: open umbrella ticket + Phase A diary entry
- **Phase B prod deploy (2026-05-16 22:12 UTC):**
  - Rollback anchor SHA: `8341769` (pré-deploy)
  - `.env` backup: `/opt/voxprobabilis/.env.bak.20260516-221224`
  - `git pull origin master` → HEAD `f207038` (17 files / 3655 insertions)
  - `systemctl restart voxprobabilis` → active, 161MB memory, workers up
  - Smoke curl laptop (via Cloudflare): 9 rotas 200 (`/` `/app` `/coach/terms` `/academic/terms` `/terms` `/m/static/style.css` `/m/audiencia_cartesian.png` `/api/health` `/privacy`)
  - Content verified: 3 tabs no `/`, Art. 1º + R$ 1.000 em /coach/terms, R$ 500 em /academic/terms, hub linkando ambos
  - Security headers v0.1 herdados nas novas rotas: X-Frame-Options DENY, X-Content-Type-Options nosniff, Referrer-Policy strict-origin-when-cross-origin
  - Sem regressão v0.1: `/privacy` 200, `/api/health` 200 OK, `/assets/*` mantido
- **Checklist:**
  - [x] A.1-A.8 Phase A local (9 commits + 11 tests + uvicorn smoke local)
  - [x] PR #1 created + merged via merge-commit (preserva 9 commits atômicos per CLAUDE.md)
  - [x] B.1-9 prod deploy + smoke (9 rotas verde, 6 content checks verde)
  - [ ] B.10 sacred files update Phase B (este commit)
  - [ ] Faustão browser test incognito (3 tabs, Coach Terms Art. 1º-8º, /app v0.1 ainda funcional)
  - [ ] Faustão manda URL pro adv amigo (gatilho do produto)
  - [ ] Confirmação escrita Faustão pra fechar VOX-LANDING-A (rule #13)
- **Pricing tier:** CTA-only (`href="#"`) per SPRINT.md §0 #6 + resposta humana 2026-05-16 q2 — checkout real fica pra sprint posterior (`VOX-COACH-B` candidato)
- **Pre-merge tests local:** 16/16 verdes em test_landing.py + non-audio test_api.py. 10 falhas pré-existentes em audio tests (llvmlite ABI mismatch venv local Python 3.13) — não relacionadas a esta mudança. Prod usa Python 3.12.
- **Rollback (se quebrar):** `cd /opt/voxprobabilis && sudo -u vox git reset --hard 8341769 && sudo systemctl restart voxprobabilis` (~30s recovery)

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
