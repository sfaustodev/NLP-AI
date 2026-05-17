# JIRA.md — ticket index

> Cross-link tickets ↔ commits ↔ PRs.
> No Jira project opened yet for Vox Probabilis. Tracking by phase label until Juan creates one.

---

## Active

### VOX-COACH-B · Coach backend T1 bare (sem checkout)
- **Status:** 🟡 In Review · deploy prod 2026-05-17 03:48 UTC · aguardando Faustão browser test escrito (rule #13)
- **Branch:** `feat/vox-coach-b` → merged via PR #3 em master `d1377d1` (17 commits atomic preservados)
- **Live URL:** https://voxprobabilis.com/coach (dashboard) · `/coach/session/{token}` (live view)
- **PR:** [NLP-AI#3](https://github.com/sfaustodev/NLP-AI/pull/3) · merged 2026-05-17 03:21 UTC
- **Deploy prod:** HEAD `d1377d1` · PREV_SHA `f207038` (rollback anchor) · .env.bak.20260517-034647 · systemctl restart OK 140MB
- **Smoke prod (via Cloudflare, 12/12 verde):** `/coach` `/coach/terms.pdf` `/coach/consent-template.pdf` `/coach/static/*` 200 · `/api/coach/quota` 401 sem cookie ✓ · regression `/` `/app` `/coach/terms` `/api/health` `/privacy` mantidos
- **Faustão ativado:** `usr_Xe1YvJemmYNS9_k0wdUFMg` · FREE_TRIAL · expira 2026-06-16 · activation URL `https://voxprobabilis.com/coach/activate?token=ScUA4AI30FkzmlqcO2jPts-iMqGTEYca` (single-use, 7d TTL)
- **Spec:** `landing_page/SPECS/SPEC_COACH.md` v0.1.1
- **Goal:** advogado clica "Trial grátis" no /coach, faz sessão real (calibração + 3-5 respostas), recebe relatório Sonnet, baixa PDF. Sente o produto antes de pagar.
- **Plan file:** `/Users/peluche/.claude/plans/pr-ximo-movimento-faust-o-sobe-snuggly-balloon.md` (VOX-COACH-B revision)
- **Phase B.1 commits (9):**
  - `c371d1c` docs: sprint start + close VOX-LANDING-A + HUMAN cleanup (Q-06/07/08 resolved)
  - `b5237c3` chore: anthropic 0.40.0 + reportlab 4.2.5 + matplotlib 3.9.2 deps
  - `6322e61` feat: migrations 007/008/009 + coach pkg init
  - `1435f22` feat: session state machine + DB CRUD + **27 tests**
  - `4eb2d0c` feat: HMAC session_token + lawyer cookie + **12 tests**
  - `356d54c` feat: mic quality (SNR + sr + centroid) + **24 tests**
  - `8961e49` feat: baseline + feedback delta+cartesian+label + **34 tests**
  - `426207d` feat: tier pricing + lawyer accounts CRUD + **23 tests**
- **Tests Phase B.1:** **120 verde** em modules Coach. Audio tests v0.1 (10 falhas) pré-existente local llvmlite py3.13 — não relacionado.
- **Sub-phases (checklist):**
  - [x] B.1 backend skeleton (7 modules: session/auth/mic_quality/baseline/feedback/pricing/users + migrations 007-009 + tests)
  - [x] B.2 routes (8 endpoints `/api/coach/*`) + Sonnet 4.6 LLM + 3 PDFs reportlab + middleware + responses module
  - [x] B.3 frontend (`/coach` dashboard + session live view + MediaRecorder + polling 2s + CTA wiring)
  - [x] B.4.1 CLI tier activation + 7 tests verde
  - [x] B.4.2 smoke uvicorn local e2e (auth + quota + state machine + redirect + cookies) verde
  - [x] B.4.3 PR #3 merged master (17 commits atomic preservados)
  - [x] B.4.4 súplica prod + deploy + smoke verde + Faustão ativado FREE_TRIAL
  - [ ] Faustão browser test real + confirmação escrita "testei tudo passou"
  - [ ] (futuro) Faustão envia URL adv amigo quando se sentir confortável
- **Scope IN:** FREE_TRIAL + TIER_1_MONTHLY tiers · 7 endpoints `/api/coach/*` · Sonnet 4.6 reports · 3 PDFs reportlab · manual tier activation via CLI
- **Scope OUT (pra VOX-COACH-C/D):** Cofre features (clients/trajectory/diff/brief/tags) · Opus reports · Lemon Squeezy/Stripe checkout · Safari/mobile
- **Decisões agente sem perguntar:**
  - Lawyer auth: magic-link via CLI activation (Faustão envia link adv)
  - Real-time transport: polling 1s (upgrade futuro WebSocket/SSE se medir >4s p95)
  - Audio retention: 0s pós-feature-extraction (SPEC literal)
- **Decisões que viraram bloqueio:** CSP update separado em VOX-CSP-FIX (não bloqueia B); Anthropic API key precisa Faustão gerar manual antes do deploy
- **Deps novos:** `anthropic==0.40.0`, `reportlab==4.2.5`, `matplotlib==3.9.2`

---

### VOX-LANDING-A · marketing landing 3-tab + per-product Terms
- **Status:** 🟢 Done · agent-verified Playwright 5/5 visualmente 2026-05-16 22:55 UTC (Faustão delegou test "controla meu chrome e testa tudo q vc quiser")
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
  - [x] B.10 sacred files update Phase B (commit `852cb59` + PR #2 merged `259fa22`)
  - [x] Browser test Playwright 5/5 verde (Faustão delegou): home / + Coach tab + /coach/terms + /terms hub + /app v0.1 funcional
  - [ ] Faustão manda URL pro adv amigo (gatilho do produto)
- **Pricing tier:** CTA-only (`href="#"`) per SPRINT.md §0 #6 + resposta humana 2026-05-16 q2 — checkout real fica pra `VOX-COACH-D` (Lemon Squeezy/Stripe)
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
