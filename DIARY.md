# DIARY.md — execution log

> Append-only. Newest entry on top.

---

## 2026-05-09 — Phase A local code work · v0.1 deploy prep

**Tickets touched:** `VOX-DEPLOY-A` (sub-phases A0–A8)

**Done:**
- A0 bootstrap SPRINT/DIARY/JIRA/HUMAN at repo root (commit `8b24839`)
- A1 `LIVENESS_MODE = "off"` constant in `app/__init__.py`, Settings extended w/ `liveness_mode` + `hostname` + `metrics_key`, `.env.example` adds `VOX_LIVENESS_MODE` / `VOX_HOSTNAME` / `VOX_METRICS_KEY` (commit `08280f0`, bundles pre-existing `VOX_TLS_CERT/KEY_PATH` + `backend/secrets/` gitignore)
- A2 `/api/metrics` private endpoint (`app/api/metrics.py`), 5 tests (`test_metrics.py`), conftest reload list extended (commit `337dcb4`). Empty key → 404, wrong key → 404 (no leak)
- A3 frontend wired to real backend: `VOX_USE_MOCK = false` + deleted `window.VOX_MOCK` (~108 lines) (commit `7448297`). Existing `apiCalibrate`/`apiAnalyze` already correct against `/api/calibrate` + `/api/analyze` — DEPLOY §5.2's prescribed `static/wire.js` extraction skipped, logged in HUMAN Q-08
- A4 LGPD `landing_page/privacy.html` + `terms.html` (verbatim DEPLOY §9.1.1/§9.1.2), `/privacy` + `/terms` routes in `app/main.py`, footer Legal column + consent line above calibrate button referencing LGPD Art. 11 (commit `e49368d`). 2 tests added
- A5 nginx hardened (commit `5d6cde4`): Cloudflare real-IP block (15 prefixes) + `CF-Connecting-IP`, `limit_req_zone voxapi 10r/s` + `voxupload 2r/s`, dedicated upload location matching `/api/(calibrate|analyze)$`, explicit `:80 → 301`, Cloudflare Origin cert paths, X-Frame-Options DENY, Permissions-Policy microphone=(self), full CSP, HSTS 6mo no-subdomains. Upstream port 8002
- A6 systemd port 8000→8002 + `--no-server-header` + `--log-level info --access-log` + `MemoryMax=1G CPUQuota=200%` + `StartLimitBurst=3` + `ProtectKernelModules` / `RestrictAddressFamilies` / `RestrictNamespaces` / `LockPersonality` (commit `738e9b7`). `ReadWritePaths` now includes `/var/log/voxprobabilis`
- A7 daily SQLite backup script `backend/deploy/voxprobabilis-backup.sh` — sqlite3 `.backup` (WAL-aware), 14d retention, gzip after 1d (commit `6e88ce4`). Install via `/etc/cron.daily/`
- A8 README refreshed (commit `eef165a`): architecture matches Cloudflare model, first-time VPS setup updated to vox user + `/var/lib` + `/etc/ssl` + cert paste + `cron.daily` install, ROLLBACK section, LGPD section, env table extended

**In flight:**
- _none_ — Phase A complete.

**Blocked:**
- Phase C (VPS deploy) blocked on Phase B (Juan's Porkbun → Cloudflare → DNS → Origin Cert → VPS apt prep) and answers to HUMAN Q-01..Q-06.
- Smoke fixtures `audios_claude/{ai_truth,ai_uncertain,ai_lie}.wav` still missing — HUMAN Q-07.

**Files changed (net new + modified):**
- `SPRINT.md`, `DIARY.md`, `JIRA.md`, `HUMAN.md` (created)
- `backend/app/__init__.py`, `backend/app/config.py`, `backend/app/main.py`
- `backend/app/api/metrics.py` (new)
- `backend/.env.example`
- `backend/tests/conftest.py`, `backend/tests/test_api.py`, `backend/tests/test_metrics.py` (new)
- `backend/deploy/nginx.conf`, `backend/deploy/voxprobabilis.service`, `backend/deploy/voxprobabilis-backup.sh` (new)
- `backend/README.md`
- `landing_page/index.html`, `landing_page/privacy.html` (new), `landing_page/terms.html` (new)
- `.gitignore`

**Tests:** 43/43 green throughout (`pytest -q`).

**Next session should start with:** wait for Juan's answers to HUMAN.md Q-01..Q-07 + Phase B completion (DNS active, Cloudflare Origin Cert generated). Then run Phase C from `backend/README.md` "First-time VPS setup" + DEPLOY §10 smoke tests + DEPLOY §15 DoD walkthrough. Per global rule #13 task stays open until Juan writes "testei tudo, passou sem bugs".

---

<!-- Older entries follow below in reverse chronological order. -->
