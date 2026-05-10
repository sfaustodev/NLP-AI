# DIARY.md — execution log

> Append-only. Newest entry on top.

---

## 2026-05-10 — Phase B/C/D · live deploy

**Tickets touched:** `VOX-DEPLOY-A`

**Done:**
- Phase B (Juan): Cloudflare DNS + Origin Cert + SSL Full(strict) + HSTS + min TLS 1.2 — confirmed live.
- VPS probe (read-only, ssh root@89.116.73.118): Ubuntu 24.04, Python 3.12, 3.8GB RAM, 1 vCPU. Existing partial deploy from Apr 19/23 (uvicorn on :8000 since Apr 23, vox user uid 999, /opt/voxprobabilis at `40f0eb9`). Resolved Q-01..Q-05 in HUMAN.md.
- Local fixes (commits `e9c8ea8`, `d35c9c0`, `3d1c45d`, `8341769`):
  - `nginx.conf`: drop `:80 → 301` block (Docker owns :80 on this VPS for appnda.com); comment records the reason.
  - `README.md`: bump `python3.11` → `python3.12` (Ubuntu 24.04 default).
  - `voxprobabilis.service`: move `StartLimitIntervalSec`/`Burst` from `[Service]` to `[Unit]` (systemd ignored them silently); add `NUMBA_CACHE_DIR=/tmp/numba` to fix librosa.effects lazy-loaded JIT cache crash under `ProtectSystem=strict`.
  - `app/main.py`: `@app.api_route(methods=["GET","HEAD"])` on `/`, `/privacy`, `/terms` so `curl -I` (and uptime monitors) get 200, not 405.
- Phase C steps:
  - `systemctl stop voxprobabilis` (release :8000), `git pull` to `8341769`, `pip install -r requirements.txt` (refresh).
  - Patch `/opt/voxprobabilis/.env`: `VOX_COOKIE_SECURE=true`, `VOX_DB_PATH=/var/lib/voxprobabilis/vox.db`, `VOX_HOSTNAME=voxprobabilis.com`, `VOX_LIVENESS_MODE=off`, `VOX_METRICS_KEY=<token_urlsafe(32)>`. `chmod 600`.
  - `chmod 600 /var/lib/voxprobabilis/vox.db`.
  - `mkdir -p /var/log/voxprobabilis /etc/ssl/voxprobabilis`.
  - `scp` Cloudflare Origin cert + key from `backend/secrets/` → `/etc/ssl/voxprobabilis/` (cert 0644, key 0600, root:root).
  - `cp voxprobabilis.service /etc/systemd/system/`, `daemon-reload`, `start`. `:8002` LISTEN, health 200 ok.
  - `install -m 755 voxprobabilis-backup.sh /etc/cron.daily/`. First run produced `vox-2026-05-10.db` in `/var/backups/voxprobabilis/`.
  - UFW allow 22/80/443, `ufw enable`.
  - `rm /etc/nginx/sites-enabled/appnda` (Q-09 option b; Juan's appnda.com keeps serving via Docker on :80).
  - `cp nginx.conf /etc/nginx/sites-available/voxprobabilis`, symlink to `sites-enabled/`. `nginx -t` ok. `systemctl enable --now nginx`.
- Phase D (smoke + DoD):
  - DEPLOY §10 #1–#8 all green from laptop.
  - **Mid-smoke discovery (Q-10):** Cloudflare DNS had 3 A records for `@` — our VPS plus two AWS Oregon IPs (Linkly leftovers from a previous Porkbun trial). CF round-robin caused multipart POST to randomly land on Linkly, which redirected to `voxprobabilis-com.l.ink`. Juan deleted the orphan A records + wildcard CNAME `* → uixie.porkbun.com`. Smoke #6/#7/#8 then passed.
  - Lie sample landed in **`OVER_CONTROLLED_TENSE`** with confidence high, jitter -39%, mfcc_delta_var -65%, microtremor +32% — the deception signature.
  - Rate limit: 1–3 = 200, 4 = 429 RATE_LIMITED.

**In flight:**
- Juan's browser smoke §10.1 (incognito, three points correct quadrants, cookie HttpOnly+Secure) — pending Juan.
- DoD §15 final walkthrough — pending Juan.

**Blocked:**
- Per global rule #13, task **open** until Juan writes "testei tudo aqui, passou sem bugs".

**Files changed (this session):**
- `backend/deploy/nginx.conf`, `backend/deploy/voxprobabilis.service`, `backend/README.md`, `backend/app/main.py`
- VPS-side: `/opt/voxprobabilis/.env`, `/etc/systemd/system/voxprobabilis.service`, `/etc/nginx/sites-{available,enabled}/voxprobabilis`, `/etc/ssl/voxprobabilis/{cert,key}.pem`, `/etc/cron.daily/voxprobabilis-backup`, removed `/etc/nginx/sites-enabled/appnda`.
- Cloudflare DNS: deleted A `@ → 44.230.85.241`, A `@ → 52.33.207.7`, CNAME `* → uixie.porkbun.com`.

**VOX_METRICS_KEY (private, do not commit):** `FMeAuNOBEJrxkdBtsHepgkE25vnc-TfyThkpr-Zq8Zg`
URL: https://voxprobabilis.com/api/metrics?key=FMeAuNOBEJrxkdBtsHepgkE25vnc-TfyThkpr-Zq8Zg

**Tests:** 43/43 local pytest green. Production-side: §10 1–8 smoke green.

**Next session should start with:** Juan's confirmation of browser smoke + DoD review. Then Phase E close-out + `transitionJiraIssue` if Jira project exists. Watch logs for first 6 h before broader announce (DEPLOY §16).

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
