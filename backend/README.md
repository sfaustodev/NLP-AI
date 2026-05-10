# Vox Probabilis — Backend v0.1

Python/FastAPI service that powers the Vox Probabilis landing page. Extracts four spectral features from voice samples and projects them onto a two-axis Cartesian plane (naturalness × involuntary stress). See [`../landing_page/SPEC.md`](../landing_page/SPEC.md) for the full contract; this README covers **how to run and deploy it**.

## Quick start (local dev)

```bash
cd backend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Required env. Copy the template, then generate a salt.
cp .env.example .env
python3 -c "import secrets; print('VOX_SECRET_SALT=' + secrets.token_urlsafe(32))" >> .env
# Edit .env so VOX_SECRET_SALT has only the generated line (remove the placeholder).

# Boot
uvicorn app.main:app --reload --port 8000
```

Visit `http://localhost:8000/`. The landing page in `../landing_page/index.html` is served at `/`; API lives under `/api/*`.

The backend auto-applies `migrations/001_initial.sql` on every start, so the first run creates `./vox.db` on its own.

## Environment

All config is env-driven; see [`.env.example`](.env.example). The only required var is `VOX_SECRET_SALT` — the service refuses to boot without it, because that salt is what makes the stored IP hash non-reversible.

| Var | Default | Notes |
|---|---|---|
| `VOX_SECRET_SALT` | *(required)* | 32 random bytes, URL-safe base64. |
| `VOX_DB_PATH` | `./vox.db` | Anywhere writable. On the VPS: `/var/lib/voxprobabilis/vox.db`. |
| `VOX_STATIC_DIR` | `../landing_page` | Directory containing `index.html`. |
| `VOX_COOKIE_SECURE` | `false` | **Set to `true` in production.** Also gates XFF-trust. |
| `VOX_MAX_UPLOAD_MB` | `10` | Matches SPEC §11.1. |
| `VOX_FREE_DAILY_QUOTA` | `3` | Normal analyses per UTC day; ritual freebies don't count. |
| `VOX_CORS_ORIGINS` | *(empty)* | Comma-separated allowlist; empty disables CORS middleware. |
| `VOX_LOG_LEVEL` | `INFO` | `DEBUG` / `INFO` / `WARNING`. |
| `VOX_HOSTNAME` | `voxprobabilis.com` | Public hostname (used for self-references). |
| `VOX_LIVENESS_MODE` | `off` | DEPLOY.md §4. `off` / `boolean` / `full`. Unknown → `off`. |
| `VOX_METRICS_KEY` | *(empty)* | Empty disables `/api/metrics`. Generate with `secrets.token_urlsafe(32)`. |
| `VOX_TLS_CERT_PATH`, `VOX_TLS_KEY_PATH` | *(reference only)* | Document where the Cloudflare Origin cert + key live; the app does not load them. |

## Running the tests

```bash
pytest -q
```

The suite is hermetic: synthesised audio, throwaway sqlite under `tmp_path`, every app module reloaded per test. No network, no binary fixtures. All tests should complete in under 30 seconds.

Covered:

- **Unit**: audio load (accept/reject paths), feature extractors (invariants on sines and chirps), baseline (clamping, source provenance, SPEC key names), projection (every quadrant + tanh saturation + ORIGIN dead zone).
- **Integration**: session cookie lifecycle, calibrate → analyze happy path, rate limit exhaustion, ritual freebie accounting, error codes for bad inputs, `X-Vox-Version` header.

## Smoke test (SPEC §14.3)

The single test that matters most: does the extractor reproduce the paper's numbers?

```bash
# With venv active and server running:
curl -F "audio=@path/to/juans_truth_sample.wav" http://localhost:8000/api/calibrate | jq

# Compare the returned baseline.* values against Juan's paper.
# Tolerance: ±5%. If this diverges, the scientific claim is broken.
```

## Deployment

Authoritative runbook: [`../landing_page/DEPLOY.md`](../landing_page/DEPLOY.md). The summary below is for operators who already read it once.

Architecture (DEPLOY.md §2):

- **Cloudflare proxied** (orange cloud) for `voxprobabilis.com` + `www.voxprobabilis.com`.
- **Cloudflare Origin certificate** (15-yr validity) installed at `/etc/ssl/voxprobabilis/{cert,key}.pem`. SSL/TLS mode = **Full (strict)**.
- **nginx** owns :443 (and :80 for the 301 redirect), terminates TLS with the origin cert, proxies to `127.0.0.1:8002`.
- **uvicorn** binds `127.0.0.1:8002`, 2 workers, under systemd as `vox` user.
- The existing Rust service on this VPS stays untouched — different hostname / different vhost.

### Pre-flight (Juan's tasks before code lands on the box)

Walk-through in DEPLOY.md §3:

1. Porkbun → Cloudflare nameservers (5–30 min propagation).
2. Cloudflare DNS A records `@` and `www` → VPS IPv4, both proxied.
3. Cloudflare SSL/TLS = Full (strict); HSTS 6 mo, no subdomains; min TLS 1.2.
4. Cloudflare → SSL/TLS → Origin Server → Create Certificate (15 yr, PEM). Save cert + key — **shown once**.
5. Optional: Porkbun email forwarding for `contact@voxprobabilis.com`.

### First-time VPS setup (DEPLOY.md §6.2)

```bash
# System deps (one-time)
sudo apt update
sudo apt install -y python3.11 python3.11-venv ffmpeg libsndfile1 sqlite3 git nginx ufw fail2ban

# Service user + dirs
sudo useradd -r -s /bin/false vox || true
sudo mkdir -p /opt/voxprobabilis /var/lib/voxprobabilis /var/log/voxprobabilis /etc/ssl/voxprobabilis
sudo chown -R vox:vox /var/lib/voxprobabilis /var/log/voxprobabilis

# Code + venv
sudo -u vox git clone https://github.com/sfaustodev/NLP-AI.git /opt/voxprobabilis
sudo -u vox python3.11 -m venv /opt/voxprobabilis/venv
sudo -u vox /opt/voxprobabilis/venv/bin/pip install -U pip
sudo -u vox /opt/voxprobabilis/venv/bin/pip install -r /opt/voxprobabilis/backend/requirements.txt

# DB
sudo -u vox sqlite3 /var/lib/voxprobabilis/vox.db < /opt/voxprobabilis/backend/migrations/001_initial.sql
sudo -u vox chmod 600 /var/lib/voxprobabilis/vox.db

# Env (set VOX_COOKIE_SECURE=true, VOX_DB_PATH=/var/lib/voxprobabilis/vox.db,
#      VOX_SECRET_SALT=<generated>, VOX_LIVENESS_MODE=off, VOX_METRICS_KEY=<generated>)
sudo -u vox cp /opt/voxprobabilis/backend/.env.example /opt/voxprobabilis/.env
sudo -u vox chmod 600 /opt/voxprobabilis/.env
# (edit /opt/voxprobabilis/.env in your editor of choice)

# Cloudflare Origin TLS — paste content from §3.4 dialog
sudo nano /etc/ssl/voxprobabilis/cert.pem
sudo nano /etc/ssl/voxprobabilis/key.pem
sudo chmod 644 /etc/ssl/voxprobabilis/cert.pem
sudo chmod 600 /etc/ssl/voxprobabilis/key.pem
sudo chown root:root /etc/ssl/voxprobabilis/*

# systemd
sudo cp /opt/voxprobabilis/backend/deploy/voxprobabilis.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voxprobabilis
sudo journalctl -u voxprobabilis -f   # watch it boot

# nginx
sudo ln -s /opt/voxprobabilis/backend/deploy/nginx.conf /etc/nginx/sites-enabled/voxprobabilis
sudo nginx -t                                # MUST pass
sudo systemctl reload nginx

# Daily SQLite backup (DEPLOY §11)
sudo install -m 755 /opt/voxprobabilis/backend/deploy/voxprobabilis-backup.sh \
                    /etc/cron.daily/voxprobabilis-backup
sudo /etc/cron.daily/voxprobabilis-backup     # one dry run

# UFW
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### Verifying the deploy

```bash
# On the VPS
curl -s http://127.0.0.1:8002/api/health | jq
curl -sk https://voxprobabilis.com/api/health | jq
curl -s  https://voxprobabilis.com/privacy | grep -c '<title>'   # expect 1
curl -s  https://voxprobabilis.com/terms   | grep -c '<title>'   # expect 1
```

Health should return `{"status":"ok","version":"0.1.0"}`. A `503 reason: ffmpeg not available` means the `apt install ffmpeg` step didn't stick — fix it before anything else. A `503 reason: parselmouth missing` means the venv install dropped praat-parselmouth — `pip install --force-reinstall praat-parselmouth==0.4.5`.

Full smoke + DoD: DEPLOY.md §10 + §15.

## ROLLBACK

If the deploy breaks something (especially the existing Rust service), the path back is short and reversible. Memorize the first three lines.

```bash
# 1. Disable the Vox Probabilis nginx site
sudo rm /etc/nginx/sites-enabled/voxprobabilis
sudo nginx -t && sudo systemctl reload nginx

# 2. Stop and disable the service
sudo systemctl stop voxprobabilis
sudo systemctl disable voxprobabilis

# 3. (Optional) Remove the daily backup hook
sudo rm /etc/cron.daily/voxprobabilis-backup

# 4. (Optional, only if removing fully) wipe code + DB
#    sudo rm -rf /opt/voxprobabilis /var/lib/voxprobabilis \
#                /etc/systemd/system/voxprobabilis.service
#    sudo systemctl daemon-reload
```

DNS at Cloudflare: pause the site or change A record back to its previous value. Note that this won't unbreak the Rust service if the breakage was nginx — fix nginx first.

### Updating after a code push

```bash
sudo -u vox git -C /opt/voxprobabilis pull
sudo -u vox /opt/voxprobabilis/venv/bin/pip install -r /opt/voxprobabilis/backend/requirements.txt
sudo systemctl restart voxprobabilis
```

Migrations are idempotent and apply on boot, so no manual SQL step.

## LGPD compliance

Brazilian Lei Geral de Proteção de Dados Art. 11 classifies voice as sensitive personal data. Three pieces are non-negotiable before public link:

- `/privacy` — verbatim policy from DEPLOY.md §9.1.1, served by FastAPI from `landing_page/privacy.html`.
- `/terms` — verbatim ToS from DEPLOY.md §9.1.2, from `landing_page/terms.html`.
- `contact@voxprobabilis.com` — DPO inbox, set up via Porkbun email forwarding (DEPLOY §3.1).

The landing page links both pages in the footer "Legal" column and shows a consent line above the calibrate button referencing Art. 11 explicit consent. **Do not deploy without these three pieces.**

## Retention (SPEC §10.3)

A weekly cron should prune stale data. Install once:

```bash
# Crontab for the vox user:
#   Weekly at 03:00 UTC on Sundays.
0 3 * * 0 sqlite3 /var/lib/voxprobabilis/vox.db <<'SQL'
    DELETE FROM sessions WHERE last_seen_at < strftime('%s', 'now', '-60 days');
    DELETE FROM analyses WHERE created_at    < strftime('%s', 'now', '-90 days');
SQL
```

`dataset_optins` is kept indefinitely — it is anonymous research data by design (SPEC §10.2).

## Observability

Logs go to stderr in a JSON-ish line format; systemd redirects them to journald. Useful filters:

```bash
# Live
sudo journalctl -u voxprobabilis -f

# Last hour's errors
sudo journalctl -u voxprobabilis --since "1 hour ago" -p err
```

Never log raw audio, full session IDs (hash them first), or raw IPs (use the stored IP hash).

## What's NOT in v0.1

Out of scope until v0.2: user accounts, payment processing, API keys, batch upload, PDF export, admin dashboard, real-time `MediaRecorder` capture in the frontend, PT-BR translation. The pricing page is CTA-only.

## Lineage

- Companion paper: <https://doi.org/10.5281/zenodo.19396809>
- Companion book: <https://doi.org/10.5281/zenodo.19478167>
- License: MIT (see repo root).
