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

Architecture on the VPS (decided 2026-04-18, see commit history):

- **Docker stack owns :80** for the existing `infra-*` containers — do **not** touch it.
- **nginx owns :443 only**, terminates TLS for `voxprobabilis.com`, and proxies everything to `127.0.0.1:8000` (uvicorn).
- **Cloudflare** handles the 80 → 443 redirect at the edge.
- The existing Rust `nda-backend` on :3000 is untouched — different hostname.

### First-time VPS setup

```bash
# System deps (one-time)
sudo apt update
sudo apt install -y python3.12 python3.12-venv ffmpeg libsndfile1 sqlite3

# Service user + dirs
sudo useradd -r -s /bin/false vox
sudo mkdir -p /opt/voxprobabilis /var/lib/voxprobabilis
sudo chown -R vox:vox /opt/voxprobabilis /var/lib/voxprobabilis

# Code + venv
sudo -u vox git clone https://github.com/<org>/NLP-AI.git /opt/voxprobabilis
sudo -u vox python3.12 -m venv /opt/voxprobabilis/venv
sudo -u vox /opt/voxprobabilis/venv/bin/pip install -r /opt/voxprobabilis/backend/requirements.txt

# Env
sudo -u vox cp /opt/voxprobabilis/backend/.env.example /opt/voxprobabilis/.env
sudo -u vox python3 -c "import secrets; print('VOX_SECRET_SALT=' + secrets.token_urlsafe(32))" \
    | sudo tee -a /opt/voxprobabilis/.env
# Edit /opt/voxprobabilis/.env: set VOX_COOKIE_SECURE=true, VOX_DB_PATH=/var/lib/voxprobabilis/vox.db

# systemd
sudo cp /opt/voxprobabilis/backend/deploy/voxprobabilis.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now voxprobabilis
sudo journalctl -u voxprobabilis -f   # watch it boot

# nginx (only after DNS is live)
sudo ln -s /opt/voxprobabilis/backend/deploy/nginx.conf /etc/nginx/sites-enabled/voxprobabilis
sudo nginx -t && sudo systemctl reload nginx
sudo systemctl enable nginx    # it was installed but inactive — enable now

# TLS (once DNS resolves)
sudo certbot --nginx -d voxprobabilis.com -d www.voxprobabilis.com
```

### Verifying the deploy

```bash
# On the VPS
curl -s http://127.0.0.1:8000/api/health | jq    # local, before nginx
curl -sk https://voxprobabilis.com/api/health | jq
```

Both should return `{"status":"ok","version":"0.1.0"}`. A `503` with `reason: ffmpeg not available` means the `apt install ffmpeg` step didn't stick — fix it before anything else.

### Updating after a code push

```bash
sudo -u vox git -C /opt/voxprobabilis pull
sudo -u vox /opt/voxprobabilis/venv/bin/pip install -r /opt/voxprobabilis/backend/requirements.txt
sudo systemctl restart voxprobabilis
```

Migrations are idempotent and apply on boot, so no manual SQL step.

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
