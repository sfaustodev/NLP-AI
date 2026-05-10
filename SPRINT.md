# SPRINT.md — Vox Probabilis v0.1 production deploy

> Authoritative implementation policy. Read before any code change.
> Active sprint goal: ship `https://voxprobabilis.com` per `landing_page/DEPLOY.md`.

---

## 0. Hard rules (non-negotiable)

These cannot be overridden. Surface conflicts to the human before acting.

1. **Do not break Juan's existing Rust service.** Every nginx change is additive. Run `nginx -t` before every reload.
2. **No raw audio persisted.** Audio cleanup in `finally` blocks, verified post-deploy.
3. **No secrets in source.** `.env` only, mode `0600`, owned by `vox`.
4. **HTTPS only.** Cloudflare SSL mode = Full (strict). HSTS on.
5. **LGPD compliance pages must exist before public link.** `/privacy` + `/terms` before announcement.
6. **No payment integration in v0.1.** Pricing buttons remain CTA-only.
7. **No PT-BR translation in v0.1.**
8. **`SPRINT.md` cannot be edited by the agent.** Propose changes via DIARY/HUMAN; Juan applies.
9. **Atomic commits per logical change.** Format: `<type>(scope): <imperative>`. One commit per change.
10. **No `--no-verify` / hook bypass.** Investigate failures, do not skip.
11. **Server-side rate limit + nginx `limit_req`.** Defense in depth.
12. **`SPEC.md` and `DEPLOY.md` are authoritative.** If they conflict, `DEPLOY.md` wins (newer).
13. **Task closes only on Juan's written confirmation** ("testei tudo aqui, passou sem bugs"). PR merged ≠ done.

---

## 1. Sprint identity

- **Goal:** Deploy Vox Probabilis v0.1 to `https://voxprobabilis.com` end-to-end (frontend wired, LGPD pages live, nginx hardened, systemd active, smoke + DoD green).
- **Active dates:** 2026-05-09 → ship + 48h post-launch watch
- **Critical path:** Phase A code work → Juan's Cloudflare/VPS prep (Phase B) → Phase C VPS deploy → Phase D smoke + DoD
- **External blockers:** DNS propagation (5–30 min), Cloudflare Origin Cert generation (Juan)

---

## 2. Recommended order of execution

| Order | Phase | Deliverable | Notes |
|---|---|---|---|
| 1 | A0 | discipline files (SPRINT/DIARY/JIRA/HUMAN) | bootstrap |
| 2 | A1 | LIVENESS_MODE constant + .env additions | mode = "off" per §4 decision |
| 3 | A2 | `/api/metrics` endpoint + tests | DEPLOY §12.2 |
| 4 | A3 | `static/wire.js` + index.html mock swap | DEPLOY §5 |
| 5 | A4 | privacy.html + terms.html + footer/consent links | DEPLOY §9 |
| 6 | A5 | harden nginx.conf | DEPLOY §7 |
| 7 | A6 | fix systemd port + flags | DEPLOY §8 |
| 8 | A7 | daily SQLite backup script | DEPLOY §11 |
| 9 | A8 | README ROLLBACK + LGPD pointers | DEPLOY §13 |
| 10 | C | VPS deploy via §6.2 setup script | after Phase B done |
| 11 | D | smoke §10 + DoD §15 | gate before announce |

Phase A and Phase B (Juan's Porkbun/Cloudflare/VPS prep) run in parallel.

---

## 3. Stack-specific rules

### Backend (FastAPI)
- Python 3.11+. uvicorn 2 workers. Bind `127.0.0.1:8002`.
- All routers in `app/api/`. Register in `app/main.py`.
- Errors raise `VoxError`; central handler emits SPEC §6.5 envelope.
- Audio cleanup in `finally`. Tempfiles in `/tmp` only.
- SQLite WAL. DB at `/var/lib/voxprobabilis/vox.db`, mode `0600`.
- Pinned deps. No `>=`.

### Frontend (vanilla JS)
- No bundler. Plain `<script src="/static/wire.js">`.
- Same-origin fetch (`API_BASE = ''`). `credentials: 'same-origin'`.
- Do not change HTML structure or visible copy. Only JS bodies + the two new footer links + the consent line.
- Plot mapping per SPEC §12.1: `svg_x = 200 + naturalness * 180; svg_y = 200 - involuntary_stress * 180`.

### nginx
- Listen :443 ssl http2. :80 → 301 redirect.
- Cloudflare real-IP block (15 prefixes) + `real_ip_header CF-Connecting-IP`.
- `limit_req_zone voxapi 10r/s` + `voxupload 2r/s`. Burst 20 / 5 nodelay.
- `client_max_body_size 12M`. `proxy_request_buffering on` for upload locations.
- Server name explicitly `voxprobabilis.com www.voxprobabilis.com` (avoid default_server collision).

### systemd
- User `vox`, group `vox`. Hardening flags per DEPLOY §8.
- `ReadWritePaths=/var/lib/voxprobabilis /var/log/voxprobabilis /tmp`.
- `Restart=on-failure RestartSec=5s StartLimitBurst=3`.

### Cloudflare
- Free plan. Proxied (orange cloud) for both static and `/api/*`.
- Origin cert 15-yr. Saved in `/etc/ssl/voxprobabilis/{cert,key}.pem`. Key 0600.
- Web Analytics auto-injected.

---

## 4. Security checklist (every PR)

- [ ] Audio deleted in `finally`. `find /tmp -name '*.wav' -mtime -1` returns empty after smoke.
- [ ] Cookie: HttpOnly + Secure (prod) + SameSite=Lax.
- [ ] `VOX_SECRET_SALT` = 32 random bytes, not placeholder.
- [ ] `.env` mode 0600, owned `vox`.
- [ ] `vox.db` mode 0600, owned `vox`.
- [ ] Logs do not contain raw IPs, full session IDs, audio file names.
- [ ] CSP header set. `X-Frame-Options DENY`. `Permissions-Policy microphone=(self)`.
- [ ] CORS not enabled (same-origin).
- [ ] `--no-server-header` (uvicorn) + `server_tokens off` (nginx).
- [ ] Rate limit on `/api/calibrate` + `/api/analyze` at both app and nginx layers.
- [ ] `/api/metrics` requires `VOX_METRICS_KEY` (constant-time compare).

---

## 5. Conflict resolution

- `DEPLOY.md` ↔ `SPEC.md` → `DEPLOY.md` wins (newer).
- Ticket / chat ↔ `SPRINT.md` → `SPRINT.md` wins until Juan amends.
- "Quick fix" that bypasses §0 → refuse, surface conflict.

---

## 6. Out of scope (deferred to v0.2)

Browser MediaRecorder · payments · API keys for paid tier · batch CSV · PDF export · admin dashboard · PT-BR · Postgres · Docker · off-VPS backups · UptimeRobot · Liveness as exposed feature · IPv6 · staging subdomain · transactional email · stricter CSP (no `'unsafe-inline'`).

---

## 7. Pointers

- Master runbook: `landing_page/DEPLOY.md`
- Spec: `landing_page/SPEC.md`
- TTS finding: `TTS_DISCOVERY.md`
- Backend README: `backend/README.md`
- Diary: `DIARY.md`
- Open questions: `HUMAN.md`
- Ticket index: `JIRA.md`

---

## 8. When in doubt

Stop. Read DEPLOY.md / SPEC.md section. If silent, escalate to HUMAN.md. Do not improvise on:
- Cloudflare SSL mode
- Cookie flags
- Audio retention behavior
- LGPD page wording (Juan's exact text)
- Existing Rust service nginx routing

---

_Last updated: 2026-05-09 — bootstrap. Sprint expires when Juan confirms ship + 48h watch passes._
