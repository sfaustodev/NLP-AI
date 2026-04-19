# Vox Probabilis — Backend v0.1 · SPEC

```
Authors          Juan Fausto, Claude (Anthropic)
Version          0.1.0-draft · 2026-04-18
Companion paper  https://doi.org/10.5281/zenodo.19396809
Companion book   https://doi.org/10.5281/zenodo.19478167
Frontend         ./vox_probabilis.html (existing, do not modify)
License          MIT
```

---

## 0 · For Claude Code · Read This First

You are about to implement a small Python backend that runs audio analysis and serves an existing static HTML page. The HTML is **already designed and approved by the user** — do not modify its layout, copy, or styles. Your job is to make the buttons actually work.

**Before writing any code, ask Juan these four questions:**

1. VPS OS and Python version available (we assume Debian/Ubuntu + Python 3.11+, but confirm).
2. Which port is free for this service (we default to `8000`, but the Rust service may conflict).
3. Domain name chosen for the product (we default to `voxprobabilis.com` in config examples).
4. Whether ffmpeg is already installed on the VPS (`ffmpeg -version` check).

If any of these are unknown, stop and ask — do not guess your way into broken deploy.

**Boundaries:**

- This is v0.1. If a feature is not listed as "in scope", it is out of scope.
- Billing (Lemon Squeezy / Stripe) is **not** in this spec. Paid tiers show CTA only, no payment flow yet.
- No Docker. Plain systemd.
- No Postgres. SQLite.
- No auth framework. Session cookies for free tier; API keys deferred to v0.2.
- No Celery / Redis. Analysis runs synchronously in a threadpool (FFT on 10s of audio is ~100ms, no queue needed).

**Do not touch Juan's existing Rust service on the same VPS.** Nginx will route around it. Assume nothing about it.

---

## 1 · Mission

Serve a web API and a static HTML page that let any user upload a 3–60 second voice sample and receive back:

1. Four spectral features (jitter, MFCC delta variance, spectral flux, microtremor)
2. A 2D Cartesian projection (Naturalness × Involuntary Stress) and quadrant label
3. An honest confidence indicator reflecting whether per-user baseline calibration was performed

The backend must be **light** (share a VPS with an unrelated heavy Rust service), **honest** (never produce confident output from uncalibrated input), and **privacy-preserving** (raw audio is never written to disk).

---

## 2 · Scope

### 2.1 · In scope for v0.1

- Static file serving for the existing landing page and its assets
- `POST /api/calibrate` — accept truth sample, store per-session baseline
- `POST /api/analyze` — accept any sample, return features + Cartesian projection + quadrant
- `GET /api/session` — inspect current session state (has baseline? how many free uses today?)
- `GET /api/health` — liveness for nginx/uptime monitoring
- SQLite persistence for session state, rate limits, and opt-in feature vectors (not audio)
- Session cookie (HTTPOnly, SameSite=Lax, 30-day rolling)
- Per-IP-and-cookie rate limit: 3 analyses / 24h, ritual 3 samples do not count against quota
- systemd service unit and nginx config snippet
- Minimal pytest suite covering feature extraction correctness and API contract

### 2.2 · Out of scope for v0.1 (do not implement)

- User accounts / password auth / OAuth
- Payment processing (Lemon Squeezy, Stripe) — pricing page is display-only
- API keys for paid tiers
- Batch CSV upload
- PDF export of results
- Admin dashboard
- Email notifications
- Real-time microphone recording (frontend uses uploads; browser MediaRecorder integration is v0.2)
- PT-BR translation of the landing page
- Live retraining of baselines across users
- Any ML model beyond the four hand-crafted features

---

## 3 · Architecture

```
                        Cloudflare (free tier, DNS + CDN + TLS)
                                       │
                                       ▼
                           nginx on VPS (port 443)
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 │                     │                     │
                 ▼                     ▼                     ▼
        /api-rust/*              /api/*                  / (static)
        existing Rust          FastAPI (uvicorn)        FastAPI StaticFiles
        service                port 8000                 same process
        (untouched)            2 workers
                                       │
                                       ▼
                               SQLite (file on disk)
                               /var/lib/voxprobabilis/vox.db
```

- Single Python process hosts both the API and the static page. No CORS needed.
- SQLite file is append-mostly; WAL mode enabled for concurrent reads during writes.
- All heavy work (FFT, librosa, parselmouth) runs inside `fastapi.concurrency.run_in_threadpool` so the asyncio event loop never blocks.

---

## 4 · Tech Stack

| Dependency | Version | Why |
|---|---|---|
| Python | 3.11+ | Speed, good asyncio, `tomllib` stdlib |
| FastAPI | ^0.110 | Async, auto-docs, tiny footprint |
| uvicorn | ^0.27 | Standard ASGI server, systemd-friendly |
| parselmouth | ^0.4.3 | Praat bindings; jitter is Praat's specialty |
| librosa | ^0.10 | MFCC, spectral flux; well-tested |
| numpy | ^1.26 | Underpins everything |
| scipy | ^1.12 | Butterworth bandpass, Hilbert transform |
| soundfile | ^0.12 | WAV/FLAC read |
| pydub | ^0.25 | Format conversion via ffmpeg |
| pytest | ^8.0 | Tests |
| httpx | ^0.27 | Test client for API |

System dependencies (install via apt):

- `ffmpeg` — required by pydub for MP3/M4A/OGG decoding
- `libsndfile1` — required by soundfile

Lock versions in `requirements.txt`. Do **not** use poetry or pipenv — plain pip + venv keeps it light.

---

## 5 · Repository Layout

```
voxprobabilis/
├── SPEC.md                    ← this file
├── README.md                  ← ops guide for Juan (created last)
├── requirements.txt
├── .env.example
├── .gitignore
├── app/
│   ├── __init__.py
│   ├── main.py                ← FastAPI app, route registration, startup
│   ├── config.py              ← env var loading
│   ├── db.py                  ← SQLite connection + migrations
│   ├── sessions.py            ← session cookie management
│   ├── rate_limit.py          ← 3/day logic
│   ├── audio/
│   │   ├── __init__.py
│   │   ├── load.py            ← decode any format → mono 16kHz numpy array
│   │   ├── features.py        ← the four extractors
│   │   ├── baseline.py        ← per-session + global fallback
│   │   └── projection.py      ← Cartesian + quadrant
│   └── api/
│       ├── __init__.py
│       ├── calibrate.py
│       ├── analyze.py
│       ├── session.py
│       └── health.py
├── static/
│   └── vox_probabilis.html    ← the existing frontend (unchanged)
├── migrations/
│   └── 001_initial.sql
├── tests/
│   ├── fixtures/
│   │   ├── truth_sample.wav       ← Juan provides; 5s of plain speech
│   │   ├── lie_sample.wav         ← Juan provides; 5s of deliberate lie
│   │   └── silence.wav            ← 5s of digital silence for edge case
│   ├── test_audio_load.py
│   ├── test_features.py
│   ├── test_baseline.py
│   ├── test_projection.py
│   └── test_api.py
└── deploy/
    ├── voxprobabilis.service   ← systemd unit
    └── nginx.conf.snippet       ← to be dropped into existing nginx config
```

---

## 6 · API Contract

All request/response bodies are JSON except where audio is uploaded (multipart/form-data). All responses include `Content-Type: application/json` and an `X-Vox-Version: 0.1.0` header.

### 6.1 · `POST /api/calibrate`

Accept a truth sample and compute the user's personal baseline. This is the first step of the 3-sample ritual. Does not count against rate limit.

**Request:** `multipart/form-data`

| Field | Type | Constraints |
|---|---|---|
| `audio` | file | WAV/MP3/M4A/OGG/FLAC · ≤ 10 MB · 3–60 seconds |
| `label` | string | optional; defaults to "truth" |

**Response 200:**

```json
{
  "session_id": "a1b2c3d4...",
  "baseline_established": true,
  "baseline": {
    "jitter_local": 0.01823,
    "mfcc_delta_var_mean": 0.04715,
    "spectral_flux_mean": 0.12841,
    "microtremor_envelope": 0.00342
  },
  "sample_duration_s": 4.87,
  "voiced_frame_ratio": 0.82
}
```

**Response 4xx:** see §11.

### 6.2 · `POST /api/analyze`

Accept any voice sample and return features + projection. Counts against the 3/day quota **unless** the request includes `ritual_step=uncertain` or `ritual_step=lie` AND a session with an established baseline but no prior uncertain/lie submissions in the current calendar day.

**Request:** `multipart/form-data`

| Field | Type | Constraints |
|---|---|---|
| `audio` | file | WAV/MP3/M4A/OGG/FLAC · ≤ 10 MB · 3–60 seconds |
| `ritual_step` | string | optional; one of `uncertain`, `lie`, `ai_bonus`, or absent |
| `opt_in_dataset` | bool | optional; if `true`, store the 4 features (not audio) for v2 |

**Response 200:**

```json
{
  "features": {
    "jitter_local": 0.01330,
    "mfcc_delta_var_mean": 0.02829,
    "spectral_flux_mean": 0.06164,
    "microtremor_envelope": 0.00400
  },
  "deltas": {
    "jitter_local_pct": -27.05,
    "mfcc_delta_var_pct": -40.00,
    "spectral_flux_pct": -52.00,
    "microtremor_envelope_pct": 16.95
  },
  "projection": {
    "naturalness": -0.712,
    "involuntary_stress": 0.418,
    "quadrant": "OVER_CONTROLLED_TENSE"
  },
  "confidence": "medium",
  "confidence_reason": "Per-session baseline from calibration used.",
  "baseline_source": "session",
  "sample_duration_s": 4.92,
  "voiced_frame_ratio": 0.78,
  "quota": { "remaining_today": 2, "resets_at": "2026-04-19T00:00:00Z" }
}
```

**Quadrant enum values:**

- `OVER_CONTROLLED_TENSE` — naturalness<0, stress>0  (the deception signature)
- `NATURAL_STRESSED` — naturalness>0, stress>0
- `OVER_CONTROLLED_CALM` — naturalness<0, stress<0
- `NATURAL_CALM` — naturalness>0, stress<0
- `ORIGIN` — both axes within ±0.05 (near zero; report as ambiguous)

**Confidence enum values:**

- `high` — per-session baseline + ≥0.75 voiced frame ratio + duration ≥5s
- `medium` — per-session baseline + either lower voiced ratio or shorter sample
- `low` — global fallback baseline used OR voiced frame ratio < 0.4
- `unreliable` — voiced frame ratio < 0.2 OR duration < 3s (response still returned but UI should warn heavily)

### 6.3 · `GET /api/session`

**Response 200:**

```json
{
  "session_id": "a1b2c3d4...",
  "has_baseline": true,
  "ritual_complete": false,
  "ritual_steps_done": ["truth"],
  "quota": { "remaining_today": 3, "resets_at": "2026-04-19T00:00:00Z" },
  "created_at": "2026-04-18T14:03:22Z"
}
```

### 6.4 · `GET /api/health`

Returns `200 {"status": "ok", "version": "0.1.0"}` when ffmpeg is callable and DB is writable. Returns `503` with a reason otherwise.

### 6.5 · Error response shape

All 4xx/5xx responses use this shape:

```json
{
  "error": {
    "code": "AUDIO_TOO_SHORT",
    "message": "Audio must be at least 3 seconds long. Received 1.2s.",
    "hint": "Try recording for 5 seconds or longer."
  }
}
```

Error codes: see §11.

---

## 7 · Feature Extraction Spec

All extractors work on a mono 16 kHz float32 numpy array of shape `(n_samples,)`. Audio loading (§7.1) converts any input to this format.

### 7.1 · Audio normalization pipeline

```
upload bytes → mime/magic-byte check → save to /tmp with UUID name →
pydub.AudioSegment.from_file → set to mono, resample to 16000 Hz →
export to bytes → read with soundfile → numpy float32 normalized to [-1, 1] →
delete /tmp file → return array
```

- Maximum 60 seconds of audio; truncate silently if longer (but log a warning).
- Minimum 3 seconds; reject with `AUDIO_TOO_SHORT` otherwise.
- Detect voiced frames using `librosa.effects.split(y, top_db=30)` and compute `voiced_frame_ratio = voiced_samples / total_samples`. Reject with `NO_VOICE_DETECTED` if ratio < 0.1.

### 7.2 · Jitter (local)

**Library:** `parselmouth` (Praat wrapper)

**Procedure:**
```python
import parselmouth
snd = parselmouth.Sound(values=y, sampling_frequency=16000)
point_process = parselmouth.praat.call(snd, "To PointProcess (periodic, cc)", 75, 500)
jitter_local = parselmouth.praat.call(
    point_process, "Get jitter (local)",
    0, 0, 0.0001, 0.02, 1.3
)
```

- Praat parameters are the defaults; document this in code comments.
- If `point_process` yields < 10 periods, return `None` and set confidence to `unreliable`.
- Output is a dimensionless fraction (typical voiced speech: 0.005–0.03).

### 7.3 · MFCC Delta Variance

**Library:** `librosa`

**Procedure:**
```python
mfcc = librosa.feature.mfcc(
    y=y, sr=16000,
    n_mfcc=13, n_fft=400, hop_length=160, win_length=400
)  # 25ms window, 10ms hop at 16kHz
delta = librosa.feature.delta(mfcc, order=1)
# Compute variance per coefficient, then mean across the 13 coefficients
mfcc_delta_var_mean = float(np.mean(np.var(delta, axis=1)))
```

- We use mean-of-variances (not variance-of-means) because the paper's proxy for "voluntary variation" is how much each MFCC coefficient fluctuates frame-to-frame.
- Typical range for relaxed speech: 0.03–0.12.

### 7.4 · Spectral Flux

**Library:** `librosa`

**Procedure:**
```python
S = np.abs(librosa.stft(y, n_fft=400, hop_length=160))
# Normalize each frame's magnitude spectrum
S_norm = S / (np.sum(S, axis=0, keepdims=True) + 1e-9)
# Frame-to-frame L2 difference
flux = np.sqrt(np.sum(np.diff(S_norm, axis=1) ** 2, axis=0))
spectral_flux_mean = float(np.mean(flux))
```

- This is the standard flux definition (Dixon 2006). Document reference in code.
- Typical range: 0.08–0.20.

### 7.5 · Microtremor Envelope (8–12 Hz band)

**Library:** `scipy.signal`

**Procedure:**
```python
from scipy.signal import butter, filtfilt, hilbert

# Step 1: extract amplitude envelope of the voice
analytic = hilbert(y)
envelope = np.abs(analytic)

# Step 2: downsample envelope to 1000 Hz (8-12 Hz fits easily)
from scipy.signal import resample_poly
env_1k = resample_poly(envelope, up=1, down=16)  # 16k → 1k

# Step 3: bandpass 8-12 Hz on the envelope (this is where microtremor lives)
b, a = butter(N=4, Wn=[8/500, 12/500], btype='bandpass')
tremor_band = filtfilt(b, a, env_1k)

# Step 4: RMS amplitude in that band
microtremor_envelope = float(np.sqrt(np.mean(tremor_band ** 2)))
```

- This follows the classical Lippold microtremor observation adapted for voice: physiological tremor modulates the amplitude envelope in the 8–12 Hz band.
- Typical range on relaxed speech: 0.002–0.005. Elevated values (the "involuntary stress" signal) go above 0.005.
- Note in code comments: **this exact formula is the v0.1 definition; it may be refined in v0.2 based on real usage data**.

---

## 8 · Baseline & Cartesian Projection

### 8.1 · Per-session baseline (primary path)

- When `POST /api/calibrate` succeeds, the four features of the truth sample are stored in the `sessions` table under the user's session ID.
- Every subsequent `/api/analyze` call for that session computes deltas against those stored values.

### 8.2 · Global fallback baseline (secondary path)

If `/api/analyze` is called without a per-session baseline (user skipped ritual), use these hardcoded values derived from Juan's paper sample-of-3 mean of the "truth" condition:

```python
GLOBAL_BASELINE = {
    "jitter_local": 0.0182,
    "mfcc_delta_var_mean": 0.0471,
    "spectral_flux_mean": 0.1284,
    "microtremor_envelope": 0.0034,
}
```

- Responses using this baseline MUST set `baseline_source = "global"` and `confidence = "low"` at best.
- Document in `app/audio/baseline.py` that these numbers come from n=3 and are placeholders until real population data is collected.

### 8.3 · Delta computation

For each feature `f`:

```
delta_pct = ((sample[f] - baseline[f]) / baseline[f]) * 100
```

Report both raw features and deltas. Clamp delta_pct to `[-100, +500]` to prevent outliers from breaking the UI.

### 8.4 · Naturalness axis

```
voluntary_signals = [
    delta_pct("jitter_local"),
    delta_pct("mfcc_delta_var_mean"),
    delta_pct("spectral_flux_mean"),
]
# Mean of voluntary-variation deltas, converted to a [-1, +1] signal via tanh
naturalness_raw = np.mean(voluntary_signals) / 100.0
naturalness = float(np.tanh(naturalness_raw * 2))  # 2 = sharpness factor
```

Interpretation:

- `naturalness > 0` → voluntary variation elevated vs baseline → more natural speech
- `naturalness < 0` → voluntary variation suppressed → over-controlled

The `* 2` sharpness factor is a design choice: it maps ±50% delta to roughly ±0.76 on the plot, which feels right visually. Document this.

### 8.5 · Involuntary Stress axis

```
microtremor_signal = delta_pct("microtremor_envelope") / 100.0
involuntary_stress = float(np.tanh(microtremor_signal * 2))
```

Interpretation:

- `involuntary_stress > 0` → elevated microtremor → involuntary arousal present
- `involuntary_stress < 0` → reduced microtremor → calm

### 8.6 · Quadrant assignment

```python
def quadrant(nat, stress):
    if abs(nat) < 0.05 and abs(stress) < 0.05:
        return "ORIGIN"
    over_ctrl = nat < 0
    tense = stress > 0
    if over_ctrl and tense:     return "OVER_CONTROLLED_TENSE"
    if not over_ctrl and tense: return "NATURAL_STRESSED"
    if over_ctrl and not tense: return "OVER_CONTROLLED_CALM"
    return "NATURAL_CALM"
```

---

## 9 · Sessions & Rate Limiting

### 9.1 · Session cookie

- Cookie name: `vox_session`
- Value: URL-safe base64 of a 32-byte random ID
- Flags: `HttpOnly; SameSite=Lax; Secure` (in production); `Path=/`
- Expiry: 30 days rolling (refresh on any successful API call)
- Set on first visit via a middleware; do not require the frontend to do anything.

### 9.2 · SQLite schema (`migrations/001_initial.sql`)

```sql
CREATE TABLE IF NOT EXISTS sessions (
    session_id TEXT PRIMARY KEY,
    created_at INTEGER NOT NULL,         -- unix ts
    last_seen_at INTEGER NOT NULL,
    ip_hash TEXT NOT NULL,                -- sha256(ip + SALT), first 16 chars
    baseline_jitter REAL,
    baseline_mfcc_delta_var REAL,
    baseline_spectral_flux REAL,
    baseline_microtremor REAL,
    baseline_established_at INTEGER,
    ritual_uncertain_done_at INTEGER,    -- unix ts or null
    ritual_lie_done_at INTEGER            -- unix ts or null
);

CREATE TABLE IF NOT EXISTS analyses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    day_bucket TEXT NOT NULL,             -- 'YYYY-MM-DD' in UTC
    ritual_step TEXT,                     -- 'uncertain' | 'lie' | 'ai_bonus' | null
    counted_against_quota INTEGER NOT NULL, -- 0 or 1
    quadrant TEXT NOT NULL
);
CREATE INDEX idx_analyses_session_day ON analyses(session_id, day_bucket);

CREATE TABLE IF NOT EXISTS dataset_optins (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at INTEGER NOT NULL,
    jitter_local REAL NOT NULL,
    mfcc_delta_var_mean REAL NOT NULL,
    spectral_flux_mean REAL NOT NULL,
    microtremor_envelope REAL NOT NULL,
    ritual_step TEXT,
    quadrant TEXT NOT NULL
    -- Note: no session_id, no ip, no anything identifying. Pure features.
);
```

Enable WAL mode on startup: `PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;`

### 9.3 · Rate limit logic

For an `/api/analyze` call with a given session:

1. Compute `day_bucket = utcnow().strftime("%Y-%m-%d")`
2. If `ritual_step == "uncertain"` AND session has a baseline AND no row in `analyses` with this session+day+`ritual_step="uncertain"` exists → free, set `counted_against_quota = 0`
3. Same rule for `ritual_step == "lie"` (one freebie per day)
4. `ritual_step == "ai_bonus"` → counts against quota like normal analysis (it's a bonus but not free)
5. Otherwise count against quota: if `SELECT COUNT(*) FROM analyses WHERE session_id=? AND day_bucket=? AND counted_against_quota=1` ≥ 3 → return `429 RATE_LIMITED`

The `/api/calibrate` call never counts against quota.

---

## 10 · Privacy & Retention

### 10.1 · Audio handling

- **Raw audio is never written to disk for more than 100ms.** It hits `/tmp` during format conversion then gets deleted in a `finally` block.
- Never log raw audio bytes, not even in debug mode.
- Never log audio file names from user uploads (they can be exfil vectors).

### 10.2 · What IS stored

- Session ID (random, not tied to identity)
- IP hash (sha256 with a server-side salt, first 16 chars — enough for rate-limit dedup, not enough to reverse)
- Timestamps
- Feature values (4 floats per analysis)
- Quadrant string
- If `opt_in_dataset=true`: the 4 feature values go to `dataset_optins` table with NO session/IP link

### 10.3 · Retention

- `sessions` rows idle for > 60 days are deleted by a weekly cron (write a simple SQL cleanup; document in README).
- `analyses` rows older than 90 days are deleted (same cron).
- `dataset_optins` rows are kept indefinitely (they are anonymous research data).

### 10.4 · Honest disclaimer in responses

Every analyze response includes:

```json
"methodology_note": "Features are computed per the methods described in DOI 10.5281/zenodo.19396809. The deception signature was validated on n=3 recordings from a single speaker. Generalization to your voice is hypothesized but not proven."
```

The frontend already displays this note visibly.

---

## 11 · Validation & Error Handling

### 11.1 · Error codes

| Code | HTTP | Meaning |
|---|---|---|
| `AUDIO_MISSING` | 400 | No `audio` field in request |
| `AUDIO_TOO_LARGE` | 413 | File > 10 MB |
| `AUDIO_TOO_SHORT` | 400 | Decoded audio < 3 seconds |
| `AUDIO_TOO_LONG` | 400 | Decoded audio > 60 seconds (soft: truncate + warn; see §7.1) |
| `AUDIO_UNSUPPORTED_FORMAT` | 400 | MIME + magic byte mismatch, or ffmpeg refused |
| `AUDIO_CORRUPT` | 400 | Decoding raised an exception |
| `NO_VOICE_DETECTED` | 400 | Voiced frame ratio < 0.1 |
| `RATE_LIMITED` | 429 | Free quota exceeded; include `retry_after` in response |
| `BASELINE_REQUIRED` | 400 | `ritual_step=uncertain` or `lie` with no prior calibration |
| `RITUAL_ALREADY_USED` | 400 | e.g. `ritual_step=lie` when today's `lie` freebie is already spent |
| `INTERNAL_ERROR` | 500 | Unexpected; log full traceback server-side, return generic message |

### 11.2 · Validation order

Always validate in this sequence (fail fast, cheap checks first):

1. Session cookie valid / assignable
2. `audio` field present
3. Content-Length ≤ 10 MB
4. MIME + magic bytes match an allowed format
5. Rate limit check (before expensive decoding)
6. Decode + resample
7. Duration check
8. Voiced frame ratio check
9. Run feature extraction

Steps 1–5 should complete in < 10ms. Steps 6–9 take ~100–300ms.

---

## 12 · Frontend Integration

The existing `vox_probabilis.html` is in `static/` and is served at `/`. It currently has mock data and simulated flows. The backend integration requires minimal JS changes — Claude Code should ship a small patch file describing the exact `fetch()` calls to add but **must not** restyle or reorganize the HTML.

### 12.1 · JS patch plan

Add to the existing `<script>` block:

```javascript
// Replace the mock runStep(step) with real API calls.
// Replace the mock runAnalysis() with a real fetch to /api/analyze.
```

Specifically:

- `runStep(1)` → `POST /api/calibrate` with the uploaded blob, parse response, show result
- `runStep(2)` → `POST /api/analyze` with `ritual_step=uncertain`
- `runStep(3)` → `POST /api/analyze` with `ritual_step=lie`
- `runStep(5)` → `POST /api/analyze` with `ritual_step=ai_bonus`
- Main analyzer's `runAnalysis()` → `POST /api/analyze` with no ritual_step

**Audio source for v0.1:** file upload only. Browser `MediaRecorder` integration is explicitly deferred to v0.2. The existing "Record 5 seconds" buttons should open the file picker for now — a small UX compromise Juan has agreed to.

The response `projection.naturalness` and `projection.involuntary_stress` values (both in [-1, +1]) should be mapped to SVG coordinates via:

```
svg_x = 200 + naturalness * 180
svg_y = 200 - involuntary_stress * 180
```

This keeps points inside the 400x400 plot with a small margin.

### 12.2 · CORS

Not needed. The frontend is served from the same origin as the API.

---

## 13 · Deployment

### 13.1 · systemd unit (`deploy/voxprobabilis.service`)

```ini
[Unit]
Description=Vox Probabilis backend
After=network.target

[Service]
Type=simple
User=vox
Group=vox
WorkingDirectory=/opt/voxprobabilis
Environment="PATH=/opt/voxprobabilis/venv/bin"
EnvironmentFile=/opt/voxprobabilis/.env
ExecStart=/opt/voxprobabilis/venv/bin/uvicorn app.main:app \
  --host 127.0.0.1 --port 8000 --workers 2
Restart=on-failure
RestartSec=5
# Harden
PrivateTmp=true
NoNewPrivileges=true
ProtectSystem=strict
ReadWritePaths=/var/lib/voxprobabilis /tmp

[Install]
WantedBy=multi-user.target
```

### 13.2 · nginx snippet (`deploy/nginx.conf.snippet`)

Drop this inside the existing `server { ... }` block for the domain. Do **not** overwrite Juan's existing nginx config.

```nginx
# Vox Probabilis API + static
location /api/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    proxy_set_header X-Forwarded-Proto $scheme;
    client_max_body_size 12M;
    proxy_read_timeout 30s;
}

location = / {
    proxy_pass http://127.0.0.1:8000/;
    proxy_set_header Host $host;
}

location /static/ {
    proxy_pass http://127.0.0.1:8000/static/;
    proxy_set_header Host $host;
}

# Leave /api-rust/ and anything else untouched — they already route to Juan's Rust service.
```

### 13.3 · Environment variables (`.env.example`)

```
# Required
VOX_SECRET_SALT=CHANGE_ME_TO_32_RANDOM_BYTES_BASE64
VOX_DB_PATH=/var/lib/voxprobabilis/vox.db

# Optional
VOX_LOG_LEVEL=INFO
VOX_COOKIE_SECURE=true    # set to false only for local dev
VOX_MAX_UPLOAD_MB=10
VOX_FREE_DAILY_QUOTA=3
```

### 13.4 · Install script (document in README, do not write a script)

```
# One-time setup on the VPS
sudo apt update && sudo apt install -y python3.11 python3.11-venv ffmpeg libsndfile1
sudo useradd -r -s /bin/false vox
sudo mkdir -p /opt/voxprobabilis /var/lib/voxprobabilis
sudo chown -R vox:vox /opt/voxprobabilis /var/lib/voxprobabilis
# Clone repo to /opt/voxprobabilis, create venv, pip install -r requirements.txt
# Run migrations: sqlite3 /var/lib/voxprobabilis/vox.db < migrations/001_initial.sql
# Copy .env.example → .env and fill in VOX_SECRET_SALT
# Copy deploy/voxprobabilis.service → /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now voxprobabilis
# Add nginx snippet to existing server block, reload nginx
```

---

## 14 · Testing

### 14.1 · Unit tests (required)

- `test_audio_load.py` — decode WAV, MP3, OGG fixtures; reject a corrupt file; reject a text file renamed `.wav`.
- `test_features.py` — on a known sine wave, jitter should be near 0, flux should be near 0 after first frame; on the truth fixture, all four features fall in expected ranges documented in §7.
- `test_baseline.py` — calibrate establishes baseline; subsequent analyze computes correct deltas; no session → uses global baseline with `confidence="low"`.
- `test_projection.py` — known feature deltas produce expected quadrant and axis values; edge case at origin returns `"ORIGIN"`.

### 14.2 · API integration tests (required)

- Full ritual flow: calibrate → analyze(uncertain) → analyze(lie) → analyze (counts against quota)
- Rate limit: 4th normal analyze in same day returns 429
- Calibration does not count against quota
- Invalid audio returns correct error code and HTTP status
- Session cookie is set on first visit and reused on subsequent calls

### 14.3 · Smoke test (document in README)

Manual: Juan runs `curl -F "audio=@my_voice.wav" http://localhost:8000/api/calibrate` with his own voice, then reads the SQLite DB to confirm his baseline numbers match what the paper reported for him (±5% tolerance). This is the single most important test — if this fails, the scientific claim fails.

---

## 15 · Observability

- stdlib `logging`, configured to emit JSON lines at INFO, pipe to journald via systemd
- Log format: `{"ts": "...", "level": "INFO", "event": "analyze", "session_id_hash": "abc...", "duration_ms": 142, "quadrant": "NATURAL_CALM", "confidence": "medium"}`
- Never log: raw audio, full session IDs (hash them in logs), IPs (use the hash)
- `/api/health` returns DB-writable and ffmpeg-callable status
- No Prometheus for v0.1. Juan can `tail -f /var/log/syslog | grep voxprobabilis` if needed.

---

## 16 · Explicitly Deferred (v0.2 and beyond)

- Browser microphone recording (MediaRecorder)
- Lemon Squeezy / Stripe payment flow
- API keys for paid tiers
- Batch CSV upload
- PDF export
- Admin dashboard
- PT-BR localization
- Postgres migration
- Docker packaging
- A/B testing framework
- Webhook notifications
- Retraining the global baseline from opt-in data

---

## 17 · Definition of Done

All of these must be true before Juan is told "it's ready":

1. `pytest` passes with all tests green.
2. `curl` smoke test for all four endpoints documented in the README works on localhost.
3. Running Juan's own voice through `/api/calibrate` reproduces his paper's baseline numbers within 5% (see §14.3).
4. Running his "lie" sample through `/api/analyze` after calibration produces `quadrant: "OVER_CONTROLLED_TENSE"`. If it does not, the product does not work — raise the problem to Juan before "finishing".
5. systemd service starts and survives `systemctl restart`.
6. nginx snippet drops cleanly next to the existing Rust service without breaking `/api-rust/`.
7. `README.md` is written and includes: setup, config, run, deploy, operate, troubleshoot sections.
8. No raw audio ever hits disk for more than the transient `/tmp` window; verify by running the full flow then `find /var/lib/voxprobabilis /tmp -name '*.wav' -o -name '*.mp3'` returning nothing.
9. The frontend HTML works end-to-end from intro → ritual → reveal with real backend data.

---

## 18 · For Claude Code: Workflow Suggestion

1. Read this entire SPEC.md once through.
2. Ask Juan the four questions from §0.
3. Scaffold the repo structure (§5).
4. Write `requirements.txt` and `.env.example`.
5. Implement in this order: `audio/load.py` → `audio/features.py` → tests for both → `audio/baseline.py` and `projection.py` + tests → `db.py` + migrations → `sessions.py` → `rate_limit.py` → `api/*.py` + tests → `main.py` → systemd + nginx files → README.
6. Run smoke test (§14.3) with Juan's voice before declaring done.
7. If any test from §14 fails or DoD (§17) is not met, report specifically what failed — do not paper over.

---

## Appendix A · References

- Fausto, J. & Claude (2026). *An Entropy-Oriented FFT Analysis of Voice Stress.* Zenodo. [10.5281/zenodo.19396809](https://doi.org/10.5281/zenodo.19396809)
- Fausto, J. & Claude (2026). *AGI✕ Logos Probabilis: The Senses of a New Species.* Zenodo. [10.5281/zenodo.19478167](https://doi.org/10.5281/zenodo.19478167)
- Dixon, S. (2006). *Onset detection revisited.* DAFx. (spectral flux definition)
- Boersma, P. (2001). *Praat, a system for doing phonetics by computer.* Glot International. (jitter measurement)
- Lippold, O. (1971). *Physiological tremor.* Scientific American. (8–12 Hz tremor band)

---

*This spec is a living document. If Claude Code finds ambiguity, it should ask Juan rather than guess. If Juan wants to change something fundamental, he updates this file and it becomes v0.2 of the spec, not an undocumented drift from v0.1.*
