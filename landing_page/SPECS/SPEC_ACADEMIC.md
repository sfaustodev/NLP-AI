# Vox Probabilis Academic — SPEC.md

```
Authors          Juan Fausto, Claude (Anthropic)
Version          0.1.0-academic · 2026-05-16
Companion        ./SPEC.md (V1 original, foundational)
                 ./DEPLOY.md (production deploy of V1)
                 ./SPEC_COACH.md (sibling product)
Target           Law students and law professors
Domain           voxprobabilis.com/academic (shares backend with V1)
License          MIT
Status           DRAFT — first SPEC of the educational product line
```

---

## 0 · For Claude Code · Read This First

The V1 backend (`./SPEC.md`) is already implemented and deployed (see `./DEPLOY.md`). This SPEC describes the **educational sibling product**, which:

- Reuses the V1 backend feature extraction (jitter, MFCC delta variance, spectral flux, microtremor) **without modification to the core math**
- Adds **multi-segment analysis** instead of single-baseline-vs-single-sample
- Adds **per-witness baseline** (recalibrated for each speaker in a hearing)
- Adds **phase detection heuristics** (Fase 1: Opening → Fase 2: Calibration → Fase 3: Judge inquiry → Fase 4: Adversarial counsel) — see §6
- Adds a **legal-format Term of Use** with restrictive purpose clause (educational only, see §8)
- Adds a **declarative eligibility filter** at upload (user attests: "this hearing is public and/or transit-judged, contains no sealed proceedings")
- Wraps the cartesian projection in **comparative-relative framing** rather than absolute scoring — see §7

**Do not touch the V1 endpoints in ways that break backward compatibility.** Academic endpoints live under `/api/academic/*`. V1 endpoints (`/api/calibrate`, `/api/analyze`) stay as-is.

**Before coding, ask Juan these four questions:**

1. Should Academic share the same SQLite database as V1 (preferred for simplicity) or a separate `academic.db` file?
2. Pricing flag location: env var (`VOX_ACADEMIC_PRICING_MODE`) or hardcoded in `app/academic/pricing.py`?
3. Hearing audio: support only `.wav` first (lowest friction), or also `.mp3/.opus/.m4a` from day one?
4. Report export format: HTML (in-browser) only, or HTML + PDF download?

Defaults if Juan doesn't answer in 24h: same DB, env var, all audio formats (the V1 stack already handles them), HTML only with PDF deferred to v0.2.

---

## 1 · Product Definition

Vox Probabilis Academic is a **comparative prosodic analysis tool for legal hearings**, intended for **post-verdict** material used in **educational contexts**. It is **not** a forensic device, **not** a lie detector, **not** evidence-producing.

What it produces: a structured report containing
- Per-segment prosodic features (4 numbers per segment)
- A 2D cartesian projection showing each segment as a point
- A colorimetric transcript marking shifts in prosodic consistency
- A textual summary describing what *changed* across phases of the hearing

What it does **not** produce: any binary classification of truth/lie, any score assigning probability of deception, any output usable as judicial evidence.

The intended user: a law student or law professor studying the structure of oral evidence, witness consistency under cross-examination, and the textures of legal procedure. The product is a teaching instrument; conclusions are the user's, not the system's.

---

## 2 · Architecture · Final State After Deploy

```
                                    browser
                                       │
                                       ▼
                            Cloudflare (free tier)
                            voxprobabilis.com
                                       │
                                       ▼
                              nginx (reuse V1 config)
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
       /  → V1 landing       /academic → Academic        /coach → Coach
                                       │
                                       ▼
                                   :8002
                                       │
                ┌──────────────────────┼─────────────────────┐
                ▼                      ▼                     ▼
        app.main (V1)         app.academic.routes     app.coach.routes
                                       │
                                       ▼
                              app.features (shared)
                                       │
                                       ▼
                          /var/lib/voxprobabilis/vox.db
```

Backend code organization:

```
/opt/voxprobabilis/
  app/
    main.py                    # V1 root — unchanged
    features/                  # shared feature extraction — unchanged
    academic/
      __init__.py
      routes.py                # /api/academic/* endpoints
      phase_detect.py          # heuristics from §6
      report.py                # builds comparative-relative report
      eligibility.py           # declarative filter (§8)
    coach/                     # sibling product — see SPEC_COACH.md
```

---

## 3 · User Flow

1. User navigates to `voxprobabilis.com/academic` and sees the educational landing page
2. User scrolls to the analyzer or clicks "Start Analysis"
3. **Onboarding modal** (3 screens):
   - Screen 1: what this is / what it is not (mirrors the §1 framing)
   - Screen 2: eligibility declaration (single checkbox + 3-line statement, see §8)
   - Screen 3: brief technical guide ("upload up to 60 min of audio, segment by speaker, get report")
4. User uploads an audio file (`.wav`, `.mp3`, `.opus`, `.m4a` accepted, max 200 MB)
5. Backend transcribes via Whisper-1 OpenAI API, returns transcript with timestamps and speaker diarization
6. Frontend renders transcript with auto-detected phase boundaries (§6) highlighted; user can manually adjust boundaries (drag handles on timeline)
7. User confirms boundaries → clicks "Analyze"
8. Backend extracts features per segment, computes per-witness baselines, computes deltas, generates cartesian projection
9. Frontend renders the report (§7) in the same view, with transcript on the left and cartesian plot on the right
10. User can:
    - Download the report as HTML (v0.1) or PDF (v0.2)
    - Save the session to their account for later review (paid tiers)
    - Share a read-only link with classmates (paid tiers, future)

---

## 4 · Endpoints

All new endpoints under `/api/academic/`. Authentication via session cookie (same as V1) plus tier check (free, student, professor, institution).

### 4.1 · `POST /api/academic/transcribe`

Receives multipart upload of audio file. Returns transcription with timestamps, speaker diarization, and **auto-detected phase boundaries** (see §6).

Request:
```
multipart/form-data:
  audio: file (.wav/.mp3/.opus/.m4a, max 200 MB)
  eligibility_attested: bool (must be true; backend rejects with 400 if false)
```

Response 200:
```json
{
  "transcription_id": "trn_abc123",
  "duration_s": 1572,
  "transcript": [
    {
      "start": 64.0, "end": 130.0,
      "speaker": "S1",
      "text": "...",
      "tokens": [{"start": 64.0, "end": 64.3, "text": "Boa"}, ...]
    },
    ...
  ],
  "auto_phases": [
    {"start": 0.0, "end": 64.0, "phase": "OPENING", "confidence": 0.92},
    {"start": 64.0, "end": 130.0, "phase": "CALIBRATION", "speaker_under_inquiry": "S1", "interlocutor": "JUDGE", "confidence": 0.88},
    {"start": 130.0, "end": 380.0, "phase": "JUDGE_INQUIRY", "speaker_under_inquiry": "S1", "interlocutor": "JUDGE", "confidence": 0.79},
    {"start": 390.0, "end": 790.0, "phase": "ADVERSARIAL_COUNSEL", "speaker_under_inquiry": "S1", "interlocutor": "OPPOSING_COUNSEL", "confidence": 0.71},
    ...
  ],
  "warnings": [
    "Audio codec detected: Opus (YouTube). Microtremor band may be partially attenuated.",
    "Two speakers had highly similar voice profiles (S2, S3) — diarization may have errors. Review boundaries before analyzing."
  ]
}
```

Error 400 if `eligibility_attested != true`:
```json
{ "error": { "code": "ELIGIBILITY_REQUIRED", "message": "You must attest that the hearing is public and contains no sealed proceedings." } }
```

### 4.2 · `POST /api/academic/analyze`

Receives transcription ID plus user-confirmed phase boundaries. Returns full prosodic analysis.

Request:
```json
{
  "transcription_id": "trn_abc123",
  "phases": [
    {"start": 64.0, "end": 130.0, "phase": "CALIBRATION", "witness_id": "W1", "interlocutor": "JUDGE"},
    {"start": 130.0, "end": 380.0, "phase": "JUDGE_INQUIRY", "witness_id": "W1", "interlocutor": "JUDGE"},
    ...
  ],
  "witnesses": [
    {"id": "W1", "label": "Ana Luiza (informante)", "role": "WITNESS_FOR_PLAINTIFF"},
    {"id": "W2", "label": "Gabriel (informante)", "role": "WITNESS_FOR_PLAINTIFF"}
  ]
}
```

Response 200:
```json
{
  "analysis_id": "anl_xyz789",
  "witnesses": {
    "W1": {
      "label": "Ana Luiza (informante)",
      "baseline_segment_id": "seg_001",
      "baseline_features": { "jitter": 0.0229, "mfcc_delta_var": 21.62, "spectral_flux": 44.63, "microtremor_rms": 0.0203 },
      "segments": [
        {
          "id": "seg_001", "phase": "CALIBRATION", "interlocutor": "JUDGE",
          "features": { ... }, "delta_pct_vs_baseline": { ... },
          "cartesian": { "x": 0.21, "y": -0.07 },
          "consistency_label": "BASELINE"
        },
        {
          "id": "seg_002", "phase": "JUDGE_INQUIRY",
          "features": { ... }, "delta_pct_vs_baseline": { "jitter": -8.0, "spectral_flux": -8.0, ... },
          "cartesian": { "x": -0.83, "y": -0.24 },
          "consistency_label": "SLIGHT_SHIFT"
        },
        {
          "id": "seg_003", "phase": "ADVERSARIAL_COUNSEL",
          "features": { ... }, "delta_pct_vs_baseline": { "jitter": 4.0, "spectral_flux": -26.0, "microtremor_rms": -28.0 },
          "cartesian": { "x": -0.54, "y": 1.34 },
          "consistency_label": "MAJOR_SHIFT"
        }
      ]
    },
    "W2": { ... }
  },
  "cartesian_summary": {
    "axis_x": "PROSODIC_NATURALNESS",
    "axis_y": "INVOLUNTARY_TENSION",
    "method": "z-score normalization across all segments in this hearing"
  },
  "warnings": [
    "Microtremor decreased under pressure for W1 — diverges from V1 paper baseline (n=3, controlled recording). See documentation: voxprobabilis.com/academic/methodology#microtremor-context-dependence"
  ]
}
```

### 4.3 · `GET /api/academic/report/{analysis_id}.html`

Returns the rendered HTML report (see §7). Includes:
- Header with hearing metadata (user-provided title, date, court)
- Transcript with colorimetric overlay
- Cartesian plot (SVG, inline, paleta Vox Probabilis)
- Per-witness summary tables
- Methodology footer with DOI reference

Public read-only link variant (paid tiers only): `/api/academic/share/{share_token}` returns the same HTML but with no editing UI.

### 4.4 · `GET /api/academic/quota`

Returns remaining quota for the authenticated session/user.

Response 200:
```json
{
  "tier": "STUDENT_MONTHLY",
  "analyses_used_this_period": 12,
  "analyses_remaining": 48,
  "period_resets_at": "2026-06-16T00:00:00Z"
}
```

### 4.5 · `POST /api/academic/upgrade`

Stub for v0.2. Returns 501 NOT_IMPLEMENTED in v0.1.0. Payment integration deferred.

---

## 5 · Database · Schema Additions

Migrations live in `/opt/voxprobabilis/migrations/`. Numbered sequentially after existing V1 migrations.

```sql
-- 003_academic_transcriptions.sql
CREATE TABLE IF NOT EXISTS academic_transcriptions (
    id TEXT PRIMARY KEY,                  -- 'trn_' + ulid
    session_id TEXT NOT NULL,
    user_id TEXT,                         -- nullable for free tier
    duration_s REAL NOT NULL,
    transcript_json TEXT NOT NULL,        -- full Whisper response
    auto_phases_json TEXT NOT NULL,
    eligibility_attested INTEGER NOT NULL DEFAULT 0,
    created_at INTEGER NOT NULL,          -- unix timestamp
    deleted_at INTEGER                    -- soft delete, retention §11
);
CREATE INDEX idx_trn_session ON academic_transcriptions(session_id);
CREATE INDEX idx_trn_created ON academic_transcriptions(created_at);

-- 004_academic_analyses.sql
CREATE TABLE IF NOT EXISTS academic_analyses (
    id TEXT PRIMARY KEY,                  -- 'anl_' + ulid
    transcription_id TEXT NOT NULL,
    session_id TEXT NOT NULL,
    user_id TEXT,
    phases_json TEXT NOT NULL,
    witnesses_json TEXT NOT NULL,
    result_json TEXT NOT NULL,            -- full analysis result
    created_at INTEGER NOT NULL,
    deleted_at INTEGER,
    FOREIGN KEY (transcription_id) REFERENCES academic_transcriptions(id)
);
CREATE INDEX idx_anl_transcription ON academic_analyses(transcription_id);
CREATE INDEX idx_anl_session ON academic_analyses(session_id);

-- 005_academic_quotas.sql
CREATE TABLE IF NOT EXISTS academic_quotas (
    session_id TEXT NOT NULL,
    period_start INTEGER NOT NULL,        -- unix timestamp of period start
    period_end INTEGER NOT NULL,
    tier TEXT NOT NULL,                   -- FREE | STUDENT_MONTHLY | STUDENT_ANNUAL | PROFESSOR | INSTITUTION
    analyses_used INTEGER NOT NULL DEFAULT 0,
    analyses_limit INTEGER NOT NULL,
    PRIMARY KEY (session_id, period_start)
);

-- 006_academic_share_tokens.sql
CREATE TABLE IF NOT EXISTS academic_share_tokens (
    token TEXT PRIMARY KEY,               -- random 32 chars
    analysis_id TEXT NOT NULL,
    created_by_session TEXT NOT NULL,
    created_at INTEGER NOT NULL,
    expires_at INTEGER,                   -- nullable = never expires
    revoked_at INTEGER,
    FOREIGN KEY (analysis_id) REFERENCES academic_analyses(id)
);
```

Retention policy:
- Free tier transcriptions: 30 days then hard delete
- Paid tier transcriptions: 365 days then hard delete (or user-triggered earlier)
- Analyses: same as parent transcription
- Quotas: rolling 90-day retention
- Share tokens: until `expires_at` or `revoked_at`, then hard delete

---

## 6 · Phase Detection Heuristics

The product's distinctive technical contribution. Live in `app/academic/phase_detect.py`.

### 6.1 · Inputs

- Whisper transcript with word-level timestamps
- Diarization output (speaker labels: S1, S2, ...)
- (Optional) user-provided hint: "this is a juizado especial" / "this is a varada de família" / "this is a júri" — informs prior

### 6.2 · Five canonical phases

```python
class Phase(Enum):
    OPENING = "OPENING"
    CALIBRATION = "CALIBRATION"
    JUDGE_INQUIRY = "JUDGE_INQUIRY"
    PLAINTIFF_COUNSEL_INQUIRY = "PLAINTIFF_COUNSEL_INQUIRY"
    ADVERSARIAL_COUNSEL = "ADVERSARIAL_COUNSEL"
    TRANSITION = "TRANSITION"   # between witnesses
    CLOSING = "CLOSING"
```

### 6.3 · Three independent signals (each scored 0.0 to 1.0)

**Signal A — Speaker change.** Diarization detects new dominant speaker. If new speaker holds the floor for >15 consecutive seconds, score = 1.0.

**Signal B — Tribunal vocative.** Regex match on speaker's turn: `\b(excelência|vossa excelência|doutor[a]?\s+\w+|meritíssim[ao])\b`. Match = 0.8; "excelência" specifically = 1.0.

**Signal C — Question-type semantic shift.** Single Haiku call per turn classifies the speaker's utterance into:
- `IDENTITY_CALIBRATION` (name, age, profession, relationship)
- `FACTUAL_INQUIRY` (what did you see, when, where)
- `INTERPRETATIVE_PRESSURE` (how could you know, are you certain, why did you assume)
- `PROCEDURAL` (housekeeping, technical, dismissal)

Phase transition from CALIBRATION to JUDGE_INQUIRY: signal C transitions from `IDENTITY_CALIBRATION` to `FACTUAL_INQUIRY`.

Phase transition from JUDGE_INQUIRY to ADVERSARIAL_COUNSEL: signals A + B + C all fire within a 30-second window.

### 6.4 · Confidence scoring

Phase boundary confidence = weighted average of the three signals at the transition point. Conservative weights for v0.1.0:
- Signal A (speaker change): 0.5
- Signal B (vocative): 0.2
- Signal C (semantic shift): 0.3

Confidence threshold for auto-marking: 0.65. Below that, the boundary is suggested but flagged as "REVIEW_RECOMMENDED" in the frontend.

### 6.5 · User override

Frontend always shows auto-detected boundaries with drag handles. User can adjust freely. The `POST /api/academic/analyze` request submits the **user-confirmed** boundaries, not the auto-detected ones.

### 6.6 · Forensic platform templates (v0.2)

Different proceeding types have different canonical structures. Templates to add in v0.2:
- Juizado Especial Cível (current default)
- Vara Criminal — has interrogatório do réu + oitiva de testemunhas separately
- Tribunal do Júri — has additional layer of jury questions
- Vara da Família — often sealed, lower elegibility

---

## 7 · Output Report · Structure and Framing

The report is what the user pays for. Its quality determines retention.

### 7.1 · Header section

- Title (user-provided): "Audiência: [hearing name]"
- Court: optional, user-provided
- Date of hearing: optional, user-provided
- Duration analyzed: auto from audio
- Number of witnesses: auto
- Methodology: hyperlink to `/academic/methodology` page

### 7.2 · Comparative-relative framing — the language matters

The report **never** says:
- "This witness was lying"
- "Probability of deception: X%"
- "Truth score: Y"

The report **does** say:
- "Witness A's prosodic signature shifted significantly between Phase 2 (calibration) and Phase 4 (adversarial inquiry): spectral flux decreased by 26%."
- "Witness B remained prosodically consistent across all phases (variation under 8% in all features)."
- "During Witness B's voluntary statement at 18:58, prosodic features moved in a direction consistent with conviction rather than tension: jitter decreased while MFCC delta variance increased."

Wording lives in `app/academic/report_strings.py`. All strings versioned and reviewed for legal exposure.

### 7.3 · Cartesian plot

Inline SVG, paleta Vox Probabilis (background #0a0a0a, axes #3a1a1a, witness 1 #c33a3a vermelho, witness 2 #d9b873 bege). Generated server-side with matplotlib + savefig to SVG, then inlined. Use the existing `plot_audiencia.py` from the validation session as the template.

Axes:
- X = `(spectral_flux_z + mfcc_delta_var_z) / 2` — labeled "Prosodic Naturalness ←→ Flattening"
- Y = `(jitter_z - microtremor_z) / 2` — labeled "Laryngeal Relaxation ←→ Involuntary Tension"
- Markers: `○` calibration, `▢` inquiry phases, `★` voluntary or anomalous statements (user-tagged)

### 7.4 · Colorimetric transcript

Each segment colored by deviation from witness's own baseline:
- **Green** (#3a7a3a): consistency_label = `BASELINE` or `MINOR_VARIATION` (delta < 10%)
- **Yellow** (#a89878): `SLIGHT_SHIFT` (10–20%)
- **Orange** (#b87a3a): `NOTABLE_SHIFT` (20–35%)
- **Red** (#c33a3a): `MAJOR_SHIFT` (>35%)

Hover over a segment: tooltip with the four feature deltas and the phase label. Click: expands to show the underlying transcript text and a 10-second mini-spectrogram of that segment.

### 7.5 · Per-witness summary card

For each witness:
- Baseline established at: [phase], [duration] s
- Number of segments analyzed
- Most consistent phase: [name]
- Most shifted phase: [name] with [most-changed feature] at [delta]%
- Trajectory description: "Witness migrated from EXPRESSIVO_ESTAVEL toward OVER_CONTROL_TENSO during cross-examination."

### 7.6 · Methodology footer

Inline at the bottom of every report:

> Methodology: Comparative prosodic analysis based on spectral features (jitter via Praat, MFCC delta variance, spectral flux, microtremor envelope 8–12 Hz). Per-witness baseline calibration. Method documented in *AGI~~ Logos Probabilis*, Chapter 1, and in foundational paper DOI 10.5281/zenodo.19396809. This report does not classify truth or deception; it documents prosodic consistency across procedural phases. Interpretation is the responsibility of the reader. See terms of use at voxprobabilis.com/academic/terms.

---

## 8 · Eligibility Filter & Term of Use

### 8.1 · Eligibility declaration (single screen, single checkbox, three lines)

```
Declaro, sob minhas responsabilidades civil e acadêmica, que:

1. O áudio que estou enviando é de audiência pública ou transitada em julgado.
2. Não contém matéria coberta por segredo de justiça, sigilo de menor, ou sigilo familiar.
3. Estou usando esta análise para fins educacionais, pedagógicos, ou de pesquisa acadêmica.

☐ Confirmo as três declarações acima.
```

Without the checkbox, `POST /api/academic/transcribe` returns 400 `ELIGIBILITY_REQUIRED`. The session is marked `eligibility_attested = true` for 24 hours after confirmation; subsequent uploads in that window skip the screen.

### 8.2 · Term of Use · structured as legal document

Located at `voxprobabilis.com/academic/terms`. Formatted with article numbering (`Art. 1º`, `Art. 2º`, ...). Three pages target length.

Key clauses:

**Art. 1º — Definição do serviço.** Define what Academic is (analysis tool) and what it is not (not evidence, not polygraph, not forensic).

**Art. 2º — Finalidade exclusivamente educacional.** Use restricted to: classroom teaching, academic research, individual study, pedagogical preparation. Use as evidence in active proceedings is **vedada** (forbidden).

**Art. 3º — Vedação ao uso processual em curso.** Reproduces the §1 language: outputs of Academic must not be incorporated into petitions, technical opinions submitted to court, or any document intended to produce legal effects in active proceedings.

**Art. 4º — Responsabilidade do usuário.** User declares eligibility of source material per §8.1. Service operator is not responsible for misuse.

**Art. 5º — LGPD.** Voice is sensitive personal data under Lei Geral de Proteção de Dados Art. 11. Service operator's lawful basis for processing is the academic purpose exception (Art. 7, IV) read together with the user's eligibility declaration. User accepts that data is processed for academic purposes.

**Art. 6º — Cláusula de reprodutibilidade.** All numerical outputs can be reproduced by re-running the open-source pipeline (link to GitHub repo) on the same audio. This is the integrity guarantee: no black box, no proprietary scoring, no hidden weights.

**Art. 7º — Limitação de responsabilidade.**

I. O serviço é fornecido em modalidade "no estado em que se encontra" (*as is*). O operador não garante adequação a finalidade processual ou forense específica.

II. Em caso de falha técnica do serviço — indisponibilidade temporária, erro de análise, perda de transcrições por falha do sistema — a responsabilidade do operador é limitada ao maior dos seguintes valores:
  (a) o valor pago pelo usuário nos 30 dias anteriores ao incidente;
  (b) R$ 500,00 (quinhentos reais).

III. A limitação prevista no inciso II não se aplica a casos de dolo ou culpa grave do operador, nem afasta direitos do consumidor previstos no Código de Defesa do Consumidor.

IV. O operador não é responsável por decisões acadêmicas, processuais ou interpretativas tomadas pelo usuário com base nos outputs do Academic. O Academic é instrumento de análise comparativa; a interpretação e a decisão são exclusivas do usuário.

**Art. 8º — Foro.** Para dirimir quaisquer questões decorrentes deste Termo de Uso, fica eleito o foro da Comarca de Porto Seguro, Estado da Bahia, com renúncia expressa a qualquer outro, por mais privilegiado que seja.

### 8.3 · Acceptance UI

Modal on first visit to `/academic`. Three screens (carousel). Final screen has the checkbox + "Aceitar e continuar" button. Acceptance is logged to `academic_quotas.terms_accepted_at`. Re-acceptance required when Term of Use version changes (track via env var `VOX_ACADEMIC_TOU_VERSION`).

---

## 9 · Pricing & Tiers

Authoritative pricing in `app/academic/pricing.py`:

```python
PRICING_USD = {
    "FREE":               {"analyses_per_period": 1,  "period_days": 30,  "price": 0.00},
    "AVULSA":             {"analyses_per_period": 1,  "period_days": 30,  "price": 2.22, "one_time": True},
    "STUDENT_MONTHLY":    {"analyses_per_period": 60, "period_days": 30,  "price": 16.60},
    "STUDENT_ANNUAL":     {"analyses_per_period": 82, "period_days": 30,  "price": 127.49, "annual": True},
    "PROFESSOR":          {"analyses_per_period": -1, "period_days": 30,  "price": 36.00},  # -1 = unlimited
    "INSTITUTION_SEAT":   {"analyses_per_period": -1, "period_days": 365, "price": 289.44}, # 33% off annual professor
}
```

`STUDENT_ANNUAL` gets 36% discount + 36% more monthly analyses (60 × 1.36 = 82).

`INSTITUTION_SEAT` triggers automatically when 3+ professor seats are purchased under the same institution_id. Existing seats are retroactively repriced.

Free tier: 1 analysis per 30 days. Marketing/funnel design — the goal is to get the user to feel the value, not to give the product away.

Payment integration: Lemon Squeezy. Deferred to v0.2 — for v0.1.0, all paid tiers are activated manually by Juan via direct DB edit or a CLI script. **Do not block product launch on payment.**

---

## 10 · Frontend

### 10.1 · Page structure

`/academic` is a single-page app. Sections:

1. **Hero** — Title "Vox Probabilis Academic", subtitle "Análise prosódica de audiências públicas para fins educacionais", a small DOI badge linking to Zenodo.
2. **What this is / what it is not** — two columns. Left: green checkmarks of legitimate uses. Right: red X marks of prohibited uses.
3. **How it works** — three steps with icons: upload → confirm phase boundaries → read the report.
4. **Methodology summary** — 4-paragraph plain-language explanation of the pipeline. Links to the paper.
5. **Pricing** — four cards. Sticky CTA on the right side as user scrolls.
6. **The Analyzer** — embedded directly in the page, below pricing. No separate route. Reduces friction.
7. **Footer** — links to terms, privacy, methodology page, GitHub repo, paper DOI, book DOI.

### 10.2 · Reuse from V1

- Color tokens (`--bg`, `--ink`, `--accent-red`, `--accent-beige`) — define once in `static/css/tokens.css`, import everywhere
- Modal component, button styles, form styles
- Cartesian plot widget (extract to `static/js/cartesian.js`, parameterize for any data shape)

### 10.3 · Specific to Academic

- File upload widget with drag-and-drop, max 200 MB visible counter, audio waveform preview after upload
- Phase boundary editor — horizontal timeline with draggable boundary markers, each boundary labeled with phase name, double-click to edit
- Witness assignment panel — auto-detected speakers as cards, user labels each ("Ana Luiza", "Gabriel")
- Report view — split-pane: transcript left (colorimetric), cartesian plot top-right, witness summaries bottom-right

### 10.4 · Accessibility & responsive

Minimum supported: 1280×800 desktop. Mobile is **not** supported in v0.1.0 — explicit message on smaller viewports: "Vox Probabilis Academic é otimizado para tela de pelo menos 1280px. Acesse de um computador para a melhor experiência." Mobile defer to v0.3.

---

## 11 · Definition of Done · v0.1.0

1. All endpoints in §4 implemented and responding correctly to happy-path requests
2. Migrations 003–006 applied and pass integrity checks
3. Phase detection signals A, B, C implemented (Haiku integration for C)
4. Eligibility filter blocks uploads without attestation
5. Term of Use page live at `/academic/terms`, accessible from footer and onboarding modal
6. Report rendering produces a valid HTML report with all sections from §7
7. Cartesian plot renders correctly with paleta Vox Probabilis
8. Free tier quota enforces 1 analysis per 30 days
9. Manual upgrade to paid tier via CLI script works
10. The Santo André hearing (validation case from elo #42 mycorrhiza) reproduces the same numerical results when uploaded through the Academic frontend as it did during validation
11. Privacy policy at `/academic/privacy` covers Academic-specific data handling
12. Smoke test: full flow upload → analyze → report → terms → privacy works in Chrome and Firefox on desktop
13. nginx routing: `/academic/*` routes to FastAPI `/api/academic/*`
14. systemd reload deploys without breaking V1 endpoints
15. README in `/opt/voxprobabilis/app/academic/README.md` documents architecture, endpoint contracts, and how to add a new pricing tier

---

## 12 · Out of Scope for v0.1.0

- Payment processing (Lemon Squeezy) — manual upgrade only
- Mobile responsive layout — desktop only
- PDF report export — HTML only
- Public share links — deferred to v0.2
- Account system — session-based only in v0.1
- Email notifications — none
- Multilingual UI — Portuguese only at launch (English locale planned, see §13)
- Forensic platform templates beyond Juizado Especial Cível
- Real-time analysis during upload (full-pipeline only)
- Audio formats beyond .wav/.mp3/.opus/.m4a
- Spectrogram visualization in the report (planned for v0.2)

---

## 13 · Internationalization · Sketch for Future Session

Frontend uses a `data-i18n="key.path"` attribute on every translatable element. Locale files at `static/locales/{lang}.json`. Language detection order:
1. `?lang=pt-BR` query parameter (explicit override)
2. `vox_lang` cookie (user preference)
3. `Accept-Language` HTTP header from browser
4. Default: `pt-BR`

Locale switch UI: dropdown in footer, top-right of header.

Backend response language: API responses include user-facing text in the language requested via `Accept-Language` header. Internal codes (`OVER_CONTROLLED_TENSE`, etc.) stay in English. Only display strings localize.

Implementation order (separate sprint, not v0.1.0):
1. Extract all hardcoded strings into `pt-BR.json`
2. Add `i18n.js` runtime that swaps strings on DOM ready
3. Create `en.json` as English translation of all keys
4. Wire detection logic
5. Test toggle works across all pages

---

## 14 · Appendix · Environment Variables

```ini
# Academic-specific (add to /opt/voxprobabilis/.env)
VOX_ACADEMIC_ENABLED=true
VOX_ACADEMIC_FREE_QUOTA=1                          # analyses per 30 days for free tier
VOX_ACADEMIC_MAX_UPLOAD_MB=200
VOX_ACADEMIC_WHISPER_API_KEY=sk-...                # OpenAI Whisper API
VOX_ACADEMIC_HAIKU_API_KEY=sk-ant-...              # Anthropic Haiku for phase classification
VOX_ACADEMIC_TOU_VERSION=2026-05-16                # bump on terms change → force re-accept
VOX_ACADEMIC_RETENTION_FREE_DAYS=30
VOX_ACADEMIC_RETENTION_PAID_DAYS=365
```

---

## 15 · Handoff Script · When Done

When DoD §11 is met, Claude Code reports back:

```
ACADEMIC v0.1.0 — READY FOR LAUNCH

✓ All 4 endpoints live and tested
✓ Phase detection (A/B/C) producing reasonable results on Santo André hearing
✓ Eligibility filter blocks unattested uploads
✓ Term of Use accessible and accepted in onboarding
✓ Cartesian plot renders in paleta Vox Probabilis
✓ Free tier quota enforces 1/30d
✓ V1 endpoints unaffected (smoke test passed)
✓ nginx routes /academic/* correctly

URL: https://voxprobabilis.com/academic
Terms: https://voxprobabilis.com/academic/terms
Methodology: https://voxprobabilis.com/academic/methodology

Manual upgrade CLI: `python -m app.academic.cli upgrade --session SID --tier STUDENT_MONTHLY`

Known limitations of v0.1.0:
- Desktop only (mobile message shown)
- Manual payment tier activation
- HTML reports only (PDF v0.2)
- Portuguese only (English locale v0.2)
- Juizado Especial only (other platforms v0.2)

Next sprint candidates: Lemon Squeezy integration, English locale, PDF export.
```
