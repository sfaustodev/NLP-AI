# Vox Probabilis Coach — SPEC.md

```
Authors          Juan Fausto, Claude (Anthropic)
Version          0.1.1-coach · 2026-05-16
Companion        ./SPEC.md (V1 original)
                 ./DEPLOY.md (production deploy of V1)
                 ./SPEC_ACADEMIC.md (sibling product)
Target           Practicing lawyers preparing clients/witnesses for hearings
Domain           voxprobabilis.com/coach
License          MIT
Status           DRAFT — first SPEC of the professional product line
Changelog        v0.1.1 (2026-05-16):
                 - Pricing finalized: FREE / Tier 1 $36 / Tier 2 $77 / Tier 3 $141.90
                 - Annual discount: 36% on all paid tiers
                 - Tier 2 retention features fleshed out: Cofre, Trajectory, Diff,
                   Tags, Pre-Hearing Brief, Personalized Reports
                 - Art. 7º liability cap clarified (max of: 12mo paid OR BRL 1.000)
                 - Safari officially out of scope (no fallback work)
                 - Mic quality: warning only, not hard block
                 - PDF generation (Terms, Consent, Reports, Brief) moved into v0.1.1 DoD
```

---

## 0 · For Claude Code · Read This First

The V1 backend (`./SPEC.md`) is already implemented and deployed. The Academic sibling (`./SPEC_ACADEMIC.md`) extends V1 for post-verdict educational analysis. **Coach is different in nature** — not analysis of recorded material, but **live interactive sessions** between a lawyer and their client.

Key differences from V1 and Academic:

| Dimension | V1 | Academic | **Coach** |
|---|---|---|---|
| Use case | Curiosity / research | Education / study | **Practice preparation** |
| Timing | Single recording | Post-verdict batch | **Live, interactive** |
| Audio source | File upload | File upload | **Browser MediaRecorder** |
| Baseline | None | Per-witness | **Per-session, immutable after calibration** |
| Audience | Public | Students / professors | **Lawyer + their client (1-on-1)** |
| Output framing | Result | Comparative report | **Real-time coaching feedback** |
| Report LLM | None | Haiku | **Sonnet reasoning low (T1) or Opus high (T2/T3)** |
| Cross-session features | No | No | **Cofre + Trajectory + Diff + Brief (T2+)** |

The product is **not** an evidence-producing device. It is a **practice tool** — analogous to a sports coach reviewing video with an athlete before a match.

**Before coding, ask Juan these three remaining questions** (most others resolved in v0.1.1):

1. Audio retention behavior in Tier 2/3: **features only** (current default) OR add an opt-in feature-flag for retaining anonymized 5-second snippets for later review? (Higher LGPD complexity, but lawyers may request it.)
2. Concurrent sessions per account: 1 at a time, or N parallel? (v0.1.1 default: 1 at a time per lawyer account.)
3. Naming of the Tier 2 feature suite — Juan suggested **Cofre** for storage; confirm naming of Trajetória, Diff Inter-Sessão, Pre-Hearing Brief before launch (or propose Portuguese variants).

Resolved in v0.1.1:
- Pricing — see §9
- Safari — explicitly out of scope
- Mic quality block — warning only
- PDF generation — included in v0.1.1

---

## 1 · Product Definition

Vox Probabilis Coach is a **live interactive practice tool** for lawyers preparing clients or witnesses for hearings. Flow: lawyer creates session → client joins → calibration phrase establishes per-session baseline → lawyer/client cycle of question/response → real-time prosodic feedback per response → session report at the end.

What it produces:
- Per-answer prosodic feedback (color indicator + 4 feature deltas vs baseline)
- Session summary at end with aggregate stats
- For Tier 1: standard LLM-generated report (Sonnet reasoning low)
- For Tier 2/3: personalized Opus report drawing on cross-session history + Pre-Hearing Brief

What it does **not** produce:
- Any document usable as judicial evidence
- Any binary truth/lie classification
- Any "probability of deception" score

---

## 2 · Architecture

```
                                  browser
                                     │
                                     ▼
                          Cloudflare (free tier)
                          voxprobabilis.com
                                     │
                                     ▼
                            nginx (shared)
                                     │
                                     ▼
                                   :8002
                                     │
                                     ▼
                          app.coach.routes
                                     │
                                     ▼
                          app.features (shared with V1)
                                     │
                                     ▼
                  /var/lib/voxprobabilis/vox.db (shared)
```

```
/opt/voxprobabilis/
  app/
    main.py                    # V1 — unchanged
    features/                  # shared
    academic/                  # sibling product
    coach/
      __init__.py
      routes.py                # /api/coach/* endpoints
      session.py               # state machine
      mic_quality.py           # SNR estimator
      baseline.py              # per-session baseline
      feedback.py              # real-time scoring
      reports/
        sonnet_standard.py     # Tier 1
        opus_personalized.py   # Tier 2/3
      retention/               # Tier 2+ features
        vault.py               # Cofre — session storage and retrieval
        trajectory.py          # cross-session evolution chart
        diff.py                # inter-session comparison
        tags.py                # private annotations
        brief.py               # Pre-Hearing Brief generator
      pdf/                     # all PDF generation via reportlab
        terms.py               # Term of Use PDF
        consent.py             # Client consent template PDF
        session_report.py
        brief.py
```

---

## 3 · User Flow

### Pre-session (lawyer side)

1. Lawyer logs into `/coach` (existing account or starts free trial)
2. Creates new session: name, optional question list, **optional client_id from Cofre** (Tier 2+)
3. System generates session URL with token
4. Lawyer joins immediately, or shares URL with client to join remotely

### Joining the session

5. Both parties grant mic permission (HTTPS for `getUserMedia`)
6. Mic quality check runs (§6); GREEN/YELLOW silent or warn, RED warns strongly but allows proceed
7. Client speaks one of three rotated calibration phrases:
   - "Meu nome completo é [nome], hoje é [data], estou em [cidade]."
   - "Acordei hoje às sete da manhã, tomei café, e venho aqui falar sobre o caso."
   - "Eu trabalho com [profissão], moro em [cidade], e estou colaborando com a preparação."
8. Backend records 8 seconds, extracts features, stores as `session.baseline_features` (immutable)

### Practice loop

9. **🎤 Pergunta** → lawyer asks; not analyzed, just context
10. **🎤 Resposta** → before client speaks
11. Client answers
12. **⏹️ Parar** when finished
13. Backend extracts features, computes deltas vs baseline, returns within 2.5s
14. Response card shows: color indicator, four deltas, plain-language summary
15. **⏭️ Próxima** to advance, **🔁 Repetir** to redo
16. (Tier 2+) Lawyer can tag any response with custom tags

### Session end

17. Lawyer ends session → backend computes summary
18. Tier 1: Sonnet reasoning low generates standard report
19. Tier 2/3: Opus high generates personalized report with cross-session history (if client linked)
20. Session report available as HTML and PDF
21. Tier 2/3: session auto-saved to Cofre under linked client (if any)

---

## 4 · Endpoints

All under `/api/coach/`. Authentication via lawyer session cookie + per-session token for joiners.

### 4.1 · `POST /api/coach/session/create`
```json
{
  "session_name": "...",
  "planned_questions": ["..."],
  "attach_to_client_id": "cli_abc123"   // optional, Tier 2+ only
}
```

### 4.2 · `GET /api/coach/session/{session_token}`
Returns session state.

### 4.3 · `POST /api/coach/session/{session_token}/calibrate`
Submits calibration audio, returns mic_quality + baseline_features.

### 4.4 · `POST /api/coach/session/{session_token}/response`
Submits response audio, returns deltas + cartesian position + narrative.

### 4.5 · `POST /api/coach/session/{session_token}/response/{response_id}/tag` *(T2+)*
Adds private tags.
```json
{ "tags": ["fraco", "ensaiar"] }
```

### 4.6 · `POST /api/coach/session/{session_token}/end`
Triggers tier-appropriate report generation.

### 4.7 · `GET /api/coach/session/{session_id}/report.html`
HTML report.

### 4.8 · `GET /api/coach/session/{session_id}/report.pdf`
PDF report (reportlab).

### 4.9 · `GET /api/coach/quota`
Quota and feature flags.

### 4.10 · Cofre endpoints (T2+ feature-gated)

| Endpoint | Description |
|---|---|
| `GET /api/coach/cofre/clients` | List of lawyer's clients |
| `POST /api/coach/cofre/clients` | Create new client profile |
| `GET /api/coach/cofre/clients/{client_id}` | Client profile + sessions list |
| `DELETE /api/coach/cofre/clients/{client_id}` | LGPD erasure |
| `GET /api/coach/cofre/clients/{client_id}/trajectory` | Cross-session trajectory data |
| `POST /api/coach/cofre/clients/{client_id}/diff` | Compare two sessions |
| `POST /api/coach/cofre/clients/{client_id}/brief` | Generate Pre-Hearing Brief |
| `GET /api/coach/cofre/brief/{brief_id}.html` | Brief HTML |
| `GET /api/coach/cofre/brief/{brief_id}.pdf` | Brief PDF |

---

## 5 · Session State Machine

```
CREATED → AWAITING_CALIBRATION → READY → IN_PRACTICE ⟲ → ENDED
```

Transitions enforced server-side.

---

## 6 · Microphone Quality Check

Three signals: SNR (dB), sample rate (Hz), spectral centroid (Hz).

```python
def mic_label(snr_db, sr_hz, centroid_hz):
    base = "GREEN" if snr_db > 25 else "YELLOW" if snr_db > 15 else "RED"
    if sr_hz < 22000:
        base = "YELLOW" if base == "GREEN" else "RED"
    if not (800 <= centroid_hz <= 4500):
        base = {"GREEN": "YELLOW", "YELLOW": "RED", "RED": "RED"}[base]
    return base
```

**Frontend behavior — warning only, no hard block** (v0.1.1 decision):
- GREEN: silent
- YELLOW: banner "Qualidade moderada. Resultados podem ter precisão reduzida."
- RED: banner "Qualidade insuficiente. Resultados podem não ser confiáveis. Sugestões: fones com microfone, ambiente silencioso, próximo da boca." User can proceed.

---

## 7 · Per-Session Baseline & Tier 2+ Retention Features

### 7.1 · Per-session baseline (all tiers)

Each session establishes its **own** baseline from the calibration audio. All subsequent analysis is **relative to that baseline**, in **that microphone**, in **that environment**, with **that speaker**. No comparison against population baseline, no universal thresholds.

This eliminates entire classes of generalization failure documented in the V1 paper (n=3 microtremor pattern not generalizing → Santo André validation showed microtremor inverted; with per-session baseline, the absolute direction doesn't matter, only relative shift).

### 7.2 · Consistency thresholds

```python
def consistency_label(deltas):
    max_abs = max(abs(v) for v in deltas.values())
    if max_abs < 10:  return ("BASELINE", "GREEN")
    if max_abs < 20:  return ("SLIGHT_SHIFT", "YELLOW")
    if max_abs < 35:  return ("NOTABLE_SHIFT", "ORANGE")
    return ("MAJOR_SHIFT", "RED")
```

### 7.3 · Tier 2/3 Cofre system — the retention feature suite

The Cofre is what justifies paying Tier 2 over Tier 1. It transforms Coach from a one-off tool into a **client preparation continuum**.

#### 🗄️ **Cofre de Sessões**

Encrypted-at-rest persistent storage of session features (NOT audio — see §8.3). Searchable by client name, date, process reference, custom tags. Retention: 365 days from session end by default, lawyer can manually delete earlier, or set per-session retention shorter at creation time.

#### 👤 **Perfil de Cliente**

Lawyer creates client profile: name, optional process reference, free-form private notes. Multiple sessions attach to one profile and accumulate into a continuous record.

#### 📈 **Trajetória do Cliente**

Cross-session visualization. Each session's aggregate cartesian position is plotted, connected chronologically. Lawyer can immediately see whether the client is **improving** (response cloud converging toward baseline-naturalness across sessions) or **deteriorating** (responses diverging).

Practical example: Lawyer prepares João over 5 sessions. Trajectory shows João's mean response drifted from "over-control tenso" in session 1 to "expressivo estável" by session 5. Preparation worked.

#### 🔀 **Diff Inter-Sessão**

Pick any two sessions of the same client. System matches comparable questions (by `question_text` similarity or `question_index`), shows side-by-side:
- Questions that stayed stable → rehearsed narrative consistent
- Questions that improved → coaching worked
- Questions that deteriorated → something changed, lawyer should investigate

Each notable change gets an Opus-generated commentary line.

#### 🏷️ **Tags & Anotações Privadas**

Custom hashtags per response (#fraco, #ensaiar, #boa, #cuidado) + free-form notes. Searchable later. Visible only to the lawyer's own account.

#### 📋 **Pre-Hearing Brief** — the killer feature

The night before the hearing, lawyer hits "Gerar Brief". Opus reads:
- All sessions of this client
- All tags and annotations
- The cartesian trajectory
- Questions consistently stable, consistently unstable, improved, regressed

Outputs a structured pre-hearing brief (PDF, 2–3 pages):
- **Áreas mais sólidas** — sections where client is consistently natural; lawyer can lean on these in direct examination
- **Áreas de atenção** — sections where instability persists; lawyer should anticipate cross-examination attack here
- **Recomendações táticas** — order of direct examination questions (start where confidence is highest), reactive strategy for cross
- **Evolução** — narrative summary of preparation arc across sessions

This is the feature that justifies $77/month. It's the difference between "I rehearsed with my client" and "I have a quantified preparation record that my LLM partner synthesized into tactical guidance".

#### 🎯 **Highlight Reel** *(deferred to v0.2)*

Auto-generated 3-clip summary of the most prosodically anomalous moments per session, with brief LLM-generated commentary. Useful for sending to junior associate.

### 7.4 · Tier-aware report generation

```python
def generate_report(session, tier):
    if tier == "FREE_TRIAL":
        return basic_template_report(session)            # no LLM, just stats
    elif tier == "TIER_1_MONTHLY":
        return sonnet_standard_report(session)           # Sonnet reasoning low
    elif tier in ("TIER_2_MONTHLY", "TIER_3_MONTHLY"):
        history = load_client_history(session.client_id) if session.client_id else None
        return opus_personalized_report(session, history)   # Opus high, with history
```

---

## 8 · Term of Use & Legal Framing

### 8.1 · Term of Use · structured as legal document

Located at `voxprobabilis.com/coach/terms`. Article-numbered. **PDF generated via reportlab and downloadable from the page.**

**Art. 1º — Definição.** Coach é ferramenta interativa de preparação. Não é dispositivo forense, não é polígrafo, não é evidência judicial.

**Art. 2º — Finalidade exclusiva.** Uso restrito a sessões 1-on-1 entre advogado(a) e cliente(s) ou testemunha(s) da causa em que atua. Vedado o uso em sessões com partes contrárias, com testemunhas arroladas por outras partes sem o conhecimento delas, ou em qualquer contexto que viole o sigilo profissional do advogado.

**Art. 3º — Vedação ao uso processual.** É absolutamente vedado anexar relatórios produzidos pelo Coach a peças processuais, ou submetê-los como evidência em qualquer foro. Coach é ferramenta de preparação privada; seus outputs são para uso interno do escritório.

**Art. 4º — Consentimento do cliente.** O advogado declara, ao iniciar uma sessão, que obteve consentimento informado do cliente para a análise prosódica. Modelo de consentimento fornecido em `/coach/consent-template` (PDF para download e impressão).

**Art. 5º — LGPD.** Voz é dado pessoal sensível. Coach processa o áudio em memória, armazena apenas features extraídas (números) nos planos Tier 2+ que oferecem retenção, descarta o áudio bruto em todos os planos dentro de 60 segundos do processamento, e permite ao advogado solicitar exclusão da sessão a qualquer momento. Base legal: consentimento explícito do titular do dado (cliente/testemunha) intermediado pelo advogado.

**Art. 6º — Sigilo profissional.** O Coach não armazena conteúdo semântico das respostas. Apenas features prosódicas (4 números por resposta) e timestamps. As perguntas do advogado, quando fornecidas como texto, são armazenadas para o relatório — o advogado é responsável por não incluir matéria coberta por sigilo nesse campo.

**Art. 7º — Limitação de responsabilidade.** A responsabilidade civil máxima agregada do Operador, decorrente de qualquer reclamação relacionada ao serviço, fica limitada ao **maior** dos seguintes valores: **(i)** o total efetivamente pago pelo Usuário ao Operador nos 12 (doze) meses anteriores ao evento que ensejar a reclamação; ou **(ii)** R$ 1.000,00 (mil reais). Esta limitação não se aplica em caso de dolo ou culpa grave comprovada do Operador.

**Art. 8º — Foro.** Foro da Comarca de Porto Seguro, Bahia, com renúncia expressa a qualquer outro, por mais privilegiado que seja.

### 8.2 · Consent template (PDF)

`/coach/consent-template` serves a printable PDF (reportlab) for the lawyer to obtain client signature:

- Identification of lawyer, client, case
- Plain-language description of Coach
- Explicit authorization clause
- Right to revoke consent at any time
- Confirmation: no audio retained, only numerical features
- Signature lines (lawyer + client + witness optional)

Not legally mandatory under LGPD (lawyer is data controller), but recommended as defense-in-depth.

### 8.3 · No-audio-retention default

For **all** tiers, audio is deleted from disk within 60 seconds of feature extraction:
- `/tmp/coach/` wiped on session end
- systemd timer runs `coach_cleanup` every 5 minutes deleting orphan files
- DB stores only feature JSON, never audio paths

### 8.4 · LGPD data subject rights

Via dashboard, lawyer can (acting on behalf of client):
- Export all data for a given client (LGPD Art. 18 — portability)
- Delete all data for a given client (LGPD Art. 18 — erasure)
- Receive confirmation of what data is processed (LGPD Art. 18 — transparency)

### 8.5 · What is stored vs. what is not

| Data type | Tier 1 | Tier 2/3 |
|---|---|---|
| Audio file (raw) | Deleted <60s | Deleted <60s |
| Calibration features (4 numbers) | Session lifetime only | Retained until manual delete or 365d |
| Response features (4 numbers per response) | Session lifetime only | Retained until manual delete or 365d |
| Question text (if provided) | Session lifetime only | Retained until manual delete |
| Tags and annotations | Not available | Retained until manual delete |
| Client profile metadata | Not available | Retained until manual delete |
| Session reports (HTML/PDF) | 7 days then auto-delete | 365 days then auto-delete |

---

## 9 · Pricing · v0.1.1 final

Authoritative in `app/coach/pricing.py`:

```python
PRICING_USD = {
    "FREE_TRIAL": {
        "price_monthly": 0.00,
        "sessions_per_period": 1,
        "responses_per_session": -1,    # unlimited within the 1 session
        "reports_per_period": 0,        # no LLM, basic template only
        "report_model": None,
        "period_days": 30,
        "retention_enabled": False,
        "personalized_reports": False,
        "history_features": False,
        "label": "Trial — uma vez por conta"
    },
    "TIER_1_MONTHLY": {
        "price_monthly": 36.00,
        "sessions_per_period": 45,
        "responses_per_session": -1,
        "reports_per_period": 15,
        "report_model": "sonnet-reasoning-low",
        "period_days": 30,
        "retention_enabled": False,
        "personalized_reports": False,
        "history_features": False,
        "label": "Premium"
    },
    "TIER_2_MONTHLY": {
        "price_monthly": 77.00,
        "sessions_per_period": -1,
        "responses_per_session": -1,
        "reports_per_period": 50,
        "report_model": "opus-high",
        "period_days": 30,
        "retention_enabled": True,
        "personalized_reports": True,
        "history_features": True,
        "label": "Profissional"
    },
    "TIER_3_MONTHLY": {
        "price_monthly": 141.90,
        "sessions_per_period": -1,
        "responses_per_session": -1,
        "reports_per_period": -1,
        "report_model": "opus-high",
        "period_days": 30,
        "retention_enabled": True,
        "personalized_reports": True,
        "history_features": True,
        "label": "Escritório"
    },
}

# Annual discount applied to all paid tiers
ANNUAL_DISCOUNT = 0.36

def annual_price(tier_key: str) -> float:
    monthly = PRICING_USD[tier_key]["price_monthly"]
    return round(monthly * 12 * (1 - ANNUAL_DISCOUNT), 2)

# Computed annual prices (paid 1× per year):
# Tier 1 annual: $276.48 ($23.04/mês equivalent)
# Tier 2 annual: $591.36 ($49.28/mês equivalent)
# Tier 3 annual: $1089.79 ($90.82/mês equivalent)
```

**Funnel logic:**
- **FREE_TRIAL** — lawyer feels the experience once: 1 session, 0 LLM reports, basic template summary. Sees the cartesian plot and the per-response feedback. No polished narrative.
- **TIER_1 ($36/mês)** — daily use unlocked: 45 sessions cover 1–2 hearings/week × multiple practice rounds, 15 polished Sonnet reports for cases that matter. No retention — each session is standalone.
- **TIER_2 ($77/mês)** — crosses from "useful tool" to "permanent partner": ilimitado em sessões, 50 relatórios Opus personalizados, Cofre + Trajetória + Diff + Pre-Hearing Brief. The Brief feature alone justifies the price for any lawyer with weekly hearings.
- **TIER_3 ($141.90/mês)** — for firms with multiple lawyers and high hearing volume: tudo ilimitado com Opus.

**Sustainability check:**

Per-report Opus cost (estimated 8k input tokens reading session + history + 3k output tokens for narrative): ~$0.45.

- Tier 2 with 50 reports/month max → ~$22.50 LLM cost on $77 revenue → **71% gross margin**
- Tier 3 unlimited (assume p95 lawyer = 80 reports/month) → ~$36 LLM cost on $141.90 revenue → **75% gross margin**

Sonnet reasoning low for Tier 1 reports: ~$0.10/report. 15 reports × $0.10 = $1.50 LLM cost on $36 revenue → **96% gross margin**.

All healthy. Pricing has room.

**Payment integration**: Lemon Squeezy. Deferred to v0.2. For v0.1.1, all paid tiers activated manually by Juan via CLI script.

---

## 10 · Frontend

### 10.1 · Browser requirements

**Supported**: Chrome 90+, Firefox 90+, Edge 90+ (Chromium) on desktop (macOS, Windows, Linux).

**Explicitly out of scope** (v0.1.1):
- Safari (any version) — MediaRecorder Opus output is non-standard, would require AudioContext fallback
- Mobile browsers
- Tablets

Detection: on entry to `/coach`, check `navigator.userAgent` for Safari/iOS. Show clear message: "Vox Probabilis Coach atualmente suporta Chrome e Firefox em desktop. Suporte ao Safari está planejado para futura versão. Acesse de Chrome ou Firefox para usar o produto."

### 10.2 · UI structure

`/coach` (lawyer's home):
- Hero
- Active sessions panel
- Cofre panel (Tier 2+ only): client list, recent sessions, "Gerar Brief" CTA
- "Nova sessão" button
- Pricing card (collapsed if subscribed)

`/coach/session/[token]` (live session view):

```
┌─────────────────────────────────────────────────────────────┐
│ Vox Probabilis Coach — Sessão: [name]            [Encerrar] │
├─────────────────────────────────────────────────────────────┤
│   ┌─────────────────┐         ┌─────────────────────────┐   │
│   │ Mic: ● GREEN    │         │  Cartesian view         │   │
│   │ Baseline: ✓     │         │       •baseline         │   │
│   └─────────────────┘         │   • • responses         │   │
│   [cliente: João]             │      •                  │   │
│                                └─────────────────────────┘   │
│                                                              │
│   🎤 Pergunta   🎤 Resposta   ⏹️ Parar   ⏭️ Próxima         │
│                                                              │
│   Histórico da sessão:                                       │
│   Q1                                       🟢 BASELINE       │
│   Q2  [🏷 #ensaiar]                          🟡 SLIGHT        │
│   Q3  [🏷 #fraco]                            🟠 NOTABLE       │
│   Q4                                       🔴 MAJOR          │
└─────────────────────────────────────────────────────────────┘
```

`/coach/cofre/clients/[id]` (Tier 2+ client profile):

```
┌─────────────────────────────────────────────────────────────┐
│ João Silva · 5 sessões · Processo 1234567-89                │
├─────────────────────────────────────────────────────────────┤
│   [Trajetória cartesiana — 5 pontos conectados]              │
│                                                              │
│   Sessões anteriores:                                        │
│   ▸ Sessão 1 · 14 mai · 12 respostas · maior shift -32%     │
│   ▸ Sessão 2 · 16 mai · 14 respostas · maior shift -28%     │
│   ▸ Sessão 3 · 18 mai · 12 respostas · maior shift -19%     │
│   ▸ Sessão 4 · 20 mai · 14 respostas · maior shift -12%     │
│   ▸ Sessão 5 · 22 mai · 13 respostas · maior shift -8%      │
│                                                              │
│   [📋 Gerar Pre-Hearing Brief]   [🔀 Diff entre duas]        │
│   [➕ Nova sessão com este cliente]                           │
└─────────────────────────────────────────────────────────────┘
```

### 10.3 · Real-time latency budget

Target: response card appears within 2.5s of ⏹️ Parar; p95 < 4s.

| Stage | Budget |
|---|---|
| Browser → server upload | < 800 ms |
| Server feature extraction | < 1200 ms |
| Server → browser response | < 200 ms |
| UI render | < 300 ms |

### 10.4 · Key JS components

```
static/js/coach/
  recorder.js        - MediaRecorder wrapper, Opus encoding
  session.js         - State machine, WebSocket
  ui.js              - Button states, history rendering
  cartesian.js       - Shared with Academic
  consent_modal.js   - Initial consent dialog
  cofre.js           - Client list/profile/trajectory (T2+)
  brief.js           - Brief generator UI (T2+)
  tagger.js          - Tag and annotation UI (T2+)
```

---

## 11 · Definition of Done · v0.1.1

1. All 10 core endpoints in §4.1–4.9 implemented and happy-path tested
2. All 9 Cofre endpoints in §4.10 implemented and feature-gated (Tier 2+)
3. Session state machine enforces transitions correctly (no bypass paths)
4. Microphone quality check produces sensible labels; warns but does not block
5. Per-session baseline computed correctly and stored immutably after calibration
6. Real-time response analysis returns within 4s p95
7. Cartesian plot updates in real time as responses come in
8. **Term of Use PDF** generates correctly via reportlab and is downloadable from `/coach/terms`
9. **Consent template PDF** generates correctly and is downloadable from `/coach/consent-template`
10. **Session Report PDF** generates correctly via reportlab
11. **Pre-Hearing Brief PDF** generates correctly (Tier 2+)
12. Free trial enforces 1-session limit
13. Tier 1 enforces 45 session + 15 report monthly limit
14. Tier 2/3 LLM model switch works (Sonnet reasoning low vs Opus high)
15. Cofre persistence: sessions retrievable, queryable, deletable
16. Trajectory chart renders correctly with 2+ sessions of a single client
17. Diff Inter-Sessão produces sensible side-by-side output
18. Pre-Hearing Brief generates with proper section structure
19. Tags/annotations: create, retrieve, search by tag
20. Audio verified deleted within 60s of session end (no orphan files in `/tmp/coach/`)
21. nginx routes `/coach/*` correctly without affecting V1 or Academic
22. systemd reload deploys without breaking siblings
23. Browser flow tested in Chrome and Firefox on macOS and Windows
24. Safari shows clean unsupported-browser message
25. Privacy policy at `/coach/privacy` accurately describes retention by tier
26. LGPD data subject rights endpoints (export, erasure) work
27. Smoke test: Juan's lawyer friend runs a full mock session end-to-end and provides written feedback

---

## 12 · Out of Scope for v0.1.1

- Safari, mobile, tablet
- Lemon Squeezy payment integration (manual upgrade only)
- Multi-party sessions (1 lawyer + 1 client only)
- Concurrent sessions per account (1 at a time)
- Live transcription during session
- Spectrogram visualization
- Case management integrations (Astrea, ADVBOX — v0.3)
- Multilingual UI — Portuguese only at launch
- Custom calibration phrases (3 pre-canned only)
- Highlight Reel (sketched §7.3, deferred to v0.2)
- Audio retention as a feature (currently always deleted <60s)
- WebRTC peer-to-peer audio
- Mid-session real-time feedback to client (only lawyer sees results)

---

## 13 · Internationalization · Same as Academic

Reuses i18n infrastructure designed in `SPEC_ACADEMIC.md §13`. Coach keys prefixed `coach.*`.

---

## 14 · Appendix · Environment Variables

```ini
VOX_COACH_ENABLED=true
VOX_COACH_FREE_TRIAL_SESSIONS=1
VOX_COACH_MAX_RESPONSE_DURATION_S=120
VOX_COACH_MIN_RESPONSE_DURATION_S=2
VOX_COACH_AUDIO_DELETE_DELAY_S=60
VOX_COACH_THRESHOLD_BASELINE=10
VOX_COACH_THRESHOLD_SLIGHT=20
VOX_COACH_THRESHOLD_NOTABLE=35
VOX_COACH_SESSION_TIMEOUT_MIN=60
VOX_COACH_TOU_VERSION=2026-05-16
VOX_COACH_RETENTION_MAX_DAYS=365

VOX_COACH_SONNET_API_KEY=sk-ant-...
VOX_COACH_OPUS_API_KEY=sk-ant-...
VOX_COACH_SONNET_MODEL=claude-sonnet-4-7-reasoning-low
VOX_COACH_OPUS_MODEL=claude-opus-4-7-high
```

---

## 15 · Database · Schema Additions

```sql
-- 007_coach_sessions.sql
CREATE TABLE IF NOT EXISTS coach_sessions (
    id TEXT PRIMARY KEY,
    session_token TEXT UNIQUE NOT NULL,
    owner_user_id TEXT NOT NULL,
    client_id TEXT,                          -- Tier 2+ links session to client
    session_name TEXT NOT NULL,
    state TEXT NOT NULL,
    baseline_features TEXT,
    mic_quality_label TEXT,
    mic_quality_snr_db REAL,
    planned_questions_json TEXT,
    created_at INTEGER NOT NULL,
    expires_at INTEGER NOT NULL,
    ended_at INTEGER,
    deleted_at INTEGER,
    FOREIGN KEY (client_id) REFERENCES coach_clients(id)
);

-- 008_coach_responses.sql
CREATE TABLE IF NOT EXISTS coach_responses (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    response_index INTEGER NOT NULL,
    question_text TEXT,
    question_index INTEGER,
    duration_s REAL NOT NULL,
    features_json TEXT NOT NULL,
    delta_pct_json TEXT NOT NULL,
    cartesian_x REAL,
    cartesian_y REAL,
    consistency_label TEXT NOT NULL,
    color TEXT NOT NULL,
    narrative TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (session_id) REFERENCES coach_sessions(id)
);

-- 009_coach_clients.sql  (Tier 2+ writes here)
CREATE TABLE IF NOT EXISTS coach_clients (
    id TEXT PRIMARY KEY,
    owner_user_id TEXT NOT NULL,
    client_name TEXT NOT NULL,
    process_reference TEXT,
    notes TEXT,
    created_at INTEGER NOT NULL,
    deleted_at INTEGER
);
CREATE INDEX idx_coach_clients_owner ON coach_clients(owner_user_id);

-- 010_coach_tags.sql  (Tier 2+)
CREATE TABLE IF NOT EXISTS coach_response_tags (
    response_id TEXT NOT NULL,
    tag TEXT NOT NULL,
    note TEXT,
    created_at INTEGER NOT NULL,
    PRIMARY KEY (response_id, tag),
    FOREIGN KEY (response_id) REFERENCES coach_responses(id)
);
CREATE INDEX idx_coach_tags_tag ON coach_response_tags(tag);

-- 011_coach_briefs.sql  (Tier 2+)
CREATE TABLE IF NOT EXISTS coach_briefs (
    id TEXT PRIMARY KEY,
    client_id TEXT NOT NULL,
    owner_user_id TEXT NOT NULL,
    sessions_analyzed INTEGER NOT NULL,
    summary_json TEXT NOT NULL,
    html_content TEXT,
    pdf_path TEXT,
    created_at INTEGER NOT NULL,
    FOREIGN KEY (client_id) REFERENCES coach_clients(id)
);
```

---

## 16 · Handoff Script · When Done

```
COACH v0.1.1 — READY FOR FIRST CLIENT

✓ All core + Cofre endpoints live
✓ State machine working
✓ MediaRecorder integration tested in Chrome + Firefox
✓ Real-time analysis p95 latency: [X]s (target < 4s)
✓ Per-session baseline immutable after calibration
✓ Audio NOT retained — verified (no orphan files)
✓ Tier feature gating works (Free / T1 / T2 / T3)
✓ Cofre persistence working (T2+)
✓ Trajectory + Diff + Pre-Hearing Brief generating correctly (T2+)
✓ PDF generation working for: Terms, Consent, Session Report, Brief
✓ Safari users see clean unsupported-browser message
✓ V1 and Academic endpoints unaffected (smoke tests passed)

URL: https://voxprobabilis.com/coach
Terms: https://voxprobabilis.com/coach/terms (HTML + PDF download)
Consent template: https://voxprobabilis.com/coach/consent-template (PDF)

Pricing live (with 36% annual discount on all paid tiers):
  FREE — 1 sessão única por conta
  Tier 1 Premium       — $36/mês  ($276.48/ano)  — 45 sessões + 15 relatórios Sonnet
  Tier 2 Profissional  — $77/mês  ($591.36/ano)  — ilimitado + 50 relatórios Opus + Cofre completo
  Tier 3 Escritório    — $141.90/mês ($1089.79/ano) — tudo ilimitado com Opus

Manual upgrade: `python -m app.coach.cli upgrade --user UID --tier TIER_2_MONTHLY`

Recommended first paying customer: Juan's lawyer friend.

Next sprint candidates: Lemon Squeezy, Safari fallback, Highlight Reel,
multi-session concurrent, mobile.
```
