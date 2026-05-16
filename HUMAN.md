# HUMAN.md — perguntas pro humano

> Fila de decisões que o agente NÃO tem autoridade pra tomar sozinho.
> Filtro: irreversibilidade · contrato externo · segurança · conflito com SPRINT.md.
> Tom: criatividade encorajada, honestidade 100%, humildade granted.

---

## Open questions

> Q-01..Q-05 já foram movidas pra Resolved (probe SSH 2026-05-10). Restam apenas Q-06, Q-07, Q-08 (todas low-priority, herdadas de VOX-DEPLOY-A).

---

### Q-06 · Email forwarding `contact@voxprobabilis.com` · raised 2026-05-09 · context: VOX-DEPLOY-A

**Why I'm asking:** DEPLOY.md §3.1 + §9 (LGPD) + §10 (Terms) referenciam esse endereço. Se não configurar no Porkbun, o e-mail bate em vazio e o requisito LGPD vira só fachada (você não recebe acionamento de titular).

**Tentative:** recomendado fortemente. 2 min no Porkbun. Forward pro seu pessoal.

**Ask:** quer que eu te dê o passo-a-passo? Se sim, pode ir lá no Porkbun → voxprobabilis.com → Email Forwarding e cria `contact@` → seu email pessoal. Salva. Pronto.

---

### Q-08 · Mantive wiring inline em `index.html` em vez de extrair pra `static/wire.js` · raised 2026-05-09 · context: VOX-DEPLOY-A

**Why I'm asking:** DEPLOY.md §5.2 manda *literalmente* "implement this patch in a separate file `static/wire.js`". Mas o `index.html` que você já tinha tem `apiCalibrate`/`apiAnalyze` inline e o dispatcher `data-action`. A implementação que o DEPLOY descreve (`runStep(N)` + `runAnalysis()`) **não existe** — DEPLOY foi escrito antes do HTML maturar.

**Que fiz:** flipei `VOX_USE_MOCK = false` e deletei o bloco `window.VOX_MOCK` (~108 linhas). Resultado: backend real bate em `/api/calibrate` e `/api/analyze`, comportamento idêntico ao spec, diff mínimo.

**Tradeoff:**
- (a) Inline (atual): 1 arquivo HTML, fácil de inspecionar; cache-bust ao editar HTML invalida JS junto.
- (b) Extrair pra `static/wire.js` per literal §5.2: cache separado pro JS (HTML pode mudar sem invalidar JS no Cloudflare CDN), mas exige mover ~150 linhas (state, $/$$, go, painters, dispatcher) ou só a parte de `apiCalibrate`/`apiAnalyze` (split artificial).

**Tentative (já aplicado):** (a). Ganho de cache-isolation é teórico (HTML é dominado pelos painters, não pelo wiring; mudança no wiring é rara).

**Ask:** OK manter inline ou você quer §5.2 ao pé da letra? Se quiser (b), me diz e extraio.

---

### Q-07 · Smoke fixtures `audios_claude/{ai_truth,ai_uncertain,ai_lie}.wav` · raised 2026-05-09 · context: VOX-DEPLOY-A

**Why I'm asking:** DEPLOY.md §1 cita esses três WAVs como fixtures de regressão. Não achei em `backend/tests/` nem `landing_page/samples/`. Decide se smoke pós-deploy (§10 curl 6 e 7) usa eles ou usa as suas opus locais.

**Tentative:** smoke pós-deploy roda da sua máquina, com seus opus de calibração. Os WAVs são opcionais.

**Ask:** onde estão? você quer commitar pro repo (com risco de PII embora seja sua voz só) ou deixa local?

---

## Resolved

### Q-01..Q-05 · VPS probe (provider/OS/nginx/Rust/certbot) · raised 2026-05-09 · resolved 2026-05-10

**Asked:** dimensões de hardware, layout nginx, porta Rust, domínio existente, certbot.
**Answer (probe via SSH read-only):** Ubuntu 24.04 LTS Noble, Python 3.12, 3.8 GB RAM, 1 vCPU. nginx 1.24 com `appnda` único site (dormente). Rust em :3000 servido por Docker direto na :80, hostname `appnda.com`. certbot ausente.
**Followed-up in:** SPRINT.md §3, DIARY 2026-05-10, plan file.

### Q-09 · appnda nginx vhost (precisava resolver pra ligar nginx) · raised 2026-05-10 · resolved 2026-05-10

**Asked:** appnda nginx site dormente, mas `listen 80 default_server` choca com Docker. Manter, editar, ou remover symlink?
**Answer (from human):** "aplica a B" — remove symlink. Projeto appnda parado por ≥1 mês.
**Followed-up in:** `rm /etc/nginx/sites-enabled/appnda` no VPS, comentário em `nginx.conf` registrando o motivo.

### Q-10 · DNS records órfãos no Cloudflare · raised 2026-05-10 · resolved 2026-05-10

**Asked:** 3 A records pra `voxprobabilis.com` em vez de 1 — `89.116.73.118` (nosso VPS), `44.230.85.241` e `52.33.207.7` (AWS Oregon, = Linkly). CNAME wildcard `* → uixie.porkbun.com` (parking Porkbun).
**Why blocking:** Cloudflare round-robin entre os 3 IPs. POST multipart caía nos AWS → Linkly redirecionava pra `voxprobabilis-com.l.ink`. GET passava OK porque os AWS retornavam parking page com cache, não interceptavam.
**Answer (from human):** deletei AWS A records + CNAME wildcard.
**Followed-up in:** smoke §10 6/7/8 passaram após DNS cleanup. Lie sample landed em `OVER_CONTROLLED_TENSE`.

### Q-00 · Liveness mode A/B/C · raised 2026-05-09 · resolved 2026-05-09

**Asked:** DEPLOY §4 — TTS_DISCOVERY achou microtremor ~zero em TTS, real second capability, mas n=3-vs-n=3 fino. Pick: A (off) / B (boolean) / C (full).
**Answer (from human):** A — document only.
**Followed-up in:** SPRINT §6 (out of scope v0.2), `LIVENESS_MODE = "off"` em `backend/app/__init__.py`, `VOX_LIVENESS_MODE=off` em `.env.example`. Frontend sem badge.

### Q-00b · Bootstrap discipline files · raised 2026-05-09 · resolved 2026-05-09

**Asked:** SPRINT/DIARY/JIRA/HUMAN não existiam. Bootstrap?
**Answer (from human):** Bootstrap.
**Followed-up in:** estes 4 arquivos.

### Q-00c · Phase A vs Phase B parallel · raised 2026-05-09 · resolved 2026-05-09

**Asked:** Local code (Phase A) em paralelo com sua prep Cloudflare/VPS (Phase B), ou serial?
**Answer (from human):** Parallel.
**Followed-up in:** SPRINT §1.
