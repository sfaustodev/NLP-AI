# HUMAN.md — perguntas pro humano

> Fila de decisões que o agente NÃO tem autoridade pra tomar sozinho.
> Filtro: irreversibilidade · contrato externo · segurança · conflito com SPRINT.md.
> Tom: criatividade encorajada, honestidade 100%, humildade granted.

---

## Open questions

> Todas resolvidas 2026-05-16 com inicio VOX-COACH-B. Fila vazia.

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

### Q-06 · Email forwarding `contact@voxprobabilis.com` · raised 2026-05-09 · resolved 2026-05-16

**Asked:** Configurar forwarding no Porkbun pra LGPD não virar fachada?
**Answer (from human + dig verify):** Faustão configurou no Porkbun. `dig +short MX voxprobabilis.com` confirma: `10 fwd1.porkbun.com / 20 fwd2.porkbun.com` + SPF record `v=spf1 include:_spf.porkbun.com ~all`. Email forwarding ATIVO.
**Followed-up in:** sem mudança código — verificação via dig basta. Endereço `contact@voxprobabilis.com` agora aceita email LGPD.

### Q-07 · Smoke fixtures `audios_claude/{ai_truth,ai_uncertain,ai_lie}.wav` · raised 2026-05-09 · resolved 2026-05-16

**Asked:** Onde estão? commit pro repo ou deixa local?
**Answer (from human + local find):** Faustão disse "estão na pasta NLP-AI". `find` confirma: `landing_page/samples/audios_claude/{ai_truth,ai_lie,ai_uncertain}.wav` existem. Ainda untracked no git (privacidade).
**Followed-up in:** path documentado. Coach (VOX-COACH-B) usa esses fixtures pros tests de smoke flow (calibrate→response). Mantém untracked por ora (risco PII embora seja voz Faustão only).

### Q-08 · wire.js inline vs extract · raised 2026-05-09 · resolved 2026-05-16

**Asked:** OK manter inline ou extrair pra static/wire.js per literal DEPLOY §5.2?
**Answer (from human):** "cara nem sei o que é isso confio em vc pra decidir".
**Decision (agent):** Manter inline (já é o estado atual). Ganho cache-isolation teórico não justifica refactor ~150 LOC + risco regressão. Se mudar wire em sprint futuro + cache realmente perceptível, considera split então.
**Followed-up in:** sem mudança código — `landing_page/index.html` v0.1 mantém wiring inline.
