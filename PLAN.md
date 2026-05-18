# PLAN.md — active sprint plan log

> 5º sacred file. Append-only durante sprint. AI-readable, terse.
> Rotaciona em fim de sprint → `PLAN_SPRINT_<id>.md`.
> Rules específicas SPRINT only. Rules globais → CLAUDE.md.

---

## 2026-05-17 — fix/csp-cloudflare-beacon — VOX-CSP-FIX

**Contexto:** Sub-ticket isolado descoberto durante smoke VOX-LANDING-A (2026-05-16 22:55 UTC, Playwright). Cloudflare Web Analytics beacon `static.cloudflareinsights.com/beacon.min.js` bloqueado por CSP `script-src 'self' 'unsafe-inline'` do v0.1. Web Analytics ativado no dashboard CF nunca dispara. NÃO bloqueia VOX-COACH-B em flight.

**Escopo:**
- Edit `backend/deploy/nginx.conf` linha 96 CSP header.
- Add `https://static.cloudflareinsights.com` ao `script-src` (carrega beacon.min.js).
- Defensive: add `media-src 'self' blob:` pra futuro preview de gravação Coach (recorder.js usa MediaRecorder; hoje só FormData upload, mas blob playback é cenário plausível v0.2).
- Hardening grátis (Codex P2): add `object-src 'none'; base-uri 'self'` (base-uri não herda de default-src; sem isso `<base>` injetado reescreve URLs relativos).
- Codex P2 #1 revisto: `connect-src cloudflareinsights.com` NÃO é necessário em proxied (orange-cloud) mode — beacon POSTa same-origin `/cdn-cgi/rum`. Removido da CSP. Só seria preciso em setup non-proxied.

**Arquivos críticos:**
- `backend/deploy/nginx.conf` (single-line edit linha 96)
- VPS prod `/etc/nginx/sites-available/voxprobabilis` (cp + reload pós-PR merge)

**Branch base:** `origin/master` (5d7c5af) — NÃO branch atual `c/adoring-darwin-c2011d` que tem commit WEBM (`e2fbf1a`) pendente VOX-COACH-B fora master.

**Verificação:**
1. `nginx -t` syntax check pós cp servidor.
2. `curl -sI https://voxprobabilis.com/ | grep -i content-security` → header com 3 novos itens.
3. Playwright console verify zero CSP violations.
4. Cloudflare dashboard Analytics começa registrar tráfego ~5 min após reload.

**Pre-merge defenses (rule #20):**
- pre-merge-coverage skip (config nginx, sem code path testável unit).
- `codex-cross-review` no PR diff antes merge.

**Súplica prod:** rule #16D formato obrigatório antes SSH/reload nginx prod.

**Next single-step:** checkout branch nova de origin/master, edit nginx.conf, commit atômico.

**Status final (2026-05-18 00:58 UTC):** ✅ LIVE prod · PR #5 merged master `faa9779` · deploy SHA `8479bf3` · backup `.bak.20260518-005717` · smoke 7/7 verde · beacon POST /cdn-cgi/rum 204 ✓. Aguarda Faustão confirmar Cloudflare dashboard Analytics tráfego (~5 min).

---

## 2026-05-18 — docs/csp-fix-closeout — VOX-CSP-FIX close-out

**Contexto:** Close-out sacred files pós VOX-CSP-FIX deploy verde. Trivial doc-only branch.

**Escopo:**
- Update JIRA.md: add VOX-CSP-FIX block Active (status Done · deploy details · smoke results)
- Update DIARY.md: prepend entry 2026-05-18
- Update PLAN.md: status final + new entry close-out
- Bundle `.gitignore` mod (+`reports_fausto/`) que veio da skill codex-cross-review

**Verificação:** PR merge fast (sem CI tests pra docs).

---
