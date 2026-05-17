/* Coach dashboard — /coach lawyer landing.
 *
 * Loads quota + tier on mount, handles "Nova sessão" form, redirects to
 * session view on success. All endpoints under /api/coach/*. Cookie
 * auth (coach_session) is automatic via fetch credentials:'same-origin'.
 *
 * Security note: no innerHTML on user-controlled or server-derived data.
 * All dynamic text goes through textContent / explicit DOM construction.
 */

(function () {
  "use strict";

  const $ = (sel) => document.querySelector(sel);

  const tierBadge = $("#tier-badge");
  const quotaLine = $("#quota-line");
  const form      = $("#new-session-form");
  const sessionNameInput = $("#session-name");
  const plannedInput = $("#planned-questions");
  const createBtn = $("#create-btn");
  const errBanner = $("#new-session-error");
  const msgBanner = $("#new-session-msg");

  function show(el, text, cls) {
    el.textContent = text;
    el.classList.remove("hide");
    if (cls) { el.classList.add(cls); }
  }
  function hide(el) {
    el.classList.add("hide");
    el.textContent = "";
  }

  async function fetchQuota() {
    try {
      const r = await fetch("/api/coach/quota", { credentials: "same-origin" });
      if (r.status === 401) {
        tierBadge.textContent = "não ativado";
        quotaLine.textContent = "Sua conta ainda não foi ativada. Peça o link de ativação ao operador.";
        createBtn.disabled = true;
        return null;
      }
      if (!r.ok) {
        tierBadge.textContent = "erro";
        quotaLine.textContent = "Erro ao carregar quota: HTTP " + r.status;
        return null;
      }
      const body = await r.json();
      renderTier(body);
      return body;
    } catch (e) {
      tierBadge.textContent = "offline";
      quotaLine.textContent = "Falha de rede ao consultar quota.";
      return null;
    }
  }

  function renderTier(q) {
    tierBadge.textContent = q.tier_label || q.tier_key;
    tierBadge.classList.remove("tier-trial", "tier-t1", "tier-t2", "tier-t3");
    const map = {
      FREE_TRIAL: "tier-trial",
      TIER_1_MONTHLY: "tier-t1",
      TIER_2_MONTHLY: "tier-t2",
      TIER_3_MONTHLY: "tier-t3",
    };
    tierBadge.classList.add(map[q.tier_key] || "tier-trial");

    const sessLeft = q.sessions_remaining === -1 ? "ilimitadas" :
                       q.sessions_remaining + "/" + q.sessions_limit;
    const repLeft  = q.reports_remaining === -1 ? "ilimitados" :
                       q.reports_remaining + "/" + q.reports_limit;

    // Build DOM nodes explicitly — never use innerHTML with server data.
    quotaLine.textContent = "";
    quotaLine.appendChild(document.createTextNode("Sessões restantes: "));
    const sessStrong = document.createElement("strong");
    sessStrong.textContent = sessLeft;
    quotaLine.appendChild(sessStrong);
    quotaLine.appendChild(document.createTextNode(" · Relatórios restantes: "));
    const repStrong = document.createElement("strong");
    repStrong.textContent = repLeft;
    quotaLine.appendChild(repStrong);
  }

  form.addEventListener("submit", async (ev) => {
    ev.preventDefault();
    hide(errBanner); hide(msgBanner);

    const name = sessionNameInput.value.trim();
    if (!name) {
      show(errBanner, "Dê um nome à sessão.", "warning");
      return;
    }
    const planned = plannedInput.value
      .split("\n").map((s) => s.trim()).filter(Boolean);

    createBtn.disabled = true;
    createBtn.textContent = "Criando…";
    try {
      const r = await fetch("/api/coach/session/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        credentials: "same-origin",
        body: JSON.stringify({
          session_name: name,
          planned_questions: planned,
        }),
      });
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        const code = body.error && body.error.code;
        const msg  = (body.error && body.error.message) || ("HTTP " + r.status);
        show(errBanner,
          code === "COACH_QUOTA_EXCEEDED"
            ? "Quota de sessões esgotada neste período. " + msg
            : "Erro ao criar sessão: " + msg,
          "error");
        return;
      }
      const body = await r.json();
      show(msgBanner,
        'Sessão criada. Abrindo "' + name + '" agora…',
        "success");
      // Redirect after a short beat so the user sees the confirmation.
      setTimeout(() => {
        window.location.href = "/coach/session/" + encodeURIComponent(body.session_token);
      }, 700);
    } catch (e) {
      show(errBanner, "Falha de rede: " + e.message, "error");
    } finally {
      createBtn.disabled = false;
      createBtn.textContent = "Criar sessão";
    }
  });

  // Bootstrap.
  fetchQuota();
})();
