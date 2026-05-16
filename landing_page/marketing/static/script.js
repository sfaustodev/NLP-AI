/* ================================================================
   VOX PROBABILIS — Site Script
   Tabs, pricing toggle, deep linking, reveal on scroll.
   ================================================================ */

(function () {
  "use strict";

  /* ---------- TABS ---------- */

  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".tab-panel");

  function activateTab(tabName, pushState = true) {
    tabs.forEach((t) => {
      const isActive = t.dataset.tab === tabName;
      t.setAttribute("aria-selected", isActive ? "true" : "false");
    });
    panels.forEach((p) => {
      p.classList.toggle("active", p.id === `tab-${tabName}`);
    });
    if (pushState && history.pushState) {
      const url = new URL(window.location.href);
      url.hash = tabName;
      history.pushState({ tab: tabName }, "", url);
    }
    // Scroll to top on tab change (UX nicer)
    window.scrollTo({ top: 0, behavior: "smooth" });
  }

  tabs.forEach((t) => {
    t.addEventListener("click", () => activateTab(t.dataset.tab));
    t.addEventListener("keydown", (e) => {
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        const tabsArr = Array.from(tabs);
        const idx = tabsArr.indexOf(t);
        const next = e.key === "ArrowRight"
          ? tabsArr[(idx + 1) % tabsArr.length]
          : tabsArr[(idx - 1 + tabsArr.length) % tabsArr.length];
        next.focus();
        activateTab(next.dataset.tab);
      }
    });
  });

  // Deep linking — abrir aba via #hash da URL
  function readHash() {
    const hash = window.location.hash.replace("#", "");
    if (["v1", "academic", "coach"].includes(hash)) {
      activateTab(hash, false);
    }
  }
  window.addEventListener("popstate", readHash);
  readHash();

  /* ---------- PRICING TOGGLE (Coach mensal / anual) ---------- */

  const toggles = document.querySelectorAll(".toggle-switch");

  toggles.forEach((toggle) => {
    const labels = toggle.parentElement.querySelectorAll(".pricing-toggle-label");

    function setActive(mode) {
      toggle.dataset.active = mode;
      labels.forEach((l) => l.classList.toggle("active", l.dataset.value === mode));

      // Atualizar preços e periodicidade nos cards do panel
      const panel = toggle.closest(".tab-panel");
      if (!panel) return;
      const cards = panel.querySelectorAll(".price-card[data-monthly]");
      cards.forEach((card) => {
        const monthly = parseFloat(card.dataset.monthly);
        const annualTotal = parseFloat(card.dataset.annual);
        const valueEl = card.querySelector(".value");
        const periodEl = card.querySelector(".price-period");
        if (!valueEl || !periodEl) return;

        if (mode === "annual" && !isNaN(annualTotal)) {
          const perMonth = (annualTotal / 12).toFixed(2);
          valueEl.textContent = perMonth;
          periodEl.innerHTML = `/mês · cobrado anualmente <span class="text-beige">${annualTotal.toFixed(2)}</span>`;
        } else if (!isNaN(monthly)) {
          valueEl.textContent = monthly.toFixed(2);
          periodEl.textContent = "/mês";
        }
      });
    }

    toggle.addEventListener("click", () => {
      const current = toggle.dataset.active || "monthly";
      setActive(current === "monthly" ? "annual" : "monthly");
    });

    labels.forEach((l) => {
      l.addEventListener("click", () => setActive(l.dataset.value));
    });
  });

  /* ---------- REVEAL ON SCROLL (sutil) ---------- */

  if ("IntersectionObserver" in window) {
    const reveals = document.querySelectorAll(".reveal");
    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (entry.isIntersecting) {
          entry.target.style.opacity = "1";
          entry.target.style.transform = "translateY(0)";
          observer.unobserve(entry.target);
        }
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -60px 0px" });

    reveals.forEach((el) => {
      el.style.opacity = "0";
      el.style.transform = "translateY(20px)";
      el.style.transition = "opacity 600ms ease, transform 600ms ease";
      observer.observe(el);
    });
  }

  /* ---------- LANG TOGGLE (placeholder — i18n vem depois) ---------- */

  const langToggle = document.querySelector(".lang-toggle");
  if (langToggle) {
    langToggle.addEventListener("click", () => {
      alert(
        "Internacionalização em construção. PT-BR fixo nesta versão. " +
        "Locale /en chegará em sprint dedicada — ver SPEC §13."
      );
    });
  }

  /* ---------- KEYBOARD: pressionar 1/2/3 troca aba (acessibilidade rápida) ---------- */

  document.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "TEXTAREA") return;
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === "1") activateTab("v1");
    if (e.key === "2") activateTab("academic");
    if (e.key === "3") activateTab("coach");
  });

})();
