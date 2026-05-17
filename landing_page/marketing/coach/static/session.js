/* Coach session live-view orchestrator.
 *
 * State machine mirror (server is authoritative):
 *   loading
 *     → unsupported       (no MediaRecorder)
 *     → permission        (mic not granted)
 *     → calibrate         (CREATED, server state)
 *     → practice          (READY / IN_PRACTICE)
 *     → ended             (ENDED → render report)
 *
 * Polls GET /api/coach/session/{token} every 2s while the user is on a
 * non-terminal state. Stops when ENDED.
 *
 * Security note: the Sonnet-generated report.html is rendered inside a
 * sandboxed iframe (srcdoc + sandbox="") so any LLM-injected script
 * cannot execute. All other dynamic text uses textContent.
 */

(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);

  // -------- token from URL path --------
  const pathParts = window.location.pathname.split("/").filter(Boolean);
  const token = pathParts[pathParts.length - 1];

  const panels = {
    permission: $("#permission-panel"),
    calibrate:  $("#calibrate-panel"),
    practice:   $("#practice-panel"),
    report:     $("#report-panel"),
    unsupported: $("#unsupported"),
    header:     $("#header"),
    main:       $("#main-panel"),
  };
  const els = {
    sessionTitle:    $("#session-title"),
    sessionSub:      $("#session-sub"),
    sessionBadge:    $("#session-name-badge"),
    grantMicBtn:     $("#grant-mic-btn"),
    micError:        $("#mic-error"),
    calibrateRecordBtn: $("#calibrate-record-btn"),
    calibrateStopBtn:   $("#calibrate-stop-btn"),
    calibrateStatus:    $("#calibrate-status"),
    calibrateError:     $("#calibrate-error"),
    micDot:        $("#mic-dot"),
    micLabel:      $("#mic-label"),
    micSnr:        $("#mic-snr"),
    questionText:  $("#question-text"),
    responseRecordBtn: $("#response-record-btn"),
    responseStopBtn:   $("#response-stop-btn"),
    responseStatus:    $("#response-status"),
    responseError:     $("#response-error"),
    endBtn:        $("#end-btn"),
    cartesianSvg:  $("#cartesian-svg"),
    history:       $("#history"),
    reportHtml:    $("#report-html"),
    reportPdfLink: $("#report-pdf-link"),
  };

  // Track responses we've already rendered to avoid duplicating on poll.
  let renderedResponseIds = new Set();
  let pollTimer = null;
  let micGranted = false;
  let currentState = null;

  function showPanel(name) {
    Object.keys(panels).forEach((k) => {
      if (k === "header" || k === "main" || k === "unsupported") { return; }
      panels[k].classList.add("hide");
    });
    if (panels[name]) { panels[name].classList.remove("hide"); }
  }

  function showBanner(el, text, cls) {
    el.textContent = text;
    el.classList.remove("hide");
    el.classList.add(cls || "warning");
  }
  function hideBanner(el) {
    el.classList.add("hide");
    el.classList.remove("warning", "error", "success");
  }

  // -------- unsupported browser splash --------
  if (!window.CoachRecorder || !window.CoachRecorder.isSupported()) {
    panels.unsupported.classList.remove("hide");
    panels.header.classList.add("hide");
    panels.main.classList.add("hide");
    return;
  }

  // -------- cartesian init --------
  window.CoachCartesian.init(els.cartesianSvg);

  // -------- poll session state --------
  async function pollState() {
    try {
      const r = await fetch("/api/coach/session/" + encodeURIComponent(token),
                             { credentials: "same-origin" });
      if (r.status === 404 || r.status === 410 || r.status === 401) {
        els.sessionSub.textContent = "Sessão indisponível (404/410/401).";
        return;
      }
      if (!r.ok) {
        els.sessionSub.textContent = "Erro HTTP " + r.status + " ao consultar sessão.";
        return;
      }
      const body = await r.json();
      onState(body);
    } catch (e) {
      els.sessionSub.textContent = "Falha de rede ao consultar sessão.";
    }
  }

  function onState(s) {
    els.sessionTitle.textContent = s.session_name || "Sessão";
    els.sessionBadge.textContent = s.tier_label || s.state;
    els.sessionSub.textContent = "Estado: " + s.state;
    currentState = s.state;

    // Render mic quality if known.
    if (s.mic_quality_label) {
      els.micDot.className = "mic-dot " + s.mic_quality_label;
      els.micLabel.textContent = s.mic_quality_label;
      if (typeof s.mic_quality_snr_db === "number") {
        els.micSnr.textContent = "· SNR " + s.mic_quality_snr_db.toFixed(1) + " dB";
      }
    }

    // Render new responses.
    if (Array.isArray(s.responses)) {
      s.responses.forEach((resp) => {
        if (renderedResponseIds.has(resp.id)) { return; }
        renderedResponseIds.add(resp.id);
        window.CoachCartesian.addPoint(
          els.cartesianSvg, resp.cartesian_x, resp.cartesian_y,
          resp.color, "#" + (resp.response_index || ""),
        );
        appendHistoryRow(resp);
      });
    }

    // Panel decisions.
    if (s.state === "ENDED") {
      stopPolling();
      renderReport(s);
      showPanel("report");
      return;
    }
    if (s.state === "CREATED") {
      if (!micGranted) { showPanel("permission"); return; }
      showPanel("calibrate");
      return;
    }
    if (s.state === "READY" || s.state === "IN_PRACTICE") {
      if (!micGranted) { showPanel("permission"); return; }
      showPanel("practice");
      return;
    }
  }

  function clearChildren(node) {
    while (node.firstChild) { node.removeChild(node.firstChild); }
  }

  function appendHistoryRow(resp) {
    // Drop the initial placeholder if still present.
    const placeholder = els.history.querySelector(".muted");
    if (placeholder) { clearChildren(els.history); }

    const row = document.createElement("div");
    row.className = "history-row";

    const dot = document.createElement("span");
    dot.className = "history-dot " + (resp.color || "GREEN");
    row.appendChild(dot);

    const text = document.createElement("div");
    text.className = "history-text";

    const label = document.createElement("span");
    label.className = "label";
    label.textContent = "Resposta #" + (resp.response_index || "?") +
                          " · " + (resp.consistency_label || "BASELINE");
    text.appendChild(label);

    const body = document.createElement("span");
    body.textContent = resp.narrative || "(sem narrativa)";
    text.appendChild(body);

    row.appendChild(text);
    els.history.appendChild(row);
  }

  function renderReport(s) {
    els.reportPdfLink.href =
      "/api/coach/session/" + encodeURIComponent(token) + "/report.pdf";

    fetch("/api/coach/session/" + encodeURIComponent(token) + "/report.html",
          { credentials: "same-origin" })
      .then((r) => r.ok ? r.text() : Promise.reject("HTTP " + r.status))
      .then((html) => {
        // Render the report inside a sandboxed iframe so any LLM-injected
        // script or untrusted markup cannot touch this page's DOM, cookies,
        // or storage. sandbox="" = no scripts, no forms, no same-origin.
        clearChildren(els.reportHtml);
        const iframe = document.createElement("iframe");
        iframe.setAttribute("sandbox", "");
        iframe.setAttribute("srcdoc", html);
        iframe.style.width = "100%";
        iframe.style.minHeight = "420px";
        iframe.style.border = "0";
        iframe.style.background = "var(--bone)";
        els.reportHtml.appendChild(iframe);
      })
      .catch((e) => {
        els.reportHtml.textContent = "Erro ao carregar relatório: " + e;
      });
  }

  function startPolling(intervalMs) {
    stopPolling();
    pollTimer = setInterval(pollState, intervalMs || 2000);
  }
  function stopPolling() {
    if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
  }

  // -------- mic permission flow --------
  els.grantMicBtn.addEventListener("click", async () => {
    hideBanner(els.micError);
    try {
      await window.CoachRecorder.requestMic();
      micGranted = true;
      pollState();   // re-render with appropriate panel
    } catch (e) {
      showBanner(els.micError,
        "Falha ao acessar microfone: " + (e.message || e) +
        ". Verifique permissão do navegador.", "error");
    }
  });

  // -------- calibrate flow --------
  els.calibrateRecordBtn.addEventListener("click", async () => {
    hideBanner(els.calibrateError);
    try {
      window.CoachRecorder.startRecording({ timeoutMs: 12000 });
      els.calibrateRecordBtn.classList.add("hide");
      els.calibrateStopBtn.classList.remove("hide");
      els.calibrateStatus.textContent = "Gravando… (máx. 12s)";
    } catch (e) {
      showBanner(els.calibrateError, "Falha ao gravar: " + e.message, "error");
    }
  });
  els.calibrateStopBtn.addEventListener("click", async () => {
    els.calibrateStopBtn.classList.add("hide");
    els.calibrateStatus.textContent = "Enviando calibração…";
    try {
      const blob = await window.CoachRecorder.stopRecording();
      const fd = new FormData();
      fd.append("audio", blob, "calibrate.webm");
      const r = await fetch(
        "/api/coach/session/" + encodeURIComponent(token) + "/calibrate",
        { method: "POST", credentials: "same-origin", body: fd },
      );
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error((body.error && body.error.message) || ("HTTP " + r.status));
      }
      els.calibrateStatus.textContent = "Calibração ok. Avançando…";
      pollState();
    } catch (e) {
      showBanner(els.calibrateError, "Falha na calibração: " + e.message, "error");
      els.calibrateRecordBtn.classList.remove("hide");
    }
  });

  // -------- response flow --------
  els.responseRecordBtn.addEventListener("click", () => {
    hideBanner(els.responseError);
    try {
      window.CoachRecorder.startRecording({ timeoutMs: 30000 });
      els.responseRecordBtn.classList.add("hide");
      els.responseStopBtn.classList.remove("hide");
      els.responseStatus.textContent = "Gravando resposta… (máx. 30s)";
    } catch (e) {
      showBanner(els.responseError, "Falha ao gravar: " + e.message, "error");
    }
  });
  els.responseStopBtn.addEventListener("click", async () => {
    els.responseStopBtn.classList.add("hide");
    els.responseStatus.textContent = "Enviando resposta…";
    try {
      const blob = await window.CoachRecorder.stopRecording();
      const fd = new FormData();
      fd.append("audio", blob, "response.webm");
      const qt = els.questionText.value.trim();
      if (qt) { fd.append("question_text", qt); }
      const r = await fetch(
        "/api/coach/session/" + encodeURIComponent(token) + "/response",
        { method: "POST", credentials: "same-origin", body: fd },
      );
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error((body.error && body.error.message) || ("HTTP " + r.status));
      }
      els.responseStatus.textContent = "Resposta processada.";
      els.responseRecordBtn.classList.remove("hide");
      els.questionText.value = "";
      pollState();
    } catch (e) {
      showBanner(els.responseError, "Falha na resposta: " + e.message, "error");
      els.responseRecordBtn.classList.remove("hide");
    }
  });

  // -------- end flow --------
  els.endBtn.addEventListener("click", async () => {
    if (!confirm("Encerrar sessão? Não dá pra gravar mais respostas depois.")) {
      return;
    }
    els.endBtn.disabled = true;
    try {
      const r = await fetch(
        "/api/coach/session/" + encodeURIComponent(token) + "/end",
        { method: "POST", credentials: "same-origin" },
      );
      if (!r.ok) {
        const body = await r.json().catch(() => ({}));
        throw new Error((body.error && body.error.message) || ("HTTP " + r.status));
      }
      window.CoachRecorder.releaseMic();
      pollState();
    } catch (e) {
      showBanner(els.responseError, "Falha ao encerrar: " + e.message, "error");
      els.endBtn.disabled = false;
    }
  });

  // -------- bootstrap --------
  pollState();
  startPolling(2000);
})();
