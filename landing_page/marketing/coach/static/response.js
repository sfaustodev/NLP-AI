/* Per-response detail page.
 *
 * URL pattern: /coach/session/<token>/response/<id>
 * Reads token + id from the path, GETs /api/coach/session/<token>/response/<id>,
 * renders question, transcription (when phase C lands), narrative + cartesian
 * point + raw features for debugging.
 *
 * All dynamic strings use textContent — never innerHTML on server data.
 */

(function () {
  "use strict";

  const $ = (s) => document.querySelector(s);

  const parts = window.location.pathname.split("/").filter(Boolean);
  // Expected: ["coach", "session", "<token>", "response", "<id>"]
  if (parts.length < 5 || parts[0] !== "coach" || parts[1] !== "session"
      || parts[3] !== "response") {
    $("#sub").textContent = "URL inválida.";
    return;
  }
  const sessionToken = parts[2];
  const responseId   = parts[4];

  $("#back-link").href = "/coach/session/" + encodeURIComponent(sessionToken);

  function showErr(text) {
    const el = $("#err");
    el.textContent = text;
    el.classList.remove("hide");
  }

  fetch("/api/coach/session/" + encodeURIComponent(sessionToken)
        + "/response/" + encodeURIComponent(responseId),
        { credentials: "same-origin" })
    .then((r) => {
      if (!r.ok) {
        return r.json().catch(() => ({})).then((body) => {
          throw new Error((body.error && body.error.message) || ("HTTP " + r.status));
        });
      }
      return r.json();
    })
    .then((d) => {
      $("#title").textContent = "Resposta #" + (d.response_index || "?");
      $("#sub").textContent = d.session_name
        ? ("Sessão: " + d.session_name + " · " + (d.duration_s || 0).toFixed(1) + "s")
        : ((d.duration_s || 0).toFixed(1) + "s");
      $("#question-text").textContent = d.question_text || "(sem pergunta registrada)";

      if (d.transcription) {
        $("#transcription-text").textContent = d.transcription;
        $("#transcription-text").classList.remove("muted");
      }

      const dot = $("#color-dot");
      dot.className = "mic-dot " + (d.color || "GREEN");
      $("#consistency-label").textContent = d.consistency_label || "BASELINE";
      $("#narrative-text").textContent = d.narrative || "(sem leitura prática gerada)";

      // Cartesian: init widget + plot just this point.
      window.CoachCartesian.init($("#cartesian-svg"));
      window.CoachCartesian.addPoint(
        $("#cartesian-svg"), d.cartesian.x, d.cartesian.y,
        d.color, "#" + d.response_index,
      );

      // Raw features for debug.
      const featLines = Object.keys(d.features || {}).sort().map((k) =>
        k.padEnd(28) + " " + d.features[k]
      ).concat(
        ["", "deltas vs baseline (%):"],
        Object.keys(d.delta_pct || {}).sort().map((k) =>
          "  " + k.padEnd(26) + " " + (d.delta_pct[k] > 0 ? "+" : "") + d.delta_pct[k]
        ),
      );
      $("#features-pre").textContent = featLines.join("\n");
    })
    .catch((e) => {
      showErr("Falha ao carregar resposta: " + e.message);
    });
})();
