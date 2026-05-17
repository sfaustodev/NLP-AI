/* Tiny cartesian plot widget — pure SVG, no deps.
 *
 * Exposes window.CoachCartesian.{init, addPoint, clear}.
 * Coordinates: x,y ∈ [-1, +1] → SVG 400×400 with 20px margin axes.
 *
 * Color tokens: GREEN/YELLOW/ORANGE/RED match consistency_label palette.
 */

(function () {
  "use strict";

  const W = 400;
  const H = 400;
  const M = 32;  // margin
  const PR = W / 2;  // projection radius

  const COLOR_MAP = {
    GREEN:  "#5C8C5A",
    YELLOW: "#C9A24A",
    ORANGE: "#D17C3F",
    RED:    "#B2231F",
    BASELINE: "#5C8C5A",
  };

  function svgNS(tag, attrs, parent) {
    const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
    if (attrs) {
      for (const k in attrs) {
        if (Object.prototype.hasOwnProperty.call(attrs, k)) {
          el.setAttribute(k, attrs[k]);
        }
      }
    }
    if (parent) { parent.appendChild(el); }
    return el;
  }

  function toSvgX(x) { return PR + x * (PR - M); }
  function toSvgY(y) { return PR - y * (PR - M); }   // flip — positive y is up

  function drawAxes(svg) {
    // Quadrant grid: subtle crosshair + outer border.
    svgNS("rect", {
      x: M / 2, y: M / 2, width: W - M, height: H - M,
      fill: "none", stroke: "#3C2F2C", "stroke-width": 1,
    }, svg);

    svgNS("line", {
      x1: PR, y1: M / 2, x2: PR, y2: H - M / 2,
      stroke: "#2A211F", "stroke-width": 1, "stroke-dasharray": "3,4",
    }, svg);
    svgNS("line", {
      x1: M / 2, y1: PR, x2: W - M / 2, y2: PR,
      stroke: "#2A211F", "stroke-width": 1, "stroke-dasharray": "3,4",
    }, svg);

    // Origin marker.
    svgNS("circle", {
      cx: PR, cy: PR, r: 3,
      fill: "none", stroke: "#7A6F6A", "stroke-width": 1,
    }, svg);

    // Quadrant labels (muted).
    const labels = [
      { x: PR + 60, y: M + 14, text: "NATURAL TENSA" },
      { x: M + 14, y: M + 14, text: "OVER-CONTROLLED TENSA", anchor: "start" },
      { x: M + 14, y: H - M / 2 - 6, text: "OVER-CONTROLLED CALMA", anchor: "start" },
      { x: PR + 60, y: H - M / 2 - 6, text: "NATURAL CALMA" },
    ];
    for (const l of labels) {
      svgNS("text", {
        x: l.x, y: l.y,
        "font-family": "JetBrains Mono, ui-monospace, monospace",
        "font-size": "8.5",
        fill: "#7A6F6A",
        "letter-spacing": "0.08em",
        "text-anchor": l.anchor || "start",
      }, svg).textContent = l.text;
    }
  }

  function init(svgEl) {
    while (svgEl.firstChild) { svgEl.removeChild(svgEl.firstChild); }
    svgEl.setAttribute("viewBox", `0 0 ${W} ${H}`);
    drawAxes(svgEl);
    // Layer group for points so they can be cleared without redrawing axes.
    const g = svgNS("g", { id: "cartesian-points" }, svgEl);
    return g;
  }

  function addPoint(svgEl, x, y, color, label) {
    let g = svgEl.querySelector("#cartesian-points");
    if (!g) { g = init(svgEl); }
    const fill = COLOR_MAP[color] || "#BFB4A4";
    const cx = toSvgX(x);
    const cy = toSvgY(y);
    const c = svgNS("circle", {
      cx: cx, cy: cy, r: 5,
      fill: fill,
      stroke: "#0A0808", "stroke-width": 1,
    }, g);
    if (label) {
      const t = svgNS("text", {
        x: cx + 8, y: cy + 3,
        "font-family": "JetBrains Mono, ui-monospace, monospace",
        "font-size": "9",
        fill: fill,
      }, g);
      t.textContent = label;
    }
  }

  function clear(svgEl) {
    const g = svgEl.querySelector("#cartesian-points");
    if (g) { while (g.firstChild) { g.removeChild(g.firstChild); } }
  }

  window.CoachCartesian = { init: init, addPoint: addPoint, clear: clear };
})();
