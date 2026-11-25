/* ChurnGuard motion layer: purely visual, reads the DOM that app.js renders and never changes data.
   Remove the <script src="/static/motion.js"> tag to switch it all off. */
(() => {
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (REDUCED) return;
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const SPRING = "cubic-bezier(.34,1.56,.64,1)", OUT = "cubic-bezier(.16,1,.3,1)";
  const COLORS = { high: "#dc2626", medium: "#f59e0b", low: "#16a34a" };

  /* ---------- 1. page entrance with staggered children ---------- */
  function stagger(root) {
    const kids = [...root.children, ...$$(".grid > *", root)];
    kids.forEach((el, i) => { el.classList.remove("cg-rise"); el.style.setProperty("--i", Math.min(i, 12)); });
    void root.offsetWidth;
    kids.forEach((el) => el.classList.add("cg-rise"));
  }
  function onRoute() {
    const page = $(".page.active");
    if (page) stagger(page);
    moveBlob();
  }

  /* ---------- 2. liquid nav indicator ---------- */
  const nav = $(".sidebar nav");
  const blob = Object.assign(document.createElement("div"), { className: "cg-blob" });
  nav.prepend(blob);
  let blobTop = null;
  function moveBlob() {
    const a = $(".sidebar nav a.active");
    if (!a) { blob.style.opacity = 0; return; }
    blob.style.opacity = 1;
    const top = a.offsetTop, h = a.offsetHeight;
    if (blobTop === null || matchMedia("(max-width: 760px)").matches) {
      Object.assign(blob.style, { top: `${top}px`, height: `${h}px`, left: `${a.offsetLeft}px`, width: `${a.offsetWidth}px` });
    } else {
      // phase 1: stretch over both items, phase 2: snap to the target (a liquid "jump")
      const from = blobTop, lo = Math.min(from, top), hi = Math.max(from, top) + h;
      Object.assign(blob.style, { top: `${lo}px`, height: `${hi - lo}px` });
      setTimeout(() => Object.assign(blob.style, { top: `${top}px`, height: `${h}px` }), 170);
    }
    blobTop = top;
  }
  window.addEventListener("hashchange", () => requestAnimationFrame(onRoute));
  window.addEventListener("resize", () => { blobTop = null; moveBlob(); });

  /* ---------- 3. cursor spotlight ---------- */
  document.addEventListener("pointermove", (e) => {
    const c = e.target.closest?.(".card");
    if (!c) return;
    const r = c.getBoundingClientRect();
    c.style.setProperty("--mx", `${e.clientX - r.left}px`);
    c.style.setProperty("--my", `${e.clientY - r.top}px`);
  });

  /* ---------- 4. odometer numbers ---------- */
  function odometer(el) {
    if (el.dataset.cg || el.children.length) return;
    const text = el.textContent.trim();
    if (!/\d/.test(text) || /[:\-]/.test(text)) return;
    el.dataset.cg = 1;
    el.setAttribute("aria-label", text);
    el.innerHTML = `<span class="cg-odo" aria-hidden="true">${[...text].map((ch) => /\d/.test(ch)
      ? `<span class="cg-col"><span data-d="${ch}">${"0123456789".split("").map((d) => `<b>${d}</b>`).join("")}</span></span>`
      : `<span>${ch === " " ? "&nbsp;" : ch}</span>`).join("")}</span>`;
    const cols = $$("[data-d]", el);
    requestAnimationFrame(() => requestAnimationFrame(() => cols.forEach((c, i) => {
      c.style.transitionDelay = `${i * 70}ms`;
      c.style.transform = `translateY(-${c.dataset.d * 1.18}em)`;
    })));
  }

  /* ---------- bars grow from zero ---------- */
  function grow(el, prop, i = 0) {
    if (el.dataset.cg) return;
    el.dataset.cg = 1;
    const target = el.style[prop];
    if (!target) return;
    el.animate([{ [prop]: prop === "height" ? "3px" : "0%" }, { [prop]: target }], { duration: 900, delay: 120 + i * 60, easing: OUT, fill: "backwards" });
  }

  /* ---------- 5. customer constellation (overview) ---------- */
  const head = $("#page-overview .page-head");
  const cv = Object.assign(document.createElement("canvas"), { className: "cg-orbit", title: "Each dot is a customer. Red dots are drifting away; hover to rescue them." });
  const legend = Object.assign(document.createElement("div"), { className: "cg-orbit-legend", innerHTML: '<span style="color:#4f46e5">●</span> retained &nbsp;<span style="color:#dc2626">●</span> at risk · hover to rescue' });
  head.append(cv, legend);
  const ctx = cv.getContext("2d");
  let churnShare = 0.26, mouse = { x: -999, y: -999 }, rescued = 0;
  const N = 110, P = [];
  const spawn = (p = {}) => Object.assign(p, { a: Math.random() * 6.283, r: 0.25 + Math.random() * 0.55, sp: (0.0016 + Math.random() * 0.003) * (Math.random() < .5 ? 1 : -1), risk: false, out: 0, life: 1, flash: 0 });
  for (let i = 0; i < N; i++) P.push(spawn());
  cv.addEventListener("pointermove", (e) => { const r = cv.getBoundingClientRect(); mouse = { x: e.clientX - r.left, y: e.clientY - r.top }; });
  cv.addEventListener("pointerleave", () => (mouse = { x: -999, y: -999 }));
  function frame() {
    requestAnimationFrame(frame);
    if (!cv.offsetParent) return;
    const dpr = devicePixelRatio || 1, w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== w * dpr) { cv.width = w * dpr; cv.height = h * dpr; }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const cx = w * 0.55, cy = h / 2, rx = w * 0.42, ry = h * 0.46;
    // core
    const g = ctx.createRadialGradient(cx, cy, 0, cx, cy, 46);
    g.addColorStop(0, "rgba(79,70,229,.25)"); g.addColorStop(1, "rgba(79,70,229,0)");
    ctx.fillStyle = g; ctx.beginPath(); ctx.arc(cx, cy, 46, 0, 6.283); ctx.fill();
    const atRisk = P.filter((p) => p.risk).length;
    if (atRisk < churnShare * N * 0.35 && Math.random() < 0.05) { const c = P[(Math.random() * N) | 0]; if (!c.risk) c.risk = true; }
    for (const p of P) {
      p.a += p.sp;
      if (p.risk) { p.out += 0.0022; if (p.out > 0.55) { p.life -= 0.02; if (p.life <= 0) spawn(p); } }
      else p.out = Math.max(0, p.out - 0.01);
      const rr = p.r + p.out;
      const x = cx + Math.cos(p.a) * rx * rr, y = cy + Math.sin(p.a) * ry * rr;
      if (p.risk && Math.hypot(x - mouse.x, y - mouse.y) < 42) { p.risk = false; p.life = 1; p.flash = 1; rescued++; }
      const col = p.risk ? `rgba(220,38,38,${0.9 * p.life})` : "rgba(79,70,229,.75)";
      ctx.fillStyle = col; ctx.beginPath(); ctx.arc(x, y, p.risk ? 2.6 : 2.1, 0, 6.283); ctx.fill();
      if (p.flash > 0) { ctx.strokeStyle = `rgba(22,163,74,${p.flash})`; ctx.lineWidth = 1.5; ctx.beginPath(); ctx.arc(x, y, 4 + (1 - p.flash) * 16, 0, 6.283); ctx.stroke(); p.flash -= 0.03; }
    }
    if (rescued) { ctx.fillStyle = "#16a34a"; ctx.font = "600 11px Inter, sans-serif"; ctx.fillText(`${rescued} rescued`, cx - 24, cy + 4); }
  }
  requestAnimationFrame(frame);
  function readChurnShare() {
    const k = $$("#kpis .kpi").find((c) => /high-risk share/i.test(c.textContent));
    const v = parseFloat(k?.querySelector(".value")?.getAttribute("aria-label") || k?.querySelector(".value")?.textContent);
    if (!isNaN(v) && v > 0) churnShare = Math.min(0.9, v / 100);
  }

  /* ---------- 6. gauge sweep, heartbeat, what-if ---------- */
  let lastGauge = {};
  function celebrate(anchor) {
    const r = anchor.getBoundingClientRect(), cx = r.left + r.width / 2, cy = r.top + r.height * 0.6;
    for (let i = 0; i < 46; i++) {
      const d = document.createElement("i"), ang = Math.random() * Math.PI * 2, dist = 70 + Math.random() * 140;
      d.className = "cg-confetti";
      d.style.cssText = `left:${cx}px;top:${cy}px;background:${["#16a34a", "#4f46e5", "#86efac", "#818cf8", "#f59e0b"][i % 5]}`;
      document.body.append(d);
      d.animate([{ transform: "translate(-50%,-50%) scale(1)", opacity: 1 },
        { transform: `translate(${Math.cos(ang) * dist}px, ${Math.sin(ang) * dist + 80}px) rotate(${Math.random() * 720}deg) scale(.6)`, opacity: 0 }],
        { duration: 1100 + Math.random() * 500, easing: OUT }).onfinish = () => d.remove();
    }
    const t = document.getElementById("toast");
    if (t) { t.textContent = "Risk dropped to low: customer retained"; t.className = "toast show success"; clearTimeout(t._h); t._h = setTimeout(() => (t.className = "toast"), 2600); }
  }
  const arcPoint = (p) => { const a = Math.PI * (1 - p); return [100 + 80 * Math.cos(a), 96 - 80 * Math.sin(a)]; };
  function enhanceResult(res) {
    const svg = $(".gauge", res);
    if (!svg || svg.dataset.cg) return;
    svg.dataset.cg = 1;
    const arc = $$("path", svg)[1], num = $(".gauge-num", svg);
    const p = parseFloat(num.textContent) / 100;
    const level = ["high", "medium", "low"].find((l) => $(`.result-meta .risk.${l}`, res)) || "medium";
    arc.classList.add("cg-arc");
    // morph from the previous prediction (or from 0) by redrawing the arc each frame
    const from = lastGauge.p ?? 0, t0 = performance.now(), D = 1300;
    const ease = (k) => { const c = 1.70158 * 1.2; return 1 + (c + 1) * Math.pow(k - 1, 3) + c * Math.pow(k - 1, 2); };
    (function draw(t) {
      const k = Math.min(1, (t - t0) / D), q = Math.max(0.003, from + (p - from) * ease(k));
      const [x, y] = arcPoint(Math.min(0.999, q));
      arc.setAttribute("d", `M20 96 A80 80 0 0 1 ${x.toFixed(1)} ${y.toFixed(1)}`);
      num.textContent = `${Math.round(Math.max(0, Math.min(1, q)) * 100)}%`;
      if (k < 1) requestAnimationFrame(draw); else num.textContent = `${Math.round(p * 100)}%`;
    })(t0);
    if (lastGauge.level && lastGauge.level !== "low" && level === "low") setTimeout(() => celebrate(svg), 900);
    lastGauge = { p, level };
    svg.style.setProperty("--glow", COLORS[level]);
    svg.style.setProperty("--beat", { high: ".9s", medium: "1.6s", low: "3s" }[level]);
    setTimeout(() => svg.classList.add("beat"), 1300);
    // what-if ghost arc
    const ghost = document.createElementNS("http://www.w3.org/2000/svg", "path");
    ghost.setAttribute("class", "cg-ghost"); ghost.setAttribute("fill", "none"); ghost.setAttribute("stroke", "#16a34a");
    ghost.setAttribute("stroke-width", "16"); ghost.setAttribute("stroke-linecap", "round");
    svg.insertBefore(ghost, num);
    const wi = Object.assign(document.createElement("div"), { className: "cg-whatif" });
    svg.after(wi);
    $$(".reason", res).forEach((row) => {
      const imp = parseFloat($(".bar-track", row)?.title?.replace("+", "")) / 100 || 0;
      row.addEventListener("pointerenter", () => {
        const q = Math.max(0.005, p - imp), [x, y] = arcPoint(q);
        ghost.setAttribute("d", `M20 96 A80 80 0 0 1 ${x.toFixed(1)} ${y.toFixed(1)}`);
        const gl = ghost.getTotalLength();
        ghost.style.strokeDasharray = gl;
        ghost.animate([{ strokeDashoffset: gl }, { strokeDashoffset: 0 }], { duration: 500, easing: OUT });
        svg.classList.add("whatif");
        wi.textContent = `If this were fixed: ${Math.round(p * 100)}% → ${Math.round(q * 100)}%`;
        wi.classList.add("show");
      });
      row.addEventListener("pointerleave", () => { svg.classList.remove("whatif"); wi.classList.remove("show"); });
    });
    // cascade the rest of the result
    $$(".result-meta, .stat, .section-title, .reason, .act, .btn-row", res).forEach((el, i) => { el.style.setProperty("--i", i); el.classList.add("cg-deal"); });
  }

  /* ---------- 8. scanning sweep while predicting ---------- */
  const pbtn = $("#predict-btn"), form = $("#customer-form");
  new MutationObserver(() => form.classList.toggle("cg-scan", pbtn.disabled)).observe(pbtn, { attributes: true, attributeFilter: ["disabled"] });

  /* ---------- 9. typewriter for the retention message ---------- */
  function typeOut(box) {
    if (box.dataset.cg) return;
    box.dataset.cg = 1;
    const full = box.textContent;
    box.textContent = ""; box.classList.add("cg-typing");
    let i = 0;
    (function tick() { i = Math.min(full.length, i + 3); box.textContent = full.slice(0, i); if (i < full.length) setTimeout(tick, 12); else box.classList.remove("cg-typing"); })();
  }

  /* ---------- 10. drop zone gravity + shockwave ---------- */
  const dz = $("#dropzone");
  dz.addEventListener("dragover", (e) => {
    const r = dz.getBoundingClientRect(), dx = e.clientX - (r.left + r.width / 2), dy = e.clientY - (r.top + r.height / 2);
    dz.style.setProperty("--dx", `${Math.max(-40, Math.min(40, dx / 6))}px`);
    dz.style.setProperty("--dy", `${Math.max(-18, Math.min(18, dy / 6))}px`);
  });
  dz.addEventListener("drop", (e) => {
    const r = dz.getBoundingClientRect();
    const w = Object.assign(document.createElement("span"), { className: "cg-wave" });
    Object.assign(w.style, { left: `${e.clientX - r.left}px`, top: `${e.clientY - r.top}px` });
    dz.append(w); setTimeout(() => w.remove(), 1000);
  });

  /* ---------- 11. call list: rows start scrambled and glide into ranked order ---------- */
  function rankGlide(tbody) {
    const rows = [...tbody.rows];
    if (rows.length < 2) return;
    const shuffled = [...rows].sort(() => Math.random() - 0.5);
    shuffled.forEach((r) => tbody.append(r));
    const first = new Map(rows.map((r) => [r, r.getBoundingClientRect().top]));
    rows.forEach((r) => tbody.append(r));
    rows.forEach((r, i) => {
      const dy = first.get(r) - r.getBoundingClientRect().top;
      r.animate([{ transform: `translateY(${dy}px)`, opacity: 0.35 }, { transform: "none", opacity: 1 }], { duration: 1000, delay: 250 + i * 12, easing: SPRING, fill: "backwards" });
    });
    rows.slice(0, 3).forEach((r) => setTimeout(() => r.classList.add("cg-sorted"), 1200));
  }

  /* ---------- observe everything app.js renders ---------- */
  function process(node) {
    if (node.nodeType !== 1) return;
    $$(".kpi .value, .metric .value", node).concat(node.matches?.(".kpi .value, .metric .value") ? [node] : []).forEach(odometer);
    $$(".bar-fill, .mini i, .stack i, .psi-track i", node).forEach((el, i) => grow(el, "width", i));
    $$(".daily i", node).forEach((el, i) => grow(el, "height", i));
    $$(".psi-track i", node).forEach((el) => { if (/red/.test(el.style.background)) el.classList.add("cg-quake"); });
    const res = node.closest?.("#result") || $("#result", node);
    if (res) enhanceResult(res);
    $$(".message-box", node).concat(node.matches?.(".message-box") ? [node] : []).forEach(typeOut);
    if (node.closest?.("#kpis") || node.id === "kpis") readChurnShare();
    if ((node.closest?.("#batch-out") || node.id === "batch-out") && $(".grid.kpis", node.closest("#batch-out") || node)) {
      const fresh = node.matches?.(".grid.kpis") || $(".grid.kpis", node);
      if (fresh) { stagger($("#batch-out")); const tb = $("#b-table tbody"); if (tb) setTimeout(() => rankGlide(tb), 60); }
    }
    if (node.closest?.("#mon-out") && node.parentElement?.id === "mon-out") stagger($("#mon-out"));
  }
  new MutationObserver((muts) => muts.forEach((m) => m.addedNodes.forEach(process))).observe($("main"), { childList: true, subtree: true });

  // first paint
  const boot = () => { moveBlob(); onRoute(); };
  if (document.readyState === "complete") setTimeout(boot, 50); else addEventListener("load", () => setTimeout(boot, 50));
})();
