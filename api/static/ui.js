/* ChurnGuard UI upgrade: top bar, dark mode with circular reveal, Ctrl+K command palette,
   keyboard shortcuts, segmented toggles, range sliders, live prediction, sparklines, metric rings, ripples. */
(() => {
  const $ = (s, el = document) => el.querySelector(s);
  const $$ = (s, el = document) => [...el.querySelectorAll(s)];
  const REDUCED = matchMedia("(prefers-reduced-motion: reduce)").matches;
  const store = { get: (k) => { try { return localStorage.getItem(k); } catch { return null; } }, set: (k, v) => { try { localStorage.setItem(k, v); } catch {} } };
  const PAGES = { overview: "Overview", predict: "Predict Customer", batch: "Batch Scoring", model: "Model Insights", monitoring: "Monitoring" };

  /* ---------- theme ---------- */
  const logo = $(".brand img");
  function applyTheme(t) {
    document.documentElement.dataset.theme = t;
    logo.src = t === "dark" ? "/static/logo-dark.png" : "/static/logo.png";
    $("#theme-btn") && ($("#theme-btn").textContent = t === "dark" ? "☀️" : "🌙");
    store.set("cg-theme", t);
  }
  applyTheme(store.get("cg-theme") || (matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light"));
  function toggleTheme(e) {
    const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    if (!document.startViewTransition || REDUCED) return applyTheme(next);
    const x = e?.clientX ?? innerWidth - 40, y = e?.clientY ?? 30;
    const r = Math.hypot(Math.max(x, innerWidth - x), Math.max(y, innerHeight - y));
    document.startViewTransition(() => applyTheme(next)).ready.then(() => {
      document.documentElement.animate({ clipPath: [`circle(0px at ${x}px ${y}px)`, `circle(${r}px at ${x}px ${y}px)`] },
        { duration: 650, easing: "cubic-bezier(.16,1,.3,1)", pseudoElement: "::view-transition-new(root)" });
    });
  }

  /* ---------- top bar ---------- */
  const main = $("main");
  const bar = document.createElement("div");
  bar.className = "topbar";
  bar.innerHTML = `<div class="crumbs">ChurnGuard <span>/</span> <b id="crumb">Overview</b>
      <span class="date">${new Date().toLocaleDateString(undefined, { weekday: "short", day: "numeric", month: "short" })}</span></div>
    <span class="spacer"></span>
    <button class="kbtn" id="k-open" aria-label="Open command palette">🔍 <span>Search or jump to…</span><kbd>Ctrl K</kbd></button>
    <button class="iconbtn" id="theme-btn" title="Toggle theme (T)" aria-label="Toggle dark mode"></button>`;
  main.prepend(bar);
  applyTheme(document.documentElement.dataset.theme);
  $("#theme-btn").addEventListener("click", toggleTheme);
  addEventListener("scroll", () => bar.classList.toggle("scrolled", scrollY > 8), { passive: true });
  const setCrumb = () => { const p = (location.hash || "#overview").slice(1); $("#crumb").textContent = PAGES[p] || "Overview"; };
  addEventListener("hashchange", setCrumb); setCrumb();

  // greeting by time of day
  const h = new Date().getHours();
  const h1 = $("#page-overview h1");
  if (h1) h1.textContent = h < 12 ? "Good morning" : h < 17 ? "Good afternoon" : "Good evening";

  /* ---------- command palette ---------- */
  const ACTIONS = [
    ...Object.entries(PAGES).map(([k, v], i) => ({ ic: ["▦", "👤", "📄", "📊", "📈"][i], label: `Go to ${v}`, hint: `${i + 1}`, run: () => (location.hash = k) })),
    { ic: "⚠", label: "Predict the high-risk example", run: () => { location.hash = "predict"; setTimeout(() => $('[data-example="risky"]').click(), 120); } },
    { ic: "♥", label: "Predict the loyal example", run: () => { location.hash = "predict"; setTimeout(() => $('[data-example="loyal"]').click(), 120); } },
    { ic: "⬆", label: "Upload a CSV to score", run: () => { location.hash = "batch"; setTimeout(() => $("#csv-input").click(), 150); } },
    { ic: "⬇", label: "Download the sample CSV", run: () => (location.href = "/sample.csv") },
    { ic: "◐", label: "Toggle dark mode", hint: "T", run: () => toggleTheme() },
    { ic: "⟳", label: "Refresh monitoring", run: () => { location.hash = "monitoring"; setTimeout(() => $("#refresh-mon").click(), 120); } },
    { ic: "{}", label: "Open API docs", run: () => open("/docs", "_blank") },
  ];
  const back = document.createElement("div");
  back.className = "palette-back";
  back.innerHTML = `<div class="palette" role="dialog" aria-label="Command palette"><input id="k-in" placeholder="Type a command or page…" autocomplete="off">
    <ul id="k-list"></ul><div class="foot"><span><kbd>↑</kbd> <kbd>↓</kbd> move</span><span><kbd>Enter</kbd> run</span><span><kbd>Esc</kbd> close</span></div></div>`;
  document.body.append(back);
  let sel = 0, shown = ACTIONS;
  const renderList = () => {
    $("#k-list").innerHTML = shown.map((a, i) => `<li class="${i === sel ? "sel" : ""}" data-i="${i}"><span class="ic">${a.ic}</span>${a.label}${a.hint ? `<small><kbd>${a.hint}</kbd></small>` : ""}</li>`).join("")
      || `<li style="cursor:default"><span class="ic">∅</span>No matches</li>`;
    $(".sel", back)?.scrollIntoView({ block: "nearest" });
  };
  const fuzzy = (q, s) => { let i = 0; s = s.toLowerCase(); for (const c of q.toLowerCase()) { i = s.indexOf(c, i); if (i < 0) return false; i++; } return true; };
  function openPalette() { back.classList.add("open"); $("#k-in").value = ""; shown = ACTIONS; sel = 0; renderList(); setTimeout(() => $("#k-in").focus(), 10); }
  function closePalette() { back.classList.remove("open"); $("#k-in").blur(); }
  function runSel() { const a = shown[sel]; if (a) { closePalette(); a.run(); } }
  $("#k-open").addEventListener("click", openPalette);
  back.addEventListener("click", (e) => { if (e.target === back) closePalette(); const li = e.target.closest("li[data-i]"); if (li) { sel = +li.dataset.i; runSel(); } });
  $("#k-in").addEventListener("input", (e) => { shown = ACTIONS.filter((a) => fuzzy(e.target.value, a.label)); sel = 0; renderList(); });
  $("#k-in").addEventListener("keydown", (e) => {
    if (e.key === "ArrowDown") { sel = Math.min(shown.length - 1, sel + 1); renderList(); e.preventDefault(); }
    if (e.key === "ArrowUp") { sel = Math.max(0, sel - 1); renderList(); e.preventDefault(); }
    if (e.key === "Enter") runSel();
    if (e.key === "Escape") closePalette();
  });

  /* ---------- keyboard shortcuts ---------- */
  addEventListener("keydown", (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === "k") { e.preventDefault(); back.classList.contains("open") ? closePalette() : openPalette(); return; }
    if (e.target.closest("input, select, textarea") || e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === "/") { e.preventDefault(); openPalette(); }
    else if (e.key.toLowerCase() === "t") toggleTheme();
    else if (/^[1-5]$/.test(e.key)) location.hash = Object.keys(PAGES)[+e.key - 1];
    else if (e.key === "Escape") closePalette();
  });

  /* ---------- form: segmented toggles, sliders, live prediction ---------- */
  const form = $("#customer-form");
  function buildSegments() {
    $$("select", form).filter((s) => s.options.length === 2).forEach((s) => {
      if (s.previousElementSibling?.classList.contains("seg")) return;
      const seg = document.createElement("div");
      seg.className = "seg"; seg.setAttribute("role", "radiogroup"); seg.setAttribute("aria-label", s.closest(".field").querySelector("label").textContent);
      seg.innerHTML = `<span class="thumb"></span>${[...s.options].map((o) => `<button type="button" role="radio" data-v="${o.value}">${o.textContent}</button>`).join("")}`;
      s.before(seg);
      seg.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; s.value = b.dataset.v; s.dispatchEvent(new Event("change", { bubbles: true })); syncSeg(seg, s); });
      syncSeg(seg, s);
    });
  }
  function syncSeg(seg, s) {
    const btns = $$("button", seg), i = Math.max(0, btns.findIndex((b) => b.dataset.v === s.value));
    btns.forEach((b, k) => { b.classList.toggle("on", k === i); b.setAttribute("aria-checked", k === i); });
    const th = $(".thumb", seg);
    th.style.width = `calc((100% - 6px) / ${btns.length})`;
    th.style.transform = `translateX(${i * 100}%)`;
  }
  const RANGES = { tenure: [0, 72, 1], MonthlyCharges: [18, 120, 0.5] };
  function buildRanges() {
    for (const [name, [min, max, step]] of Object.entries(RANGES)) {
      const inp = $(`[name="${name}"]`, form);
      if (!inp || inp.nextElementSibling?.classList.contains("rng")) continue;
      const r = Object.assign(document.createElement("input"), { type: "range", min, max, step, className: "rng" });
      r.setAttribute("aria-label", `${name} slider`);
      inp.after(r);
      r.addEventListener("input", () => { inp.value = r.value; inp.dispatchEvent(new Event("input", { bubbles: true })); });
      inp.addEventListener("input", () => (r.value = inp.value));
      r.value = inp.value;
    }
  }
  const syncAll = () => {
    $$(".seg", form).forEach((seg) => syncSeg(seg, seg.nextElementSibling));
    for (const name of Object.keys(RANGES)) { const inp = $(`[name="${name}"]`, form); if (inp?.nextElementSibling) inp.nextElementSibling.value = inp.value; }
  };
  $$("[data-example]").forEach((b) => b.addEventListener("click", () => requestAnimationFrame(syncAll)));

  // live prediction toggle (off by default: every prediction is logged for monitoring)
  const foot = $(".form-foot", form);
  const live = document.createElement("div");
  live.className = "livebar";
  live.innerHTML = `<button type="button" class="switch" id="live-sw" role="switch" aria-checked="false" aria-label="Live prediction"></button><span>Live prediction as you edit</span>`;
  foot.prepend(live);
  let liveOn = store.get("cg-live") === "1", tm;
  const setLive = (v) => { liveOn = v; $("#live-sw").classList.toggle("on", v); $("#live-sw").setAttribute("aria-checked", v); store.set("cg-live", v ? "1" : "0"); };
  setLive(liveOn);
  $("#live-sw").addEventListener("click", () => setLive(!liveOn));
  form.addEventListener("input", () => liveOn && (clearTimeout(tm), (tm = setTimeout(() => form.requestSubmit(), 500))));
  form.addEventListener("change", () => liveOn && (clearTimeout(tm), (tm = setTimeout(() => form.requestSubmit(), 300))));

  /* ---------- ripple + magnetic primary buttons ---------- */
  document.addEventListener("pointerdown", (e) => {
    const b = e.target.closest(".btn");
    if (!b || REDUCED || b.tagName === "SELECT") return;
    const r = b.getBoundingClientRect(), d = Math.max(r.width, r.height) * 2.2;
    const s = Object.assign(document.createElement("span"), { className: "cg-ripple" });
    Object.assign(s.style, { left: `${e.clientX - r.left}px`, top: `${e.clientY - r.top}px`, width: `${d}px`, height: `${d}px` });
    b.append(s); setTimeout(() => s.remove(), 650);
  });
  if (!REDUCED) {
    document.addEventListener("pointermove", (e) => {
      const k = e.target.closest?.(".kpi");
      if (k) {
        const r = k.getBoundingClientRect(), px = (e.clientX - r.left) / r.width - 0.5, py = (e.clientY - r.top) / r.height - 0.5;
        k.style.transform = `translateY(-3px) rotateX(${(-py * 7).toFixed(2)}deg) rotateY(${(px * 9).toFixed(2)}deg)`;
      }
      const b = e.target.closest?.(".btn.primary");
      if (b) { const r = b.getBoundingClientRect(); b.style.translate = `${((e.clientX - r.left) / r.width - 0.5) * 6}px ${((e.clientY - r.top) / r.height - 0.5) * 5}px`; }
    });
    document.addEventListener("pointerout", (e) => {
      const k = e.target.closest?.(".kpi"); if (k && !k.contains(e.relatedTarget)) k.style.transform = "";
      const b = e.target.closest?.(".btn.primary"); if (b && !b.contains(e.relatedTarget)) b.style.translate = "";
    });
  }

  /* ---------- sparkline on "Predictions logged" + rings on model metrics ---------- */
  const SVGNS = "http://www.w3.org/2000/svg";
  async function sparkline(card) {
    if (card.querySelector(".spark")) return;
    let daily;
    try { daily = (await (await fetch("/monitoring")).json()).daily; } catch { return; }
    if (!daily?.length || card.querySelector(".spark")) return;
    const vals = daily.map((d) => d.n), mx = Math.max(...vals, 1), W = 96, H = 34;
    const pts = vals.map((v, i) => [(i / Math.max(1, vals.length - 1)) * W, H - 3 - (v / mx) * (H - 8)]);
    const line = pts.map((p, i) => `${i ? "L" : "M"}${p[0].toFixed(1)} ${p[1].toFixed(1)}`).join(" ");
    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("class", "spark"); svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("aria-hidden", "true");
    svg.innerHTML = `<defs><linearGradient id="cg-spark-grad" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#6366f1" stop-opacity=".3"/><stop offset="1" stop-color="#6366f1" stop-opacity="0"/></linearGradient></defs>
      <path class="area" d="${line} L${W} ${H} L0 ${H} Z"/><path class="line" d="${line}"/>`;
    card.append(svg);
    const pl = svg.querySelector(".line"), L = pl.getTotalLength();
    if (!REDUCED) { pl.style.strokeDasharray = L; pl.animate([{ strokeDashoffset: L }, { strokeDashoffset: 0 }], { duration: 1400, easing: "cubic-bezier(.16,1,.3,1)" }); }
  }
  function ring(card) {
    if (card.querySelector(".ring")) return;
    const v = parseFloat(card.querySelector(".value")?.getAttribute("aria-label") || card.querySelector(".value")?.textContent);
    if (isNaN(v) || v > 1) return;
    const C = 2 * Math.PI * 18;
    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("class", "ring"); svg.setAttribute("viewBox", "0 0 46 46"); svg.setAttribute("aria-hidden", "true");
    svg.innerHTML = `<circle class="bg" cx="23" cy="23" r="18"/><circle class="fg" cx="23" cy="23" r="18" stroke-dasharray="${C}" stroke-dashoffset="${C * (1 - v)}"/>`;
    card.append(svg);
    if (!REDUCED) svg.querySelector(".fg").animate([{ strokeDashoffset: C }, { strokeDashoffset: C * (1 - v) }], { duration: 1300, delay: 200, easing: "cubic-bezier(.34,1.56,.64,1)", fill: "backwards" });
  }

  /* ---------- watch what app.js renders ---------- */
  new MutationObserver((muts) => {
    for (const m of muts) for (const n of m.addedNodes) {
      if (n.nodeType !== 1) continue;
      if (n.closest?.("#customer-form") || n.querySelector?.("#customer-form select")) { buildSegments(); buildRanges(); }
      const kp = n.closest?.("#kpis") ? n : null;
      if (kp?.matches(".kpi") && /predictions logged/i.test(kp.textContent)) sparkline(kp);
      if (n.matches?.(".metric")) setTimeout(() => ring(n), 30);
      if (n.closest?.("#batch-out") && n.querySelector?.(".progress") && !n.querySelector(".skel")) {
        n.insertAdjacentHTML("beforeend", `<div style="margin-top:14px">${[90, 75, 82, 60, 70].map((w) => `<div class="skel" style="width:${w}%"></div>`).join("")}</div>`);
      }
    }
  }).observe(main, { childList: true, subtree: true });
  buildSegments(); buildRanges();
  $$("#kpis .kpi").forEach((k) => /predictions logged/i.test(k.textContent) && sparkline(k));
  $$(".metric").forEach(ring);
})();
