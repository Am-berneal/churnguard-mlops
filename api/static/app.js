/* ChurnGuard dashboard — plain JavaScript, no build step. */
const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const pct = (x, d = 0) => `${(x * 100).toFixed(d)}%`;
const money = (x) => `$${Number(x).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const RISK_COLOR = { high: "var(--red)", medium: "var(--amber)", low: "var(--green)" };
const PRETTY = {
  tenure: "Tenure", Contract: "Contract type", InternetService: "Internet service",
  OnlineSecurity: "Online security", TechSupport: "Tech support", PaperlessBilling: "Paperless billing",
  PaymentMethod: "Payment method", MonthlyCharges: "Monthly charges", OnlineBackup: "Online backup",
  DeviceProtection: "Device protection", SeniorCitizen: "Senior citizen", StreamingTV: "Streaming TV",
  StreamingMovies: "Streaming movies", MultipleLines: "Multiple lines", PhoneService: "Phone service",
  Partner: "Partner", Dependents: "Dependents", gender: "Gender", TotalCharges: "Total charges",
};

const MODEL_NAMES = { hist_gb: "Gradient boosting", random_forest: "Random forest", logreg: "Logistic regression" };
const state = { info: null, lastCustomer: null, batch: [], shown: 25 };

async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  const body = res.headers.get("content-type")?.includes("json") ? await res.json() : await res.text();
  if (!res.ok) {
    const detail = body?.detail;
    throw new Error(Array.isArray(detail) ? detail.map((d) => `${d.loc?.slice(-1)}: ${d.msg}`).join("; ") : detail || res.statusText);
  }
  return body;
}

function toast(msg, error = false) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = `toast show${error ? " error" : ""}`;
  clearTimeout(t._h);
  t._h = setTimeout(() => (t.className = "toast"), 2600);
}

function probCell(p, level) {
  return `<div class="prob"><div class="mini"><i style="width:${pct(p)};background:${RISK_COLOR[level]}"></i></div>${pct(p, 1)}</div>`;
}

function bars(el, rows, { max, fmt = (v) => v, highlight } = {}) {
  const top = max ?? Math.max(...rows.map((r) => r.value), 1e-9);
  el.innerHTML = rows.length
    ? rows.map((r) => `<div class="bar-row${highlight?.(r) ? " champion" : ""}">
        <span class="name" title="${esc(r.label)}">${esc(r.label)}</span>
        <div class="bar-track"><div class="bar-fill" style="width:${(100 * r.value) / top}%"></div></div>
        <span class="num">${fmt(r.value)}</span></div>`).join("")
    : `<p class="muted">No data yet.</p>`;
}

/* ---------------- routing ---------------- */
const loaders = { overview: loadOverview, model: loadModel, monitoring: loadMonitoring };
function route() {
  const page = (location.hash || "#overview").slice(1);
  $$(".page").forEach((p) => p.classList.toggle("active", p.id === `page-${page}`));
  $$(".sidebar nav a").forEach((a) => a.classList.toggle("active", a.dataset.page === page));
  loaders[page]?.();
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);

/* ---------------- status ---------------- */
async function loadStatus() {
  const pill = $("#model-pill");
  try {
    const h = await api("/health");
    if (h.model_loaded) {
      pill.className = "pill pill-live";
      pill.innerHTML = `<span class="dot"></span><span>Model v${esc(h.version ?? "?")} · Live</span>`;
      state.info = await api("/model-info");
    } else {
      pill.className = "pill pill-down";
      pill.innerHTML = `<span class="dot"></span><span>No model loaded</span>`;
    }
  } catch {
    pill.className = "pill pill-down";
    pill.innerHTML = `<span class="dot"></span><span>API offline</span>`;
  }
}

/* ---------------- overview ---------------- */
async function loadOverview() {
  const info = state.info;
  const m = info?.metrics || {};
  const mon = await api("/monitoring").catch(() => ({}));
  const highShare = mon.n_predictions ? (mon.risk_mix?.high || 0) / mon.n_predictions : null;
  $("#kpis").innerHTML = [
    { label: "Model ROC AUC", value: m.test_roc_auc?.toFixed(3) ?? "—", sub: "on unseen customers", accent: true },
    { label: "Churners caught", value: m.test_recall != null ? pct(m.test_recall) : "—", sub: "recall on the test set" },
    { label: "Predictions logged", value: mon.n_predictions ?? 0, sub: mon.last_prediction ? `last ${mon.last_prediction.replace("T", " ").slice(0, 16)} UTC` : "none yet" },
    { label: "High-risk share", value: highShare != null ? pct(highShare) : "—", sub: `training churn rate ${info?.train_churn_rate ? pct(info.train_churn_rate, 1) : "—"}` },
  ].map((k) => `<div class="card kpi${k.accent ? " accent" : ""}"><div class="label">${k.label}</div><div class="value">${k.value}</div><div class="sub">${k.sub}</div></div>`).join("");

  bars($("#drivers"), (info?.drivers || []).slice(0, 7).map((d) => ({ label: PRETTY[d.feature] || d.feature, value: d.mean_impact })),
    { fmt: (v) => `${(v * 100).toFixed(1)} pt` });

  const mix = mon.risk_mix || {};
  const n = mon.n_predictions || 0;
  $("#live-mini").innerHTML = n
    ? `<div class="stack">${["high", "medium", "low"].map((k) => `<i style="width:${pct((mix[k] || 0) / n)};background:${RISK_COLOR[k]}"></i>`).join("")}</div>
       <div class="legend">${["high", "medium", "low"].map((k) => `<span style="--c:${RISK_COLOR[k]}">${k} ${mix[k] || 0}</span>`).join("")}</div>
       <div class="stat-row" style="margin-top:16px"><div class="stat"><div class="k">Average churn probability</div><div class="v">${pct(mon.avg_probability || 0, 1)}</div></div>
       <div class="stat"><div class="k">Data drift</div><div class="v"><span class="status ${mon.overall_status?.split(" ")[0]}">${esc(mon.overall_status)}</span></div></div></div>`
    : `<p class="muted">No predictions yet. Try <a href="#predict">a single customer</a> or <a href="#batch">upload a CSV</a>.</p>`;

  const recent = mon.recent || [];
  $("#recent").innerHTML = `<table><thead><tr><th>Time (UTC)</th><th>Source</th><th>Contract</th><th>Tenure</th><th>Monthly</th><th>Churn probability</th><th>Risk</th></tr></thead><tbody>
    ${recent.length ? recent.map((r) => `<tr><td class="nowrap">${esc(r.ts.replace("T", " ").slice(0, 19))}</td><td>${esc(r.source)}</td><td>${esc(r.Contract)}</td><td>${r.tenure} mo</td>
      <td>$${Number(r.MonthlyCharges).toFixed(2)}</td><td>${probCell(r.probability, r.risk_level)}</td><td><span class="risk ${r.risk_level}">${r.risk_level}</span></td></tr>`).join("")
      : `<tr><td colspan="7" class="empty-row">Predictions will appear here.</td></tr>`}</tbody></table>`;
}

/* ---------------- predict ---------------- */
const YN = ["Yes", "No"];
const ADDON = ["No", "Yes", "No internet service"];
const FIELDS = {
  "f-account": [
    ["tenure", "Tenure (months)", "number", { min: 0, max: 120 }],
    ["Contract", "Contract", ["Month-to-month", "One year", "Two year"]],
    ["PaymentMethod", "Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]],
    ["MonthlyCharges", "Monthly charges ($)", "number", { min: 0, step: 0.01 }],
    ["TotalCharges", "Total charges ($)", "number", { min: 0, step: 0.01 }],
    ["PaperlessBilling", "Paperless billing", YN],
  ],
  "f-services": [
    ["PhoneService", "Phone service", YN],
    ["MultipleLines", "Multiple lines", ["No", "Yes", "No phone service"]],
    ["InternetService", "Internet service", ["Fiber optic", "DSL", "No"]],
    ["OnlineSecurity", "Online security", ADDON], ["OnlineBackup", "Online backup", ADDON],
    ["DeviceProtection", "Device protection", ADDON], ["TechSupport", "Tech support", ADDON],
    ["StreamingTV", "Streaming TV", ADDON], ["StreamingMovies", "Streaming movies", ADDON],
  ],
  "f-demo": [
    ["gender", "Gender", ["Female", "Male"]], ["SeniorCitizen", "Senior citizen", [["0", "No"], ["1", "Yes"]]],
    ["Partner", "Partner", YN], ["Dependents", "Dependents", ["No", "Yes"]],
  ],
};
const EXAMPLES = {
  risky: { gender: "Female", SeniorCitizen: 0, Partner: "No", Dependents: "No", tenure: 2, PhoneService: "Yes", MultipleLines: "No",
    InternetService: "Fiber optic", OnlineSecurity: "No", OnlineBackup: "No", DeviceProtection: "No", TechSupport: "No",
    StreamingTV: "Yes", StreamingMovies: "Yes", Contract: "Month-to-month", PaperlessBilling: "Yes",
    PaymentMethod: "Electronic check", MonthlyCharges: 95.5, TotalCharges: 190.1 },
  loyal: { gender: "Male", SeniorCitizen: 0, Partner: "Yes", Dependents: "Yes", tenure: 60, PhoneService: "Yes", MultipleLines: "Yes",
    InternetService: "DSL", OnlineSecurity: "Yes", OnlineBackup: "Yes", DeviceProtection: "Yes", TechSupport: "Yes",
    StreamingTV: "No", StreamingMovies: "No", Contract: "Two year", PaperlessBilling: "No",
    PaymentMethod: "Credit card (automatic)", MonthlyCharges: 64.9, TotalCharges: 3894 },
};

function buildForm() {
  for (const [id, fields] of Object.entries(FIELDS)) {
    $(`#${id}`).innerHTML = fields.map(([name, label, type, attrs = {}]) => {
      const input = Array.isArray(type)
        ? `<select name="${name}">${type.map((o) => (Array.isArray(o) ? `<option value="${o[0]}">${o[1]}</option>` : `<option>${o}</option>`)).join("")}</select>`
        : `<input name="${name}" type="number" required ${Object.entries(attrs).map(([k, v]) => `${k}="${v}"`).join(" ")}>`;
      return `<div class="field"><label>${label}</label>${input}</div>`;
    }).join("");
  }
  fillForm(EXAMPLES.risky);
  $("#customer-form").addEventListener("submit", (e) => { e.preventDefault(); runPrediction(); });
  $$("[data-example]").forEach((b) => b.addEventListener("click", () => { fillForm(EXAMPLES[b.dataset.example]); runPrediction(); }));
}

function fillForm(c) {
  for (const [k, v] of Object.entries(c)) {
    const el = $(`#customer-form [name="${k}"]`);
    if (el) el.value = String(v);
  }
}

function readForm() {
  const c = Object.fromEntries(new FormData($("#customer-form")).entries());
  for (const k of ["tenure", "SeniorCitizen"]) c[k] = parseInt(c[k], 10);
  for (const k of ["MonthlyCharges", "TotalCharges"]) c[k] = parseFloat(c[k]);
  return c;
}

function gauge(p, level) {
  const r = 80, cx = 100, cy = 96, a = Math.PI * (1 - p);
  const x = cx + r * Math.cos(a), y = cy - r * Math.sin(a);
  const t = state.info?.threshold ?? 0.5, ta = Math.PI * (1 - t);
  return `<svg class="gauge" width="220" height="130" viewBox="0 0 200 118">
    <path d="M20 96 A80 80 0 0 1 180 96" fill="none" stroke="#eef2f7" stroke-width="16" stroke-linecap="round"/>
    <path d="M20 96 A80 80 0 0 1 ${x.toFixed(1)} ${y.toFixed(1)}" fill="none" stroke="${RISK_COLOR[level]}" stroke-width="16" stroke-linecap="round"/>
    <line x1="${cx + 66 * Math.cos(ta)}" y1="${cy - 66 * Math.sin(ta)}" x2="${cx + 94 * Math.cos(ta)}" y2="${cy - 94 * Math.sin(ta)}" stroke="#94a3b8" stroke-width="2" stroke-dasharray="3 2"/>
    <text x="100" y="86" text-anchor="middle" class="gauge-num">${pct(p)}</text>
    <text x="100" y="106" text-anchor="middle" class="gauge-label">churn probability</text></svg>`;
}

async function runPrediction() {
  const btn = $("#predict-btn");
  btn.disabled = true; btn.textContent = "Predicting…";
  const customer = readForm();
  try {
    const r = await api("/predict", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(customer) });
    state.lastCustomer = customer;
    renderResult(r);
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = "Predict churn risk";
  }
}

function renderResult(r) {
  const verdict = { high: "This customer is likely to leave.", medium: "This customer shows some warning signs.", low: "This customer looks loyal." }[r.risk_level];
  const maxImpact = Math.max(...r.reasons.map((x) => x.impact), 0.01);
  $("#result").innerHTML = `
    ${gauge(r.churn_probability, r.risk_level)}
    <div class="result-meta"><span class="risk ${r.risk_level}">${r.risk_level} risk</span><span class="muted">${verdict}</span></div>
    <div class="stat-row">
      <div class="stat"><div class="k">Priority</div><div class="v"><span class="prio ${r.priority}" style="font-size:18px">${r.priority}</span></div></div>
      <div class="stat"><div class="k">Revenue at risk (12 mo)</div><div class="v">${money(r.revenue_at_risk)}</div></div>
    </div>
    <div class="section-title">Why this risk?</div>
    ${r.reasons.length ? r.reasons.map((x) => `<div class="reason"><span>${esc(x.reason)}</span>
      <div class="bar-track" title="+${(x.impact * 100).toFixed(1)} points"><div class="bar-fill" style="width:${(100 * x.impact) / maxImpact}%"></div></div></div>`).join("")
      : `<p class="muted">No single factor stands out.</p>`}
    <div class="section-title">Recommended actions</div>
    ${r.actions.length ? r.actions.map((a) => `<div class="act"><strong>${esc(a.action)}</strong><small>${esc(a.channel)} · ${esc(a.cost)} cost · because: ${esc(a.because.toLowerCase())}</small></div>`).join("")
      : `<p class="muted">No action needed. Keep up the good service.</p>`}
    ${r.risk_level !== "low" ? `<div class="btn-row" style="margin-top:6px"><select id="tone" class="btn ghost small"><option>friendly</option><option>professional</option><option>concise</option></select>
      <button class="btn primary small" id="msg-btn">✨ Write retention message</button></div><div id="msg"></div>` : ""}
    <p class="result-note" style="margin-top:12px">Impact bars show how much the probability would drop if that factor matched a loyal customer.</p>`;
  $("#msg-btn")?.addEventListener("click", writeMessage);
}

async function writeMessage() {
  const btn = $("#msg-btn");
  btn.disabled = true; btn.textContent = "Writing…";
  try {
    const m = await api("/retention-message", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ customer: state.lastCustomer, tone: $("#tone").value }) });
    const src = m.source.startsWith("gemini") ? `Written by ${m.source.replace("gemini:", "Gemini ")}` : "Template (set GEMINI_API_KEY to use Gemini)";
    $("#msg").innerHTML = `<div class="message-box">${esc(m.message)}</div>
      <div class="message-meta"><span>${esc(src)}</span><button class="btn ghost small" id="copy-msg">Copy</button></div>`;
    $("#copy-msg").onclick = () => navigator.clipboard.writeText(m.message).then(() => toast("Message copied"));
  } catch (e) {
    toast(e.message, true);
  } finally {
    btn.disabled = false; btn.textContent = "✨ Write retention message";
  }
}

/* ---------------- batch ---------------- */
function setupBatch() {
  const dz = $("#dropzone"), input = $("#csv-input");
  input.addEventListener("change", () => input.files[0] && uploadCsv(input.files[0]));
  ["dragenter", "dragover"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => dz.addEventListener(ev, (e) => { e.preventDefault(); dz.classList.remove("drag"); }));
  dz.addEventListener("drop", (e) => e.dataTransfer.files[0] && uploadCsv(e.dataTransfer.files[0]));
}

async function uploadCsv(file) {
  $("#batch-out").innerHTML = `<div class="card"><strong>Scoring ${esc(file.name)}…</strong><div class="progress"><i></i></div></div>`;
  const fd = new FormData();
  fd.append("file", file);
  try {
    const res = await api("/predict/csv", { method: "POST", body: fd });
    state.batch = res.results;
    renderBatch(res.summary, file.name);
    toast(`Scored ${res.summary.n_customers} customers`);
  } catch (e) {
    $("#batch-out").innerHTML = `<div class="card banner significant">Could not score this file: ${esc(e.message)}</div>`;
  }
  $("#csv-input").value = "";
}

function renderBatch(s, name) {
  const n = s.n_customers || 1;
  $("#batch-out").innerHTML = `
    <div class="grid kpis">
      <div class="card kpi"><div class="label">Customers scored</div><div class="value">${s.n_customers.toLocaleString()}</div><div class="sub">${esc(name)}</div></div>
      <div class="card kpi"><div class="label">High risk</div><div class="value" style="color:var(--red)">${s.n_high}</div><div class="sub">${pct(s.n_high / n)} of the list</div></div>
      <div class="card kpi"><div class="label">Call today (P1)</div><div class="value">${s.n_p1}</div><div class="sub">likely to leave and valuable</div></div>
      <div class="card kpi accent"><div class="label">Revenue at risk</div><div class="value">${money(s.revenue_at_risk_total)}</div><div class="sub">expected loss over 12 months</div></div>
    </div>
    <div class="card">
      <div class="stack">${[["high", s.n_high], ["medium", s.n_medium], ["low", s.n_low]].map(([k, v]) => `<i style="width:${pct(v / n)};background:${RISK_COLOR[k]}"></i>`).join("")}</div>
      <div class="legend">${[["high", s.n_high], ["medium", s.n_medium], ["low", s.n_low]].map(([k, v]) => `<span style="--c:${RISK_COLOR[k]}">${k} ${v}</span>`).join("")}</div>
    </div>
    <div class="card">
      <div class="toolbar">
        <h3>Call list</h3><span class="hint">sorted by revenue at risk</span><span class="spacer"></span>
        <input id="b-search" placeholder="Search customer ID…">
        <select id="b-risk"><option value="">All risks</option><option value="high">High</option><option value="medium">Medium</option><option value="low">Low</option></select>
        <button class="btn primary small" id="b-export">Export CSV</button>
      </div>
      <div class="table-wrap" id="b-table"></div>
    </div>`;
  const reset = () => { state.shown = 25; drawBatchTable(); };
  $("#b-search").addEventListener("input", reset);
  $("#b-risk").addEventListener("change", reset);
  $("#b-export").addEventListener("click", exportBatch);
  drawBatchTable();
}

function drawBatchTable() {
  const q = $("#b-search").value.toLowerCase(), risk = $("#b-risk").value;
  const rows = state.batch.filter((r) => (!risk || r.risk_level === risk) && (!q || String(r.customerID ?? "").toLowerCase().includes(q)));
  $("#b-table").innerHTML = `<table><thead><tr><th>#</th><th>Customer</th><th>Priority</th><th>Churn probability</th><th>Risk</th><th>Revenue at risk</th><th>Top reasons</th><th>Next best action</th></tr></thead><tbody>
    ${rows.slice(0, state.shown).map((r, i) => `<tr><td class="muted">${i + 1}</td><td class="nowrap"><strong>${esc(r.customerID ?? "—")}</strong></td><td><span class="prio ${r.priority}">${r.priority}</span></td>
      <td>${probCell(r.churn_probability, r.risk_level)}</td><td><span class="risk ${r.risk_level}">${r.risk_level}</span></td><td>${money(r.revenue_at_risk)}</td>
      <td class="reasons-cell">${esc(r.reasons.slice(0, 2).map((x) => x.reason).join(" · ") || "—")}</td><td>${esc(r.actions[0]?.action ?? "—")}</td></tr>`).join("")
      || `<tr><td colspan="8" class="empty-row">No customers match.</td></tr>`}</tbody></table>
    ${rows.length > state.shown ? `<div class="more"><button class="btn ghost small" id="b-more">Show more (${rows.length - state.shown} left)</button></div>` : ""}`;
  $("#b-more")?.addEventListener("click", () => { state.shown += 25; drawBatchTable(); });
}

function exportBatch() {
  const head = ["customerID", "priority", "churn_probability", "risk_level", "revenue_at_risk", "top_reasons", "recommended_action"];
  const q = (v) => `"${String(v ?? "").replace(/"/g, '""')}"`;
  const lines = state.batch.map((r) => [r.customerID, r.priority, r.churn_probability, r.risk_level, r.revenue_at_risk,
    r.reasons.map((x) => x.reason).join("; "), r.actions[0]?.action ?? ""].map(q).join(","));
  const blob = new Blob([[head.join(","), ...lines].join("\n")], { type: "text/csv" });
  const a = Object.assign(document.createElement("a"), { href: URL.createObjectURL(blob), download: "churnguard_call_list.csv" });
  a.click();
}

/* ---------------- model ---------------- */
async function loadModel() {
  const info = state.info;
  if (!info) { $("#model-head").innerHTML = `<p class="muted">No model loaded. Run <code>make pipeline</code> first.</p>`; return; }
  const m = info.metrics || {};
  const trained = info.trained_at ? new Date(info.trained_at).toLocaleString() : "—";
  const typeName = MODEL_NAMES[info.model_type] || info.model_type;
  $("#model-head").innerHTML = `<div><h2 style="font-size:18px">${esc(info.name)}</h2>
      <div class="tags"><span class="tag indigo">v${esc(info.version)} · champion</span><span class="tag">${esc(typeName)}</span>
      <span class="tag">Threshold ${info.threshold.toFixed(2)}</span><span class="tag">Trained ${esc(trained)}</span><span class="tag">Tracked with MLflow</span></div></div>
    <a class="btn ghost" href="http://localhost:5000" target="_blank">Open in MLflow ↗</a>`;
  const M = [
    ["ROC AUC", m.test_roc_auc, "How well it ranks churners above loyal customers (1.0 = perfect)"],
    ["PR AUC", m.test_pr_auc, "Ranking quality focused on the churners"],
    ["F1", m.test_f1, "Balance of precision and recall"],
    ["Precision", m.test_precision, "Of customers flagged, how many really left"],
    ["Recall", m.test_recall, "Of customers who left, how many we caught"],
    ["Accuracy", m.test_accuracy, "Share of all predictions that were right"],
  ];
  $("#metrics").innerHTML = M.map(([l, v, e]) => `<div class="card metric" title="${e}"><div class="label">${l}</div><div class="value">${v != null ? v.toFixed(3) : "—"}</div><div class="explain">${e}</div></div>`).join("");

  const runs = await api("/model/runs").catch(() => []);
  const best = {};
  for (const r of runs) {
    if (r.cv_roc_auc == null) continue;
    const key = r.champion ? "champion" : r.run_name;
    if (!best[key] || r.cv_roc_auc > best[key].cv_roc_auc) best[key] = r;
  }
  const rows = Object.values(best).sort((a, b) => b.cv_roc_auc - a.cv_roc_auc).slice(0, 6)
    .map((r) => ({ label: r.champion ? `★ Tuned ${MODEL_NAMES[r.model_type] || r.model_type}` : MODEL_NAMES[r.model_type] || r.run_name, value: r.cv_roc_auc, champion: r.champion }));
  const min = Math.min(...rows.map((r) => r.value)) - 0.02;
  bars($("#runs"), rows.map((r) => ({ ...r, value: r.value - min })), { fmt: (v) => (v + min).toFixed(3), highlight: (r) => r.champion });
  $$("img[data-plot]").forEach((img) => { img.src = `/model/plots/${img.dataset.plot}`; img.onerror = () => img.replaceWith(Object.assign(document.createElement("p"), { className: "muted", textContent: "Plot not available." })); });
}

/* ---------------- monitoring ---------------- */
async function loadMonitoring() {
  const out = $("#mon-out");
  const m = await api("/monitoring").catch((e) => ({ error: e.message }));
  if (m.error || m.enabled === false) { out.innerHTML = `<div class="card muted">${esc(m.error || "Prediction logging is disabled (LOG_PREDICTIONS=0).")}</div>`; return; }
  if (!m.n_predictions) { out.innerHTML = `<div class="banner insufficient">No predictions logged yet. Make some predictions, then come back to check for drift.</div>`; return; }
  const status = m.overall_status;
  const msg = {
    stable: "Live data looks like the training data. The model is working in familiar territory.",
    moderate: "Some inputs have shifted moderately. Keep an eye on them.",
    significant: "Live data has shifted significantly. Consider retraining the model.",
    "insufficient data": `Drift is measured once at least ${m.min_samples} predictions are logged (${m.n_predictions} so far).`,
  }[status];
  const maxDay = Math.max(...m.daily.map((d) => d.n), 1);
  out.innerHTML = `
    <div class="banner ${status.split(" ")[0]}"><strong>${status === "insufficient data" ? "Collecting data" : `Drift: ${esc(status)}`}</strong><span>${msg}</span></div>
    <div class="grid kpis">
      <div class="card kpi"><div class="label">Predictions</div><div class="value">${m.n_predictions.toLocaleString()}</div><div class="sub">${Object.entries(m.by_source).map(([k, v]) => `${k} ${v}`).join(" · ")}</div></div>
      <div class="card kpi"><div class="label">Avg churn probability</div><div class="value">${pct(m.avg_probability, 1)}</div><div class="sub">across logged predictions</div></div>
      <div class="card kpi"><div class="label">High risk</div><div class="value" style="color:var(--red)">${m.risk_mix.high || 0}</div><div class="sub">${pct((m.risk_mix.high || 0) / m.n_predictions)} of traffic</div></div>
      <div class="card kpi"><div class="label">Last prediction</div><div class="value" style="font-size:18px">${esc(m.last_prediction.replace("T", " ").slice(0, 16))}</div><div class="sub">UTC</div></div>
    </div>
    <div class="grid two">
      <div class="card"><div class="card-head"><h3>Daily volume</h3><span class="hint">last 14 days</span></div>
        <div class="daily">${m.daily.map((d) => `<div title="${d.n} predictions, avg ${pct(d.avg_probability, 1)}"><span>${d.n}</span><i style="height:${(90 * d.n) / maxDay}px"></i><span>${d.day.slice(5)}</span></div>`).join("")}</div></div>
      <div class="card"><div class="card-head"><h3>How to read drift</h3></div>
        <p class="muted">The <strong>Population Stability Index (PSI)</strong> compares the distribution of each input in live traffic with the training data.</p>
        <p style="margin-top:10px"><span class="status stable">&lt; 0.10 stable</span> <span class="status moderate">0.10–0.25 moderate</span> <span class="status significant">≥ 0.25 significant</span></p>
        <p class="muted" style="margin-top:10px">Significant drift means the model is seeing customers unlike the ones it learned from, and its predictions may be less reliable.</p></div>
    </div>
    <div class="card"><div class="card-head"><h3>Feature drift</h3><span class="hint">live traffic vs training data</span></div>
      <div class="table-wrap"><table><thead><tr><th>Feature</th><th>PSI</th><th></th><th>Status</th><th>Training</th><th>Live</th></tr></thead><tbody>
      ${m.drift.map((d) => `<tr><td><strong>${esc(PRETTY[d.feature] || d.feature)}</strong></td><td>${d.psi ?? "—"}</td>
        <td><div class="psi-track"><i style="width:${Math.min(100, ((d.psi || 0) / 0.5) * 100)}%;background:${d.status === "significant" ? "var(--red)" : d.status === "moderate" ? "var(--amber)" : "var(--green)"}"></i></div></td>
        <td><span class="status ${d.status.split(" ")[0]}">${esc(d.status)}</span></td>
        <td class="muted">${d.type === "numeric" ? `mean ${d.train_mean}` : esc(d.train_top)}</td><td>${d.type === "numeric" ? `mean ${d.live_mean}` : esc(d.live_top)}</td></tr>`).join("")}
      </tbody></table></div></div>`;
}

/* ---------------- boot ---------------- */
(async function init() {
  buildForm();
  setupBatch();
  $("#refresh-mon").addEventListener("click", loadMonitoring);
  await loadStatus();
  route();
})();
