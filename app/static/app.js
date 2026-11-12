// Dashboard de Cupo Seguro: galería, subida de imágenes, dibujo de espacios y resultados.
// Sin dependencias externas.
"use strict";

const SVG_NS = "http://www.w3.org/2000/svg";
const STATE_LABEL = { libre: "Libre", ocupado: "Ocupado", verificar: "Por verificar" };
const MIN_BOX = 4; // lado mínimo de una caja dibujada, en píxeles de la imagen

const ui = {
  scenario: "comercial",
  model: null,
  gallery: [],
  source: null,      // {type: "gallery", item} | {type: "upload", file, url}
  result: null,
  selected: null,
  showGt: true,
  drawMode: false,
  boxes: [],         // cajas dibujadas en píxeles de la imagen
  dragStart: null,
  dragBox: null,
};

const $ = (id) => document.getElementById(id);
const photo = $("photo");
const overlay = $("overlay");
const canvas = $("draw-canvas");

// ---------------------------------------------------------------------------
// Utilidades
// ---------------------------------------------------------------------------

function showBanner(message, kind = "error") {
  const banner = $("banner");
  banner.textContent = message;
  banner.className = `banner ${kind}`;
}

function hideBanner() {
  $("banner").className = "banner hidden";
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  let body = null;
  try {
    body = await response.json();
  } catch (_) {
    body = null;
  }
  if (!response.ok) {
    const detail = body && body.detail ? body.detail : `Error ${response.status}`;
    const error = new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
    error.status = response.status;
    throw error;
  }
  return body;
}

function fmt(value, digits = 2) {
  return value === null || value === undefined || Number.isNaN(value) ? "–" : Number(value).toFixed(digits);
}

function fmtQ(value) {
  if (value === null || value === undefined) return "–";
  return "Q " + Math.round(value).toLocaleString("es-GT");
}

function setSource(source) {
  // Libera la URL de la imagen subida anterior
  if (ui.source && ui.source.type === "upload" && ui.source.url && (!source || source.url !== ui.source.url)) {
    URL.revokeObjectURL(ui.source.url);
  }
  ui.source = source;
}

function setBusy(on) {
  $("spinner").classList.toggle("hidden", !on);
}

function loadImage(url) {
  return new Promise((resolve, reject) => {
    photo.onload = () => {
      // Los patches (150x150) se amplían para que se vean bien
      $("stage").classList.toggle("small", photo.naturalWidth < 400);
      resolve();
    };
    photo.onerror = () => reject(new Error("El archivo no es una imagen válida (usa JPG o PNG)."));
    photo.src = url;
  });
}

// ---------------------------------------------------------------------------
// Modelo y escenario
// ---------------------------------------------------------------------------

function thresholds(scenario) {
  if (ui.result) return ui.result.thresholds[scenario];
  return ui.model ? ui.model.thresholds[scenario] : null;
}

function renderThresholds() {
  for (const scenario of ["comercial", "uvg"]) {
    const th = thresholds(scenario);
    const node = $(`th-${scenario}`);
    node.innerHTML = th ? `t<sub>L</sub> ${fmt(th.t_L, 3)}, t<sub>H</sub> ${fmt(th.t_H, 3)}` : "t<sub>L</sub> –, t<sub>H</sub> –";
  }
}

function renderModel() {
  const m = ui.model;
  const box = $("model-info");
  if (!m) return;
  const val = m.val_metrics || {};
  const bin = val.binary_t05 || {};
  const value = val.value || {};
  const com = value.comercial || {};
  const uvg = value.uvg || {};
  const bench = m.benchmark;
  const rows = [
    ["Nombre", m.name],
    ["Run", m.run_id],
    ["Arquitectura", m.model],
    ["Tamaño", `${fmt(m.size_mb, 2)} MB`],
    ["Temperatura", `${fmt(m.temperature, 3)}${m.calibrated ? " (calibrado)" : ""}`],
    ["Val F1 macro (t = 0.5)", fmt(bin.f1_macro, 3)],
    ["Val AUC", fmt(bin.auc, 3)],
    ["Val ocupado→libre (t = 0.5)", bin.rate_busy_as_free === undefined ? "–" : `${fmt(100 * bin.rate_busy_as_free, 2)} %`],
    ["Valor comercial (val)", `${fmtQ(com.objective_value)} /mes`],
    ["Valor UVG (val)", `${fmtQ(uvg.objective_value)} /semestre`],
  ];
  if (bench) {
    rows.push(["CPU por frame (p50)", `${fmt(bench.cpu_per_frame_p50_ms, 1)} ms (${bench.slots_per_frame || "–"} espacios)`]);
    rows.push(["CPU por espacio (p50)", `${fmt(bench.cpu_per_slot_p50_ms, 2)} ms`]);
    rows.push(["Costo por 1,000 predicciones", `Q ${fmt(bench.costo_por_1000_pred_Q, 5)}`]);
    if (bench.matches_model === false) rows.push(["Aviso", "la latencia es de otro modelo; vuelve a correr el benchmark"]);
  }
  box.className = "";
  const dl = document.createElement("dl");
  dl.className = "kv";
  for (const [k, v] of rows) {
    const dt = document.createElement("dt");
    dt.textContent = k;
    const dd = document.createElement("dd");
    dd.textContent = v ?? "–";
    dl.append(dt, dd);
  }
  const rule = document.createElement("p");
  rule.className = "hint";
  rule.textContent = m.rule || "";
  box.replaceChildren(dl, rule);
}

async function loadModel() {
  try {
    ui.model = await api("/api/model");
    hideBanner();
    renderModel();
    renderThresholds();
  } catch (error) {
    ui.model = null;
    showBanner(error.message, error.status === 503 ? "error" : "warn");
    $("model-info").textContent = "Sin modelo cargado.";
  }
}

function setScenario(scenario) {
  ui.scenario = scenario;
  document.querySelectorAll(".scenario-btn").forEach((btn) => {
    const on = btn.dataset.scenario === scenario;
    btn.classList.toggle("active", on);
    btn.setAttribute("aria-checked", on ? "true" : "false");
  });
  renderResult();
}

// ---------------------------------------------------------------------------
// Galería
// ---------------------------------------------------------------------------

function thumb(item) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "thumb";
  button.title = item.note ? `${item.title}: ${item.note}` : item.title;
  button.disabled = !item.available;
  const img = document.createElement("img");
  img.loading = "lazy";
  img.alt = item.title;
  if (item.available) img.src = item.image_url;
  const caption = document.createElement("span");
  caption.textContent = item.available ? item.title : `${item.title} (no está en DATA_ROOT)`;
  button.append(img, caption);
  if (item.note) {
    const note = document.createElement("em");
    note.textContent = item.note;
    button.append(note);
  }
  button.addEventListener("click", () => analyzeGallery(item, button));
  return button;
}

async function loadGallery() {
  try {
    const data = await api("/api/gallery");
    ui.gallery = data.items;
    const frames = $("gallery-frames");
    const patches = $("gallery-patches");
    frames.innerHTML = "";
    patches.innerHTML = "";
    for (const item of data.items) (item.kind === "frame" ? frames : patches).append(thumb(item));
    if (!data.items.length) frames.textContent = "La galería está vacía (genera app/gallery.json).";
    // Enlace directo a un ejemplo (útil en la demo): /?ejemplo=<id>&escenario=uvg
    const params = new URLSearchParams(window.location.search);
    if (params.get("escenario") === "uvg") setScenario("uvg");
    const wanted = data.items.find((item) => item.id === params.get("ejemplo"));
    if (wanted && wanted.available) analyzeGallery(wanted, null);
  } catch (error) {
    $("gallery-frames").textContent = `No se pudo cargar la galería: ${error.message}`;
  }
}

async function analyzeGallery(item, button) {
  document.querySelectorAll(".thumb").forEach((b) => b.classList.remove("active"));
  if (button) button.classList.add("active");
  setSource({ type: "gallery", item });
  exitDrawMode();
  const endpoint = item.kind === "frame" ? "/api/predict/frame" : "/api/predict/patch";
  await runPrediction(item.image_url, () =>
    api(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gallery_id: item.id, scenario: ui.scenario }),
    }), item.title, item.note || item.split_origin || "");
}

// ---------------------------------------------------------------------------
// Predicción
// ---------------------------------------------------------------------------

async function runPrediction(imageUrl, request, title, subtitle) {
  hideBanner();
  setBusy(true);
  $("viewer-title").textContent = title;
  $("viewer-sub").textContent = subtitle || "";
  try {
    await loadImage(imageUrl);
    $("stage").classList.remove("empty");
    ui.result = await request();
    // Si el modelo se reemplazó en el servidor, se actualiza el panel "Modelo"
    if (ui.result.model && (!ui.model || ui.model.run_id !== ui.result.model.run_id)) loadModel();
    // Con un solo espacio (modo patch) queda seleccionado para poder reportarlo
    ui.selected = ui.result.slots.length === 1 ? ui.result.slots[0].slot_id : null;
    renderResult();
  } catch (error) {
    ui.result = null;
    renderResult();
    showBanner(error.message, error.status === 503 ? "error" : "warn");
  } finally {
    setBusy(false);
  }
}

function uploadForm(file, extra = {}) {
  const form = new FormData();
  form.append("file", file);
  form.append("scenario", ui.scenario);
  for (const [k, v] of Object.entries(extra)) form.append(k, v);
  return form;
}

function onSubmitFrame(event) {
  event.preventDefault();
  const form = event.target;
  const file = form.file.files[0];
  if (!file) return showBanner("Elige una imagen para analizar.", "warn");
  const camera = form.camera.value;
  const url = URL.createObjectURL(file);
  setSource({ type: "upload", file, url });
  exitDrawMode();
  runPrediction(url, () => api("/api/predict/frame", { method: "POST", body: uploadForm(file, { camera }) }),
    `Imagen subida (cámara ${camera})`, file.name);
}

function onSubmitPatch(event) {
  event.preventDefault();
  const file = event.target.file.files[0];
  if (!file) return showBanner("Elige una imagen para analizar.", "warn");
  const url = URL.createObjectURL(file);
  setSource({ type: "upload", file, url });
  exitDrawMode();
  runPrediction(url, () => api("/api/predict/patch", { method: "POST", body: uploadForm(file) }),
    "Espacio subido", file.name);
}

// ---------------------------------------------------------------------------
// Resultados
// ---------------------------------------------------------------------------

function outcome(slot) {
  // "ok", "error" o null (sin etiqueta o por verificar)
  if (slot.ground_truth === null || slot.ground_truth === undefined) return null;
  const state = slot.states[ui.scenario];
  if (state === "verificar") return null;
  return (state === "ocupado") === (slot.ground_truth === 1) ? "ok" : "error";
}

function renderResult() {
  renderThresholds();
  overlay.innerHTML = "";
  const r = ui.result;
  const gtAvailable = Boolean(r && r.ground_truth && r.ground_truth.available);
  $("gt-toggle-wrap").classList.toggle("hidden", !gtAvailable);
  renderWarnings(r ? r.warnings : []);
  if (!r) {
    $("c-free").textContent = $("c-busy").textContent = $("c-verify").textContent = "–";
    $("latency").textContent = "–";
    $("gt-summary").classList.add("hidden");
    renderSlotDetail();
    return;
  }
  overlay.setAttribute("viewBox", `0 0 ${r.image.width} ${r.image.height}`);
  overlay.setAttribute("preserveAspectRatio", "none");
  const stroke = Math.max(2, Math.round(r.image.width / 400));
  for (const slot of r.slots) overlay.append(slotShape(slot, stroke, gtAvailable && ui.showGt));
  const s = r.summary_by_scenario[ui.scenario];
  $("c-free").textContent = s.libres;
  $("c-busy").textContent = s.ocupados;
  $("c-verify").textContent = s.por_verificar;
  $("latency").textContent = `${fmt(r.latency_ms.total, 0)} ms en total (modelo ${fmt(r.latency_ms.model, 0)} ms, ${r.latency_ms.n_slots} espacios)`;
  renderGtSummary(gtAvailable);
  renderSlotDetail();
}

function slotShape(slot, stroke, markGt) {
  const g = document.createElementNS(SVG_NS, "g");
  const state = slot.states[ui.scenario];
  const rect = document.createElementNS(SVG_NS, "rect");
  const { x, y, w, h } = slot.box;
  rect.setAttribute("x", x);
  rect.setAttribute("y", y);
  rect.setAttribute("width", w);
  rect.setAttribute("height", h);
  rect.setAttribute("stroke-width", ui.selected === slot.slot_id ? stroke * 2 : stroke);
  rect.setAttribute("class", `slot ${state}${ui.selected === slot.slot_id ? " selected" : ""}`);
  // Si la caja es toda la imagen (modo patch), solo se dibuja el borde para no teñir la foto
  if (w >= ui.result.image.width && h >= ui.result.image.height) rect.classList.add("whole");
  const title = document.createElementNS(SVG_NS, "title");
  title.textContent = `Espacio ${slot.slot_id}: ${STATE_LABEL[state]}, p(ocupado) = ${fmt(slot.p_busy, 3)}`;
  rect.append(title);
  g.append(rect);
  const mark = markGt ? outcome(slot) : null;
  if (mark) {
    rect.classList.add(mark === "error" ? "wrong" : "right");
    const text = document.createElementNS(SVG_NS, "text");
    const size = Math.max(14, Math.min(w, h) * 0.45);
    text.setAttribute("x", x + w / 2);
    text.setAttribute("y", y + h / 2 + size * 0.35);
    text.setAttribute("font-size", size);
    text.setAttribute("class", `gt-mark ${mark}`);
    text.textContent = mark === "error" ? "✗" : "✓";
    g.append(text);
  }
  g.addEventListener("mousemove", (event) => showTooltip(event, slot));
  g.addEventListener("mouseleave", hideTooltip);
  g.addEventListener("click", () => {
    ui.selected = slot.slot_id;
    renderResult();
  });
  return g;
}

function slotText(slot) {
  const state = slot.states[ui.scenario];
  let text = `Espacio ${slot.slot_id} · ${STATE_LABEL[state]} · p(ocupado) = ${fmt(slot.p_busy, 3)}`;
  if (slot.ground_truth !== null && slot.ground_truth !== undefined) {
    text += ` · real: ${slot.ground_truth === 1 ? "ocupado" : "libre"}`;
  }
  return text;
}

function showTooltip(event, slot) {
  const tip = $("tooltip");
  tip.textContent = slotText(slot);
  tip.classList.remove("hidden");
  tip.style.left = `${event.clientX + 14}px`;
  tip.style.top = `${event.clientY + 14}px`;
}

function hideTooltip() {
  $("tooltip").classList.add("hidden");
}

function renderWarnings(warnings) {
  const box = $("warnings");
  box.innerHTML = "";
  for (const message of warnings || []) {
    const p = document.createElement("p");
    p.className = "warning";
    p.textContent = `⚠ ${message}`;
    box.append(p);
  }
}

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function renderGtSummary(available) {
  const box = $("gt-summary");
  if (!available || !ui.showGt) {
    box.classList.add("hidden");
    return;
  }
  const g = ui.result.ground_truth.by_scenario[ui.scenario];
  box.classList.remove("hidden");
  const line1 = el("p");
  line1.append(el("span", "ok", `✓ ${g.aciertos} aciertos`), " · ",
    el("span", "bad", `✗ ${g.errores} errores`), ` · ${g.por_verificar} por verificar`);
  const line2 = el("p");
  line2.append("Ocupado mostrado como libre: ", el("strong", "", String(g.ocupado_como_libre)),
    " · libre mostrado como ocupado: ", el("strong", "", String(g.libre_como_ocupado)),
    g.sin_etiqueta ? ` · sin etiqueta: ${g.sin_etiqueta}` : "");
  box.replaceChildren(el("h3", "", "Contra la etiqueta real"), line1, line2,
    el("p", "hint", ui.result.ground_truth.note));
}

function renderSlotDetail() {
  const box = $("slot-detail");
  const feedback = $("feedback");
  const slot = ui.result && ui.result.slots.find((s) => s.slot_id === ui.selected);
  if (!slot) {
    box.className = "hint";
    box.textContent = ui.result
      ? "Pasa el cursor sobre un espacio para ver la probabilidad; haz clic para seleccionarlo."
      : "Todavía no hay resultados.";
    feedback.classList.add("hidden");
    return;
  }
  const state = slot.states[ui.scenario];
  const tagClass = state === "libre" ? "free" : state === "ocupado" ? "busy" : "verify";
  const head = el("p");
  head.append(el("span", `tag ${tagClass}`, STATE_LABEL[state]), " Espacio ", el("strong", "", slot.slot_id));
  const prob = el("p");
  prob.append("P(ocupado) = ", el("strong", "", fmt(slot.p_busy, 3)));
  const parts = [head, prob];
  if (slot.ground_truth !== null && slot.ground_truth !== undefined) {
    parts.push(el("p", "", `Etiqueta real: ${slot.ground_truth === 1 ? "ocupado" : "libre"}`));
  }
  box.className = "";
  box.replaceChildren(...parts);
  feedback.classList.remove("hidden");
  $("fb-msg").textContent = "";
}

async function sendFeedback(reported) {
  const slot = ui.result && ui.result.slots.find((s) => s.slot_id === ui.selected);
  if (!slot) return;
  const source = ui.source && ui.source.type === "gallery" ? ui.source.item.id : "upload";
  try {
    const res = await api("/api/feedback", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        slot_id: slot.slot_id,
        reported_state: reported,
        predicted_state: slot.states[ui.scenario],
        p_busy: slot.p_busy,
        scenario: ui.scenario,
        source,
      }),
    });
    $("fb-msg").textContent = res.message;
  } catch (error) {
    $("fb-msg").textContent = `No se pudo registrar: ${error.message}`;
  }
}

// ---------------------------------------------------------------------------
// Modo "dibujar espacios"
// ---------------------------------------------------------------------------

function enterDrawMode(keepBoxes = false) {
  if (!photo.naturalWidth) return showBanner("Primero sube una imagen o elige una de la galería.", "warn");
  ui.drawMode = true;
  if (!keepBoxes) ui.boxes = [];
  ui.result = null;
  renderResult();
  canvas.classList.remove("hidden");
  $("draw-edit").classList.add("hidden");
  $("stage").classList.add("drawing");
  resizeCanvas();
}

function exitDrawMode() {
  ui.drawMode = false;
  canvas.classList.add("hidden");
  $("stage").classList.remove("drawing");
}

function resizeCanvas() {
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(photo.clientWidth * ratio);
  canvas.height = Math.round(photo.clientHeight * ratio);
  canvas.style.width = `${photo.clientWidth}px`;
  canvas.style.height = `${photo.clientHeight}px`;
  drawBoxes();
}

function toImageCoords(event) {
  const rect = canvas.getBoundingClientRect();
  const sx = photo.naturalWidth / rect.width;
  const sy = photo.naturalHeight / rect.height;
  const x = Math.min(Math.max(event.clientX - rect.left, 0), rect.width) * sx;
  const y = Math.min(Math.max(event.clientY - rect.top, 0), rect.height) * sy;
  return { x, y };
}

function drawBoxes() {
  const ctx = canvas.getContext("2d");
  const k = canvas.width / photo.naturalWidth;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 2 * (window.devicePixelRatio || 1);
  ctx.font = `${12 * (window.devicePixelRatio || 1)}px sans-serif`;
  const all = ui.dragBox ? [...ui.boxes, ui.dragBox] : ui.boxes;
  all.forEach((b, i) => {
    ctx.strokeStyle = "#38bdf8";
    ctx.fillStyle = "rgba(56, 189, 248, 0.15)";
    ctx.fillRect(b.x * k, b.y * k, b.w * k, b.h * k);
    ctx.strokeRect(b.x * k, b.y * k, b.w * k, b.h * k);
    ctx.fillStyle = "#0c4a6e";
    ctx.fillText(String(i + 1), b.x * k + 4, b.y * k + 14 * (window.devicePixelRatio || 1));
  });
  $("box-count").textContent = `${ui.boxes.length} espacio${ui.boxes.length === 1 ? "" : "s"} dibujado${ui.boxes.length === 1 ? "" : "s"}`;
}

function normBox(a, b) {
  return { x: Math.min(a.x, b.x), y: Math.min(a.y, b.y), w: Math.abs(a.x - b.x), h: Math.abs(a.y - b.y) };
}

function onPointerDown(event) {
  if (!ui.drawMode) return;
  canvas.setPointerCapture(event.pointerId);
  ui.dragStart = toImageCoords(event);
  ui.dragBox = null;
}

function onPointerMove(event) {
  if (!ui.drawMode || !ui.dragStart) return;
  ui.dragBox = normBox(ui.dragStart, toImageCoords(event));
  drawBoxes();
}

function onPointerUp(event) {
  if (!ui.drawMode || !ui.dragStart) return;
  const box = normBox(ui.dragStart, toImageCoords(event));
  ui.dragStart = null;
  ui.dragBox = null;
  if (box.w >= MIN_BOX && box.h >= MIN_BOX) {
    ui.boxes.push({ x: Math.round(box.x), y: Math.round(box.y), w: Math.round(box.w), h: Math.round(box.h) });
  }
  drawBoxes();
}

async function onCustomFile(event) {
  const file = event.target.files[0];
  if (!file) return;
  const url = URL.createObjectURL(file);
  setSource({ type: "upload", file, url });
  ui.result = null;
  try {
    await loadImage(url);
  } catch (error) {
    return showBanner(error.message, "warn");
  }
  hideBanner();
  $("stage").classList.remove("empty");
  $("viewer-title").textContent = "Dibuja los espacios sobre tu imagen";
  $("viewer-sub").textContent = file.name;
  enterDrawMode();
}

function useCurrentImage() {
  if (!ui.source) return showBanner("Primero elige un ejemplo de la galería o sube una imagen.", "warn");
  $("viewer-title").textContent = "Dibuja los espacios sobre la imagen actual";
  enterDrawMode();
}

function runCustom() {
  if (!ui.source || !photo.naturalWidth) return showBanner("Primero sube una imagen o elige una de la galería.", "warn");
  if (!ui.boxes.length) return showBanner("No hay espacios dibujados: arrastra el mouse sobre la imagen para marcar al menos uno.", "warn");
  const boxes = ui.boxes.map((b, i) => ({ ...b, slot_id: String(i + 1) }));
  let request;
  let imageUrl;
  if (ui.source.type === "gallery") {
    imageUrl = ui.source.item.image_url;
    request = () => api("/api/predict/custom", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ gallery_id: ui.source.item.id, boxes, scenario: ui.scenario }),
    });
  } else {
    imageUrl = ui.source.url;
    request = () => api("/api/predict/custom", {
      method: "POST",
      body: uploadForm(ui.source.file, { boxes: JSON.stringify(boxes) }),
    });
  }
  exitDrawMode();
  $("draw-edit").classList.remove("hidden");
  runPrediction(imageUrl, request, "Espacios dibujados", `${boxes.length} espacio(s)`);
}

// ---------------------------------------------------------------------------
// Pestañas y eventos
// ---------------------------------------------------------------------------

function setTab(tab) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-body").forEach((b) => b.classList.toggle("hidden", b.dataset.tab !== tab));
  if (tab !== "custom") exitDrawMode();
}

function bindEvents() {
  document.querySelectorAll(".scenario-btn").forEach((b) => b.addEventListener("click", () => setScenario(b.dataset.scenario)));
  document.querySelectorAll(".tab").forEach((b) => b.addEventListener("click", () => setTab(b.dataset.tab)));
  $("form-frame").addEventListener("submit", onSubmitFrame);
  $("form-patch").addEventListener("submit", onSubmitPatch);
  $("custom-file").addEventListener("change", onCustomFile);
  $("custom-use-current").addEventListener("click", useCurrentImage);
  $("draw-undo").addEventListener("click", () => { ui.boxes.pop(); drawBoxes(); });
  $("draw-clear").addEventListener("click", () => { ui.boxes = []; drawBoxes(); });
  $("draw-run").addEventListener("click", runCustom);
  $("draw-edit").addEventListener("click", () => enterDrawMode(true));
  $("gt-toggle").addEventListener("change", (e) => { ui.showGt = e.target.checked; renderResult(); });
  $("fb-busy").addEventListener("click", () => sendFeedback("ocupado"));
  $("fb-free").addEventListener("click", () => sendFeedback("libre"));
  canvas.addEventListener("pointerdown", onPointerDown);
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerup", onPointerUp);
  window.addEventListener("resize", () => { if (ui.drawMode) resizeCanvas(); });
}

bindEvents();
loadModel();
loadGallery();
