const API_KEY = "agribot.piUrl.v1";
const CACHE_KEY = "agribot.android.latestRun.v5";
const CONFIDENCE_THRESHOLD = 0.765;
const I18N = window.AgribotI18N;
const DEFAULT_PI_URL = "http://10.42.0.1:8080";
const FALLBACK_PI_URLS = [DEFAULT_PI_URL, "http://192.168.1.25:8080"];
const REQUEST_TIMEOUT_MS = 4000;

const els = {
  badge: document.getElementById("connectionBadge"),
  languageSelect: document.getElementById("languageSelect"),
  piUrl: document.getElementById("piUrl"),
  saveUrl: document.getElementById("saveUrl"),
  connectionError: document.getElementById("connectionError"),
  dashboardLink: document.getElementById("dashboardLink"),
  refresh: document.getElementById("refreshButton"),
  latestLabel: document.getElementById("latestLabel"),
  latestMeta: document.getElementById("latestMeta"),
  decisionCount: document.getElementById("decisionCount"),
  runTime: document.getElementById("runTime"),
  sickCount: document.getElementById("sickCount"),
  uncertainCount: document.getElementById("uncertainCount"),
  confidenceValue: document.getElementById("confidenceValue"),
  coverageNote: document.getElementById("coverageNote"),
  decisionRows: document.getElementById("decisionRows"),
  emptyRows: document.getElementById("emptyRows"),
  downloadLink: document.getElementById("downloadLink"),
  runStatus: document.getElementById("runStatus"),
  runId: document.getElementById("runId"),
  startedAt: document.getElementById("startedAt"),
  updatedAt: document.getElementById("updatedAt"),
  cameraSource: document.getElementById("cameraSource"),
  modelName: document.getElementById("modelName"),
  labelBars: document.getElementById("labelBars"),
  fieldName: document.getElementById("fieldName"),
  fieldLayoutForm: document.getElementById("fieldLayoutForm"),
  addField: document.getElementById("addFieldButton"),
  editField: document.getElementById("editFieldButton"),
  addRow: document.getElementById("addRowButton"),
  fieldSelector: document.getElementById("fieldSelector"),
  fieldModal: document.getElementById("fieldModal"),
  fieldModalTitle: document.getElementById("fieldModalTitle"),
  closeFieldModal: document.getElementById("closeFieldModal"),
  cancelFieldModal: document.getElementById("cancelFieldModal"),
  layoutFieldId: document.getElementById("layoutFieldId"),
  layoutRows: document.getElementById("layoutRows"),
  layoutPlants: document.getElementById("layoutPlants"),
  layoutRowSpacing: document.getElementById("layoutRowSpacing"),
  layoutPlantSpacing: document.getElementById("layoutPlantSpacing"),
  rowTableBody: document.getElementById("rowTableBody"),
  fieldDetailTitle: document.getElementById("fieldDetailTitle"),
  fieldDetailMeta: document.getElementById("fieldDetailMeta"),
  fieldDetailBadge: document.getElementById("fieldDetailBadge"),
  fieldTotalDetected: document.getElementById("fieldTotalDetected"),
  fieldSickDetected: document.getElementById("fieldSickDetected"),
  fieldUncertainDetected: document.getElementById("fieldUncertainDetected"),
  fieldDiseaseSummary: document.getElementById("fieldDiseaseSummary"),
  fieldIssueRows: document.getElementById("fieldIssueRows"),
  fieldDetailMap: document.getElementById("fieldDetailMap"),
  farmRobotMap: document.getElementById("farmRobotMap"),
  liveLocation: document.getElementById("liveLocation"),
  robotFieldBadge: document.getElementById("robotFieldBadge"),
  recentCatalogue: document.getElementById("recentCatalogue"),
};

const DEFAULT_FIELD = {
  id: "field_1",
  name: "Field 1",
  active_row_id: "A",
  row_count: 1,
  plants_per_row: 100,
  row_spacing_m: 1.0,
  plant_spacing_m: 0.5,
  start_plant: 1,
  plant_step: 1,
  plant_cooldown_sec: 2.0,
  rows: [],
};

let selectedFieldId = DEFAULT_FIELD.id;
let selectedRowId = DEFAULT_FIELD.active_row_id;
let currentFarmLayout = { active_field_id: DEFAULT_FIELD.id, fields: [{ ...DEFAULT_FIELD }] };
let lastRenderedData = null;
let layoutSaveInFlight = false;
let layoutSaveTimer = null;
let fieldSelectionTouched = false;
let fieldModalMode = "add";
let fieldModalFieldId = null;

function apiBase() {
  return (localStorage.getItem(API_KEY) || DEFAULT_PI_URL).replace(/\/+$/, "");
}

function saveApiBase(value) {
  const cleaned = String(value || "").trim().replace(/\/+$/, "");
  localStorage.setItem(API_KEY, cleaned || DEFAULT_PI_URL);
  if (els.piUrl) els.piUrl.value = apiBase();
  if (els.dashboardLink) els.dashboardLink.href = apiBase();
}

function candidateBases() {
  const seen = new Set();
  return [apiBase(), ...FALLBACK_PI_URLS].filter((item) => {
    const cleaned = item.replace(/\/+$/, "");
    if (seen.has(cleaned)) return false;
    seen.add(cleaned);
    return true;
  });
}

async function fetchWithTimeout(url, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), timeoutMs);
  try {
    return await fetch(url, { ...options, signal: controller.signal });
  } finally {
    clearTimeout(timeout);
  }
}

function firstSuccessfulBase(path, options = {}, timeoutMs = REQUEST_TIMEOUT_MS) {
  const bases = candidateBases();
  return new Promise((resolve, reject) => {
    let settled = false;
    let pending = bases.length;
    const errors = [];
    for (const base of bases) {
      fetchWithTimeout(`${base}${path}`, options, timeoutMs)
        .then((response) => {
          if (!response.ok) throw new Error(`${base} HTTP ${response.status}`);
          return response.json().then((payload) => ({ base, payload }));
        })
        .then((result) => {
          if (!settled) {
            settled = true;
            resolve(result);
          }
        })
        .catch((error) => {
          errors.push(error);
          pending -= 1;
          if (!settled && pending === 0) {
            reject(errors[0] || new Error("No Pi URL reachable"));
          }
        });
    }
  });
}

function fmtTime(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleString([], { dateStyle: "medium", timeStyle: "short" });
}

function fmtClock(value) {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
}

function pct(value) {
  if (!Number.isFinite(value)) return "--";
  return `${(value * 100).toFixed(1)}%`;
}

function numberValue(value, fallback, min, max) {
  const parsed = Number(value);
  if (!Number.isFinite(parsed)) return fallback;
  return Math.min(max, Math.max(min, parsed));
}

function intValue(value, fallback, min, max) {
  return Math.round(numberValue(value, fallback, min, max));
}

function normalizeLabel(label) {
  return String(label || "").toLowerCase().replace(/[_-]/g, " ").trim();
}

function isUncertain(row) {
  return row?.status === "uncertain" || normalizeLabel(row?.label) === "uncertain" || Number(row?.confidence) < CONFIDENCE_THRESHOLD;
}

function isSick(row) {
  const label = normalizeLabel(row?.label);
  return row?.status === "ok" && !isUncertain(row) && label && label !== "healthy";
}

function rowTime(row) {
  return row?.timestamp || row?.time || null;
}

function appendText(parent, tag, text) {
  const el = document.createElement(tag);
  el.textContent = text;
  parent.appendChild(el);
  return el;
}

function rowIdForIndex(index) {
  let number = index;
  let value = "";
  while (true) {
    const rem = number % 26;
    value = String.fromCharCode(65 + rem) + value;
    number = Math.floor(number / 26) - 1;
    if (number < 0) return value;
  }
}

function rowIds(count) {
  return Array.from({ length: Math.max(1, Number(count) || 1) }, (_, index) => rowIdForIndex(index));
}

function cleanFieldId(value, fallback) {
  const cleaned = String(value || "")
    .trim()
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return cleaned || fallback;
}

function normalizeField(raw = {}, index = 1) {
  const name = String(raw.name || raw.field_name || raw.field_id || `Field ${index}`).trim() || `Field ${index}`;
  const rowCount = intValue(raw.row_count, DEFAULT_FIELD.row_count, 1, 100);
  const defaultPlants = intValue(raw.plants_per_row, DEFAULT_FIELD.plants_per_row, 1, 1000);
  const defaultPlantSpacing = numberValue(raw.plant_spacing_m, DEFAULT_FIELD.plant_spacing_m, 0, 1000);
  const defaultRowSpacing = numberValue(raw.row_spacing_m, DEFAULT_FIELD.row_spacing_m, 0, 1000);
  const ids = rowIds(rowCount);
  const rowsById = new Map((raw.rows || []).map((row) => [String(row.id || row.row_id || "").toUpperCase(), row]));
  let y = 0;
  const rows = ids.map((rowId, rowIndex) => {
    const row = rowsById.get(rowId) || {};
    const rowSpacing = numberValue(row.row_spacing_m, defaultRowSpacing, 0, 1000);
    const normalized = {
      id: rowId,
      row_id: rowId,
      row_index: rowIndex + 1,
      plants_per_row: intValue(row.plants_per_row, defaultPlants, 1, 1000),
      plant_spacing_m: numberValue(row.plant_spacing_m, defaultPlantSpacing, 0, 1000),
      row_spacing_m: rowSpacing,
      y_m: numberValue(row.y_m, y, 0, 100000),
    };
    y += rowSpacing;
    return normalized;
  });
  const activeRow = ids.includes(String(raw.active_row_id || raw.row_id || "").toUpperCase())
    ? String(raw.active_row_id || raw.row_id).toUpperCase()
    : ids[0];
  return {
    id: cleanFieldId(raw.id || raw.active_field_id || name, `field_${index}`),
    name,
    field_id: name,
    active_row_id: activeRow,
    row_count: rowCount,
    plants_per_row: defaultPlants,
    row_spacing_m: defaultRowSpacing,
    plant_spacing_m: defaultPlantSpacing,
    start_plant: intValue(raw.start_plant, DEFAULT_FIELD.start_plant, 1, 100000),
    plant_step: intValue(raw.plant_step, DEFAULT_FIELD.plant_step, 1, 1000),
    plant_cooldown_sec: numberValue(raw.plant_cooldown_sec, DEFAULT_FIELD.plant_cooldown_sec, 0, 120),
    rows,
  };
}

function normalizeFarmLayout(layout = {}) {
  const sourceFields = Array.isArray(layout.fields) && layout.fields.length ? layout.fields : [layout];
  const used = new Set();
  const fields = sourceFields.map((raw, index) => {
    const field = normalizeField(raw, index + 1);
    const base = field.id;
    let suffix = 2;
    while (used.has(field.id)) {
      field.id = `${base}_${suffix}`;
      suffix += 1;
    }
    used.add(field.id);
    return field;
  });
  const activeId = used.has(layout.active_field_id) ? layout.active_field_id : fields[0].id;
  return { active_field_id: activeId, fields, updated_at: layout.updated_at || new Date().toISOString() };
}

function selectedField() {
  return currentFarmLayout.fields.find((field) => field.id === selectedFieldId) || currentFarmLayout.fields[0];
}

function selectedRow(field = selectedField()) {
  return field.rows.find((row) => row.id === selectedRowId) || field.rows[0];
}

function ensureSelectedRow(field = selectedField()) {
  if (field.active_row_id && field.rows.some((row) => row.id === field.active_row_id)) {
    selectedRowId = field.active_row_id;
  } else if (!field.rows.some((row) => row.id === selectedRowId)) {
    selectedRowId = field.active_row_id && field.rows.some((row) => row.id === field.active_row_id) ? field.active_row_id : field.rows[0]?.id || "A";
  }
  return selectedRowId;
}

function plantName(row) {
  if (I18N) return I18N.plantName(row);
  return row?.plant_display || `${row?.field_id || ""} / Row ${row?.row_id || "A"} / Plant ${row?.plant_number ?? row?.plant_id ?? "--"}`;
}

function locationText(row) {
  if (!row) return "--";
  return `${row.field_id || selectedField()?.name || "Field"} / Row ${row.row_id || "A"} / Plant ${row.plant_number ?? row.plant_id ?? row.sequence ?? "--"}`;
}

function plantKeyFor(field, rowId, plantNumber) {
  return `${field.name}|${rowId}|${plantNumber}`;
}

function latestByPlant(decisions, field) {
  const latest = new Map();
  for (const row of decisions) {
    if ((row.field_id || field.name) !== field.name) continue;
    const key = row.plant_key || plantKeyFor(field, row.row_id || "A", row.plant_number ?? row.plant_id ?? row.sequence);
    latest.set(key, row);
  }
  return latest;
}

function fieldRows(decisions, field) {
  return Array.from(latestByPlant(decisions, field).values());
}

function diseaseSummary(rows) {
  const byDisease = new Map();
  for (const row of rows.filter(isSick)) {
    const label = row.label || "Unknown";
    const entry = byDisease.get(label) || { label, count: 0, best: null };
    entry.count += 1;
    if (!entry.best || Number(row.confidence) > Number(entry.best.confidence)) entry.best = row;
    byDisease.set(label, entry);
  }
  return Array.from(byDisease.values()).sort((a, b) => b.count - a.count);
}

async function fetchLatest() {
  const { base, payload } = await firstSuccessfulBase("/api/runs/latest", { cache: "no-store" });
  localStorage.setItem(API_KEY, base);
  localStorage.setItem(CACHE_KEY, JSON.stringify({ savedAt: new Date().toISOString(), payload }));
  return { payload, cached: false };
}

async function saveFarmLayout(layout) {
  const { payload } = await firstSuccessfulBase("/api/farm-layout", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(layout),
  }, 3000);
  return payload;
}

function cachedLatest() {
  const raw = localStorage.getItem(CACHE_KEY);
  if (!raw) return null;
  try {
    const parsed = JSON.parse(raw);
    return { payload: parsed.payload, cached: true, savedAt: parsed.savedAt };
  } catch {
    return null;
  }
}

function setBadge(text, online) {
  els.badge.textContent = I18N ? I18N.t(text) : text;
  els.badge.style.background = online ? "rgba(185, 232, 208, 0.24)" : "rgba(255, 213, 128, 0.24)";
}

function populateFieldControls(field) {
  ensureSelectedRow(field);
  els.fieldSelector.replaceChildren();
  for (const item of currentFarmLayout.fields) {
    const option = document.createElement("option");
    option.value = item.id;
    option.textContent = item.name;
    els.fieldSelector.appendChild(option);
  }
  els.fieldSelector.value = field.id;
  els.fieldName.textContent = field.name;
}

function populateFieldModal(field) {
  els.layoutFieldId.value = field.name;
  els.layoutRows.value = field.row_count;
  els.layoutPlants.value = field.plants_per_row;
  els.layoutRowSpacing.value = field.row_spacing_m;
  els.layoutPlantSpacing.value = field.plant_spacing_m;
}

function collectFieldForm(existing = selectedField()) {
  const rowCount = intValue(els.layoutRows.value, existing.row_count, 1, 100);
  const base = {
    ...existing,
    name: els.layoutFieldId.value,
    field_id: els.layoutFieldId.value,
    row_count: rowCount,
    plants_per_row: els.layoutPlants.value,
    row_spacing_m: els.layoutRowSpacing.value,
    plant_spacing_m: els.layoutPlantSpacing.value,
    active_row_id: existing.rows.some((row) => row.id === existing.active_row_id) ? existing.active_row_id : "A",
    start_plant: existing.start_plant || DEFAULT_FIELD.start_plant,
    plant_step: existing.plant_step || DEFAULT_FIELD.plant_step,
    rows: collectRowsFromTable(existing),
  };
  return normalizeField(base, currentFarmLayout.fields.findIndex((field) => field.id === existing.id) + 1 || 1);
}

function collectRowsFromTable(field) {
  return field.rows.map((row) => {
    const plants = document.querySelector(`[data-row-plants="${row.id}"]`);
    const spacing = document.querySelector(`[data-row-spacing="${row.id}"]`);
    const rowGap = document.querySelector(`[data-row-gap="${row.id}"]`);
    return {
      ...row,
      plants_per_row: plants ? plants.value : row.plants_per_row,
      plant_spacing_m: spacing ? spacing.value : row.plant_spacing_m,
      row_spacing_m: rowGap ? rowGap.value : row.row_spacing_m,
    };
  });
}

function collectFarmLayoutFromRows() {
  const field = selectedField();
  const updated = normalizeField({ ...field, rows: collectRowsFromTable(field) }, currentFarmLayout.fields.findIndex((item) => item.id === field.id) + 1 || 1);
  const fields = currentFarmLayout.fields.map((field) => (field.id === selectedFieldId ? updated : field));
  selectedFieldId = updated.id;
  return normalizeFarmLayout({ active_field_id: selectedFieldId, fields });
}

function collectFarmLayoutFromModal() {
  const existing =
    fieldModalMode === "add"
      ? normalizeField({ ...DEFAULT_FIELD, id: `field_${currentFarmLayout.fields.length + 1}`, name: els.layoutFieldId.value || `Field ${currentFarmLayout.fields.length + 1}` }, currentFarmLayout.fields.length + 1)
      : currentFarmLayout.fields.find((field) => field.id === fieldModalFieldId) || selectedField();
  const updated = collectFieldForm(existing);
  const fields =
    fieldModalMode === "add"
      ? [...currentFarmLayout.fields, updated]
      : currentFarmLayout.fields.map((field) => (field.id === existing.id ? updated : field));
  selectedFieldId = updated.id;
  fieldSelectionTouched = true;
  selectedRowId = updated.active_row_id;
  return normalizeFarmLayout({ active_field_id: updated.id, fields });
}

function openFieldModal(mode) {
  fieldModalMode = mode;
  const next = currentFarmLayout.fields.length + 1;
  const draft =
    mode === "add"
      ? normalizeField({ ...DEFAULT_FIELD, id: `field_${next}`, name: `Field ${next}` }, next)
      : selectedField();
  fieldModalFieldId = mode === "add" ? null : draft.id;
  els.fieldModalTitle.textContent = I18N.t(mode === "add" ? "addField" : "editField");
  populateFieldModal(draft);
  els.fieldModal.hidden = false;
  document.body.classList.add("modal-open");
  requestAnimationFrame(() => els.layoutFieldId.focus());
}

function closeFieldModal() {
  els.fieldModal.hidden = true;
  document.body.classList.remove("modal-open");
}

async function persistFarmLayout(layout, { refreshAfter = false } = {}) {
  layoutSaveInFlight = true;
  try {
    const result = await saveFarmLayout(layout);
    currentFarmLayout = normalizeFarmLayout(result.farm_layout);
    if (!currentFarmLayout.fields.some((field) => field.id === selectedFieldId)) {
      selectedFieldId = currentFarmLayout.active_field_id || currentFarmLayout.fields[0].id;
    }
    ensureSelectedRow(selectedField());
    setBadge("saved", true);
    if (refreshAfter) {
      await refresh();
    } else if (lastRenderedData) {
      render({
        payload: {
          ...lastRenderedData.payload,
          farm_layout: currentFarmLayout,
          field_layout: selectedField(),
        },
        cached: lastRenderedData.cached,
      });
    }
    return result;
  } catch (error) {
    setBadge("saveFailed", false);
    throw error;
  } finally {
    layoutSaveInFlight = false;
  }
}

function scheduleFarmLayoutSave() {
  clearTimeout(layoutSaveTimer);
  layoutSaveTimer = setTimeout(() => {
    try {
      const layout = collectFarmLayoutFromRows();
      persistFarmLayout(layout).catch(() => {});
    } catch {
      setBadge("saveFailed", false);
    }
  }, 450);
}

function renderRowsTable(decisions, field) {
  els.rowTableBody.replaceChildren();
  const detected = fieldRows(decisions, field);
  for (const rowSpec of field.rows) {
    const rowDecisions = detected.filter((row) => String(row.row_id || "A").toUpperCase() === rowSpec.id);
    const tr = document.createElement("tr");
    appendText(tr, "td", I18N.t("rowShort", { row: rowSpec.id }));

    const plantsCell = document.createElement("td");
    const plants = document.createElement("input");
    plants.type = "number";
    plants.min = "1";
    plants.max = "1000";
    plants.value = rowSpec.plants_per_row;
    plants.dataset.rowPlants = rowSpec.id;
    plantsCell.appendChild(plants);

    const spacingCell = document.createElement("td");
    const spacing = document.createElement("input");
    spacing.type = "number";
    spacing.min = "0";
    spacing.max = "1000";
    spacing.step = "0.01";
    spacing.value = rowSpec.plant_spacing_m;
    spacing.dataset.rowSpacing = rowSpec.id;
    spacingCell.appendChild(spacing);

    const gapCell = document.createElement("td");
    const gap = document.createElement("input");
    gap.type = "number";
    gap.min = "0";
    gap.max = "1000";
    gap.step = "0.01";
    gap.value = rowSpec.row_spacing_m;
    gap.dataset.rowGap = rowSpec.id;
    gapCell.appendChild(gap);

    tr.append(plantsCell, spacingCell, gapCell);
    appendText(tr, "td", rowDecisions.length);
    appendText(tr, "td", rowDecisions.filter(isSick).length);
    appendText(tr, "td", rowDecisions.filter(isUncertain).length);
    els.rowTableBody.appendChild(tr);
  }
}

function renderOverview(payload, data) {
  const summary = payload.summary || {};
  const metadata = payload.metadata || {};
  const decisions = payload.decisions || [];
  const latest = summary.latest_decision || decisions[decisions.length - 1] || null;
  els.latestLabel.textContent = latest ? I18N.label(latest.label) : I18N.t("noData");
  els.latestMeta.textContent = latest ? `${locationText(latest)} | ${fmtClock(rowTime(latest))}` : I18N.t("waitingForInference");
  els.decisionCount.textContent = summary.decisions ?? decisions.length ?? 0;
  els.runTime.textContent = summary.completed ? I18N.t("completed") : I18N.t("running");
  els.sickCount.textContent = summary.sick ?? decisions.filter(isSick).length;
  els.uncertainCount.textContent = I18N.t("uncertainCount", { count: summary.uncertain ?? decisions.filter(isUncertain).length });
  els.confidenceValue.textContent = latest ? pct(Number(latest.confidence)) : "--";
  els.coverageNote.textContent = data.cached ? I18N.t("phoneCache") : I18N.t("liveFromPi");
  els.runStatus.textContent = summary.completed ? I18N.t("completed") : I18N.t("live");
  els.runId.textContent = payload.run_id || summary.run_id || "--";
  els.startedAt.textContent = fmtTime(summary.started_at || metadata.started_at);
  els.updatedAt.textContent = fmtTime(summary.last_updated_at);
  els.cameraSource.textContent = metadata.camera?.source || "--";
  els.modelName.textContent = metadata.model?.classifier || "--";
  els.downloadLink.href = payload.csv_url ? (payload.csv_url.startsWith("http") ? payload.csv_url : `${apiBase()}${payload.csv_url}`) : "#";
  if (els.dashboardLink) els.dashboardLink.href = apiBase();
}

function renderFieldDetail(decisions) {
  const field = selectedField();
  const rows = fieldRows(decisions, field);
  const sickRows = rows.filter(isSick).sort((a, b) => String(a.row_id).localeCompare(String(b.row_id)) || Number(a.plant_number) - Number(b.plant_number));
  const uncertainRows = rows.filter(isUncertain).sort((a, b) => String(a.row_id).localeCompare(String(b.row_id)) || Number(a.plant_number) - Number(b.plant_number));
  els.fieldName.textContent = field.name;
  els.fieldDetailTitle.textContent = field.name;
  els.fieldDetailMeta.textContent = I18N.t("mappedPlants", {
    rows: field.row_count,
    plants: field.rows.reduce((sum, row) => sum + Number(row.plants_per_row), 0),
  });
  els.fieldDetailBadge.textContent = field.id === currentFarmLayout.active_field_id ? I18N.t("activeField") : I18N.t("savedField");
  els.fieldTotalDetected.textContent = rows.length;
  els.fieldSickDetected.textContent = sickRows.length;
  els.fieldUncertainDetected.textContent = uncertainRows.length;

  els.fieldDiseaseSummary.replaceChildren();
  const diseases = diseaseSummary(rows);
  if (!diseases.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = I18N.t("noDiseasesField");
    els.fieldDiseaseSummary.appendChild(empty);
  } else {
    for (const disease of diseases) {
      const item = document.createElement("div");
      item.className = "disease-card";
      appendText(item, "strong", I18N.label(disease.label));
      appendText(item, "span", I18N.t("plantCount", { count: disease.count }));
      appendText(item, "small", I18N.t("mostProbable", { location: locationText(disease.best), confidence: pct(Number(disease.best?.confidence)) }));
      els.fieldDiseaseSummary.appendChild(item);
    }
  }

  els.fieldIssueRows.replaceChildren();
  const max = Math.max(sickRows.length, uncertainRows.length, 1);
  for (let index = 0; index < max; index += 1) {
    const tr = document.createElement("tr");
    appendText(tr, "td", sickRows[index] ? `${locationText(sickRows[index])} - ${I18N.label(sickRows[index].label)}` : "--");
    appendText(tr, "td", uncertainRows[index] ? `${locationText(uncertainRows[index])} - ${pct(Number(uncertainRows[index].confidence))}` : "--");
    els.fieldIssueRows.appendChild(tr);
  }

  renderFieldMap(els.fieldDetailMap, field, rows);
}

function renderFieldMap(container, field, rows) {
  container.replaceChildren();
  const latest = latestByPlant(rows, field);
  for (const rowSpec of field.rows) {
    const rowBand = document.createElement("div");
    rowBand.className = "map-row";
    appendText(rowBand, "span", rowSpec.id);
    const rowTiles = document.createElement("div");
    rowTiles.className = "map-row-tiles";
    for (let index = 0; index < rowSpec.plants_per_row; index += 1) {
      const plantNumber = field.start_plant + index * field.plant_step;
      const row = latest.get(plantKeyFor(field, rowSpec.id, plantNumber));
      const tile = document.createElement("button");
      tile.type = "button";
      tile.className = `plant-tile ${row ? (isUncertain(row) ? "uncertain" : isSick(row) ? "sick" : "healthy") : "pending"}`;
      tile.title = row ? `${locationText(row)}: ${I18N.label(row.label)} (${pct(Number(row.confidence))})` : `${field.name} / Row ${rowSpec.id} / Plant ${plantNumber}`;
      appendText(tile, "strong", `${rowSpec.id}-${plantNumber}`);
      appendText(tile, "span", row?.label ? I18N.label(row.label) : I18N.t("notScanned"));
      rowTiles.appendChild(tile);
    }
    rowBand.appendChild(rowTiles);
    container.appendChild(rowBand);
  }
}

function renderFarmMap(decisions) {
  els.farmRobotMap.replaceChildren();
  const latest = decisions[decisions.length - 1];
  els.liveLocation.textContent = latest
    ? I18N.t("robotAt", { location: locationText(latest), label: I18N.label(latest.label) })
    : I18N.t("waitingLivePosition");
  els.robotFieldBadge.textContent = latest?.field_id || I18N.t("robot");
  for (const field of currentFarmLayout.fields) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = `farm-field ${field.id === selectedFieldId ? "selected" : ""} ${latest?.field_id === field.name ? "robot-here" : ""}`;
    card.dataset.fieldId = field.id;
    appendText(card, "strong", field.name);
    appendText(card, "span", I18N.t("rowCount", { count: field.row_count }));
    if (latest?.field_id === field.name) {
      appendText(card, "small", I18N.t("robotPlant", { row: latest.row_id, plant: latest.plant_number }));
    }
    els.farmRobotMap.appendChild(card);
  }
}

function renderCatalogue(decisions) {
  els.recentCatalogue.replaceChildren();
  const recent = decisions.slice(-24).reverse();
  for (const row of recent) {
    const item = document.createElement("div");
    item.className = `catalogue-card ${isUncertain(row) ? "uncertain" : isSick(row) ? "sick" : "healthy"}`;
    appendText(item, "strong", locationText(row));
    appendText(item, "span", I18N.label(row.label));
    appendText(item, "small", `${pct(Number(row.confidence))} / ${fmtClock(rowTime(row))}`);
    els.recentCatalogue.appendChild(item);
  }
}

function renderRows(rows) {
  els.decisionRows.replaceChildren();
  if (!rows.length) {
    els.decisionRows.appendChild(els.emptyRows.content.cloneNode(true));
    return;
  }
  for (const row of rows) {
    const tr = document.createElement("tr");
    const statusClass = row.status === "ok" ? "status-ok" : "status-uncertain";
    appendText(tr, "td", fmtClock(rowTime(row)));
    appendText(tr, "td", locationText(row));
    appendText(tr, "td", row.label ? I18N.label(row.label) : "--");
    appendText(tr, "td", pct(Number(row.confidence)));
    appendText(tr, "td", I18N.action(row.action || (isSick(row) ? "Inspect or treat" : isUncertain(row) ? "Rescan this plant" : "No action"), row));
    const status = appendText(tr, "td", I18N.status(row.status));
    status.className = statusClass;
    els.decisionRows.appendChild(tr);
  }
}

function renderLabels(labels) {
  els.labelBars.replaceChildren();
  const entries = Object.entries(labels).sort((a, b) => b[1] - a[1]);
  if (!entries.length) {
    const empty = document.createElement("p");
    empty.className = "empty";
    empty.textContent = I18N.t("noLabels");
    els.labelBars.appendChild(empty);
    return;
  }
  const max = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
  for (const [label, value] of entries) {
    const row = document.createElement("div");
    row.className = "label-row";
    const width = Math.max(3, (Number(value) / max) * 100);
    appendText(row, "span", I18N.label(label));
    const bar = document.createElement("span");
    bar.className = "bar";
    const fill = document.createElement("span");
    fill.style.width = `${width}%`;
    bar.appendChild(fill);
    row.appendChild(bar);
    appendText(row, "strong", value);
    els.labelBars.appendChild(row);
  }
}

function render(data) {
  lastRenderedData = data;
  const payload = data.payload || {};
  const decisions = payload.decisions || [];
  currentFarmLayout = normalizeFarmLayout(payload.farm_layout || payload.field_layout || {});
  if (!fieldSelectionTouched || !currentFarmLayout.fields.some((field) => field.id === selectedFieldId)) {
    selectedFieldId = currentFarmLayout.active_field_id || currentFarmLayout.fields[0].id;
  }
  ensureSelectedRow(selectedField());
  renderOverview(payload, data);
  populateFieldControls(selectedField());
  renderRowsTable(decisions, selectedField());
  renderFieldDetail(decisions);
  renderFarmMap(decisions);
  renderCatalogue(decisions);
  renderRows(decisions.slice(-80).reverse());
  renderLabels(payload.summary?.labels || {});
}

async function refresh() {
  if (layoutSaveInFlight) return;
  try {
    const data = await fetchLatest();
    setBadge("live", true);
    if (els.connectionError) els.connectionError.textContent = "";
    if (els.piUrl) els.piUrl.value = apiBase();
    if (els.dashboardLink) els.dashboardLink.href = apiBase();
    render(data);
  } catch (error) {
    const cached = cachedLatest();
    if (cached) {
      setBadge("cached", false);
      if (els.connectionError) els.connectionError.textContent = `${I18N.t("usingPhoneCache")} ${apiBase()}`;
      render(cached);
    } else {
      setBadge("offline", false);
      if (els.connectionError) els.connectionError.textContent = `${apiBase()} ${I18N.t("connectHint")}`;
    }
  }
}

if (els.saveUrl) {
  els.saveUrl.addEventListener("click", () => {
    saveApiBase(els.piUrl.value);
    refresh();
  });
}
els.refresh.addEventListener("click", refresh);
els.addField.addEventListener("click", () => openFieldModal("add"));
els.editField.addEventListener("click", () => openFieldModal("edit"));
els.closeFieldModal.addEventListener("click", closeFieldModal);
els.cancelFieldModal.addEventListener("click", closeFieldModal);
els.fieldModal.addEventListener("click", (event) => {
  if (event.target === els.fieldModal) closeFieldModal();
});
window.addEventListener("keydown", (event) => {
  if (event.key === "Escape" && !els.fieldModal.hidden) closeFieldModal();
});
els.addRow.addEventListener("click", () => {
  try {
    const field = selectedField();
    const updated = normalizeField(
      {
        ...field,
        row_count: Number(field.row_count) + 1,
        rows: collectRowsFromTable(field),
      },
      currentFarmLayout.fields.findIndex((item) => item.id === field.id) + 1 || 1,
    );
    currentFarmLayout = normalizeFarmLayout({
      active_field_id: selectedFieldId,
      fields: currentFarmLayout.fields.map((item) => (item.id === field.id ? updated : item)),
    });
    render({ payload: { ...lastRenderedData?.payload, farm_layout: currentFarmLayout, decisions: lastRenderedData?.payload?.decisions || [] }, cached: true });
    persistFarmLayout(currentFarmLayout).catch(() => {});
  } catch {
    setBadge("saveFailed", false);
  }
});
els.fieldSelector.addEventListener("change", () => {
  try {
    const layout = collectFarmLayoutFromRows();
    selectedFieldId = els.fieldSelector.value;
    fieldSelectionTouched = true;
    layout.active_field_id = selectedFieldId;
    currentFarmLayout = normalizeFarmLayout(layout);
    selectedRowId = selectedField().active_row_id;
    render({ payload: { ...lastRenderedData?.payload, farm_layout: currentFarmLayout, decisions: lastRenderedData?.payload?.decisions || [] }, cached: true });
    persistFarmLayout(currentFarmLayout).catch(() => {});
  } catch {
    setBadge("saveFailed", false);
  }
});
els.farmRobotMap.addEventListener("click", (event) => {
  const target = event.target.closest("[data-field-id]");
  if (!target) return;
  currentFarmLayout = collectFarmLayoutFromRows();
  selectedFieldId = target.dataset.fieldId;
  fieldSelectionTouched = true;
  currentFarmLayout.active_field_id = selectedFieldId;
  selectedRowId = selectedField().active_row_id;
  if (lastRenderedData) render(lastRenderedData);
  persistFarmLayout(currentFarmLayout).catch(() => {});
});
els.rowTableBody.addEventListener("input", scheduleFarmLayoutSave);
els.rowTableBody.addEventListener("change", scheduleFarmLayoutSave);
els.fieldLayoutForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  try {
    const layout = collectFarmLayoutFromModal();
    closeFieldModal();
    await persistFarmLayout(layout, { refreshAfter: true });
    if (els.connectionError) els.connectionError.textContent = "";
  } catch {
    setBadge("saveFailed", false);
    if (els.connectionError) els.connectionError.textContent = `Could not save to ${apiBase()}.`;
  }
});

window.addEventListener("load", () => {
  I18N.init(els.languageSelect, () => {
    if (lastRenderedData) render(lastRenderedData);
  });
  if (els.piUrl) els.piUrl.value = apiBase();
  if (els.dashboardLink) els.dashboardLink.href = apiBase();
  const cached = cachedLatest();
  if (cached) render(cached);
  refresh();
  setInterval(refresh, 5000);
});
