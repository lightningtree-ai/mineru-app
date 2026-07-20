/* mineru-app frontend: staging + upload, SSE queue, library, viewer. */

const $ = (sel) => document.querySelector(sel);

let SUPPORTED = [".pdf", ".docx", ".pptx", ".xlsx", ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff", ".jp2"];
let PLATFORM = "";               // darwin | win32 | linux, from /api/meta
let staged = [];                 // File objects waiting to be processed
const jobs = new Map();          // job id -> job
let docs = [];                   // library entries (newest first)
let currentDoc = null;
let pdfjsModule = null;          // lazy-loaded ESM
const selectedIds = new Set();   // export selection

/* ---------------- boot ---------------- */

async function boot() {
  try {
    const meta = await (await fetch("/api/meta")).json();
    SUPPORTED = meta.supported_suffixes.map((s) => s.toLowerCase());
    PLATFORM = meta.platform;
    setDevice(meta.device);
    $("#viewer-reveal").textContent = revealLabel();
  } catch { /* defaults are fine */ }
  await refreshJobs();
  await refreshLibrary();
  connectEvents();
  restoreOptions();
  setInterval(tickElapsed, 1000);
  // Coming back to the tab (e.g. after the machine slept mid-job) is exactly
  // when a stale "still running" line would be visible — reconcile on wake, in
  // case the SSE socket was half-open and never fired onerror.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible") reconcile();
  });
}

function setDevice(device) {
  const badge = $("#device-badge");
  if (device) {
    badge.textContent = `device: ${device}`;
    badge.hidden = false;
  }
}

function revealLabel() {
  if (PLATFORM === "darwin") return "Reveal in Finder";
  if (PLATFORM === "win32") return "Show in Explorer";
  return "Show in file manager";
}

async function revealDoc(docId) {
  await fetch(`/api/docs/${docId}/reveal`, { method: "POST" });
}

/* ---------------- drag & drop / staging ---------------- */

const dropzone = $("#dropzone");

["dragenter", "dragover"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach((ev) =>
  dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));

dropzone.addEventListener("drop", async (e) => {
  const files = await filesFromDataTransfer(e.dataTransfer);
  stageFiles(files);
});
dropzone.addEventListener("click", () => $("#file-input").click());
$("#pick-files").addEventListener("click", (e) => { e.stopPropagation(); $("#file-input").click(); });
$("#pick-folder").addEventListener("click", (e) => { e.stopPropagation(); $("#folder-input").click(); });
$("#file-input").addEventListener("change", (e) => stageFiles([...e.target.files]));
$("#folder-input").addEventListener("change", (e) => stageFiles([...e.target.files]));

async function filesFromDataTransfer(dt) {
  // Capture entries synchronously — DataTransferItems go stale after an await.
  const entries = [...dt.items]
    .filter((i) => i.kind === "file")
    .map((i) => (i.webkitGetAsEntry ? i.webkitGetAsEntry() : null));
  if (!entries.some(Boolean)) return [...dt.files];

  const files = [];
  async function walk(entry) {
    if (!entry) return;
    if (entry.isFile) {
      files.push(await new Promise((res, rej) => entry.file(res, rej)));
    } else if (entry.isDirectory) {
      const reader = entry.createReader();
      let batch;
      do {
        batch = await new Promise((res, rej) => reader.readEntries(res, rej));
        for (const e of batch) await walk(e);
      } while (batch.length);
    }
  }
  for (const entry of entries) await walk(entry);
  return files;
}

function isSupported(name) {
  const dot = name.lastIndexOf(".");
  return dot >= 0 && SUPPORTED.includes(name.slice(dot).toLowerCase());
}

function stageFiles(files) {
  const ok = files.filter((f) => isSupported(f.name));
  const seen = new Set(staged.map((f) => f.name + ":" + f.size));
  for (const f of ok) {
    const key = f.name + ":" + f.size;
    if (!seen.has(key)) { staged.push(f); seen.add(key); }
  }
  renderStaged(files.length - ok.length);
}

function renderStaged(skipped = 0) {
  const section = $("#staged");
  section.hidden = staged.length === 0;
  $("#staged-count").textContent =
    `${staged.length} file${staged.length === 1 ? "" : "s"} staged` +
    (skipped > 0 ? ` (${skipped} unsupported skipped)` : "");
  const ul = $("#staged-list");
  ul.innerHTML = "";
  for (const f of staged) {
    const li = document.createElement("li");
    li.textContent = f.name;
    ul.appendChild(li);
  }
  $("#process-btn").textContent = `Process ${staged.length} file${staged.length === 1 ? "" : "s"}`;
}

$("#staged-clear").addEventListener("click", () => { staged = []; renderStaged(); });

/* ---------------- options ---------------- */

const OPT_IDS = ["opt-lang", "opt-backend", "opt-method", "opt-device", "opt-formula", "opt-table", "opt-start", "opt-end"];

function gatherOptions() {
  const num = (v) => (v === "" ? null : Number(v));
  return {
    lang: $("#opt-lang").value,
    backend: $("#opt-backend").value,
    method: $("#opt-method").value,
    device: $("#opt-device").value,
    formula: $("#opt-formula").checked,
    table: $("#opt-table").checked,
    start_page: num($("#opt-start").value),
    end_page: num($("#opt-end").value),
  };
}

function restoreOptions() {
  let saved;
  try { saved = JSON.parse(localStorage.getItem("mineru-options") || "null"); } catch { return; }
  if (!saved) return;
  for (const id of OPT_IDS) {
    const el = document.getElementById(id);
    if (!(id in saved)) continue;
    if (el.type === "checkbox") el.checked = saved[id]; else el.value = saved[id];
  }
  $("#backend-warning").hidden = $("#opt-backend").value !== "vlm-transformers";
}

function persistOptions() {
  const out = {};
  for (const id of OPT_IDS) {
    const el = document.getElementById(id);
    out[id] = el.type === "checkbox" ? el.checked : el.value;
  }
  localStorage.setItem("mineru-options", JSON.stringify(out));
}

/* ---------------- upload / process ---------------- */

$("#process-btn").addEventListener("click", async () => {
  if (!staged.length) return;
  const btn = $("#process-btn");
  btn.disabled = true;
  btn.textContent = "Uploading…";
  persistOptions();
  try {
    const form = new FormData();
    for (const f of staged) form.append("files", f, f.name);
    form.append("options", JSON.stringify(gatherOptions()));
    const resp = await fetch("/api/jobs", { method: "POST", body: form });
    if (!resp.ok) throw new Error((await resp.json()).detail || resp.statusText);
    staged = [];
    renderStaged();
    await refreshJobs();
  } catch (e) {
    alert(`Upload failed: ${e.message}`);
  } finally {
    btn.disabled = false;
    renderStaged();
  }
});

/* ---------------- queue + SSE ---------------- */

function connectEvents() {
  const es = new EventSource("/api/events");
  let reconnecting = false;  // true after a drop, until the stream comes back
  es.onopen = () => {
    // The event bus has no replay, so job "done"/"doc" events that fired while
    // we were disconnected are gone. Re-sync from the REST endpoints so a job
    // that finished during the outage doesn't stay frozen on its last progress
    // line. Skip the initial connect — boot() already fetched both.
    if (reconnecting) {
      reconnecting = false;
      reconcile();
    }
  };
  es.onmessage = (e) => {
    const event = JSON.parse(e.data);
    if (event.type === "hello") setDevice(event.device);
    if (event.type === "job") {
      event.job._receivedAt = Date.now();
      jobs.set(event.job.id, event.job);
      if (event.job.device) setDevice(event.job.device);
      renderQueue();
    }
    if (event.type === "doc") {
      refreshLibrary().then(() => { if (!currentDoc) openDoc(event.doc.id); });
    }
  };
  es.onerror = () => { reconnecting = true; /* EventSource auto-reconnects */ };
}

// Re-sync queue + library from the server's authoritative state. Cheap and
// idempotent; safe to call on any reconnect or tab wake.
function reconcile() {
  refreshJobs();
  refreshLibrary();
}

async function refreshJobs() {
  const data = await (await fetch("/api/jobs")).json();
  jobs.clear();
  for (const j of data.jobs) {
    j._receivedAt = Date.now();
    jobs.set(j.id, j);
  }
  setDevice(data.device);
  renderQueue();
}

function elapsed(job) {
  if (!job.started_at) return "";
  const end = job.finished_at || Date.now() / 1000;
  const s = Math.max(0, Math.round(end - job.started_at));
  return s >= 60 ? `${Math.floor(s / 60)}m${String(s % 60).padStart(2, "0")}s` : `${s}s`;
}

function renderQueue() {
  const list = [...jobs.values()].sort((a, b) => b.queued_at - a.queued_at);
  $("#queue-section").hidden = list.length === 0;
  const ul = $("#queue-list");
  ul.innerHTML = "";
  for (const job of list) {
    const li = document.createElement("li");
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = job.name;
    name.title = job.name;
    const time = document.createElement("span");
    time.className = "dim";
    time.dataset.jobTime = job.id;
    time.textContent = elapsed(job);
    const chip = document.createElement("span");
    chip.className = `chip ${job.status}`;
    chip.textContent = job.status;
    li.append(name, time, chip);
    if (job.status === "queued" || job.status === "running") {
      const running = job.status === "running";
      const x = document.createElement("button");
      x.className = "job-cancel";
      x.type = "button";
      x.title = running ? "Terminate this run" : "Cancel this queued job";
      x.textContent = "✕";
      x.addEventListener("click", () => {
        if (running && !confirm(`Terminate "${job.name}"?\n\nThe models will need to reload for the next job.`)) return;
        fetch(`/api/jobs/${job.id}/cancel`, { method: "POST" });
      });
      li.appendChild(x);
    }
    ul.appendChild(li);
    if (job.status === "running") {
      const prog = document.createElement("div");
      prog.className = "job-progress";
      prog.dataset.jobProgress = job.id;
      prog.textContent = progressText(job);
      prog.title = job.progress || "";
      ul.appendChild(prog);
    }
    if (job.status === "failed" && job.error) {
      const err = document.createElement("div");
      err.className = "job-error";
      err.textContent = job.error;
      ul.appendChild(err);
    }
    if (job.status === "done") {
      li.style.cursor = "pointer";
      li.title = "Open in viewer";
      li.addEventListener("click", () => openDoc(job.doc_id));
    }
  }
}

function progressText(job) {
  let text = job.progress || "working… (first run loads models, which takes a while)";
  const quiet = job._receivedAt ? Date.now() - job._receivedAt : 0;
  if (quiet > 90_000) text += `  · no output for ${Math.round(quiet / 60_000)}m`;
  return text;
}

function tickElapsed() {
  for (const job of jobs.values()) {
    if (job.status !== "running") continue;
    const el = document.querySelector(`[data-job-time="${job.id}"]`);
    if (el) el.textContent = elapsed(job);
    const prog = document.querySelector(`[data-job-progress="${job.id}"]`);
    if (prog) prog.textContent = progressText(job);
  }
}

/* ---------------- library + export ---------------- */

async function refreshLibrary() {
  docs = (await (await fetch("/api/docs")).json()).docs;
  renderLibrary();
}

function renderLibrary() {
  const ul = $("#library-list");
  ul.innerHTML = "";
  $("#library-empty").hidden = docs.length > 0;
  for (const doc of docs) {
    const li = document.createElement("li");
    if (currentDoc && currentDoc.id === doc.id) li.classList.add("selected-doc");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.checked = selectedIds.has(doc.id);
    cb.addEventListener("click", (e) => {
      e.stopPropagation();
      cb.checked ? selectedIds.add(doc.id) : selectedIds.delete(doc.id);
      renderExportBar();
    });
    const name = document.createElement("span");
    name.className = "name";
    name.textContent = doc.original_name.replace(/\.[^.]+$/, "");
    name.title = doc.original_name;
    const meta = document.createElement("span");
    meta.className = "meta";
    meta.textContent = `${doc.blocks} blocks · ${doc.seconds}s`;
    const reveal = document.createElement("button");
    reveal.className = "icon-btn";
    reveal.type = "button";
    reveal.title = revealLabel();
    reveal.textContent = "📂";
    reveal.addEventListener("click", (e) => { e.stopPropagation(); revealDoc(doc.id); });
    li.append(cb, name, meta, reveal);
    li.addEventListener("click", () => openDoc(doc.id));
    ul.appendChild(li);
  }
  renderExportBar();
}

$("#lib-select-all").addEventListener("change", (e) => {
  selectedIds.clear();
  if (e.target.checked) for (const d of docs) selectedIds.add(d.id);
  renderLibrary();
});

function renderExportBar() {
  const n = selectedIds.size;
  $("#export-bar").hidden = n === 0;
  $("#export-count").textContent = `${n} selected`;
}

async function downloadExport(mode) {
  const resp = await fetch("/api/export", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ids: [...selectedIds], mode }),
  });
  if (!resp.ok) { alert((await resp.json()).detail || "Export failed"); return; }
  triggerDownload(await resp.blob(), "mineru-export.zip");
}

function triggerDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 30_000);
}

$("#export-md").addEventListener("click", () => downloadExport("md-only"));
$("#export-md-img").addEventListener("click", () => downloadExport("md+images"));

/* ---------------- viewer ---------------- */

const tabState = { sideRendered: false, blocks: null };

async function openDoc(docId) {
  const doc = docs.find((d) => d.id === docId);
  if (!doc) return;
  currentDoc = doc;
  tabState.sideRendered = false;
  tabState.blocks = null;
  renderLibrary();

  $("#viewer-empty").hidden = true;
  $("#viewer-doc").hidden = false;
  $("#viewer-title").textContent = doc.original_name;
  $("#viewer-meta").textContent =
    `${(doc.markdown_chars / 1000).toFixed(1)}k chars · ${doc.blocks} blocks · ${doc.seconds}s · ${doc.device || ""}`;
  $("#viewer-zip").href = `/api/docs/${doc.id}/zip`;

  const md = await (await fetch(`/api/docs/${doc.id}/markdown`)).text();
  tabState.markdown = md;
  renderMarkdown(md, $("#md-rendered"), doc.id);
  $("#raw-md").textContent = md;
  $("#side-md").innerHTML = "";
  $("#side-pdf").innerHTML = "";
  selectTab(document.querySelector(".tab.active").dataset.tab, true);
}

document.querySelectorAll(".tab").forEach((btn) =>
  btn.addEventListener("click", () => selectTab(btn.dataset.tab)));

function selectTab(tab, force = false) {
  document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b.dataset.tab === tab));
  document.querySelectorAll(".tab-pane").forEach((p) => p.classList.toggle("active", p.id === `tab-${tab}`));
  if (!currentDoc) return;
  if (tab === "side" && (!tabState.sideRendered || force)) {
    tabState.sideRendered = true;
    renderMarkdown(tabState.markdown, $("#side-md"), currentDoc.id);
    renderSidePdf(currentDoc);
  }
  if (tab === "blocks" && (tabState.blocks === null || force)) loadBlocks(currentDoc);
}

/* Markdown rendering with math protection: pull $$…$$ / $…$ out before marked
   so underscores etc. inside TeX survive, then substitute KaTeX HTML back in. */
function renderMarkdown(src, container, docId) {
  const math = [];
  let text = src.replace(/\$\$([\s\S]+?)\$\$/g, (_, body) => {
    math.push({ body, display: true });
    return `§§MATH${math.length - 1}§§`;
  });
  text = text.replace(/\$([^$\n]+?)\$/g, (_, body) => {
    math.push({ body, display: false });
    return `§§MATH${math.length - 1}§§`;
  });
  text = text
    .replace(/(!\[[^\]]*\]\()images\//g, `$1/api/docs/${docId}/images/`)
    .replace(/(<img[^>]+src=["'])images\//g, `$1/api/docs/${docId}/images/`);
  let html = marked.parse(text);
  html = html.replace(/§§MATH(\d+)§§/g, (m, i) => {
    const { body, display } = math[Number(i)];
    try {
      return katex.renderToString(body, { displayMode: display, throwOnError: false });
    } catch {
      return m;
    }
  });
  container.innerHTML = html;
}

async function renderSidePdf(doc) {
  const pane = $("#side-pdf");
  pane.innerHTML = "";
  if (!/\.pdf$/i.test(doc.original_name)) {
    if (/\.(png|jpe?g|webp|gif|bmp|tiff|jp2)$/i.test(doc.original_name)) {
      const img = document.createElement("img");
      img.src = `/api/docs/${doc.id}/source`;
      img.style.maxWidth = "100%";
      pane.appendChild(img);
    } else {
      pane.innerHTML = `<p class="dim" style="color:#cdd1d6">No inline preview for this file type — ` +
        `<a style="color:#9ec1ff" href="/api/docs/${doc.id}/source">download the original</a>.</p>`;
    }
    return;
  }
  if (!pdfjsModule) {
    pdfjsModule = await import("/static/vendor/pdfjs/pdf.min.mjs");
    pdfjsModule.GlobalWorkerOptions.workerSrc = "/static/vendor/pdfjs/pdf.worker.min.mjs";
  }
  const pdf = await pdfjsModule.getDocument(`/api/docs/${doc.id}/source`).promise;
  for (let i = 1; i <= pdf.numPages; i++) {
    if (currentDoc !== doc) return; // user switched docs mid-render
    const page = await pdf.getPage(i);
    const viewport = page.getViewport({ scale: 1.4 });
    const canvas = document.createElement("canvas");
    canvas.width = viewport.width;
    canvas.height = viewport.height;
    pane.appendChild(canvas);
    await page.render({ canvasContext: canvas.getContext("2d"), viewport }).promise;
  }
}

async function loadBlocks(doc) {
  const resp = await fetch(`/api/docs/${doc.id}/content-list`);
  if (!resp.ok) {
    $("#blocks-table").innerHTML = `<p class="dim">No content list available.</p>`;
    tabState.blocks = [];
    return;
  }
  tabState.blocks = await resp.json();
  $("#blocks-raw").textContent = JSON.stringify(tabState.blocks, null, 2);
  const types = [...new Set(tabState.blocks.map((b) => b.type))].sort();
  const sel = $("#blocks-filter");
  sel.innerHTML = `<option value="">all (${tabState.blocks.length})</option>` +
    types.map((t) => `<option value="${t}">${t}</option>`).join("");
  renderBlocksTable();
}

function blockPreview(b) {
  if (b.type === "image") return b.img_caption?.join(" ") || b.img_path || "";
  if (b.type === "table") return b.table_caption?.join(" ") || "[table]";
  if (b.type === "equation") return b.text || b.latex || "";
  return b.text || "";
}

function renderBlocksTable() {
  const filter = $("#blocks-filter").value;
  const rows = tabState.blocks
    .map((b, i) => ({ ...b, _i: i }))
    .filter((b) => !filter || b.type === filter);
  const esc = (s) => String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;");
  $("#blocks-table").innerHTML =
    `<table><thead><tr><th>#</th><th>type</th><th>page</th><th>content</th></tr></thead><tbody>` +
    rows.map((b) =>
      `<tr><td>${b._i}</td><td>${esc(b.type)}</td><td>${b.page_idx ?? ""}</td>` +
      `<td>${esc(blockPreview(b).slice(0, 220))}</td></tr>`).join("") +
    `</tbody></table>`;
}

$("#blocks-filter").addEventListener("change", renderBlocksTable);
$("#blocks-json").addEventListener("change", (e) => {
  $("#blocks-raw").hidden = !e.target.checked;
  $("#blocks-table").hidden = e.target.checked;
});

$("#raw-copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText(tabState.markdown || "");
  $("#raw-copy").textContent = "Copied!";
  setTimeout(() => ($("#raw-copy").textContent = "Copy markdown"), 1500);
});

$("#viewer-reveal").addEventListener("click", () => { if (currentDoc) revealDoc(currentDoc.id); });

$("#opt-backend").addEventListener("change", (e) => {
  $("#backend-warning").hidden = e.target.value !== "vlm-transformers";
});

$("#viewer-delete").addEventListener("click", async () => {
  if (!currentDoc) return;
  if (!confirm(`Delete "${currentDoc.original_name}" and all its outputs?`)) return;
  await fetch(`/api/docs/${currentDoc.id}`, { method: "DELETE" });
  selectedIds.delete(currentDoc.id);
  currentDoc = null;
  $("#viewer-doc").hidden = true;
  $("#viewer-empty").hidden = false;
  await refreshLibrary();
});

boot();
