"use strict";

const $ = (sel) => document.querySelector(sel);
const LANG_LABEL = { hi: "हिंदी", en: "English", hinglish: "Hinglish", mixed: "Hindi + English" };

const state = {
  docs: [],
  selected: new Set(),
  history: [], // [{role, content}] sent back to the server for follow-up questions
  busy: false,
};

// ---------------------------------------------------------------- helpers
function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

/** Minimal, XSS-safe markdown: escape first, then add a few inline/block formats. */
function renderMarkdown(text) {
  const inline = (s) =>
    escapeHtml(s)
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\[(\d+)\]/g, '<button type="button" class="cite" data-n="$1">$1</button>');

  const out = [];
  let list = null; // "ul" | "ol"
  const closeList = () => { if (list) { out.push(`</${list}>`); list = null; } };
  let para = [];
  const flushPara = () => { if (para.length) { out.push(`<p>${para.map(inline).join("<br>")}</p>`); para = []; } };

  for (const raw of text.split("\n")) {
    const line = raw.trimEnd();
    const ul = line.match(/^\s*[-*•]\s+(.*)$/);
    const ol = line.match(/^\s*\d+[.)]\s+(.*)$/);
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (ul || ol) {
      flushPara();
      const kind = ul ? "ul" : "ol";
      if (list !== kind) { closeList(); out.push(`<${kind}>`); list = kind; }
      out.push(`<li>${inline((ul || ol)[1])}</li>`);
    } else if (h) {
      flushPara(); closeList();
      out.push(`<h4>${inline(h[2])}</h4>`);
    } else if (!line.trim()) {
      flushPara(); closeList();
    } else {
      closeList();
      para.push(line);
    }
  }
  flushPara(); closeList();
  return out.join("");
}

async function api(path, options = {}) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(body.detail || `Request failed (${res.status})`);
  return body;
}

// ---------------------------------------------------------------- documents
async function refreshStats() {
  try {
    const s = await api("/api/health");
    $("#stats").textContent =
      `${s.documents} docs · ${s.chunks} chunks · LLM: ${s.llm || "none"}` + (s.reranker ? " · reranker on" : "");
  } catch {
    $("#stats").textContent = "Server not ready…";
  }
}

async function refreshDocs() {
  state.docs = await api("/api/documents");
  const ids = new Set(state.docs.map((d) => d.id));
  for (const id of [...state.selected]) if (!ids.has(id)) state.selected.delete(id);
  renderDocs();
  refreshStats();
}

function renderDocs() {
  const list = $("#doc-list");
  if (!state.docs.length) {
    list.innerHTML = '<li class="empty-docs">No documents yet. Upload one to get started.</li>';
  } else {
    list.innerHTML = state.docs.map((d) => `
      <li class="doc">
        <input type="checkbox" data-id="${d.id}" ${state.selected.has(d.id) ? "checked" : ""} aria-label="Use ${escapeHtml(d.filename)}">
        <div>
          <div class="doc-name">${escapeHtml(d.filename)}</div>
          <div class="doc-meta">
            <span class="badge lang">${LANG_LABEL[d.language] || d.language}</span>
            <span class="badge">${d.num_pages} page${d.num_pages === 1 ? "" : "s"}</span>
            <span class="badge">${d.num_chunks} chunks</span>
          </div>
          ${d.warnings.map((w) => `<div class="doc-warn">⚠ ${escapeHtml(w)}</div>`).join("")}
        </div>
        <button class="icon-btn" type="button" data-delete="${d.id}" title="Remove from knowledge base">✕</button>
      </li>`).join("");
  }
  renderScope();
}

function renderScope() {
  const n = state.selected.size;
  $("#scope").textContent = n ? `Searching ${n} selected document${n === 1 ? "" : "s"}` : "Searching all documents";
  $("#select-all").textContent = n && n === state.docs.length ? "Clear" : "Select all";
}

async function uploadFiles(files) {
  if (!files.length) return;
  const status = $("#upload-status");
  const pending = document.createElement("li");
  pending.textContent = `Processing ${files.length} file(s)… (embedding can take a moment)`;
  status.prepend(pending);

  const form = new FormData();
  for (const f of files) form.append("files", f);
  try {
    const { results } = await api("/api/documents", { method: "POST", body: form });
    pending.remove();
    for (const r of results) {
      const li = document.createElement("li");
      li.className = r.status;
      li.textContent =
        r.status === "added" ? `✓ ${r.filename}: ${r.document.num_chunks} chunks indexed`
        : r.status === "duplicate" ? `• ${r.filename}: already in the knowledge base`
        : `✕ ${r.filename}: ${r.error}`;
      status.prepend(li);
      setTimeout(() => li.remove(), 12000);
    }
  } catch (err) {
    pending.className = "error";
    pending.textContent = `Upload failed: ${err.message}`;
  }
  refreshDocs();
}

async function deleteDoc(id) {
  const doc = state.docs.find((d) => d.id === id);
  if (!doc || !confirm(`Remove "${doc.filename}" from the knowledge base?`)) return;
  try {
    await api(`/api/documents/${id}`, { method: "DELETE" });
  } catch (err) {
    alert(err.message);
  }
  refreshDocs();
}

// ---------------------------------------------------------------- chat
function addMessage(role, html, extraClass = "") {
  $("#empty-state")?.remove();
  const el = document.createElement("div");
  el.className = `msg ${role} ${extraClass}`.trim();
  el.innerHTML = html;
  $("#messages").append(el);
  el.scrollIntoView({ behavior: "smooth", block: "end" });
  return el;
}

function renderAnswer(res) {
  const sources = res.sources.map((s, i) => `
    <div class="source" data-source="${i + 1}">
      <div class="source-head">
        <span>[${i + 1}] ${escapeHtml(s.filename)} · p.${s.page}</span>
        <span>${s.rerank_score != null ? `rerank ${s.rerank_score.toFixed(2)}` : `rrf ${s.rrf_score.toFixed(3)}`}</span>
      </div>
      <div class="source-text">${escapeHtml(s.text)}</div>
    </div>`).join("");

  const t = res.timings || {};
  const meta = [
    `<span class="badge lang">${LANG_LABEL[res.language] || res.language}</span>`,
    t.retrieval_ms != null ? `<span>retrieval ${t.retrieval_ms} ms</span>` : "",
    t.generation_ms != null ? `<span>generation ${(t.generation_ms / 1000).toFixed(1)} s</span>` : "",
    res.search_query && res.search_query !== res.question ? `<span title="Follow-up rewritten for search">🔎 ${escapeHtml(res.search_query)}</span>` : "",
  ].join("");

  return `${renderMarkdown(res.answer)}
    <div class="msg-meta">${meta}</div>
    ${res.sources.length ? `<details class="sources"><summary>${res.sources.length} sources</summary>${sources}</details>` : ""}`;
}

async function ask(question) {
  if (state.busy || !question.trim()) return;
  state.busy = true;
  $("#send").disabled = true;
  addMessage("user", escapeHtml(question));
  const thinking = addMessage("assistant", '<span class="typing"><i></i><i></i><i></i></span>');

  try {
    const res = await api("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        history: state.history,
        doc_ids: state.selected.size ? [...state.selected] : null,
      }),
    });
    res.question = question;
    thinking.innerHTML = renderAnswer(res);
    state.history.push({ role: "user", content: question }, { role: "assistant", content: res.answer });
  } catch (err) {
    thinking.classList.add("error");
    thinking.textContent = `Error: ${err.message}`;
  } finally {
    state.busy = false;
    $("#send").disabled = false;
    thinking.scrollIntoView({ behavior: "smooth", block: "end" });
  }
}

// ---------------------------------------------------------------- wiring
function init() {
  const input = $("#file-input");
  const drop = $("#dropzone");
  input.addEventListener("change", () => { uploadFiles([...input.files]); input.value = ""; });
  drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); input.click(); } });
  ["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
  drop.addEventListener("drop", (e) => uploadFiles([...e.dataTransfer.files]));

  $("#doc-list").addEventListener("click", (e) => {
    const del = e.target.closest("[data-delete]");
    if (del) deleteDoc(del.dataset.delete);
  });
  $("#doc-list").addEventListener("change", (e) => {
    if (e.target.matches("input[type=checkbox]")) {
      e.target.checked ? state.selected.add(e.target.dataset.id) : state.selected.delete(e.target.dataset.id);
      renderScope();
    }
  });
  $("#select-all").addEventListener("click", () => {
    if (state.selected.size === state.docs.length) state.selected.clear();
    else state.docs.forEach((d) => state.selected.add(d.id));
    renderDocs();
  });

  const q = $("#question");
  const autosize = () => { q.style.height = "auto"; q.style.height = `${q.scrollHeight}px`; };
  q.addEventListener("input", autosize);
  q.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#chat-form").requestSubmit(); }
  });
  $("#chat-form").addEventListener("submit", (e) => {
    e.preventDefault();
    const text = q.value;
    q.value = "";
    autosize();
    ask(text);
  });
  document.querySelectorAll(".example").forEach((b) => b.addEventListener("click", () => ask(b.textContent)));

  $("#messages").addEventListener("click", (e) => {
    const cite = e.target.closest(".cite");
    if (!cite) return;
    const msg = cite.closest(".msg");
    const details = msg.querySelector("details.sources");
    const src = msg.querySelector(`[data-source="${cite.dataset.n}"]`);
    if (!details || !src) return;
    details.open = true;
    src.classList.add("flash");
    src.scrollIntoView({ behavior: "smooth", block: "center" });
    setTimeout(() => src.classList.remove("flash"), 1500);
  });

  $("#clear-chat").addEventListener("click", () => {
    state.history = [];
    $("#messages").innerHTML = "";
    addMessage("assistant", "New conversation started. Ask a question about your documents.");
  });

  refreshDocs().catch(() => setTimeout(() => refreshDocs(), 2000));
}

init();
