"use strict";
const $ = id => document.getElementById(id);
let selected = null, detail = null, pane = "input", records = [], state = null, operating = false, stamp = "", actionError = "";
let apiKey = sessionStorage.getItem("strata.monitor.key") || "";
document.documentElement.dataset.theme = localStorage.getItem("strata.theme") || "dark";
const seconds = n => typeof n === "number" ? `${n.toFixed(2)} s` : "—";
const speed = n => typeof n === "number" ? n.toFixed(1) : "—";
const fmt = n => typeof n === "number" ? n.toLocaleString() : "—";
// A rate that can be in the thousands (a prompt read) keeps its digits apart: 2,336 t/s, 84.9 t/s.
const rate = n => typeof n !== "number" || !(n > 0) ? "—" : n >= 100 ? Math.round(n).toLocaleString() : n.toFixed(1);
const active = r => !["completed", "error", "disconnected"].includes(r.state);
// How far the prompt read has got: the engine's position over the request's whole prompt (a reused prefix counts as
// read, so this is real progress, not the tokens this pass had to read).
const readPct = r => typeof r.prompt_read === "number" && r.prompt_total > 0
  ? Math.min(100, 100 * r.prompt_read / r.prompt_total) : null;
// The prompt reading speed: the engine's own figure while the read runs (prefill_tok_s, one PP line per chunk), and
// the request's own prompt time once it has finished (timings.prompt_per_second, which leaves the reused prefix out).
const prefillOf = r => {
  const t = r.timings;
  if (t && typeof t.prompt_per_second === "number" && t.prompt_per_second > 0) return t.prompt_per_second;
  if (typeof r.prefill_tok_s === "number" && r.prefill_tok_s > 0) return r.prefill_tok_s;
  return null;
};
// The run's average decode speed: every token produced over the run divided by the run's decode time, from each
// finished request's engine timings (predicted_n over predicted_ms). Token-weighted, so a short request does not
// outweigh a long one - the same figure the engine's DONE lines and the server's totals report. Not the last request.
function averageDecodeSpeed(rows) {
  let tokens = 0, ms = 0;
  for (const r of rows) {
    const t = r.timings;
    if (!t) continue;
    const n = t.predicted_n, d = t.predicted_ms;
    if (typeof n === "number" && typeof d === "number" && n >= 2 && d > 0) { tokens += n; ms += d; }
  }
  return ms > 0 ? tokens / (ms / 1000) : null;
}
function text(id, value) { const el = $(id); if (el.textContent !== String(value)) el.textContent = value; }
async function api(path, body) {
  const response = await fetch(path, {cache:"no-store", headers:{...(apiKey ? {Authorization:`Bearer ${apiKey}`} : {}), ...(body !== undefined ? {"Content-Type":"application/json"} : {})}, ...(body !== undefined ? {method:"POST", body:JSON.stringify(body)} : {})});
  const value = await response.json();
  if (!response.ok) {
    if (response.status === 401) $("auth").hidden = false;
    throw new Error(value.error?.message || `HTTP ${response.status}`);
  }
  return value;
}
function notice(message) { $("error").hidden = !message; text("error", message || ""); }
function list() {
  const query = $("search").value.toLowerCase(), filter = $("filter").value;
  const visible = records.filter(r => `${r.id} ${r.path} ${r.model}`.toLowerCase().includes(query) && (filter === "all" || (filter === "active" ? active(r) : r.state === filter)));
  const nextStamp = JSON.stringify([visible, selected]);
  if (stamp === nextStamp) return;
  stamp = nextStamp;
  const focusId = document.activeElement?.dataset.request;
  $("requests").replaceChildren();
  if (!visible.length) {
    const empty = document.createElement("div"); empty.className = "empty";
    empty.textContent = records.length ? "No matching requests." : "No requests yet. API calls will appear here automatically.";
    $("requests").append(empty);
  }
  for (const r of visible) {
    const button = document.createElement("button"); button.className = `request${selected === r.id ? " selected" : ""}`;
    button.dataset.request = r.id; button.setAttribute("aria-label", `Request ${r.id}`); button.setAttribute("aria-pressed", String(selected === r.id));
    const row = document.createElement("div"); row.className = "row";
    const time = document.createElement("span"); time.textContent = new Date(r.started_at * 1000).toLocaleTimeString();
    const tag = document.createElement("span"); tag.className = `tag ${r.state}`; tag.textContent = r.state;
    row.append(time, tag);
    const endpoint = document.createElement("div"); endpoint.className = "endpoint"; endpoint.textContent = `POST ${r.path}`;
    const meta = document.createElement("div"); meta.className = "small";
    meta.textContent = `${r.id} · ${seconds(r.wallclock_s)} · ${r.stream ? "stream" : "JSON response"}${r.http_status ? ` · HTTP ${r.http_status}` : ""}${readNote(r)}`;
    button.append(row, endpoint, meta);
    button.onclick = async () => { selected = r.id; detail = null; list(); await showDetail(); };
    $("requests").append(button);
    if (focusId === r.id) button.focus({preventScroll:true});
  }
}
// The "Prompt read" card: the live read's own percentage and speed while a prompt is being read, and otherwise the
// prompt speed of the last request. Reading a prompt is most of an agent turn's wall-clock, so it belongs on the
// front page next to the decode speed.
function prefillCard() {
  const reading = records.find(r => active(r) && r.state === "reading" && readPct(r) != null);
  if (reading) {
    text("prefill", `${Math.round(readPct(reading))}%`);
    text("prefill-note", `${fmt(reading.prompt_read)} of ${fmt(reading.prompt_total)} tokens · ${rate(prefillOf(reading))} t/s`);
    return;
  }
  const live = records.find(r => active(r) && prefillOf(r) != null);
  if (live) { text("prefill", rate(prefillOf(live))); text("prefill-note", "Reading now · tokens / second"); return; }
  const done = records.find(r => !active(r) && prefillOf(r) != null);
  if (done) {
    text("prefill", rate(prefillOf(done)));
    text("prefill-note", `Last request · ${fmt(done.usage?.prompt_tokens ?? done.usage?.input_tokens)} prompt tokens`);
    return;
  }
  text("prefill", "—"); text("prefill-note", "Prompt speed · tokens / second");
}
// A row's own reading line: how far the prompt read has got, then how fast the prompt was read.
function readNote(r) {
  const pct = readPct(r);
  if (pct != null && r.state === "reading") {
    return ` · reading ${Math.round(pct)}% (${fmt(r.prompt_read)} of ${fmt(r.prompt_total)} tokens, ${rate(prefillOf(r))} t/s)`;
  }
  const pre = prefillOf(r);
  return pre ? ` · prompt ${rate(pre)} t/s` : "";
}
function content() {
  if (!detail) return;
  let value = detail[pane] || (pane === "response" && detail.stream ? "Streaming response: see Output and the request error, if any." : "No content.");
  if (pane === "output") { try { value = JSON.stringify(JSON.parse(value), null, 2); } catch (_) {} }
  text("content", value);
  const note = pane === "output" && detail.error ? "Raw model output retained for diagnosis. The API returned an error." : pane === "input" ? "Original request body." : pane === "output" ? "Model answer. JSON is formatted here for readability." : pane === "reasoning" ? "Separate reasoning content, when enabled." : "API response body.";
  text("content-note", note + (detail[`${pane}_truncated`] ? " Monitor capture truncated at 256K characters; the API response was not shortened." : ""));
}
async function showDetail() {
  if (!selected) return;
  const id = selected;
  const value = await api(`/api/requests?id=${encodeURIComponent(id)}`);
  if (selected !== id) return;
  detail = value; $("selection").hidden = false; $("no-selection").hidden = true;
  text("detail-title", `${value.id} · ${value.path}`);
  text("detail-status", value.state); $("detail-status").className = `tag ${value.state}`;
  text("detail-time", new Date(value.started_at * 1000).toLocaleString());
  text("detail-wall", seconds(value.wallclock_s ?? ((Date.now() / 1000) - value.started_at)));
  text("detail-load", seconds(value.load_s)); text("detail-queue", seconds(value.queue_s)); text("detail-first", seconds(value.first_token_s));
  text("detail-prompt", value.usage?.prompt_tokens ?? value.usage?.input_tokens ?? "—");
  text("detail-tokens", value.usage?.completion_tokens ?? value.usage?.output_tokens ?? "—");
  text("detail-speed", speed(value.timings?.predicted_per_second)); text("detail-format", value.response_format || "text");
  text("detail-prefill", rate(prefillOf(value)));
  const t = value.timings;
  text("detail-reused", t && typeof t.cache_n === "number" ? `${fmt(t.cache_n)} of ${fmt(t.prompt_n)}` : "—");
  $("detail-error").hidden = !value.error; text("detail-error", value.error?.message || ""); content();
}
async function refresh() {
  try {
    const [status, history] = await Promise.all([api("/v1/status"), api("/api/requests")]);
    state = status; records = history.requests;
    status.loaded = history.loaded; status.auto_load = history.auto_load;
    if (selected && !records.some(r => r.id === selected)) {
      selected = detail = null; $("selection").hidden = true; $("no-selection").hidden = false;
    }
    const inflight = Math.max(status.activity.in_flight, records.filter(active).length);
    const loading = records.some(r => r.state === "loading");
    text("loaded", loading ? "Loading…" : inflight ? "In use" : status.loaded ? "Loaded" : "Unloaded");
    $("model-state").className = `value ${inflight || loading ? "busy" : status.loaded ? "loaded" : ""}`;
    text("model", status.model); text("count", records.length);
    text("active", inflight ? `${inflight} active / queued` : status.auto_load ? "Loads automatically on the next request" : "No active requests");
    const last = records.find(r => !active(r));
    text("wall", last ? seconds(last.wallclock_s) : "—"); text("speed", speed(averageDecodeSpeed(records.filter(r => !active(r)))));
    prefillCard();
    $("load").disabled = operating || Boolean(inflight) || status.loaded; $("unload").disabled = operating || Boolean(inflight) || !status.loaded;
    list(); if (selected) await showDetail();
    text("updated", `Live · updated ${new Date().toLocaleTimeString()}`);
    if (!operating) notice(actionError);
  } catch (error) { notice(error.message); text("updated", "Disconnected · retrying"); $("load").disabled = $("unload").disabled = true; }
}
async function control(load) {
  actionError = "";
  operating = true; $("load").disabled = $("unload").disabled = true;
  text("loaded", load ? "Loading…" : "Unloading…");
  try { await api(load ? "/load" : "/unload", {}); notice(""); }
  catch (error) { actionError = error.message; notice(actionError); }
  finally { operating = false; await refresh(); }
}
$("load").onclick = () => control(true); $("unload").onclick = () => control(false);
$("key").onclick = () => { $("auth").hidden = !$("auth").hidden; if (!$("auth").hidden) $("api-key").focus(); };
$("auth").onsubmit = async event => { event.preventDefault(); apiKey = $("api-key").value.trim(); sessionStorage.setItem("strata.monitor.key", apiKey); $("auth").hidden = true; await refresh(); };
$("theme").onclick = () => { const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark"; document.documentElement.dataset.theme = theme; localStorage.setItem("strata.theme", theme); };
$("search").oninput = list; $("filter").onchange = list;
document.querySelectorAll("[data-pane]").forEach(button => { button.onclick = () => { pane = button.dataset.pane; document.querySelectorAll("[data-pane]").forEach(b => b.setAttribute("aria-selected", String(b === button))); content(); }; });
$("copy").onclick = async () => { try { await navigator.clipboard.writeText($("content").textContent); text("copy", "Copied"); setTimeout(() => text("copy", "Copy"), 1500); } catch (_) { notice("Clipboard unavailable; select the text to copy it."); } };
async function poll() { await refresh(); setTimeout(poll, 2000); }
poll();
