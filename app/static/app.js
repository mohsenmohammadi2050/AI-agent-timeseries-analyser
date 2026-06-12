const apiKeyInput = document.getElementById("apiKey");
const threadIdInput = document.getElementById("threadId");
const datasetIdInput = document.getElementById("datasetId");
const datasetFileInput = document.getElementById("datasetFile");
const modelNameSelect = document.getElementById("modelName");
const forecastStartInput = document.getElementById("forecastStart");
const forecastEndInput = document.getElementById("forecastEnd");
const sampleRateInput = document.getElementById("sampleRate");
const chatLog = document.getElementById("chatLog");
const statusText = document.getElementById("statusText");
const toolsBox = document.getElementById("toolsBox");
const traceBox = document.getElementById("traceBox");
const intentBox = document.getElementById("intentBox");
const evaluationBox = document.getElementById("evaluationBox");
const evaluateBtn = document.getElementById("evaluateBtn");
const openEvaluationBtn = document.getElementById("openEvaluationBtn");
const closeEvaluationBtn = document.getElementById("closeEvaluationBtn");
const evaluationModal = document.getElementById("evaluationModal");
const evaluationReport = document.getElementById("evaluationReport");
const evaluationModalSubtitle = document.getElementById("evaluationModalSubtitle");
const messageForm = document.getElementById("messageForm");
const messageInput = document.getElementById("messageInput");

let latestEvalArtifactPath = null;
let latestEvaluationReport = null;

function headers(json = false) {
  const result = { "X-API-Key": apiKeyInput.value.trim() };
  if (json) result["Content-Type"] = "application/json";
  return result;
}

function setStatus(text) {
  statusText.textContent = text;
}

function addMessage(role, content, kind = "") {
  const item = document.createElement("div");
  item.className = `message ${kind || role}`;
  const meta = document.createElement("div");
  meta.className = "meta";
  meta.textContent = role === "user" ? "You" : role === "agent" ? "Agent" : "System";
  const body = document.createElement("div");
  body.textContent = content;
  item.append(meta, body);
  chatLog.appendChild(item);
  chatLog.scrollTop = chatLog.scrollHeight;
}

function showJson(element, value) {
  element.textContent = JSON.stringify(value, null, 2);
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function markdownToHtml(value) {
  const lines = escapeHtml(value).split(/\r?\n/);
  const html = [];
  let inList = false;

  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed) {
      if (inList) {
        html.push("</ul>");
        inList = false;
      }
      continue;
    }

    const formatted = trimmed
      .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
      .replace(/`(.+?)`/g, "<code>$1</code>");

    if (formatted.startsWith("- ")) {
      if (!inList) {
        html.push("<ul>");
        inList = true;
      }
      html.push(`<li>${formatted.slice(2)}</li>`);
    } else {
      if (inList) {
        html.push("</ul>");
        inList = false;
      }
      html.push(`<p>${formatted}</p>`);
    }
  }

  if (inList) html.push("</ul>");
  return html.join("");
}

function formatNumber(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return "n/a";
  return Number(value).toFixed(digits);
}

function scoreRow(label, item) {
  const score = formatNumber(item?.score, 1);
  const max = formatNumber(item?.max_score, 0);
  return `<div class="score-row"><span>${escapeHtml(label)}</span><strong>${score}/${max}</strong></div>`;
}

function renderEvaluationSummary(report) {
  if (!report) {
    evaluationBox.textContent = "No evaluation yet";
    return;
  }
  evaluationBox.innerHTML = `
    <div class="score-pill">${formatNumber(report.score_total, 1)}/${formatNumber(report.score_max, 0)}</div>
    <div>MAE: ${formatNumber(report.prediction_metrics?.mae)}</div>
    <div>MAPE: ${formatNumber(report.prediction_metrics?.mape)}%</div>
    <div>High-error hours: ${escapeHtml(report.prediction_metrics?.high_error_count ?? "n/a")}</div>
  `;
}

function renderEvaluationReport(report) {
  latestEvaluationReport = report;
  openEvaluationBtn.disabled = !report;
  renderEvaluationSummary(report);
  if (!report) {
    evaluationReport.innerHTML = "";
    evaluationModalSubtitle.textContent = "No report loaded";
    return;
  }

  const metrics = report.prediction_metrics || {};
  const scores = report.scores || {};
  const verdict = report.evaluator_agent?.verdict || "No evaluator verdict was returned.";
  const worstHours = metrics.worst_hours || [];

  evaluationModalSubtitle.textContent = `Score ${formatNumber(report.score_total, 1)}/${formatNumber(report.score_max, 0)}`;
  evaluationReport.innerHTML = `
    <section class="report-section report-score">
      <div class="score-big">${formatNumber(report.score_total, 1)}</div>
      <div>
        <h3>Overall Score</h3>
        <p>Saved report: <code>${escapeHtml(report.evaluation_report_path)}</code></p>
      </div>
    </section>

    <section class="report-section">
      <h3>Score Breakdown</h3>
      ${scoreRow("Tool use and intent", scores.tool_use_and_intent)}
      ${scoreRow("Reliability judgment", scores.reliability_judgment)}
      ${scoreRow("Grounding in tools", scores.grounding_in_tool_outputs)}
      ${scoreRow("Temporal fairness", scores.temporal_fairness)}
      ${scoreRow("Communication quality", scores.communication_quality)}
    </section>

    <section class="report-section">
      <h3>Prediction Metrics</h3>
      <div class="metric-grid">
        <div><span>Matched points</span><strong>${escapeHtml(metrics.matched_points ?? "n/a")}</strong></div>
        <div><span>MAE</span><strong>${formatNumber(metrics.mae)}</strong></div>
        <div><span>MAPE</span><strong>${formatNumber(metrics.mape)}%</strong></div>
        <div><span>Max error</span><strong>${formatNumber(metrics.max_absolute_error)}</strong></div>
        <div><span>Bias</span><strong>${formatNumber(metrics.bias)}</strong></div>
        <div><span>High-error hours</span><strong>${escapeHtml(metrics.high_error_count ?? "n/a")}</strong></div>
      </div>
    </section>

    <section class="report-section">
      <h3>Evaluator Verdict</h3>
      <div class="markdown-body">${markdownToHtml(verdict)}</div>
    </section>

    <section class="report-section">
      <h3>Worst Hours</h3>
      <table>
        <thead>
          <tr>
            <th>Timestamp</th>
            <th>Predicted</th>
            <th>Actual</th>
            <th>Abs Error</th>
            <th>APE</th>
          </tr>
        </thead>
        <tbody>
          ${worstHours
            .map(
              (row) => `
                <tr>
                  <td>${escapeHtml(row.timestamp)}</td>
                  <td>${formatNumber(row.predicted_value)}</td>
                  <td>${formatNumber(row.actual_value)}</td>
                  <td>${formatNumber(row.absolute_error)}</td>
                  <td>${formatNumber(row.absolute_percent_error)}%</td>
                </tr>
              `
            )
            .join("")}
        </tbody>
      </table>
    </section>
  `;
}

function openEvaluationModal() {
  if (!latestEvaluationReport) return;
  evaluationModal.hidden = false;
}

function closeEvaluationModal() {
  evaluationModal.hidden = true;
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  let data = {};
  if (text) {
    try {
      data = JSON.parse(text);
    } catch {
      data = { detail: text };
    }
  }
  if (!response.ok) {
    const detail = data.detail || response.statusText;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return data;
}

async function ensureThread() {
  if (threadIdInput.value.trim()) return threadIdInput.value.trim();
  const data = await apiFetch("/v1/chats", { method: "POST", headers: headers() });
  threadIdInput.value = data.thread_id;
  return data.thread_id;
}

async function createChat() {
  setStatus("Creating chat...");
  const data = await apiFetch("/v1/chats", { method: "POST", headers: headers() });
  threadIdInput.value = data.thread_id;
  chatLog.innerHTML = "";
  latestEvalArtifactPath = null;
  latestEvaluationReport = null;
  evaluateBtn.disabled = true;
  openEvaluationBtn.disabled = true;
  intentBox.textContent = "None yet";
  renderEvaluationReport(null);
  toolsBox.textContent = "No tool calls yet";
  traceBox.textContent = "No trace yet";
  setStatus("New chat created");
}

async function loadModels() {
  setStatus("Loading models...");
  const models = await apiFetch("/v1/models", { headers: headers() });
  modelNameSelect.innerHTML = '<option value="">No model</option>';
  for (const model of models) {
    const option = document.createElement("option");
    option.value = model.name;
    option.textContent = `${model.name}${model.available ? "" : " (unavailable)"}`;
    option.disabled = !model.available;
    modelNameSelect.appendChild(option);
  }
  setStatus("Models loaded");
}

async function uploadDataset() {
  const file = datasetFileInput.files[0];
  if (!file) {
    addMessage("system", "Choose a CSV file first.", "error");
    return;
  }
  setStatus("Uploading dataset...");
  const form = new FormData();
  form.append("file", file);
  const data = await apiFetch("/v1/datasets/upload", {
    method: "POST",
    headers: headers(),
    body: form,
  });
  datasetIdInput.value = data.dataset_id;
  addMessage(
    "system",
    `Dataset uploaded: ${data.filename}\nRows: ${data.row_count}\nDataset ID: ${data.dataset_id}`
  );
  setStatus("Dataset uploaded");
}

function buildPayload(message) {
  const payload = { message };
  const datasetId = datasetIdInput.value.trim();
  const modelName = modelNameSelect.value.trim();
  const forecastStart = forecastStartInput.value.trim();
  const forecastEnd = forecastEndInput.value.trim();
  const sampleRate = Number(sampleRateInput.value || 3600);
  if (datasetId) payload.dataset_id = datasetId;
  if (modelName) payload.model_name = modelName;
  if (forecastStart) payload.forecast_start = forecastStart;
  if (forecastEnd) payload.forecast_end = forecastEnd;
  if (sampleRate) payload.sample_rate_seconds = sampleRate;
  return payload;
}

async function sendMessage(event) {
  event.preventDefault();
  const message = messageInput.value.trim();
  if (!message) return;
  messageInput.value = "";
  addMessage("user", message);

  try {
    const threadId = await ensureThread();
    setStatus("Agent is thinking...");
    const data = await apiFetch(`/v1/chats/${threadId}/messages`, {
      method: "POST",
      headers: headers(true),
      body: JSON.stringify(buildPayload(message)),
    });
    addMessage("agent", data.answer || "(No answer)");
    latestEvalArtifactPath = data.eval_artifact_path || null;
    latestEvaluationReport = null;
    evaluateBtn.disabled = !latestEvalArtifactPath;
    openEvaluationBtn.disabled = true;
    if (!latestEvalArtifactPath) {
      evaluationBox.textContent = "No prediction-analysis artifact for this answer";
    } else {
      evaluationBox.textContent = "Ready to evaluate this answer";
    }
    showJson(intentBox, {
      intent: data.intent,
      run_id: data.run_id,
      eval_artifact_path: data.eval_artifact_path || null,
    });
    showJson(toolsBox, data.tools || []);
    showJson(traceBox, data.agent_trace || []);
    setStatus("Ready");
  } catch (error) {
    addMessage("system", error.message, "error");
    setStatus("Error");
  }
}

async function evaluateLatest() {
  if (!latestEvalArtifactPath) {
    addMessage("system", "There is no prediction-analysis artifact to evaluate yet.", "error");
    return;
  }
  try {
    evaluateBtn.disabled = true;
    setStatus("Evaluating latest prediction analysis...");
    const data = await apiFetch("/v1/evaluations/prediction-analysis", {
      method: "POST",
      headers: headers(true),
      body: JSON.stringify({
        artifact_path: latestEvalArtifactPath,
        use_llm: true,
      }),
    });
    renderEvaluationReport(data);
    openEvaluationModal();
    addMessage(
      "system",
      `Evaluation score: ${data.score_total}/${data.score_max}\nReport: ${data.evaluation_report_path}`
    );
    setStatus("Evaluation complete");
  } catch (error) {
    addMessage("system", error.message, "error");
    setStatus("Evaluation error");
  } finally {
    evaluateBtn.disabled = !latestEvalArtifactPath;
  }
}

document.getElementById("newChatBtn").addEventListener("click", () => {
  createChat().catch((error) => addMessage("system", error.message, "error"));
});
document.getElementById("loadModelsBtn").addEventListener("click", () => {
  loadModels().catch((error) => addMessage("system", error.message, "error"));
});
document.getElementById("uploadBtn").addEventListener("click", () => {
  uploadDataset().catch((error) => addMessage("system", error.message, "error"));
});
evaluateBtn.addEventListener("click", evaluateLatest);
openEvaluationBtn.addEventListener("click", openEvaluationModal);
closeEvaluationBtn.addEventListener("click", closeEvaluationModal);
evaluationModal.addEventListener("click", (event) => {
  if (event.target === evaluationModal) closeEvaluationModal();
});
document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") closeEvaluationModal();
});
messageForm.addEventListener("submit", sendMessage);

loadModels().catch(() => {});
