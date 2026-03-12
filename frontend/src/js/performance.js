import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";
import * as Renderer from "./renderer.js";
import { resolveGesture } from "./gestures.js";
import { getModelPrediction, getModelInferenceStatus } from "./model_inference.js";

const PERF_LOG_KEY = "jamjam.perf.logs.v1";

const video = document.getElementById("perfVideo");
const canvas = document.getElementById("perfCanvas");
const ctx = canvas.getContext("2d");

const statusEl = document.getElementById("perfStatus");
const testBtn = document.getElementById("perfTestButton");
const resetBtn = document.getElementById("perfClearButton");
const ruleToggleBtn = document.getElementById("ruleToggleButton");
const ruleModeBadge = document.getElementById("ruleModeBadge");

const rawLabelEl = document.getElementById("rawLabel");
const rawConfidenceEl = document.getElementById("rawConfidence");
const rawVersionEl = document.getElementById("rawVersion");
const rawClassIdEl = document.getElementById("rawClassId");
const finalLabelEl = document.getElementById("finalLabel");
const finalSourceEl = document.getElementById("finalSource");

const netEndpointEl = document.getElementById("netEndpoint");
const netConfiguredEl = document.getElementById("netConfigured");
const netInFlightEl = document.getElementById("netInFlight");
const netDisabledEl = document.getElementById("netDisabled");
const landmarkStateEl = document.getElementById("landmarkState");
const benchResultEl = document.getElementById("benchResult");
const soundLatencySummaryEl = document.getElementById("soundLatencySummary");
const audioUnlockSummaryEl = document.getElementById("audioUnlockSummary");
const soundLatencyBodyEl = document.getElementById("soundLatencyBody");

let handLandmarker = null;
let cameraStream = null;
let lastVideoTime = -1;
let lastInferenceAt = 0;
let latestLandmarks = null;
let pendingBenchmark = false;

const DEFAULT_INFER_FPS = 30;
const MIN_INFER_FPS = 8;
const MAX_INFER_FPS = 60;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getInferIntervalMs() {
  const raw = Number(new URLSearchParams(window.location.search).get("inferFps"));
  const fps = Number.isFinite(raw) ? clamp(Math.round(raw), MIN_INFER_FPS, MAX_INFER_FPS) : DEFAULT_INFER_FPS;
  return Math.round(1000 / fps);
}

const INFER_INTERVAL_MS = getInferIntervalMs();

function getConfiguredEndpoint() {
  const queryEndpoint = new URLSearchParams(window.location.search).get("inferEndpoint");
  if (queryEndpoint && queryEndpoint.trim()) return queryEndpoint.trim();

  const globalEndpoint = window.__JAMJAM_MODEL_ENDPOINT;
  if (typeof globalEndpoint === "string" && globalEndpoint.trim()) return globalEndpoint.trim();

  return null;
}

function parseDelegate() {
  const raw = (new URLSearchParams(window.location.search).get("mpDelegate") || "gpu").trim().toUpperCase();
  return raw === "CPU" ? "CPU" : "GPU";
}

function getGestureModeFromUrl() {
  const raw = (new URLSearchParams(window.location.search).get("gestureMode") || "hybrid").trim().toLowerCase();
  if (raw === "rules" || raw === "model" || raw === "hybrid") return raw;
  return "hybrid";
}

function updateRuleToggleButton() {
  const mode = getGestureModeFromUrl();
  const enabled = mode !== "model";
  ruleToggleBtn.textContent = `규칙기반: ${enabled ? "ON" : "OFF"}`;
  if (enabled) {
    statusEl.textContent = "규칙기반 실행중";
    if (ruleModeBadge) ruleModeBadge.style.display = "block";
  } else {
    if (ruleModeBadge) ruleModeBadge.style.display = "none";
  }
}

function toggleRuleBasedMode() {
  const params = new URLSearchParams(window.location.search);
  const mode = getGestureModeFromUrl();
  const nextMode = mode === "model" ? "hybrid" : "model";
  params.set("gestureMode", nextMode);
  if (!params.has("interactionMode")) params.set("interactionMode", "gesture");
  window.location.search = params.toString();
}

function setCanvasSize() {
  const rect = video.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width));
  canvas.height = Math.max(1, Math.round(rect.height));
}

function formatNumber(value, digits = 3) {
  if (!Number.isFinite(value)) return "-";
  return value.toFixed(digits);
}

function readLatencyLogs() {
  try {
    const raw = localStorage.getItem(PERF_LOG_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function toDisplayGestureLabel(label, classId = null) {
  const normalized = String(label || "").trim().toLowerCase();
  if (normalized === "none" || normalized === "class0" || classId === 0) return "아무것도 아님";
  if (normalized === "fist" || normalized === "first" || normalized === "class1" || classId === 1) return "주먹";
  if (
    normalized === "openpalm" ||
    normalized === "open palm" ||
    normalized === "open_palm" ||
    normalized === "paper" ||
    normalized === "class2" ||
    classId === 2
  ) return "손바닥";
  if (normalized === "v" || normalized === "class3" || classId === 3) return "브이";
  if (normalized === "pinky" || normalized === "pinky class" || normalized === "pinky_class" || normalized === "class4" || normalized === "class 4" || classId === 4) return "새끼손가락";
  if (normalized === "animal" || normalized === "class5" || classId === 5) return "애니멀";
  if (normalized === "kheart" || normalized === "k-heart" || normalized === "is_k_heart" || normalized === "class6" || classId === 6) return "K-하트";
  if (!label) return "아무것도 아님";
  return String(label);
}

function formatMs(value) {
  return Number.isFinite(value) ? value.toFixed(1) : "-";
}

function renderLatencyLogPanel() {
  const logs = readLatencyLogs();
  const soundLogs = logs.filter((log) => log?.soundKey !== "audio_unlock" && log?.soundKey !== "audio_unlock_fail");
  const unlockLogs = logs.filter((log) => log?.soundKey === "audio_unlock" || log?.soundKey === "audio_unlock_fail");

  const latencyValues = soundLogs
    .map((log) => Number(log?.latencyMs))
    .filter((n) => Number.isFinite(n))
    .sort((a, b) => a - b);

  const unlockValues = unlockLogs
    .map((log) => Number(log?.latencyMs))
    .filter((n) => Number.isFinite(n))
    .sort((a, b) => a - b);

  if (latencyValues.length === 0) {
    soundLatencySummaryEl.textContent = "아직 기록 없음";
  } else {
    const avg = latencyValues.reduce((sum, n) => sum + n, 0) / latencyValues.length;
    const p95 = latencyValues[Math.min(latencyValues.length - 1, Math.round((latencyValues.length - 1) * 0.95))];
    const max = latencyValues[latencyValues.length - 1];
    soundLatencySummaryEl.textContent = `건수 ${latencyValues.length}, avg ${avg.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms, max ${max.toFixed(1)}ms`;
  }

  if (unlockValues.length === 0) {
    audioUnlockSummaryEl.textContent = "아직 기록 없음";
  } else {
    const avg = unlockValues.reduce((sum, n) => sum + n, 0) / unlockValues.length;
    const p95 = unlockValues[Math.min(unlockValues.length - 1, Math.round((unlockValues.length - 1) * 0.95))];
    const max = unlockValues[unlockValues.length - 1];
    audioUnlockSummaryEl.textContent = `건수 ${unlockValues.length}, avg ${avg.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms, max ${max.toFixed(1)}ms`;
  }

  soundLatencyBodyEl.innerHTML = "";
  const rows = logs.slice(-40).reverse();
  rows.forEach((log) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td style="padding:6px; border-bottom:1px solid rgba(255,255,255,0.1);">${new Date(log.at).toLocaleTimeString()}</td>
      <td style="padding:6px; border-bottom:1px solid rgba(255,255,255,0.1);">${toDisplayGestureLabel(log.gestureLabel || "None")}</td>
      <td style="padding:6px; border-bottom:1px solid rgba(255,255,255,0.1);">${log.instrumentId || "-"}</td>
      <td style="padding:6px; border-bottom:1px solid rgba(255,255,255,0.1);">${log.soundKey || "-"}</td>
      <td style="padding:6px; border-bottom:1px solid rgba(255,255,255,0.1);">${log.gestureSource || "-"}</td>
      <td style="padding:6px; border-bottom:1px solid rgba(255,255,255,0.1); text-align:right;">${formatMs(log.latencyMs)}</td>
    `;
    soundLatencyBodyEl.appendChild(tr);
  });
}

function updateNetworkPanel() {
  const status = getModelInferenceStatus(performance.now());
  netEndpointEl.textContent = getConfiguredEndpoint() || "(없음)";
  netConfiguredEl.textContent = status.endpointConfigured ? "Yes" : "No";
  netInFlightEl.textContent = status.inFlight ? "Yes" : "No";
  netDisabledEl.textContent = status.disabled ? "Yes" : "No";
  landmarkStateEl.textContent = latestLandmarks ? "Detected" : "None";
  renderLatencyLogPanel();
}

async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const modelAssetPath = new URL("../../public/hand_landmarker.task", import.meta.url).toString();

  const preferred = parseDelegate();
  const fallback = preferred === "GPU" ? "CPU" : "GPU";

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath, delegate: preferred },
      runningMode: "VIDEO",
      numHands: 2
    });
  } catch {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath, delegate: fallback },
      runningMode: "VIDEO",
      numHands: 2
    });
  }
}

async function initCamera() {
  cameraStream = await navigator.mediaDevices.getUserMedia({
    video: {
      width: { ideal: 960 },
      height: { ideal: 540 },
      facingMode: "user"
    },
    audio: false
  });
  video.srcObject = cameraStream;
  await video.play();
  setCanvasSize();
}

function resetPanel() {
  rawLabelEl.textContent = "-";
  rawConfidenceEl.textContent = "-";
  rawVersionEl.textContent = "-";
  rawClassIdEl.textContent = "-";
  finalLabelEl.textContent = "-";
  finalSourceEl.textContent = "-";
  benchResultEl.textContent = "아직 실행 전";
}

function drawAndInfer() {
  if (!handLandmarker) {
    requestAnimationFrame(drawAndInfer);
    return;
  }

  const now = performance.now();
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  updateNetworkPanel();

  const hasFreshFrame = video.currentTime !== lastVideoTime && video.readyState >= 2 && video.videoWidth > 0;
  const inferenceDue = now - lastInferenceAt >= INFER_INTERVAL_MS;

  if (hasFreshFrame && inferenceDue) {
    lastVideoTime = video.currentTime;
    lastInferenceAt = now;

    try {
      const result = handLandmarker.detectForVideo(video, now);
      if (result.landmarks.length > 0) {
        latestLandmarks = result.landmarks[0];
        Renderer.drawHand(ctx, latestLandmarks, canvas, now * 0.001);

        if (pendingBenchmark && !testBtn.disabled) {
          pendingBenchmark = false;
          runBenchmark();
        }

        const raw = getModelPrediction(latestLandmarks, now);
        const finalGesture = resolveGesture(latestLandmarks, now, true);

        rawLabelEl.textContent = toDisplayGestureLabel(raw?.label, raw?.classId ?? null);
        rawConfidenceEl.textContent = formatNumber(raw?.confidence, 3);
        rawVersionEl.textContent = raw?.modelVersion ?? "-";
        rawClassIdEl.textContent = raw?.classId ?? "-";

        finalLabelEl.textContent = toDisplayGestureLabel(finalGesture?.label ?? "None");
        finalSourceEl.textContent = finalGesture?.source ?? "-";
      } else {
        latestLandmarks = null;
        resetPanel();
      }
    } catch {
      statusEl.textContent = "추론 중 오류가 발생했습니다. 모델 서버 상태를 확인해주세요.";
    }
  }

  requestAnimationFrame(drawAndInfer);
}

function getBenchmarkPayload() {
  const landmarks = Array.isArray(latestLandmarks) && latestLandmarks.length >= 21
    ? latestLandmarks.slice(0, 21).map((p) => ({
      x: Number.isFinite(p?.x) ? p.x : 0,
      y: Number.isFinite(p?.y) ? p.y : 0,
      z: Number.isFinite(p?.z) ? p.z : 0
    }))
    : null;

  if (!landmarks) return null;

  return {
    version: 1,
    ts_ms: Math.round(performance.now()),
    landmarks
  };
}

async function postWithTimeout(url, payload, timeoutMs) {
  const controller = new AbortController();
  const timeoutId = window.setTimeout(() => controller.abort(), timeoutMs);
  try {
    const res = await fetch(url, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal
    });
    return res;
  } finally {
    clearTimeout(timeoutId);
  }
}

function percentile(sortedValues, p) {
  if (sortedValues.length === 0) return 0;
  const idx = Math.min(sortedValues.length - 1, Math.max(0, Math.round((sortedValues.length - 1) * p)));
  return sortedValues[idx];
}

async function runBenchmark() {
  const endpoint = getConfiguredEndpoint();
  if (!endpoint) {
    benchResultEl.textContent = "Endpoint 없음";
    return;
  }

  const payload = getBenchmarkPayload();
  if (!payload) {
    pendingBenchmark = true;
    benchResultEl.textContent = "손 감지 대기 중... 감지되면 자동으로 테스트를 시작합니다.";
    return;
  }

  testBtn.disabled = true;
  benchResultEl.textContent = "테스트 중...";

  const total = 30;
  const latencies = [];
  let success = 0;

  for (let i = 0; i < total; i += 1) {
    const started = performance.now();
    try {
      const res = await postWithTimeout(endpoint, payload, 1200);
      if (res.ok) {
        success += 1;
        latencies.push(performance.now() - started);
      }
    } catch {
      // ignore
    }
  }

  testBtn.disabled = false;

  if (latencies.length === 0) {
    benchResultEl.textContent = "실패: 응답 없음";
    return;
  }

  const sorted = latencies.slice().sort((a, b) => a - b);
  const avg = latencies.reduce((sum, n) => sum + n, 0) / latencies.length;
  const p95 = percentile(sorted, 0.95);
  const max = sorted[sorted.length - 1];
  const successRate = (success / total) * 100;
  benchResultEl.textContent = `성공 ${success}/${total} (${successRate.toFixed(0)}%), avg ${avg.toFixed(1)}ms, p95 ${p95.toFixed(1)}ms, max ${max.toFixed(1)}ms`;
}

async function init() {
  statusEl.textContent = "카메라/모델을 준비 중입니다...";
  try {
    await Promise.all([initMediaPipe(), initCamera()]);
    statusEl.textContent = "손 인식 관측 중입니다. (이 페이지는 소리를 재생하지 않습니다.)";
    drawAndInfer();
  } catch {
    statusEl.textContent = "초기화 실패: 카메라 권한/모델 서버를 확인해주세요.";
  }
}

testBtn.addEventListener("click", runBenchmark);
resetBtn.addEventListener("click", () => {
  latestLandmarks = null;
  pendingBenchmark = false;
  resetPanel();
  localStorage.removeItem(PERF_LOG_KEY);
  renderLatencyLogPanel();
  statusEl.textContent = "표시값을 초기화했습니다.";
});

ruleToggleBtn.addEventListener("click", toggleRuleBasedMode);

window.addEventListener("resize", setCanvasSize);
window.addEventListener("beforeunload", () => {
  if (!cameraStream) return;
  cameraStream.getTracks().forEach((track) => {
    track.stop();
  });
});

resetPanel();
updateRuleToggleButton();
init();
