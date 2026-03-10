const REQUEST_TIMEOUT_MS = 120;
const REQUEST_INTERVAL_MS = 90;
const FAIL_OPEN_AFTER = 3;
const DISABLE_FOR_MS = 3000;

let lastRequestAt = 0;
let inFlight = false;
let consecutiveFailures = 0;
let disabledUntil = 0;
let lastPrediction = null;

function getConfiguredEndpoint() {
  const queryEndpoint = new URLSearchParams(window.location.search).get("inferEndpoint");
  if (queryEndpoint && queryEndpoint.trim()) return queryEndpoint.trim();

  const globalEndpoint = window.__JAMJAM_MODEL_ENDPOINT;
  if (typeof globalEndpoint === "string" && globalEndpoint.trim()) return globalEndpoint.trim();

  return null;
}

function sanitizeLandmarks(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return null;

  return landmarks.slice(0, 21).map((point) => ({
    x: Number.isFinite(point?.x) ? point.x : 0,
    y: Number.isFinite(point?.y) ? point.y : 0,
    z: Number.isFinite(point?.z) ? point.z : 0
  }));
}

function normalizePrediction(json, tsMs) {
  if (!json || typeof json !== "object") return null;

  const rawLabel = typeof json.label === "string" ? json.label : "None";
  const confidence = Number(json.confidence);

  return {
    label: rawLabel,
    confidence: Number.isFinite(confidence) ? confidence : 0,
    probs: Array.isArray(json.probs) ? json.probs : null,
    modelVersion: typeof json.model_version === "string" ? json.model_version : "unknown",
    source: "model",
    ts_ms: tsMs
  };
}

function scheduleModelRequest(landmarks, now) {
  const endpoint = getConfiguredEndpoint();
  if (!endpoint) return;
  if (inFlight) return;
  if (now < disabledUntil) return;
  if (now - lastRequestAt < REQUEST_INTERVAL_MS) return;

  const payloadLandmarks = sanitizeLandmarks(landmarks);
  if (!payloadLandmarks) return;

  const tsMs = Math.round(now);
  const payload = {
    version: 1,
    ts_ms: tsMs,
    landmarks: payloadLandmarks
  };

  inFlight = true;
  lastRequestAt = now;

  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
    signal: controller.signal
  })
    .then((response) => {
      if (!response.ok) throw new Error("HTTP_" + response.status);
      return response.json();
    })
    .then((json) => {
      const prediction = normalizePrediction(json, tsMs);
      if (!prediction) throw new Error("INVALID_PAYLOAD");

      lastPrediction = prediction;
      consecutiveFailures = 0;
    })
    .catch(() => {
      consecutiveFailures += 1;
      if (consecutiveFailures >= FAIL_OPEN_AFTER) {
        disabledUntil = performance.now() + DISABLE_FOR_MS;
        consecutiveFailures = 0;
      }
    })
    .finally(() => {
      clearTimeout(timeoutId);
      inFlight = false;
    });
}

export function getModelPrediction(landmarks, now = performance.now()) {
  scheduleModelRequest(landmarks, now);
  return lastPrediction;
}

export function getModelInferenceStatus(now = performance.now()) {
  return {
    endpointConfigured: Boolean(getConfiguredEndpoint()),
    inFlight,
    disabled: now < disabledUntil
  };
}
