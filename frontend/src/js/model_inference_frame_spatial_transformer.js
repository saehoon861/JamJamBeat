// [model_inference_frame_spatial_transformer.js] Frame-based Spatial Transformer 모델 추론 엔진
// pos_scale 정규화를 사용하는 단일 프레임 모델 전용

import * as ort from "onnxruntime-web";

const DEFAULT_TAU = 0.65;
const DEFAULT_REQUEST_INTERVAL_MS = 60;
const POS_SCALE_EPS = 1e-8;
const ORIGIN_IDX = 0; // wrist landmark
const SCALE_IDX = 9; // middle finger MCP landmark

const LEFT_HAND_MIRROR_ENABLED = (() => {
  const raw = new URLSearchParams(window.location.search).get("leftHandMirror");
  if (raw === "0" || raw === "false" || raw === "off") return false;
  return true;
})();

const PERF_ENABLED = (() => {
  const raw = new URLSearchParams(window.location.search).get("profilePerf");
  if (raw === "1" || raw === "true") return true;
  if (raw === "0" || raw === "false") return false;
  return Boolean(import.meta.env.DEV);
})();

const PERF_LOG_INTERVAL_MS = 2000;
const ORT_CDN_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
const INIT_RETRY_INTERVAL_MS = 1500;

function normalizeRootPath(value, fallback = "/") {
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  if (!trimmed) return fallback;
  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed.replace(/\/+$/, "") + "/";
  }
  const normalized = "/" + trimmed.replace(/^\/+/, "").replace(/\/+$/, "") + "/";
  return normalized.replace(/\/+/g, "/");
}

function resolveRuntimeRoot() {
  const params = new URLSearchParams(window.location.search);
  const queryRoot = params.get("runtimeRoot");
  const globalRoot = window.__JAMJAM_RUNTIME_ROOT;
  const envRoot = import.meta.env.VITE_RUNTIME_ROOT;
  const baseUrl = import.meta.env.BASE_URL || "/";

  if (queryRoot && queryRoot.trim()) return normalizeRootPath(queryRoot, "/runtime_frame_spatial_transformer/");
  if (typeof globalRoot === "string" && globalRoot.trim()) return normalizeRootPath(globalRoot, "/runtime_frame_spatial_transformer/");
  if (typeof envRoot === "string" && envRoot.trim()) return normalizeRootPath(envRoot, "/runtime_frame_spatial_transformer/");

  return normalizeRootPath(baseUrl + "runtime_frame_spatial_transformer", "/runtime_frame_spatial_transformer/");
}

function resolveOrtWasmRoot() {
  const params = new URLSearchParams(window.location.search);
  const queryRoot = params.get("ortWasmRoot");
  const globalRoot = window.__JAMJAM_ORT_WASM_ROOT;
  const envRoot = import.meta.env.VITE_ORT_WASM_ROOT;
  const baseUrl = import.meta.env.BASE_URL || "/";

  if (queryRoot && queryRoot.trim()) return normalizeRootPath(queryRoot, "/");
  if (typeof globalRoot === "string" && globalRoot.trim()) return normalizeRootPath(globalRoot, "/");
  if (typeof envRoot === "string" && envRoot.trim()) return normalizeRootPath(envRoot, "/");

  return normalizeRootPath(baseUrl, "/");
}

const RUNTIME_ROOT = resolveRuntimeRoot();
const ORT_LOCAL_WASM_BASE = resolveOrtWasmRoot();
const MODEL_PATH = RUNTIME_ROOT + "model.onnx";
const CLASS_NAMES_PATH = RUNTIME_ROOT + "class_names.json";
const MODEL_EXTERNAL_DATA_PATH = RUNTIME_ROOT + "model.onnx.data";

let ortApi = null;
let onnxSession = null;
let classNames = ["neutral", "fist", "open_palm", "V", "pinky", "animal", "k-heart"];
let isInitializing = false;
let initializationError = null;
let lastInitFailedAt = 0;

const requestStateByHand = new Map();
let globalRequestInFlight = false;
let lastGlobalRequestAt = 0;

const perfWindow = {
  startedAt: performance.now(),
  lastLogAt: performance.now(),
  requestCount: 0,
  successCount: 0,
  failureCount: 0,
  totalMs: 0,
  maxMs: 0
};

function normalizeHandKey(handKey = "default") {
  return String(handKey || "default").trim().toLowerCase();
}

function isInferenceEnabledForHand(handKey = "default") {
  return normalizeHandKey(handKey) !== "left";
}

function createDisabledPrediction() {
  return {
    label: "None",
    confidence: 0,
    classId: 0,
    source: "disabled",
    disabled: true,
    status: "disabled"
  };
}

function createNoHandPrediction() {
  return {
    label: "neutral",
    classId: 0,
    confidence: 0,
    probs: [1, 0, 0, 0, 0, 0, 0],
    status: "no_hand",
    mode: "frame",
    framesCollected: 0,
    effective_seq_len: 1,
    model_seq_len: 1,
    padded_seq_input: false,
    source: "frame_spatial_transformer"
  };
}

function flushPerf(now = performance.now()) {
  if (!PERF_ENABLED || now - perfWindow.lastLogAt < PERF_LOG_INTERVAL_MS) return;
  const requestCount = Math.max(1, perfWindow.requestCount);
  console.info("[Perf][FrameSpatialTransformer]", {
    windowMs: Math.round(now - perfWindow.startedAt),
    requests: perfWindow.requestCount,
    successes: perfWindow.successCount,
    failures: perfWindow.failureCount,
    avgMs: Number((perfWindow.totalMs / requestCount).toFixed(2)),
    maxMs: Number(perfWindow.maxMs.toFixed(2)),
    mode: "onnx-frame-spatial-transformer"
  });
  perfWindow.startedAt = now;
  perfWindow.lastLogAt = now;
  perfWindow.requestCount = 0;
  perfWindow.successCount = 0;
  perfWindow.failureCount = 0;
  perfWindow.totalMs = 0;
  perfWindow.maxMs = 0;
}

function getRequestIntervalMs() {
  const raw = Number(new URLSearchParams(window.location.search).get("modelIntervalMs"));
  if (!Number.isFinite(raw)) return DEFAULT_REQUEST_INTERVAL_MS;
  return Math.max(60, Math.min(400, Math.round(raw)));
}

function getGlobalRequestGapMs() {
  return Math.max(45, Math.round(getRequestIntervalMs() * 0.5));
}

function getHandRequestState(handKey = "default") {
  if (!requestStateByHand.has(handKey)) {
    requestStateByHand.set(handKey, {
      lastRequestAt: 0,
      inFlight: false,
      lastPrediction: null
    });
  }
  return requestStateByHand.get(handKey);
}

function isLikelyWasmBootError(error) {
  const message = String(error?.message || error || "").toLowerCase();
  return (
    message.includes("wasm") ||
    message.includes("backend") ||
    message.includes("fetch") ||
    message.includes("instantiate") ||
    message.includes("no available backend")
  );
}

async function createOnnxSession(ort, wasmBase) {
  if (ort?.env?.wasm) {
    ort.env.wasm.wasmPaths = wasmBase;
    if (typeof SharedArrayBuffer === "undefined") {
      ort.env.wasm.numThreads = 1;
    }
  }

  return ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    externalData: [
      {
        path: "model.onnx.data",
        data: MODEL_EXTERNAL_DATA_PATH
      }
    ]
  });
}

function readCandidateShape(candidate) {
  const isObjectLike = candidate && (typeof candidate === "object" || typeof candidate === "function");
  if (!isObjectLike) {
    return {
      session: null,
      tensor: null,
      env: null,
      keys: []
    };
  }

  const session = candidate.InferenceSession || candidate?.default?.InferenceSession || null;
  const tensor = candidate.Tensor || candidate?.default?.Tensor || null;
  const env = candidate.env || candidate?.default?.env || null;
  const keys = Object.keys(candidate || {}).slice(0, 20);

  return { session, tensor, env, keys };
}

function buildOrtApi(candidate) {
  const shape = readCandidateShape(candidate);
  if (!shape.session || typeof shape.session.create !== "function") return null;

  const fallbackTensor =
    readCandidateShape(ort).tensor ||
    readCandidateShape(globalThis?.ort).tensor ||
    null;

  const tensorCtor = shape.tensor || fallbackTensor;
  if (!tensorCtor) return null;

  return {
    InferenceSession: shape.session,
    Tensor: tensorCtor,
    env: shape.env || readCandidateShape(ort).env || null
  };
}

async function ensureOrtApi() {
  if (ortApi) return ortApi;

  const debugRows = [];
  const inspect = (label, candidate) => {
    const shape = readCandidateShape(candidate);
    debugRows.push({
      label,
      hasSessionCreate: Boolean(shape.session && typeof shape.session.create === "function"),
      hasTensor: Boolean(shape.tensor),
      hasEnv: Boolean(shape.env),
      keys: shape.keys
    });
    return buildOrtApi(candidate);
  };

  ortApi = inspect("ort", ort);

  if (!ortApi) {
    ortApi = inspect("ort.default", ort?.default);
  }

  if (!ortApi) {
    ortApi = inspect("globalThis.ort", globalThis?.ort) || inspect("globalThis.ort.default", globalThis?.ort?.default);
  }

  if (!ortApi) {
    const mod = await import("onnxruntime-web");
    ortApi = inspect("dynamicImport", mod) || inspect("dynamicImport.default", mod?.default);
  }

  if (!ortApi) {
    const wasmMod = await import("onnxruntime-web/wasm");
    ortApi = inspect("wasmImport", wasmMod) || inspect("wasmImport.default", wasmMod?.default);
  }

  if (!ortApi) {
    const debugText = debugRows
      .map((row) => `${row.label}: create=${row.hasSessionCreate}, tensor=${row.hasTensor}, env=${row.hasEnv}, keys=[${row.keys.join(",")}]`)
      .join(" | ");
    throw new Error(`onnxruntime-web 초기화 실패: InferenceSession/Tensor API를 찾지 못했습니다. ${debugText}`);
  }

  if (ortApi.env?.wasm) {
    if (!ortApi.env.wasm.wasmPaths) {
      ortApi.env.wasm.wasmPaths = ORT_LOCAL_WASM_BASE;
    }
    if (typeof SharedArrayBuffer === "undefined") {
      ortApi.env.wasm.numThreads = 1;
    }
  }

  return ortApi;
}

async function initializeModel() {
  if (onnxSession) return true;
  if (isInitializing) {
    while (isInitializing) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    return onnxSession !== null;
  }
  if (initializationError) {
    if (performance.now() - lastInitFailedAt < INIT_RETRY_INTERVAL_MS) return false;
    initializationError = null;
  }

  isInitializing = true;
  try {
    const classResponse = await fetch(CLASS_NAMES_PATH);
    if (!classResponse.ok) throw new Error(`Failed to load class names: ${classResponse.status}`);
    classNames = await classResponse.json();

    const ortInstance = await ensureOrtApi();

    try {
      onnxSession = await createOnnxSession(ortInstance, ORT_LOCAL_WASM_BASE);
    } catch (localError) {
      const shouldRetryWithCdn = ORT_LOCAL_WASM_BASE !== ORT_CDN_WASM_BASE && isLikelyWasmBootError(localError);
      if (!shouldRetryWithCdn) throw localError;

      console.warn("[FrameSpatialTransformer] local WASM 경로 실패, CDN으로 재시도합니다.", {
        localWasmBase: ORT_LOCAL_WASM_BASE,
        cdnWasmBase: ORT_CDN_WASM_BASE,
        error: String(localError?.message || localError || "")
      });

      onnxSession = await createOnnxSession(ortInstance, ORT_CDN_WASM_BASE);
    }

    console.info("[FrameSpatialTransformer] ✅ 모델 로드 완료", {
      modelPath: MODEL_PATH,
      runtimeRoot: RUNTIME_ROOT,
      wasmBase: ortInstance?.env?.wasm?.wasmPaths,
      classes: classNames.length,
      inputNames: onnxSession.inputNames,
      outputNames: onnxSession.outputNames
    });

    initializationError = null;
    lastInitFailedAt = 0;
    isInitializing = false;
    return true;
  } catch (error) {
    console.error("[FrameSpatialTransformer] ❌ 모델 로드 실패:", error);
    initializationError = error;
    lastInitFailedAt = performance.now();
    isInitializing = false;
    return false;
  }
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sumExps);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getMirrorPivotX(landmarks) {
  const wristX = Number.isFinite(landmarks?.[0]?.x) ? landmarks[0].x : 0.5;
  const indexMcpX = Number.isFinite(landmarks?.[5]?.x) ? landmarks[5].x : wristX;
  const pinkyMcpX = Number.isFinite(landmarks?.[17]?.x) ? landmarks[17].x : wristX;
  return (wristX + indexMcpX + pinkyMcpX) / 3;
}

// MediaPipe 랜드마크를 63차원 배열로 변환 (미러링 포함)
function sanitizeLandmarksTo63(landmarks, handKey = "default") {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return null;

  const normalizedHandKey = String(handKey || "default").trim().toLowerCase();
  const shouldMirrorLeft = LEFT_HAND_MIRROR_ENABLED && normalizedHandKey === "left";
  const mirrorPivotX = shouldMirrorLeft ? getMirrorPivotX(landmarks) : 0;

  const features = new Float32Array(63);
  for (let i = 0; i < 21; i++) {
    const point = landmarks[i];
    const rawX = Number.isFinite(point?.x) ? point.x : 0;
    const mirroredX = shouldMirrorLeft ? clamp(mirrorPivotX * 2 - rawX, 0, 1) : rawX;
    const offset = i * 3;
    features[offset] = mirroredX;
    features[offset + 1] = Number.isFinite(point?.y) ? point.y : 0;
    features[offset + 2] = Number.isFinite(point?.z) ? point.z : 0;
  }
  return features;
}

// pos_scale 정규화: (pts - pts[0]) / ||pts[9] - pts[0]||
function normalizePosScale(features63) {
  if (!features63 || features63.length !== 63) return null;

  const originX = features63[0];
  const originY = features63[1];
  const originZ = features63[2];

  // pts[9] = middle finger MCP (인덱스 27, 28, 29)
  const scaleX = features63[27] - originX;
  const scaleY = features63[28] - originY;
  const scaleZ = features63[29] - originZ;
  const denom = Math.hypot(scaleX, scaleY, scaleZ);
  const scale = denom <= POS_SCALE_EPS ? 1 : 1 / denom;

  const normalized = new Float32Array(63);
  for (let i = 0; i < 63; i += 3) {
    normalized[i] = (features63[i] - originX) * scale;
    normalized[i + 1] = (features63[i + 1] - originY) * scale;
    normalized[i + 2] = (features63[i + 2] - originZ) * scale;
  }
  return normalized;
}

function normalizePrediction(predIndex, confidence, probs, tsMs, tau = DEFAULT_TAU) {
  let finalIndex = predIndex;
  let tauNeutralized = false;

  // tau 후처리: 신뢰도가 임계값보다 낮으면 neutral(0)로 강제
  if (confidence < tau && predIndex !== 0) {
    finalIndex = 0;
    tauNeutralized = true;
  }

  return {
    label: classNames[finalIndex] || "None",
    confidence,
    probs,
    classId: finalIndex,
    modelVersion: "frame-spatial-transformer",
    source: "frame_spatial_transformer",
    ts_ms: tsMs,
    tau_applied: tau,
    tau_neutralized: tauNeutralized,
    raw_pred_index: predIndex,
    raw_pred_label: classNames[predIndex] || "None",
    status: "ready",
    mode: "frame",
    framesCollected: 1,
    effective_seq_len: 1,
    model_seq_len: 1,
    padded_seq_input: false
  };
}

async function runFrameInference(session, landmarks, handKey = "default") {
  if (!landmarks || !Array.isArray(landmarks) || landmarks.length < 21) {
    return createNoHandPrediction();
  }

  const raw63 = sanitizeLandmarksTo63(landmarks, handKey);
  const normalized63 = normalizePosScale(raw63);
  if (!normalized63) {
    return createNoHandPrediction();
  }

  const ortInstance = await ensureOrtApi();
  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const tensor = new ortInstance.Tensor("float32", normalized63, [1, 21, 3]);
  const results = await session.run({ [inputName]: tensor });
  const outputTensor = results[outputName] || results[Object.keys(results)[0]];
  const logits = Array.from(outputTensor.data);
  const probs = softmax(logits);
  const predIndex = probs.indexOf(Math.max(...probs));
  const confidence = probs[predIndex];

  return normalizePrediction(predIndex, confidence, probs, Math.round(performance.now()));
}

async function scheduleModelRequest(landmarks, now, handKey = "default") {
  const handState = getHandRequestState(handKey);
  if (!isInferenceEnabledForHand(handKey)) {
    handState.inFlight = false;
    handState.lastRequestAt = 0;
    handState.lastPrediction = createDisabledPrediction();
    return;
  }

  const requestIntervalMs = getRequestIntervalMs();
  if (!onnxSession) {
    const initialized = await initializeModel();
    if (!initialized) return;
  }

  if (handState.inFlight) return;
  if (globalRequestInFlight) return;
  if (now - handState.lastRequestAt < requestIntervalMs) return;
  if (now - lastGlobalRequestAt < getGlobalRequestGapMs()) return;

  globalRequestInFlight = true;
  lastGlobalRequestAt = now;
  handState.inFlight = true;
  handState.lastRequestAt = now;
  const requestStartedAt = PERF_ENABLED ? performance.now() : 0;
  if (PERF_ENABLED) perfWindow.requestCount += 1;

  try {
    const prediction = await runFrameInference(onnxSession, landmarks, handKey);
    handState.lastPrediction = prediction;

    if (PERF_ENABLED) {
      const elapsedMs = performance.now() - requestStartedAt;
      perfWindow.successCount += 1;
      perfWindow.totalMs += elapsedMs;
      perfWindow.maxMs = Math.max(perfWindow.maxMs, elapsedMs);
    }
  } catch (error) {
    console.error("[FrameSpatialTransformer] 추론 실패:", error);
    if (PERF_ENABLED) {
      const elapsedMs = performance.now() - requestStartedAt;
      perfWindow.failureCount += 1;
      perfWindow.totalMs += elapsedMs;
      perfWindow.maxMs = Math.max(perfWindow.maxMs, elapsedMs);
    }
  } finally {
    globalRequestInFlight = false;
    handState.inFlight = false;
    flushPerf();
  }
}

// 메인 API: 손동작 예측 가져오기
export function getModelPrediction(landmarks, now = performance.now(), handKey = "default") {
  const handState = getHandRequestState(handKey);
  if (!isInferenceEnabledForHand(handKey)) {
    handState.inFlight = false;
    handState.lastRequestAt = 0;
    handState.lastPrediction = createDisabledPrediction();
    return handState.lastPrediction;
  }

  // 손이 없으면 no_hand 처리
  if (!landmarks || !Array.isArray(landmarks) || landmarks.length < 21) {
    handState.lastPrediction = createNoHandPrediction();
    return handState.lastPrediction;
  }

  scheduleModelRequest(landmarks, now, handKey);
  return handState.lastPrediction;
}

export function getModelInferenceStatus(now = performance.now()) {
  const states = [...requestStateByHand.values()];
  return {
    endpointConfigured: onnxSession !== null,
    inFlight: states.some((state) => state.inFlight),
    disabled: false,
    mode: "onnx-frame-spatial-transformer"
  };
}

export function preloadModel() {
  return initializeModel();
}
