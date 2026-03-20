// [model_inference_sequence.js] 시퀀스 기반 손동작 인식 모델 추론 엔진
// pos_scale_mlp_sequence_delta 모델 전용 - 8프레임 시퀀스를 버퍼링하고 델타 특성을 계산합니다.

import * as ort from "onnxruntime-web";

const DEFAULT_TAU = 0.85;
const DEFAULT_REQUEST_INTERVAL_MS = 150;
const SEQUENCE_LENGTH = 8; // 모델이 요구하는 시퀀스 길이
const FEATURE_DIM = 126; // 관절63 + 델타63
const EPS = 1e-8; // pos_scale 정규화 시 0으로 나누기 방지

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

  if (queryRoot && queryRoot.trim()) return normalizeRootPath(queryRoot, "/runtime_sequence/");
  if (typeof globalRoot === "string" && globalRoot.trim()) return normalizeRootPath(globalRoot, "/runtime_sequence/");
  if (typeof envRoot === "string" && envRoot.trim()) return normalizeRootPath(envRoot, "/runtime_sequence/");

  return normalizeRootPath(baseUrl + "runtime_sequence", "/runtime_sequence/");
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

// 손별 상태 관리 (버퍼, 요청 상태 등)
const handStateByKey = new Map();
let globalRequestInFlight = false;
let lastGlobalRequestAt = 0;

const perfWindow = {
  startedAt: performance.now(),
  lastLogAt: performance.now(),
  requestCount: 0,
  successCount: 0,
  failureCount: 0,
  totalMs: 0,
  maxMs: 0,
  warmupCount: 0
};

function flushPerf(now = performance.now()) {
  if (!PERF_ENABLED || now - perfWindow.lastLogAt < PERF_LOG_INTERVAL_MS) return;
  const requestCount = Math.max(1, perfWindow.requestCount);
  console.info("[Perf][ModelInferenceSequence]", {
    windowMs: Math.round(now - perfWindow.startedAt),
    requests: perfWindow.requestCount,
    successes: perfWindow.successCount,
    failures: perfWindow.failureCount,
    warmups: perfWindow.warmupCount,
    avgMs: Number((perfWindow.totalMs / requestCount).toFixed(2)),
    maxMs: Number(perfWindow.maxMs.toFixed(2)),
    mode: "onnx-sequence"
  });
  perfWindow.startedAt = now;
  perfWindow.lastLogAt = now;
  perfWindow.requestCount = 0;
  perfWindow.successCount = 0;
  perfWindow.failureCount = 0;
  perfWindow.totalMs = 0;
  perfWindow.maxMs = 0;
  perfWindow.warmupCount = 0;
}

function getRequestIntervalMs() {
  const raw = Number(new URLSearchParams(window.location.search).get("modelIntervalMs"));
  if (!Number.isFinite(raw)) return DEFAULT_REQUEST_INTERVAL_MS;
  return Math.max(60, Math.min(400, Math.round(raw)));
}

function getGlobalRequestGapMs() {
  return Math.max(45, Math.round(getRequestIntervalMs() * 0.5));
}

function getHandState(handKey = "default") {
  if (!handStateByKey.has(handKey)) {
    handStateByKey.set(handKey, {
      jointBuffer: [], // 정규화된 63차원 프레임 버퍼
      lastRequestAt: 0,
      inFlight: false,
      lastPrediction: null,
      lastStartedAt: 0,
      lastCompletedAt: 0,
      lastDurationMs: null
    });
  }
  return handStateByKey.get(handKey);
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

      console.warn("[ModelInferenceSequence] local WASM 경로 실패, CDN으로 재시도합니다.", {
        localWasmBase: ORT_LOCAL_WASM_BASE,
        cdnWasmBase: ORT_CDN_WASM_BASE,
        error: String(localError?.message || localError || "")
      });

      onnxSession = await createOnnxSession(ortInstance, ORT_CDN_WASM_BASE);
    }

    console.info("[ModelInferenceSequence] ✅ 시퀀스 모델 로드 완료", {
      modelPath: MODEL_PATH,
      runtimeRoot: RUNTIME_ROOT,
      wasmBase: ortInstance?.env?.wasm?.wasmPaths,
      classes: classNames.length,
      sequenceLength: SEQUENCE_LENGTH,
      inputNames: onnxSession.inputNames,
      outputNames: onnxSession.outputNames
    });

    initializationError = null;
    lastInitFailedAt = 0;
    isInitializing = false;
    return true;
  } catch (error) {
    console.error("[ModelInferenceSequence] ❌ 모델 로드 실패:", error);
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
function sanitizeLandmarks(landmarks, handKey = "default") {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return null;

  const normalizedHandKey = String(handKey || "default").trim().toLowerCase();
  const shouldMirrorLeft = LEFT_HAND_MIRROR_ENABLED && normalizedHandKey === "left";
  const mirrorPivotX = shouldMirrorLeft ? getMirrorPivotX(landmarks) : 0;
  const features = [];
  for (let i = 0; i < 21; i++) {
    const point = landmarks[i];
    const rawX = Number.isFinite(point?.x) ? point.x : 0;
    const mirroredX = shouldMirrorLeft ? clamp(mirrorPivotX * 2 - rawX, 0, 1) : rawX;
    features.push(
      mirroredX,
      Number.isFinite(point?.y) ? point.y : 0,
      Number.isFinite(point?.z) ? point.z : 0
    );
  }
  return features;
}

// pos_scale 정규화: (pts - pts[0]) / ||pts[9] - pts[0]||
function normalizePosScale(frame63) {
  const normalized = new Float32Array(63);
  const originX = frame63[0];
  const originY = frame63[1];
  const originZ = frame63[2];

  // pts[9]는 인덱스 27, 28, 29 (9번째 관절 = 중지 MCP)
  const dx = frame63[27] - originX;
  const dy = frame63[28] - originY;
  const dz = frame63[29] - originZ;
  const denom = Math.hypot(dx, dy, dz);
  const scale = denom <= EPS ? 1 : 1 / denom;

  for (let i = 0; i < 63; i += 3) {
    normalized[i] = (frame63[i] - originX) * scale;
    normalized[i + 1] = (frame63[i + 1] - originY) * scale;
    normalized[i + 2] = (frame63[i + 2] - originZ) * scale;
  }

  return normalized;
}

// 8프레임 버퍼로부터 [8, 126] 특성 텐서 생성
function buildFeatureTensor(buffer) {
  if (buffer.length !== SEQUENCE_LENGTH) {
    throw new Error(`Expected ${SEQUENCE_LENGTH} buffered frames, got ${buffer.length}`);
  }

  const featureTensor = new Float32Array(SEQUENCE_LENGTH * FEATURE_DIM);
  for (let t = 0; t < SEQUENCE_LENGTH; t++) {
    const baseOffset = t * FEATURE_DIM;
    const current = buffer[t];
    const previous = t > 0 ? buffer[t - 1] : null;

    for (let i = 0; i < 63; i++) {
      // 관절 위치 (0~62)
      featureTensor[baseOffset + i] = current[i];
      // 델타 (63~125)
      featureTensor[baseOffset + 63 + i] = previous === null ? 0 : current[i] - previous[i];
    }
  }

  return featureTensor;
}

function normalizePrediction(predIndex, confidence, probs, tsMs, status, tau = DEFAULT_TAU) {
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
    modelVersion: "onnx-sequence-delta",
    source: "onnx-sequence",
    ts_ms: tsMs,
    tau_applied: tau,
    tau_neutralized: tauNeutralized,
    raw_pred_index: predIndex,
    raw_pred_label: classNames[predIndex] || "None",
    status // warmup, ready, no_hand 등
  };
}

// 손이 사라졌을 때 버퍼 리셋
export function pushNoHand(handKey = "default") {
  const handState = getHandState(handKey);
  handState.jointBuffer = [];
  handState.lastPrediction = normalizePrediction(
    0, // neutral
    0,
    new Array(classNames.length).fill(0).map((_, i) => (i === 0 ? 1 : 0)),
    Math.round(performance.now()),
    "no_hand"
  );
  return handState.lastPrediction;
}

async function scheduleModelRequest(landmarks, now, handKey = "default") {
  const handState = getHandState(handKey);
  const requestIntervalMs = getRequestIntervalMs();

  if (!onnxSession) {
    const initialized = await initializeModel();
    if (!initialized) return;
  }

  if (handState.inFlight) return;
  if (globalRequestInFlight) return;
  if (now - handState.lastRequestAt < requestIntervalMs) return;
  if (now - lastGlobalRequestAt < getGlobalRequestGapMs()) return;

  const rawFeatures = sanitizeLandmarks(landmarks, handKey);
  if (!rawFeatures) return;

  const tsMs = Math.round(now);
  globalRequestInFlight = true;
  lastGlobalRequestAt = now;
  handState.inFlight = true;
  handState.lastRequestAt = now;
  handState.lastStartedAt = performance.now();
  const requestStartedAt = PERF_ENABLED ? performance.now() : 0;
  if (PERF_ENABLED) perfWindow.requestCount += 1;

  try {
    // 1. pos_scale 정규화
    const normalized63 = normalizePosScale(rawFeatures);

    // 2. 버퍼에 추가
    handState.jointBuffer.push(normalized63);
    if (handState.jointBuffer.length > SEQUENCE_LENGTH) {
      handState.jointBuffer.shift();
    }

    // 3. warmup 체크
    if (handState.jointBuffer.length < SEQUENCE_LENGTH) {
      const prediction = normalizePrediction(
        0,
        0,
        new Array(classNames.length).fill(0).map((_, i) => (i === 0 ? 1 : 0)),
        tsMs,
        "warmup"
      );
      prediction.framesCollected = handState.jointBuffer.length;
      handState.lastPrediction = prediction;
      handState.lastCompletedAt = performance.now();
      handState.lastDurationMs = handState.lastCompletedAt - handState.lastStartedAt;

      if (PERF_ENABLED) {
        perfWindow.warmupCount += 1;
        const elapsedMs = performance.now() - requestStartedAt;
        perfWindow.totalMs += elapsedMs;
      }
      return;
    }

    // 4. 특성 텐서 생성 [8, 126]
    const features = buildFeatureTensor(handState.jointBuffer);

    // 5. ONNX 추론
    const ortInstance = await ensureOrtApi();
    const inputTensor = new ortInstance.Tensor(
      "float32",
      features,
      [1, SEQUENCE_LENGTH, FEATURE_DIM]
    );
    const inputName = onnxSession.inputNames[0];
    const outputName = onnxSession.outputNames[0];
    const results = await onnxSession.run({ [inputName]: inputTensor });

    // 6. 출력 처리
    const logits = Array.from(results[outputName].data);
    const probs = softmax(logits);
    const predIndex = probs.indexOf(Math.max(...probs));
    const confidence = probs[predIndex];
    const elapsedMs = performance.now() - handState.lastStartedAt;

    const prediction = normalizePrediction(predIndex, confidence, probs, tsMs, "ready");
    prediction.elapsed_ms = elapsedMs;
    prediction.completed_at_ms = performance.now();
    prediction.framesCollected = handState.jointBuffer.length;
    handState.lastPrediction = prediction;
    handState.lastCompletedAt = prediction.completed_at_ms;
    handState.lastDurationMs = elapsedMs;

    if (PERF_ENABLED) {
      perfWindow.successCount += 1;
      perfWindow.totalMs += elapsedMs;
      perfWindow.maxMs = Math.max(perfWindow.maxMs, elapsedMs);
    }
  } catch (error) {
    console.error("[ModelInferenceSequence] 추론 실패:", error);
    // 에러 발생 시 버퍼 리셋
    handState.jointBuffer = [];
    handState.lastCompletedAt = performance.now();
    handState.lastDurationMs = handState.lastStartedAt > 0 ? (handState.lastCompletedAt - handState.lastStartedAt) : null;

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
  const handState = getHandState(handKey);

  // 손이 없으면 no_hand 처리
  if (!landmarks || !Array.isArray(landmarks) || landmarks.length < 21) {
    return pushNoHand(handKey);
  }

  scheduleModelRequest(landmarks, now, handKey);
  return handState.lastPrediction;
}

export function getModelInferenceStatus(now = performance.now()) {
  const states = [...handStateByKey.values()];
  const lastCompletedAt = states.reduce((max, state) => Math.max(max, state.lastCompletedAt || 0), 0);
  const lastDurationMs = states.reduce((latest, state) => {
    if (!Number.isFinite(state.lastCompletedAt) || state.lastCompletedAt <= 0) return latest;
    if (!latest || state.lastCompletedAt > latest.completedAt) {
      return { completedAt: state.lastCompletedAt, durationMs: state.lastDurationMs };
    }
    return latest;
  }, null);

  return {
    endpointConfigured: onnxSession !== null,
    inFlight: states.some((state) => state.inFlight),
    recentInference: states.some((state) => state.inFlight || (state.lastCompletedAt > 0 && now - state.lastCompletedAt <= 1200)),
    lastCompletedAgoMs: lastCompletedAt > 0 ? now - lastCompletedAt : null,
    lastDurationMs: lastDurationMs?.durationMs ?? null,
    disabled: false,
    mode: "onnx-sequence"
  };
}

export function preloadModel() {
  return initializeModel();
}
