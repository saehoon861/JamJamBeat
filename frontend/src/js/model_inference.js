// [model_inference_onnx.js] ONNX Runtime Web을 이용해 브라우저에서 직접 AI 모델을 실행하는 '로컬 추론 엔진' 파일입니다.
// 백엔드 서버 없이 브라우저에서만 손모양을 판단할 수 있습니다.

import * as ortNamespace from "onnxruntime-web";

const DEFAULT_TAU = 0.85; // 선정_모델_스펙.md 기준 tau 값
const DEFAULT_REQUEST_INTERVAL_MS = 150;
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

  if (queryRoot && queryRoot.trim()) return normalizeRootPath(queryRoot, "/runtime/");
  if (typeof globalRoot === "string" && globalRoot.trim()) return normalizeRootPath(globalRoot, "/runtime/");
  if (typeof envRoot === "string" && envRoot.trim()) return normalizeRootPath(envRoot, "/runtime/");

  return normalizeRootPath(baseUrl + "runtime", "/runtime/");
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

  // Tensor는 번들/interop 형태에 따라 다른 객체에 있을 수 있어 안전하게 탐색
  const fallbackTensor =
    readCandidateShape(ortNamespace).tensor ||
    readCandidateShape(globalThis?.ort).tensor ||
    null;

  const tensorCtor = shape.tensor || fallbackTensor;
  if (!tensorCtor) return null;

  return {
    InferenceSession: shape.session,
    Tensor: tensorCtor,
    env: shape.env || readCandidateShape(ortNamespace).env || null
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

  // 1) 정적 import 네임스페이스
  ortApi = inspect("ortNamespace", ortNamespace);

  // 2) ESM/CJS interop default shape 대응
  if (!ortApi) {
    ortApi = inspect("ortNamespace.default", ortNamespace?.default);
  }

  // 3) window/global 주입형 대응
  if (!ortApi) {
    ortApi = inspect("globalThis.ort", globalThis?.ort) || inspect("globalThis.ort.default", globalThis?.ort?.default);
  }

  // 4) 동적 import fallback
  if (!ortApi) {
    const mod = await import("onnxruntime-web");
    ortApi = inspect("dynamicImport", mod) || inspect("dynamicImport.default", mod?.default);
  }

  // 5) 서브 엔트리 fallback (WASM 전용 번들)
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

  // WASM 경로/스레드 기본값 설정 (create 전에 적용)
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
  // 왼손 추론을 허용합니다. (미러링 전처리가 적용됨)
  return true;
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

function flushPerf(now = performance.now()) {
  if (!PERF_ENABLED || now - perfWindow.lastLogAt < PERF_LOG_INTERVAL_MS) return;
  const requestCount = Math.max(1, perfWindow.requestCount);
  console.info("[Perf][ModelInferenceONNX]", {
    windowMs: Math.round(now - perfWindow.startedAt),
    requests: perfWindow.requestCount,
    successes: perfWindow.successCount,
    failures: perfWindow.failureCount,
    avgMs: Number((perfWindow.totalMs / requestCount).toFixed(2)),
    maxMs: Number(perfWindow.maxMs.toFixed(2)),
    mode: "onnx-local"
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
      lastPrediction: null,
      lastStartedAt: 0,
      lastCompletedAt: 0,
      lastDurationMs: null
    });
  }
  return requestStateByHand.get(handKey);
}

// ONNX 모델과 클래스 이름을 한 번만 로드합니다.
async function initializeModel() {
  if (onnxSession) return true; // 이미 로드됨
  if (isInitializing) {
    // 초기화 중이면 대기
    while (isInitializing) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    return onnxSession !== null;
  }
  if (initializationError) {
    if (performance.now() - lastInitFailedAt < INIT_RETRY_INTERVAL_MS) return false; // 너무 짧은 간격 재시도 방지
    initializationError = null; // 일정 시간 뒤 자동 재시도 허용
  }

  isInitializing = true;
  try {
    // 클래스 이름 로드
    const classResponse = await fetch(CLASS_NAMES_PATH);
    if (!classResponse.ok) throw new Error(`Failed to load class names: ${classResponse.status}`);
    classNames = await classResponse.json();

    const ort = await ensureOrtApi();

    // ONNX 모델 로드 (로컬 wasm 우선, 필요 시 CDN fallback)
    try {
      onnxSession = await createOnnxSession(ort, ORT_LOCAL_WASM_BASE);
    } catch (localError) {
      const shouldRetryWithCdn = ORT_LOCAL_WASM_BASE !== ORT_CDN_WASM_BASE && isLikelyWasmBootError(localError);
      if (!shouldRetryWithCdn) throw localError;

      console.warn("[ModelInferenceONNX] local WASM 경로 실패, CDN으로 재시도합니다.", {
        localWasmBase: ORT_LOCAL_WASM_BASE,
        cdnWasmBase: ORT_CDN_WASM_BASE,
        error: String(localError?.message || localError || "")
      });

      onnxSession = await createOnnxSession(ort, ORT_CDN_WASM_BASE);
    }

    console.info("[ModelInferenceONNX] ✅ 모델 로드 완료", {
      modelPath: MODEL_PATH,
      runtimeRoot: RUNTIME_ROOT,
      wasmBase: ort?.env?.wasm?.wasmPaths,
      classes: classNames.length,
      inputNames: onnxSession.inputNames,
      outputNames: onnxSession.outputNames
    });

    initializationError = null;
    lastInitFailedAt = 0;
    isInitializing = false;
    return true;
  } catch (error) {
    console.error("[ModelInferenceONNX] ❌ 모델 로드 실패:", error);
    initializationError = error;
    lastInitFailedAt = performance.now();
    isInitializing = false;
    return false;
  }
}

// Softmax 함수 (ONNX 모델은 logits을 출력하므로 확률로 변환 필요)
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

// 카메라로 찍은 손 위치 데이터가 올바른 형식인지 검사하고 63차원 배열로 변환합니다.
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
  return features; // 63개 요소 배열
}

function normalizePrediction(predIndex, confidence, probs, tsMs, tau = DEFAULT_TAU) {
  let finalIndex = predIndex;
  let tauNeutralized = false;

  // tau 후처리: 신뢰도가 임계값보다 낮으면 neutral(0)로 강제
  if (confidence < tau) {
    finalIndex = 0; // neutral
    tauNeutralized = predIndex !== 0;
  }

  return {
    label: classNames[finalIndex] || "None",
    confidence,
    probs,
    classId: finalIndex,
    modelVersion: "onnx-runtime-web",
    source: "onnx",
    ts_ms: tsMs,
    tau_applied: tau,
    tau_neutralized: tauNeutralized,
    raw_pred_index: predIndex,
    raw_pred_label: classNames[predIndex] || "None"
  };
}

async function scheduleModelRequest(landmarks, now, handKey = "default") {
  const handState = getHandRequestState(handKey);
  if (!isInferenceEnabledForHand(handKey)) {
    handState.inFlight = false;
    handState.lastRequestAt = 0;
    handState.lastStartedAt = 0;
    handState.lastCompletedAt = 0;
    handState.lastDurationMs = null;
    handState.lastPrediction = createDisabledPrediction();
    return;
  }
  const requestIntervalMs = getRequestIntervalMs();
  if (!onnxSession) {
    // 모델이 아직 로드 안 됐으면 초기화 시도
    const initialized = await initializeModel();
    if (!initialized) return;
  }
  if (handState.inFlight) return; // 이미 추론 중이면 새 요청을 보내지 않습니다.
  if (globalRequestInFlight) return; // 양손이 동시에 세션을 점유하지 않도록 직렬화합니다.
  if (now - handState.lastRequestAt < requestIntervalMs) return; // 너무 자주 요청하지 않도록 간격을 지킵니다.
  if (now - lastGlobalRequestAt < getGlobalRequestGapMs()) return; // 양손 활성 시에도 전체 요청 밀도를 제한합니다.

  const features = sanitizeLandmarks(landmarks, handKey);
  if (!features) return; // 손 좌표가 이상하면 추론하지 않습니다.

  const tsMs = Math.round(now);
  globalRequestInFlight = true;
  lastGlobalRequestAt = now;
  handState.inFlight = true;
  handState.lastRequestAt = now;
  handState.lastStartedAt = performance.now();
  const requestStartedAt = PERF_ENABLED ? performance.now() : 0;
  if (PERF_ENABLED) perfWindow.requestCount += 1;

  try {
    // ONNX 추론 실행
    const ort = await ensureOrtApi();
    const inputTensor = new ort.Tensor("float32", Float32Array.from(features), [1, 63]);
    const feeds = { joints: inputTensor };
    const results = await onnxSession.run(feeds);

    // 출력 처리
    const logits = Array.from(results.logits.data); // [7개 클래스 logits]
    const probs = softmax(logits);
    const predIndex = probs.indexOf(Math.max(...probs));
    const confidence = probs[predIndex];
    const elapsedMs = performance.now() - handState.lastStartedAt;

    const prediction = normalizePrediction(predIndex, confidence, probs, tsMs);
    prediction.elapsed_ms = elapsedMs;
    prediction.completed_at_ms = performance.now();
    handState.lastPrediction = prediction;
    handState.lastCompletedAt = prediction.completed_at_ms;
    handState.lastDurationMs = elapsedMs;

    if (PERF_ENABLED) {
      perfWindow.successCount += 1;
      perfWindow.totalMs += elapsedMs;
      perfWindow.maxMs = Math.max(perfWindow.maxMs, elapsedMs);
    }
  } catch (error) {
    console.error("[ModelInferenceONNX] 추론 실패:", error);
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

// ONNX로 현재 내 손 모양 데이터를 추론하고, AI가 생각하는 정답이 무엇인지 가져오는 기능입니다.
export function getModelPrediction(landmarks, now = performance.now(), handKey = "default") {
  const handState = getHandRequestState(handKey);
  if (!isInferenceEnabledForHand(handKey)) {
    handState.inFlight = false;
    handState.lastRequestAt = 0;
    handState.lastStartedAt = 0;
    handState.lastCompletedAt = 0;
    handState.lastDurationMs = null;
    handState.lastPrediction = createDisabledPrediction();
    return handState.lastPrediction;
  }
  scheduleModelRequest(landmarks, now, handKey); // 필요하면 새 추론을 예약하거나 바로 실행합니다.
  return handState.lastPrediction; // 지금 시점에서 가장 최근에 받은 답을 돌려줍니다.
}

export function getModelInferenceStatus(now = performance.now()) {
  const states = [...requestStateByHand.values()];
  const lastCompletedAt = states.reduce((max, state) => Math.max(max, state.lastCompletedAt || 0), 0);
  const lastDurationMs = states.reduce((latest, state) => {
    if (!Number.isFinite(state.lastCompletedAt) || state.lastCompletedAt <= 0) return latest;
    if (!latest || state.lastCompletedAt > latest.completedAt) {
      return { completedAt: state.lastCompletedAt, durationMs: state.lastDurationMs };
    }
    return latest;
  }, null);
  return {
    endpointConfigured: onnxSession !== null, // ONNX 모델이 로드되어 있는지 알려줍니다.
    inFlight: states.some((state) => state.inFlight), // 현재 추론 중인지 알려줍니다.
    recentInference: states.some((state) => state.inFlight || (state.lastCompletedAt > 0 && now - state.lastCompletedAt <= 1200)),
    lastCompletedAgoMs: lastCompletedAt > 0 ? now - lastCompletedAt : null,
    lastDurationMs: lastDurationMs?.durationMs ?? null,
    disabled: false, // ONNX는 fail-open 정책이 없으므로 항상 활성화
    mode: "onnx-local"
  };
}

// 앱 시작 시 모델 미리 로드 (선택 사항)
export function preloadModel() {
  return initializeModel();
}
