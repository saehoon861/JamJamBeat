// [model_inference_onnx.js] ONNX Runtime Web을 이용해 브라우저에서 직접 AI 모델을 실행하는 '로컬 추론 엔진' 파일입니다.
// 백엔드 서버 없이 브라우저에서만 손모양을 판단할 수 있습니다.

import * as ort from "onnxruntime-web";

const MODEL_PATH = "/runtime/model.onnx";
const CLASS_NAMES_PATH = "/runtime/class_names.json";
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

let onnxSession = null;
let classNames = ["neutral", "fist", "open_palm", "V", "pinky", "animal", "k-heart"];
let isInitializing = false;
let initializationError = null;

const requestStateByHand = new Map();
const perfWindow = {
  startedAt: performance.now(),
  lastLogAt: performance.now(),
  requestCount: 0,
  successCount: 0,
  failureCount: 0,
  totalMs: 0,
  maxMs: 0
};

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
  if (initializationError) return false; // 이전에 실패했으면 재시도 안 함

  isInitializing = true;
  try {
    // ONNX Runtime Web 설정 (WASM 백엔드 사용)
    ort.env.wasm.wasmPaths = "/node_modules/onnxruntime-web/dist/";

    // 클래스 이름 로드
    const classResponse = await fetch(CLASS_NAMES_PATH);
    if (!classResponse.ok) throw new Error(`Failed to load class names: ${classResponse.status}`);
    classNames = await classResponse.json();

    // ONNX 모델 로드
    onnxSession = await ort.InferenceSession.create(MODEL_PATH, {
      executionProviders: ["wasm"],
      graphOptimizationLevel: "all"
    });

    console.info("[ModelInferenceONNX] ✅ 모델 로드 완료", {
      modelPath: MODEL_PATH,
      classes: classNames.length,
      inputNames: onnxSession.inputNames,
      outputNames: onnxSession.outputNames
    });

    isInitializing = false;
    return true;
  } catch (error) {
    console.error("[ModelInferenceONNX] ❌ 모델 로드 실패:", error);
    initializationError = error;
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
  if (!onnxSession) {
    // 모델이 아직 로드 안 됐으면 초기화 시도
    const initialized = await initializeModel();
    if (!initialized) return;
  }
  if (handState.inFlight) return; // 이미 추론 중이면 새 요청을 보내지 않습니다.
  if (now - handState.lastRequestAt < getRequestIntervalMs()) return; // 너무 자주 요청하지 않도록 간격을 지킵니다.

  const features = sanitizeLandmarks(landmarks, handKey);
  if (!features) return; // 손 좌표가 이상하면 추론하지 않습니다.

  const tsMs = Math.round(now);
  handState.inFlight = true;
  handState.lastRequestAt = now;
  const requestStartedAt = PERF_ENABLED ? performance.now() : 0;
  if (PERF_ENABLED) perfWindow.requestCount += 1;

  try {
    // ONNX 추론 실행
    const inputTensor = new ort.Tensor("float32", Float32Array.from(features), [1, 63]);
    const feeds = { joints: inputTensor };
    const results = await onnxSession.run(feeds);

    // 출력 처리
    const logits = Array.from(results.logits.data); // [7개 클래스 logits]
    const probs = softmax(logits);
    const predIndex = probs.indexOf(Math.max(...probs));
    const confidence = probs[predIndex];

    const prediction = normalizePrediction(predIndex, confidence, probs, tsMs);
    handState.lastPrediction = prediction;

    if (PERF_ENABLED) {
      const elapsedMs = performance.now() - requestStartedAt;
      perfWindow.successCount += 1;
      perfWindow.totalMs += elapsedMs;
      perfWindow.maxMs = Math.max(perfWindow.maxMs, elapsedMs);
    }
  } catch (error) {
    console.error("[ModelInferenceONNX] 추론 실패:", error);
    if (PERF_ENABLED) {
      const elapsedMs = performance.now() - requestStartedAt;
      perfWindow.failureCount += 1;
      perfWindow.totalMs += elapsedMs;
      perfWindow.maxMs = Math.max(perfWindow.maxMs, elapsedMs);
    }
  } finally {
    handState.inFlight = false;
    flushPerf();
  }
}

// ONNX로 현재 내 손 모양 데이터를 추론하고, AI가 생각하는 정답이 무엇인지 가져오는 기능입니다.
export function getModelPrediction(landmarks, now = performance.now(), handKey = "default") {
  const handState = getHandRequestState(handKey);
  scheduleModelRequest(landmarks, now, handKey); // 필요하면 새 추론을 예약하거나 바로 실행합니다.
  return handState.lastPrediction; // 지금 시점에서 가장 최근에 받은 답을 돌려줍니다.
}

export function getModelInferenceStatus(now = performance.now()) {
  const states = [...requestStateByHand.values()];
  return {
    endpointConfigured: onnxSession !== null, // ONNX 모델이 로드되어 있는지 알려줍니다.
    inFlight: states.some((state) => state.inFlight), // 현재 추론 중인지 알려줍니다.
    disabled: false, // ONNX는 fail-open 정책이 없으므로 항상 활성화
    mode: "onnx-local"
  };
}

// 앱 시작 시 모델 미리 로드 (선택 사항)
export function preloadModel() {
  return initializeModel();
}
