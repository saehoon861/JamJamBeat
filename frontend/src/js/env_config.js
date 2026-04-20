// [env_config.js] 환경별 API 주소를 한 곳에서 읽어오도록 정리한 설정 모듈입니다.

const ENV_MODEL_ENDPOINT = typeof import.meta?.env?.VITE_MODEL_ENDPOINT === "string"
  ? import.meta.env.VITE_MODEL_ENDPOINT.trim()
  : "";
const ENV_MEDIAPIPE_WASM_ROOT = typeof import.meta?.env?.VITE_MEDIAPIPE_WASM_ROOT === "string"
  ? import.meta.env.VITE_MEDIAPIPE_WASM_ROOT.trim()
  : "";
const ENV_HAND_LANDMARKER_TASK_PATH = typeof import.meta?.env?.VITE_HAND_LANDMARKER_TASK_PATH === "string"
  ? import.meta.env.VITE_HAND_LANDMARKER_TASK_PATH.trim()
  : "";
const ENV_SPLIT_HAND_INFERENCE = typeof import.meta?.env?.VITE_SPLIT_HAND_INFERENCE === "string"
  ? import.meta.env.VITE_SPLIT_HAND_INFERENCE.trim()
  : "";

function normalizePath(value, fallback) {
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  return trimmed || fallback;
}

// endpoint 우선순위: URL 파라미터 > window 전역 오버라이드 > Vite .env
export function getConfiguredModelEndpoint() {
  const queryEndpoint = new URLSearchParams(window.location.search).get("inferEndpoint");
  if (queryEndpoint && queryEndpoint.trim()) return queryEndpoint.trim();

  const globalEndpoint = window.__JAMJAM_MODEL_ENDPOINT;
  if (typeof globalEndpoint === "string" && globalEndpoint.trim()) return globalEndpoint.trim();

  if (ENV_MODEL_ENDPOINT) return ENV_MODEL_ENDPOINT;

  return null;
}

export function getConfiguredMediaPipeWasmRoot() {
  const queryRoot = new URLSearchParams(window.location.search).get("mediapipeRoot");
  if (queryRoot && queryRoot.trim()) return normalizePath(queryRoot, "/mediapipe");

  const globalRoot = window.__JAMJAM_MEDIAPIPE_WASM_ROOT;
  if (typeof globalRoot === "string" && globalRoot.trim()) return normalizePath(globalRoot, "/mediapipe");

  if (ENV_MEDIAPIPE_WASM_ROOT) return normalizePath(ENV_MEDIAPIPE_WASM_ROOT, "/mediapipe");

  return "/mediapipe";
}

export function getConfiguredHandLandmarkerTaskPath() {
  const queryPath = new URLSearchParams(window.location.search).get("handLandmarkerTask");
  if (queryPath && queryPath.trim()) return normalizePath(queryPath, "/hand_landmarker.task");

  const globalPath = window.__JAMJAM_HAND_LANDMARKER_TASK_PATH;
  if (typeof globalPath === "string" && globalPath.trim()) return normalizePath(globalPath, "/hand_landmarker.task");

  if (ENV_HAND_LANDMARKER_TASK_PATH) return normalizePath(ENV_HAND_LANDMARKER_TASK_PATH, "/hand_landmarker.task");

  return "/hand_landmarker.task";
}

export function getConfiguredSplitHandInference() {
  const queryValue = new URLSearchParams(window.location.search).get("splitHands");
  if (typeof queryValue === "string" && queryValue.trim()) {
    const normalized = queryValue.trim().toLowerCase();
    return normalized === "1" || normalized === "true" || normalized === "on";
  }

  const globalValue = window.__JAMJAM_SPLIT_HAND_INFERENCE;
  if (typeof globalValue === "boolean") return globalValue;
  if (typeof globalValue === "string" && globalValue.trim()) {
    const normalized = globalValue.trim().toLowerCase();
    return normalized === "1" || normalized === "true" || normalized === "on";
  }

  if (ENV_SPLIT_HAND_INFERENCE) {
    const normalized = ENV_SPLIT_HAND_INFERENCE.trim().toLowerCase();
    return normalized === "1" || normalized === "true" || normalized === "on";
  }

  return false;
}
