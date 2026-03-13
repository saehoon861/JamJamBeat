// [env_config.js] 환경별 API 주소를 한 곳에서 읽어오도록 정리한 설정 모듈입니다.

const ENV_MODEL_ENDPOINT = typeof import.meta?.env?.VITE_MODEL_ENDPOINT === "string"
  ? import.meta.env.VITE_MODEL_ENDPOINT.trim()
  : "";

// endpoint 우선순위: URL 파라미터 > window 전역 오버라이드 > Vite .env
export function getConfiguredModelEndpoint() {
  const queryEndpoint = new URLSearchParams(window.location.search).get("inferEndpoint");
  if (queryEndpoint && queryEndpoint.trim()) return queryEndpoint.trim();

  const globalEndpoint = window.__JAMJAM_MODEL_ENDPOINT;
  if (typeof globalEndpoint === "string" && globalEndpoint.trim()) return globalEndpoint.trim();

  if (ENV_MODEL_ENDPOINT) return ENV_MODEL_ENDPOINT;

  return null;
}
