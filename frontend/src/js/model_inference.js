// [model_inference.js] AI 선생님에게 "이 손모양이 뭐야?"라고 물어보고 답을 받아오는 '통신병' 역할의 파일입니다.
// 복잡한 손동작은 컴퓨터가 직접 계산하기 어렵기 때문에 AI의 도움을 받습니다.

import { getConfiguredModelEndpoint } from "./env_config.js";

const REQUEST_TIMEOUT_MS = 180;
const DEFAULT_REQUEST_INTERVAL_MS = 120;
const FAIL_OPEN_AFTER = 5;
const DISABLE_FOR_MS = 1200;

const requestStateByHand = new Map();

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
      consecutiveFailures: 0,
      disabledUntil: 0,
      lastPrediction: null
    });
  }
  return requestStateByHand.get(handKey);
}

// AI 선생님이 어디 계시는지(서버 주소) 확인하는 기능입니다.
function getConfiguredEndpoint() {
  return getConfiguredModelEndpoint(); // URL/전역/.env 우선순위로 주소를 가져옵니다.
}

// 카메라로 찍은 손 위치 데이터가 올바른 형식인지 검사하고 정리하는 기능입니다.
function sanitizeLandmarks(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return null; // 손 좌표가 21개보다 적으면 제대로 된 손이 아니라고 봅니다.

  return landmarks.slice(0, 21).map((point) => ({
    x: Number.isFinite(point?.x) ? point.x : 0, // x값이 이상하면 0으로 보정합니다.
    y: Number.isFinite(point?.y) ? point.y : 0, // y값이 이상하면 0으로 보정합니다.
    z: Number.isFinite(point?.z) ? point.z : 0 // z값이 이상하면 0으로 보정합니다.
  }));
}

function normalizePrediction(json, tsMs) {
  if (!json || typeof json !== "object") return null; // 서버 응답이 객체가 아니면 잘못된 응답입니다.

  const rawLabel = typeof json.label === "string" ? json.label : "None"; // 라벨이 없으면 None으로 간주합니다.
  const confidence = Number(json.confidence); // confidence를 숫자로 바꿉니다.
  const classId = Number(json.class_id);

  return {
    label: rawLabel, // AI가 예측한 이름입니다.
    confidence: Number.isFinite(confidence) ? confidence : 0, // 신뢰도가 숫자가 아니면 0으로 둡니다.
    probs: Array.isArray(json.probs) ? json.probs : null, // 클래스별 확률표가 있으면 같이 저장합니다.
    classId: Number.isFinite(classId) ? classId : null,
    modelVersion: typeof json.model_version === "string" ? json.model_version : "unknown", // 어떤 버전의 모델인지 기록합니다.
    source: "model", // 이 결과는 AI 모델에서 왔다는 뜻입니다.
    ts_ms: tsMs // 어느 시점의 손 좌표에 대한 답인지 기억합니다.
  };
}

function scheduleModelRequest(landmarks, now, handKey = "default") {
  const endpoint = getConfiguredEndpoint(); // 실제로 요청을 보낼 주소를 알아냅니다.
  const handState = getHandRequestState(handKey);
  if (!endpoint) return; // 주소가 없으면 요청하지 않습니다.
  if (handState.inFlight) return; // 이미 요청 중이면 새 요청을 보내지 않습니다.
  if (now < handState.disabledUntil) return; // 실패가 많아서 쉬는 시간이라면 요청하지 않습니다.
  if (now - handState.lastRequestAt < getRequestIntervalMs()) return; // 너무 자주 요청하지 않도록 간격을 지킵니다.

  const payloadLandmarks = sanitizeLandmarks(landmarks); // 손 좌표를 안전한 형식으로 정리합니다.
  if (!payloadLandmarks) return; // 손 좌표가 이상하면 요청을 보내지 않습니다.

  const tsMs = Math.round(now); // 현재 시간을 반올림해서 기록합니다.
  const payload = {
    version: 1, // 서버와 약속한 데이터 형식 버전입니다.
    ts_ms: tsMs, // 이 손 좌표가 언제 찍힌 것인지 보냅니다.
    landmarks: payloadLandmarks // 손 좌표 본문을 넣습니다.
  };

  handState.inFlight = true; // 이제 요청이 진행 중이라고 표시합니다.
  handState.lastRequestAt = now; // 마지막 요청 시각을 지금으로 갱신합니다.

  const controller = new AbortController(); // 너무 오래 걸리면 요청을 끊기 위한 장치입니다.
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS); // 제한 시간이 지나면 강제로 취소합니다.

  fetch(endpoint, {
    method: "POST", // 서버에 데이터를 보낼 때는 POST 방식을 씁니다.
    headers: { "Content-Type": "application/json" }, // JSON 형식으로 보낸다고 알려줍니다.
    body: JSON.stringify(payload), // 자바스크립트 객체를 JSON 글자로 바꿔서 보냅니다.
    signal: controller.signal // 시간이 초과되면 이 신호로 요청을 끊습니다.
  })
    .then((response) => {
      if (!response.ok) throw new Error("HTTP_" + response.status); // 서버가 실패 코드를 보내면 오류로 처리합니다.
      return response.json(); // 성공이면 응답 본문을 JSON으로 읽습니다.
    })
    .then((json) => {
      const prediction = normalizePrediction(json, tsMs); // 받은 JSON을 우리 형식으로 정리합니다.
      if (!prediction) throw new Error("INVALID_PAYLOAD"); // 형식이 이상하면 오류로 처리합니다.

      handState.lastPrediction = prediction; // 최근 예측 결과를 새 값으로 교체합니다.
      handState.consecutiveFailures = 0; // 성공했으니 실패 횟수는 다시 0으로 만듭니다.
    })
    .catch(() => {
      handState.consecutiveFailures += 1; // 실패가 났으니 실패 횟수를 1 올립니다.
      if (handState.consecutiveFailures >= FAIL_OPEN_AFTER) {
        handState.disabledUntil = performance.now() + DISABLE_FOR_MS; // 너무 많이 실패했으니 잠깐 요청을 멈춥니다.
        handState.consecutiveFailures = 0; // 다음 휴식 이후를 위해 실패 횟수를 초기화합니다.
      }
    })
    .finally(() => {
      clearTimeout(timeoutId); // 타임아웃 타이머를 정리합니다.
      handState.inFlight = false; // 요청이 끝났다고 표시합니다.
    });
}

// AI에게 현재 내 손 모양 데이터를 보내고, AI가 생각하는 정답이 무엇인지 가져오는 기능입니다.
export function getModelPrediction(landmarks, now = performance.now(), handKey = "default") {
  const handState = getHandRequestState(handKey);
  scheduleModelRequest(landmarks, now, handKey); // 필요하면 새 요청을 예약하거나 바로 보냅니다.
  return handState.lastPrediction; // 지금 시점에서 가장 최근에 받은 답을 돌려줍니다.
}

export function getModelInferenceStatus(now = performance.now()) {
  const states = [...requestStateByHand.values()];
  return {
    endpointConfigured: Boolean(getConfiguredEndpoint()), // 서버 주소가 설정되어 있는지 알려줍니다.
    inFlight: states.some((state) => state.inFlight), // 현재 요청 중인지 알려줍니다.
    disabled: states.some((state) => now < state.disabledUntil) // 잠시 비활성화 상태인지 알려줍니다.
  };
}
