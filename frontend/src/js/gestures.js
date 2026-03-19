// [gestures.js] 손모양을 보고 어떤 동작(주먹, 가위, 보 등)인지 맞히는 '퀴즈 정답지' 같은 파일입니다.
// 손가락이 펴졌는지 굽혀졌는지를 계산해서 손동작의 이름을 결정합니다.
// 손의 랜드마크 데이터에서 규칙 기반/모델 기반 제스처를 통합 감지하는 모듈입니다.
import { getModelPrediction } from "./model_inference.js";

function distance(a, b) {
  const dx = a.x - b.x; // 두 점의 가로 차이입니다.
  const dy = a.y - b.y; // 두 점의 세로 차이입니다.
  return Math.hypot(dx, dy); // 피타고라스 방식으로 실제 거리를 계산합니다.
}

// 손가락 사이의 거리나 위치를 보고 "지금 주먹인가?" 하고 판단을 내리는 기능입니다.
export function detectGesture(landmarks, now, sessionStarted) {
  if (!sessionStarted) { // 아직 시작 전이면 손동작을 굳이 판정하지 않습니다.
    return { isV: false, isPaper: false, isFist: false }; // 아무 동작도 아니라고 돌려줍니다.
  }

  const indexTip = landmarks[8]; // 검지 끝 좌표입니다.
  const indexPip = landmarks[6]; // 검지 중간 마디 좌표입니다.
  const indexMcp = landmarks[5]; // 검지 시작 마디 좌표입니다.

  const middleTip = landmarks[12]; // 중지 끝 좌표입니다.
  const middlePip = landmarks[10]; // 중지 중간 마디 좌표입니다.
  const middleMcp = landmarks[9]; // 중지 시작 마디 좌표입니다.

  const ringTip = landmarks[16]; // 약지 끝 좌표입니다.
  const ringPip = landmarks[14]; // 약지 중간 마디 좌표입니다.
  const ringMcp = landmarks[13]; // 약지 시작 마디 좌표입니다.

  const pinkyTip = landmarks[20]; // 새끼손가락 끝 좌표입니다.
  const pinkyPip = landmarks[18]; // 새끼손가락 중간 마디 좌표입니다.
  const pinkyMcp = landmarks[17]; // 새끼손가락 시작 마디 좌표입니다.

  const wrist = landmarks[0]; // 손목 좌표입니다.
  const palmScale = Math.max(distance(wrist, landmarks[9]), 0.001); // 손 크기에 따라 기준값이 달라지지 않도록 손바닥 크기를 구합니다.
  const vSpread = distance(indexTip, middleTip) / palmScale; // 검지와 중지가 얼마나 벌어졌는지 비율로 계산합니다.

  const indexExtended = indexTip.y < indexPip.y && indexPip.y < indexMcp.y; // 검지가 위로 펴졌는지 확인합니다.
  const middleExtended = middleTip.y < middlePip.y && middlePip.y < middleMcp.y; // 중지가 위로 펴졌는지 확인합니다.
  const ringExtended = ringTip.y < ringPip.y && ringPip.y < ringMcp.y; // 약지가 위로 펴졌는지 확인합니다.
  const pinkyExtended = pinkyTip.y < pinkyPip.y && pinkyPip.y < pinkyMcp.y; // 새끼손가락이 위로 펴졌는지 확인합니다.

  const ringFolded = ringTip.y > ringPip.y; // 약지가 접혔는지 확인합니다.
  const pinkyFolded = pinkyTip.y > pinkyPip.y; // 새끼손가락이 접혔는지 확인합니다.
  const indexFolded = indexTip.y > indexPip.y; // 검지가 접혔는지 확인합니다.
  const middleFolded = middleTip.y > middlePip.y; // 중지가 접혔는지 확인합니다.

  const isVGesture = indexExtended && middleExtended && ringFolded && pinkyFolded && vSpread > 0.38; // 브이 모양인지 판정합니다.
  const isPaperGesture = indexExtended && middleExtended && ringExtended && pinkyExtended; // 손바닥을 편 상태인지 판정합니다.
  const isFistGesture = indexFolded && middleFolded && ringFolded && pinkyFolded; // 주먹 상태인지 판정합니다.

  return {
    isV: isVGesture, // 브이 여부를 담습니다.
    isPaper: isPaperGesture, // 손바닥 여부를 담습니다.
    isFist: isFistGesture // 주먹 여부를 담습니다.
  };
}

const DEFAULT_CONFIDENCE_ENTER = 0.58;
const DEFAULT_CONFIDENCE_HOLD = 0.46;
const DEFAULT_STABLE_FRAMES = 1;
const DEFAULT_CLEAR_FRAMES = 1;

function parseNumberParam(name, fallback, min, max) {
  const value = Number(new URLSearchParams(window.location.search).get(name));
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

const CONFIDENCE_ENTER = parseNumberParam("gestureEnter", DEFAULT_CONFIDENCE_ENTER, 0.2, 0.95);
const CONFIDENCE_HOLD = parseNumberParam("gestureHold", DEFAULT_CONFIDENCE_HOLD, 0.1, CONFIDENCE_ENTER);
const STABLE_FRAMES = Math.round(parseNumberParam("gestureStableFrames", DEFAULT_STABLE_FRAMES, 1, 8));
const CLEAR_FRAMES = Math.round(parseNumberParam("gestureClearFrames", DEFAULT_CLEAR_FRAMES, 1, 6));

const CLASS_SPECIFIC_ENTER = {
  Pinky: parseNumberParam("gestureEnterPinky", 0.36, 0.2, 0.95),
  Animal: parseNumberParam("gestureEnterAnimal", 0.34, 0.2, 0.95),
  KHeart: parseNumberParam("gestureEnterKHeart", 0.36, 0.2, 0.95)
};

const CLASS_SPECIFIC_HOLD = {
  Pinky: parseNumberParam("gestureHoldPinky", 0.24, 0.1, CLASS_SPECIFIC_ENTER.Pinky),
  Animal: parseNumberParam("gestureHoldAnimal", 0.22, 0.1, CLASS_SPECIFIC_ENTER.Animal),
  KHeart: parseNumberParam("gestureHoldKHeart", 0.24, 0.1, CLASS_SPECIFIC_ENTER.KHeart)
};

const CLASS_SPECIFIC_STABLE_FRAMES = {
  Pinky: Math.round(parseNumberParam("gestureStableFramesPinky", 1, 1, 8)),
  Animal: Math.round(parseNumberParam("gestureStableFramesAnimal", 1, 1, 8)),
  KHeart: Math.round(parseNumberParam("gestureStableFramesKHeart", 1, 1, 8))
};

const stableStateByHand = new Map();

function getStableState(handKey = "default") {
  if (!stableStateByHand.has(handKey)) {
    stableStateByHand.set(handKey, {
      candidateLabel: "None",
      candidateFrames: 0,
      noneFrames: 0,
      stableLabel: "None",
      confidence: 0,
      source: "rules"
    });
  }
  return stableStateByHand.get(handKey);
}

function resolveInitialGestureMode() {
  const params = new URLSearchParams(window.location.search); // 주소창의 옵션 값을 읽습니다.
  const queryMode = params.get("gestureMode"); // URL에 적힌 gestureMode 값을 가져옵니다.
  const globalMode = typeof window.__JAMJAM_GESTURE_MODE === "string"
    ? window.__JAMJAM_GESTURE_MODE // 전역 설정값이 있으면 그것도 후보로 씁니다.
    : null; // 없으면 비워둡니다.
  const raw = (queryMode || globalMode || "model").trim().toLowerCase(); // 없으면 기본값 model을 사용합니다 (ONNX 모델만 사용).

  if (raw === "rules" || raw === "model" || raw === "hybrid") {
    return raw; // 허용된 값이면 그대로 사용합니다.
  }
  return "model"; // 이상한 값이 들어오면 model 모드로 돌아갑니다.
}

let gestureMode = resolveInitialGestureMode();

export function getGestureMode() {
  return gestureMode; // 현재 사용 중인 손동작 판정 방식을 알려줍니다.
}

export function setGestureMode(nextMode) {
  const raw = String(nextMode || "").trim().toLowerCase();
  if (raw === "rules" || raw === "model" || raw === "hybrid") {
    gestureMode = raw;
    return gestureMode;
  }
  return gestureMode;
}

function normalizeModelLabel(rawLabel) {
  const label = String(rawLabel || "").trim().toLowerCase(); // AI가 보낸 라벨을 비교하기 쉬운 글자 형태로 바꿉니다.
  if (label === "none" || label === "class0") return "None"; // 0번 클래스(중립)는 None으로 처리합니다.
  if (label === "fist" || label === "isfist" || label === "class1") return "Fist"; // 주먹 관련 이름들을 하나로 통일합니다.
  if (
    label === "open palm" ||
    label === "open_palm" ||
    label === "openpalm" ||
    label === "paper" ||
    label === "ispaper" ||
    label === "class2"
  ) return "OpenPalm"; // 손바닥 관련 이름들을 하나로 통일합니다.
  if (label === "v" || label === "isv" || label === "class3") return "V"; // 브이 관련 이름들을 하나로 통일합니다.
  if (label === "pinky" || label === "ispinky" || label === "pinky class" || label === "pinky_class" || label === "class4" || label === "class 4" || label === "4") return "Pinky"; // 새끼손가락 제스처 이름을 맞춥니다.
  if (label === "animal" || label === "isanimal" || label === "class5" || label === "class 5" || label === "5") return "Animal"; // 애니멀 제스처 이름을 맞춥니다.
  if (label === "k-heart" || label === "kheart" || label === "is_k_heart" || label === "class6" || label === "class 6" || label === "6") return "KHeart"; // K-하트 제스처 이름을 맞춥니다.
  return "None"; // 어디에도 해당하지 않으면 인식 실패로 처리합니다.
}

function getStableFramesForLabel(label) {
  return CLASS_SPECIFIC_STABLE_FRAMES[label] ?? STABLE_FRAMES;
}

function mapRulesToResult(rules) {
  if (rules.isFist) return { label: "Fist", confidence: 1, source: "rules" }; // 규칙상 주먹이면 확신 100%로 적습니다.
  if (rules.isV) return { label: "V", confidence: 1, source: "rules" }; // 규칙상 브이면 확신 100%로 적습니다.
  if (rules.isPaper) return { label: "OpenPalm", confidence: 1, source: "rules" }; // 규칙상 손바닥이면 확신 100%로 적습니다.
  return { label: "None", confidence: 0, source: "rules" }; // 아니면 아무 제스처도 아니라고 적습니다.
}

function mapModelToResult(modelPrediction, stableState) {
  if (!modelPrediction) return null; // 아직 AI 답변이 없으면 더 할 일이 없습니다.

  const normalized = normalizeModelLabel(modelPrediction.label); // AI 라벨 이름을 우리 코드 기준 이름으로 바꿉니다.
  const classLabel = Number.isFinite(modelPrediction.classId)
    ? normalizeModelLabel(`class${modelPrediction.classId}`)
    : "None";
  const label = normalized !== "None" ? normalized : classLabel;
  const confidence = Number.isFinite(modelPrediction.confidence) ? modelPrediction.confidence : 0; // 신뢰도가 숫자가 아니면 0으로 처리합니다.
  const enterThreshold = CLASS_SPECIFIC_ENTER[label] ?? CONFIDENCE_ENTER;
  const holdBaseThreshold = CLASS_SPECIFIC_HOLD[label] ?? CONFIDENCE_HOLD;
  const holdThreshold = stableState.stableLabel === label ? holdBaseThreshold : enterThreshold;

  if (label === "None" || confidence < holdThreshold) {
    return { label: "None", confidence, source: "model" }; // 신뢰도가 낮으면 아직 확실하지 않다고 봅니다.
  }

  return { label, confidence, source: "model" }; // AI가 낸 결론을 우리 형식으로 돌려줍니다.
}

// 같은 동작이 여러 번 연달아 인식되어야 "진짜 이 동작을 하고 있구나"라고 확신하는 '안정화' 기능입니다.
function stabilize(rawResult, stableState) {
  if (rawResult.label === "None") {
    stableState.noneFrames += 1; // 아무 동작도 안 보인 프레임 수를 늘립니다.
    stableState.candidateLabel = "None"; // 후보 동작도 비웁니다.
    stableState.candidateFrames = 0; // 후보가 유지된 횟수도 초기화합니다.

    if (stableState.noneFrames >= CLEAR_FRAMES) {
      stableState.stableLabel = "None"; // 충분히 안 보였으면 최종 상태도 없음으로 바꿉니다.
      stableState.confidence = 0; // 확신도도 0으로 내립니다.
      stableState.source = rawResult.source; // 어떤 방식으로 판정했는지 기억합니다.
    }
  } else {
    stableState.noneFrames = 0; // 동작이 보이면 '없음' 카운트는 리셋합니다.

    if (stableState.candidateLabel === rawResult.label) {
      stableState.candidateFrames += 1; // 같은 후보가 계속 나오면 횟수를 늘립니다.
    } else {
      stableState.candidateLabel = rawResult.label; // 새로운 동작이 보이면 후보를 교체합니다.
      stableState.candidateFrames = 1; // 첫 번째 등장으로 기록합니다.
    }

    if (stableState.candidateFrames >= getStableFramesForLabel(rawResult.label)) {
      stableState.stableLabel = rawResult.label; // 여러 번 반복되면 최종 동작으로 채택합니다.
      stableState.confidence = rawResult.confidence; // 그때의 확신도도 저장합니다.
      stableState.source = rawResult.source; // 규칙인지 모델인지도 같이 저장합니다.
    }
  }

  return {
    label: stableState.stableLabel, // 안정화가 끝난 최종 라벨입니다.
    confidence: stableState.confidence, // 최종 신뢰도입니다.
    source: stableState.source, // 어떤 방식의 결과인지 알려줍니다.
    isV: stableState.stableLabel === "V", // 브이인지 빠르게 확인할 수 있도록 함께 넣어둡니다.
    isPaper: stableState.stableLabel === "OpenPalm" // 손바닥인지도 함께 넣어둡니다.
  };
}

// [최종 판단] 규칙(단순 계산)과 모델(AI)의 결과를 종합하여 지금 어떤 손동작인지 최종 결론을 내립니다.
export function resolveGesture(landmarks, now, sessionStarted, handKey = "default") {
  if (!sessionStarted) {
    return {
      label: "None", // 시작 전에는 아무 동작도 없다고 판단합니다.
      confidence: 0, // 신뢰도도 0입니다.
      source: "rules", // 기본 출처는 규칙 방식으로 둡니다.
      isV: false, // 브이도 아닙니다.
      isPaper: false // 손바닥도 아닙니다.
    };
  }

  const stableState = getStableState(handKey);
  const modelPrediction = gestureMode === "rules"
    ? null
    : getModelPrediction(landmarks, now, handKey); // 규칙 전용 모드에서는 모델 통신 자체를 하지 않습니다.
  const modelResult = mapModelToResult(modelPrediction, stableState) || {
    label: "None", // AI 답변이 없으면 일단 인식 안 됨으로 둡니다.
    confidence: 0, // 신뢰도도 0입니다.
    source: "model" // 출처는 모델이라고 표시합니다.
  };

  let rawResult; // 안정화 전에 사용할 임시 결과를 담을 변수입니다.
  if (gestureMode === "model") {
    rawResult = modelResult; // 모델만 쓰는 모드면 AI 답변을 그대로 씁니다.
  } else {
    const rules = detectGesture(landmarks, now, sessionStarted); // 손가락 위치만 보고 규칙 판정을 실행합니다.
    const ruleResult = mapRulesToResult(rules); // 그 결과를 공통 형식으로 바꿉니다.

    if (gestureMode === "rules") {
      rawResult = ruleResult; // 규칙만 쓰는 모드면 이 값을 그대로 씁니다.
    } else {
      rawResult = ruleResult.label === "Fist"
        ? ruleResult // 주먹은 규칙 기반이 더 단순하고 안정적이어서 우선합니다.
        : (modelResult.label !== "None" ? modelResult : ruleResult); // 그 외에는 AI가 확실하면 AI를, 아니면 규칙 결과를 씁니다.
    }
  }

  return stabilize(rawResult, stableState); // 마지막으로 흔들리는 결과를 안정화해서 돌려줍니다.
}
