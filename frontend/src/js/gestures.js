// 손의 랜드마크 데이터에서 규칙 기반/모델 기반 제스처를 통합 감지하는 모듈입니다.
import { getModelPrediction } from "./model_inference.js";

function distance(a, b) {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  return Math.hypot(dx, dy);
}

export function detectGesture(landmarks, now, sessionStarted, lastHitTime, lastGuitarTime, triggerDrum, triggerGuitar, frogSprite, frogIdleSrc) {
  if (!sessionStarted) {
    return { isV: false, isPaper: false };
  }

  const indexTip = landmarks[8];
  const indexPip = landmarks[6];
  const indexMcp = landmarks[5];

  const middleTip = landmarks[12];
  const middlePip = landmarks[10];
  const middleMcp = landmarks[9];

  const ringTip = landmarks[16];
  const ringPip = landmarks[14];
  const ringMcp = landmarks[13];

  const pinkyTip = landmarks[20];
  const pinkyPip = landmarks[18];
  const pinkyMcp = landmarks[17];

  const wrist = landmarks[0];
  const palmScale = Math.max(distance(wrist, landmarks[9]), 0.001);
  const vSpread = distance(indexTip, middleTip) / palmScale;

  const indexExtended = indexTip.y < indexPip.y && indexPip.y < indexMcp.y;
  const middleExtended = middleTip.y < middlePip.y && middlePip.y < middleMcp.y;
  const ringExtended = ringTip.y < ringPip.y && ringPip.y < ringMcp.y;
  const pinkyExtended = pinkyTip.y < pinkyPip.y && pinkyPip.y < pinkyMcp.y;

  const ringFolded = ringTip.y > ringPip.y;
  const pinkyFolded = pinkyTip.y > pinkyPip.y;

  const isVGesture = indexExtended && middleExtended && ringFolded && pinkyFolded && vSpread > 0.38;
  const isPaperGesture = indexExtended && middleExtended && ringExtended && pinkyExtended;

  return {
    isV: isVGesture,
    isPaper: isPaperGesture
  };
}

const CONFIDENCE_ENTER = 0.72;
const STABLE_FRAMES = 3;
const CLEAR_FRAMES = 2;

const stableState = {
  candidateLabel: "None",
  candidateFrames: 0,
  noneFrames: 0,
  stableLabel: "None",
  confidence: 0,
  source: "rules"
};

const GESTURE_MODE = (() => {
  const params = new URLSearchParams(window.location.search);
  const queryMode = params.get("gestureMode");
  const globalMode = typeof window.__JAMJAM_GESTURE_MODE === "string"
    ? window.__JAMJAM_GESTURE_MODE
    : null;
  const raw = (queryMode || globalMode || "hybrid").trim().toLowerCase();

  if (raw === "rules" || raw === "model" || raw === "hybrid") {
    return raw;
  }
  return "hybrid";
})();

export function getGestureMode() {
  return GESTURE_MODE;
}

function normalizeModelLabel(rawLabel) {
  const label = String(rawLabel || "").trim().toLowerCase();
  if (label === "fist" || label === "isfist" || label === "class1") return "Fist";
  if (
    label === "open palm" ||
    label === "open_palm" ||
    label === "openpalm" ||
    label === "paper" ||
    label === "ispaper" ||
    label === "class2"
  ) return "OpenPalm";
  if (label === "v" || label === "isv") return "V";
  if (label === "pinky" || label === "ispinky" || label === "class4") return "Pinky";
  if (label === "animal" || label === "isanimal" || label === "class5") return "Animal";
  if (label === "k-heart" || label === "kheart" || label === "is_k_heart" || label === "class6") return "KHeart";
  return "None";
}

function mapRulesToResult(rules) {
  if (rules.isV) return { label: "V", confidence: 1, source: "rules" };
  if (rules.isPaper) return { label: "OpenPalm", confidence: 1, source: "rules" };
  return { label: "None", confidence: 0, source: "rules" };
}

function mapModelToResult(modelPrediction) {
  if (!modelPrediction) return null;

  const label = normalizeModelLabel(modelPrediction.label);
  const confidence = Number.isFinite(modelPrediction.confidence) ? modelPrediction.confidence : 0;

  if (label === "None" || confidence < CONFIDENCE_ENTER) {
    return { label: "None", confidence, source: "model" };
  }

  return { label, confidence, source: "model" };
}

function stabilize(rawResult) {
  if (rawResult.label === "None") {
    stableState.noneFrames += 1;
    stableState.candidateLabel = "None";
    stableState.candidateFrames = 0;

    if (stableState.noneFrames >= CLEAR_FRAMES) {
      stableState.stableLabel = "None";
      stableState.confidence = 0;
      stableState.source = rawResult.source;
    }
  } else {
    stableState.noneFrames = 0;

    if (stableState.candidateLabel === rawResult.label) {
      stableState.candidateFrames += 1;
    } else {
      stableState.candidateLabel = rawResult.label;
      stableState.candidateFrames = 1;
    }

    if (stableState.candidateFrames >= STABLE_FRAMES) {
      stableState.stableLabel = rawResult.label;
      stableState.confidence = rawResult.confidence;
      stableState.source = rawResult.source;
    }
  }

  return {
    label: stableState.stableLabel,
    confidence: stableState.confidence,
    source: stableState.source,
    isV: stableState.stableLabel === "V",
    isPaper: stableState.stableLabel === "OpenPalm"
  };
}

export function resolveGesture(landmarks, now, sessionStarted) {
  if (!sessionStarted) {
    return {
      label: "None",
      confidence: 0,
      source: "rules",
      isV: false,
      isPaper: false
    };
  }

  const modelPrediction = getModelPrediction(landmarks, now);
  const modelResult = mapModelToResult(modelPrediction) || {
    label: "None",
    confidence: 0,
    source: "model"
  };

  let rawResult;
  if (GESTURE_MODE === "model") {
    // 모델 전용 모드: 규칙 기반 로직을 전혀 호출하지 않습니다.
    rawResult = modelResult;
  } else {
    const rules = detectGesture(landmarks, now, sessionStarted, 0, 0);
    const ruleResult = mapRulesToResult(rules);

    if (GESTURE_MODE === "rules") {
      rawResult = ruleResult;
    } else {
      rawResult = modelResult.label !== "None" ? modelResult : ruleResult;
    }
  }

  return stabilize(rawResult);
}
