// [main.js] 이 프로그램의 '두뇌' 역할을 하는 가장 중요한 파일입니다.
// 카메라를 켜고, 내 손이 어디 있는지 찾아내고, 그 손이 악기에 닿았을 때 소리를 내라고 명령하는 모든 과정을 지휘합니다.

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision"; // 미디어파이프(손 인식 도구)를 가져옵니다.
import * as Audio from "./audio.js"; // 소리 재생 관련 기능을 가져옵니다.
import * as Renderer from "./renderer.js"; // 화면 그리기 관련 기능을 가져옵니다.
import { resolveGesture, setModelPredictionProvider, resetGestureState } from "./gestures.js"; // 손동작 인식 로직을 가져옵니다.

// 시퀀스 모델 사용 여부를 URL 파라미터로 결정 (기본값: 시퀀스 모델 사용)
const USE_SEQUENCE_MODEL = (() => {
  const params = new URLSearchParams(window.location.search);
  const seqModel = params.get("seqModel");
  // seqModel=0 또는 false면 기존 모델, 그 외에는 시퀀스 모델 사용
  if (seqModel === "0" || seqModel === "false") return false;
  return true; // 기본값: 시퀀스 모델 사용
})();

// 모델 추론 함수들 (init에서 동적 로딩 후 할당됨)
let getModelPrediction = null;
let getModelInferenceStatus = null;
const TEST_MODE_HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];
const TEST_MODE_HAND_COLORS = {
  left: {
    stroke: "rgba(113, 220, 255, 0.95)",
    fill: "rgba(214, 248, 255, 0.95)"
  },
  right: {
    stroke: "rgba(255, 204, 113, 0.95)",
    fill: "rgba(255, 244, 214, 0.95)"
  }
};
import { getConfiguredHandLandmarkerTaskPath, getConfiguredMediaPipeWasmRoot, getConfiguredSplitHandInference } from "./env_config.js";
import { setupSeamlessBackgroundLoop, applySceneMode } from "./scene_runtime.js";
import { createParticleSystem, restartClassAnimation } from "./particle_system.js";
import { DEFAULT_SOUND_MAPPING, loadSoundMapping, getSoundProfileForInstrument, loadGestureMapping } from "./sound_mapping.js";
import { createInteractionRuntime } from "./interaction_runtime.js";
import { createHandTrackingRuntime } from "./hand_tracking_runtime.js";
import { createControlRuntime } from "./control_runtime.js";

// CSS 기반 애니메이션 시스템 사용 (Lottie 대신 경량 CSS 애니메이션)
async function loadAnimationManagerFactory() {
  console.log("[Animation] Using CSS-based animation system");
  const animModule = await import("./instrument_animations_simple.js");
  return animModule.createInstrumentAnimationManager;
}

function createNoopAnimationManager() {
  return {
    initAnimation() {},
    setState() {},
    trigger() {},
    hover() {},
    setFeverMode() {},
    updateProximity() {},
    setPaused() {},
    destroy() {},
    getState() { return "idle"; },
    isLoaded() { return false; }
  };
}

function createNoopFeverController() {
  return {
    registerHit() {},
    triggerFever() {},
    updateFeverState() {},
    isFever: () => false
  };
}

// 화면에 보이는 각종 버튼, 글자, 캔버스(그림판)들을 컴퓨터가 찾을 수 있게 주소를 연결해두는 곳입니다.
let video = null; // 현재 활성화된 웹캠 비디오 태그를 가리킵니다.
let handCanvas = null; // 현재 활성화된 손 오버레이 캔버스를 가리킵니다.
const effectCanvas = document.getElementById("effectCanvas"); // 터치 효과(반짝임)를 그릴 투명 도화지를 가져옵니다.
let handCtx = null; // 현재 손 캔버스의 2D context 입니다.
const effectCtx = effectCanvas.getContext("2d"); // 효과 도화지에 그림을 그리기 위한 도구(붓)를 꺼냅니다.
const statusText = document.getElementById("status"); // 현재 상태(안내 문구)를 보여줄 글자 칸을 가져옵니다.
const scene = document.getElementById("scene"); // 전체 배경이 되는 공간을 가져옵니다.
const handCursor = document.getElementById("handCursor"); // 내 손가락 끝을 따라다닐 동그란 커서를 가져옵니다.
const landingOverlay = document.getElementById("landingOverlay"); // 처음 시작할 때 보이는 덮개 화면을 가져옵니다.
const landingStartButton = document.getElementById("landingStartButton"); // 시작하기 버튼을 가져옵니다.
const pulseMessage = document.getElementById("pulseMessage"); // 화면 중앙에 뜨는 안내 메시지를 가져옵니다.
const gestureSquirrelEffect = document.getElementById("gestureSquirrelEffect"); // 다람쥐 효과 이미지를 가져옵니다.
const testModeToggleButton = document.getElementById("testModeToggleButton");
const testModeDock = document.getElementById("testModeDock");
const testModePanel = document.getElementById("testModePanel");
const testModeVision = document.getElementById("testModeVision");
const testModeSummary = document.getElementById("testModeSummary");
const testModeVisionSummary = document.getElementById("testModeVisionSummary");
const testModeSession = document.getElementById("testModeSession");
const testModeHands = document.getElementById("testModeHands");
const testModeModel = document.getElementById("testModeModel");
const testModeInFlight = document.getElementById("testModeInFlight");
const testModeLastInference = document.getElementById("testModeLastInference");
const testModeAudioState = document.getElementById("testModeAudioState");
const testModeSoundEnabled = document.getElementById("testModeSoundEnabled");
const testModeAudioUnlocked = document.getElementById("testModeAudioUnlocked");
const testModeAudioVoices = document.getElementById("testModeAudioVoices");
const testModeAudioPanMode = document.getElementById("testModeAudioPanMode");
const testModeLastSoundKey = document.getElementById("testModeLastSoundKey");
const testModeLastPlayMode = document.getElementById("testModeLastPlayMode");
const testModeLastPan = document.getElementById("testModeLastPan");
const testModeLastGainMode = document.getElementById("testModeLastGainMode");
const testModeLastSound = document.getElementById("testModeLastSound");
const testModePlayBeepButton = document.getElementById("testModePlayBeepButton");
const testModeCenterPanButton = document.getElementById("testModeCenterPanButton");
const testModeRestorePanButton = document.getElementById("testModeRestorePanButton");
const testModeWebcamPreview = document.getElementById("testModeWebcamPreview");
const testModeRawCanvas = document.getElementById("testModeRawCanvas");
const testModeNormalizedCanvas = document.getElementById("testModeNormalizedCanvas");
const testModeNormalizedGestureDisplay = document.getElementById("testModeNormalizedGestureDisplay");
const testModeRawCtx = testModeRawCanvas ? testModeRawCanvas.getContext("2d") : null;
const testModeNormalizedCtx = testModeNormalizedCanvas ? testModeNormalizedCanvas.getContext("2d") : null;
const testModeLeftRaw = document.getElementById("testModeLeftRaw");
const testModeLeftFinal = document.getElementById("testModeLeftFinal");
const testModeLeftGestureDisplay = document.getElementById("testModeLeftGestureDisplay");
const testModeLeftSource = document.getElementById("testModeLeftSource");
const testModeLeftObject = document.getElementById("testModeLeftObject");
const testModeLeftInferenceMs = document.getElementById("testModeLeftInferenceMs");
const testModeLeftSoundMs = document.getElementById("testModeLeftSoundMs");
const testModeLeftMelody = document.getElementById("testModeLeftMelody");
const testModeRightRaw = document.getElementById("testModeRightRaw");
const testModeRightFinal = document.getElementById("testModeRightFinal");
const testModeRightGestureDisplay = document.getElementById("testModeRightGestureDisplay");
const testModeRightSource = document.getElementById("testModeRightSource");
const testModeRightObject = document.getElementById("testModeRightObject");
const testModeRightInferenceMs = document.getElementById("testModeRightInferenceMs");
const testModeRightSoundMs = document.getElementById("testModeRightSoundMs");
const testModeRightMelody = document.getElementById("testModeRightMelody");
const testModeFieldEls = {
  left: {
    raw: testModeLeftRaw,
    final: testModeLeftFinal,
    gestureDisplay: testModeLeftGestureDisplay,
    source: testModeLeftSource,
    object: testModeLeftObject,
    inferenceMs: testModeLeftInferenceMs,
    soundMs: testModeLeftSoundMs,
    melody: testModeLeftMelody
  },
  right: {
    raw: testModeRightRaw,
    final: testModeRightFinal,
    gestureDisplay: testModeRightGestureDisplay,
    source: testModeRightSource,
    object: testModeRightObject,
    inferenceMs: testModeRightInferenceMs,
    soundMs: testModeRightSoundMs,
    melody: testModeRightMelody
  }
};

function syncRuntimeDomRefs() {
  video = document.getElementById("webcam");
  handCanvas = document.getElementById("handCanvas");
  handCtx = handCanvas ? handCanvas.getContext("2d") : null;
  return { video, handCanvas, handCtx };
}

syncRuntimeDomRefs();

const instrumentElements = { // 각 동물 악기들의 HTML 요소를 하나로 묶어둡니다.
  drum: document.getElementById("instrumentDrum"), // 고슴도치 드럼 DOM입니다.
  xylophone: document.getElementById("instrumentXylophone"), // 아기 사슴 오브젝트를 찾습니다.
  tambourine: document.getElementById("instrumentTambourine"), // 아기 토끼 오브젝트를 찾습니다.
  a: document.getElementById("instrumentA"), // 다람쥐 DOM입니다.
  cat: document.getElementById("instrumentCat"), // 고양이 DOM입니다.
  penguin: document.getElementById("instrumentPenguin") // 팽귄 DOM입니다.
};
const VIDEO_INSTRUMENT_IDS = ["drum", "penguin"];
const videoInstruments = {}; // { [id]: { video, canvas, ctx, workCanvas, workCtx, lastFrameAt, raf } }

let gestureObjectActive = false;
const VIDEO_RENDER_FPS = 15; 
const VIDEO_RENDER_INTERVAL_MS = 1000 / VIDEO_RENDER_FPS;
const VIDEO_PROCESS_MAX_DIM = 480; 

const VIDEO_BLACK_THRESHOLD = 58;
const VIDEO_SOFT_BLACK_THRESHOLD = 96;

const COLLISION_PADDING = 12; // 손이 악기에 완전히 닿지 않아도 조금 근처에만 가도 인식되게 하는 '여유 공간'입니다.
const START_HOVER_MS = 220; // 시작 버튼 위에 손을 얼마나 오래 올려두어야 게임이 시작되는지 결정하는 시간(0.52초)입니다.
const ENABLE_AMBIENT_AUDIO = false; // 동작 인식 시에만 소리가 나도록 배경 앰비언트는 끕니다.

const DEFAULT_INFER_FPS = 15; // 별도 설정이 없을 때 1초에 몇 번 손 위치를 계산할지 나타냅니다. (기본 15회)
const MIN_INFER_FPS = 12; // 너무 느려지지 않게 최대로 늦출 수 있는 계산 횟수입니다.
const MAX_INFER_FPS = 60; // 컴퓨터 성능이 좋아도 최대 60회까지만 계산하도록 제한합니다.
const LANDMARK_STALE_MS = 260; // 손이 화면에서 사라졌을 때 약 0.26초 동안은 마지막 위치를 기억하고 보여줍니다.

let handLandmarker; // 미디어파이프 손 인식 도구 객체를 담아둘 변수입니다.
let sessionStarted = false; // 게임이 실제로 시작되었는지 여부를 나타냅니다.
let cameraStream = null; // 웹캠에서 나오는 영상 신호를 담아둡니다.
let adminEditMode = new URLSearchParams(window.location.search).get("admin") === "1"; // 현재 악기 배치를 수정하는 관리자 모드인지 확인합니다.

const GESTURE_TRIGGER_COOLDOWN_MS = 280; // 손동작 인식이 너무 자주 일어나지 않게 하는 대기 시간(0.28초)입니다.
const BG_VIDEO_CROSSFADE_SEC = 0.42; // 배경 영상이 바뀔 때 자연스럽게 겹치는 시간(0.42초)입니다.
const PERF_LOG_KEY = "jamjam.perf.logs.v1";
const PERF_LOG_LIMIT = 200;

const SOUND_PROFILES = {
  drum: { soundTag: "드럼 비트", burstType: "drum", playbackMode: "oneshot", melodyType: "drum", play: (note) => Audio.playKids_Drum(note) },
  piano: { soundTag: "피아노 선율", burstType: "xylophone", playbackMode: "oneshot", melodyType: "piano", play: (note) => Audio.playKids_Piano(note) },
  guitar: { soundTag: "기타 스트럼", burstType: "tambourine", playbackMode: "oneshot", melodyType: "guitar", play: (note) => Audio.playKids_Guitar(note) },
  flute: { soundTag: "플룻 멜로디", burstType: "heart", playbackMode: "oneshot", melodyType: "flute", play: (note) => Audio.playKids_Flute(note) },
  violin: { soundTag: "바이올린 하모니", burstType: "animal", playbackMode: "oneshot", melodyType: "violin", play: (note) => Audio.playKids_Violin(note) },
  bell: { soundTag: "벨 포인트", burstType: "pinky", playbackMode: "oneshot", melodyType: "bell", play: (note) => Audio.playKids_Bell(note) }
};

// GESTURE_SOUND_PROFILES는 제거됨 - gestureMapping과 SOUND_PROFILES 조합으로 대체

// 주소창(URL)에 적힌 설정값을 보고, 손 위치를 얼마나 자주 계산할지(FPS) 결정하는 기능입니다.
function parseInferFps() {
  const params = new URLSearchParams(window.location.search); // 주소창의 파라미터(?표 뒤의 글자들)를 읽어옵니다.
  const raw = Number(params.get("inferFps")); // 'inferFps'라는 이름의 숫자를 가져옵니다.
  if (!Number.isFinite(raw)) return DEFAULT_INFER_FPS; // 숫자가 아니면 기본값(15)을 사용합니다.
  return clamp(Math.round(raw), MIN_INFER_FPS, MAX_INFER_FPS); // 설정된 범위(8~60) 안으로 숫자를 맞춥니다.
}

const INFER_INTERVAL_MS = Math.round(1000 / parseInferFps()); // 계산 횟수를 보고 몇 밀리초(초의 1000분의 1)마다 계산할지 정합니다.

function parsePreferredDelegate() { // 인공지능 계산을 CPU로 할지 GPU(그래픽카드)로 할지 정하는 기능입니다.
  const params = new URLSearchParams(window.location.search); // 주소창 파라미터를 읽습니다.
  const raw = (params.get("mpDelegate") || "gpu").trim().toUpperCase(); // 'mpDelegate' 값을 가져와 대문자로 바꿉니다. 기본은 GPU입니다.
  return raw === "CPU" ? "CPU" : "GPU"; // CPU가 아니면 무조건 GPU를 사용하도록 합니다.
}

function parseInteractionMode() { // 터치로 할지 손동작으로 할지 플레이 방식을 정하는 기능입니다.
  const params = new URLSearchParams(window.location.search); // 주소창 파라미터를 읽습니다.
  const raw = (params.get("interactionMode") || "gesture").trim().toLowerCase(); // 'interactionMode' 값을 읽습니다. 기본은 손동작입니다.
  return raw === "touch" ? "touch" : "gesture"; // touch 가 아니면 gesture 방식을 사용합니다.
}

function parseNumHands() {
  const params = new URLSearchParams(window.location.search);
  const raw = Number(params.get("numHands"));
  if (!Number.isFinite(raw)) return 1;
  return clamp(Math.round(raw), 1, 2);
}

// 프로그램이 동작하면서 기억해야 할 '현재 상태' 값들입니다. (예: 지금 카메라가 켜져 있는지, 피버 타임인지 등)
const INTERACTION_MODE = parseInteractionMode();
const NUM_HANDS = parseNumHands();
const HAND_DETECTION_TARGET = NUM_HANDS;
const ENABLE_SPLIT_HAND_INFERENCE = getConfiguredSplitHandInference();
let soundMapping = loadSoundMapping(SOUND_PROFILES);
const gestureMapping = loadGestureMapping();
const particleSystem = createParticleSystem(effectCtx, effectCanvas);
let animationManager = createNoopAnimationManager();
const feverController = createNoopFeverController();
const lastSoundEventByHand = new Map();

let testModeEnabled = (() => {
  const params = new URLSearchParams(window.location.search);
  const queryValue = params.get("testMode");
  if (queryValue === "1" || queryValue === "true") return true;
  if (queryValue === "0" || queryValue === "false") return false;
  return false;
})();

function formatDisplayGesture(label, confidence = null, classId = null) {
  const normalized = String(label || "").trim().toLowerCase();
  let displayLabel = "아무것도 아님";
  if (normalized === "fist" || classId === 1) displayLabel = "주먹";
  else if (normalized === "openpalm" || normalized === "open_palm" || normalized === "open palm" || classId === 2) displayLabel = "손바닥";
  else if (normalized === "v" || classId === 3) displayLabel = "브이";
  else if (normalized === "pinky" || classId === 4) displayLabel = "새끼손가락";
  else if (normalized === "animal" || classId === 5) displayLabel = "애니멀";
  else if (normalized === "kheart" || normalized === "k-heart" || classId === 6) displayLabel = "K-하트";
  else if (normalized && normalized !== "none" && normalized !== "class0") displayLabel = label;

  if (!Number.isFinite(confidence)) return displayLabel;
  return `${displayLabel} ${(confidence * 100).toFixed(0)}%`;
}

function getInstrumentName(instrumentId) {
  const instrument = instruments.find((item) => item.id === instrumentId);
  return instrument?.name || "-";
}

function formatMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}ms` : "-";
}

function setPanelValue(element, value) {
  if (!element) return;
  element.textContent = value;
}

function setGestureDisplay(element, label, active) {
  if (!element) return;
  element.textContent = label;
  element.classList.toggle("is-active", active);
  element.classList.toggle("is-idle", !active);
}

function formatBoolean(value) {
  return value ? "Yes" : "No";
}

function hasMeaningfulGesture(label, classId = null) {
  const normalized = String(label || "").trim().toLowerCase();
  return !(
    !normalized ||
    normalized === "none" ||
    normalized === "class0" ||
    normalized === "neutral" ||
    classId === 0
  );
}

function getTestModeHandColor(handKey = "default") {
  return TEST_MODE_HAND_COLORS[handKey] || {
    stroke: "rgba(255, 255, 255, 0.92)",
    fill: "rgba(245, 245, 245, 0.92)"
  };
}

function getMirrorPivotX(landmarks) {
  const wristX = Number.isFinite(landmarks?.[0]?.x) ? landmarks[0].x : 0.5;
  const indexMcpX = Number.isFinite(landmarks?.[5]?.x) ? landmarks[5].x : wristX;
  const pinkyMcpX = Number.isFinite(landmarks?.[17]?.x) ? landmarks[17].x : wristX;
  return (wristX + indexMcpX + pinkyMcpX) / 3;
}

function sanitizePreviewLandmarks(landmarks, handKey = "default") {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return null;
  const normalizedHandKey = String(handKey || "default").trim().toLowerCase();
  const shouldMirrorLeft = normalizedHandKey === "left";
  const mirrorPivotX = shouldMirrorLeft ? getMirrorPivotX(landmarks) : 0;
  const features = new Float32Array(63);
  for (let i = 0; i < 21; i += 1) {
    const point = landmarks[i];
    const rawX = Number.isFinite(point?.x) ? point.x : 0;
    const baseOffset = i * 3;
    features[baseOffset] = shouldMirrorLeft ? clamp(mirrorPivotX * 2 - rawX, 0, 1) : rawX;
    features[baseOffset + 1] = Number.isFinite(point?.y) ? point.y : 0;
    features[baseOffset + 2] = Number.isFinite(point?.z) ? point.z : 0;
  }
  return features;
}

function normalizeSequencePreviewFrame(frame63) {
  if (!frame63 || frame63.length < 63) return null;
  const normalized = new Float32Array(63);
  const originX = frame63[0];
  const originY = frame63[1];
  const originZ = frame63[2];
  const dx = frame63[27] - originX;
  const dy = frame63[28] - originY;
  const dz = frame63[29] - originZ;
  const denom = Math.hypot(dx, dy, dz);
  const scale = denom <= 1e-8 ? 1 : 1 / denom;

  for (let i = 0; i < 63; i += 3) {
    normalized[i] = (frame63[i] - originX) * scale;
    normalized[i + 1] = (frame63[i + 1] - originY) * scale;
    normalized[i + 2] = (frame63[i + 2] - originZ) * scale;
  }

  return normalized;
}

function drawHandGraph(ctx, points, handKey, { pointRadius = 4, lineWidth = 2.2 } = {}) {
  if (!ctx || !Array.isArray(points) || points.length < 21) return;
  const { stroke, fill } = getTestModeHandColor(handKey);

  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = stroke;
  ctx.shadowBlur = 10;
  ctx.shadowColor = stroke;

  TEST_MODE_HAND_CONNECTIONS.forEach(([from, to]) => {
    const start = points[from];
    const end = points[to];
    if (!start || !end) return;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
  });

  ctx.fillStyle = fill;
  ctx.shadowBlur = 0;
  points.forEach((point, index) => {
    if (!point) return;
    ctx.beginPath();
    ctx.arc(point.x, point.y, index === 8 ? pointRadius + 1.4 : pointRadius, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawNormalizedPreviewGuide(ctx, width, height) {
  if (!ctx) return;
  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 5]);
  ctx.beginPath();
  ctx.moveTo(width * 0.5, 14);
  ctx.lineTo(width * 0.5, height - 14);
  ctx.moveTo(14, height * 0.5);
  ctx.lineTo(width - 14, height * 0.5);
  ctx.stroke();
  ctx.restore();
}

function getModelInputPreviewSummary() {
  return USE_SEQUENCE_MODEL
    ? "wrist 원점 + scale 정규화된 모델 입력"
    : "미러링을 반영한 모델 입력 좌표";
}

function renderNormalizedVisionCanvas(ctx, canvas, debugSnapshot, handKeys) {
  if (!ctx || !canvas) return;

  const normalizedWidth = canvas.width;
  const normalizedHeight = canvas.height;
  ctx.clearRect(0, 0, normalizedWidth, normalizedHeight);
  drawNormalizedPreviewGuide(ctx, normalizedWidth, normalizedHeight);

  handKeys.forEach((handKey) => {
    const hand = debugSnapshot[handKey];
    const landmarks = hand?.lastLandmarks;
    if (!Array.isArray(landmarks) || landmarks.length < 21) return;

    const sanitized = sanitizePreviewLandmarks(landmarks, handKey);
    const modelInput = USE_SEQUENCE_MODEL ? normalizeSequencePreviewFrame(sanitized) : sanitized;
    if (!modelInput) return;

    const normalizedPoints = [];
    if (USE_SEQUENCE_MODEL) {
      let maxAbs = 0.25;
      for (let i = 0; i < 63; i += 3) {
        maxAbs = Math.max(maxAbs, Math.abs(modelInput[i]), Math.abs(modelInput[i + 1]));
      }
      const scale = (Math.min(normalizedWidth, normalizedHeight) * 0.34) / maxAbs;
      for (let i = 0; i < 63; i += 3) {
        normalizedPoints.push({
          x: normalizedWidth * 0.5 + modelInput[i] * scale,
          y: normalizedHeight * 0.5 + modelInput[i + 1] * scale
        });
      }
    } else {
      for (let i = 0; i < 63; i += 3) {
        normalizedPoints.push({
          x: modelInput[i] * normalizedWidth,
          y: modelInput[i + 1] * normalizedHeight
        });
      }
    }
    drawHandGraph(ctx, normalizedPoints, handKey, { pointRadius: 3.8, lineWidth: 2.1 });
  });
}

function renderTestModeVision(debugSnapshot, handKeys) {
  if (!testModeEnabled) return;
  if (!testModeRawCanvas || !testModeNormalizedCanvas || !testModeRawCtx || !testModeNormalizedCtx) return;

  const rawWidth = testModeRawCanvas.width;
  const rawHeight = testModeRawCanvas.height;

  testModeRawCtx.clearRect(0, 0, rawWidth, rawHeight);
  const summary = USE_SEQUENCE_MODEL
    ? "좌: 원본 손 / 우: wrist 원점 + scale 정규화"
    : "좌: 원본 손 / 우: 모델 입력 좌표(미러링 포함)";
  setPanelValue(testModeVisionSummary, summary);

  handKeys.forEach((handKey) => {
    const hand = debugSnapshot[handKey];
    const landmarks = hand?.lastLandmarks;
    if (!Array.isArray(landmarks) || landmarks.length < 21) return;

    const rawPoints = landmarks.map((point) => ({
      x: (1 - point.x) * rawWidth,
      y: point.y * rawHeight
    }));
    drawHandGraph(testModeRawCtx, rawPoints, handKey, { pointRadius: 3.4, lineWidth: 2 });
  });

  const focusHandKey = handKeys.includes("right") ? "right" : (handKeys[0] || null);
  const focusHand = focusHandKey ? debugSnapshot[focusHandKey] : null;
  const focusResolved = focusHand?.lastResolvedGesture || null;
  const focusRaw = focusHand?.lastRawModelPrediction || null;
  const finalActive = hasMeaningfulGesture(focusResolved?.label);
  const rawActive = hasMeaningfulGesture(focusRaw?.label, focusRaw?.classId ?? null);
  const gestureHeadline = finalActive
    ? formatDisplayGesture(focusResolved?.label, focusResolved?.confidence)
    : (rawActive ? `${formatDisplayGesture(focusRaw?.label, focusRaw?.confidence, focusRaw?.classId ?? null)} 후보` : "대기 중");
  setGestureDisplay(testModeNormalizedGestureDisplay, gestureHeadline, finalActive || rawActive);

  renderNormalizedVisionCanvas(testModeNormalizedCtx, testModeNormalizedCanvas, debugSnapshot, handKeys);
}

function renderTutorialModelVision(debugSnapshot, handKeys) {
  const tutorialCanvas = document.getElementById("tutorialNormalizedCanvas");
  if (!(tutorialCanvas instanceof HTMLCanvasElement)) return;
  const tutorialCtx = tutorialCanvas.getContext("2d");
  if (!tutorialCtx) return;

  const tutorialSummary = document.getElementById("tutorialVisionSummary");
  setPanelValue(tutorialSummary, `왼쪽은 원본 손, 오른쪽은 ${getModelInputPreviewSummary()}입니다.`);
  renderNormalizedVisionCanvas(tutorialCtx, tutorialCanvas, debugSnapshot, handKeys);
}

function syncTestModeUI() {
  if (testModeToggleButton) {
    testModeToggleButton.textContent = testModeEnabled ? "테스트 모드 끄기" : "테스트 모드 켜기";
    testModeToggleButton.setAttribute("aria-pressed", String(testModeEnabled));
  }
  if (testModeDock) {
    testModeDock.classList.toggle("is-hidden", !testModeEnabled);
  }
}

function renderTestModePanel() {
  const debugSnapshot = interactionRuntime.getDebugSnapshot?.() || {};
  const handKeys = Object.keys(debugSnapshot).filter((handKey) => {
    const hand = debugSnapshot[handKey];
    return Boolean(hand?.lastUpdatedAt);
  });

  renderTutorialModelVision(debugSnapshot, handKeys);

  if (!testModeEnabled) return;

  const modelStatus = getModelInferenceStatus(performance.now());

  setPanelValue(testModeSummary, "Raw와 Final을 동시에 보며 추론/후처리를 구분합니다.");
  setPanelValue(testModeSession, sessionStarted ? "시작됨" : "대기");
  const visibleHandKeys = handKeys.filter((handKey) => handKey !== "left");
  setPanelValue(testModeHands, visibleHandKeys.length > 0 ? visibleHandKeys.join(", ") : "없음");
  setPanelValue(testModeModel, modelStatus.endpointConfigured ? `${modelStatus.mode} ready` : "loading");
  setPanelValue(
    testModeInFlight,
    modelStatus.recentInference
      ? `Yes${Number.isFinite(modelStatus.lastCompletedAgoMs) ? ` (${Math.round(modelStatus.lastCompletedAgoMs)}ms 전)` : ""}`
      : "No"
  );
  setPanelValue(testModeLastInference, formatMs(modelStatus.lastDurationMs));
  const audioDebugState = Audio.getAudioDebugState();
  const lastPlayback = audioDebugState.lastPlayback || null;
  setPanelValue(testModeAudioState, audioDebugState.contextState);
  setPanelValue(testModeSoundEnabled, formatBoolean(audioDebugState.soundEnabled));
  setPanelValue(testModeAudioUnlocked, formatBoolean(audioDebugState.unlocked));
  setPanelValue(testModeAudioVoices, String(audioDebugState.activeVoiceCount));
  setPanelValue(testModeAudioPanMode, Audio.getPanDebugMode());
  setPanelValue(testModeLastSoundKey, lastPlayback?.soundKey || "-");
  setPanelValue(testModeLastPlayMode, lastPlayback?.playMode || "-");
  setPanelValue(testModeLastPan, Number.isFinite(lastPlayback?.pan) ? String(lastPlayback.pan) : "-");
  setPanelValue(testModeLastGainMode, lastPlayback?.gainMode || "-");
  setPanelValue(
    testModeLastSound,
    lastPlayback
      ? `${lastPlayback.soundKey || "-"} / ${lastPlayback.playMode || "-"} / pan ${Number.isFinite(lastPlayback.pan) ? lastPlayback.pan : "-"}`
      : "-"
  );

  ["left", "right"].forEach((handKey) => {
    const hand = debugSnapshot[handKey];
    const rawModel = hand?.lastRawModelPrediction || null;
    const resolved = hand?.lastResolvedGesture || null;
    const soundEvent = lastSoundEventByHand.get(handKey) || null;
    const instrumentId = resolved?.label && resolved.label !== "None"
      ? (gestureMapping[resolved.label] || null)
      : null;
    const fields = testModeFieldEls[handKey];
    if (!fields) return;

    setPanelValue(fields.raw, formatDisplayGesture(rawModel?.label, rawModel?.confidence, rawModel?.classId ?? null));
    setPanelValue(fields.final, formatDisplayGesture(resolved?.label, resolved?.confidence));
    const finalActive = hasMeaningfulGesture(resolved?.label);
    const rawActive = hasMeaningfulGesture(rawModel?.label, rawModel?.classId ?? null);
    const gestureHeadline = finalActive
      ? formatDisplayGesture(resolved?.label, resolved?.confidence)
      : (rawActive ? `${formatDisplayGesture(rawModel?.label, rawModel?.confidence, rawModel?.classId ?? null)} 후보` : "대기 중");
    setGestureDisplay(fields.gestureDisplay, gestureHeadline, finalActive || rawActive);
    setPanelValue(fields.source, resolved?.source || "-");
    setPanelValue(fields.object, instrumentId ? getInstrumentName(instrumentId) : "-");
    setPanelValue(fields.inferenceMs, formatMs(rawModel?.elapsed_ms));
    setPanelValue(fields.soundMs, formatMs(soundEvent?.inferenceLatencyMs));
    setPanelValue(fields.melody, hand?.currentMelodyType || "-");
  });

  renderTestModeVision(debugSnapshot, handKeys);
}

function startTestModeLoop() {
  const tick = () => {
    renderTestModePanel();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function getMappedSoundProfile(instrumentId) {
  return getSoundProfileForInstrument(soundMapping, DEFAULT_SOUND_MAPPING, SOUND_PROFILES, instrumentId);
}

function playMappedInstrumentSound(instrumentId, element, { note, spawnEffect = true } = {}) {
  const profile = getMappedSoundProfile(instrumentId);
  profile.play(note);
  if (spawnEffect && element) {
    spawnBurst(profile.burstType, element);
  }
  return profile;
}

function getGestureSoundProfile(label, instrumentId) {
  // gestureMapping에서 제스쳐에 맞는 오브젝트를 찾고,
  // 그 오브젝트의 사운드 프로필을 반환
  return getMappedSoundProfile(instrumentId);
}

function playGestureMappedSound(label, instrumentId, { note, spawnEffect = true } = {}) {
  const element = instrumentElements[instrumentId] || null;
  const profile = getGestureSoundProfile(label, instrumentId);
  profile.play(note);
  if (spawnEffect && element) {
    spawnBurst(profile.burstType, element);
  }
  return profile;
}

// 우리가 연주할 수 있는 '동물 악기'들의 정보입니다. 이름과 소리, 그리고 닿았을 때 어떤 행동을 할지 적혀 있습니다.
const instruments = [
  {
    id: "drum", // 악기의 고유 ID입니다.
    name: "고슴도치 드럼", // 화면에 보일 실제 이름입니다.
    soundTag: "쿵", // 연주했을 때 표시될 소리 느낌표입니다.
    el: instrumentElements.drum, // 실제 HTML 이미지를 연결합니다.
    cooldownMs: 320, // 한 번 연주한 뒤 다시 연주하기 위해 기다려야 하는 시간입니다.
    lastHitAt: 0, // 마지막으로 연주된 시간을 기록합니다.
    onHit(note) { // 연주되었을 때 실행할 행동입니다.
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id); // Lottie 애니메이션 트리거
      return profile.soundTag || this.soundTag;
    }
  },
  {
    id: "xylophone", // 실로폰 악기입니다.
    name: "아기 사슴", // 이름입니다.
    soundTag: "사슴 멜로디", // 글자 표시입니다.
    el: instrumentElements.xylophone, // 이미지를 찾습니다.
    cooldownMs: 360, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit(note) { // 누르면 실행합니다.
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id); // Lottie 애니메이션 트리거
      return profile.soundTag || this.soundTag;
    }
  },
  {
    id: "tambourine", // 탬버린 악기입니다.
    name: "아기 토끼", // 이름입니다.
    soundTag: "토끼 리듬", // 글자 표시입니다.
    el: instrumentElements.tambourine, // 이미지를 찾습니다.
    cooldownMs: 380, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit(note) { // 누르면 실행합니다.
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id); // Lottie 애니메이션 트리거
      return profile.soundTag || this.soundTag;
    }
  },
  {
    id: "a", // 다람쥐입니다.
    name: "다람쥐", // 이름입니다.
    soundTag: "다람쥐 포인트", // 글자 표시입니다.
    el: instrumentElements.a, // 이미지를 찾습니다.
    cooldownMs: 380, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit(note) { // 누르면 실행합니다.
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id); // Lottie 애니메이션 트리거
      return profile.soundTag || this.soundTag;
    }
  },
  {
    id: "cat",
    name: "고양이",
    soundTag: "고양이 멜로디",
    el: instrumentElements.cat,
    cooldownMs: 380,
    lastHitAt: 0,
    onHit(note) {
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id);
      return profile.soundTag || this.soundTag;
    }
  },
  {
    id: "penguin",
    name: "팽귄 실로폰",
    soundTag: "팽귄 선율",
    el: instrumentElements.penguin,
    cooldownMs: 380,
    lastHitAt: 0,
    onHit(note) {
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id);
      return profile.soundTag || this.soundTag;
    }
  }
];

// 브라우저 화면 크기가 바뀔 때마다 그림판(캔버스)의 크기도 똑같이 맞춰주는 기능입니다.
function setCanvasSize() {
  const { handCanvas: activeHandCanvas } = syncRuntimeDomRefs();
  if (activeHandCanvas) {
    activeHandCanvas.width = window.innerWidth; // 손 도화지 가로 길이를 창 크기에 맞춥니다.
    activeHandCanvas.height = window.innerHeight; // 손 도화지 세로 길이를 창 크기에 맞춥니다.
  }
  effectCanvas.width = window.innerWidth; // 효과 도화지 가로 길이를 창 크기에 맞춥니다.
  effectCanvas.height = window.innerHeight; // 효과 도화지 세로 길이를 창 크기에 맞춥니다.
}

// 숫자가 너무 작거나 너무 크지 않게 특정 범위 안에만 있도록 잡아주는 기능입니다.
function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// 화면 오른쪽 아래의 '소리 켜기/끄기' 버튼의 글자를 현재 상태에 맞춰 바꿔주는 기능입니다.
function activateStart() { // 게임을 실제로 시작하는 기능입니다.
  sessionStarted = true; // 세션이 시작되었음을 표시합니다.
  landingOverlay.classList.add("is-hidden"); // 시작 화면 덮개를 숨깁니다.
  const audioState = Audio.getAudioState(); // 현재 오디오 상태를 가져옵니다.
  const playGuide = INTERACTION_MODE === "gesture" // 플레이 방식에 따라 안내 문구를 정합니다.
    ? "손동작으로 숲을 연주해 보세요."
    : "손으로 동물 악기를 터치해 보세요.";
  if (audioState.running) { // 오디오가 켜져 있으면
    statusText.textContent = playGuide; // 안내 문구를 보여줍니다.
    if (ENABLE_AMBIENT_AUDIO) {
      Audio.startAmbientLoop(); // 숲의 배경음을 재생하기 시작합니다.
    } else {
      Audio.stopAmbientLoop();
    }
  } else { // 오디오가 꺼져 있으면
    statusText.textContent = "소리를 들으려면 '소리 켜기' 버튼을 눌러주세요."; // 소리를 켜라는 메시지를 보여줍니다.
  }
}

function registerHit(now) {
  feverController.registerHit(now);
}

function spawnBurst(type, element) {
  particleSystem.spawnBurst(type, element);
}

function getVideoProcessSize(width, height) {
  if (!width || !height) return { processWidth: 0, processHeight: 0 };
  const scale = Math.min(1, VIDEO_PROCESS_MAX_DIM / Math.max(width, height));
  return {
    processWidth: Math.max(1, Math.round(width * scale)),
    processHeight: Math.max(1, Math.round(height * scale))
  };
}

function syncVideoInstrumentSize(id) {
  const inst = videoInstruments[id];
  if (!inst || !inst.video || !inst.canvas) return;
  const { video, canvas, workCanvas } = inst;

  const width = video.videoWidth || 0;
  const height = video.videoHeight || 0;
  if (!width || !height) return;

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const { processWidth, processHeight } = getVideoProcessSize(width, height);
  if (workCanvas.width !== processWidth || workCanvas.height !== processHeight) {
    workCanvas.width = processWidth;
    workCanvas.height = processHeight;
  }
}

function drawVideoInstrumentFrame(id, now = performance.now()) {
  const inst = videoInstruments[id];
  if (!inst || !inst.video || !inst.canvas || !inst.ctx || !inst.workCtx) return;

  const { video, canvas, ctx, workCanvas, workCtx, lastFrameAt } = inst;

  if (now - lastFrameAt < VIDEO_RENDER_INTERVAL_MS) {
    inst.raf = requestAnimationFrame((t) => drawVideoInstrumentFrame(id, t));
    return;
  }
  inst.lastFrameAt = now;

  syncVideoInstrumentSize(id);

  const displayWidth = canvas.width || video.videoWidth || 0;
  const displayHeight = canvas.height || video.videoHeight || 0;
  const processWidth = workCanvas.width || 0;
  const processHeight = workCanvas.height || 0;

  if (!displayWidth || !displayHeight || !processWidth || !processHeight || video.readyState < 2) {
    inst.raf = requestAnimationFrame((t) => drawVideoInstrumentFrame(id, t));
    return;
  }

  workCtx.clearRect(0, 0, processWidth, processHeight);
  workCtx.drawImage(video, 0, 0, processWidth, processHeight);
  const frame = workCtx.getImageData(0, 0, processWidth, processHeight);
  const pixels = frame.data;

  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];
    const brightness = Math.max(r, g, b);

    if (brightness <= VIDEO_BLACK_THRESHOLD) {
      pixels[i + 3] = 0;
      continue;
    }

    if (brightness < VIDEO_SOFT_BLACK_THRESHOLD) {
      const alphaScale = (brightness - VIDEO_BLACK_THRESHOLD) / (VIDEO_SOFT_BLACK_THRESHOLD - VIDEO_BLACK_THRESHOLD);
      pixels[i + 3] = Math.round(pixels[i + 3] * alphaScale);
    }
  }

  workCtx.putImageData(frame, 0, 0);
  ctx.clearRect(0, 0, displayWidth, displayHeight);
  ctx.drawImage(workCanvas, 0, 0, displayWidth, displayHeight);
  inst.raf = requestAnimationFrame((t) => drawVideoInstrumentFrame(id, t));
}

function ensureVideoRenderLoop(id) {
  const inst = videoInstruments[id];
  if (!inst || !inst.video) return;
  if (inst.raf) return;
  inst.raf = requestAnimationFrame((t) => drawVideoInstrumentFrame(id, t));
}

function syncVideoPlayback(id) {
  const inst = videoInstruments[id];
  if (!inst || !inst.video) return;
  const { video } = inst;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  const playPromise = video.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {});
  }
  ensureVideoRenderLoop(id);
}

function setGestureObjectVariant(isGestureActive, instrumentId = "drum") {
  const inst = videoInstruments[instrumentId];
  if (!inst || !inst.video) return;
  const { video, canvas } = inst;

  const nextSrc = isGestureActive
    ? video.dataset.variantGesture
    : video.dataset.variantBase;

  const isGlobalActive = isGestureActive || Object.values(videoInstruments).some(v => v.video?.dataset.active === "true"); // simplistic check
  // For now, let's just toggle the specific one
  canvas?.classList.toggle("is-gesture-variant", isGestureActive);
  
  if (!nextSrc) return;

  const currentSrc = video.getAttribute("src") || "";
  if (currentSrc !== nextSrc) {
    video.setAttribute("src", nextSrc);
    video.load();
  }

  syncVideoPlayback(instrumentId);
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function stopCameraTracks(stream) {
  if (!stream || typeof stream.getTracks !== "function") return;
  stream.getTracks().forEach((track) => {
    try {
      track.stop();
    } catch {}
  });
}

function clearCameraSource() {
  const { video: activeVideo } = syncRuntimeDomRefs();
  if (activeVideo?.srcObject && typeof activeVideo.srcObject.getTracks === "function") {
    stopCameraTracks(activeVideo.srcObject);
  }
  if (cameraStream && cameraStream !== activeVideo?.srcObject) {
    stopCameraTracks(cameraStream);
  }

  cameraStream = null;
  try {
    activeVideo?.pause();
  } catch {}
  if (activeVideo) {
    activeVideo.srcObject = null;
  }
  if (testModeWebcamPreview) {
    try {
      testModeWebcamPreview.pause();
    } catch {}
    testModeWebcamPreview.srcObject = null;
  }
}

async function initCamera() {
  try {
    const { video: activeVideo } = syncRuntimeDomRefs();
    if (!activeVideo) {
      statusText.textContent = "카메라 화면을 준비하는 중입니다...";
      return;
    }
    console.info("[MediaPipe] initCamera:start");
    statusText.textContent = "카메라를 준비하는 중입니다...";

    clearCameraSource();

    const cameraAttempts = [
      {
        label: "detailed",
        constraints: {
          width: { ideal: 640 },
          height: { ideal: 360 },
          frameRate: { ideal: 30, max: 30 }
        }
      },
      {
        label: "compat",
        constraints: {
          width: { ideal: 640 },
          height: { ideal: 360 },
          frameRate: { ideal: 24, max: 30 }
        }
      },
      {
        label: "basic",
        constraints: true
      }
    ];

    let stream = null;
    let lastError = null;

    for (let index = 0; index < cameraAttempts.length; index += 1) {
      const attempt = cameraAttempts[index];
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: attempt.constraints });
        console.info("[MediaPipe] initCamera:getUserMedia " + attempt.label + " success");
        break;
      } catch (attemptError) {
        lastError = attemptError;
        const errorName = String(attemptError?.name || "Error");
        console.warn("[MediaPipe] initCamera:getUserMedia " + attempt.label + " failed", {
          name: errorName,
          message: String(attemptError?.message || "")
        });

        if (errorName === "NotAllowedError" || errorName === "SecurityError" || errorName === "NotFoundError") {
          break;
        }

        if (errorName === "NotReadableError") {
          clearCameraSource();
          await wait(200 * (index + 1));
        }
      }
    }

    if (!stream) {
      throw lastError || new Error("Could not start video source");
    }

    activeVideo.srcObject = stream;
    cameraStream = stream;
    activeVideo.playsInline = true;
    activeVideo.muted = true;
    activeVideo.setAttribute("playsinline", "");
    if (testModeWebcamPreview) {
      testModeWebcamPreview.srcObject = stream;
      testModeWebcamPreview.playsInline = true;
      testModeWebcamPreview.muted = true;
      testModeWebcamPreview.setAttribute("playsinline", "");
    }
    activeVideo.onloadedmetadata = () => {
      console.info("[MediaPipe] initCamera:loadedmetadata", {
        width: activeVideo.videoWidth,
        height: activeVideo.videoHeight,
        readyState: activeVideo.readyState
      });
      activeVideo.play().catch(() => {});
      if (testModeWebcamPreview) {
        testModeWebcamPreview.play().catch(() => {});
      }
      statusText.textContent = handLandmarker
        ? "준비 완료! 시작 버튼에 손을 올려주세요."
        : "카메라 준비 완료! 손 인식 모델을 불러오는 중입니다.";
      trackingRuntime.predict();
    };
  } catch (error) {
    console.error("[MediaPipe] initCamera:failed", error);
    const errorName = String(error?.name || "Error");
    if (errorName === "NotAllowedError" || errorName === "SecurityError") {
      statusText.textContent = "카메라 권한을 허용해 주세요.";
    } else if (errorName === "NotFoundError") {
      statusText.textContent = "사용 가능한 카메라를 찾지 못했습니다.";
    } else if (errorName === "NotReadableError") {
      statusText.textContent = "카메라를 시작할 수 없습니다. 다른 앱에서 카메라 사용 중인지 확인해 주세요.";
    } else {
      statusText.textContent = "카메라를 시작하지 못했습니다. 잠시 후 다시 시도해 주세요.";
    }
    statusText.style.color = "var(--danger)";
  }
}

async function initMediaPipe() {
  console.info("[MediaPipe] init:start", {
    delegate: parsePreferredDelegate(),
    numHands: HAND_DETECTION_TARGET,
    splitHands: ENABLE_SPLIT_HAND_INFERENCE
  });
  statusText.textContent = "손 인식 모델을 불러오는 중입니다...";
  const vision = await FilesetResolver.forVisionTasks(getConfiguredMediaPipeWasmRoot());
  console.info("[MediaPipe] init:vision tasks loaded");
  const modelAssetPath = getConfiguredHandLandmarkerTaskPath();
  const preferredDelegate = parsePreferredDelegate(); // 먼저 시도할 계산 장치를 정합니다.
  const fallbackDelegate = preferredDelegate === "GPU" ? "CPU" : "GPU"; // 실패했을 때 쓸 대체 장치도 준비합니다.

  try {
    console.info("[MediaPipe] init:createFromOptions primary", {
      delegate: preferredDelegate,
      modelAssetPath,
      runningMode: "VIDEO"
    });
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath, // 사용할 모델 파일 경로입니다.
        delegate: preferredDelegate // 첫 번째 선택 장치로 계산합니다.
      },
      runningMode: "VIDEO", // 실시간 스트림 분석 모드입니다.
      numHands: HAND_DETECTION_TARGET,
      minHandDetectionConfidence: 0.50,
      minHandPresenceConfidence: 0.50,
      minTrackingConfidence: 0.50
    });
    console.info("[MediaPipe] init:createFromOptions primary success", {
      constructor: handLandmarker?.constructor?.name,
      hasDetectForVideo: typeof handLandmarker?.detectForVideo === "function",
      runningMode: "VIDEO"
    });
  } catch (delegateError) {
    console.warn(`${preferredDelegate} delegate failed, fallback to ${fallbackDelegate}.`, delegateError); // 첫 시도 실패 이유를 콘솔에 남깁니다.
    console.info("[MediaPipe] init:createFromOptions fallback", {
      delegate: fallbackDelegate,
      modelAssetPath,
      runningMode: "VIDEO"
    });
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath, // 모델 파일은 그대로 사용합니다.
        delegate: fallbackDelegate // 이번에는 대체 장치로 다시 시도합니다.
      },
      runningMode: "VIDEO", // 모드는 동일합니다.
      numHands: HAND_DETECTION_TARGET,
      minHandDetectionConfidence: 0.50,
      minHandPresenceConfidence: 0.50,
      minTrackingConfidence: 0.50
    });
    console.info("[MediaPipe] init:createFromOptions fallback success", {
      constructor: handLandmarker?.constructor?.name,
      hasDetectForVideo: typeof handLandmarker?.detectForVideo === "function"
    });
    console.info("[MediaPipe] init:createFromOptions fallback success");
  }

  const { video: activeVideo } = syncRuntimeDomRefs();
  if (activeVideo?.srcObject && activeVideo.readyState >= 2) {
    statusText.textContent = "준비 완료! 시작 버튼에 손을 올려주세요.";
  }
  console.info("[MediaPipe] init:ready (VIDEO mode)");
}

async function refreshCameraTarget() {
  const { video: activeVideo } = syncRuntimeDomRefs();
  setCanvasSize();
  if (!activeVideo || adminEditMode) return;

  if (cameraStream) {
    if (activeVideo.srcObject !== cameraStream) {
      activeVideo.srcObject = cameraStream;
    }
    activeVideo.playsInline = true;
    activeVideo.muted = true;
    activeVideo.setAttribute("playsinline", "");
    activeVideo.play().catch(() => {});
    return;
  }

  await initCamera();
}

const controlRuntime = createControlRuntime({
  audioApi: Audio,
  statusText,
  interactionMode: INTERACTION_MODE,
  ambientAudioEnabled: ENABLE_AMBIENT_AUDIO,
  perfLogKey: PERF_LOG_KEY,
  perfLogLimit: PERF_LOG_LIMIT,
  getSessionStarted: () => sessionStarted,
  getAdminEditMode: () => adminEditMode,
  onActivateStart: activateStart
});

const interactionRuntime = createInteractionRuntime({
  landingOverlay,
  landingStartButton,
  statusText,
  handCursor,
  gestureSquirrelEffect,
  audioApi: Audio,
  resolveGesture,
  resetGestureState: (handKey) => resetGestureState(handKey),
  getModelPrediction: (...args) => getModelPrediction?.(...args),
  restartClassAnimation,
  activateStart,
  registerHit,
  spawnBurst,
  setGestureObjectVariant,
  getGestureInstrumentId: (label) => gestureMapping[label] || null,
  getGesturePlayback: (label, instrumentId) => getGestureSoundProfile(label, instrumentId),
  getInstrumentPlayback: (instrumentId) => getMappedSoundProfile(instrumentId),
  playGestureSound: (label, instrumentId, note) => {
    return playGestureMappedSound(label, instrumentId, { note, spawnEffect: false });
  },
  playInstrumentSound: (instrumentId, note) => {
    const element = instrumentElements[instrumentId] || null;
    return playMappedInstrumentSound(instrumentId, element, { note, spawnEffect: false });
  },
  instruments,
  interactionMode: INTERACTION_MODE,
  collisionPadding: COLLISION_PADDING,
  startHoverMs: START_HOVER_MS,
  gestureCooldownMs: GESTURE_TRIGGER_COOLDOWN_MS,
  isAdminEditMode: () => adminEditMode,
  isSessionStarted: () => sessionStarted,
  feverController,
  checkBubbleCollision: (points) => particleSystem.checkBubbleCollision(points)
});

const trackingRuntime = createHandTrackingRuntime({
  video,
  handCanvas,
  handCtx,
  getVideo: () => syncRuntimeDomRefs().video,
  getHandCanvas: () => syncRuntimeDomRefs().handCanvas,
  getHandCtx: () => syncRuntimeDomRefs().handCtx,
  handCursor,
  renderer: Renderer,
  particleSystem,
  feverController,
  interactionRuntime,
  onBeforeFrame: (now, started) => {
    feverController.updateFeverState(now, started);
  },
  onDetectionError: (error) => {
    console.warn("HandTracking loop error:", error);
  },
  getHandLandmarker: () => handLandmarker,
  getSessionStarted: () => sessionStarted,
  inferIntervalMs: INFER_INTERVAL_MS,
  landmarkStaleMs: LANDMARK_STALE_MS
  ,
  splitHandInference: ENABLE_SPLIT_HAND_INFERENCE
});

async function init() {
  // 모델 추론 모듈 동적 로딩 (top-level await 회피)
  const modelInferenceModule = USE_SEQUENCE_MODEL
    ? await import("./model_inference_sequence.js")
    : await import("./model_inference.js");

  getModelPrediction = modelInferenceModule.getModelPrediction;
  getModelInferenceStatus = modelInferenceModule.getModelInferenceStatus;
  setModelPredictionProvider(getModelPrediction);

  // 사용 중인 모델 타입 로깅
  console.info(`[JamJamBeat] 🎵 모델 타입: ${USE_SEQUENCE_MODEL ? "시퀀스 모델 (8프레임 버퍼)" : "기존 단일 프레임 모델"}`);
  if (USE_SEQUENCE_MODEL) {
    console.info("[JamJamBeat] ⏱️ warmup 대기: 첫 8프레임 수집 (~0.5~1.2초)");
  }

  setupSeamlessBackgroundLoop({ crossfadeSec: BG_VIDEO_CROSSFADE_SEC }); // 배경 영상 반복 시스템을 먼저 준비합니다.
  setCanvasSize(); // 현재 화면 크기에 맞게 캔버스를 조정합니다.
  scene.dataset.fever = scene.dataset.fever || "off"; // 피버 상태의 기본값을 off로 맞춥니다.
  window.addEventListener("resize", setCanvasSize); // 창 크기가 바뀌면 캔버스도 다시 맞춥니다.
  window.addEventListener("beforeunload", () => {
    clearCameraSource(); // 페이지를 떠날 때 카메라 점유를 반드시 해제합니다.
  });
  window.addEventListener("jamjam:refresh-camera-target", () => {
    refreshCameraTarget().catch((error) => {
      console.warn("[MediaPipe] refreshCameraTarget failed:", error);
    });
  });

  controlRuntime.bind(); // 버튼과 입력 이벤트를 연결합니다.
  controlRuntime.syncSoundButtonUI(); // 소리 버튼 글자를 현재 상태에 맞춥니다.
  window.addEventListener("jamjam:sound-played", (event) => {
    const detail = event.detail || {};
    const handKey = String(detail.handKey || "").toLowerCase();
    if (!handKey) return;
    lastSoundEventByHand.set(handKey, {
      latencyMs: Number.isFinite(detail.latencyMs) ? detail.latencyMs : null,
      inferenceLatencyMs: Number.isFinite(detail.inferenceLatencyMs) ? detail.inferenceLatencyMs : null,
      at: Number.isFinite(detail.at) ? detail.at : Date.now()
    });
  });
  if (testModeToggleButton) {
    testModeToggleButton.addEventListener("click", () => {
      testModeEnabled = !testModeEnabled;
      syncTestModeUI();
    });
  }
  if (testModePlayBeepButton) {
    testModePlayBeepButton.addEventListener("click", async () => {
      const unlockedOk = await Audio.unlockAudioContext();
      if (!unlockedOk) return;
      Audio.playDebugBeep();
    });
  }
  if (testModeCenterPanButton) {
    testModeCenterPanButton.addEventListener("click", () => {
      Audio.setPanDebugMode("center");
    });
  }
  if (testModeRestorePanButton) {
    testModeRestorePanButton.addEventListener("click", () => {
      Audio.setPanDebugMode("context");
    });
  }
  syncTestModeUI();
  startTestModeLoop();
  VIDEO_INSTRUMENT_IDS.forEach((id) => {
    const el = instrumentElements[id];
    if (!el) return;
    const videoEl = el.querySelector(".instrument-video-source");
    const canvasEl = el.querySelector(".instrument-art-canvas");
    if (!videoEl || !canvasEl) return;

    videoInstruments[id] = {
      video: videoEl,
      canvas: canvasEl,
      ctx: canvasEl.getContext("2d", { willReadFrequently: true }),
      workCanvas: document.createElement("canvas"),
      workCtx: null,
      lastFrameAt: 0,
      raf: 0
    };
    videoInstruments[id].workCtx = videoInstruments[id].workCanvas.getContext("2d", { willReadFrequently: true });
    
    syncVideoPlayback(id);
  });

  const params = new URLSearchParams(window.location.search); // URL에 적힌 옵션을 읽습니다.
  const mode = params.get("mode") || "calm"; // 모드가 없으면 calm을 기본으로 씁니다.
  applySceneMode(scene, mode); // 장면 분위기를 적용합니다.

  const createInstrumentAnimationManager = await loadAnimationManagerFactory();
  animationManager = createInstrumentAnimationManager();

  // 악기별 Lottie 애니메이션 초기화 (같은 DOM 요소에 중복 초기화 방지)
  const animatedElements = new Set();
  instruments.forEach((instrument) => {
    if (instrument.el && !animatedElements.has(instrument.el)) {
      animationManager.initAnimation(instrument.id, instrument.el);
      animatedElements.add(instrument.el);
    }
  });

  const autoStart = params.get("session") === "start" || params.get("start") === "1"; // 자동 시작 옵션을 계산합니다.
  if (autoStart && !adminEditMode) {
    activateStart(); // 자동 시작 조건이면 바로 시작합니다.
  }

  if (adminEditMode) {
    return; // 관리자 모드에서는 카메라 초기화를 하지 않고 끝냅니다.
  }

  try {
    await Promise.all([
      initCamera(),
      initMediaPipe()
    ]); // 카메라와 손 인식 모델을 동시에 준비해서 체감 대기 시간을 줄입니다.
  } catch (error) {
    console.error("Initialization failed:", error); // 실패 이유는 콘솔에 남깁니다.
    statusText.textContent = "초기화 실패: 새로고침 후 다시 시도해 주세요."; // 사용자에게는 쉬운 문구로 알려줍니다.
    statusText.style.color = "var(--danger)"; // 실패 문구를 경고 색상으로 표시합니다.
  }
}

init(); // 파일이 로드되면 전체 앱 준비를 시작합니다.
