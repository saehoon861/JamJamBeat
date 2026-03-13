// [main.js] 이 프로그램의 '두뇌' 역할을 하는 가장 중요한 파일입니다.
// 카메라를 켜고, 내 손이 어디 있는지 찾아내고, 그 손이 악기에 닿았을 때 소리를 내라고 명령하는 모든 과정을 지휘합니다.

import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm"; // 미디어파이프(손 인식 도구)를 가져옵니다.
import * as Audio from "./audio.js"; // 소리 재생 관련 기능을 가져옵니다.
import * as Renderer from "./renderer.js"; // 화면 그리기 관련 기능을 가져옵니다.
import { resolveGesture } from "./gestures.js"; // 손동작 인식 로직을 가져옵니다.
import { getModelPrediction } from "./model_inference.js"; // 모델의 원본 예측 정보를 가져옵니다.
import { DEFAULT_LAYOUT, loadInstrumentLayout, applyInstrumentLayout, setupAdminDragMode } from "./instrument_layout.js";
import { setupSeamlessBackgroundLoop, applySceneMode, createFeverController } from "./scene_runtime.js";
import { createParticleSystem, restartClassAnimation } from "./particle_system.js";
import { DEFAULT_SOUND_MAPPING, loadSoundMapping, getSoundProfileForInstrument } from "./sound_mapping.js";
import { createInteractionRuntime } from "./interaction_runtime.js";
import { createHandTrackingRuntime } from "./hand_tracking_runtime.js";

// 화면에 보이는 각종 버튼, 글자, 캔버스(그림판)들을 컴퓨터가 찾을 수 있게 주소를 연결해두는 곳입니다.
const video = document.getElementById("webcam"); // 웹캠 영상을 보여줄 비디오 태그를 가져옵니다.
const handCanvas = document.getElementById("handCanvas"); // 손의 뼈대를 그릴 투명 도화지를 가져옵니다.
const effectCanvas = document.getElementById("effectCanvas"); // 터치 효과(반짝임)를 그릴 투명 도화지를 가져옵니다.
const handCtx = handCanvas.getContext("2d"); // 손 도화지에 그림을 그리기 위한 도구(붓)를 꺼냅니다.
const effectCtx = effectCanvas.getContext("2d"); // 효과 도화지에 그림을 그리기 위한 도구(붓)를 꺼냅니다.
const statusText = document.getElementById("status"); // 현재 상태(안내 문구)를 보여줄 글자 칸을 가져옵니다.
const scene = document.getElementById("scene"); // 전체 배경이 되는 공간을 가져옵니다.
const handCursor = document.getElementById("handCursor"); // 내 손가락 끝을 따라다닐 동그란 커서를 가져옵니다.
const landingOverlay = document.getElementById("landingOverlay"); // 처음 시작할 때 보이는 덮개 화면을 가져옵니다.
const landingStartButton = document.getElementById("landingStartButton"); // 시작하기 버튼을 가져옵니다.
const soundUnlockButton = document.getElementById("soundUnlockButton"); // 소리 켜기/끄기 버튼을 가져옵니다.
const pulseMessage = document.getElementById("pulseMessage"); // 화면 중앙에 뜨는 안내 메시지를 가져옵니다.
const adminControls = document.getElementById("adminControls"); // 관리자용 설정 버튼들을 가져옵니다.
const adminSaveButton = document.getElementById("adminSaveButton"); // 관리자 배구 저장 버튼을 가져옵니다.
const adminResetButton = document.getElementById("adminResetButton"); // 관리자 배치 초기화 버튼을 가져옵니다.
const gestureSquirrelEffect = document.getElementById("gestureSquirrelEffect"); // 다람쥐 효과 이미지를 가져옵니다.

// 로고 이미지 크기 정보를 표시합니다.
const landingLogo = document.querySelector(".landing-logo");
const logoDimensionsEl = document.getElementById("logoDimensions");
if (landingLogo && logoDimensionsEl) {
  landingLogo.addEventListener("load", () => {
    const w = landingLogo.naturalWidth;
    const h = landingLogo.naturalHeight;
    logoDimensionsEl.textContent = `${w} × ${h}`;
  });
  // 이미지가 이미 로드된 경우
  if (landingLogo.complete && landingLogo.naturalWidth > 0) {
    logoDimensionsEl.textContent = `${landingLogo.naturalWidth} × ${landingLogo.naturalHeight}`;
  }
}

const instrumentElements = { // 각 악기 식물들의 HTML 요소를 하나로 묶어둡니다.
  drum: document.getElementById("instrumentDrum"), // 드럼 식물을 찾습니다.
  xylophone: document.getElementById("instrumentXylophone"), // 실로폰 식물을 찾습니다.
  tambourine: document.getElementById("instrumentTambourine"), // 탬버린 식물을 찾습니다.
  fern: document.getElementById("instrumentFern"), // 고사리 식물을 찾습니다.
  owl: document.getElementById("instrumentOwl") // 사슴 식물을 찾습니다.
};

const COLLISION_PADDING = 12; // 손이 악기에 완전히 닿지 않아도 조금 근처에만 가도 인식되게 하는 '여유 공간'입니다.
const START_HOVER_MS = 220; // 시작 버튼 위에 손을 얼마나 오래 올려두어야 게임이 시작되는지 결정하는 시간(0.52초)입니다.
const FEVER_TRIGGER_WINDOW_MS = 5000; // 피버 타임을 위해 6번의 터치를 모아야 하는 시간 제한(5초)입니다.
const FEVER_TRIGGER_HITS = 30; // 피버 타임을 터뜨리기 위해 필요한 최소 터치 횟수입니다.
const FEVER_DURATION_MS = 6200; // 피버 타임이 한 번 시작되면 얼마나 오랫동안 지속될지 결정하는 시간(6.2초)입니다.
const ENABLE_AMBIENT_AUDIO = false; // 동작 인식 시에만 소리가 나도록 배경 앰비언트는 끕니다.

const DEFAULT_INFER_FPS = 45; // 별도 설정이 없을 때 1초에 몇 번 손 위치를 계산할지 나타냅니다. (기본 45회)
const MIN_INFER_FPS = 12; // 너무 느려지지 않게 최대로 늦출 수 있는 계산 횟수입니다.
const MAX_INFER_FPS = 90; // 컴퓨터 성능이 좋아도 최대 90회까지만 계산하도록 제한합니다.
const LANDMARK_STALE_MS = 260; // 손이 화면에서 사라졌을 때 약 0.26초 동안은 마지막 위치를 기억하고 보여줍니다.

let handLandmarker; // 미디어파이프 손 인식 도구 객체를 담아둘 변수입니다.
let sessionStarted = false; // 게임이 실제로 시작되었는지 여부를 나타냅니다.
let cameraStream = null; // 웹캠에서 나오는 영상 신호를 담아둡니다.
let adminEditMode = false; // 현재 악기 배치를 수정하는 관리자 모드인지 확인합니다.
let perfLogBound = false;

const GESTURE_TRIGGER_COOLDOWN_MS = 280; // 손동작 인식이 너무 자주 일어나지 않게 하는 대기 시간(0.28초)입니다.
const BG_VIDEO_CROSSFADE_SEC = 0.42; // 배경 영상이 바뀔 때 자연스럽게 겹치는 시간(0.42초)입니다.
const INSTRUMENT_LAYOUT_KEY = "jamjam.instrumentLayout.v2"; // 악기 배치 정보를 저장할 때 사용할 열쇠 이름입니다.
const PERF_LOG_KEY = "jamjam.perf.logs.v1";
const PERF_LOG_LIMIT = 200;

const SOUND_PROFILES = {
  drum: { soundTag: "베이스 멜로디", burstType: "drum", play: () => Audio.playKids_Drum() },
  xylophone: { soundTag: "메인 멜로디", burstType: "xylophone", play: () => Audio.playKids_Xylophone() },
  tambourine: { soundTag: "리듬 멜로디", burstType: "tambourine", play: () => Audio.playKids_Tambourine() },
  pinky: { soundTag: "반짝 하모니", burstType: "pinky", play: () => Audio.playKids_Triangle() },
  heart: { soundTag: "휘파람 멜로디", burstType: "xylophone", play: () => Audio.playKids_Whistle() },
  animal: { soundTag: "애니멀 포인트", burstType: "pinky", play: () => Audio.playKids_AnimalSurprise() },
  fist: { soundTag: "타격", burstType: "fist", play: () => Audio.playFistBeat() }
};

// 주소창(URL)에 적힌 설정값을 보고, 손 위치를 얼마나 자주 계산할지(FPS) 결정하는 기능입니다.
function parseInferFps() {
  const params = new URLSearchParams(window.location.search); // 주소창의 파라미터(?표 뒤의 글자들)를 읽어옵니다.
  const raw = Number(params.get("inferFps")); // 'inferFps'라는 이름의 숫자를 가져옵니다.
  if (!Number.isFinite(raw)) return DEFAULT_INFER_FPS; // 숫자가 아니면 기본값(30)을 사용합니다.
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

// 프로그램이 동작하면서 기억해야 할 '현재 상태' 값들입니다. (예: 지금 카메라가 켜져 있는지, 피버 타임인지 등)
const INTERACTION_MODE = parseInteractionMode();
let soundMapping = loadSoundMapping(SOUND_PROFILES);
const particleSystem = createParticleSystem(effectCtx, effectCanvas);
const feverController = createFeverController({
  scene,
  pulseMessage,
  statusText,
  triggerWindowMs: FEVER_TRIGGER_WINDOW_MS,
  triggerHits: FEVER_TRIGGER_HITS,
  durationMs: FEVER_DURATION_MS,
  getIdleStatusText: () => (
    INTERACTION_MODE === "gesture"
      ? "손동작으로 숲을 연주해 보세요."
      : "손으로 악기 식물을 터치해 보세요."
  )
});

function playMappedInstrumentSound(instrumentId, element) {
  const profile = getSoundProfileForInstrument(soundMapping, DEFAULT_SOUND_MAPPING, SOUND_PROFILES, instrumentId);
  profile.play();
  spawnBurst(profile.burstType, element);
  return profile.soundTag;
}

function readPerfLogs() {
  try {
    const raw = localStorage.getItem(PERF_LOG_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

function appendPerfLog(entry) {
  const logs = readPerfLogs();
  logs.push(entry);
  if (logs.length > PERF_LOG_LIMIT) {
    logs.splice(0, logs.length - PERF_LOG_LIMIT);
  }
  try {
    localStorage.setItem(PERF_LOG_KEY, JSON.stringify(logs));
  } catch {
    // 성능 로그 저장 실패가 메인 동작(오디오/인식)을 막지 않도록 무시합니다.
  }
}

// 우리가 연주할 수 있는 '악기 식물'들의 정보입니다. 이름과 소리, 그리고 닿았을 때 어떤 행동을 할지 적혀 있습니다.
const instruments = [
  {
    id: "drum", // 악기의 고유 ID입니다.
    name: "쿵 (북/드럼)", // 화면에 보일 실제 이름입니다.
    soundTag: "쿵", // 연주했을 때 표시될 소리 느낌표입니다.
    el: instrumentElements.drum, // 실제 HTML 이미지를 연결합니다.
    cooldownMs: 320, // 한 번 연주한 뒤 다시 연주하기 위해 기다려야 하는 시간입니다.
    lastHitAt: 0, // 마지막으로 연주된 시간을 기록합니다.
    onHit() { // 연주되었을 때 실행할 행동입니다.
      Audio.playKids_Drum();
      spawnBurst("drum", this.el);
      return this.soundTag;
    }
  },
  {
    id: "xylophone", // 실로폰 악기입니다.
    name: "도레미 (실로폰)", // 이름입니다.
    soundTag: "도레미", // 글자 표시입니다.
    el: instrumentElements.xylophone, // 이미지를 찾습니다.
    cooldownMs: 360, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit() { // 누르면 실행합니다.
      Audio.playKids_Xylophone();
      spawnBurst("xylophone", this.el);
      return this.soundTag;
    }
  },
  {
    id: "tambourine", // 탬버린 악기입니다.
    name: "챵 (탬버린)", // 이름입니다.
    soundTag: "챵", // 글자 표시입니다.
    el: instrumentElements.tambourine, // 이미지를 찾습니다.
    cooldownMs: 380, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit() { // 누르면 실행합니다.
      Audio.playKids_Tambourine();
      spawnBurst("tambourine", this.el);
      return this.soundTag;
    }
  },
  {
    id: "fern", // 고사리 악기입니다.
    name: "반짝 (트라이앵글)", // 이름입니다.
    soundTag: "반짝", // 글자 표시입니다.
    el: instrumentElements.fern, // 이미지를 찾습니다.
    cooldownMs: 380, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit() { // 누르면 실행합니다.
      Audio.playKids_Triangle();
      spawnBurst("pinky", this.el);
      return this.soundTag;
    }
  },
  {
    id: "owl", // 다람쥐 악기입니다.
    name: "휘파람 (리코더)", // 이름입니다.
    soundTag: "휘파람", // 글자 표시입니다.
    el: instrumentElements.owl, // 이미지를 찾습니다.
    cooldownMs: 380, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit() { // 누르면 실행합니다.
      Audio.playKids_Whistle();
      spawnBurst("heart", this.el);
      return this.soundTag;
    }
  }
];

// 브라우저 화면 크기가 바뀔 때마다 그림판(캔버스)의 크기도 똑같이 맞춰주는 기능입니다.
function setCanvasSize() {
  handCanvas.width = window.innerWidth; // 손 도화지 가로 길이를 창 크기에 맞춥니다.
  handCanvas.height = window.innerHeight; // 손 도화지 세로 길이를 창 크기에 맞춥니다.
  effectCanvas.width = window.innerWidth; // 효과 도화지 가로 길이를 창 크기에 맞춥니다.
  effectCanvas.height = window.innerHeight; // 효과 도화지 세로 길이를 창 크기에 맞춥니다.
}

// 숫자가 너무 작거나 너무 크지 않게 특정 범위 안에만 있도록 잡아주는 기능입니다.
function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// 화면 오른쪽 아래의 '소리 켜기/끄기' 버튼의 글자를 현재 상태에 맞춰 바꿔주는 기능입니다.
function updateSoundButtonUI() {
  const state = Audio.getAudioState(); // 현재 소리 설정 상태를 가져옵니다.
  soundUnlockButton.textContent = state.running ? "소리 끄기" : "소리 켜기"; // 소리가 켜져 있으면 '끄기', 꺼져 있으면 '켜기'로 글자를 바꿉니다.
}

function activateStart() { // 게임을 실제로 시작하는 기능입니다.
  sessionStarted = true; // 세션이 시작되었음을 표시합니다.
  landingOverlay.classList.add("is-hidden"); // 시작 화면 덮개를 숨깁니다.
  const audioState = Audio.getAudioState(); // 현재 오디오 상태를 가져옵니다.
  const playGuide = INTERACTION_MODE === "gesture" // 플레이 방식에 따라 안내 문구를 정합니다.
    ? "손동작으로 숲을 연주해 보세요."
    : "손으로 악기 식물을 터치해 보세요.";
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

async function initCamera() {
  try {
    let stream; // 카메라에서 받아올 영상 스트림을 담는 변수입니다.
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 960 }, // 가능하면 가로 960 정도 화질을 요청합니다.
          height: { ideal: 540 }, // 가능하면 세로 540 정도 화질을 요청합니다.
          frameRate: { ideal: 60, max: 60 } // 초당 60프레임 정도를 목표로 합니다.
        }
      });
    } catch {
      stream = await navigator.mediaDevices.getUserMedia({ video: true }); // 상세 조건이 안 맞으면 기본 카메라 요청으로 다시 시도합니다.
    }

    video.srcObject = stream; // 받은 카메라 영상을 비디오 태그에 연결합니다.
    cameraStream = stream; // 나중에 종료할 수 있도록 별도로 저장합니다.
    video.playsInline = true; // 모바일에서 전체화면으로 튀지 않게 합니다.
    video.muted = true; // 카메라 영상 자체 소리는 끕니다.
    video.setAttribute("playsinline", ""); // 일부 브라우저용 호환 속성도 함께 넣습니다.
    video.onloadedmetadata = () => {
      video.play().catch(() => { }); // 영상 정보가 준비되면 재생을 시작합니다.
      statusText.textContent = "준비 완료! 시작 버튼에 손을 올려주세요."; // 사용자에게 준비 완료를 안내합니다.
      trackingRuntime.predict(); // 손 추적 루프를 시작합니다.
    };
  } catch {
    statusText.textContent = "카메라 권한을 허용해 주세요."; // 카메라 권한이 없으면 메시지를 보여줍니다.
    statusText.style.color = "var(--danger)"; // 경고 색으로 강조합니다.
  }
}

async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"); // MediaPipe 엔진이 필요한 파일들을 불러옵니다.
  const modelAssetPath = "/hand_landmarker.task"; // 손 인식 모델 파일의 실제 주소를 만듭니다.
  const preferredDelegate = parsePreferredDelegate(); // 먼저 시도할 계산 장치를 정합니다.
  const fallbackDelegate = preferredDelegate === "GPU" ? "CPU" : "GPU"; // 실패했을 때 쓸 대체 장치도 준비합니다.

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath, // 사용할 모델 파일 경로입니다.
        delegate: preferredDelegate // 첫 번째 선택 장치로 계산합니다.
      },
      runningMode: "VIDEO", // 실시간 영상 분석 모드입니다.
      numHands: 2 // 손은 최대 두 개까지 추적합니다.
    });
  } catch (delegateError) {
    console.warn(`${preferredDelegate} delegate failed, fallback to ${fallbackDelegate}.`, delegateError); // 첫 시도 실패 이유를 콘솔에 남깁니다.
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath, // 모델 파일은 그대로 사용합니다.
        delegate: fallbackDelegate // 이번에는 대체 장치로 다시 시도합니다.
      },
      runningMode: "VIDEO", // 모드는 동일합니다.
      numHands: 2 // 손 개수도 동일합니다.
    });
  }
}

function bindUIEvents() {
  if (!perfLogBound) {
    window.addEventListener("jamjam:sound-played", (event) => {
      const detail = event.detail || {};
      appendPerfLog({
        at: Date.now(),
        soundKey: detail.soundKey || null,
        playMode: detail.playMode || null,
        instrumentId: detail.instrumentId || null,
        gestureLabel: detail.gestureLabel || null,
        gestureSource: detail.gestureSource || null,
        latencyMs: Number.isFinite(detail.latencyMs) ? detail.latencyMs : null
      });
    });
    perfLogBound = true;
  }

  const measureAudioUnlock = async (reason) => {
    const startedAt = performance.now();
    const unlocked = await Audio.unlockAudioContext();
    const elapsed = performance.now() - startedAt;
    try {
      appendPerfLog({
        at: Date.now(),
        soundKey: unlocked ? "audio_unlock" : "audio_unlock_fail",
        playMode: "control",
        instrumentId: null,
        gestureLabel: "Unlock",
        gestureSource: reason,
        latencyMs: elapsed
      });
    } catch {
      // 로그 저장 실패는 무시하고 오디오 언락 결과만 반환합니다.
    }
    return unlocked;
  };

  const tryUnlockFromGesture = async () => {
    const unlocked = await measureAudioUnlock("first-input"); // 브라우저가 막아둔 오디오 권한을 풀어보는 함수입니다.
    if (!unlocked) return; // 실패하면 여기서 멈춥니다.
    if (sessionStarted && ENABLE_AMBIENT_AUDIO) Audio.startAmbientLoop(); // 게임 중이고 배경음이 허용되면 배경음을 켭니다.
    else Audio.stopAmbientLoop(); // 아니면 배경음을 꺼둡니다.
    updateSoundButtonUI(); // 버튼 문구를 현재 상태로 갱신합니다.
  };

  const onFirstGesture = async () => {
    await tryUnlockFromGesture();
    if (Audio.getAudioState().running) {
      window.removeEventListener("pointerdown", onFirstGesture);
      window.removeEventListener("keydown", onFirstGesture);
    }
  };

  window.addEventListener("pointerdown", onFirstGesture);
  window.addEventListener("keydown", onFirstGesture);

  landingStartButton.onclick = async () => {
    if (adminEditMode) return; // 관리자 모드에서는 게임 시작을 막습니다.
    const unlocked = await measureAudioUnlock("start-button"); // 시작 버튼에서도 오디오 권한을 확보해 봅니다.
    if (unlocked) updateSoundButtonUI(); // 성공했으면 버튼 문구를 갱신합니다.
    activateStart(); // 실제 게임을 시작합니다.
  };

  soundUnlockButton.onclick = async () => {
    const state = Audio.getAudioState(); // 현재 오디오 상태를 확인합니다.
    if (state.running) {
      Audio.toggleSound(); // 켜져 있으면 꺼줍니다.
      Audio.stopAmbientLoop(); // 배경음도 같이 멈춥니다.
    } else {
      const unlocked = await measureAudioUnlock("sound-button"); // 꺼져 있으면 재생 권한부터 확보합니다.
      if (!unlocked) return; // 실패하면 종료합니다.
      if (sessionStarted && ENABLE_AMBIENT_AUDIO) Audio.startAmbientLoop(); // 게임 중이고 허용되면 배경음을 켭니다.
      else Audio.stopAmbientLoop(); // 아니면 배경음을 꺼둡니다.
      statusText.textContent = INTERACTION_MODE === "gesture"
        ? "소리가 켜졌어요! 손동작으로 숲을 연주해 보세요." // 제스처 모드 안내 문구입니다.
        : "소리가 켜졌어요! 손으로 숲을 연주해 보세요."; // 터치 모드 안내 문구입니다.
    }
    updateSoundButtonUI(); // 버튼 문구를 다시 맞춥니다.
  };
}

const interactionRuntime = createInteractionRuntime({
  landingOverlay,
  landingStartButton,
  statusText,
  handCursor,
  gestureSquirrelEffect,
  audioApi: Audio,
  resolveGesture,
  getModelPrediction,
  restartClassAnimation,
  activateStart,
  registerHit,
  spawnBurst,
  instruments,
  interactionMode: INTERACTION_MODE,
  collisionPadding: COLLISION_PADDING,
  startHoverMs: START_HOVER_MS,
  gestureCooldownMs: GESTURE_TRIGGER_COOLDOWN_MS,
  isAdminEditMode: () => adminEditMode,
  isSessionStarted: () => sessionStarted
});

const trackingRuntime = createHandTrackingRuntime({
  video,
  handCanvas,
  handCtx,
  handCursor,
  renderer: Renderer,
  particleSystem,
  feverController,
  interactionRuntime,
  onBeforeFrame: (now, started) => {
    feverController.updateFeverState(now, started);
  },
  onDetectionError: (error) => {
    console.warn("MediaPipe detection error:", error);
  },
  getHandLandmarker: () => handLandmarker,
  getSessionStarted: () => sessionStarted,
  inferIntervalMs: INFER_INTERVAL_MS,
  landmarkStaleMs: LANDMARK_STALE_MS
});

async function init() {
  setupSeamlessBackgroundLoop({ crossfadeSec: BG_VIDEO_CROSSFADE_SEC }); // 배경 영상 반복 시스템을 먼저 준비합니다.
  setCanvasSize(); // 현재 화면 크기에 맞게 캔버스를 조정합니다.
  scene.dataset.fever = scene.dataset.fever || "off"; // 피버 상태의 기본값을 off로 맞춥니다.
  window.addEventListener("resize", setCanvasSize); // 창 크기가 바뀌면 캔버스도 다시 맞춥니다.
  window.addEventListener("beforeunload", () => {
    if (!cameraStream) return; // 카메라가 없으면 정리할 것이 없습니다.
    cameraStream.getTracks().forEach((track) => {
      track.stop(); // 페이지를 떠날 때 카메라를 완전히 끕니다.
    });
  });

  bindUIEvents(); // 버튼과 입력 이벤트를 연결합니다.
  updateSoundButtonUI(); // 소리 버튼 글자를 현재 상태에 맞춥니다.

  const params = new URLSearchParams(window.location.search); // URL에 적힌 옵션을 읽습니다.
  const mode = params.get("mode") || "calm"; // 모드가 없으면 calm을 기본으로 씁니다.
  applySceneMode(scene, mode); // 장면 분위기를 적용합니다.

  if (params.get("admin") === "1") {
    setupAdminDragMode({
      scene,
      landingOverlay,
      statusText,
      adminControls,
      adminSaveButton,
      adminResetButton,
      instrumentElements,
      storageKey: INSTRUMENT_LAYOUT_KEY,
      defaultLayout: DEFAULT_LAYOUT,
      clamp,
      onAdminModeEnabled: () => {
        adminEditMode = true;
      }
    }); // 관리자 옵션이 있으면 배치 편집 모드로 들어갑니다.
  }

  const savedLayout = loadInstrumentLayout(INSTRUMENT_LAYOUT_KEY); // 저장된 악기 배치가 있는지 확인합니다.
  applyInstrumentLayout(savedLayout || DEFAULT_LAYOUT, instrumentElements, clamp); // 있으면 불러오고 없으면 기본 배치를 씁니다.

  const autoStart = params.get("session") === "start" || params.get("start") === "1"; // 자동 시작 옵션을 계산합니다.
  if (autoStart && !adminEditMode) {
    activateStart(); // 자동 시작 조건이면 바로 시작합니다.
  }

  if (adminEditMode) {
    return; // 관리자 모드에서는 카메라 초기화를 하지 않고 끝냅니다.
  }

  try {
    await initMediaPipe(); // 손 인식 엔진을 먼저 준비합니다.
    await initCamera(); // 그 다음 카메라를 켭니다.
  } catch (error) {
    console.error("Initialization failed:", error); // 실패 이유는 콘솔에 남깁니다.
    statusText.textContent = "초기화 실패: 새로고침 후 다시 시도해 주세요."; // 사용자에게는 쉬운 문구로 알려줍니다.
    statusText.style.color = "var(--danger)"; // 실패 문구를 경고 색상으로 표시합니다.
  }
}

init(); // 파일이 로드되면 전체 앱 준비를 시작합니다.
