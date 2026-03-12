// [main.js] 이 프로그램의 '두뇌' 역할을 하는 가장 중요한 파일입니다.
// 카메라를 켜고, 내 손이 어디 있는지 찾아내고, 그 손이 악기에 닿았을 때 소리를 내라고 명령하는 모든 과정을 지휘합니다.

import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm"; // 미디어파이프(손 인식 도구)를 가져옵니다.
import * as Audio from "./audio.js"; // 소리 재생 관련 기능을 가져옵니다.
import * as Renderer from "./renderer.js"; // 화면 그리기 관련 기능을 가져옵니다.
import { resolveGesture } from "./gestures.js"; // 손동작 인식 로직을 가져옵니다.
import { getModelPrediction } from "./model_inference.js"; // 모델의 원본 예측 정보를 가져옵니다.

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

const DEFAULT_INFER_FPS = 30; // 별도 설정이 없을 때 1초에 몇 번 손 위치를 계산할지 나타냅니다. (기본 30회)
const MIN_INFER_FPS = 8; // 너무 느려지지 않게 최대로 늦출 수 있는 계산 횟수입니다.
const MAX_INFER_FPS = 60; // 컴퓨터 성능이 좋아도 최대 60회까지만 계산하도록 제한합니다.
const LANDMARK_STALE_MS = 260; // 손이 화면에서 사라졌을 때 약 0.26초 동안은 마지막 위치를 기억하고 보여줍니다.

let handLandmarker; // 미디어파이프 손 인식 도구 객체를 담아둘 변수입니다.
let lastVideoTime = -1; // 이전 프레임의 시간을 기억해서 중복 계산을 방지합니다.
let sessionStarted = false; // 게임이 실제로 시작되었는지 여부를 나타냅니다.
let hoverStartedAt = 0; // 버튼 위에 손을 올리기 시작한 시각을 기록합니다.
let hoverActive = false; // 현재 버튼 위에 손이 올라가 있는지 여부입니다.
let feverUntil = 0; // 피버 타임이 언제 종료될지 기록하는 시간값입니다.
let hitStamps = []; // 짧은 시간 동안 몇 번 터치했는지 기록하는 목록입니다.
let cameraStream = null; // 웹캠에서 나오는 영상 신호를 담아둡니다.
let lastGestureHitAt = 0; // 마지막으로 손동작이 인식된 시간을 기록합니다.
let lastGestureLabel = "None"; // 마지막으로 인식된 손동작의 이름을 저장합니다.
let adminEditMode = false; // 현재 악기 배치를 수정하는 관리자 모드인지 확인합니다.
let dragState = null; // 관리자 모드에서 어떤 악기를 드래그하고 있는지 소중한 정보를 담습니다.
let lastInferenceAt = 0; // 마지막으로 AI 추론(계산)을 수행한 시간입니다.
let cachedLandmarks = null; // 가장 최근에 계산된 손가락 위치들을 임시로 저장합니다.
let cachedLandmarksAt = 0; // 임시 저장된 손가락 위치가 언제 기록된 것인지 저장합니다.
let perfLogBound = false;

const GESTURE_TRIGGER_COOLDOWN_MS = 520; // 손동작 인식이 너무 자주 일어나지 않게 하는 대기 시간(0.52초)입니다.
const BG_VIDEO_CROSSFADE_SEC = 0.42; // 배경 영상이 바뀔 때 자연스럽게 겹치는 시간(0.42초)입니다.
const INSTRUMENT_LAYOUT_KEY = "jamjam.instrumentLayout.v2"; // 악기 배치 정보를 저장할 때 사용할 열쇠 이름입니다.
const SOUND_MAPPING_KEY = "jamjam.soundMapping.v1"; // 오브젝트별 사운드 매핑을 저장할 때 사용할 열쇠 이름입니다.
const PERF_LOG_KEY = "jamjam.perf.logs.v1";
const PERF_LOG_LIMIT = 200;
const DEFAULT_SOUND_MAPPING = {
  drum: "drum",
  xylophone: "xylophone",
  tambourine: "tambourine",
  fern: "pinky",
  owl: "heart"
};
const DEFAULT_LAYOUT = { // 아무런 저장이 없을 때 사용할 기본 악기 배치 값들입니다.
  drum: { x: 10, y: 14 },
  xylophone: { x: 38, y: 22 },
  tambourine: { x: 68, y: 18 },
  fern: { x: 52, y: 10 },
  owl: { x: 24, y: 7 }
};

const particles = []; // 화면에 날아다니는 반짝이 조각들을 모두 담아두는 바구니입니다.

const SOUND_PROFILES = {
  drum: { soundTag: "드럼", burstType: "drum", play: () => Audio.playDrumMushroom() },
  xylophone: { soundTag: "피리", burstType: "xylophone", play: () => Audio.playXylophoneVine() },
  tambourine: { soundTag: "피아노", burstType: "tambourine", play: () => Audio.playTambourineFlower() },
  pinky: { soundTag: "심벌즈", burstType: "pinky", play: () => Audio.playPinkyChime() },
  heart: { soundTag: "고양이/랜덤", burstType: "xylophone", play: () => Audio.playHeartBloom() },
  animal: { soundTag: "애니멀", burstType: "pinky", play: () => Audio.playAnimalRoll() },
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

function loadSoundMapping() {
  try {
    const raw = localStorage.getItem(SOUND_MAPPING_KEY);
    if (!raw) return { ...DEFAULT_SOUND_MAPPING };
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return { ...DEFAULT_SOUND_MAPPING };

    const merged = { ...DEFAULT_SOUND_MAPPING };
    Object.keys(merged).forEach((instrumentId) => {
      const candidate = String(parsed?.[instrumentId] || "");
      if (candidate in SOUND_PROFILES) {
        merged[instrumentId] = candidate;
      }
    });
    return merged;
  } catch {
    return { ...DEFAULT_SOUND_MAPPING };
  }
}

let soundMapping = loadSoundMapping();

function getSoundProfileForInstrument(instrumentId) {
  const profileKey = soundMapping?.[instrumentId] || DEFAULT_SOUND_MAPPING[instrumentId] || "drum";
  return SOUND_PROFILES[profileKey] || SOUND_PROFILES.drum;
}

function playMappedInstrumentSound(instrumentId, element) {
  const profile = getSoundProfileForInstrument(instrumentId);
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
    name: "고슴도치 드럼", // 화면에 보일 실제 이름입니다.
    soundTag: "쿵", // 연주했을 때 표시될 소리 느낌표입니다.
    el: instrumentElements.drum, // 실제 HTML 이미지를 연결합니다.
    cooldownMs: 320, // 한 번 연주한 뒤 다시 연주하기 위해 기다려야 하는 시간입니다.
    lastHitAt: 0, // 마지막으로 연주된 시간을 기록합니다.
    onHit() { // 연주되었을 때 실행할 행동입니다.
      return playMappedInstrumentSound(this.id, this.el);
    }
  },
  {
    id: "xylophone", // 실로폰 악기입니다.
    name: "노래하는 백합", // 이름입니다.
    soundTag: "도레미", // 글자 표시입니다.
    el: instrumentElements.xylophone, // 이미지를 찾습니다.
    cooldownMs: 360, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit() { // 누르면 실행합니다.
      return playMappedInstrumentSound(this.id, this.el);
    }
  },
  {
    id: "tambourine", // 탬버린 악기입니다.
    name: "행운 클로버", // 이름입니다.
    soundTag: "챵", // 글자 표시입니다.
    el: instrumentElements.tambourine, // 이미지를 찾습니다.
    cooldownMs: 380, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit() { // 누르면 실행합니다.
      return playMappedInstrumentSound(this.id, this.el);
    }
  },
  {
    id: "fern", // 고사리 악기입니다.
    name: "마법 고사리", // 이름입니다.
    soundTag: "링", // 글자 표시입니다.
    el: instrumentElements.fern, // 이미지를 찾습니다.
    cooldownMs: 380, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit() { // 누르면 실행합니다.
      return playMappedInstrumentSound(this.id, this.el);
    }
  },
  {
    id: "owl", // 다람쥐 악기입니다.
    name: "다람쥐", // 이름입니다.
    soundTag: "삐요", // 글자 표시입니다.
    el: instrumentElements.owl, // 이미지를 찾습니다.
    cooldownMs: 380, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit() { // 누르면 실행합니다.
      return playMappedInstrumentSound(this.id, this.el);
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

// 컴퓨터에 저장해둔 악기 배치 정보를 불러오는 기능입니다.
function loadInstrumentLayout() {
  try {
    const raw = localStorage.getItem(INSTRUMENT_LAYOUT_KEY); // 저장된 글자를 가져옵니다.
    if (!raw) return null; // 만약 저장된 게 없으면 비어있다고 알려줍니다.
    const parsed = JSON.parse(raw); // 글자(JSON)를 데이터 객체로 변환합니다.
    if (!parsed || typeof parsed !== "object") return null; // 데이터 형식이 이상하면 비어있다고 합니다.
    return parsed; // 성공적으로 읽어온 데이터를 반환합니다.
  } catch {
    return null; // 오류가 나면 비어있다고 합니다.
  }
}

// 현재 악기 배치 정보를 컴퓨터에 안전하게 저장하는 기능입니다.
function saveInstrumentLayout(layout) {
  try {
    localStorage.setItem(INSTRUMENT_LAYOUT_KEY, JSON.stringify(layout)); // 데이터를 글자(JSON)로 바꿔서 브라우저에 저장합니다.
  } catch {
    // ignore storage errors // 저장 중 오류가 나면 그냥 무시합니다.
  }
}

// 불러온 배치 정보에 맞춰 화면상의 악기들을 실제 위치로 옮겨주는 기능입니다.
function applyInstrumentLayout(layout) {
  if (!layout || typeof layout !== "object") return; // 데이터가 올바르지 않으면 아무것도 하지 않습니다.

  const ids = ["drum", "xylophone", "tambourine", "fern", "owl"]; // 처리할 악기들의 이름 상자입니다.
  ids.forEach((id) => { // 각 악기 이름을 하나씩 꺼내서 반복합니다.
    const target = instrumentElements[id]; // 화면상의 악기 이미지를 찾습니다.
    const pos = layout[id]; // 저장된 위치 정보를 가져옵니다.
    if (!target || !pos || typeof pos !== "object") return; // 이미지나 위치 정보가 없으면 넘어갑니다.

    const x = Number(pos.x); // 저장된 가로 위치(x)를 숫자로 바꿉니다.
    const y = Number(pos.y); // 저장된 세로 위치(y)를 숫자로 바꿉니다.
    if (!Number.isFinite(x) || !Number.isFinite(y)) return; // 숫자가 아니면(잘못된 값이면) 넘어갑니다.

    target.style.left = `${clamp(x, 0, 90)}vw`; // 왼쪽에서 얼마나 떨어질지 설정합니다. (화면 너비의 0~90% 사이)
    target.style.bottom = `${clamp(y, 0, 80)}vh`; // 아래에서 얼마나 올라올지 설정합니다. (화면 높이의 0~80% 사이)
    target.style.right = "auto"; // 오른쪽 정렬은 사용하지 않도록 초기화합니다.
  });
}

function readCurrentInstrumentLayout() { // 현재 화면에 보이는 악기들의 위치를 숫자로 읽어오는 기능입니다.
  const layout = {}; // 읽어온 정보를 담을 빈 상자를 만듭니다.
  const width = Math.max(1, window.innerWidth); // 현재 화면의 가로 길이를 가져옵니다. (최소 1)
  const height = Math.max(1, window.innerHeight); // 현재 화면의 세로 길이를 가져옵니다. (최소 1)

  ["drum", "xylophone", "tambourine", "fern", "owl"].forEach((id) => { // 각 악기를 하나씩 살펴봅니다.
    const el = instrumentElements[id]; // 악기의 이미지를 가져옵니다.
    if (!el) return; // 이미지가 없으면 넘어갑니다.
    const rect = el.getBoundingClientRect(); // 이미지의 실제 사각형 영역 정보를 가져옵니다.
    layout[id] = { // 해당 악기의 위치를 비율(%)로 계산해서 저장합니다.
      x: clamp((rect.left / width) * 100, 0, 90), // 가로 위치 비율입니다.
      y: clamp(((height - rect.bottom) / height) * 100, 0, 80) // 세로 위치 비율입니다.
    };
  });

  return layout; // 정돈된 정보를 돌려줍니다.
}

// 관리자 모드에서 악기를 마우스나 손으로 직접 옮길 수 있게 해주는 기능입니다.
function setupAdminDragMode() {
  adminEditMode = true; // 관리자 수정 모드를 켭니다.
  scene.dataset.admin = "on"; // 화면의 설정을 '관리자 모드'로 바꿉니다.
  landingOverlay.classList.add("is-hidden"); // 시작 화면 조각을 숨깁니다.
  statusText.textContent = "관리자 모드: 오브젝트를 드래그하고 저장하세요."; // 안내 문구를 바꿉니다.

  if (adminControls) {
    adminControls.classList.add("is-visible"); // 관리자용 버튼들을 화면에 보여줍니다.
  }

  const onPointerMove = (event) => { // 마우스를 움직일 때 실행되는 기능입니다.
    if (!dragState) return; // 드래그 중이 아니면 아무것도 안 합니다.

    const width = Math.max(1, window.innerWidth); // 화면 너비를 가져옵니다.
    const height = Math.max(1, window.innerHeight); // 화면 높이를 가져옵니다.
    const rect = dragState.el.getBoundingClientRect(); // 드래그 중인 물체의 크기를 가져옵니다.

    const leftPx = clamp(event.clientX - dragState.offsetX, 0, width - rect.width); // 마우스 위치에 맞춰 왼쪽 좌표를 계산합니다.
    const topPx = clamp(event.clientY - dragState.offsetY, 0, height - rect.height); // 마우스 위치에 맞춰 위쪽 좌표를 계산합니다.
    const x = clamp((leftPx / width) * 100, 0, 90); // 좌표를 화면 비율(%)로 바꿉니다.
    const y = clamp(((height - (topPx + rect.height)) / height) * 100, 0, 80); // 좌표를 화면 비율(%)로 바꿉니다.

    dragState.el.style.left = `${x}vw`; // 계산된 위치를 실제로 적용합니다.
    dragState.el.style.bottom = `${y}vh`; // 계산된 위치를 실제로 적용합니다.
    dragState.el.style.right = "auto"; // 오른쪽 위치 설정을 끕니다.
  };

  const onPointerUp = () => { // 마우스를 뗄 때 실행되는 기능입니다.
    if (!dragState) return; // 드래그 중이 아니면 관둡니다.
    dragState.el.classList.remove("is-dragging"); // '잡고 있음' 표시를 지웁니다.
    dragState = null; // 드래그 정보를 비웁니다.
  };

  ["drum", "xylophone", "tambourine", "fern", "owl"].forEach((id) => { // 각 악기에게 드래그 기능을 심어줍니다.
    const el = instrumentElements[id]; // 악기 요소를 가져옵니다.
    if (!el) return; // 없으면 넘어갑니다.

    el.addEventListener("pointerdown", (event) => { // 악기를 꾹 눌렀을 때 실행합니다.
      event.preventDefault(); // 기본 행동을 막습니다.
      const rect = el.getBoundingClientRect(); // 위치 정보를 가져옵니다.
      dragState = { // 드래그 상태를 기록합니다.
        id,
        el,
        offsetX: event.clientX - rect.left, // 클릭한 지점과 물체 왼쪽 끝 사이의 거리를 잽니다.
        offsetY: event.clientY - rect.top // 클릭한 지점과 물체 위쪽 끝 사이의 거리를 잽니다.
      };
      el.classList.add("is-dragging"); // '드래그 중임' 애니메이션을 켭니다.
      if (el.setPointerCapture) { // 마우스가 밖으로 나가도 계속 추적하게 설정합니다.
        try {
          el.setPointerCapture(event.pointerId);
        } catch {
          // no-op
        }
      }
    });
  });

  window.addEventListener("pointermove", onPointerMove); // 창 전체에서 마우스 움직임을 감시합니다.
  window.addEventListener("pointerup", onPointerUp); // 창 전체에서 마우스 떼는 것을 감시합니다.
  window.addEventListener("pointercancel", onPointerUp); // 클릭이 취소되었을 때도 손을 뗍니다.

  if (adminSaveButton) { // 저장 버튼이 있으면
    adminSaveButton.onclick = () => { // 클릭했을 때 저장합니다.
      const layout = readCurrentInstrumentLayout(); // 현재 위치들을 읽어옵니다.
      saveInstrumentLayout(layout); // 컴퓨터에 저장합니다.
      adminSaveButton.textContent = "저장됨"; // 버튼 글자를 바꿉니다.
      window.setTimeout(() => { // 잠시 후
        adminSaveButton.textContent = "배치 저장"; // 원래 글자로 돌려놓습니다.
      }, 900);
    };
  }

  if (adminResetButton) { // 초기화 버튼이 있으면
    adminResetButton.onclick = () => { // 클릭했을 때 초기화합니다.
      localStorage.removeItem(INSTRUMENT_LAYOUT_KEY); // 저장된 정보를 지웁니다.
      applyInstrumentLayout(DEFAULT_LAYOUT); // 기본 위치로 돌립니다.
    };
  }
}

function setupSeamlessBackgroundLoop() { // 배경 영상이 끊기지 않고 자연스럽게 계속 반복되도록 해주는 기능입니다.
  const videoA = document.querySelector(".bg-video-a"); // 첫 번째 배경 영상 요소를 가져옵니다.
  const videoB = document.querySelector(".bg-video-b"); // 두 번째 배경 영상 요소를 가져옵니다.
  if (!videoA || !videoB) return; // 영상이 하나라도 없으면 종료합니다.

  const videos = [videoA, videoB]; // 두 영상을 목록으로 만듭니다.
  videos.forEach((video, idx) => { // 각 영상에 초기 설정을 해줍니다.
    video.muted = true; // 소리를 끕니다. (브라우저 정책상 필수)
    video.playsInline = true; // 모바일에서도 화면 안에서 재생되게 합니다.
    video.loop = false; // 자체 반복 기능은 끕니다. (우리가 직접 제어하기 위해)
    video.playbackRate = 1; // 재생 속도를 정상 속도로 맞춥니다.
    video.classList.toggle("is-active", idx === 0); // 첫 번째 영상만 활성화 상태로 시작합니다.
    video.classList.toggle("is-preload", idx !== 0); // 나머지 영상은 미리 로딩 상태로 둡니다.
  });

  let active = videoA; // 현재 재생 중인 영상을 가리킵니다.
  let standby = videoB; // 다음에 재생될 대기 영상을 가리킵니다.
  let rafId = 0; // 프레임 반복(AnimationFrame) 번호를 저장합니다.
  let isSwitching = false; // 영상이 교체 중인지 확인하는 깃발입니다.
  let switchSeq = 0; // 교체 순서를 기록하는 숫자입니다.

  const safePlay = (video) => { // 안전하게 영상을 재생하는 보조 기능입니다.
    const p = video.play(); // 재생을 시도합니다.
    if (p && typeof p.catch === "function") p.catch(() => { }); // 오류가 나도 무시합니다.
  };

  const swap = () => { // 두 영상을 부드럽게 교체하는 핵심 기능입니다.
    if (isSwitching) return; // 이미 교체 중이면 중복 실행을 막습니다.
    isSwitching = true; // 교체 시작을 알립니다.

    const fromVideo = active; // 물러날 영상을 기억합니다.
    const toVideo = standby; // 새로 나타날 영상을 기억합니다.
    const currentSeq = ++switchSeq; // 이번 교체 작업의 번호를 매깁니다.

    toVideo.currentTime = 0; // 대기 영상의 시간을 0으로 돌립니다.
    toVideo.classList.remove("is-preload"); // 로딩 중 표시를 지웁니다.
    toVideo.classList.add("is-active"); // 대기 영상을 뚜렷하게 보이게 합니다.
    safePlay(toVideo); // 대기 영상 재생을 시작합니다.

    window.setTimeout(() => { // 0.52초 뒤에 물러날 영상을 정리합니다. (겹치는 시간)
      if (switchSeq !== currentSeq || active !== fromVideo || standby !== toVideo) { // 상황이 바뀌었으면 관둡니다.
        isSwitching = false;
        return;
      }

      fromVideo.pause(); // 물러난 영상을 일시정지합니다.
      fromVideo.currentTime = 0; // 시간을 0으로 되돌립니다.
      fromVideo.classList.remove("is-active"); // 화면에서 숨깁니다.
      fromVideo.classList.add("is-preload"); // 다시 대기 상태로 만듭니다.
      active = toVideo; // 활성 영상 정보를 갱신합니다.
      standby = fromVideo; // 대기 영상 정보를 갱신합니다.
      isSwitching = false; // 교체 종료를 알립니다.
    }, 520);
  };

  const tick = () => { // 매 프레임마다 영상이 끝날 때가 되었는지 감시하는 기능입니다.
    if (!active.paused) { // 영상이 재생 중일 때만 확인합니다.
      const duration = Number(active.duration); // 전체 길이를 확인합니다.
      if (Number.isFinite(duration) && duration > 0) { // 길이가 정상적일 때
        const remaining = duration - active.currentTime; // 남은 시간을 계산합니다.
        if (Number.isFinite(remaining) && remaining <= BG_VIDEO_CROSSFADE_SEC) { // 끝나기 직전(0.42초 전)이면
          swap(); // 다음 영상으로 교체합니다.
        }
      }
    }
    rafId = requestAnimationFrame(tick); // 다음 화면 프레임에서도 똑같이 감찰합니다.
  };

  const start = () => { // 배경 반복 시스템을 가동하는 기능입니다.
    safePlay(active); // 현재 활성 영상을 재생합니다.
    cancelAnimationFrame(rafId); // 혹시 중복된 감시가 있다면 끕니다.
    rafId = requestAnimationFrame(tick); // 실시간 감시(tick)를 시작합니다.
  };

  if (active.readyState >= 2) { // 영상이 이미 불러와져 있다면
    start(); // 바로 시작합니다.
  } else {
    active.addEventListener("canplay", start, { once: true }); // 불러와지면 시작하도록 대기합니다.
  }

  document.addEventListener("visibilitychange", () => { // 탭이 바뀌었을 때 영상 재생을 어떻게 할지 결정합니다.
    if (document.hidden) { // 화면이 안 보이면
      switchSeq += 1; // 교체 순서를 올립니다.
      isSwitching = false; // 교체 중단 상태로 만듭니다.
      active.pause(); // 재생 중인 영상을 멈춥니다.
      standby.pause(); // 대기 중인 영상도 멈춥니다.
      cancelAnimationFrame(rafId); // 실시간 감시를 중단합니다.
      return;
    }
    start(); // 화면이 다시 보이면 영상을 다시 시작합니다.
  });
}

// 손 위치(0~1 사이의 비율)를 화면상의 실제 위치(픽셀 단위)로 변환해주는 기능입니다.
function createInstrumentPoint(landmark) {
  return {
    x: (1 - landmark.x) * handCanvas.width, // 가로 위치를 반전시켜 화면 너비를 곱합니다.
    y: landmark.y * handCanvas.height // 세로 위치를 화면 높이와 곱합니다.
  };
}

// 특정 점(손가락 끝 등)이 악기 그림 안에 들어와 있는지(닿았는지) 확인하는 핵심 기능입니다.
function isInsideElement(point, element, padding = COLLISION_PADDING) {
  if (!element) return false; // 물체가 없으면 확인할 수 없습니다.
  const rect = element.getBoundingClientRect(); // 물체의 실제 화면상 사각형 영역을 가져옵니다.
  return ( // 점이 사각형 안에 있는지 확인하여 참/거짓을 돌려줍니다.
    point.x >= rect.left - padding && // 왼쪽 경계보다 오른쪽에 있는지
    point.x <= rect.right + padding && // 오른쪽 경계보다 왼쪽에 있는지
    point.y >= rect.top - padding && // 위쪽 경계보다 아래에 있는지
    point.y <= rect.bottom + padding // 아래쪽 경계보다 위에 있는지
  );
}

// 화면 오른쪽 아래의 '소리 켜기/끄기' 버튼의 글자를 현재 상태에 맞춰 바꿔주는 기능입니다.
function updateSoundButtonUI() {
  const state = Audio.getAudioState(); // 현재 소리 설정 상태를 가져옵니다.
  soundUnlockButton.textContent = state.running ? "소리 끄기" : "소리 켜기"; // 소리가 켜져 있으면 '끄기', 꺼져 있으면 '켜기'로 글자를 바꿉니다.
}

function activateStart() { // 게임을 실제로 시작하는 기능입니다.
  sessionStarted = true; // 세션이 시작되었음을 표시합니다.
  hoverActive = false; // 시작 버튼 호버 상태를 끕니다.
  hoverStartedAt = 0; // 호버 시작 시간을 초기화합니다.
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

// 숲 모드를 바꾸는 기능입니다. (마법 모드 혹은 차분한 모드)
function applySceneMode(mode) {
  if (mode === "magic") { // 마법 모드이면
    scene.dataset.mode = "magic"; // 화면 설정을 'magic'으로 바꿉니다.
  } else { // 아니면
    scene.dataset.mode = "calm"; // 화면 설정을 'calm'으로 바꿉니다.
  }
}

// '피버 타임(대흥분 상태)'을 시작하는 기능입니다. 숲이 반짝이며 더 신나게 변합니다.
function triggerFever(now) {
  feverUntil = now + FEVER_DURATION_MS; // 피버 타임이 끝날 시간을 계산합니다.
  scene.dataset.fever = "on"; // 화면 설정을 'fever on'으로 바꿉니다.
  pulseMessage.textContent = "피버 타임! 숲이 깨어났어요!"; // 화면에 큰 메시지를 띄웁니다.
  statusText.textContent = "피버 타임 진행 중 - 더 많이 터치해봐!"; // 상태 안내를 바꿉니다.
}

function updateFeverState(now) { // 현재 피버 타임인지 확인하고 상태를 업데이트하는 기능입니다.
  if (feverUntil > now) return; // 피버 타임이 아직 안 끝났으면 그냥 넘어갑니다.
  if (scene.dataset.fever === "on") { // 피버 타임 중이었는데 시간이 다 됐으면
    scene.dataset.fever = "off"; // 피버 타임을 끕니다.
    pulseMessage.textContent = "손을 잼잼! 해서 숲을 깨워봐!"; // 다시 유혹하는 메시지를 띄웁니다.
    if (sessionStarted) statusText.textContent = "손으로 악기 식물을 터치해 보세요."; // 상태 표시를 바꿉니다.
  }
}

function registerHit(now) { // 터치가 발생했을 때 이를 기록하고 피버 타임 여부를 계산하는 기능입니다.
  hitStamps.push(now); // 터치한 시간을 목록에 추가합니다.
  hitStamps = hitStamps.filter((ts) => now - ts <= FEVER_TRIGGER_WINDOW_MS); // 5초 이내의 터치만 남기고 나머지는 지웁니다.
  if (scene.dataset.fever === "off" && hitStamps.length >= FEVER_TRIGGER_HITS) { // 피버가 꺼져있을 때 30번 넘게 터치했으면
    hitStamps = []; // 기록을 초기화하고
    triggerFever(now); // 피버 타임을 폭발시킵니다!
  }
}

// 악기를 터치했을 때 사방으로 튀어나오는 예쁜 가루(파티클) 효과를 만드는 기능입니다.
function spawnBurst(type, element) {
  if (!element) return; // 기준이 되는 악기 요소가 없으면 파티클을 만들 수 없습니다.
  const rect = element.getBoundingClientRect(); // 악기의 현재 위치와 크기를 읽어옵니다.
  const cx = rect.left + rect.width * 0.5; // 악기 중앙의 x좌표를 계산합니다.
  const cy = rect.top + rect.height * 0.5; // 악기 중앙의 y좌표를 계산합니다.
  const baseColors =
    type === "drum"
      ? ["#ffd88b", "#ff9f68", "#fff0b0"] // 드럼 계열은 따뜻한 주황색 계열입니다.
      : type === "xylophone"
        ? ["#95f5ff", "#7ff9b8", "#ffd27f", "#ff9dc2"] // 실로폰 계열은 밝고 알록달록한 색입니다.
        : type === "fist"
          ? ["#ff9f68", "#ffd388", "#ff7a7a"] // 주먹 제스처는 강한 느낌의 색을 씁니다.
          : type === "openpalm"
            ? ["#9cf6ff", "#b5ffca", "#ffeab0"] // 손바닥 제스처는 부드러운 색을 씁니다.
            : type === "pinky"
              ? ["#ffc6ef", "#ff9bdc", "#ffd9f4"] // 새끼손가락 제스처는 분홍 계열을 씁니다.
              : type === "animal"
                ? ["#b8ffa4", "#ffe695", "#8ed6ff", "#ffb5d9"] // 애니멀 제스처는 장난감 같은 여러 색을 섞습니다.
                : type === "kheart"
                  ? ["#ff7fcb", "#ff4fa8", "#ffd0ec", "#ffe1f4"] // K-하트는 하트 느낌이 나는 분홍색 위주입니다.
                  : ["#ffc5df", "#ffe9a3", "#d0ffa8"]; // 그 외에는 기본 반짝이 색을 씁니다.

  const count =
    type === "animal" ? 28 // 애니멀 제스처는 조금 더 풍성하게 뿌립니다.
      : type === "kheart" ? 32 // K-하트는 가장 많이 뿌립니다.
        : type === "fist" ? 24 // 주먹은 중간 정도 양입니다.
          : 18; // 기본값은 18개입니다.

  for (let i = 0; i < count; i += 1) {
    const angle = (Math.PI * 2 * i) / count + Math.random() * 0.24; // 원형으로 퍼지되, 약간의 랜덤 각도를 더합니다.
    const speed = 1.6 + Math.random() * 3.5; // 각 파티클 속도를 조금씩 다르게 만듭니다.
    particles.push({
      x: cx, // 시작 위치 x는 악기의 중앙입니다.
      y: cy, // 시작 위치 y도 악기의 중앙입니다.
      vx: Math.cos(angle) * speed, // 가로 속도는 각도에 따라 정해집니다.
      vy: Math.sin(angle) * speed - 1.2, // 세로 속도는 위로 살짝 튀게 보정합니다.
      gravity: 0.07, // 시간이 갈수록 아래로 떨어지게 하는 중력 값입니다.
      life: 46 + Math.random() * 34, // 파티클이 몇 프레임 동안 살지 정합니다.
      maxLife: 80, // 투명도 계산에 사용할 최대 수명입니다.
      size: type === "kheart" ? 4 + Math.random() * 5 : 3 + Math.random() * 4, // K-하트는 조금 더 크게 만듭니다.
      color: baseColors[Math.floor(Math.random() * baseColors.length)] // 준비한 색 중 하나를 랜덤으로 고릅니다.
    });
  }
}

function updateParticles() {
  effectCtx.clearRect(0, 0, effectCanvas.width, effectCanvas.height); // 이전 프레임의 파티클 그림을 모두 지웁니다.

  for (let i = particles.length - 1; i >= 0; i -= 1) {
    const p = particles[i]; // 현재 파티클 하나를 가져옵니다.
    p.x += p.vx; // 가로 속도만큼 이동합니다.
    p.y += p.vy; // 세로 속도만큼 이동합니다.
    p.vy += p.gravity; // 중력으로 인해 아래 방향 속도를 조금 늘립니다.
    p.vx *= 0.98; // 시간이 갈수록 가로 속도를 줄입니다.
    p.life -= 1; // 수명을 한 프레임 줄입니다.

    if (p.life <= 0) {
      particles.splice(i, 1); // 수명이 다한 파티클은 제거합니다.
      continue; // 제거했으니 다음 파티클로 넘어갑니다.
    }

    const alpha = p.life / p.maxLife; // 남은 수명 비율로 투명도를 계산합니다.
    effectCtx.save(); // 현재 그리기 상태를 저장합니다.
    effectCtx.globalAlpha = alpha; // 남은 수명만큼만 보이게 합니다.
    effectCtx.fillStyle = p.color; // 파티클 색을 정합니다.
    effectCtx.beginPath(); // 새 도형을 그릴 준비를 합니다.
    effectCtx.ellipse(p.x, p.y, p.size, p.size * 0.58, 0, 0, Math.PI * 2); // 길쭉한 반짝이 모양을 그립니다.
    effectCtx.fill(); // 실제 색을 채워 넣습니다.
    effectCtx.restore(); // 저장했던 캔버스 상태로 되돌립니다.
  }
}

// 게임 시작 전, 첫 화면에서 '시작 버튼' 위에 손을 올리고 있는지 확인하고 시간을 재는 기능입니다.
function processLandingHover(point, now) {
  if (landingOverlay.classList.contains("is-hidden")) return;

  if (isInsideElement(point, landingStartButton, 24)) {
    if (!hoverActive) {
      hoverActive = true;
      hoverStartedAt = now;
    }

    const remain = Math.max(0, START_HOVER_MS - (now - hoverStartedAt));
    statusText.textContent = `시작까지 ${Math.ceil(remain / 100)}초...`;
    if (now - hoverStartedAt >= START_HOVER_MS) activateStart();
  } else {
    hoverActive = false;
    hoverStartedAt = 0;
    statusText.textContent = "시작 버튼 위에 손을 올려주세요.";
  }
}

// 여러 손가락 끝이 악기들과 충돌했는지(닿았는지) 일괄 확인하는 기능입니다.
function processInstrumentCollision(points, now) {
  if (!sessionStarted) return; // 시작 전에는 악기 충돌을 검사하지 않습니다.
  if (adminEditMode) return; // 관리자 모드에서는 연주를 막습니다.
  if (INTERACTION_MODE === "gesture") return; // 제스처 모드에서는 손가락 충돌 대신 제스처 판정을 사용합니다.

  for (const instrument of instruments) {
    const hit = points.some((point) => isInsideElement(point, instrument.el)); // 손가락 끝 중 하나라도 악기 안에 들어갔는지 검사합니다.
    if (!hit) continue; // 닿지 않았으면 다음 악기로 넘어갑니다.
    if (now - instrument.lastHitAt < instrument.cooldownMs) continue; // 방금 울린 악기면 잠깐 쉬게 합니다.

    instrument.lastHitAt = now; // 마지막 연주 시각을 기록합니다.
    Audio.setPlaybackContext({ instrumentId: instrument.id, gestureLabel: "Touch", gestureSource: "touch", triggerTs: now });
    const playedTag = instrument.onHit(); // 악기 소리와 이펙트를 실행합니다.
    instrument.el.classList.add("active"); // 악기 활성화 애니메이션을 켭니다.
    window.setTimeout(() => instrument.el.classList.remove("active"), 260); // 잠시 뒤 애니메이션을 끕니다.
    registerHit(now); // 피버 타임 계산을 위해 터치 기록을 남깁니다.
    statusText.textContent = `${instrument.name} - ${playedTag || instrument.soundTag}`; // 어떤 악기가 울렸는지 보여줍니다.
  }
}

function triggerInstrumentById(id, now, meta = {}) {
  const instrument = instruments.find((item) => item.id === id); // id에 해당하는 악기를 찾습니다.
  if (!instrument || !instrument.el) return; // 못 찾으면 끝냅니다.
  if (now - instrument.lastHitAt < instrument.cooldownMs) return; // 쿨다운 중이면 발동하지 않습니다.

  instrument.lastHitAt = now; // 연주 시각을 갱신합니다.
  Audio.setPlaybackContext({
    instrumentId: instrument.id,
    gestureLabel: meta.gestureLabel || null,
    gestureSource: meta.gestureSource || null,
    triggerTs: now
  });
  const playedTag = instrument.onHit(); // 악기별 동작을 실행합니다.
  instrument.el.classList.add("active"); // 시각 효과를 켭니다.
  window.setTimeout(() => instrument.el.classList.remove("active"), 260); // 짧게 보여주고 끕니다.
  registerHit(now); // 상호작용 횟수를 기록합니다.
  statusText.textContent = `${instrument.name} - ${playedTag || instrument.soundTag}`; // 상태 문구를 갱신합니다.
}

function triggerVisualOnlyById(id, now, burstType = "pinky") {
  const instrument = instruments.find((item) => item.id === id); // 효과만 낼 악기 정보를 찾습니다.
  if (!instrument || !instrument.el) return false; // 악기가 없으면 실패입니다.
  if (now - instrument.lastHitAt < instrument.cooldownMs) return false; // 아직 대기 시간 중이면 실패입니다.

  instrument.lastHitAt = now; // 시각 효과를 낸 시각을 저장합니다.
  spawnBurst(burstType, instrument.el); // 소리 없이 파티클만 만듭니다.
  instrument.el.classList.add("active"); // 활성화 애니메이션은 같이 보여줍니다.
  window.setTimeout(() => instrument.el.classList.remove("active"), 260); // 잠시 뒤 원래 상태로 돌립니다.
  registerHit(now); // 이것도 상호작용으로 집계합니다.
  return true; // 성공적으로 처리했다고 알려줍니다.
}

function getGestureDisplayName(label) {
  if (label === "Fist") return "주먹"; // 영어 라벨을 한국어로 바꿉니다.
  if (label === "OpenPalm") return "손바닥"; // 손바닥 제스처입니다.
  if (label === "V") return "브이"; // 브이 제스처입니다.
  if (label === "Pinky") return "새끼손가락"; // 새끼손가락 제스처입니다.
  if (label === "Animal") return "애니멀"; // 애니멀 제스처입니다.
  if (label === "KHeart") return "K-하트"; // K-하트 제스처입니다.
  return label; // 정의되지 않은 값이면 원래 이름을 그대로 씁니다.
}

function runGestureReaction(label, now) {
  if (label === "Animal") {
    if (!triggerVisualOnlyById("fern", now, "pinky")) return; // 애니멀은 먼저 고사리 쪽에 시각 효과만 냅니다.
    Audio.setPlaybackContext({ instrumentId: "fern", gestureLabel: label, gestureSource: "model", triggerTs: now });
    Audio.playAnimalRoll(); // 그 다음 전용 효과음을 재생합니다.
    return; // 애니멀은 여기서 처리를 끝냅니다.
  }

  const mapping = {
    Fist: "drum",
    OpenPalm: "xylophone",
    V: "tambourine",
    Pinky: "owl",
    Animal: "fern",
    KHeart: "fern"
  };

  const id = mapping[label]; // 이 제스처가 어떤 악기와 연결되는지 찾습니다.
  if (!id) return; // 연결된 악기가 없으면 아무것도 하지 않습니다.
  triggerInstrumentById(id, now, { gestureLabel: label, gestureSource: "model" }); // 연결된 악기를 실제로 울립니다.
}

function showSquirrelEffect() {
  if (!gestureSquirrelEffect) return; // 효과용 이미지가 없으면 종료합니다.
  gestureSquirrelEffect.classList.remove("is-visible"); // 기존 표시 상태를 한번 지웁니다.
  void gestureSquirrelEffect.offsetWidth; // 브라우저에게 다시 계산하게 해서 애니메이션을 재시작할 수 있게 합니다.
  gestureSquirrelEffect.classList.add("is-visible"); // 다시 보이게 하며 애니메이션을 실행합니다.
}

function processGestureTriggers(landmarks, now) {
  if (!sessionStarted) return; // 시작 전에는 제스처 판정을 사용하지 않습니다.
  if (adminEditMode) return; // 관리자 모드에서는 제스처 연주를 끕니다.

  const gesture = resolveGesture(landmarks, now, sessionStarted); // 현재 손 모양의 최종 판정을 받습니다.
  if (!gesture || gesture.label === "None") {
    const rawModel = getModelPrediction(landmarks, now);
    if (rawModel?.classId === 0 || String(rawModel?.label || "").trim().toLowerCase() === "class0") {
      statusText.textContent = "동작을 다시해주세요.";
    }
    lastGestureLabel = "None"; // 아무것도 인식되지 않았다면 마지막 라벨을 비웁니다.
    return; // 더 처리할 것이 없습니다.
  }

  const inCooldown = now - lastGestureHitAt < GESTURE_TRIGGER_COOLDOWN_MS; // 직전 제스처 발동 후 대기 시간 안인지 확인합니다.
  if (inCooldown && gesture.label === lastGestureLabel) {
    return; // 같은 동작이 너무 빨리 반복되면 무시합니다.
  }

  if (!Audio.getAudioState().running) {
    statusText.textContent = "소리가 꺼져 있어요. '소리 켜기' 버튼을 눌러주세요.";
    lastGestureLabel = gesture.label;
    return;
  }

  runGestureReaction(gesture.label, now); // 제스처에 맞는 악기 반응을 실행합니다.
  showSquirrelEffect(); // 다람쥐 효과 애니메이션도 함께 재생합니다.

  lastGestureHitAt = now; // 이번 제스처의 발동 시각을 저장합니다.
  lastGestureLabel = gesture.label; // 마지막으로 인식된 제스처 이름을 저장합니다.
  const displayName = getGestureDisplayName(gesture.label); // 영어 이름을 사용자용 한국어 이름으로 바꿉니다.
  statusText.textContent = `손동작 ${displayName} 인식! (${gesture.source})`; // 상태 문구를 갱신합니다.
}

// 손 위치를 나타내는 커서(포인터)를 화면상의 정확한 좌표로 옮겨주는 기능입니다.
function setPointer(point) {
  handCursor.style.opacity = 1;
  handCursor.style.left = `${point.x}px`;
  handCursor.style.top = `${point.y}px`;
}

// [가장 중요한 부분] 1초에 수십 번씩 실행되면서 손의 위치를 계산하고,
// 악기에 닿았는지 확인해서 소리를 내는 '무한 반복' 기능입니다.
async function predict() {
  if (!handLandmarker) {
    requestAnimationFrame(predict); // 아직 손 인식기가 준비되지 않았으면 다음 프레임에서 다시 시도합니다.
    return; // 이번 프레임 처리는 여기서 멈춥니다.
  }

  const now = performance.now(); // 현재 시각을 읽습니다.
  handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height); // 이전 프레임의 손 그림을 지웁니다.
  updateFeverState(now); // 피버 타임이 끝났는지 확인합니다.

  const cacheFresh = cachedLandmarks && now - cachedLandmarksAt <= LANDMARK_STALE_MS; // 마지막 손 좌표가 아직 유효한지 확인합니다.
  if (cacheFresh) {
    Renderer.drawHand(handCtx, cachedLandmarks, handCanvas, now * 0.001); // 저장해둔 손 좌표를 화면에 그립니다.
    const pointer = createInstrumentPoint(cachedLandmarks[8]); // 검지 끝을 화면 좌표로 바꿉니다.
    setPointer(pointer); // 손 커서를 해당 위치로 옮깁니다.
  } else if (cachedLandmarks) {
    cachedLandmarks = null; // 오래된 좌표는 버립니다.
    handCursor.style.opacity = 0; // 커서도 숨깁니다.
  }

  const hasFreshFrame = video.currentTime !== lastVideoTime && video.readyState >= 2 && video.videoWidth > 0; // 새로운 카메라 프레임이 들어왔는지 확인합니다.
  const inferenceDue = now - lastInferenceAt >= INFER_INTERVAL_MS; // 손 인식을 실행할 간격이 되었는지 확인합니다.

  if (hasFreshFrame && inferenceDue) {
    lastVideoTime = video.currentTime; // 이번 프레임 시간을 기억합니다.
    lastInferenceAt = now; // 추론을 수행한 시각도 저장합니다.

    try {
      const result = handLandmarker.detectForVideo(video, now); // 현재 영상 프레임에서 손을 찾습니다.

      if (result.landmarks.length > 0) {
        const hands = result.landmarks; // 감지된 모든 손 랜드마크를 가져옵니다.
        const landmarks = hands[0]; // 커서/주요 제스처 판정은 첫 번째 손을 기준으로 사용합니다.
        cachedLandmarks = landmarks; // 다음 프레임 표시를 위해 저장합니다.
        cachedLandmarksAt = now; // 저장 시각도 함께 기록합니다.

        const pointer = createInstrumentPoint(landmarks[8]); // 검지 끝 위치를 화면 좌표로 변환합니다.
        processLandingHover(pointer, now); // 시작 버튼 위에 손을 올렸는지 검사합니다.

        const triggerPoints = hands.flatMap((hand) => [4, 8, 12, 16, 20].map((idx) => createInstrumentPoint(hand[idx]))); // 감지된 모든 손의 손가락 끝 위치를 충돌 판정에 사용합니다.
        processInstrumentCollision(triggerPoints, now); // 터치 모드 충돌 판정을 실행합니다.
        processGestureTriggers(landmarks, now); // 제스처 모드 판정을 실행합니다.
      } else {
        cachedLandmarks = null; // 손이 안 보이면 저장된 좌표를 지웁니다.
        cachedLandmarksAt = 0; // 좌표 저장 시각도 초기화합니다.
        handCursor.style.opacity = 0; // 손 커서를 숨깁니다.
        if (!sessionStarted) statusText.textContent = "카메라에 손을 보여주세요."; // 시작 전이라면 손을 보여달라고 안내합니다.
      }
    } catch (error) {
      console.warn("MediaPipe detection error:", error); // 손 인식 오류는 콘솔에만 남깁니다.
    }
  }

  updateParticles(); // 파티클 애니메이션도 함께 업데이트합니다.
  requestAnimationFrame(predict); // 다음 프레임에서도 다시 실행합니다.
}

async function initCamera() {
  try {
    let stream; // 카메라에서 받아올 영상 스트림을 담는 변수입니다.
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 960 }, // 가능하면 가로 960 정도 화질을 요청합니다.
          height: { ideal: 540 }, // 가능하면 세로 540 정도 화질을 요청합니다.
          frameRate: { ideal: 30, max: 30 } // 초당 30프레임 정도를 목표로 합니다.
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
      predict(); // 손 추적 루프를 시작합니다.
    };
  } catch {
    statusText.textContent = "카메라 권한을 허용해 주세요."; // 카메라 권한이 없으면 메시지를 보여줍니다.
    statusText.style.color = "var(--danger)"; // 경고 색으로 강조합니다.
  }
}

async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm"); // MediaPipe 엔진이 필요한 파일들을 불러옵니다.
  const modelAssetPath = new URL("../../public/hand_landmarker.task", import.meta.url).toString(); // 손 인식 모델 파일의 실제 주소를 만듭니다.
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

async function init() {
  setupSeamlessBackgroundLoop(); // 배경 영상 반복 시스템을 먼저 준비합니다.
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
  applySceneMode(mode); // 장면 분위기를 적용합니다.

  if (params.get("admin") === "1") {
    setupAdminDragMode(); // 관리자 옵션이 있으면 배치 편집 모드로 들어갑니다.
  }

  const savedLayout = loadInstrumentLayout(); // 저장된 악기 배치가 있는지 확인합니다.
  applyInstrumentLayout(savedLayout || DEFAULT_LAYOUT); // 있으면 불러오고 없으면 기본 배치를 씁니다.

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
