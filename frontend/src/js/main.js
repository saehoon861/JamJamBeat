import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";
import * as Audio from "./audio.js";
import * as Renderer from "./renderer.js";
import { resolveGesture } from "./gestures.js";

const video = document.getElementById("webcam");
const handCanvas = document.getElementById("handCanvas");
const effectCanvas = document.getElementById("effectCanvas");
const handCtx = handCanvas.getContext("2d");
const effectCtx = effectCanvas.getContext("2d");
const statusText = document.getElementById("status");
const scene = document.getElementById("scene");
const handCursor = document.getElementById("handCursor");
const landingOverlay = document.getElementById("landingOverlay");
const landingStartButton = document.getElementById("landingStartButton");
const soundUnlockButton = document.getElementById("soundUnlockButton");
const pulseMessage = document.getElementById("pulseMessage");
const adminControls = document.getElementById("adminControls");
const adminSaveButton = document.getElementById("adminSaveButton");
const adminResetButton = document.getElementById("adminResetButton");
const gestureSquirrelEffect = document.getElementById("gestureSquirrelEffect");

const instrumentElements = {
  drum: document.getElementById("instrumentDrum"),
  xylophone: document.getElementById("instrumentXylophone"),
  tambourine: document.getElementById("instrumentTambourine"),
  fern: document.getElementById("instrumentFern"),
  owl: document.getElementById("instrumentOwl")
};

const COLLISION_PADDING = 12;
const START_HOVER_MS = 520;
const FEVER_TRIGGER_WINDOW_MS = 5000;
const FEVER_TRIGGER_HITS = 6;
const FEVER_DURATION_MS = 6200;

let handLandmarker;
let lastVideoTime = -1;
let sessionStarted = false;
let hoverStartedAt = 0;
let hoverActive = false;
let feverUntil = 0;
let hitStamps = [];
let cameraStream = null;
let lastGestureHitAt = 0;
let lastGestureLabel = "None";
let adminEditMode = false;
let dragState = null;

const GESTURE_TRIGGER_COOLDOWN_MS = 520;
const BG_VIDEO_CROSSFADE_SEC = 0.42;
const INSTRUMENT_LAYOUT_KEY = "jamjam.instrumentLayout.v1";
const DEFAULT_LAYOUT = {
  drum: { x: 10, y: 14 },
  xylophone: { x: 38, y: 22 },
  tambourine: { x: 68, y: 18 },
  fern: { x: 52, y: 10 },
  owl: { x: 24, y: 7 }
};

const particles = [];

const instruments = [
  {
    id: "drum",
    name: "고슴도치 드럼",
    soundTag: "쿵",
    el: instrumentElements.drum,
    cooldownMs: 320,
    lastHitAt: 0,
    onHit() {
      Audio.playDrumMushroom();
      spawnBurst("drum", this.el);
    }
  },
  {
    id: "xylophone",
    name: "노래하는 백합",
    soundTag: "도레미",
    el: instrumentElements.xylophone,
    cooldownMs: 360,
    lastHitAt: 0,
    onHit() {
      Audio.playXylophoneVine();
      spawnBurst("xylophone", this.el);
    }
  },
  {
    id: "tambourine",
    name: "행운 클로버",
    soundTag: "챵",
    el: instrumentElements.tambourine,
    cooldownMs: 380,
    lastHitAt: 0,
    onHit() {
      Audio.playTambourineFlower();
      spawnBurst("tambourine", this.el);
    }
  },
  {
    id: "fern",
    name: "마법 고사리",
    soundTag: "링",
    el: instrumentElements.fern,
    cooldownMs: 380,
    lastHitAt: 0,
    onHit() {
      Audio.playPinkyChime();
      spawnBurst("pinky", this.el);
    }
  },
  {
    id: "owl",
    name: "아기 사슴",
    soundTag: "삐요",
    el: instrumentElements.owl,
    cooldownMs: 380,
    lastHitAt: 0,
    onHit() {
      Audio.playHeartBloom();
      spawnBurst("xylophone", this.el);
    }
  }
];

function setCanvasSize() {
  handCanvas.width = window.innerWidth;
  handCanvas.height = window.innerHeight;
  effectCanvas.width = window.innerWidth;
  effectCanvas.height = window.innerHeight;
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function loadInstrumentLayout() {
  try {
    const raw = localStorage.getItem(INSTRUMENT_LAYOUT_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return parsed;
  } catch {
    return null;
  }
}

function saveInstrumentLayout(layout) {
  try {
    localStorage.setItem(INSTRUMENT_LAYOUT_KEY, JSON.stringify(layout));
  } catch {
    // ignore storage errors
  }
}

function applyInstrumentLayout(layout) {
  if (!layout || typeof layout !== "object") return;

  const ids = ["drum", "xylophone", "tambourine", "fern", "owl"];
  ids.forEach((id) => {
    const target = instrumentElements[id];
    const pos = layout[id];
    if (!target || !pos || typeof pos !== "object") return;

    const x = Number(pos.x);
    const y = Number(pos.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;

    target.style.left = `${clamp(x, 0, 90)}vw`;
    target.style.bottom = `${clamp(y, 0, 80)}vh`;
    target.style.right = "auto";
  });
}

function readCurrentInstrumentLayout() {
  const layout = {};
  const width = Math.max(1, window.innerWidth);
  const height = Math.max(1, window.innerHeight);

  ["drum", "xylophone", "tambourine", "fern", "owl"].forEach((id) => {
    const el = instrumentElements[id];
    if (!el) return;
    const rect = el.getBoundingClientRect();
    layout[id] = {
      x: clamp((rect.left / width) * 100, 0, 90),
      y: clamp(((height - rect.bottom) / height) * 100, 0, 80)
    };
  });

  return layout;
}

function setupAdminDragMode() {
  adminEditMode = true;
  scene.dataset.admin = "on";
  landingOverlay.classList.add("is-hidden");
  statusText.textContent = "관리자 모드: 오브젝트를 드래그하고 저장하세요.";

  if (adminControls) {
    adminControls.classList.add("is-visible");
  }

  const onPointerMove = (event) => {
    if (!dragState) return;

    const width = Math.max(1, window.innerWidth);
    const height = Math.max(1, window.innerHeight);
    const rect = dragState.el.getBoundingClientRect();

    const leftPx = clamp(event.clientX - dragState.offsetX, 0, width - rect.width);
    const topPx = clamp(event.clientY - dragState.offsetY, 0, height - rect.height);
    const x = clamp((leftPx / width) * 100, 0, 90);
    const y = clamp(((height - (topPx + rect.height)) / height) * 100, 0, 80);

    dragState.el.style.left = `${x}vw`;
    dragState.el.style.bottom = `${y}vh`;
    dragState.el.style.right = "auto";
  };

  const onPointerUp = () => {
    if (!dragState) return;
    dragState.el.classList.remove("is-dragging");
    dragState = null;
  };

  ["drum", "xylophone", "tambourine", "fern", "owl"].forEach((id) => {
    const el = instrumentElements[id];
    if (!el) return;

    el.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const rect = el.getBoundingClientRect();
      dragState = {
        id,
        el,
        offsetX: event.clientX - rect.left,
        offsetY: event.clientY - rect.top
      };
      el.classList.add("is-dragging");
      if (el.setPointerCapture) {
        try {
          el.setPointerCapture(event.pointerId);
        } catch {
          // no-op
        }
      }
    });
  });

  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  window.addEventListener("pointercancel", onPointerUp);

  if (adminSaveButton) {
    adminSaveButton.onclick = () => {
      const layout = readCurrentInstrumentLayout();
      saveInstrumentLayout(layout);
      adminSaveButton.textContent = "저장됨";
      window.setTimeout(() => {
        adminSaveButton.textContent = "배치 저장";
      }, 900);
    };
  }

  if (adminResetButton) {
    adminResetButton.onclick = () => {
      localStorage.removeItem(INSTRUMENT_LAYOUT_KEY);
      applyInstrumentLayout(DEFAULT_LAYOUT);
    };
  }
}

function setupSeamlessBackgroundLoop() {
  const videoA = document.querySelector(".bg-video-a");
  const videoB = document.querySelector(".bg-video-b");
  if (!videoA || !videoB) return;

  const videos = [videoA, videoB];
  videos.forEach((video, idx) => {
    video.muted = true;
    video.playsInline = true;
    video.loop = false;
    video.playbackRate = 1;
    video.classList.toggle("is-active", idx === 0);
    video.classList.toggle("is-preload", idx !== 0);
  });

  let active = videoA;
  let standby = videoB;
  let rafId = 0;
  let isSwitching = false;

  const safePlay = (video) => {
    const p = video.play();
    if (p && typeof p.catch === "function") p.catch(() => {});
  };

  const swap = () => {
    if (isSwitching) return;
    isSwitching = true;

    standby.currentTime = 0;
    standby.classList.remove("is-preload");
    standby.classList.add("is-active");
    safePlay(standby);

    window.setTimeout(() => {
      active.pause();
      active.currentTime = 0;
      active.classList.remove("is-active");
      active.classList.add("is-preload");
      const prev = active;
      active = standby;
      standby = prev;
      isSwitching = false;
    }, 520);
  };

  const tick = () => {
    if (!active.paused && Number.isFinite(active.duration)) {
      const remaining = active.duration - active.currentTime;
      if (remaining <= BG_VIDEO_CROSSFADE_SEC) {
        swap();
      }
    }
    rafId = requestAnimationFrame(tick);
  };

  const start = () => {
    safePlay(active);
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(tick);
  };

  if (active.readyState >= 2) {
    start();
  } else {
    active.addEventListener("canplay", start, { once: true });
  }

  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      active.pause();
      standby.pause();
      cancelAnimationFrame(rafId);
      return;
    }
    start();
  });
}

function createInstrumentPoint(landmark) {
  return {
    x: (1 - landmark.x) * handCanvas.width,
    y: landmark.y * handCanvas.height
  };
}

function isInsideElement(point, element, padding = COLLISION_PADDING) {
  if (!element) return false;
  const rect = element.getBoundingClientRect();
  return (
    point.x >= rect.left - padding &&
    point.x <= rect.right + padding &&
    point.y >= rect.top - padding &&
    point.y <= rect.bottom + padding
  );
}

function updateSoundButtonUI() {
  const state = Audio.getAudioState();
  soundUnlockButton.textContent = state.running ? "소리 끄기" : "소리 켜기";
}

function activateStart() {
  sessionStarted = true;
  hoverActive = false;
  hoverStartedAt = 0;
  landingOverlay.classList.add("is-hidden");
  const audioState = Audio.getAudioState();
  if (audioState.running) {
    statusText.textContent = "손으로 악기 식물을 터치해 보세요.";
    Audio.startAmbientLoop();
  } else {
    statusText.textContent = "소리를 들으려면 '소리 켜기' 버튼을 눌러주세요.";
  }
}

function applySceneMode(mode) {
  if (mode === "magic") {
    scene.dataset.mode = "magic";
  } else {
    scene.dataset.mode = "calm";
  }
}

function triggerFever(now) {
  feverUntil = now + FEVER_DURATION_MS;
  scene.dataset.fever = "on";
  pulseMessage.textContent = "피버 타임! 숲이 깨어났어요!";
  statusText.textContent = "피버 타임 진행 중 - 더 많이 터치해봐!";
}

function updateFeverState(now) {
  if (feverUntil > now) return;
  if (scene.dataset.fever === "on") {
    scene.dataset.fever = "off";
    pulseMessage.textContent = "손을 잼잼! 해서 숲을 깨워봐!";
    if (sessionStarted) statusText.textContent = "손으로 악기 식물을 터치해 보세요.";
  }
}

function registerHit(now) {
  hitStamps.push(now);
  hitStamps = hitStamps.filter((ts) => now - ts <= FEVER_TRIGGER_WINDOW_MS);
  if (scene.dataset.fever === "off" && hitStamps.length >= FEVER_TRIGGER_HITS) {
    hitStamps = [];
    triggerFever(now);
  }
}

function spawnBurst(type, element) {
  if (!element) return;
  const rect = element.getBoundingClientRect();
  const cx = rect.left + rect.width * 0.5;
  const cy = rect.top + rect.height * 0.5;
  const baseColors =
    type === "drum"
      ? ["#ffd88b", "#ff9f68", "#fff0b0"]
      : type === "xylophone"
        ? ["#95f5ff", "#7ff9b8", "#ffd27f", "#ff9dc2"]
        : type === "fist"
          ? ["#ff9f68", "#ffd388", "#ff7a7a"]
          : type === "openpalm"
            ? ["#9cf6ff", "#b5ffca", "#ffeab0"]
            : type === "pinky"
              ? ["#ffc6ef", "#ff9bdc", "#ffd9f4"]
              : type === "animal"
                ? ["#b8ffa4", "#ffe695", "#8ed6ff", "#ffb5d9"]
                : type === "kheart"
                  ? ["#ff7fcb", "#ff4fa8", "#ffd0ec", "#ffe1f4"]
                  : ["#ffc5df", "#ffe9a3", "#d0ffa8"];

  const count =
    type === "animal" ? 28
      : type === "kheart" ? 32
        : type === "fist" ? 24
          : 18;

  for (let i = 0; i < count; i += 1) {
    const angle = (Math.PI * 2 * i) / count + Math.random() * 0.24;
    const speed = 1.6 + Math.random() * 3.5;
    particles.push({
      x: cx,
      y: cy,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - 1.2,
      gravity: 0.07,
      life: 46 + Math.random() * 34,
      maxLife: 80,
      size: type === "kheart" ? 4 + Math.random() * 5 : 3 + Math.random() * 4,
      color: baseColors[Math.floor(Math.random() * baseColors.length)]
    });
  }
}

function updateParticles() {
  effectCtx.clearRect(0, 0, effectCanvas.width, effectCanvas.height);

  for (let i = particles.length - 1; i >= 0; i -= 1) {
    const p = particles[i];
    p.x += p.vx;
    p.y += p.vy;
    p.vy += p.gravity;
    p.vx *= 0.98;
    p.life -= 1;

    if (p.life <= 0) {
      particles.splice(i, 1);
      continue;
    }

    const alpha = p.life / p.maxLife;
    effectCtx.save();
    effectCtx.globalAlpha = alpha;
    effectCtx.fillStyle = p.color;
    effectCtx.beginPath();
    effectCtx.ellipse(p.x, p.y, p.size, p.size * 0.58, 0, 0, Math.PI * 2);
    effectCtx.fill();
    effectCtx.restore();
  }
}

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

function processInstrumentCollision(points, now) {
  if (!sessionStarted) return;
  if (adminEditMode) return;

  for (const instrument of instruments) {
    const hit = points.some((point) => isInsideElement(point, instrument.el));
    if (!hit) continue;
    if (now - instrument.lastHitAt < instrument.cooldownMs) continue;

    instrument.lastHitAt = now;
    instrument.onHit();
    instrument.el.classList.add("active");
    window.setTimeout(() => instrument.el.classList.remove("active"), 260);
    registerHit(now);
    statusText.textContent = `${instrument.name} - ${instrument.soundTag}`;
  }
}

function triggerInstrumentById(id, now) {
  const instrument = instruments.find((item) => item.id === id);
  if (!instrument || !instrument.el) return;
  if (now - instrument.lastHitAt < instrument.cooldownMs) return;

  instrument.lastHitAt = now;
  instrument.onHit();
  instrument.el.classList.add("active");
  window.setTimeout(() => instrument.el.classList.remove("active"), 260);
  registerHit(now);
  statusText.textContent = `${instrument.name} - ${instrument.soundTag}`;
}

function getGestureDisplayName(label) {
  if (label === "Fist") return "주먹";
  if (label === "OpenPalm") return "손바닥";
  if (label === "V") return "브이";
  if (label === "Pinky") return "새끼손가락";
  if (label === "Animal") return "애니멀";
  if (label === "KHeart") return "K-하트";
  return label;
}

function runGestureReaction(label, now) {
  const mapping = {
    Fist: "drum",
    OpenPalm: "xylophone",
    V: "tambourine",
    Pinky: "owl",
    Animal: "fern",
    KHeart: "owl"
  };

  const id = mapping[label];
  if (!id) return;
  triggerInstrumentById(id, now);
}

function showSquirrelEffect() {
  if (!gestureSquirrelEffect) return;
  gestureSquirrelEffect.classList.remove("is-visible");
  void gestureSquirrelEffect.offsetWidth;
  gestureSquirrelEffect.classList.add("is-visible");
}

function processGestureTriggers(landmarks, now) {
  if (!sessionStarted) return;
  if (adminEditMode) return;

  const gesture = resolveGesture(landmarks, now, sessionStarted);
  if (!gesture || gesture.label === "None") {
    lastGestureLabel = "None";
    return;
  }

  const inCooldown = now - lastGestureHitAt < GESTURE_TRIGGER_COOLDOWN_MS;
  if (inCooldown && gesture.label === lastGestureLabel) {
    return;
  }

  runGestureReaction(gesture.label, now);
  showSquirrelEffect();

  lastGestureHitAt = now;
  lastGestureLabel = gesture.label;
  const displayName = getGestureDisplayName(gesture.label);
  statusText.textContent = `손동작 ${displayName} 인식! (${gesture.source})`;
}

function setPointer(point) {
  handCursor.style.opacity = 1;
  handCursor.style.left = `${point.x}px`;
  handCursor.style.top = `${point.y}px`;
}

async function predict() {
  if (!handLandmarker) {
    requestAnimationFrame(predict);
    return;
  }

  if (video.currentTime !== lastVideoTime && video.readyState >= 2 && video.videoWidth > 0) {
    lastVideoTime = video.currentTime;
    const now = performance.now();

    try {
      const result = handLandmarker.detectForVideo(video, now);
      handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);
      updateFeverState(now);

      if (result.landmarks.length > 0) {
        const landmarks = result.landmarks[0];
        Renderer.drawHand(handCtx, landmarks, handCanvas, now * 0.001);

        const pointer = createInstrumentPoint(landmarks[8]);
        setPointer(pointer);
        processLandingHover(pointer, now);

        const triggerPoints = [4, 8, 12, 16, 20].map((idx) => createInstrumentPoint(landmarks[idx]));
        processInstrumentCollision(triggerPoints, now);
        processGestureTriggers(landmarks, now);
      } else {
        handCursor.style.opacity = 0;
        if (!sessionStarted) statusText.textContent = "카메라에 손을 보여주세요.";
      }
    } catch (error) {
      console.warn("MediaPipe detection error:", error);
    }
  }

  updateParticles();
  requestAnimationFrame(predict);
}

async function initCamera() {
  try {
    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 960 },
          height: { ideal: 540 },
          frameRate: { ideal: 30, max: 30 }
        }
      });
    } catch {
      stream = await navigator.mediaDevices.getUserMedia({ video: true });
    }

    video.srcObject = stream;
    cameraStream = stream;
    video.playsInline = true;
    video.muted = true;
    video.setAttribute("playsinline", "");
    video.onloadedmetadata = () => {
      video.play().catch(() => {});
      statusText.textContent = "준비 완료! 시작 버튼에 손을 올려주세요.";
      predict();
    };
  } catch {
    statusText.textContent = "카메라 권한을 허용해 주세요.";
    statusText.style.color = "var(--danger)";
  }
}

async function initMediaPipe() {
  const vision = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const modelAssetPath = new URL("../../public/hand_landmarker.task", import.meta.url).toString();

  try {
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 1
    });
  } catch (gpuError) {
    console.warn("GPU delegate failed, fallback to CPU.", gpuError);
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: "CPU"
      },
      runningMode: "VIDEO",
      numHands: 1
    });
  }
}

function bindUIEvents() {
  const tryUnlockFromGesture = async () => {
    const unlocked = await Audio.unlockAudioContext();
    if (!unlocked) return;
    if (sessionStarted) Audio.startAmbientLoop();
    updateSoundButtonUI();
  };

  window.addEventListener("pointerdown", () => {
    tryUnlockFromGesture();
  }, { once: true });

  window.addEventListener("keydown", () => {
    tryUnlockFromGesture();
  }, { once: true });

  landingStartButton.onclick = async () => {
    if (adminEditMode) return;
    const unlocked = await Audio.unlockAudioContext();
    if (unlocked) updateSoundButtonUI();
    activateStart();
  };

  soundUnlockButton.onclick = async () => {
    const state = Audio.getAudioState();
    if (state.running) {
      Audio.toggleSound();
      Audio.stopAmbientLoop();
    } else {
      const unlocked = await Audio.unlockAudioContext();
      if (!unlocked) return;
      if (sessionStarted) Audio.startAmbientLoop();
      statusText.textContent = "소리가 켜졌어요! 손으로 숲을 연주해 보세요.";
    }
    updateSoundButtonUI();
  };
}

async function init() {
  setupSeamlessBackgroundLoop();
  setCanvasSize();
  scene.dataset.fever = scene.dataset.fever || "off";
  window.addEventListener("resize", setCanvasSize);
  window.addEventListener("beforeunload", () => {
    if (!cameraStream) return;
    cameraStream.getTracks().forEach((track) => track.stop());
  });

  bindUIEvents();
  updateSoundButtonUI();

  const params = new URLSearchParams(window.location.search);
  const mode = params.get("mode") || "calm";
  applySceneMode(mode);

  if (params.get("admin") === "1") {
    setupAdminDragMode();
  }

  const savedLayout = loadInstrumentLayout();
  applyInstrumentLayout(savedLayout || DEFAULT_LAYOUT);

  const autoStart = params.get("session") === "start" || params.get("start") === "1";
  if (autoStart && !adminEditMode) {
    activateStart();
  }

  if (adminEditMode) {
    return;
  }

  try {
    await initMediaPipe();
    await initCamera();
  } catch (error) {
    console.error("Initialization failed:", error);
    statusText.textContent = "초기화 실패: 새로고침 후 다시 시도해 주세요.";
    statusText.style.color = "var(--danger)";
  }
}

init();
