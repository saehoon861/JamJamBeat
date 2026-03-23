// [theme.js] 우리 숲으로 들어가는 '입구' 역할을 하는 파일입니다.
// 여기서 어떤 모드로 플레이할지 고를 수 있습니다. 마우스 클릭 대신 손을 올려서 선택합니다.

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import * as Renderer from "./renderer.js";
import { getConfiguredHandLandmarkerTaskPath, getConfiguredMediaPipeWasmRoot } from "./env_config.js";
import { createParticleSystem } from "./particle_system.js";

const video = document.getElementById("webcam");
const canvas = document.getElementById("handCanvas");
const ctx = canvas.getContext("2d");
const effectCanvas = document.getElementById("effectCanvas");
const effectCtx = effectCanvas.getContext("2d");
const handCursor = document.getElementById("handCursor");
const statusText = document.getElementById("themeStatus");
const backButton = document.getElementById("themeBackButton");
const cards = Array.from(document.querySelectorAll(".theme-mode-card"));
const particleSystem = createParticleSystem(effectCtx, effectCanvas);

const HOVER_MS = 520;
const HOVER_PADDING = 28;
const BG_VIDEO_CROSSFADE_SEC = 0.42;
const DEFAULT_INFER_FPS = 15;
const MIN_INFER_FPS = 8;
const MAX_INFER_FPS = 60;
const LANDMARK_STALE_MS = 300;
const POINTER_TRAIL_MIN_DISTANCE = 14;
const POINTER_TRAIL_MIN_INTERVAL_MS = 28;
const LANDMARK_TRAIL_MIN_DISTANCE = 12;
const LANDMARK_TRAIL_MIN_INTERVAL_MS = 30;

let handLandmarker;
let lastVideoTime = -1;
let hoverTarget = null;
let hoverStartedAt = 0;
let cameraStream = null;
let lastInferenceAt = 0;
let cachedLandmarks = null;
let cachedLandmarksAt = 0;
let lastPointerTrailAt = 0;
let lastPointerTrailPoint = null;
let lastLandmarkTrail = null;

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function parseInferFps() {
  const params = new URLSearchParams(window.location.search);
  const raw = Number(params.get("inferFps"));
  if (!Number.isFinite(raw)) return DEFAULT_INFER_FPS;
  return clamp(Math.round(raw), MIN_INFER_FPS, MAX_INFER_FPS);
}

const INFER_INTERVAL_MS = Math.round(1000 / parseInferFps());

function parsePreferredDelegate() {
  const params = new URLSearchParams(window.location.search);
  const raw = (params.get("mpDelegate") || "gpu").trim().toUpperCase();
  return raw === "CPU" ? "CPU" : "GPU";
}

function setCanvasSize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
  effectCanvas.width = window.innerWidth;
  effectCanvas.height = window.innerHeight;
}

// 배경 동영상이 끊기지 않고 자연스럽게 계속 반복되도록 해주는 기능입니다.
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
      if (remaining <= BG_VIDEO_CROSSFADE_SEC) swap();
    }
    rafId = requestAnimationFrame(tick);
  };

  const start = () => {
    safePlay(active);
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(tick);
  };

  if (active.readyState >= 2) start();
  else active.addEventListener("canplay", start, { once: true });

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

function clearHoverStyles() {
  if (backButton) backButton.classList.remove("is-hovered");
  cards.forEach((card) => {
    card.classList.remove("is-hovered");
  });
}

// 내 손 커서가 어떤 카드(버튼) 안에 들어가 있는지 확인하는 기능입니다.
function isCursorInside(element, x, y) {
  const rect = element.getBoundingClientRect();
  return (
    x >= rect.left - HOVER_PADDING &&
    x <= rect.right + HOVER_PADDING &&
    y >= rect.top - HOVER_PADDING &&
    y <= rect.bottom + HOVER_PADDING
  );
}

function findHoverTarget(x, y) {
  if (backButton && isCursorInside(backButton, x, y)) return backButton;
  for (const card of cards) {
    if (isCursorInside(card, x, y)) return card;
  }
  return null;
}

function activateTarget(target) {
  if (!target) return;
  const nav = target.dataset.nav || target.getAttribute("href");
  if (!nav) return;
  window.location.href = nav;
}

function emitPointerTrail(x, y, now = performance.now()) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  const dx = lastPointerTrailPoint ? x - lastPointerTrailPoint.x : Infinity;
  const dy = lastPointerTrailPoint ? y - lastPointerTrailPoint.y : Infinity;
  const distance = Math.hypot(dx, dy);

  if (now - lastPointerTrailAt < POINTER_TRAIL_MIN_INTERVAL_MS && distance < POINTER_TRAIL_MIN_DISTANCE) {
    return;
  }

  particleSystem.spawnPointerTrail(x, y);
  lastPointerTrailAt = now;
  lastPointerTrailPoint = { x, y };
}

function emitLandmarkTrail(x, y, now = performance.now()) {
  if (!Number.isFinite(x) || !Number.isFinite(y)) return;
  const dx = lastLandmarkTrail ? x - lastLandmarkTrail.x : Infinity;
  const dy = lastLandmarkTrail ? y - lastLandmarkTrail.y : Infinity;
  const distance = Math.hypot(dx, dy);

  if (lastLandmarkTrail && now - lastLandmarkTrail.at < LANDMARK_TRAIL_MIN_INTERVAL_MS && distance < LANDMARK_TRAIL_MIN_DISTANCE) {
    return;
  }

  particleSystem.spawnPointerTrail(x, y);
  lastLandmarkTrail = { x, y, at: now };
}

function setupPointerEffects() {
  const isFinePointerDevice = window.matchMedia?.("(pointer: fine)").matches ?? true;
  if (!isFinePointerDevice) return;

  window.addEventListener("pointermove", (event) => {
    if (event.pointerType && event.pointerType !== "mouse") return;
    emitPointerTrail(event.clientX, event.clientY);
  }, { passive: true });

  window.addEventListener("pointerdown", (event) => {
    if (event.pointerType && event.pointerType !== "mouse") return;
    particleSystem.spawnPointerBurst(event.clientX, event.clientY);
  }, { passive: true });
}

function startEffectLoop() {
  const tick = () => {
    particleSystem.updateParticles();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// 손 커서가 특정 카드 위에 올라가 있는지 확인하고, 일정 시간 유지되면 선택하는 기능입니다.
function updateHover(x, y, now) {
  const target = findHoverTarget(x, y);
  if (target !== hoverTarget) {
    clearHoverStyles();
    hoverTarget = target;
    hoverStartedAt = target ? now : 0;
  }

  if (!hoverTarget) {
    statusText.textContent = "손 커서를 카드 위에 잠깐 올리면 선택됩니다.";
    return;
  }

  hoverTarget.classList.add("is-hovered");
  const remain = Math.max(0, HOVER_MS - (now - hoverStartedAt));
  const label =
    hoverTarget === backButton
      ? "뒤로가기"
      : hoverTarget.querySelector("strong")?.textContent || "모드";

  statusText.textContent = `${label} 선택까지 ${Math.ceil(remain / 100)}초...`;

  if (now - hoverStartedAt >= HOVER_MS) {
    activateTarget(hoverTarget);
  }
}

function predict() {
  if (!handLandmarker) {
    requestAnimationFrame(predict);
    return;
  }

  const now = performance.now();
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  const cacheFresh = cachedLandmarks && now - cachedLandmarksAt <= LANDMARK_STALE_MS;
  if (cacheFresh) {
    Renderer.drawHand(ctx, cachedLandmarks, canvas, now * 0.001);

    const x = (1 - cachedLandmarks[8].x) * canvas.width;
    const y = cachedLandmarks[8].y * canvas.height;

    handCursor.style.opacity = "1";
    handCursor.style.left = `${x}px`;
    handCursor.style.top = `${y}px`;
    emitLandmarkTrail(x, y, now);
  } else if (cachedLandmarks) {
    cachedLandmarks = null;
    lastLandmarkTrail = null;
    handCursor.style.opacity = "0";
  }

  const hasFreshFrame = video.currentTime !== lastVideoTime && video.readyState >= 2 && video.videoWidth > 0;
  const inferenceDue = now - lastInferenceAt >= INFER_INTERVAL_MS;

  if (hasFreshFrame && inferenceDue) {
    lastVideoTime = video.currentTime;
    lastInferenceAt = now;

    try {
      const result = handLandmarker.detectForVideo(video, now);

      if (result.landmarks.length > 0) {
        const landmarks = result.landmarks[0];
        cachedLandmarks = landmarks;
        cachedLandmarksAt = now;

        const x = (1 - landmarks[8].x) * canvas.width;
        const y = landmarks[8].y * canvas.height;
        emitLandmarkTrail(x, y, now);
        updateHover(x, y, now);
      } else {
        cachedLandmarks = null;
        cachedLandmarksAt = 0;
        lastLandmarkTrail = null;
        handCursor.style.opacity = "0";
        clearHoverStyles();
        hoverTarget = null;
        hoverStartedAt = 0;
        statusText.textContent = "카메라에 손을 보여주세요.";
      }
    } catch (error) {
      console.warn("Theme detection error:", error);
    }
  }

  requestAnimationFrame(predict);
}

async function initMediaPipe() {
  console.info("[Theme MediaPipe] init:start");
  const vision = await FilesetResolver.forVisionTasks(getConfiguredMediaPipeWasmRoot());
  console.info("[Theme MediaPipe] init:vision tasks loaded");
  const modelAssetPath = getConfiguredHandLandmarkerTaskPath();
  const preferredDelegate = parsePreferredDelegate();
  const fallbackDelegate = preferredDelegate === "GPU" ? "CPU" : "GPU";

  try {
    console.info("[Theme MediaPipe] init:create primary", { delegate: preferredDelegate, modelAssetPath });
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: preferredDelegate
      },
      runningMode: "VIDEO",
      numHands: 1
    });
    console.info("[Theme MediaPipe] init:create primary success");
  } catch (delegateError) {
    console.warn(`${preferredDelegate} delegate failed, fallback to ${fallbackDelegate}.`, delegateError);
    console.info("[Theme MediaPipe] init:create fallback", { delegate: fallbackDelegate, modelAssetPath });
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: fallbackDelegate
      },
      runningMode: "VIDEO",
      numHands: 1
    });
    console.info("[Theme MediaPipe] init:create fallback success");
  }
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
      statusText.textContent = "카드를 선택해 숲으로 입장하세요.";
      predict();
    };
  } catch {
    statusText.textContent = "카메라 권한을 허용해 주세요.";
    statusText.style.color = "#ff8080";
  }
}

function bindClickFallback() {
  if (backButton) {
    backButton.onclick = (event) => {
      event.preventDefault();
      activateTarget(backButton);
    };
  }

  cards.forEach((card) => {
    card.onclick = (event) => {
      event.preventDefault();
      activateTarget(card);
    };
  });
}

async function init() {
  setupSeamlessBackgroundLoop();
  setCanvasSize();
  setupPointerEffects();
  startEffectLoop();
  window.addEventListener("resize", setCanvasSize);
  window.addEventListener("beforeunload", () => {
    if (!cameraStream) return;
    cameraStream.getTracks().forEach((track) => {
      track.stop();
    });
  });

  bindClickFallback();

  try {
    await initMediaPipe();
    await initCamera();
  } catch (error) {
    console.error("Theme initialization failed:", error);
    statusText.textContent = "초기화 실패: 새로고침 후 다시 시도해 주세요.";
    statusText.style.color = "#ff8080";
  }
}

init();
