import { FilesetResolver, HandLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/+esm";
import * as Renderer from "./renderer.js";

const video = document.getElementById("webcam");
const canvas = document.getElementById("handCanvas");
const ctx = canvas.getContext("2d");
const handCursor = document.getElementById("handCursor");
const statusText = document.getElementById("themeStatus");
const backButton = document.getElementById("themeBackButton");
const cards = Array.from(document.querySelectorAll(".theme-mode-card"));

const HOVER_MS = 520;
const HOVER_PADDING = 28;
const BG_VIDEO_CROSSFADE_SEC = 0.42;

let handLandmarker;
let lastVideoTime = -1;
let hoverTarget = null;
let hoverStartedAt = 0;
let cameraStream = null;

function setCanvasSize() {
  canvas.width = window.innerWidth;
  canvas.height = window.innerHeight;
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
  cards.forEach((card) => card.classList.remove("is-hovered"));
}

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

  if (video.currentTime !== lastVideoTime && video.readyState >= 2 && video.videoWidth > 0) {
    lastVideoTime = video.currentTime;
    const now = performance.now();

    try {
      const result = handLandmarker.detectForVideo(video, now);
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      if (result.landmarks.length > 0) {
        const landmarks = result.landmarks[0];
        Renderer.drawHand(ctx, landmarks, canvas, now * 0.001);

        const x = (1 - landmarks[8].x) * canvas.width;
        const y = landmarks[8].y * canvas.height;

        handCursor.style.opacity = "1";
        handCursor.style.left = `${x}px`;
        handCursor.style.top = `${y}px`;

        updateHover(x, y, now);
      } else {
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
  window.addEventListener("resize", setCanvasSize);
  window.addEventListener("beforeunload", () => {
    if (!cameraStream) return;
    cameraStream.getTracks().forEach((track) => track.stop());
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
