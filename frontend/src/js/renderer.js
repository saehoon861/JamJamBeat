import lottie from "lottie-web";

// [renderer.js] 손 위치를 따라다니는 나비를 그리는 렌더러입니다.
// 제스처에 따라 나비의 색상, 날개짓 속도, 특수 효과가 달라집니다.

const smoothedHandLandmarksByKey = new Map();
const smoothedRenderScaleByKey = new Map();

const TARGET_PALM_PX_RATIO = 0.14;
const MIN_TARGET_PALM_PX = 84;
const MAX_TARGET_PALM_PX = 150;
const MIN_RENDER_SCALE = 0.85;
const MAX_RENDER_SCALE = 2.6;
const DEFAULT_RENDER_MODE = "simple";
const RENDER_MODE = (() => {
    const raw = new URLSearchParams(window.location.search).get("handRenderer");
    if (!raw) return DEFAULT_RENDER_MODE;
    return String(raw).trim().toLowerCase() === "fancy" ? "fancy" : DEFAULT_RENDER_MODE;
})();

function clampNumber(value, min, max) {
    return Math.min(max, Math.max(min, value));
}

function computeNormalizedRenderScale(stable, canvas, toCanvasX, toCanvasY, smoothingKey = "default") {
    const wrist = stable?.[0];
    const middleMcp = stable?.[9];
    const previousScale = smoothedRenderScaleByKey.get(smoothingKey) ?? 1;
    if (!wrist || !middleMcp) return previousScale;

    const wx = toCanvasX(wrist.x);
    const wy = toCanvasY(wrist.y);
    const mx = toCanvasX(middleMcp.x);
    const my = toCanvasY(middleMcp.y);

    const measuredPalmPx = Math.max(1, Math.hypot(wx - mx, wy - my));
    const targetPalmPx = clampNumber(
        Math.min(canvas.width, canvas.height) * TARGET_PALM_PX_RATIO,
        MIN_TARGET_PALM_PX,
        MAX_TARGET_PALM_PX
    );

    const instantScale = clampNumber(targetPalmPx / measuredPalmPx, MIN_RENDER_SCALE, MAX_RENDER_SCALE);
    const nextScale = previousScale + (instantScale - previousScale) * 0.22;
    smoothedRenderScaleByKey.set(smoothingKey, nextScale);

    return nextScale;
}

export function getSmoothedLandmarks(rawLandmarks, smoothingKey = "default") {
    const previous = smoothedHandLandmarksByKey.get(smoothingKey) || null;
    if (!previous || previous.length !== rawLandmarks.length) {
        const seeded = rawLandmarks.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0 }));
        smoothedHandLandmarksByKey.set(smoothingKey, seeded);
        return seeded;
    }

    const smoothing = 0.68;
    const nextSmoothed = previous.map((prev, i) => {
        const next = rawLandmarks[i];
        return {
            x: prev.x + (next.x - prev.x) * smoothing,
            y: prev.y + (next.y - prev.y) * smoothing,
            z: (prev.z ?? 0) + ((next.z ?? 0) - (prev.z ?? 0)) * smoothing,
        };
    });
    smoothedHandLandmarksByKey.set(smoothingKey, nextSmoothed);

    return nextSmoothed;
}

// ─── Lottie Setup ────────────────────────────────────────
const lottieContainer = document.createElement('div');
lottieContainer.style.width = '200px';
lottieContainer.style.height = '200px';
let lottieActive = false;
let currentAnimationSpeed = null;

const anim = lottie.loadAnimation({
    container: lottieContainer,
    renderer: 'canvas',
    loop: true,
    autoplay: false,
    path: '/assets/butterfly.json',
    rendererSettings: {
        clearCanvas: true
    }
});

let cachedLottieCanvas = null;
anim.addEventListener('DOMLoaded', () => {
    cachedLottieCanvas = lottieContainer.querySelector('canvas');
    if (RENDER_MODE !== "fancy") {
        anim.goToAndStop(0, true);
    }
});

// 틴트 처리를 위한 임시 캔버스
const tintCanvas = document.createElement('canvas');
tintCanvas.width = 200;
tintCanvas.height = 200;
const tintCtx = tintCanvas.getContext('2d');

// ─── 제스처별 나비 스타일 정의 ───────────────────────────
const GESTURE_CONFIG = {
    Fist: { speed: 0.5, tint: "rgba(255, 165, 0, 0.4)", glow: "rgba(255, 165, 0, 0.6)" }, // orange
    OpenPalm: { speed: 1.0, tint: "rgba(0, 128, 0, 0.4)", glow: "rgba(0, 128, 0, 0.6)" }, // green
    V: { speed: 2.0, tint: "rgba(255, 255, 0, 0.4)", glow: "rgba(255, 255, 0, 0.6)" }, // yellow
    Pinky: { speed: 3.0, tint: "rgba(0, 255, 0, 0.4)", glow: "rgba(0, 255, 0, 0.6)" }, // lime
    Animal: { speed: 2.5, tint: "rgba(128, 0, 128, 0.4)", glow: "rgba(128, 0, 128, 0.6)" }, // purple
    KHeart: { speed: 1.5, tint: "rgba(255, 192, 203, 0.4)", glow: "rgba(255, 192, 203, 0.6)" }, // pink
    None: { speed: 0.8, tint: null, glow: "rgba(200, 182, 255, 0.4)" }
};

// ─── drawHand: 메인 렌더링 진입점 ───────────────────────
export function drawHand(ctx, landmarks, canvas, t, smoothingKey = "default", currentGesture = "None") {
    return;
}

export function clearHandSmoothing(smoothingKey) {
    smoothedHandLandmarksByKey.delete(smoothingKey);
    smoothedRenderScaleByKey.delete(smoothingKey);
}

export function setHandAnimationActive(active) {
    if (RENDER_MODE !== "fancy") return;
    if (active && !lottieActive) {
        anim.play();
        lottieActive = true;
        return;
    }

    if (!active && lottieActive) {
        anim.pause();
        lottieActive = false;
    }
}

export function createFloatingNote(container) {
    if (!container) return;
    const notes = ["♪", "♫", "♬", "🎵", "🎶"];
    const note = document.createElement("span");
    note.className = "floating-note";
    note.textContent = notes[Math.floor(Math.random() * notes.length)];
    note.style.left = `${Math.random() * 90 + 5}%`;
    note.style.animationDuration = `${Math.random() * 3 + 4}s`;
    note.style.fontSize = `${Math.random() * 1.5 + 1}rem`;
    note.style.opacity = `${Math.random() * 0.4 + 0.3}`;
    container.appendChild(note);
    note.addEventListener("animationend", () => note.remove());
}
