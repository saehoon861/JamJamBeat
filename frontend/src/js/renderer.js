// [renderer.js] 손 위치를 따라다니는 나비를 그리는 렌더러입니다.
// 제스처에 따라 나비의 색상, 날개짓 속도, 특수 효과가 달라집니다.
// Note: Lottie 기능은 비활성화됨 (CSS 애니메이션 사용)

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

// ─── Lottie Setup (비활성화됨) ────────────────────────────────────────
// Lottie 애니메이션은 CSS 기반 애니메이션으로 대체되었습니다.
// const lottieContainer = document.createElement('div');
// const anim = lottie.loadAnimation({...});
// 위 코드는 더 이상 사용하지 않습니다.

// ─── 제스처별 나비 스타일 정의 (drawHand 비활성화로 현재 미사용) ───────────────────────────
// const GESTURE_CONFIG = { ... };  // drawHand 재활성화 시 복원

// ─── drawHand: 메인 렌더링 진입점 ───────────────────────
export function drawHand(ctx, landmarks, canvas, t, smoothingKey = "default", currentGesture = "None") {
    return;
}

export function clearHandSmoothing(smoothingKey) {
    smoothedHandLandmarksByKey.delete(smoothingKey);
    smoothedRenderScaleByKey.delete(smoothingKey);
}

export function setHandAnimationActive(_active) {
    // Lottie 비활성화됨 - CSS 애니메이션 사용 중
    // fancy 모드 진입 시 lottieActive/anim 미정의 오류 방지
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
