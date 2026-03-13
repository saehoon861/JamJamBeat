// [renderer.js] 화면에 예쁜 손과 반짝이는 효과를 그리는 '화가' 역할의 파일입니다.
// 손이 너무 떨리지 않게 부드럽게 보정해주고, 젤리처럼 말랑말랑한 디자인을 입혀줍니다.
// 캔버스 렌더링 및 시각 효과를 담당하는 모듈입니다.
// "어두운색의 살집 있는 말랑말랑한 손" 디자인이 포함되어 있습니다.

const smoothedHandLandmarksByKey = new Map();
const smoothedRenderScaleByKey = new Map();

const TARGET_PALM_PX_RATIO = 0.14;
const MIN_TARGET_PALM_PX = 84;
const MAX_TARGET_PALM_PX = 150;
const MIN_RENDER_SCALE = 0.85;
const MAX_RENDER_SCALE = 2.6;

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

// 손의 떨림을 줄이기 위해 이전 좌표와 현재 좌표를 보간(Interpolation)하는 함수
// 손이 떨리는 것을 방지하기 위해, 이전 위치와 현재 위치를 자연스럽게 이어주는 '부드러운 보정' 기능입니다.
export function getSmoothedLandmarks(rawLandmarks, smoothingKey = "default") {
    const previous = smoothedHandLandmarksByKey.get(smoothingKey) || null;
    if (!previous || previous.length !== rawLandmarks.length) {
        const seeded = rawLandmarks.map((p) => ({ x: p.x, y: p.y, z: p.z ?? 0 }));
        smoothedHandLandmarksByKey.set(smoothingKey, seeded);
        return seeded;
    }

    const smoothing = 0.68; // 값이 클수록 손 움직임을 더 빠르게 따라갑니다.
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

// "어두운색의 살집 있는 손"을 그리는 핵심 함수
// 화면에 실제로 손 모양을 그리는 핵심 기능입니다. 색깔이나 그림자 등을 여기서 입힙니다.
export function drawHand(ctx, landmarks, canvas, t, smoothingKey = "default") {
    ctx.save();
    ctx.globalAlpha = 0.92;
    const stable = getSmoothedLandmarks(landmarks, smoothingKey);
    const toCanvasX = (x) => (1 - x) * canvas.width; // 좌우 반전 처리
    const toCanvasY = (y) => y * canvas.height;

    const rawPoint = (idx) => ({ x: toCanvasX(stable[idx].x), y: toCanvasY(stable[idx].y) });
    const rawPalmPoints = [17, 13, 9, 5, 1, 0].map(rawPoint);

    let cx = 0, cy = 0;
    rawPalmPoints.forEach((p) => { cx += p.x; cy += p.y; });
    cx /= rawPalmPoints.length; cy /= rawPalmPoints.length;

    const renderScale = computeNormalizedRenderScale(stable, canvas, toCanvasX, toCanvasY, smoothingKey);

    // 특정 랜드마크의 픽셀 좌표를 반환하는 헬퍼
    const point = (idx) => {
        const p = rawPoint(idx);
        return {
            x: cx + (p.x - cx) * renderScale,
            y: cy + (p.y - cy) * renderScale
        };
    };

    // 손바닥 영역을 구성하는 포인트들
    const palmIndices = [17, 13, 9, 5, 1, 0];
    const palmPoints = palmIndices.map(point);

    // 손바닥 중심 계산
    cx = 0; cy = 0;
    palmPoints.forEach((p) => { cx += p.x; cy += p.y; });
    cx /= palmPoints.length; cy /= palmPoints.length;

    // --- 내부 유틸리티 그리기 함수 ---

    // 부드러운 곡선으로 연결된 경로 생성 (손바닥용)
    const drawPalmPath = () => {
        palmPoints.forEach((p, i) => {
            const next = palmPoints[(i + 1) % palmPoints.length];
            const mx = (p.x + next.x) / 2;
            const my = (p.y + next.y) / 2;
            if (i === 0) ctx.moveTo(mx, my);
            else ctx.quadraticCurveTo(p.x, p.y, mx, my);
        });
        ctx.closePath();
    };

    // 입체적인 관절 그리기 (그라데이션 적용)
    const drawJoint = (x, y, r, core, edge) => {
        const g = ctx.createRadialGradient(x - r * 0.34, y - r * 0.36, r * 0.22, x, y, r);
        g.addColorStop(0, core);
        g.addColorStop(1, edge);
        ctx.shadowColor = "rgba(255, 186, 218, 0.5)";
        ctx.shadowBlur = r * 0.7;
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.shadowBlur = 0;
    };

    // 손가락 마디(캡슐 형태) 그리기
    const drawCapsule = (a, b, radius, color, shadowColor) => {
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const len = Math.max(1, Math.hypot(dx, dy));
        const nx = -dy / len;
        const ny = dx / len;

        ctx.lineCap = "round";
        ctx.lineJoin = "round";

        // 아래쪽 그림자 효과
        ctx.strokeStyle = shadowColor;
        ctx.lineWidth = radius * 1.32;
        ctx.beginPath();
        ctx.moveTo(a.x + 2, a.y + 3);
        ctx.lineTo(b.x + 2, b.y + 3);
        ctx.stroke();

        // 메인 살집 색상 (어두운 회색/차콜)
        ctx.strokeStyle = color;
        ctx.lineWidth = radius * 1.08;
        ctx.beginPath();
        ctx.moveTo(a.x, a.y);
        ctx.lineTo(b.x, b.y);
        ctx.stroke();

        // 위쪽 하이라이트 (Glossy 효과)
        ctx.strokeStyle = "rgba(255,255,255,0.3)";
        ctx.lineWidth = radius * 0.42;
        ctx.beginPath();
        ctx.moveTo(a.x - nx * radius * 0.2, a.y - ny * radius * 0.2);
        ctx.lineTo(b.x - nx * radius * 0.2, b.y - ny * radius * 0.2);
        ctx.stroke();
    };

    // 1. 손바닥 그리기 (심장 박동처럼 미세하게 커졌다가 작아졌다가 하는 효과를 줍니다)
    const pulse = 1 + Math.sin(t * 5) * 0.015;
    ctx.save();
    ctx.translate(cx, cy);
    ctx.scale(pulse, pulse);
    ctx.translate(-cx, -cy);

    // 손바닥 그림자
    const palmShadow = ctx.createRadialGradient(cx + 10, cy + 12, 10, cx + 10, cy + 12, 100);
    palmShadow.addColorStop(0, "rgba(35, 45, 48, 0.28)");
    palmShadow.addColorStop(1, "rgba(35, 45, 48, 0)");
    ctx.fillStyle = palmShadow;
    ctx.beginPath();
    drawPalmPath();
    ctx.fill();

    // 손바닥 본체 (어두운 차콜 그레이 입체 그라데이션)
    const palmGrad = ctx.createRadialGradient(cx - 20, cy - 20, 15, cx, cy + 15, 110);
    palmGrad.addColorStop(0, "#ffe9f4");
    palmGrad.addColorStop(0.45, "#ffd4e8");
    palmGrad.addColorStop(1, "#f3a8ca");
    ctx.fillStyle = palmGrad;
    ctx.beginPath();
    drawPalmPath();
    ctx.fill();

    // 손바닥 상단 하이라이트 Rim
    ctx.strokeStyle = "rgba(255,255,255,0.62)";
    ctx.lineWidth = 5.4;
    ctx.beginPath();
    ctx.arc(cx - 15, cy - 18, 30, Math.PI * 1.0, Math.PI * 1.9);
    ctx.stroke();
    ctx.restore();

    // 2. 손가락 그리기
    const fingers = [
        { joints: [1, 2, 3, 4], color: "#ffd8ee", shadow: "#d988b5", size: 15.6 },
        { joints: [5, 6, 7, 8], color: "#ffd0ea", shadow: "#d581ae", size: 15.2 },
        { joints: [9, 10, 11, 12], color: "#ffcee8", shadow: "#cf78a7", size: 16.2 },
        { joints: [13, 14, 15, 16], color: "#ffc8e5", shadow: "#c96f9f", size: 15.1 },
        { joints: [17, 18, 19, 20], color: "#ffc2df", shadow: "#bf6596", size: 14.4 }
    ];

    fingers.forEach((finger, fIdx) => {
        const sway = Math.sin(t * 6 + fIdx * 0.9) * 1.5;
        const pts = finger.joints.map((idx, i) => {
            const p = point(idx);
            return { x: p.x, y: p.y + sway * (i / finger.joints.length) };
        });

        // 마디 연결 그리기 (끝으로 갈수록 얇아지는 Tapering 적용)
        for (let i = 1; i < pts.length; i++) {
            const r = Math.max(7.2, finger.size - i * 1.95);
            drawCapsule(pts[i - 1], pts[i], r, finger.color, finger.shadow);
        }

        // 관절 노드 그리기 (젤리 같은 느낌이 나도록 동그랗게 그립니다)
        pts.forEach((p, i) => {
            const r = Math.max(7.0, finger.size - i * 1.9);
            drawJoint(p.x, p.y, r, "rgba(255, 245, 252, 0.76)", finger.color);
        });
    });

    // 3. 손가락 끝 포인트 효과 (어두운 테마에 어울리는 은은한 빛)
    const tipColors = ["#ffc5e7", "#ffbbe1", "#ffc0e4", "#ffb2db", "#ffa9d6"];
    [4, 8, 12, 16, 20].forEach((idx, i) => {
        const p = point(idx);
        const r = 9 + Math.sin(t * 8 + i) * 1.2;
        drawJoint(p.x, p.y, r, "rgba(255,255,255,0.88)", tipColors[i]);
        // 상단 반짝임
        drawJoint(p.x - r * 0.3, p.y - r * 0.3, r * 0.42, "rgba(255,255,255,0.74)", "rgba(255,255,255,0)");
    });

    ctx.shadowBlur = 0;
    ctx.restore();
}

export function clearHandSmoothing(smoothingKey) {
    smoothedHandLandmarksByKey.delete(smoothingKey);
    smoothedRenderScaleByKey.delete(smoothingKey);
}

// 떠다니는 음표 효과 생성 (audio.js와 함께 사용)
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
