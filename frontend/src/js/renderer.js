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

const AVATAR_IMAGES = {
    "Fist": new Image(),        // drum (hedgehog)
    "OpenPalm": new Image(),    // xylophone (lily)
    "V": new Image(),           // tambourine (clover)
    "Pinky": new Image(),       // owl (squirrel)
    "Animal": new Image(),      // fern (magic fern)
    "KHeart": new Image()       // fern (magic fern)
};

// Initiate image loading
AVATAR_IMAGES["Fist"].src = "/assets/objects/hedgehog-drum.png";
AVATAR_IMAGES["OpenPalm"].src = "/assets/objects/lily-melody.png";
AVATAR_IMAGES["V"].src = "/assets/objects/clover-chime.png";
AVATAR_IMAGES["Pinky"].src = "/assets/objects/squirrel-effect.png";
AVATAR_IMAGES["Animal"].src = "/assets/objects/magic-fern.png";
AVATAR_IMAGES["KHeart"].src = "/assets/objects/magic-fern.png";


// "살아있는 오브젝트 기반 아바타"를 그리는 핵심 함수
// 화면에 실제로 손 모양 대신 캐릭터/식물 오브젝트를 그립니다. 제스처에 따라 다른 모양이 나옵니다.
export function drawHand(ctx, landmarks, canvas, t, smoothingKey = "default", currentGesture = "None") {
    ctx.save();
    ctx.globalAlpha = 0.96;
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

    // ---------------------------------------------------------
    // 살아있는 오브젝트 아바타(Living Object Avatar) 그리기
    // ---------------------------------------------------------

    // 제스처 라벨에 해당하는 아바타 이미지 선택
    const avatarImg = AVATAR_IMAGES[currentGesture];

    if (avatarImg && avatarImg.complete && avatarImg.naturalWidth > 0 && currentGesture !== "None") {
        // 제스처가 인식된 경우: 해당 오브젝트 이미지를 렌더링
        const wrist = point(0);
        const middle = point(9);
        const handDirX = middle.x - wrist.x;
        const handDirY = middle.y - wrist.y;
        
        // 손목에서 중지 첫 번째 마디를 향하는 방향으로 각도 계산
        let angle = Math.atan2(handDirY, handDirX) + Math.PI / 2;

        const pulse = 1 + Math.sin(t * 8.0) * 0.05; // 콩닥콩닥 숨쉬는 듯한 박동
        const baseSize = 140 * renderScale; // 손바닥 크기에 비례하여 아바타 크기 결정
        
        // 속도에 따른 찌그러짐(Squash and Stretch) 효과
        // 이전 손목 위치를 비교하여 속도를 유추할 수 있으나, 일단 시간에 따라 살짝 흔들거리게 적용
        const swayX = Math.sin(t * 12) * 0.04;
        const scaleX = pulse + swayX;
        const scaleY = pulse - swayX;

        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(angle * 0.2); // 손의 회전을 완전히 따라가면 부자연스러울 수 있으므로 약간만 반영
        ctx.scale(scaleX, scaleY);
        
        ctx.shadowColor = "rgba(0, 0, 0, 0.45)";
        ctx.shadowBlur = 20;

        // 중앙부를 기준으로 이미지를 그림
        ctx.drawImage(avatarImg, -baseSize / 2, -baseSize / 2, baseSize, baseSize);

        ctx.restore();
    } else {
        // 제스처가 "None"이거나 아직 이미지가 없는 경우: 작은 마법 씨앗(Magic Seed) 그리기 (손바닥 중심 위치)
        const pulse = 1 + Math.sin(t * 5.0) * 0.1;
        ctx.save();
        ctx.translate(cx, cy);
        ctx.scale(pulse, pulse);

        const seedGrad = ctx.createRadialGradient(0, -2, 0, 0, 0, 16);
        seedGrad.addColorStop(0, "rgba(255, 255, 255, 1)");
        seedGrad.addColorStop(0.3, "rgba(220, 255, 200, 0.8)");
        seedGrad.addColorStop(1, "rgba(100, 200, 100, 0)");

        ctx.fillStyle = seedGrad;
        ctx.shadowColor = "rgba(100, 255, 100, 0.6)";
        ctx.shadowBlur = 15;
        
        ctx.beginPath();
        ctx.arc(0, 0, 16, 0, Math.PI * 2);
        ctx.fill();

        ctx.restore();
    }

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
