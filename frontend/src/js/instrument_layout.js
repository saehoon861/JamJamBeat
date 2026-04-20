// [instrument_layout.js] 악기 배치를 읽고 저장하고, 관리자 드래그 편집까지 맡는 모듈입니다.

const INSTRUMENT_IDS = ["a", "drum", "xylophone", "tambourine", "cat", "penguin"];

// 저장된 배치가 없을 때 사용할 기본 위치값입니다.
export const DEFAULT_LAYOUT = {
  drum: { x: 30, y: 4 },
  xylophone: { x: 43, y: 5 },
  tambourine: { x: 56, y: 4 },
  a: { x: 39, y: 0 },
  cat: { x: 69, y: 0 },
  penguin: { x: 12, y: 2 }
};

function getElementSize(el) {
  if (!el) return { width: 0, height: 0 };
  const rect = el.getBoundingClientRect();
  return {
    width: Math.max(el.offsetWidth || 0, rect.width || 0),
    height: Math.max(el.offsetHeight || 0, rect.height || 0)
  };
}

function getLayoutBounds(el) {
  const viewportWidth = Math.max(1, window.innerWidth);
  const viewportHeight = Math.max(1, window.innerHeight);
  const { width, height } = getElementSize(el);
  const maxX = Math.min(90, Math.max(0, 100 - (width / viewportWidth) * 100));
  const maxY = Math.min(80, Math.max(0, 100 - (height / viewportHeight) * 100));
  return { maxX, maxY };
}

function normalizeLegacyLayout(layout) {
  if (!layout || typeof layout !== "object") return null;
  return {
    ...layout,
    a: layout.a || layout.a2 || layout.fern || layout.owl
  };
}

// localStorage 에 저장된 악기 배치를 읽어옵니다.
export function loadInstrumentLayout(storageKey) {
  try {
    const raw = localStorage.getItem(storageKey);
    if (!raw) return null;
    const parsed = JSON.parse(raw);
    if (!parsed || typeof parsed !== "object") return null;
    return normalizeLegacyLayout(parsed);
  } catch {
    return null;
  }
}

// 현재 배치 정보를 localStorage 에 저장합니다.
export function saveInstrumentLayout(storageKey, layout) {
  try {
    localStorage.setItem(storageKey, JSON.stringify(layout));
  } catch {
    // 저장 실패가 메인 동작을 막지 않도록 무시합니다.
  }
}

// 배치 데이터에 맞춰 실제 DOM 요소의 위치를 화면에 적용합니다.
export function applyInstrumentLayout(layout, instrumentElements, clamp) {
  if (!layout || typeof layout !== "object") return;
  const appliedTargets = new Set();

  INSTRUMENT_IDS.forEach((id) => {
    const target = instrumentElements[id];
    const pos = layout[id];
    if (!target || !pos || typeof pos !== "object") return;
    if (appliedTargets.has(target)) return;

    const x = Number(pos.x);
    const y = Number(pos.y);
    if (!Number.isFinite(x) || !Number.isFinite(y)) return;
    const { maxX, maxY } = getLayoutBounds(target);

    target.style.left = `${clamp(x, 0, maxX)}vw`;
    target.style.bottom = `${clamp(y, 0, maxY)}vh`;
    target.style.right = "auto";
    target.style.top = "auto";
    appliedTargets.add(target);
  });
}

// 현재 화면에 보이는 악기 위치를 다시 읽어서 "저장 가능한 형태"로 바꿉니다.
export function readCurrentInstrumentLayout(instrumentElements, clamp) {
  const layout = {};
  const width = Math.max(1, window.innerWidth);
  const height = Math.max(1, window.innerHeight);
  const seenTargets = new Set();

  INSTRUMENT_IDS.forEach((id) => {
    const el = instrumentElements[id];
    if (!el) return;
    if (seenTargets.has(el)) return;
    const rect = el.getBoundingClientRect();
    const { maxX, maxY } = getLayoutBounds(el);
    layout[id] = {
      x: clamp((rect.left / width) * 100, 0, maxX),
      y: clamp(((height - rect.bottom) / height) * 100, 0, maxY)
    };
    seenTargets.add(el);
  });

  return layout;
}

// 관리자 모드에서 드래그 편집 기능을 켜는 함수입니다.
// 저장/초기화 버튼 이벤트도 여기서 같이 연결합니다.
export function setupAdminDragMode({
  scene,
  landingOverlay,
  statusText,
  adminControls,
  adminSaveButton,
  adminResetButton,
  instrumentElements,
  storageKey,
  defaultLayout,
  clamp,
  onAdminModeEnabled
}) {
  // dragState 는 "지금 무엇을 잡고 움직이는지" 기록하는 임시 저장소입니다.
  let dragState = null;
  const boundElements = new Set();

  if (typeof onAdminModeEnabled === "function") {
    onAdminModeEnabled();
  }

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

    // 마우스 위치에서 오프셋을 빼서 요소의 왼쪽 상단 위치를 계산합니다.
    const leftPx = event.clientX - dragState.offsetX;
    const topPx = event.clientY - dragState.offsetY;

    // 화면 밖으로 나가지 않도록 제한합니다.
    const clampedLeftPx = clamp(leftPx, 0, width - dragState.elWidth);
    const clampedTopPx = clamp(topPx, 0, height - dragState.elHeight);

    // bottom CSS 속성은 "요소의 하단"이 "화면 하단"으로부터 떨어진 거리
    // 요소 하단 위치 = 요소 상단 + 요소 높이
    const elementBottomPx = clampedTopPx + dragState.elHeight;
    // 화면 하단에서 요소 하단까지의 거리
    const bottomPx = height - elementBottomPx;

    // 픽셀을 vw, vh로 변환합니다.
    const x = (clampedLeftPx / width) * 100;
    const y = (bottomPx / height) * 100;

    // 최종적으로 범위 제한을 적용합니다.
    const finalX = clamp(x, 0, dragState.maxX);
    const finalY = clamp(y, 0, dragState.maxY);

    dragState.el.style.left = `${finalX}vw`;
    dragState.el.style.bottom = `${finalY}vh`;
    dragState.el.style.right = "auto";
    dragState.el.style.top = "auto";
    dragState.el.style.transform = "none";
  };

  const onPointerUp = () => {
    if (!dragState) return;
    dragState.el.classList.remove("is-dragging");
    dragState = null;
  };

  INSTRUMENT_IDS.forEach((id) => {
    const el = instrumentElements[id];
    if (!el) return;
    if (boundElements.has(el)) return;

    // pointerdown 은 마우스/터치/펜 입력을 비교적 통합해서 받을 수 있는 이벤트입니다.
    el.addEventListener("pointerdown", (event) => {
      event.preventDefault();
      const rect = el.getBoundingClientRect();
      dragState = {
        id,
        el,
        offsetX: event.clientX - rect.left,
        offsetY: event.clientY - rect.top,
        elWidth: rect.width,
        elHeight: rect.height,
        ...getLayoutBounds(el)
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
    boundElements.add(el);
  });

  window.addEventListener("pointermove", onPointerMove);
  window.addEventListener("pointerup", onPointerUp);
  window.addEventListener("pointercancel", onPointerUp);

  if (adminSaveButton) {
    adminSaveButton.onclick = () => {
      // 현재 위치를 읽어서 저장합니다.
      const layout = readCurrentInstrumentLayout(instrumentElements, clamp);
      saveInstrumentLayout(storageKey, layout);
      adminSaveButton.textContent = "저장됨";
      window.setTimeout(() => {
        adminSaveButton.textContent = "배치 저장";
      }, 900);
    };
  }

  if (adminResetButton) {
    adminResetButton.onclick = () => {
      // 저장된 사용자 배치를 지우고 기본값으로 되돌립니다.
      localStorage.removeItem(storageKey);
      applyInstrumentLayout(defaultLayout, instrumentElements, clamp);
    };
  }
}
