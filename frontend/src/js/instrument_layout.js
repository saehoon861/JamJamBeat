// [instrument_layout.js] 악기 배치를 읽고 저장하고, 관리자 드래그 편집까지 맡는 모듈입니다.

const INSTRUMENT_IDS = ["a", "drum", "xylophone", "tambourine"];

// 저장된 배치가 없을 때 사용할 기본 위치값입니다.
export const DEFAULT_LAYOUT = {
  drum: { x: 10, y: 14 },
  xylophone: { x: 38, y: 22 },
  tambourine: { x: 68, y: 18 },
  a: { x: 52, y: 10 }
};

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

    target.style.left = `${clamp(x, 0, 90)}vw`;
    target.style.bottom = `${clamp(y, 0, 80)}vh`;
    target.style.right = "auto";
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
    layout[id] = {
      x: clamp((rect.left / width) * 100, 0, 90),
      y: clamp(((height - rect.bottom) / height) * 100, 0, 80)
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
    const rect = dragState.el.getBoundingClientRect();

    // 클릭했던 위치 차이를 유지한 채 자연스럽게 따라오게 계산합니다.
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
