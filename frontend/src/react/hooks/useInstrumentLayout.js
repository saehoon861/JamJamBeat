import { useEffect } from "react";
import {
  DEFAULT_LAYOUT,
  applyInstrumentLayout,
  loadInstrumentLayout,
  readCurrentInstrumentLayout,
  saveInstrumentLayout
} from "../../js/instrument_layout.js";

const INSTRUMENT_LAYOUT_KEY = "jamjam.instrumentLayout.v2";

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getInstrumentElements() {
  return {
    drum: document.getElementById("instrumentDrum"),
    xylophone: document.getElementById("instrumentXylophone"),
    tambourine: document.getElementById("instrumentTambourine"),
    a: document.getElementById("instrumentA"),
    cat: document.getElementById("instrumentCat"),
    penguin: document.getElementById("instrumentPenguin")
  };
}

function isAdminRoute() {
  return new URLSearchParams(window.location.search).get("admin") === "1";
}

export function useInstrumentLayout() {
  useEffect(() => {
    const scene = document.getElementById("scene");
    const landingOverlay = document.getElementById("landingOverlay");
    const statusText = document.getElementById("status");
    const adminControls = document.getElementById("adminControls");
    const adminSaveButton = document.getElementById("adminSaveButton");
    const adminResetButton = document.getElementById("adminResetButton");
    const instrumentElements = getInstrumentElements();

    const adminMode = isAdminRoute();
    const savedLayout = loadInstrumentLayout(INSTRUMENT_LAYOUT_KEY);
    applyInstrumentLayout(adminMode ? DEFAULT_LAYOUT : (savedLayout || DEFAULT_LAYOUT), instrumentElements, clamp);

    if (!adminMode || !scene || !landingOverlay || !statusText) {
      return undefined;
    }

    let dragState = null;
    const pointerDownCleanups = [];
    let saveResetLabelTimer = 0;

    scene.dataset.admin = "on";
    landingOverlay.classList.add("is-hidden");
    statusText.textContent = "관리자 모드: 오브젝트를 드래그하고 저장하세요.";
    adminControls?.classList.add("is-visible");

    const onPointerMove = (event) => {
      if (!dragState) return;

      const width = Math.max(1, window.innerWidth);
      const height = Math.max(1, window.innerHeight);
      const leftPx = event.clientX - dragState.offsetX;
      const topPx = event.clientY - dragState.offsetY;
      const clampedLeftPx = clamp(leftPx, 0, width - dragState.elWidth);
      const clampedTopPx = clamp(topPx, 0, height - dragState.elHeight);
      const elementBottomPx = clampedTopPx + dragState.elHeight;
      const bottomPx = height - elementBottomPx;
      const finalX = clamp((clampedLeftPx / width) * 100, 0, 90);
      const finalY = clamp((bottomPx / height) * 100, 0, 80);

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

    Object.entries(instrumentElements).forEach(([id, el]) => {
      if (!el) return;

      const onPointerDown = (event) => {
        event.preventDefault();
        const rect = el.getBoundingClientRect();
        dragState = {
          id,
          el,
          offsetX: event.clientX - rect.left,
          offsetY: event.clientY - rect.top,
          elWidth: rect.width,
          elHeight: rect.height
        };
        el.classList.add("is-dragging");
        if (el.setPointerCapture) {
          try {
            el.setPointerCapture(event.pointerId);
          } catch {
            // no-op
          }
        }
      };

      el.addEventListener("pointerdown", onPointerDown);
      pointerDownCleanups.push(() => {
        el.removeEventListener("pointerdown", onPointerDown);
      });
    });

    window.addEventListener("pointermove", onPointerMove);
    window.addEventListener("pointerup", onPointerUp);
    window.addEventListener("pointercancel", onPointerUp);

    if (adminSaveButton) {
      adminSaveButton.onclick = () => {
        const layout = readCurrentInstrumentLayout(instrumentElements, clamp);
        saveInstrumentLayout(INSTRUMENT_LAYOUT_KEY, layout);
        adminSaveButton.textContent = "저장됨";
        window.clearTimeout(saveResetLabelTimer);
        saveResetLabelTimer = window.setTimeout(() => {
          adminSaveButton.textContent = "배치 저장";
        }, 900);
      };
    }

    if (adminResetButton) {
      adminResetButton.onclick = () => {
        localStorage.removeItem(INSTRUMENT_LAYOUT_KEY);
        applyInstrumentLayout(DEFAULT_LAYOUT, instrumentElements, clamp);
      };
    }

    return () => {
      pointerDownCleanups.forEach((cleanup) => cleanup());
      window.removeEventListener("pointermove", onPointerMove);
      window.removeEventListener("pointerup", onPointerUp);
      window.removeEventListener("pointercancel", onPointerUp);
      window.clearTimeout(saveResetLabelTimer);
      if (adminSaveButton) {
        adminSaveButton.onclick = null;
      }
      if (adminResetButton) {
        adminResetButton.onclick = null;
      }
    };
  }, []);
}
