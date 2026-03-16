// [instrument_animations.js] 악기별 Lottie 애니메이션을 관리하는 모듈입니다.
// Idle, Hover, Trigger 상태를 지원하며 손 추적과 연동됩니다.

import lottie from "lottie-web";

// ─── 애니메이션 상태 정의 ───────────────────────────────
const ANIMATION_STATES = {
  IDLE: "idle",           // 기본 대기 상태
  HOVER: "hover",         // 손이 근처에 있을 때
  TRIGGER: "trigger",     // 악기가 연주될 때
  FEVER: "fever"          // 피버 모드일 때
};

// ─── 악기별 애니메이션 설정 ──────────────────────────────
const INSTRUMENT_ANIMATION_CONFIG = {
  drum: {
    path: "/assets/animations/drum.json",
    fallbackEmoji: "🦔",
    states: {
      idle: { loop: true, speed: 0.8, segments: null },
      hover: { loop: true, speed: 1.2, segments: null },
      trigger: { loop: false, speed: 1.5, segments: null },
      fever: { loop: true, speed: 2.0, segments: null }
    }
  },
  xylophone: {
    path: "/assets/animations/xylophone.json",
    fallbackEmoji: "🌿",
    states: {
      idle: { loop: true, speed: 0.6, segments: null },
      hover: { loop: true, speed: 1.0, segments: null },
      trigger: { loop: false, speed: 1.8, segments: null },
      fever: { loop: true, speed: 2.2, segments: null }
    }
  },
  tambourine: {
    path: "/assets/animations/tambourine.json",
    fallbackEmoji: "🌸",
    states: {
      idle: { loop: true, speed: 0.7, segments: null },
      hover: { loop: true, speed: 1.1, segments: null },
      trigger: { loop: false, speed: 1.6, segments: null },
      fever: { loop: true, speed: 2.0, segments: null }
    }
  },
  a: {
    path: "/assets/animations/a.json",
    fallbackEmoji: "🦌",
    states: {
      idle: { loop: true, speed: 0.9, segments: null },
      hover: { loop: true, speed: 1.3, segments: null },
      trigger: { loop: false, speed: 1.7, segments: null },
      fever: { loop: true, speed: 2.3, segments: null }
    }
  }
};

// ─── 애니메이션 인스턴스 매니저 ──────────────────────────
export function createInstrumentAnimationManager() {
  const instances = new Map(); // 악기 ID -> 애니메이션 인스턴스
  const containers = new Map(); // 악기 ID -> 컨테이너 DOM
  const states = new Map(); // 악기 ID -> 현재 상태
  const loadedAnimations = new Set(); // 로드 성공한 애니메이션

  /**
   * 악기에 Lottie 애니메이션 초기화
   * @param {string} instrumentId - 악기 ID (drum, xylophone, etc.)
   * @param {HTMLElement} targetElement - 애니메이션을 표시할 DOM 요소
   */
  function initAnimation(instrumentId, targetElement) {
    if (!targetElement) {
      console.warn(`[InstrumentAnimation] Target element not found for ${instrumentId}`);
      return false;
    }

    const config = INSTRUMENT_ANIMATION_CONFIG[instrumentId];
    if (!config) {
      console.warn(`[InstrumentAnimation] No config for ${instrumentId}`);
      return false;
    }

    // 기존 컨테이너 정리
    if (containers.has(instrumentId)) {
      const oldContainer = containers.get(instrumentId);
      if (oldContainer && oldContainer.parentNode) {
        oldContainer.parentNode.removeChild(oldContainer);
      }
    }

    // 애니메이션 컨테이너 생성
    const container = document.createElement("div");
    container.className = "instrument-lottie-container";
    container.style.position = "absolute";
    container.style.inset = "0";
    container.style.width = "100%";
    container.style.height = "100%";
    container.style.pointerEvents = "none";
    container.style.zIndex = "10"; // 이미지 위에 렌더링

    // 기존 이미지를 숨기거나 유지할지 선택
    const existingImage = targetElement.querySelector(".instrument-art");
    if (existingImage) {
      existingImage.style.opacity = "0.3"; // Lottie와 혼합 (완전히 숨기려면 0)
    }

    targetElement.style.position = "relative";
    targetElement.appendChild(container);
    containers.set(instrumentId, container);

    // Lottie 애니메이션 로드
    try {
      const anim = lottie.loadAnimation({
        container,
        renderer: "svg", // svg가 더 선명하고 가볍습니다
        loop: true,
        autoplay: true,
        path: config.path,
        rendererSettings: {
          preserveAspectRatio: "xMidYMid meet",
          clearCanvas: false,
          progressiveLoad: true,
          hideOnTransparent: true
        }
      });

      instances.set(instrumentId, anim);
      states.set(instrumentId, ANIMATION_STATES.IDLE);

      anim.addEventListener("DOMLoaded", () => {
        loadedAnimations.add(instrumentId);
        setState(instrumentId, ANIMATION_STATES.IDLE);
        console.log(`[InstrumentAnimation] ${instrumentId} loaded successfully`);
      });

      anim.addEventListener("data_failed", () => {
        console.warn(`[InstrumentAnimation] ${instrumentId} failed to load, using fallback`);
        showFallback(instrumentId, container, config.fallbackEmoji);
      });

      return true;
    } catch (error) {
      console.error(`[InstrumentAnimation] Failed to init ${instrumentId}:`, error);
      showFallback(instrumentId, container, config.fallbackEmoji);
      return false;
    }
  }

  /**
   * Lottie 로드 실패 시 이모지 대체 표시
   */
  function showFallback(instrumentId, container, emoji) {
    container.innerHTML = `
      <div class="animation-fallback" style="
        display: flex;
        align-items: center;
        justify-content: center;
        width: 100%;
        height: 100%;
        font-size: 3rem;
        animation: fallback-pulse 2s ease-in-out infinite;
      ">${emoji}</div>
    `;
  }

  /**
   * 애니메이션 상태 변경
   * @param {string} instrumentId - 악기 ID
   * @param {string} newState - 새로운 상태 (idle, hover, trigger, fever)
   */
  function setState(instrumentId, newState) {
    const anim = instances.get(instrumentId);
    const config = INSTRUMENT_ANIMATION_CONFIG[instrumentId];

    if (!anim || !config || !loadedAnimations.has(instrumentId)) {
      return;
    }

    const currentState = states.get(instrumentId);
    if (currentState === newState && newState !== ANIMATION_STATES.TRIGGER) {
      return; // 같은 상태면 스킵 (단, TRIGGER는 매번 실행)
    }

    const stateConfig = config.states[newState];
    if (!stateConfig) return;

    states.set(instrumentId, newState);

    // 애니메이션 속도 및 루프 설정
    anim.setSpeed(stateConfig.speed);
    anim.loop = stateConfig.loop;

    // 세그먼트가 있으면 특정 구간만 재생
    if (stateConfig.segments) {
      anim.playSegments(stateConfig.segments, true);
    } else {
      anim.goToAndPlay(0, true);
    }

    // TRIGGER 상태는 완료 후 IDLE로 복귀
    if (newState === ANIMATION_STATES.TRIGGER) {
      const onComplete = () => {
        anim.removeEventListener("complete", onComplete);
        setState(instrumentId, ANIMATION_STATES.IDLE);
      };
      anim.addEventListener("complete", onComplete);
    }
  }

  /**
   * 연주 트리거 (가장 강한 애니메이션)
   */
  function trigger(instrumentId) {
    setState(instrumentId, ANIMATION_STATES.TRIGGER);
  }

  /**
   * Hover 상태로 전환 (손이 근처에 있을 때)
   */
  function hover(instrumentId, isHovering) {
    if (isHovering) {
      setState(instrumentId, ANIMATION_STATES.HOVER);
    } else {
      setState(instrumentId, ANIMATION_STATES.IDLE);
    }
  }

  /**
   * 피버 모드 전환
   */
  function setFeverMode(isFever) {
    const targetState = isFever ? ANIMATION_STATES.FEVER : ANIMATION_STATES.IDLE;
    instances.forEach((anim, instrumentId) => {
      setState(instrumentId, targetState);
    });
  }

  /**
   * 손과 악기 간 거리 기반 반응 (선택적 고급 기능)
   */
  function updateProximity(instrumentId, distance, maxDistance = 200) {
    const anim = instances.get(instrumentId);
    if (!anim || !loadedAnimations.has(instrumentId)) return;

    // 거리에 따라 스케일/투명도 조절
    const proximity = Math.max(0, 1 - distance / maxDistance);
    const container = containers.get(instrumentId);

    if (container && proximity > 0.3) {
      const scale = 1 + proximity * 0.15; // 최대 15% 확대
      container.style.transform = `scale(${scale})`;
      container.style.filter = `brightness(${1 + proximity * 0.3})`;
    } else if (container) {
      container.style.transform = "scale(1)";
      container.style.filter = "brightness(1)";
    }
  }

  /**
   * 모든 애니메이션 정리
   */
  function destroy() {
    instances.forEach((anim) => {
      anim.destroy();
    });
    containers.forEach((container) => {
      if (container && container.parentNode) {
        container.parentNode.removeChild(container);
      }
    });
    instances.clear();
    containers.clear();
    states.clear();
    loadedAnimations.clear();
  }

  /**
   * 애니메이션 일시정지/재생
   */
  function setPaused(instrumentId, paused) {
    const anim = instances.get(instrumentId);
    if (!anim) return;

    if (paused) {
      anim.pause();
    } else {
      anim.play();
    }
  }

  return {
    initAnimation,
    setState,
    trigger,
    hover,
    setFeverMode,
    updateProximity,
    setPaused,
    destroy,
    // 상태 확인용
    getState: (instrumentId) => states.get(instrumentId),
    isLoaded: (instrumentId) => loadedAnimations.has(instrumentId)
  };
}

// ─── CSS 애니메이션 폴백 스타일 추가 ─────────────────────
if (typeof document !== "undefined") {
  const style = document.createElement("style");
  style.textContent = `
    @keyframes fallback-pulse {
      0%, 100% {
        transform: scale(1) rotate(0deg);
        opacity: 0.9;
      }
      50% {
        transform: scale(1.1) rotate(5deg);
        opacity: 1;
      }
    }

    .instrument-lottie-container {
      transition: transform 0.3s ease, filter 0.3s ease;
    }
  `;
  document.head.appendChild(style);
}
