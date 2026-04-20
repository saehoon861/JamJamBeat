// [instrument_animations_simple.js] 간단한 CSS 기반 애니메이션 폴백 버전
// Lottie가 로드되지 않을 경우를 위한 대안

export function createInstrumentAnimationManager() {
  const containers = new Map();
  const states = new Map();

  // CSS 클래스 기반 애니메이션
  function initAnimation(instrumentId, targetElement) {
    if (!targetElement) {
      console.warn(`[InstrumentAnimation] Target element not found for ${instrumentId}`);
      return false;
    }

    // 애니메이션 컨테이너 생성
    const container = document.createElement("div");
    container.className = `instrument-animation instrument-animation-${instrumentId}`;
    container.style.cssText = `
      position: absolute;
      inset: 0;
      width: 100%;
      height: 100%;
      pointer-events: none;
      z-index: 10;
    `;

    targetElement.appendChild(container);
    containers.set(instrumentId, container);
    states.set(instrumentId, "idle");

    // IDLE 애니메이션 시작
    setState(instrumentId, "idle");

    console.log(`[InstrumentAnimation] ${instrumentId} initialized (CSS fallback)`);
    return true;
  }

  function setState(instrumentId, newState) {
    const container = containers.get(instrumentId);
    if (!container) return;

    const currentState = states.get(instrumentId);
    if (currentState === newState && newState !== "trigger") {
      return;
    }

    states.set(instrumentId, newState);

    // 모든 상태 클래스 제거
    container.classList.remove("anim-idle", "anim-hover", "anim-trigger", "anim-fever");

    // 새 상태 클래스 추가
    container.classList.add(`anim-${newState}`);

    // TRIGGER는 원샷 후 IDLE 복귀
    if (newState === "trigger") {
      setTimeout(() => {
        setState(instrumentId, "idle");
      }, 500);
    }
  }

  function trigger(instrumentId) {
    setState(instrumentId, "trigger");
  }

  function hover(instrumentId, isHovering) {
    setState(instrumentId, isHovering ? "hover" : "idle");
  }

  function setFeverMode(isFever) {
    const targetState = isFever ? "fever" : "idle";
    containers.forEach((container, instrumentId) => {
      setState(instrumentId, targetState);
    });
  }

  function updateProximity(instrumentId, distance, maxDistance = 200) {
    // CSS 폴백 버전에서는 간단하게 구현
    const container = containers.get(instrumentId);
    if (!container) return;

    const proximity = Math.max(0, 1 - distance / maxDistance);
    if (proximity > 0.3) {
      const scale = 1 + proximity * 0.1;
      container.style.transform = `scale(${scale})`;
    } else {
      container.style.transform = "scale(1)";
    }
  }

  function setPaused(instrumentId, paused) {
    const container = containers.get(instrumentId);
    if (!container) return;
    container.style.animationPlayState = paused ? "paused" : "running";
  }

  function destroy() {
    containers.forEach((container) => {
      if (container && container.parentNode) {
        container.parentNode.removeChild(container);
      }
    });
    containers.clear();
    states.clear();
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
    getState: (instrumentId) => states.get(instrumentId),
    isLoaded: (instrumentId) => containers.has(instrumentId)
  };
}

// CSS 스타일 추가
if (typeof document !== "undefined") {
  const style = document.createElement("style");
  style.textContent = `
    .instrument-animation {
      transition: transform 0.3s ease, filter 0.3s ease;
    }

    /* IDLE 상태: 부드러운 호흡 */
    .instrument-animation.anim-idle::before {
      content: '';
      position: absolute;
      inset: -10%;
      background: radial-gradient(circle, rgba(255,255,255,0.2) 0%, transparent 70%);
      animation: breathing 3s ease-in-out infinite;
      border-radius: 50%;
    }

    @keyframes breathing {
      0%, 100% {
        transform: scale(0.95);
        opacity: 0.3;
      }
      50% {
        transform: scale(1.05);
        opacity: 0.6;
      }
    }

    /* TRIGGER 상태: 강렬한 펄스 */
    .instrument-animation.anim-trigger::before {
      content: '';
      position: absolute;
      inset: -20%;
      background: radial-gradient(circle, rgba(255,220,100,0.6) 0%, transparent 60%);
      animation: pulse-burst 0.5s ease-out;
      border-radius: 50%;
    }

    @keyframes pulse-burst {
      0% {
        transform: scale(0.5);
        opacity: 1;
      }
      100% {
        transform: scale(1.5);
        opacity: 0;
      }
    }

    /* FEVER 상태: 화려한 회전 */
    .instrument-animation.anim-fever::before {
      content: '';
      position: absolute;
      inset: -15%;
      background: conic-gradient(
        from 0deg,
        rgba(255,100,100,0.4),
        rgba(100,255,100,0.4),
        rgba(100,100,255,0.4),
        rgba(255,100,100,0.4)
      );
      animation: fever-spin 1s linear infinite;
      border-radius: 50%;
    }

    @keyframes fever-spin {
      0% {
        transform: rotate(0deg) scale(1);
      }
      50% {
        transform: rotate(180deg) scale(1.1);
      }
      100% {
        transform: rotate(360deg) scale(1);
      }
    }

    /* HOVER 상태: 살짝 강조 */
    .instrument-animation.anim-hover::before {
      content: '';
      position: absolute;
      inset: -12%;
      background: radial-gradient(circle, rgba(200,220,255,0.4) 0%, transparent 70%);
      animation: hover-glow 1.5s ease-in-out infinite;
      border-radius: 50%;
    }

    @keyframes hover-glow {
      0%, 100% {
        transform: scale(1);
        opacity: 0.5;
      }
      50% {
        transform: scale(1.08);
        opacity: 0.8;
      }
    }
  `;
  document.head.appendChild(style);
}
