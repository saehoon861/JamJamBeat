// [interaction_runtime.js] 시작 버튼 hover, 악기 충돌, 제스처 반응처럼 사용자 상호작용을 처리하는 모듈입니다.

const ACTIVE_CLASS_MS = 260;
const GESTURE_SOUND_MAP = {
  Fist: {
    id: "drum"
  },
  OpenPalm: {
    id: "xylophone"
  },
  V: {
    id: "tambourine"
  },
  Pinky: {
    id: "owl"
  },
  Animal: {
    id: "fern"
  },
  KHeart: {
    id: "fern"
  }
};

// createInteractionRuntime(...) 은 "손이 무엇을 만졌는지, 어떤 제스처를 했는지"를 해석하는 상호작용 모듈입니다.
// 실제 소리 재생이나 제스처 판정 함수는 바깥에서 주입받고, 여기서는 흐름만 관리합니다.
export function createInteractionRuntime({
  landingOverlay,
  landingStartButton,
  statusText,
  handCursor,
  gestureSquirrelEffect,
  audioApi,
  resolveGesture,
  getModelPrediction,
  restartClassAnimation,
  activateStart,
  registerHit,
  spawnBurst,
  getInstrumentPlayback,
  playInstrumentSound,
  instruments,
  interactionMode,
  collisionPadding,
  startHoverMs,
  gestureCooldownMs,
  isAdminEditMode,
  isSessionStarted
}) {
  const handStateByKey = new Map();

  function getHandState(handKey = "default") {
    if (!handStateByKey.has(handKey)) {
      handStateByKey.set(handKey, {
        lastGestureLabel: "None",
        currentMelodyType: null
      });
    }
    return handStateByKey.get(handKey);
  }

  function getGestureStartConfidenceFloor(label) {
    if (label === "Pinky" || label === "Animal" || label === "KHeart") return 0.2;
    return 0.3;
  }

  // hoverStartedAt / hoverActive 는 시작 버튼 위에 손을 올려둔 시간을 재기 위해 씁니다.
  let hoverStartedAt = 0;
  let hoverActive = false;
  // MediaPipe 손 좌표(0~1 비율)를 실제 화면 픽셀 좌표로 바꿉니다.
  function createInstrumentPoint(landmark, canvas) {
    return {
      x: (1 - landmark.x) * canvas.width,
      y: landmark.y * canvas.height
    };
  }

  function isInsideElement(point, element, padding = collisionPadding) {
    if (!element) return false;
    const rect = element.getBoundingClientRect();
    return (
      point.x >= rect.left - padding &&
      point.x <= rect.right + padding &&
      point.y >= rect.top - padding &&
      point.y <= rect.bottom + padding
    );
  }

  // 시작 화면에서 손 커서가 버튼 위에 일정 시간 머무는지 검사합니다.
  function processLandingHover(point, now) {
    if (landingOverlay.classList.contains("is-hidden")) return;

    if (isInsideElement(point, landingStartButton, 24)) {
      if (!hoverActive) {
        hoverActive = true;
        hoverStartedAt = now;
      }

      const remain = Math.max(0, startHoverMs - (now - hoverStartedAt));
      // 남은 시간을 사용자에게 보여줍니다.
      statusText.textContent = `시작까지 ${Math.ceil(remain / 100)}초...`;
      if (now - hoverStartedAt >= startHoverMs) {
        hoverActive = false;
        hoverStartedAt = 0;
        Promise.resolve(audioApi.unlockAudioContext?.())
          .catch(() => false)
          .finally(() => {
            activateStart();
          });
      }
    } else {
      hoverActive = false;
      hoverStartedAt = 0;
      statusText.textContent = "시작 버튼 위에 손을 올려주세요.";
    }
  }

  // 악기가 연주될 때 잠깐 반짝이도록 active 클래스를 붙였다가 곧 제거합니다.
  function activateInstrumentElement(instrument) {
    instrument.el.classList.add("active");
    window.setTimeout(() => instrument.el.classList.remove("active"), ACTIVE_CLASS_MS);
  }

  // id 로 특정 악기를 찾아 실제 소리, 이펙트, 상태 문구를 함께 처리합니다.
  function triggerInstrumentById(id, now, meta = {}) {
    const instrument = instruments.find((item) => item.id === id);
    if (!instrument || !instrument.el) return;
    if (now - instrument.lastHitAt < instrument.cooldownMs) return;

    instrument.lastHitAt = now;
    audioApi.setPlaybackContext({
      instrumentId: instrument.id,
      gestureLabel: meta.gestureLabel || null,
      gestureSource: meta.gestureSource || null,
      triggerTs: now
    });
    const playedTag = instrument.onHit();
    activateInstrumentElement(instrument);
    registerHit(now);
    statusText.textContent = `${instrument.name} - ${playedTag || instrument.soundTag}`;
  }

  // 소리는 내지 않고 화면 효과만 잠깐 보여주고 싶을 때 쓰는 함수입니다.
  function triggerVisualOnlyById(id, now, burstType = "pinky") {
    const instrument = instruments.find((item) => item.id === id);
    if (!instrument || !instrument.el) return false;
    if (now - instrument.lastHitAt < instrument.cooldownMs) return false;

    instrument.lastHitAt = now;
    spawnBurst(burstType, instrument.el);
    activateInstrumentElement(instrument);
    registerHit(now);
    return true;
  }

  // 여러 손가락 끝 좌표를 받아서 악기와 충돌했는지 한 번에 검사합니다.
  // 터치 모드일 때만 사용됩니다.
  function processInstrumentCollision(points, now) {
    if (!isSessionStarted()) return;
    if (isAdminEditMode()) return;
    if (interactionMode === "gesture") return;

    for (const instrument of instruments) {
      const hit = points.some((point) => isInsideElement(point, instrument.el));
      if (!hit) continue;
      if (now - instrument.lastHitAt < instrument.cooldownMs) continue;

      instrument.lastHitAt = now;
      audioApi.setPlaybackContext({
        instrumentId: instrument.id,
        gestureLabel: "Touch",
        gestureSource: "touch",
        triggerTs: now
      });
      const playedTag = instrument.onHit();
      activateInstrumentElement(instrument);
      registerHit(now);
      statusText.textContent = `${instrument.name} - ${playedTag || instrument.soundTag}`;
    }
  }

  // 영어 제스처 이름을 사용자에게 보여줄 한국어 이름으로 바꿉니다.
  function getGestureDisplayName(label) {
    if (label === "Fist") return "주먹";
    if (label === "OpenPalm") return "손바닥";
    if (label === "V") return "브이";
    if (label === "Pinky") return "새끼손가락";
    if (label === "Animal") return "애니멀";
    if (label === "KHeart") return "K-하트";
    return label;
  }

  // 특정 제스처가 인식되었을 때 어떤 악기나 효과를 낼지 연결하는 함수입니다.
  function runGestureReaction(label, now, handKey = "default") {
    const instrumentInfo = GESTURE_SOUND_MAP[label];
    if (!instrumentInfo) return;
    const playback = getInstrumentPlayback(instrumentInfo.id);
    if (!playback) return;
    const handState = getHandState(handKey);

    if (playback.playbackMode === "oneshot") {
      if (handState.currentMelodyType) {
        audioApi.stopMelodySequence(handState.currentMelodyType);
        handState.currentMelodyType = null;
      }
      triggerInstrumentById(instrumentInfo.id, now, { gestureLabel: label, gestureSource: "model" });
      return;
    }

    // 멜로디 시퀀스 시작
    if (handState.currentMelodyType !== playback.melodyType) {
      // 이전 멜로디 중지
      if (handState.currentMelodyType) {
        audioApi.stopMelodySequence(handState.currentMelodyType);
      }
      // 새 멜로디 시작
      const melodyType = `${playback.melodyType}:${handKey}`;
      audioApi.startMelodySequence(melodyType, (note) => {
        audioApi.setPlaybackContext({
          instrumentId: instrumentInfo.id,
          gestureLabel: `${label}:${handKey}`,
          gestureSource: `model:${handKey}`,
          triggerTs: performance.now()
        });
        playInstrumentSound(instrumentInfo.id, note);
      });
      handState.currentMelodyType = melodyType;
    }

    // 시각 효과만 표시 (소리는 멜로디 시퀀스가 재생)
    const instrument = instruments.find((item) => item.id === instrumentInfo.id);
    if (instrument && instrument.el) {
      activateInstrumentElement(instrument);
      spawnBurst(playback.burstType || "pinky", instrument.el);
    }
  }

  // 다람쥐 이미지에 같은 CSS 애니메이션을 다시 재생시키기 위한 함수입니다.
  function showSquirrelEffect() {
    restartClassAnimation(gestureSquirrelEffect, "is-visible");
  }

  // 제스처 모드에서 현재 손모양을 해석하고, 쿨다운/오디오 상태까지 확인한 뒤 반응을 실행합니다.
  function processGestureTriggers(landmarks, now, handKey = "default") {
    if (!isSessionStarted()) return;
    if (isAdminEditMode()) return;
    const handState = getHandState(handKey);

    const gesture = resolveGesture(landmarks, now, isSessionStarted(), handKey);

    // 제스처가 없거나 신뢰도가 매우 낮으면 멜로디 중지
    const shouldStopMelody = !gesture || gesture.label === "None" ||
                             (gesture.confidence < getGestureStartConfidenceFloor(gesture.label) && gesture.label !== handState.lastGestureLabel);

    if (shouldStopMelody) {
      // 손동작이 없거나 불확실하면 멜로디 중지
      if (handState.currentMelodyType) {
        audioApi.stopMelodySequence(handState.currentMelodyType);
        handState.currentMelodyType = null;
        statusText.textContent = `${handKey} 손 멜로디 중지됨`;
      }

      // 모델도 "아무것도 아님"이라고 보면 사용자에게 다시 동작해달라고 안내합니다.
      if (!gesture || gesture.label === "None") {
        const rawModel = getModelPrediction(landmarks, now, handKey);
        if (rawModel?.classId === 0 || String(rawModel?.label || "").trim().toLowerCase() === "class0") {
          statusText.textContent = `${handKey} 손 동작을 다시해주세요.`;
        }
      }
      handState.lastGestureLabel = "None";
      return;
    }

    // 제스처가 바뀌었을 때만 새 멜로디 시작
    if (gesture.label !== handState.lastGestureLabel) {
      if (!audioApi.getAudioState().running) {
        statusText.textContent = "소리가 꺼져 있어요. '소리 켜기' 버튼을 눌러주세요.";
        handState.lastGestureLabel = gesture.label;
        return;
      }

      runGestureReaction(gesture.label, now, handKey);
      showSquirrelEffect();

      handState.lastGestureLabel = gesture.label;
      const displayName = getGestureDisplayName(gesture.label);
      statusText.textContent = `${handKey}손 ${displayName} 인식! (신뢰도: ${(gesture.confidence * 100).toFixed(0)}%)`;
    } else {
      // 같은 제스처가 유지되는 경우 - 신뢰도도 함께 표시
      const displayName = getGestureDisplayName(gesture.label);
      statusText.textContent = `${handKey}손 ${displayName} 유지 중... 🎵 (${(gesture.confidence * 100).toFixed(0)}%)`;
    }
  }

  // 손 커서의 위치와 보이기/숨기기를 담당합니다.
  function setPointer(point) {
    handCursor.style.opacity = 1;
    handCursor.style.left = `${point.x}px`;
    handCursor.style.top = `${point.y}px`;
  }

  // 손이 사라졌을 때 커서를 숨기고, 시작 전이라면 상태 문구도 초기화합니다.
  function resetTrackingState() {
    handCursor.style.opacity = 0;

    // 손이 사라지면 멜로디도 중지
    handStateByKey.forEach((handState) => {
      if (handState.currentMelodyType) {
        audioApi.stopMelodySequence(handState.currentMelodyType);
        handState.currentMelodyType = null;
      }
      handState.lastGestureLabel = "None";
    });

    if (!isSessionStarted()) {
      statusText.textContent = "카메라에 손을 보여주세요.";
    }
  }

  // 바깥에서는 필요한 상호작용 함수들만 꺼내 쓰면 됩니다.
  return {
    createInstrumentPoint,
    processLandingHover,
    processInstrumentCollision,
    processGestureTriggers,
    setPointer,
    resetTrackingState
  };
}
