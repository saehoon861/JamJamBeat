// [interaction_runtime.js] 시작 버튼 hover, 악기 충돌, 제스처 반응처럼 사용자 상호작용을 처리하는 모듈입니다.

const ACTIVE_CLASS_MS = 260;

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
  resetGestureState,
  getModelPrediction,
  restartClassAnimation,
  activateStart,
  registerHit,
  spawnBurst,
  spawnPointerBurst,
  setGestureObjectVariant,
  getGestureInstrumentId,
  getGesturePlayback,
  getInstrumentPlayback,
  playGestureSound,
  playInstrumentSound,
  instruments,
  interactionMode,
  collisionPadding,
  startHoverMs,
  gestureCooldownMs,
  isAdminEditMode,
  isSessionStarted,
  feverController,
  checkBubbleCollision
}) {
  const handStateByKey = new Map();
  const activeGestureHands = new Set();
  const rectCache = new WeakMap();
  let lastStatusText = "";
  const HELD_ONESHOT_INTERVAL_MS = Math.max(gestureCooldownMs, 220);
  const SOUND_DEBUG = (() => {
    const raw = new URLSearchParams(window.location.search).get("soundDebug");
    return raw === "1" || raw === "true";
  })();

  function logSoundDebug(stage, payload = {}) {
    if (!SOUND_DEBUG) return;
    console.info("[SoundDebug]", {
      at: Date.now(),
      perfNow: Number(performance.now().toFixed(2)),
      stage,
      ...payload
    });
  }

  function isGestureEnabledForHand(handKey = "default") {
    // feature/mirroring 지원 모델을 위해 상위 차단을 해제합니다.
    return true;
  }

  function createDisabledGestureResult() {
    return {
      label: "None",
      confidence: 0,
      source: "disabled",
      disabled: true
    };
  }

  function setStatusText(nextText) {
    if (typeof nextText !== "string") return;
    if (nextText === lastStatusText) return;
    statusText.textContent = nextText;
    lastStatusText = nextText;
  }

  function getHandState(handKey = "default") {
    if (!handStateByKey.has(handKey)) {
      handStateByKey.set(handKey, {
        lastGestureLabel: "None",
        currentMelodyType: null,
        lastGestureTriggerAt: 0,
        lastResolvedGesture: null,
        lastRawModelPrediction: null,
        lastLandmarks: null,
        lastUpdatedAt: 0
      });
    }
    return handStateByKey.get(handKey);
  }

  function markInstrumentTriggered(instrument, now, sourceKey = "default") {
    if (!instrument) return;
    if (!(instrument.lastHitAtBySource instanceof Map)) {
      instrument.lastHitAtBySource = new Map();
    }
    instrument.lastHitAtBySource.set(sourceKey, now);
    instrument.lastHitAt = now;
  }

  function canTriggerInstrument(instrument, now, sourceKey = "default") {
    if (!instrument) return false;
    if (!(instrument.lastHitAtBySource instanceof Map)) {
      instrument.lastHitAtBySource = new Map();
    }
    const lastHitAt = instrument.lastHitAtBySource.get(sourceKey) || 0;
    if (now - lastHitAt < instrument.cooldownMs) {
      logSoundDebug("interaction.canTriggerInstrument.blocked", {
        instrumentId: instrument.id,
        sourceKey,
        elapsedMs: Number((now - lastHitAt).toFixed(2)),
        cooldownMs: instrument.cooldownMs
      });
      return false;
    }
    markInstrumentTriggered(instrument, now, sourceKey);
    return true;
  }

  // 현재 (마지막으로 인식된) 제스처 라벨을 반환합니다.
  function getCurrentGesture(handKey = "default") {
    return getHandState(handKey).lastGestureLabel;
  }

  function getDebugSnapshot() {
    const snapshot = {};
    handStateByKey.forEach((handState, handKey) => {
      snapshot[handKey] = {
        lastGestureLabel: handState.lastGestureLabel,
        currentMelodyType: handState.currentMelodyType,
        lastResolvedGesture: handState.lastResolvedGesture,
        lastRawModelPrediction: handState.lastRawModelPrediction,
        lastLandmarks: Array.isArray(handState.lastLandmarks)
          ? handState.lastLandmarks.map((point) => ({ ...point }))
          : null,
        lastUpdatedAt: handState.lastUpdatedAt
      };
    });
    return snapshot;
  }

  function hasRecognizedGesture() {
    for (const handState of handStateByKey.values()) {
      if (handState.lastGestureLabel && handState.lastGestureLabel !== "None") {
        return true;
      }
    }
    return false;
  }

  function hasRecognizedGestureExcept(handKey = "default") {
    for (const [key, handState] of handStateByKey.entries()) {
      if (key === handKey) continue;
      if (handState.lastGestureLabel && handState.lastGestureLabel !== "None") {
        return true;
      }
    }
    return false;
  }

  function getGestureStartConfidenceFloor(label) {
    if (label === "Pinky" || label === "Animal" || label === "KHeart") return 0.2;
    return 0.3;
  }

  function snapshotLandmarks(landmarks) {
    return Array.isArray(landmarks)
      ? landmarks.map((point) => ({
        x: Number.isFinite(point?.x) ? point.x : 0,
        y: Number.isFinite(point?.y) ? point.y : 0,
        z: Number.isFinite(point?.z) ? point.z : 0
      }))
      : null;
  }

  function updateTrackedHandSnapshot(landmarks, now, handKey = "default") {
    const handState = getHandState(handKey);
    handState.lastLandmarks = snapshotLandmarks(landmarks);
    handState.lastUpdatedAt = now;

    if (!isGestureEnabledForHand(handKey)) {
      const disabledResult = createDisabledGestureResult();
      if (handState.currentMelodyType) {
        audioApi.stopMelodySequence(handState.currentMelodyType);
        handState.currentMelodyType = null;
      }
      audioApi.stopMelodiesForHand?.(handKey);
      handState.lastResolvedGesture = { ...disabledResult };
      handState.lastRawModelPrediction = { ...disabledResult };
      handState.lastGestureLabel = "None";
      handState.lastGestureTriggerAt = 0;
      resetGestureState?.(handKey);
      activeGestureHands.delete(handKey);
      setGestureObjectVariant?.(activeGestureHands.size > 0);
    }

    return handState;
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
    const now = performance.now();
    const cached = rectCache.get(element);
    const rect = (!cached || now - cached.ts > 500)
      ? (() => {
        const nextRect = element.getBoundingClientRect();
        rectCache.set(element, { rect: nextRect, ts: now });
        return nextRect;
      })()
      : cached.rect;
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
      setStatusText(`시작까지 ${Math.ceil(remain / 100)}초...`);
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
      setStatusText("시작 버튼 위에 손을 올려주세요.");
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
    const sourceKey = meta.handKey || meta.gestureSource || "default";
    if (!canTriggerInstrument(instrument, now, sourceKey)) return;
    audioApi.setPlaybackContext({
      instrumentId: instrument.id,
      gestureLabel: meta.gestureLabel || null,
      gestureSource: meta.gestureSource || null,
      triggerTs: now,
      handKey: meta.handKey || null
    });
    const playedTag = instrument.onHit();
    activateInstrumentElement(instrument);
    registerHit(now);
    setStatusText(`${instrument.name} - ${playedTag || instrument.soundTag}`);
  }

  function triggerGesturePlaybackById(id, playback, now, meta = {}) {
    const instrument = instruments.find((item) => item.id === id);
    if (!instrument || !instrument.el || !playback) return;
    logSoundDebug("interaction.triggerGesturePlaybackById", {
      instrumentId: id,
      handKey: meta.handKey || null,
      gestureLabel: meta.gestureLabel || null,
      playbackMode: playback.playbackMode || null,
      soundTag: playback.soundTag || instrument.soundTag
    });
    const sourceKey = meta.handKey || meta.gestureSource || "default";
    if (!canTriggerInstrument(instrument, now, sourceKey)) return;
    audioApi.setPlaybackContext({
      instrumentId: instrument.id,
      gestureLabel: meta.gestureLabel || null,
      gestureSource: meta.gestureSource || null,
      triggerTs: now,
      inferenceTs: Number.isFinite(meta.inferenceTs) ? meta.inferenceTs : null,
      handKey: meta.handKey || null
    });
    if (typeof playGestureSound === "function") {
      playGestureSound(meta.gestureLabel || null, instrument.id, undefined);
    } else {
      playInstrumentSound(instrument.id);
    }
    activateInstrumentElement(instrument);
    registerHit(now);
    setStatusText(`${instrument.name} - ${playback.soundTag || instrument.soundTag}`);
  }

  // 소리는 내지 않고 화면 효과만 잠깐 보여주고 싶을 때 쓰는 함수입니다.
  function triggerVisualOnlyById(id, now, burstType = "pinky") {
    const instrument = instruments.find((item) => item.id === id);
    if (!instrument || !instrument.el) return false;
    if (!canTriggerInstrument(instrument, now, "visual")) return false;
    spawnBurst(burstType, instrument.el);
    activateInstrumentElement(instrument);
    registerHit(now);
    return true;
  }

  // 여러 손가락 끝 좌표를 받아서 악기와 충돌했는지 한 번에 검사합니다.
  // 터치 모드일 때만 사용됩니다.
  function processInstrumentCollision(points, now) {
    // 소리는 제스처 인식으로만 나가게 하고, 터치 충돌은 사운드 트리거로 사용하지 않습니다.
    void points;
    void now;
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
    const instrumentId = getGestureInstrumentId?.(label);
    if (!instrumentId) {
      logSoundDebug("interaction.runGestureReaction.skip", {
        handKey,
        gestureLabel: label,
        reason: "no_instrument_mapping"
      });
      return;
    }
    const playback = getGesturePlayback?.(label, instrumentId) || getInstrumentPlayback(instrumentId);
    if (!playback) {
      logSoundDebug("interaction.runGestureReaction.skip", {
        handKey,
        gestureLabel: label,
        instrumentId,
        reason: "no_playback_profile"
      });
      return;
    }
    const handState = getHandState(handKey);
    logSoundDebug("interaction.runGestureReaction.start", {
      handKey,
      gestureLabel: label,
      instrumentId,
      playbackMode: playback.playbackMode || null,
      audioState: audioApi.getAudioState()
    });

    if (playback.playbackMode === "oneshot") {
      if (handState.currentMelodyType) {
        audioApi.stopMelodySequence(handState.currentMelodyType);
        handState.currentMelodyType = null;
      }
      audioApi.stopMelodiesForHand?.(handKey);
      triggerGesturePlaybackById(instrumentId, playback, now, {
        gestureLabel: label,
        gestureSource: "model",
        handKey,
        inferenceTs: handState.lastUpdatedAt
      });
      handState.lastGestureTriggerAt = now;
      return;
    }

    // 멜로디 시퀀스 시작
    const melodyType = `${playback.melodyType}:${handKey}`;
    if (handState.currentMelodyType !== melodyType) {
      // 이전 멜로디 중지
      if (handState.currentMelodyType) {
        audioApi.stopMelodySequence(handState.currentMelodyType);
      }
      audioApi.stopMelodiesForHand?.(handKey);
      // 새 멜로디 시작
      let pendingInferenceTs = handState.lastUpdatedAt;
      audioApi.startMelodySequence(melodyType, (note) => {
        audioApi.setPlaybackContext({
          instrumentId,
          gestureLabel: `${label}:${handKey}`,
          gestureSource: `model:${handKey}`,
          triggerTs: performance.now(),
          inferenceTs: Number.isFinite(pendingInferenceTs) ? pendingInferenceTs : null,
          handKey
        });
        if (typeof playGestureSound === "function") {
          playGestureSound(label, instrumentId, note);
        } else {
          playInstrumentSound(instrumentId, note);
        }
        pendingInferenceTs = null;
      });
      handState.currentMelodyType = melodyType;
    }

    // 시각 효과만 표시 (소리는 멜로디 시퀀스가 재생)
    const instrument = instruments.find((item) => item.id === instrumentId);
    if (instrument && instrument.el) {
      activateInstrumentElement(instrument);
      spawnBurst(playback.burstType || "pinky", instrument.el);
    }
  }

  // 다람쥐 이미지에 같은 CSS 애니메이션을 다시 재생시키기 위한 함수입니다.
  function showSquirrelEffect() {
    restartClassAnimation(gestureSquirrelEffect, "is-visible");
  }

  function getGestureEffectZone() {
    if (!handCursor || typeof handCursor.getBoundingClientRect !== "function") {
      return null;
    }

    const rect = handCursor.getBoundingClientRect();
    if (!rect.width || !rect.height) return null;

    const zoneWidth = Math.max(180, rect.width * 3.2);
    const zoneHeight = Math.max(150, rect.height * 0.82);
    const centerX = rect.left + rect.width * 0.5;
    const centerY = rect.top + rect.height * 0.16;

    return {
      left: centerX - zoneWidth * 0.5,
      right: centerX + zoneWidth * 0.5,
      top: centerY - zoneHeight * 0.5,
      bottom: centerY + zoneHeight * 0.5,
      centerX,
      centerY
    };
  }

  function clampPointToGestureZone(point, zone) {
    if (!point || !zone) return point;
    return {
      x: Math.min(zone.right, Math.max(zone.left, point.x)),
      y: Math.min(zone.bottom, Math.max(zone.top, point.y))
    };
  }

  function createPointerBurstPoint(landmarks) {
    const tip = Array.isArray(landmarks) ? landmarks[8] : null;
    const wrist = Array.isArray(landmarks) ? landmarks[0] : null;
    const zone = getGestureEffectZone();
    if (
      Number.isFinite(tip?.x) &&
      Number.isFinite(tip?.y) &&
      Number.isFinite(wrist?.x) &&
      Number.isFinite(wrist?.y)
    ) {
      const anchorX = (1 - tip.x) * window.innerWidth;
      const anchorY = tip.y * window.innerHeight;
      const cursorHeight = handCursor?.offsetHeight || 214;
      const tipOffsetPx = Math.max(10, cursorHeight * 0.82);
      const dx = tip.x - wrist.x;
      const dy = tip.y - wrist.y;
      const angleRad = Math.atan2(dy, dx) + Math.PI / 2;

      return clampPointToGestureZone({
        x: anchorX + Math.sin(angleRad) * tipOffsetPx,
        y: anchorY - Math.cos(angleRad) * tipOffsetPx
      }, zone);
    }

    if (!zone) {
      return null;
    }

    return {
      x: zone.centerX,
      y: zone.centerY
    };
  }

  function showGestureRecognitionEffect(landmarks) {
    const point = createPointerBurstPoint(landmarks);
    if (point && typeof spawnPointerBurst === "function") {
      spawnPointerBurst(point.x, point.y);
      return;
    }
    showSquirrelEffect();
  }

  // 제스처 모드에서 현재 손모양을 해석하고, 쿨다운/오디오 상태까지 확인한 뒤 반응을 실행합니다.
  function processGestureTriggers(landmarks, now, handKey = "default") {
    if (!isSessionStarted()) return;
    if (isAdminEditMode()) return;
    const handState = updateTrackedHandSnapshot(landmarks, now, handKey);

    if (!isGestureEnabledForHand(handKey)) {
      return;
    }

    const gesture = resolveGesture(landmarks, now, isSessionStarted(), handKey);
    const rawModel = getModelPrediction(landmarks, now, handKey);
    handState.lastResolvedGesture = gesture ? { ...gesture } : null;
    handState.lastRawModelPrediction = rawModel ? { ...rawModel } : null;

    // 제스처가 없거나 신뢰도가 매우 낮으면 멜로디 중지
    const shouldStopMelody = !gesture || gesture.label === "None" ||
                             (gesture.confidence < getGestureStartConfidenceFloor(gesture.label) && gesture.label !== handState.lastGestureLabel);

    if (shouldStopMelody) {
      logSoundDebug("interaction.processGestureTriggers.skip", {
        handKey,
        reason: !gesture ? "no_gesture" : (gesture.label === "None" ? "gesture_none" : "confidence_below_threshold"),
        gestureLabel: gesture?.label || null,
        confidence: Number.isFinite(gesture?.confidence) ? Number(gesture.confidence.toFixed(3)) : null,
        lastGestureLabel: handState.lastGestureLabel,
        rawModelLabel: rawModel?.label || null
      });
      activeGestureHands.delete(handKey);
      setGestureObjectVariant?.(activeGestureHands.size > 0);
      const hasOtherRecognizedGesture = hasRecognizedGestureExcept(handKey);
      const hadCurrentMelody = Boolean(handState.currentMelodyType);
      // 손동작이 없거나 불확실하면 멜로디 중지
      if (handState.currentMelodyType) {
        audioApi.stopMelodySequence(handState.currentMelodyType);
        handState.currentMelodyType = null;
        handState.lastGestureTriggerAt = 0;
      }
      audioApi.stopMelodiesForHand?.(handKey);
      const hasOtherActiveMelody = audioApi.hasAnyActiveMelody?.() ?? false;
      if (hadCurrentMelody && !hasOtherRecognizedGesture && !hasOtherActiveMelody) {
        setStatusText(`${handKey} 손 멜로디 중지됨`);
      }

      // 모델도 "아무것도 아님"이라고 보면 사용자에게 다시 동작해달라고 안내합니다.
      if ((!gesture || gesture.label === "None") && !hasOtherRecognizedGesture && !hasOtherActiveMelody) {
        if (rawModel?.classId === 0 || String(rawModel?.label || "").trim().toLowerCase() === "class0") {
          setStatusText(`${handKey} 손 동작을 다시해주세요.`);
        }
      }
      handState.lastGestureLabel = "None";
      return;
    }

    // 제스처가 바뀌었을 때만 새 멜로디 시작
    if (gesture.label !== handState.lastGestureLabel) {
      logSoundDebug("interaction.processGestureTriggers.trigger", {
        handKey,
        gestureLabel: gesture.label,
        confidence: Number((gesture.confidence * 100).toFixed(1)),
        previousGestureLabel: handState.lastGestureLabel
      });
      if (gesture.label === "Fist") activeGestureHands.add(handKey);
      else activeGestureHands.delete(handKey);
      setGestureObjectVariant?.(activeGestureHands.size > 0);

      if (!audioApi.getAudioState().running) {
        logSoundDebug("interaction.processGestureTriggers.blocked", {
          handKey,
          gestureLabel: gesture.label,
          reason: "audio_not_running",
          audioState: audioApi.getAudioState()
        });
        setStatusText("소리가 꺼져 있어요. '소리 켜기' 버튼을 눌러주세요.");
        return;
      }

      runGestureReaction(gesture.label, now, handKey);
      showGestureRecognitionEffect(landmarks);

      handState.lastGestureLabel = gesture.label;

      const displayName = getGestureDisplayName(gesture.label);
      setStatusText(`${handKey}손 ${displayName} 인식! (신뢰도: ${(gesture.confidence * 100).toFixed(0)}%)`);
    } else {
      if (gesture.label === "Fist") activeGestureHands.add(handKey);
      else activeGestureHands.delete(handKey);
      setGestureObjectVariant?.(activeGestureHands.size > 0);
      const instrumentId = getGestureInstrumentId?.(gesture.label);
      const playback = getGesturePlayback?.(gesture.label, instrumentId) || getInstrumentPlayback(instrumentId);
      if (playback?.playbackMode === "oneshot" && audioApi.getAudioState().running) {
        if (now - handState.lastGestureTriggerAt >= HELD_ONESHOT_INTERVAL_MS) {
          logSoundDebug("interaction.processGestureTriggers.retrigger", {
            handKey,
            gestureLabel: gesture.label,
            elapsedMs: Number((now - handState.lastGestureTriggerAt).toFixed(2)),
            retriggerMs: HELD_ONESHOT_INTERVAL_MS
          });
          runGestureReaction(gesture.label, now, handKey);
          showGestureRecognitionEffect(landmarks);
          handState.lastGestureTriggerAt = now;
        }
      }
      const displayName = getGestureDisplayName(gesture.label);
      setStatusText(`${handKey}손 ${displayName} 유지 중... 🎵 (${(gesture.confidence * 100).toFixed(0)}%)`);
    }
  }

  // 손 커서의 위치와 보이기/숨기기를 담당합니다.
  function setPointer(point, now, landmarks = null) {
    const baseTransform = "translate(-50%, -86%) scaleY(-1)";
    handCursor.style.opacity = 1;
    handCursor.style.left = `${point.x}px`;
    handCursor.style.top = `${point.y}px`;

    // 손목과 검지/중지 끝의 평균 각도를 계산하여 세로 지휘봉 이미지를 회전합니다.
    if (landmarks && landmarks.length >= 13) {
      const wrist = landmarks[0];
      const indexTip = landmarks[8];
      const middleTip = landmarks[12];

      // 검지와 중지의 평균 위치를 사용하여 더 안정적인 방향 계산
      const fingerX = (indexTip.x + middleTip.x) / 2;
      const fingerY = (indexTip.y + middleTip.y) / 2;

      // 손가락에서 손목 방향으로 계산하여 회전 방향 반전
      const dx = wrist.x - fingerX;
      const dy = wrist.y - fingerY;
      const angle = Math.atan2(dy, dx) * (180 / Math.PI) + 90;
      handCursor.style.transform = `${baseTransform} rotate(${angle}deg)`;
      return;
    }

    // 랜드마크가 부족한 경우 검지만 사용
    if (landmarks && landmarks.length > 8) {
      const wrist = landmarks[0];
      const indexTip = landmarks[8];
      // 손가락에서 손목 방향으로 계산하여 회전 방향 반전
      const dx = wrist.x - indexTip.x;
      const dy = wrist.y - indexTip.y;
      const angle = Math.atan2(dy, dx) * (180 / Math.PI) + 90;
      handCursor.style.transform = `${baseTransform} rotate(${angle}deg)`;
      return;
    }

    handCursor.style.transform = baseTransform;
  }

  // 모든 터치 포인트(손가락 끝)에 대해 비눗방울 충돌을 검사합니다.
  function processBubbleCollisions(points) {
    // 버블 충돌은 시각 효과만 유지하고 사운드는 내지 않습니다.
    checkBubbleCollision(points);
  }

  // 손이 사라졌을 때 커서를 숨기고, 시작 전이라면 상태 문구도 초기화합니다.
  function resetTrackingState() {
    handCursor.style.opacity = 0;
    const resetAt = performance.now();
    activeGestureHands.clear();
    setGestureObjectVariant?.(false);

    // 손이 사라지면 멜로디와 홀드 시간도 중지
    handStateByKey.forEach((handState, handKey) => {
      if (handState.currentMelodyType) {
        audioApi.stopMelodySequence(handState.currentMelodyType);
        handState.currentMelodyType = null;
      }
      audioApi.stopMelodiesForHand?.(handKey);
      handState.lastGestureLabel = "None";
      handState.lastGestureTriggerAt = 0;
      handState.lastResolvedGesture = null;
      handState.lastRawModelPrediction = null;
      handState.lastLandmarks = null;
      handState.lastUpdatedAt = 0;
      resetGestureState?.(handKey);
      if (isGestureEnabledForHand(handKey)) {
        getModelPrediction?.(null, resetAt, handKey);
      }
    });

    if (!isSessionStarted()) {
      setStatusText("카메라에 손을 보여주세요.");
    }
  }

  // 바깥에서는 필요한 상호작용 함수들만 꺼내 쓰면 됩니다.
  return {
    createInstrumentPoint,
    processLandingHover,
    processInstrumentCollision,
    processGestureTriggers,
    processBubbleCollisions,
    updateTrackedHandSnapshot,
    setPointer,
    resetTrackingState,
    getCurrentGesture,
    getDebugSnapshot
  };
}
