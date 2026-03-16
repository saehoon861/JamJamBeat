// [hand_tracking_runtime.js] MediaPipe 프레임 루프와 손 랜드마크 캐시 처리를 담당하는 모듈입니다.

// createHandTrackingRuntime(...) 은 "손 추적 엔진" 하나를 만들어 돌려주는 공장 함수입니다.
// 필요한 부품(video, canvas, renderer 등)을 바깥에서 받아서, 이 파일 안에서는 추적 흐름만 담당합니다.
export function createHandTrackingRuntime({
  video,
  handCanvas,
  handCtx,
  handCursor,
  renderer,
  particleSystem,
  feverController,
  interactionRuntime,
  onBeforeFrame,
  onDetectionError,
  getHandLandmarker,
  getSessionStarted,
  inferIntervalMs,
  landmarkStaleMs
}) {
  const HAND_KEYS = ["left", "right"];
  // lastVideoTime 은 "이 프레임을 이미 처리했는지" 확인하려고 저장하는 값입니다.
  let lastVideoTime = -1;
  // lastInferenceAt 은 마지막으로 손 인식을 돌린 시각입니다.
  let lastInferenceAt = 0;
  // cachedLandmarks 는 가장 최근에 찾은 손 좌표를 잠깐 기억해두는 저장소입니다.
  let cachedHands = [];
  // cachedLandmarksAt 은 위 좌표를 언제 저장했는지 기록합니다.
  let cachedLandmarksAt = 0;

  function normalizeHandedness(result) {
    const raw = Array.isArray(result?.handednesses) ? result.handednesses : [];
    return raw.map((entry) => {
      const first = Array.isArray(entry) ? entry[0] : entry;
      const label = String(first?.displayName || first?.categoryName || "").trim().toLowerCase();
      if (label === "left" || label === "right") return label;
      return null;
    });
  }

  function buildHandsWithKeys(result) {
    const hands = Array.isArray(result?.landmarks) ? result.landmarks : [];
    const handedness = normalizeHandedness(result);
    const usedKeys = new Set();

    return hands.map((landmarks, index) => {
      const preferredKey = handedness[index];
      let handKey = preferredKey;

      if (!handKey || usedKeys.has(handKey)) {
        handKey = HAND_KEYS.find((candidate) => !usedKeys.has(candidate)) || `hand-${index}`;
      }
      usedKeys.add(handKey);

      return { handKey, landmarks };
    });
  }

  // 직전에 찾은 손 좌표가 아직 너무 오래되지 않았다면,
  // 새 계산 결과가 없어도 화면에 손 모양을 잠깐 계속 보여줍니다.
  function renderCachedHand(now) {
    const cacheFresh = cachedHands.length > 0 && now - cachedLandmarksAt <= landmarkStaleMs;
    if (cacheFresh) {
      // 저장된 손 좌표들을 다시 그립니다.
      cachedHands.forEach((hand) => {
        const currentGesture = interactionRuntime.getCurrentGesture(hand.handKey);
        renderer.drawHand(handCtx, hand.landmarks, handCanvas, now * 0.001, hand.handKey, currentGesture);
      });
      // 검지 끝 좌표를 손 커서 위치로 변환합니다.
      const primaryHand = cachedHands.find((hand) => hand.handKey === "right") || cachedHands[0];
      // 모든 손가락 끝 위치를 전달하여 비눗방울 터뜨리기를 시도합니다.
      const flickerPoints = [4, 8, 12, 16, 20].map((idx) => interactionRuntime.createInstrumentPoint(primaryHand.landmarks[idx], handCanvas));
      interactionRuntime.processBubbleCollisions(flickerPoints);

      const pointer = interactionRuntime.createInstrumentPoint(primaryHand.landmarks[8], handCanvas);
      // 커서를 실제 화면 위치로 옮깁니다.
      interactionRuntime.setPointer(pointer, now);
      return;
    }

    if (cachedHands.length > 0) {
      // 저장된 좌표가 너무 오래됐으면 버리고 커서를 숨깁니다.
      cachedHands = [];
      handCursor.style.opacity = 0;
    }
  }

  // 이번 프레임에서 실제 손 인식을 새로 돌려야 하는지 판단합니다.
  // "새 카메라 프레임이 들어왔는가?" 와 "지정한 간격이 지났는가?"를 둘 다 봅니다.
  function shouldRunInference(now) {
    const hasFreshFrame = video.currentTime !== lastVideoTime && video.readyState >= 2 && video.videoWidth > 0;
    const inferenceDue = now - lastInferenceAt >= inferIntervalMs;
    return hasFreshFrame && inferenceDue;
  }

  // MediaPipe 가 돌려준 손 인식 결과를 받아서,
  // 현재 손이 있는지, 어디 있는지, 어떤 반응을 해야 하는지 다음 단계로 넘깁니다.
  function handleDetectionResult(result, now) {
    if (result.landmarks.length === 0) {
      // 손이 하나도 안 보이면 저장된 좌표를 지우고 UI도 초기 상태로 돌립니다.
      cachedHands = [];
      cachedLandmarksAt = 0;
      interactionRuntime.resetTrackingState();
      return;
    }

    // 여러 손이 감지될 수 있으므로 hands 배열로 받습니다.
    const hands = buildHandsWithKeys(result);
    const primaryHand = hands.find((hand) => hand.handKey === "right") || hands[0];
    const landmarks = primaryHand.landmarks;
    cachedHands = hands;
    cachedLandmarksAt = now;

    // 검지 끝을 화면 좌표로 바꾸고, 시작 버튼 hover 같은 UI 반응을 처리합니다.
    hands.forEach((hand) => {
      const pointer = interactionRuntime.createInstrumentPoint(hand.landmarks[8], handCanvas);
      interactionRuntime.processLandingHover(pointer, now);
    });

    // 모든 손의 손가락 끝 좌표들을 모아서 터치 충돌 판정에 사용합니다.
    const triggerPoints = hands.flatMap((hand) => {
      if (!hand.landmarks) return [];
      return [4, 8, 12, 16, 20].map((idx) => interactionRuntime.createInstrumentPoint(hand.landmarks[idx], handCanvas));
    });

    // 손가락 끝이 악기에 닿았는지 검사합니다.
    interactionRuntime.processInstrumentCollision(triggerPoints, now);
    // 비눗방울과도 닿았는지 검사합니다.
    interactionRuntime.processBubbleCollisions(triggerPoints);
    
    // 첫 번째 손 기준으로 제스처 판정과 커서 위치를 업데이트합니다.
    if (primaryHand && primaryHand.landmarks) {
      const pointer = interactionRuntime.createInstrumentPoint(primaryHand.landmarks[8], handCanvas);
      interactionRuntime.setPointer(pointer, now);

      hands.forEach((hand) => {
        if (hand.landmarks) {
          interactionRuntime.processGestureTriggers(hand.landmarks, now, hand.handKey);
        }
      });
    }
  }

  // predict() 는 requestAnimationFrame 으로 계속 반복되는 메인 루프입니다.
  // 한 프레임마다 손을 그릴지, 새 인식을 돌릴지, 이펙트를 갱신할지를 결정합니다.
  function predict() {
    try {
      const handLandmarker = getHandLandmarker();
      if (!handLandmarker) {
        requestAnimationFrame(predict);
        return;
      }

      const now = performance.now();
      handCtx.clearRect(0, 0, handCanvas.width, handCanvas.height);
      onBeforeFrame(now, getSessionStarted());
      renderCachedHand(now);

      if (shouldRunInference(now)) {
        lastVideoTime = video.currentTime;
        lastInferenceAt = now;
        const result = handLandmarker.detectForVideo(video, now);
        handleDetectionResult(result, now);
      }

      particleSystem.updateParticles();
    } catch (error) {
      onDetectionError(error);
    } finally {
      // 에러가 나더라도 루프는 중단되지 않도록 반드시 다음 프레임을 예약합니다.
      requestAnimationFrame(predict);
    }
  }

  // 바깥에서는 predict() 만 알면 이 추적 엔진을 시작할 수 있습니다.
  return {
    predict
  };
}
