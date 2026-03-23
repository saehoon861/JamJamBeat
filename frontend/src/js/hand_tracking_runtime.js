// [hand_tracking_runtime.js] MediaPipe 프레임 루프와 손 랜드마크 캐시 처리를 담당하는 모듈입니다.

// createHandTrackingRuntime(...) 은 "손 추적 엔진" 하나를 만들어 돌려주는 공장 함수입니다.
// 필요한 부품(video, canvas, renderer 등)을 바깥에서 받아서, 이 파일 안에서는 추적 흐름만 담당합니다.
export function createHandTrackingRuntime({
  video,
  handCanvas,
  handCtx,
  getVideo,
  getHandCanvas,
  getHandCtx,
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
  landmarkStaleMs,
  splitHandInference = false
}) {
  const PERF_ENABLED = (() => {
    const raw = new URLSearchParams(window.location.search).get("profilePerf");
    if (raw === "1" || raw === "true") return true;
    if (raw === "0" || raw === "false") return false;
    return Boolean(import.meta.env.DEV);
  })();
  const PERF_LOG_INTERVAL_MS = 2000;
  const SLOW_FRAME_WARN_MS = (() => {
    const raw = Number(new URLSearchParams(window.location.search).get("slowFrameWarnMs"));
    if (!Number.isFinite(raw)) return 40;
    return Math.max(16, Math.min(200, Math.round(raw)));
  })();
  const SLOW_FRAME_WARN_INTERVAL_MS = 1000;
  const INFERENCE_MAX_WIDTH = (() => {
    const raw = Number(new URLSearchParams(window.location.search).get("inferWidth"));
    if (!Number.isFinite(raw)) return 0;
    return Math.max(0, Math.min(640, Math.round(raw)));
  })();
  const ADAPTIVE_INTERVAL_MAX_MS = (() => {
    const raw = Number(new URLSearchParams(window.location.search).get("inferIntervalMaxMs"));
    if (!Number.isFinite(raw)) return 300;
    return Math.max(inferIntervalMs, Math.min(400, Math.round(raw)));
  })();
  const FORCE_INFERENCE_IDLE_MS = (() => {
    const raw = Number(new URLSearchParams(window.location.search).get("forceInferIdleMs"));
    if (!Number.isFinite(raw)) return 600;
    return Math.max(200, Math.min(2000, Math.round(raw)));
  })();
  const FRAME_SIGNAL_STALE_MS = (() => {
    const raw = Number(new URLSearchParams(window.location.search).get("frameSignalStaleMs"));
    if (!Number.isFinite(raw)) return 240;
    return Math.max(120, Math.min(1200, Math.round(raw)));
  })();
  const perfWindow = {
    startedAt: performance.now(),
    lastLogAt: performance.now(),
    frameCount: 0,
    inferenceCount: 0,
    forcedInferenceCount: 0,
    timeFallbackInferenceCount: 0,
    staleRenderCount: 0,
    emptyDetections: 0,
    handsDetectedMax: 0,
    predictTotalMs: 0,
    predictMaxMs: 0,
    detectTotalMs: 0,
    detectMaxMs: 0,
    handleTotalMs: 0,
    handleMaxMs: 0,
    renderCacheTotalMs: 0,
    renderCacheMaxMs: 0
  };

  function recordPerf(sumKey, maxKey, value) {
    if (!PERF_ENABLED) return;
    perfWindow[sumKey] += value;
    perfWindow[maxKey] = Math.max(perfWindow[maxKey], value);
  }

  function flushPerf(now) {
    if (!PERF_ENABLED || now - perfWindow.lastLogAt < PERF_LOG_INTERVAL_MS) return;
    const frameCount = Math.max(1, perfWindow.frameCount);
    const inferenceCount = Math.max(1, perfWindow.inferenceCount);
    console.info("[Perf][HandTracking]", {
      windowMs: Math.round(now - perfWindow.startedAt),
      frames: perfWindow.frameCount,
      inferences: perfWindow.inferenceCount,
      forcedInferences: perfWindow.forcedInferenceCount,
      timeFallbackInferences: perfWindow.timeFallbackInferenceCount,
      staleRenders: perfWindow.staleRenderCount,
      emptyDetections: perfWindow.emptyDetections,
      handsDetectedMax: perfWindow.handsDetectedMax,
      avgPredictMs: Number((perfWindow.predictTotalMs / frameCount).toFixed(2)),
      maxPredictMs: Number(perfWindow.predictMaxMs.toFixed(2)),
      avgDetectMs: Number((perfWindow.detectTotalMs / inferenceCount).toFixed(2)),
      maxDetectMs: Number(perfWindow.detectMaxMs.toFixed(2)),
      avgHandleMs: Number((perfWindow.handleTotalMs / inferenceCount).toFixed(2)),
      maxHandleMs: Number(perfWindow.handleMaxMs.toFixed(2)),
      avgRenderCacheMs: Number((perfWindow.renderCacheTotalMs / frameCount).toFixed(2)),
      maxRenderCacheMs: Number(perfWindow.renderCacheMaxMs.toFixed(2)),
      inferIntervalMs: adaptiveInferIntervalMs,
      splitHandInference
    });
    perfWindow.startedAt = now;
    perfWindow.lastLogAt = now;
    perfWindow.frameCount = 0;
    perfWindow.inferenceCount = 0;
    perfWindow.forcedInferenceCount = 0;
    perfWindow.timeFallbackInferenceCount = 0;
    perfWindow.staleRenderCount = 0;
    perfWindow.emptyDetections = 0;
    perfWindow.handsDetectedMax = 0;
    perfWindow.predictTotalMs = 0;
    perfWindow.predictMaxMs = 0;
    perfWindow.detectTotalMs = 0;
    perfWindow.detectMaxMs = 0;
    perfWindow.handleTotalMs = 0;
    perfWindow.handleMaxMs = 0;
    perfWindow.renderCacheTotalMs = 0;
    perfWindow.renderCacheMaxMs = 0;
  }

  const HAND_KEYS = ["left", "right"];
  const REGION_CONFIGS = [
    { handKey: "left", startXRatio: 0, endXRatio: 0.5 },
    { handKey: "right", startXRatio: 0.5, endXRatio: 1 }
  ];
  // lastVideoTime 은 "이 프레임을 이미 처리했는지" 확인하려고 저장하는 값입니다.
  let lastVideoTime = -1;
  // lastInferenceAt 은 마지막으로 손 인식을 돌린 시각입니다.
  let lastInferenceAt = 0;
  // cachedLandmarks 는 가장 최근에 찾은 손 좌표를 잠깐 기억해두는 저장소입니다.
  let cachedHands = [];
  // cachedLandmarksAt 은 위 좌표를 언제 저장했는지 기록합니다.
  let cachedLandmarksAt = 0;
  let lastHandRenderAt = 0;
  let lastCachedBubbleCollisionAt = 0;
  const HAND_RENDER_INTERVAL_MS = 33;
  const fullInferenceCanvas = document.createElement("canvas");
  const fullInferenceCtx = fullInferenceCanvas.getContext("2d", { willReadFrequently: true });
  const CACHED_BUBBLE_COLLISION_INTERVAL_MS = 48;
  let lastDetectTimestampMs = 0;
  let lastSlowWarnAt = 0;
  let adaptiveInferIntervalMs = inferIntervalMs;
  let lastFrameSignalAt = 0;
  let rvfcProbeArmed = false;
  let latestVideoFrameSignal = null;

  const resolveVideo = () => getVideo?.() || video;
  const resolveHandCanvas = () => getHandCanvas?.() || handCanvas;
  const resolveHandCtx = () => getHandCtx?.() || handCtx;

  function getNextDetectTimestamp(baseNow) {
    const normalized = Math.max(1, Math.round(baseNow));
    const nextTimestamp = Math.max(normalized, lastDetectTimestampMs + 1);
    lastDetectTimestampMs = nextTimestamp;
    return nextTimestamp;
  }

  function normalizeHandedness(result) {
    const raw = Array.isArray(result?.handednesses)
      ? result.handednesses
      : Array.isArray(result?.handedness)
        ? result.handedness
        : [];
    return raw.map((entry) => {
      const first = Array.isArray(entry) ? entry[0] : entry;
      const label = String(first?.displayName || first?.categoryName || "").trim().toLowerCase();
      if (label === "left" || label === "right") return label;
      return null;
    });
  }

  function resizeCanvasForSource(canvas, sourceWidth, sourceHeight, maxWidth = INFERENCE_MAX_WIDTH) {
    const scale = (maxWidth > 0 && sourceWidth > maxWidth) ? (maxWidth / sourceWidth) : 1;
    const width = Math.max(1, Math.round(sourceWidth * scale));
    const height = Math.max(1, Math.round(sourceHeight * scale));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    return { width, height };
  }

  function drawVideoRegionToCanvas(canvas, ctx, sx, sy, sourceWidth, sourceHeight) {
    const activeVideo = resolveVideo();
    const { width, height } = resizeCanvasForSource(canvas, sourceWidth, sourceHeight);
    if (!ctx || !activeVideo) return null;
    ctx.clearRect(0, 0, width, height);
    ctx.drawImage(activeVideo, sx, sy, sourceWidth, sourceHeight, 0, 0, width, height);
    return canvas;
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
    const renderStartedAt = PERF_ENABLED ? performance.now() : 0;
    const activeHandCanvas = resolveHandCanvas();
    const activeHandCtx = resolveHandCtx();
    const cacheFresh = cachedHands.length > 0 && now - cachedLandmarksAt <= landmarkStaleMs;
    if (cacheFresh) {
      if (PERF_ENABLED) perfWindow.staleRenderCount += 1;
      if (activeHandCanvas && activeHandCtx && now - lastHandRenderAt >= HAND_RENDER_INTERVAL_MS) {
        renderer.setHandAnimationActive?.(true);
        cachedHands.forEach((hand) => {
          const currentGesture = interactionRuntime.getCurrentGesture(hand.handKey);
          renderer.drawHand(activeHandCtx, hand.landmarks, activeHandCanvas, now * 0.001, hand.handKey, currentGesture);
        });
        lastHandRenderAt = now;
      }
      // 검지 끝 좌표를 손 커서 위치로 변환합니다.
      const primaryHand = cachedHands.find((hand) => hand.handKey === "right") || cachedHands[0];
      if (activeHandCanvas && now - lastCachedBubbleCollisionAt >= CACHED_BUBBLE_COLLISION_INTERVAL_MS) {
        // 저장된 좌표로는 충돌 검사 빈도를 낮춰도 체감 차이가 적어 부담을 줄입니다.
        const flickerPoints = [4, 8, 12, 16, 20].map((idx) => interactionRuntime.createInstrumentPoint(primaryHand.landmarks[idx], activeHandCanvas));
        interactionRuntime.processBubbleCollisions(flickerPoints);
        lastCachedBubbleCollisionAt = now;
      }

      if (activeHandCanvas) {
        const pointer = interactionRuntime.createInstrumentPoint(primaryHand.landmarks[8], activeHandCanvas);
        // 커서를 실제 화면 위치로 옮깁니다.
        interactionRuntime.setPointer(pointer, now, primaryHand.landmarks);
      }
      recordPerf("renderCacheTotalMs", "renderCacheMaxMs", performance.now() - renderStartedAt);
      return;
    }

    if (cachedHands.length > 0) {
      // 저장된 좌표가 너무 오래됐으면 버리고 커서를 숨깁니다.
      cachedHands = [];
      handCursor.style.opacity = 0;
    }
    renderer.setHandAnimationActive?.(false);
    recordPerf("renderCacheTotalMs", "renderCacheMaxMs", performance.now() - renderStartedAt);
  }

  function ensureVideoFrameSignalProbe() {
    const activeVideo = resolveVideo();
    if (rvfcProbeArmed) return;
    if (typeof activeVideo?.requestVideoFrameCallback !== "function") return;

    rvfcProbeArmed = true;
    const onVideoFrame = (_now, metadata = {}) => {
      const currentVideo = resolveVideo();
      if (Number.isFinite(metadata.mediaTime)) {
        latestVideoFrameSignal = metadata.mediaTime;
        lastFrameSignalAt = performance.now();
      }
      if (typeof currentVideo?.requestVideoFrameCallback === "function") {
        currentVideo.requestVideoFrameCallback(onVideoFrame);
      } else {
        rvfcProbeArmed = false;
      }
    };

    activeVideo.requestVideoFrameCallback(onVideoFrame);
  }

  // 이번 프레임에서 실제 손 인식을 새로 돌려야 하는지 판단합니다.
  // "새 카메라 프레임이 들어왔는가?" 와 "지정한 간격이 지났는가?"를 둘 다 봅니다.
  function shouldRunInference(now) {
    const activeVideo = resolveVideo();
    if (!(activeVideo?.readyState >= 2 && activeVideo.videoWidth > 0)) {
      return { run: false, reason: "not_ready" };
    }

    const currentVideoTime = Number.isFinite(latestVideoFrameSignal)
      ? latestVideoFrameSignal
      : (Number.isFinite(activeVideo.currentTime) ? activeVideo.currentTime : lastVideoTime);
    const hasFreshFrame = currentVideoTime !== lastVideoTime;
    if (lastFrameSignalAt <= 0) {
      lastFrameSignalAt = now;
    }
    if (hasFreshFrame) {
      lastFrameSignalAt = now;
    }

    const inferenceDue = now - lastInferenceAt >= adaptiveInferIntervalMs;
    if (hasFreshFrame && inferenceDue) {
      return { run: true, reason: "fresh_frame", currentVideoTime };
    }

    const frameSignalStale = now - lastFrameSignalAt >= FRAME_SIGNAL_STALE_MS;
    if (frameSignalStale && inferenceDue) {
      return { run: true, reason: "time_fallback", currentVideoTime };
    }

    const idleTooLong = now - lastInferenceAt >= FORCE_INFERENCE_IDLE_MS;
    if (idleTooLong) {
      return { run: true, reason: "idle_recovery", currentVideoTime };
    }

    return { run: false, reason: "throttled", currentVideoTime };
  }

  function updateAdaptiveInferInterval(detectElapsedMs) {
    const recommendedInterval = Math.round(
      Math.max(inferIntervalMs, Math.min(ADAPTIVE_INTERVAL_MAX_MS, detectElapsedMs * 1.2))
    );
    if (recommendedInterval > adaptiveInferIntervalMs) {
      adaptiveInferIntervalMs = recommendedInterval;
      return;
    }
    adaptiveInferIntervalMs = Math.max(
      inferIntervalMs,
      Math.round(adaptiveInferIntervalMs * 0.85 + recommendedInterval * 0.15)
    );
  }

  // MediaPipe 가 돌려준 손 인식 결과를 받아서,
  // 현재 손이 있는지, 어디 있는지, 어떤 반응을 해야 하는지 다음 단계로 넘깁니다.
  function handleDetectedHands(hands, now) {
    const activeHandCanvas = resolveHandCanvas();
    if (PERF_ENABLED) {
      perfWindow.handsDetectedMax = Math.max(perfWindow.handsDetectedMax, hands.length);
      if (hands.length === 0) perfWindow.emptyDetections += 1;
    }
    if (hands.length === 0) {
      // 손이 하나도 안 보이면 저장된 좌표를 지우고 UI도 초기 상태로 돌립니다.
      cachedHands = [];
      cachedLandmarksAt = 0;
      lastHandRenderAt = 0;
      renderer.setHandAnimationActive?.(false);
      interactionRuntime.resetTrackingState();
      return;
    }

    // 여러 손이 감지될 수 있으므로 hands 배열로 받습니다.
    renderer.setHandAnimationActive?.(hands.length > 0);
    const primaryHand = hands.find((hand) => hand.handKey === "right") || hands[0];
    const landmarks = primaryHand.landmarks;
    cachedHands = hands;
    cachedLandmarksAt = now;
    if (!activeHandCanvas) return;

    // 검지 끝을 화면 좌표로 바꾸고, 시작 버튼 hover 같은 UI 반응을 처리합니다.
    hands.forEach((hand) => {
      const pointer = interactionRuntime.createInstrumentPoint(hand.landmarks[8], activeHandCanvas);
      interactionRuntime.processLandingHover(pointer, now);
    });

    // 모든 손의 손가락 끝 좌표들을 모아서 터치 충돌 판정에 사용합니다.
    const triggerPoints = hands.flatMap((hand) => {
      if (!hand.landmarks) return [];
      return [4, 8, 12, 16, 20].map((idx) => interactionRuntime.createInstrumentPoint(hand.landmarks[idx], activeHandCanvas));
    });

    // 손가락 끝이 악기에 닿았는지 검사합니다.
    interactionRuntime.processInstrumentCollision(triggerPoints, now);
    // 비눗방울과도 닿았는지 검사합니다.
    interactionRuntime.processBubbleCollisions(triggerPoints);
    
    // 첫 번째 손 기준으로 제스처 판정과 커서 위치를 업데이트합니다.
    if (primaryHand && primaryHand.landmarks) {
      const pointer = interactionRuntime.createInstrumentPoint(primaryHand.landmarks[8], activeHandCanvas);
      interactionRuntime.setPointer(pointer, now, primaryHand.landmarks);

      hands.forEach((hand) => {
        if (hand.landmarks) {
          interactionRuntime.updateTrackedHandSnapshot?.(hand.landmarks, now, hand.handKey);
          if (String(hand.handKey || "").toLowerCase() === "left") return;
          interactionRuntime.processGestureTriggers(hand.landmarks, now, hand.handKey);
        }
      });
    }
  }

  // detectForVideo/detect 결과를 받아 처리합니다.
  function onLiveStreamResult(result) {
    const now = performance.now();
    const hands = buildHandsWithKeys(result);
    const handleStartedAt = PERF_ENABLED ? performance.now() : 0;
    handleDetectedHands(hands, now);
    recordPerf("handleTotalMs", "handleMaxMs", performance.now() - handleStartedAt);
  }

  // predict() 는 requestAnimationFrame 으로 계속 반복되는 메인 루프입니다.
  // 한 프레임마다 손을 그릴지, 새 인식을 돌릴지, 이펙트를 갱신할지를 결정합니다.
  function predict() {
    try {
      const activeVideo = resolveVideo();
      const activeHandCanvas = resolveHandCanvas();
      const activeHandCtx = resolveHandCtx();
      const predictStartedAt = PERF_ENABLED ? performance.now() : 0;
      const handLandmarker = getHandLandmarker();
      if (!handLandmarker) {
        requestAnimationFrame(predict);
        return;
      }

      const now = performance.now();
      if (PERF_ENABLED) perfWindow.frameCount += 1;
      if (activeHandCanvas && activeHandCtx) {
        activeHandCtx.clearRect(0, 0, activeHandCanvas.width, activeHandCanvas.height);
      }
      onBeforeFrame(now, getSessionStarted());
      renderCachedHand(now);

      ensureVideoFrameSignalProbe();
      const inferenceGate = shouldRunInference(now);
      if (inferenceGate.run) {
        if (Number.isFinite(inferenceGate.currentVideoTime)) {
          lastVideoTime = inferenceGate.currentVideoTime;
        }
        lastInferenceAt = now;
        if (PERF_ENABLED) perfWindow.inferenceCount += 1;
        if (PERF_ENABLED && inferenceGate.reason === "idle_recovery") {
          perfWindow.forcedInferenceCount += 1;
        }
        if (PERF_ENABLED && inferenceGate.reason === "time_fallback") {
          perfWindow.timeFallbackInferenceCount += 1;
        }
        const detectStartedAt = PERF_ENABLED ? performance.now() : 0;

        // tasks-vision JS API는 detectForVideo()/detect() 경로를 사용합니다.
        const inferenceCanvas = activeVideo
          ? drawVideoRegionToCanvas(fullInferenceCanvas, fullInferenceCtx, 0, 0, activeVideo.videoWidth, activeVideo.videoHeight)
          : null;
        if (inferenceCanvas) {
          try {
            const detectTimestamp = getNextDetectTimestamp(now);
            if (typeof handLandmarker.detectForVideo === "function") {
              const result = handLandmarker.detectForVideo(inferenceCanvas, detectTimestamp);
              onLiveStreamResult(result);
            } else if (typeof handLandmarker.detect === "function") {
              const result = handLandmarker.detect(inferenceCanvas);
              onLiveStreamResult(result);
            } else {
              throw new TypeError("MediaPipe HandLandmarker supports detectForVideo()/detect() only.");
            }
          } catch (detectError) {
            onDetectionError(detectError);
          }
        }

        const detectElapsedMs = performance.now() - detectStartedAt;
        recordPerf("detectTotalMs", "detectMaxMs", detectElapsedMs);
        updateAdaptiveInferInterval(detectElapsedMs);
      }

      particleSystem.updateParticles();
      const predictElapsedMs = performance.now() - predictStartedAt;
      recordPerf("predictTotalMs", "predictMaxMs", predictElapsedMs);
      if (PERF_ENABLED && predictElapsedMs >= SLOW_FRAME_WARN_MS && now - lastSlowWarnAt >= SLOW_FRAME_WARN_INTERVAL_MS) {
        lastSlowWarnAt = now;
        console.warn("[Perf][HandTracking][SlowFrame]", {
          predictMs: Number(predictElapsedMs.toFixed(2)),
          detectMaxMs: Number(perfWindow.detectMaxMs.toFixed(2)),
          handleMaxMs: Number(perfWindow.handleMaxMs.toFixed(2)),
          renderCacheMaxMs: Number(perfWindow.renderCacheMaxMs.toFixed(2)),
          inferIntervalMs: adaptiveInferIntervalMs,
          inferenceWidth: fullInferenceCanvas.width || 0,
          inferenceHeight: fullInferenceCanvas.height || 0
        });
      }
      flushPerf(now);
    } catch (error) {
      onDetectionError(error);
    } finally {
      // 에러가 나더라도 루프는 중단되지 않도록 반드시 다음 프레임을 예약합니다.
      requestAnimationFrame(predict);
    }
  }

  // 바깥에서는 predict() 와 onLiveStreamResult() 를 사용합니다.
  return {
    predict,
    onLiveStreamResult
  };
}
