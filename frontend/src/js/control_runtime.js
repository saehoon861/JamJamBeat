// [control_runtime.js] 시작/사운드 버튼 브리지와 오디오 언락, 성능 로그 저장을 담당합니다.

export function createControlRuntime({
  audioApi,
  statusText,
  interactionMode,
  ambientAudioEnabled,
  perfLogKey,
  perfLogLimit,
  getSessionStarted,
  getAdminEditMode,
  onActivateStart
}) {
  let perfLogBound = false;

  function readPerfLogs() {
    try {
      const raw = localStorage.getItem(perfLogKey);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  function appendPerfLog(entry) {
    const logs = readPerfLogs();
    logs.push(entry);
    if (logs.length > perfLogLimit) {
      logs.splice(0, logs.length - perfLogLimit);
    }
    try {
      localStorage.setItem(perfLogKey, JSON.stringify(logs));
    } catch {
      // 성능 로그 저장 실패가 메인 동작을 막지 않도록 무시합니다.
    }
  }

  function syncSoundButtonUI() {
    const state = audioApi.getAudioState();
    window.dispatchEvent(new CustomEvent("jamjam:audio-state", {
      detail: {
        running: Boolean(state.running)
      }
    }));
  }

  async function measureAudioUnlock(reason) {
    const startedAt = performance.now();
    let unlocked = false;
    try {
      unlocked = await audioApi.unlockAudioContext();
    } catch (error) {
      console.warn("Audio unlock failed:", error);
      unlocked = false;
    }
    const elapsed = performance.now() - startedAt;
    try {
      appendPerfLog({
        at: Date.now(),
        soundKey: unlocked ? "audio_unlock" : "audio_unlock_fail",
        playMode: "control",
        instrumentId: null,
        gestureLabel: "Unlock",
        gestureSource: reason,
        latencyMs: elapsed
      });
    } catch {
      // 로그 저장 실패는 무시하고 오디오 언락 결과만 반환합니다.
    }
    return unlocked;
  }

  function bind() {
    if (!perfLogBound) {
      window.addEventListener("jamjam:sound-played", (event) => {
        const detail = event.detail || {};
        appendPerfLog({
          at: Date.now(),
          soundKey: detail.soundKey || null,
          playMode: detail.playMode || null,
          instrumentId: detail.instrumentId || null,
          gestureLabel: detail.gestureLabel || null,
          gestureSource: detail.gestureSource || null,
          latencyMs: Number.isFinite(detail.latencyMs) ? detail.latencyMs : null
        });
      });
      perfLogBound = true;
    }

    const tryUnlockFromGesture = async () => {
      const unlocked = await measureAudioUnlock("first-input");
      if (!unlocked) return;
      if (getSessionStarted() && ambientAudioEnabled) audioApi.startAmbientLoop();
      else audioApi.stopAmbientLoop();
      syncSoundButtonUI();
    };

    const onFirstGesture = async () => {
      await tryUnlockFromGesture();
      if (audioApi.getAudioState().running) {
        window.removeEventListener("pointerdown", onFirstGesture);
        window.removeEventListener("keydown", onFirstGesture);
      }
    };

    window.addEventListener("pointerdown", onFirstGesture);
    window.addEventListener("keydown", onFirstGesture);

    window.addEventListener("jamjam:start-request", async () => {
      if (getAdminEditMode()) return;
      try {
        const unlocked = await measureAudioUnlock("start-button");
        if (unlocked) syncSoundButtonUI();
      } finally {
        onActivateStart();
      }
    });

    window.addEventListener("jamjam:sound-toggle-request", async () => {
      const state = audioApi.getAudioState();
      if (state.running) {
        audioApi.toggleSound();
        audioApi.stopAmbientLoop();
      } else {
        const unlocked = await measureAudioUnlock("sound-button");
        if (!unlocked) return;
        if (getSessionStarted() && ambientAudioEnabled) audioApi.startAmbientLoop();
        else audioApi.stopAmbientLoop();
        statusText.textContent = interactionMode === "gesture"
          ? "소리가 켜졌어요! 손동작으로 숲을 연주해 보세요."
          : "소리가 켜졌어요! 손으로 숲을 연주해 보세요.";
      }
      syncSoundButtonUI();
    });
  }

  return {
    bind,
    syncSoundButtonUI,
    appendPerfLog
  };
}
