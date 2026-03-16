// [scene_runtime.js] 배경 영상 루프, 씬 모드, 피버 상태처럼 화면 전체 분위기를 담당하는 모듈입니다.

// 배경 영상을 두 개 번갈아 재생해서, 영상 끝에서 끊기지 않게 자연스럽게 이어붙입니다.
export function setupSeamlessBackgroundLoop({ crossfadeSec, cleanupDelayMs = 520 } = {}) {
  const videoA = document.querySelector(".bg-video-a");
  const videoB = document.querySelector(".bg-video-b");
  if (!videoA || !videoB) return;

  const videos = [videoA, videoB];
  // 두 영상을 같은 조건으로 준비해두고, 하나는 현재 재생용, 하나는 대기용으로 씁니다.
  videos.forEach((video, idx) => {
    video.muted = true;
    video.playsInline = true;
    video.loop = false;
    video.playbackRate = 1;
    video.classList.toggle("is-active", idx === 0);
    video.classList.toggle("is-preload", idx !== 0);
  });

  let active = videoA;
  let standby = videoB;
  let rafId = 0;
  let isSwitching = false;
  let switchSeq = 0;

  // 브라우저 자동재생 정책 때문에 play() 가 실패할 수 있으므로 안전하게 감쌉니다.
  const safePlay = (video) => {
    const p = video.play();
    if (p && typeof p.catch === "function") p.catch(() => {});
  };

  // active 영상이 끝나기 직전이면 standby 영상을 앞으로 끌어와 부드럽게 교체합니다.
  const swap = () => {
    if (isSwitching) return;
    isSwitching = true;

    const fromVideo = active;
    const toVideo = standby;
    const currentSeq = ++switchSeq;

    toVideo.currentTime = 0;
    toVideo.classList.remove("is-preload");
    toVideo.classList.add("is-active");
    safePlay(toVideo);

    window.setTimeout(() => {
      if (switchSeq !== currentSeq || active !== fromVideo || standby !== toVideo) {
        isSwitching = false;
        return;
      }

      fromVideo.pause();
      fromVideo.currentTime = 0;
      fromVideo.classList.remove("is-active");
      fromVideo.classList.add("is-preload");
      active = toVideo;
      standby = fromVideo;
      isSwitching = false;
    }, cleanupDelayMs);
  };

  // requestAnimationFrame 으로 매 프레임마다 "영상 끝날 때가 됐는지" 감시합니다.
  const tick = () => {
    if (!active.paused) {
      const duration = Number(active.duration);
      if (Number.isFinite(duration) && duration > 0) {
        const remaining = duration - active.currentTime;
        if (Number.isFinite(remaining) && remaining <= crossfadeSec) {
          swap();
        }
      }
    }
    rafId = requestAnimationFrame(tick);
  };

  // 배경 루프를 실제로 시작하는 함수입니다.
  const start = () => {
    safePlay(active);
    cancelAnimationFrame(rafId);
    rafId = requestAnimationFrame(tick);
  };

  if (active.readyState >= 2) start();
  else active.addEventListener("canplay", start, { once: true });

  document.addEventListener("visibilitychange", () => {
    // 탭이 안 보일 때는 재생과 감시를 잠시 멈춰 자원을 아낍니다.
    if (document.hidden) {
      switchSeq += 1;
      isSwitching = false;
      active.pause();
      standby.pause();
      cancelAnimationFrame(rafId);
      return;
    }
    start();
  });
}

// URL 에서 받은 모드 이름을 scene 데이터 속성으로 반영합니다.
export function applySceneMode(scene, mode) {
  scene.dataset.mode = mode === "magic" ? "magic" : "calm";
}

// createFeverController(...) 는 피버 타임 전용 상태 관리기를 만들어줍니다.
export function createFeverController({
  scene,
  pulseMessage,
  statusText,
  triggerWindowMs,
  triggerHits,
  durationMs,
  getIdleStatusText
}) {
  // feverUntil 은 피버가 언제 끝나는지 기록하는 시간값입니다.
  let feverUntil = 0;
  // hitStamps 는 최근 상호작용 시각들을 모아두는 배열입니다.
  let hitStamps = [];

  // 피버 타임을 시작하고 문구와 scene 상태를 바꿉니다.
  function triggerFever(now) {
    feverUntil = now + durationMs;
    scene.dataset.fever = "on";
    pulseMessage.textContent = "피버 타임! 숲이 깨어났어요!";
    statusText.textContent = "피버 타임 진행 중 - 더 많이 터치해봐!";
  }

  // 최근 몇 초 안에 사용자가 얼마나 많이 상호작용했는지 세서 피버 조건을 검사합니다.
  function registerHit(now) {
    hitStamps.push(now);
    hitStamps = hitStamps.filter((ts) => now - ts <= triggerWindowMs);
    if (scene.dataset.fever === "off" && hitStamps.length >= triggerHits) {
      hitStamps = [];
      triggerFever(now);
    }
  }

  // 피버가 끝났는지 매 프레임 확인하고, 끝났으면 원래 상태로 돌립니다.
  function updateFeverState(now, sessionStarted) {
    if (feverUntil > now) return;
    if (scene.dataset.fever === "on") {
      scene.dataset.fever = "off";
      pulseMessage.textContent = "손을 잼잼! 해서 숲을 깨워봐!";
      if (sessionStarted) {
        statusText.textContent = getIdleStatusText();
      }
    }
  }

  return {
    registerHit,
    triggerFever,
    updateFeverState,
    isFever: () => scene.dataset.fever === "on"
  };
}
