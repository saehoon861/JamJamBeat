# FRONTEND_WEBCAM_SEQUENCE_MONITOR_CURRENT.md

## 개요

이 문서는 `frontend/`를 수정하지 않고 `frontend-test/`에서 현재 프론트엔드 실시간 추론 파이프라인을 관찰하기 위한 기준 문서입니다.

현재 확인된 사실:

- 실제 브라우저 진입은 `frontend/index.html` -> `frontend/src/react/main.jsx`
- React shell은 `frontend/src/react/App.jsx`
- legacy runtime boot는 `frontend/src/react/hooks/useLegacyMainRuntime.js` 가 `frontend/src/js/main.js` 를 동적 import 하는 방식
- 기본 추론 경로는 `seqModel=0/false`가 아니면 `frontend/src/js/model_inference_sequence.js`
- 실제 active runtime asset은 `frontend/public/runtime_sequence/*`
- 현재 로드 대상 모델은 `bundle_id=pos_scale_mlp_sequence_delta_20260319_162806`, `model_id=mlp_sequence_delta`
- `mlp-sequential-gamma`라는 이름은 현재 frontend runtime 자산명으로 직접 존재하지 않음
- `focal gamma`는 `frontend/public/runtime_sequence/config.json`에 기록되어 있지 않으므로 frontend 런타임 자산만으로는 확인 불가

## 현재 frontend에서 이미 구현된 것

- webcam + MediaPipe hand tracking
- sequence ONNX inference
- `pos_scale` 정규화
- `8프레임 warmup`
- `delta63` feature 생성
- `tau=0.85`
- `no_hand` / `warmup` / `ready` / `tau_neutralized`
- test mode에서 raw/final/source/inference ms/normalized preview 표시

## 현재 frontend에서 아직 약한 것

- top3 확률 표시
- runtime metadata를 한 눈에 보는 monitor 전용 뷰
- `mlp-sequential-gamma` 같은 별칭이 실제 어떤 runtime asset인지 식별하는 표시
- rolling latency summary (`avg/p50/p95`)

## 현재 active runtime asset 확인 결과

- `bundle_id`: `pos_scale_mlp_sequence_delta_20260319_162806`
- `model_id`: `mlp_sequence_delta`
- `mode`: `sequence`
- `dataset_key`: `pos_scale`
- `normalization_family`: `pos_scale`
- `seq_len`: `8`
- `tau`: `0.85`
- `checkpoint_fingerprint`: `2f9e81b7a142a0cd92707eb903fe4880f55cbcf920676f6433104ddf8c006d09`
- `focal gamma`: `Not recorded in runtime asset`

## frontend-test monitor 구현 범위

`frontend-test` 앱은 아래만 집중해서 보여줍니다.

- webcam preview
- raw landmarks overlay
- normalized `pos_scale` preview
- raw model prediction
- final gesture
- status
- frames collected
- top3 probs
- current inference ms
- rolling avg / p50 / p95
- bundle id / model id / seq_len / tau / runtime root / fingerprint

## 코드 부록

아래는 현재 frontend 기준 관련 파일 원문 전체 복사본입니다.

## React Shell

Path: `/home/user/projects/JamJamBeat-model3/frontend/src/react/App.jsx`

```jsx
import React from "react";
import { DEFAULT_LAYOUT } from "../js/instrument_layout.js";
import { useInstrumentLayout } from "./hooks/useInstrumentLayout.js";
import { useMainControls } from "./hooks/useMainControls.js";
import { useLegacyMainRuntime } from "./hooks/useLegacyMainRuntime.js";
import logoWebp from "../../assets/로고-removebg-preview.webp";
// import logoPng from "../../assets/로고-removebg-preview.png";
const logoPng = logoWebp; 
import batonImage from "../../assets/objects/지휘봉.png";
import backgroundVideo from "../../assets/objects/움직이는_동화_영상_만들기.mp4?url";
import handGesturesGuide from "../../assets/hand_gestures_guide.png";
import hedgehogCreditVideo from "../../assets/objects/고슴도치1.mov?url";
import penguinCreditVideo from "../../assets/objects/팽귄1.mov?url";

const INSTRUMENTS = [
  {
    id: "drum",
    elementId: "instrumentDrum",
    className: "instrument instrument-drum mvp-button",
    label: "고슴도치 드럼",
    animal: "🦔",
    instrument: "🥁"
  },
  {
    id: "xylophone",
    elementId: "instrumentXylophone",
    className: "instrument instrument-xylophone mvp-button",
    label: "여우 실로폰",
    animal: "🦊",
    instrument: "🎹"
  },
  {
    id: "tambourine",
    elementId: "instrumentTambourine",
    className: "instrument instrument-tambourine mvp-button",
    label: "토끼 탬버린",
    animal: "🐰",
    instrument: "🪘"
  },
  {
    id: "a",
    elementId: "instrumentA",
    className: "instrument instrument-squirrel mvp-button",
    label: "다람쥐 심벌즈",
    animal: "🐿️",
    instrument: "🔔"
  },
  {
    id: "cat",
    elementId: "instrumentCat",
    className: "instrument instrument-cat mvp-button",
    label: "고양이 하트",
    animal: "🐱",
    instrument: "💖"
  },
  {
    id: "penguin",
    elementId: "instrumentPenguin",
    className: "instrument instrument-penguin mvp-button",
    label: "펭귄 기타",
    animal: "🐧",
    instrument: "🎸"
  }
];

const CREDIT_GROUPS = [
  {
    title: "Project",
    lines: ["JamJamBeat"]
  },
  {
    title: "Contributors",
    lines: ["surya2347", "saehoon861", "RohGyuMin", "OpenClaw Assistant"]
  },
  {
    title: "Built With",
    lines: ["React", "Vite", "MediaPipe Hands", "ONNX Runtime Web", "Web Audio API"]
  },
  {
    title: "Special Thanks",
    lines: ["숲의 지휘자가 되어준 모든 플레이어들"]
  }
];

function getInstrumentStyle(id, isMvpButton) {
  // MVP 버튼은 그리드 레이아웃 사용하므로 위치 스타일 불필요
  if (isMvpButton) return undefined;

  const position = DEFAULT_LAYOUT[id];
  if (!position) return undefined;

  return {
    left: `${position.x}vw`,
    bottom: `${position.y}vh`,
    right: "auto"
  };
}

function InstrumentButton({ instrument }) {
  const isMvpButton = instrument.className.includes('mvp-button');

  return (
    <button
      id={instrument.elementId}
      className={instrument.className}
      type="button"
      aria-label={instrument.label}
      style={getInstrumentStyle(instrument.id, isMvpButton)}
    >
      <div className="mvp-button-content">
        <span className="mvp-animal">{instrument.animal}</span>
        <span className="mvp-instrument">{instrument.instrument}</span>
      </div>
    </button>
  );
}

export default function App() {
  useInstrumentLayout();
  useLegacyMainRuntime();
  const { soundButtonLabel, requestStart, requestSoundToggle } = useMainControls();
  const [showTutorial, setShowTutorial] = React.useState(false);
  const [showCredits, setShowCredits] = React.useState(false);
  const [tutorialPracticeEnabled, setTutorialPracticeEnabled] = React.useState(false);

  const handleStartClick = () => {
    setTutorialPracticeEnabled(false);
    setShowTutorial(true);
  };

  const closeTutorial = () => {
    setShowTutorial(false);
    if (!tutorialPracticeEnabled) {
      requestStart();
    }
  };

  const startTutorialPractice = () => {
    if (!tutorialPracticeEnabled) {
      requestStart();
      setTutorialPracticeEnabled(true);
    }
  };

  const skipTutorialAndStart = () => {
    setShowTutorial(false);
    if (!tutorialPracticeEnabled) {
      requestStart();
    }
  };

  const openCredits = () => {
    setShowCredits(true);
  };

  const closeCredits = () => {
    setShowCredits(false);
  };

  const handleExitClick = () => {
    window.close();
    window.setTimeout(() => {
      if (!window.closed) {
        window.location.replace("about:blank");
      }
    }, 120);
  };

  React.useEffect(() => {
    if (!showTutorial || tutorialPracticeEnabled) {
      window.dispatchEvent(new CustomEvent("jamjam:refresh-camera-target"));
    }
  }, [showTutorial, tutorialPracticeEnabled]);

  return (
    <>
      <svg style={{ position: "absolute", width: 0, height: 0 }} aria-hidden="true">
        <defs>
          <filter id="remove-white" colorInterpolationFilters="sRGB">
            <feColorMatrix
              type="matrix"
              values={`
                1 0 0 0 0
                0 1 0 0 0
                0 0 1 0 0
                -1 -1 -1 2 0
              `}
            />
          </filter>
          <linearGradient id="batonGradient" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" style={{ stopColor: "#8ab4a0", stopOpacity: 0.3 }} />
            <stop offset="70%" style={{ stopColor: "#b4ffe6", stopOpacity: 0.85 }} />
            <stop offset="100%" style={{ stopColor: "#c8ffdc", stopOpacity: 0.95 }} />
          </linearGradient>
        </defs>
      </svg>

      {showTutorial && (
        <section className="tutorial-overlay" aria-live="polite">
          <div className="tutorial-modal">
            <header className="tutorial-header">
              <span className="tutorial-icon" aria-hidden="true">✨</span>
              <h2 className="tutorial-title">숲의 지휘자 연습실</h2>
            </header>
            
            <p className="tutorial-description">
              나의 손동작을 카메라에 비춰보세요!<br/>
              6가지 동작을 따라 하면 숲의 음악이 시작됩니다.
            </p>

            <div className="tutorial-interactive-area">
              <div className="tutorial-image-container">
                <img 
                  src={handGesturesGuide} 
                  alt="6가지 손동작 가이트" 
                  className="tutorial-image" 
                />
              </div>
              
              <div className="tutorial-mirror-zone">
                {tutorialPracticeEnabled ? (
                  <>
                    <div className="camera-shell tutorial-camera-shell" aria-hidden="true">
                      <video id="webcam" autoPlay playsInline muted />
                    </div>
                    <canvas id="handCanvas" className="tutorial-hand-canvas" />
                  </>
                ) : (
                  <div className="tutorial-camera-placeholder">
                    <p>연습하기를 누르면 카메라가 여기서 바로 켜집니다.</p>
                  </div>
                )}
              </div>
            </div>

            <div className="tutorial-action-stack">
              <button
                type="button"
                className="tutorial-practice-button"
                onClick={startTutorialPractice}
                disabled={tutorialPracticeEnabled}
              >
                {tutorialPracticeEnabled ? "카메라 켜짐, 손동작 연습 중" : "카메라 켜고 연습하기"}
              </button>
              <p className="tutorial-practice-note">
                연습을 건너뛰고 바로 시작할 수도 있습니다.
              </p>
              <div className="tutorial-action-row">
                <button
                  type="button"
                  className="tutorial-skip-button"
                  onClick={skipTutorialAndStart}
                >
                  연습 안 하고 바로 시작
                </button>
                <button
                  type="button"
                  className="tutorial-close-button"
                  onClick={closeTutorial}
                  disabled={!tutorialPracticeEnabled}
                >
                  연습 완료! 모험 시작하기
                </button>
              </div>
            </div>
          </div>
        </section>
      )}

      {showCredits && (
        <section className="credits-overlay" aria-live="polite" onClick={closeCredits}>
          <div className="credits-side-video credits-side-video-left" aria-hidden="true">
            <video autoPlay muted loop playsInline preload="auto">
              <source src={hedgehogCreditVideo} type="video/quicktime" />
            </video>
          </div>
          <div className="credits-side-video credits-side-video-right" aria-hidden="true">
            <video autoPlay muted loop playsInline preload="auto">
              <source src={penguinCreditVideo} type="video/quicktime" />
            </video>
          </div>
          <div
            className="credits-modal"
            role="dialog"
            aria-modal="true"
            aria-label="만든 사람들"
            onClick={(event) => event.stopPropagation()}
          >
            <header className="credits-header">
              <p className="credits-kicker">만든 사람들</p>
              <h2 className="credits-title">JamJamBeat Credits</h2>
            </header>

            <div className="credits-roll-viewport" aria-hidden="true">
              <div className="credits-roll-track">
                {CREDIT_GROUPS.map((group) => (
                  <section key={group.title} className="credits-group">
                    <h3>{group.title}</h3>
                    {group.lines.map((line) => (
                      <p key={line}>{line}</p>
                    ))}
                  </section>
                ))}
                <section className="credits-group credits-group-end">
                  <p>Thank you for playing.</p>
                </section>
              </div>
            </div>

            <button type="button" className="credits-close-button" onClick={closeCredits}>
              닫기
            </button>
          </div>
        </section>
      )}

      <main id="scene" className="scene hide-camera" data-fever="off">
        <section id="landingOverlay" className="landing-overlay" aria-live="polite">
          <div className="landing-panel">
            <div className="landing-logo-container">
              <img className="landing-logo" src={logoPng} srcSet={logoWebp} alt="JamJam Beat Logo" loading="eager" />
            </div>
            <button id="landingStartButton" type="button" onClick={handleStartClick}>시작하기</button>
            <button className="landing-credits-button" type="button" onClick={openCredits}>만든 사람들</button>
            <button className="landing-exit-button" type="button" onClick={handleExitClick}>종료하기</button>
          </div>
        </section>

        <section className="layer layer-background" aria-hidden="true">
          <div className="bg-video-wrap" aria-hidden="true">
            <video className="bg-video bg-video-a is-active" muted playsInline preload="auto" autoPlay loop>
              <source src={backgroundVideo} type="video/mp4" />
            </video>
            <video className="bg-video bg-video-b is-preload" muted playsInline preload="auto" autoPlay loop>
              <source src={backgroundVideo} type="video/mp4" />
            </video>
          </div>
          <svg className="forest-silhouette" viewBox="0 0 1600 900" preserveAspectRatio="xMidYMax slice" role="img">
            <title>Layered forest silhouette background</title>
            <defs>
              <linearGradient id="skyGrad" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="#1a1630" />
                <stop offset="100%" stopColor="#182b2d" />
              </linearGradient>
            </defs>
            <rect width="1600" height="900" fill="url(#skyGrad)" />
            <path d="M0 660 Q120 560 220 660 T460 660 T700 650 T960 670 T1260 655 T1600 660 L1600 900 L0 900 Z" fill="#111c24" />
            <path d="M0 720 Q180 640 360 715 T760 720 T1120 710 T1600 720 L1600 900 L0 900 Z" fill="#0f1621" />
            <g fill="#102418" opacity="0.95">
              <path d="M120 900 L120 620 L90 620 L90 900 Z" />
              <path d="M160 900 L160 585 L130 585 L130 900 Z" />
              <path d="M1240 900 L1240 610 L1210 610 L1210 900 Z" />
              <path d="M1288 900 L1288 570 L1262 570 L1262 900 Z" />
            </g>
          </svg>
          <div className="fog fog-a" />
          <div className="fog fog-b" />
          <div className="fog fog-c" />
        </section>

        <section id="instrumentLayer" className="layer layer-middleground" aria-label="인터랙티브 악기 숲">
          {INSTRUMENTS.map((instrument) => (
            <InstrumentButton key={instrument.id} instrument={instrument} />
          ))}
        </section>

        <section className="layer layer-foreground" aria-live="polite">
          {!showTutorial && (
            <>
              <div className="camera-shell" aria-hidden="true">
                <video id="webcam" autoPlay playsInline muted />
              </div>
              <canvas id="handCanvas" />
            </>
          )}
          <div className="guide-silhouette" aria-hidden="true" />
          <div id="gestureSquirrelEffect" className="gesture-squirrel" aria-hidden="true">🐿️</div>
          <p id="pulseMessage" className="pulse-message">손을 잼잼! 해서 숲을 깨워봐!</p>
        </section>

        <section className="hud" aria-live="polite">
          <h2>잼잼비트 숲의 입구</h2>
          <p id="status" className="status">카메라 준비 중...</p>
          <button id="soundUnlockButton" className="sound-unlock" type="button" onClick={requestSoundToggle}>{soundButtonLabel}</button>
          <button id="testModeToggleButton" className="test-mode-toggle" type="button">테스트 모드 켜기</button>
          <button className="hud-exit-button" type="button" onClick={handleExitClick}>프로그램 끝내기</button>
          <a className="hud-main-link" href="./index.html">메인화면으로</a>
          <p className="privacy-note">영상은 서버로 전송되지 않고 오직 연주에만 사용됩니다.</p>
        </section>

        <section id="adminControls" className="admin-floating-controls" aria-live="polite">
          <p>관리자 편집 모드: 오브젝트를 드래그해 배치</p>
          <div>
            <button id="adminSaveButton" type="button">배치 저장</button>
            <button id="adminResetButton" type="button">기본 위치</button>
            <button type="button" onClick={() => history.back()}>뒤로가기</button>
            <a href="./index.html">메인화면으로</a>
          </div>
        </section>

        <canvas id="effectCanvas" aria-hidden="true" />
        <div id="handCursor" className="hand-cursor" aria-hidden="true">
          <img src={batonImage} alt="" aria-hidden="true" />
        </div>

        <section id="testModeVision" className="test-mode-vision is-hidden" aria-live="polite">
          <div className="test-mode-vision-head">
            <p className="test-mode-eyebrow">손 입력 미리보기</p>
            <p id="testModeVisionSummary" className="test-mode-vision-summary">원본 손과 모델 입력 좌표를 비교합니다.</p>
          </div>
          <div className="test-mode-vision-grid">
            <article className="test-mode-vision-card">
              <h3>원본 손</h3>
              <div className="test-mode-preview-shell">
                <video id="testModeWebcamPreview" autoPlay playsInline muted aria-hidden="true" />
                <canvas id="testModeRawCanvas" width="256" height="144" aria-hidden="true" />
              </div>
            </article>
            <article className="test-mode-vision-card">
              <h3>모델 입력</h3>
              <canvas id="testModeNormalizedCanvas" className="test-mode-normalized-canvas" width="220" height="220" aria-hidden="true" />
            </article>
          </div>
        </section>

        <section id="testModePanel" className="test-mode-panel is-hidden" aria-live="polite">
          <div className="test-mode-head">
            <p className="test-mode-eyebrow">실시간 테스트 모드</p>
            <p id="testModeSummary" className="test-mode-summary">카메라/모델 상태를 관찰합니다.</p>
          </div>

          <dl className="test-mode-grid">
            <dt>세션</dt><dd id="testModeSession">-</dd>
            <dt>감지 손</dt><dd id="testModeHands">-</dd>
            <dt>모델</dt><dd id="testModeModel">-</dd>
            <dt>최근 추론</dt><dd id="testModeInFlight">-</dd>
            <dt>마지막 추론</dt><dd id="testModeLastInference">-</dd>
          </dl>

          <div className="test-mode-columns">
            <article className="test-mode-card">
              <h3>왼손</h3>
              <dl className="test-mode-grid">
                <dt>Raw</dt><dd id="testModeLeftRaw">-</dd>
                <dt>Final</dt><dd id="testModeLeftFinal">-</dd>
                <dt>Source</dt><dd id="testModeLeftSource">-</dd>
                <dt>오브젝트</dt><dd id="testModeLeftObject">-</dd>
                <dt>추론 ms</dt><dd id="testModeLeftInferenceMs">-</dd>
                <dt>소리 ms</dt><dd id="testModeLeftSoundMs">-</dd>
                <dt>루프</dt><dd id="testModeLeftMelody">-</dd>
              </dl>
            </article>

            <article className="test-mode-card">
              <h3>오른손</h3>
              <dl className="test-mode-grid">
                <dt>Raw</dt><dd id="testModeRightRaw">-</dd>
                <dt>Final</dt><dd id="testModeRightFinal">-</dd>
                <dt>Source</dt><dd id="testModeRightSource">-</dd>
                <dt>오브젝트</dt><dd id="testModeRightObject">-</dd>
                <dt>추론 ms</dt><dd id="testModeRightInferenceMs">-</dd>
                <dt>소리 ms</dt><dd id="testModeRightSoundMs">-</dd>
                <dt>루프</dt><dd id="testModeRightMelody">-</dd>
              </dl>
            </article>
          </div>
        </section>
      </main>
    </>
  );
}

```

## Legacy Runtime Boot Hook

Path: `/home/user/projects/JamJamBeat-model3/frontend/src/react/hooks/useLegacyMainRuntime.js`

```js
import { useEffect, useRef } from "react";

export function useLegacyMainRuntime() {
  const bootedRef = useRef(false);

  useEffect(() => {
    if (bootedRef.current) return;
    bootedRef.current = true;

    let cancelled = false;

    import("../../js/main.js").catch((error) => {
      if (cancelled) return;
      console.error("Failed to boot legacy main runtime:", error);
    });

    return () => {
      cancelled = true;
    };
  }, []);
}

```

## Legacy Main Runtime

Path: `/home/user/projects/JamJamBeat-model3/frontend/src/js/main.js`

```js
// [main.js] 이 프로그램의 '두뇌' 역할을 하는 가장 중요한 파일입니다.
// 카메라를 켜고, 내 손이 어디 있는지 찾아내고, 그 손이 악기에 닿았을 때 소리를 내라고 명령하는 모든 과정을 지휘합니다.

import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision"; // 미디어파이프(손 인식 도구)를 가져옵니다.
import * as Audio from "./audio.js"; // 소리 재생 관련 기능을 가져옵니다.
import * as Renderer from "./renderer.js"; // 화면 그리기 관련 기능을 가져옵니다.
import { resolveGesture, setModelPredictionProvider, resetGestureState } from "./gestures.js"; // 손동작 인식 로직을 가져옵니다.

// 시퀀스 모델 사용 여부를 URL 파라미터로 결정 (기본값: 시퀀스 모델 사용)
const USE_SEQUENCE_MODEL = (() => {
  const params = new URLSearchParams(window.location.search);
  const seqModel = params.get("seqModel");
  // seqModel=0 또는 false면 기존 모델, 그 외에는 시퀀스 모델 사용
  if (seqModel === "0" || seqModel === "false") return false;
  return true; // 기본값: 시퀀스 모델 사용
})();

// 모델 추론 함수들 (init에서 동적 로딩 후 할당됨)
let getModelPrediction = null;
let getModelInferenceStatus = null;
const TEST_MODE_HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17]
];
const TEST_MODE_HAND_COLORS = {
  left: {
    stroke: "rgba(113, 220, 255, 0.95)",
    fill: "rgba(214, 248, 255, 0.95)"
  },
  right: {
    stroke: "rgba(255, 204, 113, 0.95)",
    fill: "rgba(255, 244, 214, 0.95)"
  }
};
import { getConfiguredHandLandmarkerTaskPath, getConfiguredMediaPipeWasmRoot, getConfiguredSplitHandInference } from "./env_config.js";
import { setupSeamlessBackgroundLoop, applySceneMode } from "./scene_runtime.js";
import { createParticleSystem, restartClassAnimation } from "./particle_system.js";
import { DEFAULT_SOUND_MAPPING, loadSoundMapping, getSoundProfileForInstrument, loadGestureMapping } from "./sound_mapping.js";
import { createInteractionRuntime } from "./interaction_runtime.js";
import { createHandTrackingRuntime } from "./hand_tracking_runtime.js";
import { createControlRuntime } from "./control_runtime.js";

// CSS 기반 애니메이션 시스템 사용 (Lottie 대신 경량 CSS 애니메이션)
async function loadAnimationManagerFactory() {
  console.log("[Animation] Using CSS-based animation system");
  const animModule = await import("./instrument_animations_simple.js");
  return animModule.createInstrumentAnimationManager;
}

function createNoopAnimationManager() {
  return {
    initAnimation() {},
    setState() {},
    trigger() {},
    hover() {},
    setFeverMode() {},
    updateProximity() {},
    setPaused() {},
    destroy() {},
    getState() { return "idle"; },
    isLoaded() { return false; }
  };
}

function createNoopFeverController() {
  return {
    registerHit() {},
    triggerFever() {},
    updateFeverState() {},
    isFever: () => false
  };
}

// 화면에 보이는 각종 버튼, 글자, 캔버스(그림판)들을 컴퓨터가 찾을 수 있게 주소를 연결해두는 곳입니다.
let video = null; // 현재 활성화된 웹캠 비디오 태그를 가리킵니다.
let handCanvas = null; // 현재 활성화된 손 오버레이 캔버스를 가리킵니다.
const effectCanvas = document.getElementById("effectCanvas"); // 터치 효과(반짝임)를 그릴 투명 도화지를 가져옵니다.
let handCtx = null; // 현재 손 캔버스의 2D context 입니다.
const effectCtx = effectCanvas.getContext("2d"); // 효과 도화지에 그림을 그리기 위한 도구(붓)를 꺼냅니다.
const statusText = document.getElementById("status"); // 현재 상태(안내 문구)를 보여줄 글자 칸을 가져옵니다.
const scene = document.getElementById("scene"); // 전체 배경이 되는 공간을 가져옵니다.
const handCursor = document.getElementById("handCursor"); // 내 손가락 끝을 따라다닐 동그란 커서를 가져옵니다.
const landingOverlay = document.getElementById("landingOverlay"); // 처음 시작할 때 보이는 덮개 화면을 가져옵니다.
const landingStartButton = document.getElementById("landingStartButton"); // 시작하기 버튼을 가져옵니다.
const pulseMessage = document.getElementById("pulseMessage"); // 화면 중앙에 뜨는 안내 메시지를 가져옵니다.
const gestureSquirrelEffect = document.getElementById("gestureSquirrelEffect"); // 다람쥐 효과 이미지를 가져옵니다.
const testModeToggleButton = document.getElementById("testModeToggleButton");
const testModePanel = document.getElementById("testModePanel");
const testModeVision = document.getElementById("testModeVision");
const testModeSummary = document.getElementById("testModeSummary");
const testModeVisionSummary = document.getElementById("testModeVisionSummary");
const testModeSession = document.getElementById("testModeSession");
const testModeHands = document.getElementById("testModeHands");
const testModeModel = document.getElementById("testModeModel");
const testModeInFlight = document.getElementById("testModeInFlight");
const testModeLastInference = document.getElementById("testModeLastInference");
const testModeWebcamPreview = document.getElementById("testModeWebcamPreview");
const testModeRawCanvas = document.getElementById("testModeRawCanvas");
const testModeNormalizedCanvas = document.getElementById("testModeNormalizedCanvas");
const testModeRawCtx = testModeRawCanvas ? testModeRawCanvas.getContext("2d") : null;
const testModeNormalizedCtx = testModeNormalizedCanvas ? testModeNormalizedCanvas.getContext("2d") : null;
const testModeLeftRaw = document.getElementById("testModeLeftRaw");
const testModeLeftFinal = document.getElementById("testModeLeftFinal");
const testModeLeftSource = document.getElementById("testModeLeftSource");
const testModeLeftObject = document.getElementById("testModeLeftObject");
const testModeLeftInferenceMs = document.getElementById("testModeLeftInferenceMs");
const testModeLeftSoundMs = document.getElementById("testModeLeftSoundMs");
const testModeLeftMelody = document.getElementById("testModeLeftMelody");
const testModeRightRaw = document.getElementById("testModeRightRaw");
const testModeRightFinal = document.getElementById("testModeRightFinal");
const testModeRightSource = document.getElementById("testModeRightSource");
const testModeRightObject = document.getElementById("testModeRightObject");
const testModeRightInferenceMs = document.getElementById("testModeRightInferenceMs");
const testModeRightSoundMs = document.getElementById("testModeRightSoundMs");
const testModeRightMelody = document.getElementById("testModeRightMelody");
const testModeFieldEls = {
  left: {
    raw: testModeLeftRaw,
    final: testModeLeftFinal,
    source: testModeLeftSource,
    object: testModeLeftObject,
    inferenceMs: testModeLeftInferenceMs,
    soundMs: testModeLeftSoundMs,
    melody: testModeLeftMelody
  },
  right: {
    raw: testModeRightRaw,
    final: testModeRightFinal,
    source: testModeRightSource,
    object: testModeRightObject,
    inferenceMs: testModeRightInferenceMs,
    soundMs: testModeRightSoundMs,
    melody: testModeRightMelody
  }
};

function syncRuntimeDomRefs() {
  video = document.getElementById("webcam");
  handCanvas = document.getElementById("handCanvas");
  handCtx = handCanvas ? handCanvas.getContext("2d") : null;
  return { video, handCanvas, handCtx };
}

syncRuntimeDomRefs();

const instrumentElements = { // 각 동물 악기들의 HTML 요소를 하나로 묶어둡니다.
  drum: document.getElementById("instrumentDrum"), // 고슴도치 드럼 DOM입니다.
  xylophone: document.getElementById("instrumentXylophone"), // 아기 사슴 오브젝트를 찾습니다.
  tambourine: document.getElementById("instrumentTambourine"), // 아기 토끼 오브젝트를 찾습니다.
  a: document.getElementById("instrumentA"), // 다람쥐 DOM입니다.
  cat: document.getElementById("instrumentCat"), // 고양이 DOM입니다.
  penguin: document.getElementById("instrumentPenguin") // 팽귄 DOM입니다.
};
const VIDEO_INSTRUMENT_IDS = ["drum", "penguin"];
const videoInstruments = {}; // { [id]: { video, canvas, ctx, workCanvas, workCtx, lastFrameAt, raf } }

let gestureObjectActive = false;
const VIDEO_RENDER_FPS = 15; 
const VIDEO_RENDER_INTERVAL_MS = 1000 / VIDEO_RENDER_FPS;
const VIDEO_PROCESS_MAX_DIM = 480; 

const VIDEO_BLACK_THRESHOLD = 58;
const VIDEO_SOFT_BLACK_THRESHOLD = 96;

const COLLISION_PADDING = 12; // 손이 악기에 완전히 닿지 않아도 조금 근처에만 가도 인식되게 하는 '여유 공간'입니다.
const START_HOVER_MS = 220; // 시작 버튼 위에 손을 얼마나 오래 올려두어야 게임이 시작되는지 결정하는 시간(0.52초)입니다.
const ENABLE_AMBIENT_AUDIO = false; // 동작 인식 시에만 소리가 나도록 배경 앰비언트는 끕니다.

const DEFAULT_INFER_FPS = 15; // 별도 설정이 없을 때 1초에 몇 번 손 위치를 계산할지 나타냅니다. (기본 15회)
const MIN_INFER_FPS = 12; // 너무 느려지지 않게 최대로 늦출 수 있는 계산 횟수입니다.
const MAX_INFER_FPS = 60; // 컴퓨터 성능이 좋아도 최대 60회까지만 계산하도록 제한합니다.
const LANDMARK_STALE_MS = 260; // 손이 화면에서 사라졌을 때 약 0.26초 동안은 마지막 위치를 기억하고 보여줍니다.

let handLandmarker; // 미디어파이프 손 인식 도구 객체를 담아둘 변수입니다.
let sessionStarted = false; // 게임이 실제로 시작되었는지 여부를 나타냅니다.
let cameraStream = null; // 웹캠에서 나오는 영상 신호를 담아둡니다.
let adminEditMode = new URLSearchParams(window.location.search).get("admin") === "1"; // 현재 악기 배치를 수정하는 관리자 모드인지 확인합니다.

const GESTURE_TRIGGER_COOLDOWN_MS = 280; // 손동작 인식이 너무 자주 일어나지 않게 하는 대기 시간(0.28초)입니다.
const BG_VIDEO_CROSSFADE_SEC = 0.42; // 배경 영상이 바뀔 때 자연스럽게 겹치는 시간(0.42초)입니다.
const PERF_LOG_KEY = "jamjam.perf.logs.v1";
const PERF_LOG_LIMIT = 200;

const SOUND_PROFILES = {
  drum: { soundTag: "드럼 비트", burstType: "drum", playbackMode: "melody", melodyType: "drum", play: (note) => Audio.playKids_Drum(note) },
  piano: { soundTag: "피아노 선율", burstType: "xylophone", playbackMode: "melody", melodyType: "piano", play: (note) => Audio.playKids_Piano(note) },
  guitar: { soundTag: "기타 스트럼", burstType: "tambourine", playbackMode: "melody", melodyType: "guitar", play: (note) => Audio.playKids_Guitar(note) },
  flute: { soundTag: "플룻 멜로디", burstType: "heart", playbackMode: "melody", melodyType: "flute", play: (note) => Audio.playKids_Flute(note) },
  violin: { soundTag: "바이올린 하모니", burstType: "animal", playbackMode: "melody", melodyType: "violin", play: (note) => Audio.playKids_Violin(note) },
  bell: { soundTag: "벨 포인트", burstType: "pinky", playbackMode: "melody", melodyType: "bell", play: (note) => Audio.playKids_Bell(note) }
};

// GESTURE_SOUND_PROFILES는 제거됨 - gestureMapping과 SOUND_PROFILES 조합으로 대체

// 주소창(URL)에 적힌 설정값을 보고, 손 위치를 얼마나 자주 계산할지(FPS) 결정하는 기능입니다.
function parseInferFps() {
  const params = new URLSearchParams(window.location.search); // 주소창의 파라미터(?표 뒤의 글자들)를 읽어옵니다.
  const raw = Number(params.get("inferFps")); // 'inferFps'라는 이름의 숫자를 가져옵니다.
  if (!Number.isFinite(raw)) return DEFAULT_INFER_FPS; // 숫자가 아니면 기본값(15)을 사용합니다.
  return clamp(Math.round(raw), MIN_INFER_FPS, MAX_INFER_FPS); // 설정된 범위(8~60) 안으로 숫자를 맞춥니다.
}

const INFER_INTERVAL_MS = Math.round(1000 / parseInferFps()); // 계산 횟수를 보고 몇 밀리초(초의 1000분의 1)마다 계산할지 정합니다.

function parsePreferredDelegate() { // 인공지능 계산을 CPU로 할지 GPU(그래픽카드)로 할지 정하는 기능입니다.
  const params = new URLSearchParams(window.location.search); // 주소창 파라미터를 읽습니다.
  const raw = (params.get("mpDelegate") || "gpu").trim().toUpperCase(); // 'mpDelegate' 값을 가져와 대문자로 바꿉니다. 기본은 GPU입니다.
  return raw === "CPU" ? "CPU" : "GPU"; // CPU가 아니면 무조건 GPU를 사용하도록 합니다.
}

function parseInteractionMode() { // 터치로 할지 손동작으로 할지 플레이 방식을 정하는 기능입니다.
  const params = new URLSearchParams(window.location.search); // 주소창 파라미터를 읽습니다.
  const raw = (params.get("interactionMode") || "gesture").trim().toLowerCase(); // 'interactionMode' 값을 읽습니다. 기본은 손동작입니다.
  return raw === "touch" ? "touch" : "gesture"; // touch 가 아니면 gesture 방식을 사용합니다.
}

function parseNumHands() {
  const params = new URLSearchParams(window.location.search);
  const raw = Number(params.get("numHands"));
  if (!Number.isFinite(raw)) return 2;
  return clamp(Math.round(raw), 1, 2);
}

// 프로그램이 동작하면서 기억해야 할 '현재 상태' 값들입니다. (예: 지금 카메라가 켜져 있는지, 피버 타임인지 등)
const INTERACTION_MODE = parseInteractionMode();
const NUM_HANDS = parseNumHands();
const HAND_DETECTION_TARGET = NUM_HANDS;
const ENABLE_SPLIT_HAND_INFERENCE = getConfiguredSplitHandInference();
let soundMapping = loadSoundMapping(SOUND_PROFILES);
const gestureMapping = loadGestureMapping();
const particleSystem = createParticleSystem(effectCtx, effectCanvas);
let animationManager = createNoopAnimationManager();
const feverController = createNoopFeverController();
const lastSoundEventByHand = new Map();

let testModeEnabled = (() => {
  const params = new URLSearchParams(window.location.search);
  const queryValue = params.get("testMode");
  if (queryValue === "1" || queryValue === "true") return true;
  if (queryValue === "0" || queryValue === "false") return false;
  return false;
})();

function formatDisplayGesture(label, confidence = null, classId = null) {
  const normalized = String(label || "").trim().toLowerCase();
  let displayLabel = "아무것도 아님";
  if (normalized === "fist" || classId === 1) displayLabel = "주먹";
  else if (normalized === "openpalm" || normalized === "open_palm" || normalized === "open palm" || classId === 2) displayLabel = "손바닥";
  else if (normalized === "v" || classId === 3) displayLabel = "브이";
  else if (normalized === "pinky" || classId === 4) displayLabel = "새끼손가락";
  else if (normalized === "animal" || classId === 5) displayLabel = "애니멀";
  else if (normalized === "kheart" || normalized === "k-heart" || classId === 6) displayLabel = "K-하트";
  else if (normalized && normalized !== "none" && normalized !== "class0") displayLabel = label;

  if (!Number.isFinite(confidence)) return displayLabel;
  return `${displayLabel} ${(confidence * 100).toFixed(0)}%`;
}

function getInstrumentName(instrumentId) {
  const instrument = instruments.find((item) => item.id === instrumentId);
  return instrument?.name || "-";
}

function formatMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}ms` : "-";
}

function setPanelValue(element, value) {
  if (!element) return;
  element.textContent = value;
}

function getTestModeHandColor(handKey = "default") {
  return TEST_MODE_HAND_COLORS[handKey] || {
    stroke: "rgba(255, 255, 255, 0.92)",
    fill: "rgba(245, 245, 245, 0.92)"
  };
}

function getMirrorPivotX(landmarks) {
  const wristX = Number.isFinite(landmarks?.[0]?.x) ? landmarks[0].x : 0.5;
  const indexMcpX = Number.isFinite(landmarks?.[5]?.x) ? landmarks[5].x : wristX;
  const pinkyMcpX = Number.isFinite(landmarks?.[17]?.x) ? landmarks[17].x : wristX;
  return (wristX + indexMcpX + pinkyMcpX) / 3;
}

function sanitizePreviewLandmarks(landmarks, handKey = "default") {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return null;
  const normalizedHandKey = String(handKey || "default").trim().toLowerCase();
  const shouldMirrorLeft = normalizedHandKey === "left";
  const mirrorPivotX = shouldMirrorLeft ? getMirrorPivotX(landmarks) : 0;
  const features = new Float32Array(63);
  for (let i = 0; i < 21; i += 1) {
    const point = landmarks[i];
    const rawX = Number.isFinite(point?.x) ? point.x : 0;
    const baseOffset = i * 3;
    features[baseOffset] = shouldMirrorLeft ? clamp(mirrorPivotX * 2 - rawX, 0, 1) : rawX;
    features[baseOffset + 1] = Number.isFinite(point?.y) ? point.y : 0;
    features[baseOffset + 2] = Number.isFinite(point?.z) ? point.z : 0;
  }
  return features;
}

function normalizeSequencePreviewFrame(frame63) {
  if (!frame63 || frame63.length < 63) return null;
  const normalized = new Float32Array(63);
  const originX = frame63[0];
  const originY = frame63[1];
  const originZ = frame63[2];
  const dx = frame63[27] - originX;
  const dy = frame63[28] - originY;
  const dz = frame63[29] - originZ;
  const denom = Math.hypot(dx, dy, dz);
  const scale = denom <= 1e-8 ? 1 : 1 / denom;

  for (let i = 0; i < 63; i += 3) {
    normalized[i] = (frame63[i] - originX) * scale;
    normalized[i + 1] = (frame63[i + 1] - originY) * scale;
    normalized[i + 2] = (frame63[i + 2] - originZ) * scale;
  }

  return normalized;
}

function drawHandGraph(ctx, points, handKey, { pointRadius = 4, lineWidth = 2.2 } = {}) {
  if (!ctx || !Array.isArray(points) || points.length < 21) return;
  const { stroke, fill } = getTestModeHandColor(handKey);

  ctx.save();
  ctx.lineWidth = lineWidth;
  ctx.lineJoin = "round";
  ctx.lineCap = "round";
  ctx.strokeStyle = stroke;
  ctx.shadowBlur = 10;
  ctx.shadowColor = stroke;

  TEST_MODE_HAND_CONNECTIONS.forEach(([from, to]) => {
    const start = points[from];
    const end = points[to];
    if (!start || !end) return;
    ctx.beginPath();
    ctx.moveTo(start.x, start.y);
    ctx.lineTo(end.x, end.y);
    ctx.stroke();
  });

  ctx.fillStyle = fill;
  ctx.shadowBlur = 0;
  points.forEach((point, index) => {
    if (!point) return;
    ctx.beginPath();
    ctx.arc(point.x, point.y, index === 8 ? pointRadius + 1.4 : pointRadius, 0, Math.PI * 2);
    ctx.fill();
  });
  ctx.restore();
}

function drawNormalizedPreviewGuide(ctx, width, height) {
  if (!ctx) return;
  ctx.save();
  ctx.strokeStyle = "rgba(255, 255, 255, 0.14)";
  ctx.lineWidth = 1;
  ctx.setLineDash([4, 5]);
  ctx.beginPath();
  ctx.moveTo(width * 0.5, 14);
  ctx.lineTo(width * 0.5, height - 14);
  ctx.moveTo(14, height * 0.5);
  ctx.lineTo(width - 14, height * 0.5);
  ctx.stroke();
  ctx.restore();
}

function renderTestModeVision(debugSnapshot, handKeys) {
  if (!testModeEnabled) return;
  if (!testModeRawCanvas || !testModeNormalizedCanvas || !testModeRawCtx || !testModeNormalizedCtx) return;

  const rawWidth = testModeRawCanvas.width;
  const rawHeight = testModeRawCanvas.height;
  const normalizedWidth = testModeNormalizedCanvas.width;
  const normalizedHeight = testModeNormalizedCanvas.height;

  testModeRawCtx.clearRect(0, 0, rawWidth, rawHeight);
  testModeNormalizedCtx.clearRect(0, 0, normalizedWidth, normalizedHeight);
  drawNormalizedPreviewGuide(testModeNormalizedCtx, normalizedWidth, normalizedHeight);

  const summary = USE_SEQUENCE_MODEL
    ? "좌: 원본 손 / 우: wrist 원점 + scale 정규화"
    : "좌: 원본 손 / 우: 모델 입력 좌표(미러링 포함)";
  setPanelValue(testModeVisionSummary, summary);

  handKeys.forEach((handKey) => {
    const hand = debugSnapshot[handKey];
    const landmarks = hand?.lastLandmarks;
    if (!Array.isArray(landmarks) || landmarks.length < 21) return;

    const rawPoints = landmarks.map((point) => ({
      x: (1 - point.x) * rawWidth,
      y: point.y * rawHeight
    }));
    drawHandGraph(testModeRawCtx, rawPoints, handKey, { pointRadius: 3.4, lineWidth: 2 });

    const sanitized = sanitizePreviewLandmarks(landmarks, handKey);
    const modelInput = USE_SEQUENCE_MODEL ? normalizeSequencePreviewFrame(sanitized) : sanitized;
    if (!modelInput) return;

    const normalizedPoints = [];
    if (USE_SEQUENCE_MODEL) {
      let maxAbs = 0.25;
      for (let i = 0; i < 63; i += 3) {
        maxAbs = Math.max(maxAbs, Math.abs(modelInput[i]), Math.abs(modelInput[i + 1]));
      }
      const scale = (Math.min(normalizedWidth, normalizedHeight) * 0.34) / maxAbs;
      for (let i = 0; i < 63; i += 3) {
        normalizedPoints.push({
          x: normalizedWidth * 0.5 + modelInput[i] * scale,
          y: normalizedHeight * 0.5 + modelInput[i + 1] * scale
        });
      }
    } else {
      for (let i = 0; i < 63; i += 3) {
        normalizedPoints.push({
          x: modelInput[i] * normalizedWidth,
          y: modelInput[i + 1] * normalizedHeight
        });
      }
    }
    drawHandGraph(testModeNormalizedCtx, normalizedPoints, handKey, { pointRadius: 3.8, lineWidth: 2.1 });
  });
}

function syncTestModeUI() {
  if (testModeToggleButton) {
    testModeToggleButton.textContent = testModeEnabled ? "테스트 모드 끄기" : "테스트 모드 켜기";
    testModeToggleButton.setAttribute("aria-pressed", String(testModeEnabled));
  }
  if (testModePanel) {
    testModePanel.classList.toggle("is-hidden", !testModeEnabled);
  }
  if (testModeVision) {
    testModeVision.classList.toggle("is-hidden", !testModeEnabled);
  }
}

function renderTestModePanel() {
  if (!testModeEnabled) return;

  const debugSnapshot = interactionRuntime.getDebugSnapshot?.() || {};
  const modelStatus = getModelInferenceStatus(performance.now());
  const handKeys = Object.keys(debugSnapshot).filter((handKey) => {
    const hand = debugSnapshot[handKey];
    return Boolean(hand?.lastUpdatedAt);
  });

  setPanelValue(testModeSummary, "Raw와 Final을 동시에 보며 추론/후처리를 구분합니다.");
  setPanelValue(testModeSession, sessionStarted ? "시작됨" : "대기");
  setPanelValue(testModeHands, handKeys.length > 0 ? handKeys.join(", ") : "없음");
  setPanelValue(testModeModel, modelStatus.endpointConfigured ? `${modelStatus.mode} ready` : "loading");
  setPanelValue(
    testModeInFlight,
    modelStatus.recentInference
      ? `Yes${Number.isFinite(modelStatus.lastCompletedAgoMs) ? ` (${Math.round(modelStatus.lastCompletedAgoMs)}ms 전)` : ""}`
      : "No"
  );
  setPanelValue(testModeLastInference, formatMs(modelStatus.lastDurationMs));

  ["left", "right"].forEach((handKey) => {
    const hand = debugSnapshot[handKey];
    const rawModel = hand?.lastRawModelPrediction || null;
    const resolved = hand?.lastResolvedGesture || null;
    const soundEvent = lastSoundEventByHand.get(handKey) || null;
    const instrumentId = resolved?.label && resolved.label !== "None"
      ? (gestureMapping[resolved.label] || null)
      : null;
    const fields = testModeFieldEls[handKey];
    if (!fields) return;

    setPanelValue(fields.raw, formatDisplayGesture(rawModel?.label, rawModel?.confidence, rawModel?.classId ?? null));
    setPanelValue(fields.final, formatDisplayGesture(resolved?.label, resolved?.confidence));
    setPanelValue(fields.source, resolved?.source || "-");
    setPanelValue(fields.object, instrumentId ? getInstrumentName(instrumentId) : "-");
    setPanelValue(fields.inferenceMs, formatMs(rawModel?.elapsed_ms));
    setPanelValue(fields.soundMs, formatMs(soundEvent?.inferenceLatencyMs));
    setPanelValue(fields.melody, hand?.currentMelodyType || "-");
  });

  renderTestModeVision(debugSnapshot, handKeys);
}

function startTestModeLoop() {
  const tick = () => {
    renderTestModePanel();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function getMappedSoundProfile(instrumentId) {
  return getSoundProfileForInstrument(soundMapping, DEFAULT_SOUND_MAPPING, SOUND_PROFILES, instrumentId);
}

function playMappedInstrumentSound(instrumentId, element, { note, spawnEffect = true } = {}) {
  const profile = getMappedSoundProfile(instrumentId);
  profile.play(note);
  if (spawnEffect && element) {
    spawnBurst(profile.burstType, element);
  }
  return profile;
}

function getGestureSoundProfile(label, instrumentId) {
  // gestureMapping에서 제스쳐에 맞는 오브젝트를 찾고,
  // 그 오브젝트의 사운드 프로필을 반환
  return getMappedSoundProfile(instrumentId);
}

function playGestureMappedSound(label, instrumentId, { note, spawnEffect = true } = {}) {
  const element = instrumentElements[instrumentId] || null;
  const profile = getGestureSoundProfile(label, instrumentId);
  profile.play(note);
  if (spawnEffect && element) {
    spawnBurst(profile.burstType, element);
  }
  return profile;
}

// 우리가 연주할 수 있는 '동물 악기'들의 정보입니다. 이름과 소리, 그리고 닿았을 때 어떤 행동을 할지 적혀 있습니다.
const instruments = [
  {
    id: "drum", // 악기의 고유 ID입니다.
    name: "고슴도치 드럼", // 화면에 보일 실제 이름입니다.
    soundTag: "쿵", // 연주했을 때 표시될 소리 느낌표입니다.
    el: instrumentElements.drum, // 실제 HTML 이미지를 연결합니다.
    cooldownMs: 320, // 한 번 연주한 뒤 다시 연주하기 위해 기다려야 하는 시간입니다.
    lastHitAt: 0, // 마지막으로 연주된 시간을 기록합니다.
    onHit(note) { // 연주되었을 때 실행할 행동입니다.
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id); // Lottie 애니메이션 트리거
      return profile.soundTag || this.soundTag;
    }
  },
  {
    id: "xylophone", // 실로폰 악기입니다.
    name: "아기 사슴", // 이름입니다.
    soundTag: "사슴 멜로디", // 글자 표시입니다.
    el: instrumentElements.xylophone, // 이미지를 찾습니다.
    cooldownMs: 360, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit(note) { // 누르면 실행합니다.
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id); // Lottie 애니메이션 트리거
      return profile.soundTag || this.soundTag;
    }
  },
  {
    id: "tambourine", // 탬버린 악기입니다.
    name: "아기 토끼", // 이름입니다.
    soundTag: "토끼 리듬", // 글자 표시입니다.
    el: instrumentElements.tambourine, // 이미지를 찾습니다.
    cooldownMs: 380, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit(note) { // 누르면 실행합니다.
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id); // Lottie 애니메이션 트리거
      return profile.soundTag || this.soundTag;
    }
  },
  {
    id: "a", // 다람쥐입니다.
    name: "다람쥐", // 이름입니다.
    soundTag: "다람쥐 포인트", // 글자 표시입니다.
    el: instrumentElements.a, // 이미지를 찾습니다.
    cooldownMs: 380, // 대기 시간입니다.
    lastHitAt: 0, // 시간 기록입니다.
    onHit(note) { // 누르면 실행합니다.
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id); // Lottie 애니메이션 트리거
      return profile.soundTag || this.soundTag;
    }
  },
  {
    id: "cat",
    name: "고양이",
    soundTag: "고양이 멜로디",
    el: instrumentElements.cat,
    cooldownMs: 380,
    lastHitAt: 0,
    onHit(note) {
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id);
      return profile.soundTag || this.soundTag;
    }
  },
  {
    id: "penguin",
    name: "팽귄 실로폰",
    soundTag: "팽귄 선율",
    el: instrumentElements.penguin,
    cooldownMs: 380,
    lastHitAt: 0,
    onHit(note) {
      const profile = playMappedInstrumentSound(this.id, this.el, { note });
      animationManager.trigger(this.id);
      return profile.soundTag || this.soundTag;
    }
  }
];

// 브라우저 화면 크기가 바뀔 때마다 그림판(캔버스)의 크기도 똑같이 맞춰주는 기능입니다.
function setCanvasSize() {
  const { handCanvas: activeHandCanvas } = syncRuntimeDomRefs();
  if (activeHandCanvas) {
    activeHandCanvas.width = window.innerWidth; // 손 도화지 가로 길이를 창 크기에 맞춥니다.
    activeHandCanvas.height = window.innerHeight; // 손 도화지 세로 길이를 창 크기에 맞춥니다.
  }
  effectCanvas.width = window.innerWidth; // 효과 도화지 가로 길이를 창 크기에 맞춥니다.
  effectCanvas.height = window.innerHeight; // 효과 도화지 세로 길이를 창 크기에 맞춥니다.
}

// 숫자가 너무 작거나 너무 크지 않게 특정 범위 안에만 있도록 잡아주는 기능입니다.
function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

// 화면 오른쪽 아래의 '소리 켜기/끄기' 버튼의 글자를 현재 상태에 맞춰 바꿔주는 기능입니다.
function activateStart() { // 게임을 실제로 시작하는 기능입니다.
  sessionStarted = true; // 세션이 시작되었음을 표시합니다.
  landingOverlay.classList.add("is-hidden"); // 시작 화면 덮개를 숨깁니다.
  const audioState = Audio.getAudioState(); // 현재 오디오 상태를 가져옵니다.
  const playGuide = INTERACTION_MODE === "gesture" // 플레이 방식에 따라 안내 문구를 정합니다.
    ? "손동작으로 숲을 연주해 보세요."
    : "손으로 동물 악기를 터치해 보세요.";
  if (audioState.running) { // 오디오가 켜져 있으면
    statusText.textContent = playGuide; // 안내 문구를 보여줍니다.
    if (ENABLE_AMBIENT_AUDIO) {
      Audio.startAmbientLoop(); // 숲의 배경음을 재생하기 시작합니다.
    } else {
      Audio.stopAmbientLoop();
    }
  } else { // 오디오가 꺼져 있으면
    statusText.textContent = "소리를 들으려면 '소리 켜기' 버튼을 눌러주세요."; // 소리를 켜라는 메시지를 보여줍니다.
  }
}

function registerHit(now) {
  feverController.registerHit(now);
}

function spawnBurst(type, element) {
  particleSystem.spawnBurst(type, element);
}

function getVideoProcessSize(width, height) {
  if (!width || !height) return { processWidth: 0, processHeight: 0 };
  const scale = Math.min(1, VIDEO_PROCESS_MAX_DIM / Math.max(width, height));
  return {
    processWidth: Math.max(1, Math.round(width * scale)),
    processHeight: Math.max(1, Math.round(height * scale))
  };
}

function syncVideoInstrumentSize(id) {
  const inst = videoInstruments[id];
  if (!inst || !inst.video || !inst.canvas) return;
  const { video, canvas, workCanvas } = inst;

  const width = video.videoWidth || 0;
  const height = video.videoHeight || 0;
  if (!width || !height) return;

  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const { processWidth, processHeight } = getVideoProcessSize(width, height);
  if (workCanvas.width !== processWidth || workCanvas.height !== processHeight) {
    workCanvas.width = processWidth;
    workCanvas.height = processHeight;
  }
}

function drawVideoInstrumentFrame(id, now = performance.now()) {
  const inst = videoInstruments[id];
  if (!inst || !inst.video || !inst.canvas || !inst.ctx || !inst.workCtx) return;

  const { video, canvas, ctx, workCanvas, workCtx, lastFrameAt } = inst;

  if (now - lastFrameAt < VIDEO_RENDER_INTERVAL_MS) {
    inst.raf = requestAnimationFrame((t) => drawVideoInstrumentFrame(id, t));
    return;
  }
  inst.lastFrameAt = now;

  syncVideoInstrumentSize(id);

  const displayWidth = canvas.width || video.videoWidth || 0;
  const displayHeight = canvas.height || video.videoHeight || 0;
  const processWidth = workCanvas.width || 0;
  const processHeight = workCanvas.height || 0;

  if (!displayWidth || !displayHeight || !processWidth || !processHeight || video.readyState < 2) {
    inst.raf = requestAnimationFrame((t) => drawVideoInstrumentFrame(id, t));
    return;
  }

  workCtx.clearRect(0, 0, processWidth, processHeight);
  workCtx.drawImage(video, 0, 0, processWidth, processHeight);
  const frame = workCtx.getImageData(0, 0, processWidth, processHeight);
  const pixels = frame.data;

  for (let i = 0; i < pixels.length; i += 4) {
    const r = pixels[i];
    const g = pixels[i + 1];
    const b = pixels[i + 2];
    const brightness = Math.max(r, g, b);

    if (brightness <= VIDEO_BLACK_THRESHOLD) {
      pixels[i + 3] = 0;
      continue;
    }

    if (brightness < VIDEO_SOFT_BLACK_THRESHOLD) {
      const alphaScale = (brightness - VIDEO_BLACK_THRESHOLD) / (VIDEO_SOFT_BLACK_THRESHOLD - VIDEO_BLACK_THRESHOLD);
      pixels[i + 3] = Math.round(pixels[i + 3] * alphaScale);
    }
  }

  workCtx.putImageData(frame, 0, 0);
  ctx.clearRect(0, 0, displayWidth, displayHeight);
  ctx.drawImage(workCanvas, 0, 0, displayWidth, displayHeight);
  inst.raf = requestAnimationFrame((t) => drawVideoInstrumentFrame(id, t));
}

function ensureVideoRenderLoop(id) {
  const inst = videoInstruments[id];
  if (!inst || !inst.video) return;
  if (inst.raf) return;
  inst.raf = requestAnimationFrame((t) => drawVideoInstrumentFrame(id, t));
}

function syncVideoPlayback(id) {
  const inst = videoInstruments[id];
  if (!inst || !inst.video) return;
  const { video } = inst;
  video.muted = true;
  video.loop = true;
  video.playsInline = true;
  const playPromise = video.play();
  if (playPromise && typeof playPromise.catch === "function") {
    playPromise.catch(() => {});
  }
  ensureVideoRenderLoop(id);
}

function setGestureObjectVariant(isGestureActive, instrumentId = "drum") {
  const inst = videoInstruments[instrumentId];
  if (!inst || !inst.video) return;
  const { video, canvas } = inst;

  const nextSrc = isGestureActive
    ? video.dataset.variantGesture
    : video.dataset.variantBase;

  const isGlobalActive = isGestureActive || Object.values(videoInstruments).some(v => v.video?.dataset.active === "true"); // simplistic check
  // For now, let's just toggle the specific one
  canvas?.classList.toggle("is-gesture-variant", isGestureActive);
  
  if (!nextSrc) return;

  const currentSrc = video.getAttribute("src") || "";
  if (currentSrc !== nextSrc) {
    video.setAttribute("src", nextSrc);
    video.load();
  }

  syncVideoPlayback(instrumentId);
}

function wait(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

function stopCameraTracks(stream) {
  if (!stream || typeof stream.getTracks !== "function") return;
  stream.getTracks().forEach((track) => {
    try {
      track.stop();
    } catch {}
  });
}

function clearCameraSource() {
  const { video: activeVideo } = syncRuntimeDomRefs();
  if (activeVideo?.srcObject && typeof activeVideo.srcObject.getTracks === "function") {
    stopCameraTracks(activeVideo.srcObject);
  }
  if (cameraStream && cameraStream !== activeVideo?.srcObject) {
    stopCameraTracks(cameraStream);
  }

  cameraStream = null;
  try {
    activeVideo?.pause();
  } catch {}
  if (activeVideo) {
    activeVideo.srcObject = null;
  }
  if (testModeWebcamPreview) {
    try {
      testModeWebcamPreview.pause();
    } catch {}
    testModeWebcamPreview.srcObject = null;
  }
}

async function initCamera() {
  try {
    const { video: activeVideo } = syncRuntimeDomRefs();
    if (!activeVideo) {
      statusText.textContent = "카메라 화면을 준비하는 중입니다...";
      return;
    }
    console.info("[MediaPipe] initCamera:start");
    statusText.textContent = "카메라를 준비하는 중입니다...";

    clearCameraSource();

    const cameraAttempts = [
      {
        label: "detailed",
        constraints: {
          width: { ideal: 640 },
          height: { ideal: 360 },
          frameRate: { ideal: 30, max: 30 }
        }
      },
      {
        label: "compat",
        constraints: {
          width: { ideal: 640 },
          height: { ideal: 360 },
          frameRate: { ideal: 24, max: 30 }
        }
      },
      {
        label: "basic",
        constraints: true
      }
    ];

    let stream = null;
    let lastError = null;

    for (let index = 0; index < cameraAttempts.length; index += 1) {
      const attempt = cameraAttempts[index];
      try {
        stream = await navigator.mediaDevices.getUserMedia({ video: attempt.constraints });
        console.info("[MediaPipe] initCamera:getUserMedia " + attempt.label + " success");
        break;
      } catch (attemptError) {
        lastError = attemptError;
        const errorName = String(attemptError?.name || "Error");
        console.warn("[MediaPipe] initCamera:getUserMedia " + attempt.label + " failed", {
          name: errorName,
          message: String(attemptError?.message || "")
        });

        if (errorName === "NotAllowedError" || errorName === "SecurityError" || errorName === "NotFoundError") {
          break;
        }

        if (errorName === "NotReadableError") {
          clearCameraSource();
          await wait(200 * (index + 1));
        }
      }
    }

    if (!stream) {
      throw lastError || new Error("Could not start video source");
    }

    activeVideo.srcObject = stream;
    cameraStream = stream;
    activeVideo.playsInline = true;
    activeVideo.muted = true;
    activeVideo.setAttribute("playsinline", "");
    if (testModeWebcamPreview) {
      testModeWebcamPreview.srcObject = stream;
      testModeWebcamPreview.playsInline = true;
      testModeWebcamPreview.muted = true;
      testModeWebcamPreview.setAttribute("playsinline", "");
    }
    activeVideo.onloadedmetadata = () => {
      console.info("[MediaPipe] initCamera:loadedmetadata", {
        width: activeVideo.videoWidth,
        height: activeVideo.videoHeight,
        readyState: activeVideo.readyState
      });
      activeVideo.play().catch(() => {});
      if (testModeWebcamPreview) {
        testModeWebcamPreview.play().catch(() => {});
      }
      statusText.textContent = handLandmarker
        ? "준비 완료! 시작 버튼에 손을 올려주세요."
        : "카메라 준비 완료! 손 인식 모델을 불러오는 중입니다.";
      trackingRuntime.predict();
    };
  } catch (error) {
    console.error("[MediaPipe] initCamera:failed", error);
    const errorName = String(error?.name || "Error");
    if (errorName === "NotAllowedError" || errorName === "SecurityError") {
      statusText.textContent = "카메라 권한을 허용해 주세요.";
    } else if (errorName === "NotFoundError") {
      statusText.textContent = "사용 가능한 카메라를 찾지 못했습니다.";
    } else if (errorName === "NotReadableError") {
      statusText.textContent = "카메라를 시작할 수 없습니다. 다른 앱에서 카메라 사용 중인지 확인해 주세요.";
    } else {
      statusText.textContent = "카메라를 시작하지 못했습니다. 잠시 후 다시 시도해 주세요.";
    }
    statusText.style.color = "var(--danger)";
  }
}

async function initMediaPipe() {
  console.info("[MediaPipe] init:start", {
    delegate: parsePreferredDelegate(),
    numHands: HAND_DETECTION_TARGET,
    splitHands: ENABLE_SPLIT_HAND_INFERENCE
  });
  statusText.textContent = "손 인식 모델을 불러오는 중입니다...";
  const vision = await FilesetResolver.forVisionTasks(getConfiguredMediaPipeWasmRoot());
  console.info("[MediaPipe] init:vision tasks loaded");
  const modelAssetPath = getConfiguredHandLandmarkerTaskPath();
  const preferredDelegate = parsePreferredDelegate(); // 먼저 시도할 계산 장치를 정합니다.
  const fallbackDelegate = preferredDelegate === "GPU" ? "CPU" : "GPU"; // 실패했을 때 쓸 대체 장치도 준비합니다.

  try {
    console.info("[MediaPipe] init:createFromOptions primary", {
      delegate: preferredDelegate,
      modelAssetPath,
      runningMode: "VIDEO"
    });
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath, // 사용할 모델 파일 경로입니다.
        delegate: preferredDelegate // 첫 번째 선택 장치로 계산합니다.
      },
      runningMode: "VIDEO", // 실시간 스트림 분석 모드입니다.
      numHands: HAND_DETECTION_TARGET,
      minHandDetectionConfidence: 0.25,
      minHandPresenceConfidence: 0.25,
      minTrackingConfidence: 0.25
    });
    console.info("[MediaPipe] init:createFromOptions primary success", {
      constructor: handLandmarker?.constructor?.name,
      hasDetectForVideo: typeof handLandmarker?.detectForVideo === "function",
      runningMode: "VIDEO"
    });
  } catch (delegateError) {
    console.warn(`${preferredDelegate} delegate failed, fallback to ${fallbackDelegate}.`, delegateError); // 첫 시도 실패 이유를 콘솔에 남깁니다.
    console.info("[MediaPipe] init:createFromOptions fallback", {
      delegate: fallbackDelegate,
      modelAssetPath,
      runningMode: "VIDEO"
    });
    handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath, // 모델 파일은 그대로 사용합니다.
        delegate: fallbackDelegate // 이번에는 대체 장치로 다시 시도합니다.
      },
      runningMode: "VIDEO", // 모드는 동일합니다.
      numHands: HAND_DETECTION_TARGET,
      minHandDetectionConfidence: 0.25,
      minHandPresenceConfidence: 0.25,
      minTrackingConfidence: 0.25
    });
    console.info("[MediaPipe] init:createFromOptions fallback success", {
      constructor: handLandmarker?.constructor?.name,
      hasDetectForVideo: typeof handLandmarker?.detectForVideo === "function"
    });
    console.info("[MediaPipe] init:createFromOptions fallback success");
  }

  const { video: activeVideo } = syncRuntimeDomRefs();
  if (activeVideo?.srcObject && activeVideo.readyState >= 2) {
    statusText.textContent = "준비 완료! 시작 버튼에 손을 올려주세요.";
  }
  console.info("[MediaPipe] init:ready (VIDEO mode)");
}

async function refreshCameraTarget() {
  const { video: activeVideo } = syncRuntimeDomRefs();
  setCanvasSize();
  if (!activeVideo || adminEditMode) return;

  if (cameraStream) {
    if (activeVideo.srcObject !== cameraStream) {
      activeVideo.srcObject = cameraStream;
    }
    activeVideo.playsInline = true;
    activeVideo.muted = true;
    activeVideo.setAttribute("playsinline", "");
    activeVideo.play().catch(() => {});
    return;
  }

  await initCamera();
}

const controlRuntime = createControlRuntime({
  audioApi: Audio,
  statusText,
  interactionMode: INTERACTION_MODE,
  ambientAudioEnabled: ENABLE_AMBIENT_AUDIO,
  perfLogKey: PERF_LOG_KEY,
  perfLogLimit: PERF_LOG_LIMIT,
  getSessionStarted: () => sessionStarted,
  getAdminEditMode: () => adminEditMode,
  onActivateStart: activateStart
});

const interactionRuntime = createInteractionRuntime({
  landingOverlay,
  landingStartButton,
  statusText,
  handCursor,
  gestureSquirrelEffect,
  audioApi: Audio,
  resolveGesture,
  resetGestureState: (handKey) => resetGestureState(handKey),
  getModelPrediction: (...args) => getModelPrediction?.(...args),
  restartClassAnimation,
  activateStart,
  registerHit,
  spawnBurst,
  setGestureObjectVariant,
  getGestureInstrumentId: (label) => gestureMapping[label] || null,
  getGesturePlayback: (label, instrumentId) => getGestureSoundProfile(label, instrumentId),
  getInstrumentPlayback: (instrumentId) => getMappedSoundProfile(instrumentId),
  playGestureSound: (label, instrumentId, note) => {
    return playGestureMappedSound(label, instrumentId, { note, spawnEffect: false });
  },
  playInstrumentSound: (instrumentId, note) => {
    const element = instrumentElements[instrumentId] || null;
    return playMappedInstrumentSound(instrumentId, element, { note, spawnEffect: false });
  },
  instruments,
  interactionMode: INTERACTION_MODE,
  collisionPadding: COLLISION_PADDING,
  startHoverMs: START_HOVER_MS,
  gestureCooldownMs: GESTURE_TRIGGER_COOLDOWN_MS,
  isAdminEditMode: () => adminEditMode,
  isSessionStarted: () => sessionStarted,
  feverController,
  checkBubbleCollision: (points) => particleSystem.checkBubbleCollision(points)
});

const trackingRuntime = createHandTrackingRuntime({
  video,
  handCanvas,
  handCtx,
  getVideo: () => syncRuntimeDomRefs().video,
  getHandCanvas: () => syncRuntimeDomRefs().handCanvas,
  getHandCtx: () => syncRuntimeDomRefs().handCtx,
  handCursor,
  renderer: Renderer,
  particleSystem,
  feverController,
  interactionRuntime,
  onBeforeFrame: (now, started) => {
    feverController.updateFeverState(now, started);
  },
  onDetectionError: (error) => {
    console.warn("HandTracking loop error:", error);
  },
  getHandLandmarker: () => handLandmarker,
  getSessionStarted: () => sessionStarted,
  inferIntervalMs: INFER_INTERVAL_MS,
  landmarkStaleMs: LANDMARK_STALE_MS
  ,
  splitHandInference: ENABLE_SPLIT_HAND_INFERENCE
});

async function init() {
  // 모델 추론 모듈 동적 로딩 (top-level await 회피)
  const modelInferenceModule = USE_SEQUENCE_MODEL
    ? await import("./model_inference_sequence.js")
    : await import("./model_inference.js");

  getModelPrediction = modelInferenceModule.getModelPrediction;
  getModelInferenceStatus = modelInferenceModule.getModelInferenceStatus;
  setModelPredictionProvider(getModelPrediction);

  // 사용 중인 모델 타입 로깅
  console.info(`[JamJamBeat] 🎵 모델 타입: ${USE_SEQUENCE_MODEL ? "시퀀스 모델 (8프레임 버퍼)" : "기존 단일 프레임 모델"}`);
  if (USE_SEQUENCE_MODEL) {
    console.info("[JamJamBeat] ⏱️ warmup 대기: 첫 8프레임 수집 (~0.5~1.2초)");
  }

  setupSeamlessBackgroundLoop({ crossfadeSec: BG_VIDEO_CROSSFADE_SEC }); // 배경 영상 반복 시스템을 먼저 준비합니다.
  setCanvasSize(); // 현재 화면 크기에 맞게 캔버스를 조정합니다.
  scene.dataset.fever = scene.dataset.fever || "off"; // 피버 상태의 기본값을 off로 맞춥니다.
  window.addEventListener("resize", setCanvasSize); // 창 크기가 바뀌면 캔버스도 다시 맞춥니다.
  window.addEventListener("beforeunload", () => {
    clearCameraSource(); // 페이지를 떠날 때 카메라 점유를 반드시 해제합니다.
  });
  window.addEventListener("jamjam:refresh-camera-target", () => {
    refreshCameraTarget().catch((error) => {
      console.warn("[MediaPipe] refreshCameraTarget failed:", error);
    });
  });

  controlRuntime.bind(); // 버튼과 입력 이벤트를 연결합니다.
  controlRuntime.syncSoundButtonUI(); // 소리 버튼 글자를 현재 상태에 맞춥니다.
  window.addEventListener("jamjam:sound-played", (event) => {
    const detail = event.detail || {};
    const handKey = String(detail.handKey || "").toLowerCase();
    if (!handKey) return;
    lastSoundEventByHand.set(handKey, {
      latencyMs: Number.isFinite(detail.latencyMs) ? detail.latencyMs : null,
      inferenceLatencyMs: Number.isFinite(detail.inferenceLatencyMs) ? detail.inferenceLatencyMs : null,
      at: Number.isFinite(detail.at) ? detail.at : Date.now()
    });
  });
  if (testModeToggleButton) {
    testModeToggleButton.addEventListener("click", () => {
      testModeEnabled = !testModeEnabled;
      syncTestModeUI();
    });
  }
  syncTestModeUI();
  startTestModeLoop();
  VIDEO_INSTRUMENT_IDS.forEach((id) => {
    const el = instrumentElements[id];
    if (!el) return;
    const videoEl = el.querySelector(".instrument-video-source");
    const canvasEl = el.querySelector(".instrument-art-canvas");
    if (!videoEl || !canvasEl) return;

    videoInstruments[id] = {
      video: videoEl,
      canvas: canvasEl,
      ctx: canvasEl.getContext("2d", { willReadFrequently: true }),
      workCanvas: document.createElement("canvas"),
      workCtx: null,
      lastFrameAt: 0,
      raf: 0
    };
    videoInstruments[id].workCtx = videoInstruments[id].workCanvas.getContext("2d", { willReadFrequently: true });
    
    syncVideoPlayback(id);
  });

  const params = new URLSearchParams(window.location.search); // URL에 적힌 옵션을 읽습니다.
  const mode = params.get("mode") || "calm"; // 모드가 없으면 calm을 기본으로 씁니다.
  applySceneMode(scene, mode); // 장면 분위기를 적용합니다.

  const createInstrumentAnimationManager = await loadAnimationManagerFactory();
  animationManager = createInstrumentAnimationManager();

  // 악기별 Lottie 애니메이션 초기화 (같은 DOM 요소에 중복 초기화 방지)
  const animatedElements = new Set();
  instruments.forEach((instrument) => {
    if (instrument.el && !animatedElements.has(instrument.el)) {
      animationManager.initAnimation(instrument.id, instrument.el);
      animatedElements.add(instrument.el);
    }
  });

  const autoStart = params.get("session") === "start" || params.get("start") === "1"; // 자동 시작 옵션을 계산합니다.
  if (autoStart && !adminEditMode) {
    activateStart(); // 자동 시작 조건이면 바로 시작합니다.
  }

  if (adminEditMode) {
    return; // 관리자 모드에서는 카메라 초기화를 하지 않고 끝냅니다.
  }

  try {
    await Promise.all([
      initCamera(),
      initMediaPipe()
    ]); // 카메라와 손 인식 모델을 동시에 준비해서 체감 대기 시간을 줄입니다.
  } catch (error) {
    console.error("Initialization failed:", error); // 실패 이유는 콘솔에 남깁니다.
    statusText.textContent = "초기화 실패: 새로고침 후 다시 시도해 주세요."; // 사용자에게는 쉬운 문구로 알려줍니다.
    statusText.style.color = "var(--danger)"; // 실패 문구를 경고 색상으로 표시합니다.
  }
}

init(); // 파일이 로드되면 전체 앱 준비를 시작합니다.

```

## Hand Tracking Runtime

Path: `/home/user/projects/JamJamBeat-model3/frontend/src/js/hand_tracking_runtime.js`

```js
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
    if (!Number.isFinite(raw)) return 96;
    return Math.max(96, Math.min(640, Math.round(raw)));
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
    const scale = sourceWidth > maxWidth ? (maxWidth / sourceWidth) : 1;
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

```

## Sequence Inference Runtime

Path: `/home/user/projects/JamJamBeat-model3/frontend/src/js/model_inference_sequence.js`

```js
// [model_inference_sequence.js] 시퀀스 기반 손동작 인식 모델 추론 엔진
// pos_scale_mlp_sequence_delta 모델 전용 - 8프레임 시퀀스를 버퍼링하고 델타 특성을 계산합니다.

import * as ort from "onnxruntime-web";

const DEFAULT_TAU = 0.85;
const DEFAULT_REQUEST_INTERVAL_MS = 150;
const SEQUENCE_LENGTH = 8; // 모델이 요구하는 시퀀스 길이
const FEATURE_DIM = 126; // 관절63 + 델타63
const EPS = 1e-8; // pos_scale 정규화 시 0으로 나누기 방지

const LEFT_HAND_MIRROR_ENABLED = (() => {
  const raw = new URLSearchParams(window.location.search).get("leftHandMirror");
  if (raw === "0" || raw === "false" || raw === "off") return false;
  return true;
})();

const PERF_ENABLED = (() => {
  const raw = new URLSearchParams(window.location.search).get("profilePerf");
  if (raw === "1" || raw === "true") return true;
  if (raw === "0" || raw === "false") return false;
  return Boolean(import.meta.env.DEV);
})();

const PERF_LOG_INTERVAL_MS = 2000;
const ORT_CDN_WASM_BASE = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.24.3/dist/";
const INIT_RETRY_INTERVAL_MS = 1500;

function normalizeRootPath(value, fallback = "/") {
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  if (!trimmed) return fallback;
  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed.replace(/\/+$/, "") + "/";
  }
  const normalized = "/" + trimmed.replace(/^\/+/, "").replace(/\/+$/, "") + "/";
  return normalized.replace(/\/+/g, "/");
}

function resolveRuntimeRoot() {
  const params = new URLSearchParams(window.location.search);
  const queryRoot = params.get("runtimeRoot");
  const globalRoot = window.__JAMJAM_RUNTIME_ROOT;
  const envRoot = import.meta.env.VITE_RUNTIME_ROOT;
  const baseUrl = import.meta.env.BASE_URL || "/";

  if (queryRoot && queryRoot.trim()) return normalizeRootPath(queryRoot, "/runtime_sequence/");
  if (typeof globalRoot === "string" && globalRoot.trim()) return normalizeRootPath(globalRoot, "/runtime_sequence/");
  if (typeof envRoot === "string" && envRoot.trim()) return normalizeRootPath(envRoot, "/runtime_sequence/");

  return normalizeRootPath(baseUrl + "runtime_sequence", "/runtime_sequence/");
}

function resolveOrtWasmRoot() {
  const params = new URLSearchParams(window.location.search);
  const queryRoot = params.get("ortWasmRoot");
  const globalRoot = window.__JAMJAM_ORT_WASM_ROOT;
  const envRoot = import.meta.env.VITE_ORT_WASM_ROOT;
  const baseUrl = import.meta.env.BASE_URL || "/";

  if (queryRoot && queryRoot.trim()) return normalizeRootPath(queryRoot, "/");
  if (typeof globalRoot === "string" && globalRoot.trim()) return normalizeRootPath(globalRoot, "/");
  if (typeof envRoot === "string" && envRoot.trim()) return normalizeRootPath(envRoot, "/");

  return normalizeRootPath(baseUrl, "/");
}

const RUNTIME_ROOT = resolveRuntimeRoot();
const ORT_LOCAL_WASM_BASE = resolveOrtWasmRoot();
const MODEL_PATH = RUNTIME_ROOT + "model.onnx";
const CLASS_NAMES_PATH = RUNTIME_ROOT + "class_names.json";
const MODEL_EXTERNAL_DATA_PATH = RUNTIME_ROOT + "model.onnx.data";

let ortApi = null;
let onnxSession = null;
let classNames = ["neutral", "fist", "open_palm", "V", "pinky", "animal", "k-heart"];
let isInitializing = false;
let initializationError = null;
let lastInitFailedAt = 0;

// 손별 상태 관리 (버퍼, 요청 상태 등)
const handStateByKey = new Map();
let globalRequestInFlight = false;
let lastGlobalRequestAt = 0;

const perfWindow = {
  startedAt: performance.now(),
  lastLogAt: performance.now(),
  requestCount: 0,
  successCount: 0,
  failureCount: 0,
  totalMs: 0,
  maxMs: 0,
  warmupCount: 0
};

function normalizeHandKey(handKey = "default") {
  return String(handKey || "default").trim().toLowerCase();
}

function isInferenceEnabledForHand(handKey = "default") {
  return normalizeHandKey(handKey) !== "left";
}

function createDisabledPrediction() {
  return {
    label: "None",
    confidence: 0,
    classId: 0,
    source: "disabled",
    disabled: true,
    status: "disabled"
  };
}

function flushPerf(now = performance.now()) {
  if (!PERF_ENABLED || now - perfWindow.lastLogAt < PERF_LOG_INTERVAL_MS) return;
  const requestCount = Math.max(1, perfWindow.requestCount);
  console.info("[Perf][ModelInferenceSequence]", {
    windowMs: Math.round(now - perfWindow.startedAt),
    requests: perfWindow.requestCount,
    successes: perfWindow.successCount,
    failures: perfWindow.failureCount,
    warmups: perfWindow.warmupCount,
    avgMs: Number((perfWindow.totalMs / requestCount).toFixed(2)),
    maxMs: Number(perfWindow.maxMs.toFixed(2)),
    mode: "onnx-sequence"
  });
  perfWindow.startedAt = now;
  perfWindow.lastLogAt = now;
  perfWindow.requestCount = 0;
  perfWindow.successCount = 0;
  perfWindow.failureCount = 0;
  perfWindow.totalMs = 0;
  perfWindow.maxMs = 0;
  perfWindow.warmupCount = 0;
}

function getRequestIntervalMs() {
  const raw = Number(new URLSearchParams(window.location.search).get("modelIntervalMs"));
  if (!Number.isFinite(raw)) return DEFAULT_REQUEST_INTERVAL_MS;
  return Math.max(60, Math.min(400, Math.round(raw)));
}

function getGlobalRequestGapMs() {
  return Math.max(45, Math.round(getRequestIntervalMs() * 0.5));
}

function getHandState(handKey = "default") {
  if (!handStateByKey.has(handKey)) {
    handStateByKey.set(handKey, {
      jointBuffer: [], // 정규화된 63차원 프레임 버퍼
      lastRequestAt: 0,
      inFlight: false,
      lastPrediction: null,
      lastStartedAt: 0,
      lastCompletedAt: 0,
      lastDurationMs: null
    });
  }
  return handStateByKey.get(handKey);
}

function isLikelyWasmBootError(error) {
  const message = String(error?.message || error || "").toLowerCase();
  return (
    message.includes("wasm") ||
    message.includes("backend") ||
    message.includes("fetch") ||
    message.includes("instantiate") ||
    message.includes("no available backend")
  );
}

async function createOnnxSession(ort, wasmBase) {
  if (ort?.env?.wasm) {
    ort.env.wasm.wasmPaths = wasmBase;
  }

  return ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    externalData: [
      {
        path: "model.onnx.data",
        data: MODEL_EXTERNAL_DATA_PATH
      }
    ]
  });
}

function readCandidateShape(candidate) {
  const isObjectLike = candidate && (typeof candidate === "object" || typeof candidate === "function");
  if (!isObjectLike) {
    return {
      session: null,
      tensor: null,
      env: null,
      keys: []
    };
  }

  const session = candidate.InferenceSession || candidate?.default?.InferenceSession || null;
  const tensor = candidate.Tensor || candidate?.default?.Tensor || null;
  const env = candidate.env || candidate?.default?.env || null;
  const keys = Object.keys(candidate || {}).slice(0, 20);

  return { session, tensor, env, keys };
}

function buildOrtApi(candidate) {
  const shape = readCandidateShape(candidate);
  if (!shape.session || typeof shape.session.create !== "function") return null;

  const fallbackTensor =
    readCandidateShape(ort).tensor ||
    readCandidateShape(globalThis?.ort).tensor ||
    null;

  const tensorCtor = shape.tensor || fallbackTensor;
  if (!tensorCtor) return null;

  return {
    InferenceSession: shape.session,
    Tensor: tensorCtor,
    env: shape.env || readCandidateShape(ort).env || null
  };
}

async function ensureOrtApi() {
  if (ortApi) return ortApi;

  const debugRows = [];
  const inspect = (label, candidate) => {
    const shape = readCandidateShape(candidate);
    debugRows.push({
      label,
      hasSessionCreate: Boolean(shape.session && typeof shape.session.create === "function"),
      hasTensor: Boolean(shape.tensor),
      hasEnv: Boolean(shape.env),
      keys: shape.keys
    });
    return buildOrtApi(candidate);
  };

  ortApi = inspect("ort", ort);

  if (!ortApi) {
    ortApi = inspect("ort.default", ort?.default);
  }

  if (!ortApi) {
    ortApi = inspect("globalThis.ort", globalThis?.ort) || inspect("globalThis.ort.default", globalThis?.ort?.default);
  }

  if (!ortApi) {
    const mod = await import("onnxruntime-web");
    ortApi = inspect("dynamicImport", mod) || inspect("dynamicImport.default", mod?.default);
  }

  if (!ortApi) {
    const wasmMod = await import("onnxruntime-web/wasm");
    ortApi = inspect("wasmImport", wasmMod) || inspect("wasmImport.default", wasmMod?.default);
  }

  if (!ortApi) {
    const debugText = debugRows
      .map((row) => `${row.label}: create=${row.hasSessionCreate}, tensor=${row.hasTensor}, env=${row.hasEnv}, keys=[${row.keys.join(",")}]`)
      .join(" | ");
    throw new Error(`onnxruntime-web 초기화 실패: InferenceSession/Tensor API를 찾지 못했습니다. ${debugText}`);
  }

  if (ortApi.env?.wasm) {
    if (!ortApi.env.wasm.wasmPaths) {
      ortApi.env.wasm.wasmPaths = ORT_LOCAL_WASM_BASE;
    }
    if (typeof SharedArrayBuffer === "undefined") {
      ortApi.env.wasm.numThreads = 1;
    }
  }

  return ortApi;
}

async function initializeModel() {
  if (onnxSession) return true;
  if (isInitializing) {
    while (isInitializing) {
      await new Promise((resolve) => setTimeout(resolve, 50));
    }
    return onnxSession !== null;
  }
  if (initializationError) {
    if (performance.now() - lastInitFailedAt < INIT_RETRY_INTERVAL_MS) return false;
    initializationError = null;
  }

  isInitializing = true;
  try {
    const classResponse = await fetch(CLASS_NAMES_PATH);
    if (!classResponse.ok) throw new Error(`Failed to load class names: ${classResponse.status}`);
    classNames = await classResponse.json();

    const ortInstance = await ensureOrtApi();

    try {
      onnxSession = await createOnnxSession(ortInstance, ORT_LOCAL_WASM_BASE);
    } catch (localError) {
      const shouldRetryWithCdn = ORT_LOCAL_WASM_BASE !== ORT_CDN_WASM_BASE && isLikelyWasmBootError(localError);
      if (!shouldRetryWithCdn) throw localError;

      console.warn("[ModelInferenceSequence] local WASM 경로 실패, CDN으로 재시도합니다.", {
        localWasmBase: ORT_LOCAL_WASM_BASE,
        cdnWasmBase: ORT_CDN_WASM_BASE,
        error: String(localError?.message || localError || "")
      });

      onnxSession = await createOnnxSession(ortInstance, ORT_CDN_WASM_BASE);
    }

    console.info("[ModelInferenceSequence] ✅ 시퀀스 모델 로드 완료", {
      modelPath: MODEL_PATH,
      runtimeRoot: RUNTIME_ROOT,
      wasmBase: ortInstance?.env?.wasm?.wasmPaths,
      classes: classNames.length,
      sequenceLength: SEQUENCE_LENGTH,
      inputNames: onnxSession.inputNames,
      outputNames: onnxSession.outputNames
    });

    initializationError = null;
    lastInitFailedAt = 0;
    isInitializing = false;
    return true;
  } catch (error) {
    console.error("[ModelInferenceSequence] ❌ 모델 로드 실패:", error);
    initializationError = error;
    lastInitFailedAt = performance.now();
    isInitializing = false;
    return false;
  }
}

function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - maxLogit));
  const sumExps = exps.reduce((a, b) => a + b, 0);
  return exps.map((x) => x / sumExps);
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function getMirrorPivotX(landmarks) {
  const wristX = Number.isFinite(landmarks?.[0]?.x) ? landmarks[0].x : 0.5;
  const indexMcpX = Number.isFinite(landmarks?.[5]?.x) ? landmarks[5].x : wristX;
  const pinkyMcpX = Number.isFinite(landmarks?.[17]?.x) ? landmarks[17].x : wristX;
  return (wristX + indexMcpX + pinkyMcpX) / 3;
}

// MediaPipe 랜드마크를 63차원 배열로 변환 (미러링 포함)
function sanitizeLandmarks(landmarks, handKey = "default") {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return null;

  const normalizedHandKey = String(handKey || "default").trim().toLowerCase();
  const shouldMirrorLeft = LEFT_HAND_MIRROR_ENABLED && normalizedHandKey === "left";
  const mirrorPivotX = shouldMirrorLeft ? getMirrorPivotX(landmarks) : 0;
  const features = [];
  for (let i = 0; i < 21; i++) {
    const point = landmarks[i];
    const rawX = Number.isFinite(point?.x) ? point.x : 0;
    const mirroredX = shouldMirrorLeft ? clamp(mirrorPivotX * 2 - rawX, 0, 1) : rawX;
    features.push(
      mirroredX,
      Number.isFinite(point?.y) ? point.y : 0,
      Number.isFinite(point?.z) ? point.z : 0
    );
  }
  return features;
}

// pos_scale 정규화: (pts - pts[0]) / ||pts[9] - pts[0]||
function normalizePosScale(frame63) {
  const normalized = new Float32Array(63);
  const originX = frame63[0];
  const originY = frame63[1];
  const originZ = frame63[2];

  // pts[9]는 인덱스 27, 28, 29 (9번째 관절 = 중지 MCP)
  const dx = frame63[27] - originX;
  const dy = frame63[28] - originY;
  const dz = frame63[29] - originZ;
  const denom = Math.hypot(dx, dy, dz);
  const scale = denom <= EPS ? 1 : 1 / denom;

  for (let i = 0; i < 63; i += 3) {
    normalized[i] = (frame63[i] - originX) * scale;
    normalized[i + 1] = (frame63[i + 1] - originY) * scale;
    normalized[i + 2] = (frame63[i + 2] - originZ) * scale;
  }

  return normalized;
}

// 8프레임 버퍼로부터 [8, 126] 특성 텐서 생성
function buildFeatureTensor(buffer) {
  if (buffer.length !== SEQUENCE_LENGTH) {
    throw new Error(`Expected ${SEQUENCE_LENGTH} buffered frames, got ${buffer.length}`);
  }

  const featureTensor = new Float32Array(SEQUENCE_LENGTH * FEATURE_DIM);
  for (let t = 0; t < SEQUENCE_LENGTH; t++) {
    const baseOffset = t * FEATURE_DIM;
    const current = buffer[t];
    const previous = t > 0 ? buffer[t - 1] : null;

    for (let i = 0; i < 63; i++) {
      // 관절 위치 (0~62)
      featureTensor[baseOffset + i] = current[i];
      // 델타 (63~125)
      featureTensor[baseOffset + 63 + i] = previous === null ? 0 : current[i] - previous[i];
    }
  }

  return featureTensor;
}

function normalizePrediction(predIndex, confidence, probs, tsMs, status, tau = DEFAULT_TAU) {
  let finalIndex = predIndex;
  let tauNeutralized = false;

  // tau 후처리: 신뢰도가 임계값보다 낮으면 neutral(0)로 강제
  if (confidence < tau && predIndex !== 0) {
    finalIndex = 0;
    tauNeutralized = true;
  }

  return {
    label: classNames[finalIndex] || "None",
    confidence,
    probs,
    classId: finalIndex,
    modelVersion: "onnx-sequence-delta",
    source: "onnx-sequence",
    ts_ms: tsMs,
    tau_applied: tau,
    tau_neutralized: tauNeutralized,
    raw_pred_index: predIndex,
    raw_pred_label: classNames[predIndex] || "None",
    status // warmup, ready, no_hand 등
  };
}

// 손이 사라졌을 때 버퍼 리셋
export function pushNoHand(handKey = "default") {
  const handState = getHandState(handKey);
  if (!isInferenceEnabledForHand(handKey)) {
    handState.jointBuffer = [];
    handState.inFlight = false;
    handState.lastRequestAt = 0;
    handState.lastStartedAt = 0;
    handState.lastCompletedAt = 0;
    handState.lastDurationMs = null;
    handState.lastPrediction = createDisabledPrediction();
    return handState.lastPrediction;
  }
  handState.jointBuffer = [];
  handState.lastPrediction = normalizePrediction(
    0, // neutral
    0,
    new Array(classNames.length).fill(0).map((_, i) => (i === 0 ? 1 : 0)),
    Math.round(performance.now()),
    "no_hand"
  );
  return handState.lastPrediction;
}

async function scheduleModelRequest(landmarks, now, handKey = "default") {
  const handState = getHandState(handKey);
  if (!isInferenceEnabledForHand(handKey)) {
    handState.jointBuffer = [];
    handState.inFlight = false;
    handState.lastRequestAt = 0;
    handState.lastStartedAt = 0;
    handState.lastCompletedAt = 0;
    handState.lastDurationMs = null;
    handState.lastPrediction = createDisabledPrediction();
    return;
  }
  const requestIntervalMs = getRequestIntervalMs();

  if (!onnxSession) {
    const initialized = await initializeModel();
    if (!initialized) return;
  }

  if (handState.inFlight) return;
  if (globalRequestInFlight) return;
  if (now - handState.lastRequestAt < requestIntervalMs) return;
  if (now - lastGlobalRequestAt < getGlobalRequestGapMs()) return;

  const rawFeatures = sanitizeLandmarks(landmarks, handKey);
  if (!rawFeatures) return;

  const tsMs = Math.round(now);
  globalRequestInFlight = true;
  lastGlobalRequestAt = now;
  handState.inFlight = true;
  handState.lastRequestAt = now;
  handState.lastStartedAt = performance.now();
  const requestStartedAt = PERF_ENABLED ? performance.now() : 0;
  if (PERF_ENABLED) perfWindow.requestCount += 1;

  try {
    // 1. pos_scale 정규화
    const normalized63 = normalizePosScale(rawFeatures);

    // 2. 버퍼에 추가
    handState.jointBuffer.push(normalized63);
    if (handState.jointBuffer.length > SEQUENCE_LENGTH) {
      handState.jointBuffer.shift();
    }

    // 3. warmup 체크
    if (handState.jointBuffer.length < SEQUENCE_LENGTH) {
      const prediction = normalizePrediction(
        0,
        0,
        new Array(classNames.length).fill(0).map((_, i) => (i === 0 ? 1 : 0)),
        tsMs,
        "warmup"
      );
      prediction.framesCollected = handState.jointBuffer.length;
      handState.lastPrediction = prediction;
      handState.lastCompletedAt = performance.now();
      handState.lastDurationMs = handState.lastCompletedAt - handState.lastStartedAt;

      if (PERF_ENABLED) {
        perfWindow.warmupCount += 1;
        const elapsedMs = performance.now() - requestStartedAt;
        perfWindow.totalMs += elapsedMs;
      }
      return;
    }

    // 4. 특성 텐서 생성 [8, 126]
    const features = buildFeatureTensor(handState.jointBuffer);

    // 5. ONNX 추론
    const ortInstance = await ensureOrtApi();
    const inputTensor = new ortInstance.Tensor(
      "float32",
      features,
      [1, SEQUENCE_LENGTH, FEATURE_DIM]
    );
    const inputName = onnxSession.inputNames[0];
    const outputName = onnxSession.outputNames[0];
    const results = await onnxSession.run({ [inputName]: inputTensor });

    // 6. 출력 처리
    const logits = Array.from(results[outputName].data);
    const probs = softmax(logits);
    const predIndex = probs.indexOf(Math.max(...probs));
    const confidence = probs[predIndex];
    const elapsedMs = performance.now() - handState.lastStartedAt;

    const prediction = normalizePrediction(predIndex, confidence, probs, tsMs, "ready");
    prediction.elapsed_ms = elapsedMs;
    prediction.completed_at_ms = performance.now();
    prediction.framesCollected = handState.jointBuffer.length;
    handState.lastPrediction = prediction;
    handState.lastCompletedAt = prediction.completed_at_ms;
    handState.lastDurationMs = elapsedMs;

    if (PERF_ENABLED) {
      perfWindow.successCount += 1;
      perfWindow.totalMs += elapsedMs;
      perfWindow.maxMs = Math.max(perfWindow.maxMs, elapsedMs);
    }
  } catch (error) {
    console.error("[ModelInferenceSequence] 추론 실패:", error);
    // 에러 발생 시 버퍼 리셋
    handState.jointBuffer = [];
    handState.lastCompletedAt = performance.now();
    handState.lastDurationMs = handState.lastStartedAt > 0 ? (handState.lastCompletedAt - handState.lastStartedAt) : null;

    if (PERF_ENABLED) {
      const elapsedMs = performance.now() - requestStartedAt;
      perfWindow.failureCount += 1;
      perfWindow.totalMs += elapsedMs;
      perfWindow.maxMs = Math.max(perfWindow.maxMs, elapsedMs);
    }
  } finally {
    globalRequestInFlight = false;
    handState.inFlight = false;
    flushPerf();
  }
}

// 메인 API: 손동작 예측 가져오기
export function getModelPrediction(landmarks, now = performance.now(), handKey = "default") {
  const handState = getHandState(handKey);
  if (!isInferenceEnabledForHand(handKey)) {
    handState.jointBuffer = [];
    handState.inFlight = false;
    handState.lastRequestAt = 0;
    handState.lastStartedAt = 0;
    handState.lastCompletedAt = 0;
    handState.lastDurationMs = null;
    handState.lastPrediction = createDisabledPrediction();
    return handState.lastPrediction;
  }

  // 손이 없으면 no_hand 처리
  if (!landmarks || !Array.isArray(landmarks) || landmarks.length < 21) {
    return pushNoHand(handKey);
  }

  scheduleModelRequest(landmarks, now, handKey);
  return handState.lastPrediction;
}

export function getModelInferenceStatus(now = performance.now()) {
  const states = [...handStateByKey.values()];
  const lastCompletedAt = states.reduce((max, state) => Math.max(max, state.lastCompletedAt || 0), 0);
  const lastDurationMs = states.reduce((latest, state) => {
    if (!Number.isFinite(state.lastCompletedAt) || state.lastCompletedAt <= 0) return latest;
    if (!latest || state.lastCompletedAt > latest.completedAt) {
      return { completedAt: state.lastCompletedAt, durationMs: state.lastDurationMs };
    }
    return latest;
  }, null);

  return {
    endpointConfigured: onnxSession !== null,
    inFlight: states.some((state) => state.inFlight),
    recentInference: states.some((state) => state.inFlight || (state.lastCompletedAt > 0 && now - state.lastCompletedAt <= 1200)),
    lastCompletedAgoMs: lastCompletedAt > 0 ? now - lastCompletedAt : null,
    lastDurationMs: lastDurationMs?.durationMs ?? null,
    disabled: false,
    mode: "onnx-sequence"
  };
}

export function preloadModel() {
  return initializeModel();
}

```

## Gesture Resolution

Path: `/home/user/projects/JamJamBeat-model3/frontend/src/js/gestures.js`

```js
// [gestures.js] 손모양을 보고 어떤 동작(주먹, 가위, 보 등)인지 맞히는 '퀴즈 정답지' 같은 파일입니다.
// 손가락이 펴졌는지 굽혀졌는지를 계산해서 손동작의 이름을 결정합니다.
// 손의 랜드마크 데이터에서 규칙 기반/모델 기반 제스처를 통합 감지하는 모듈입니다.
import { getModelPrediction as getDefaultModelPrediction } from "./model_inference.js";

let modelPredictionProvider = getDefaultModelPrediction;

function isGestureEnabledForHand(handKey = "default") {
  return String(handKey || "default").trim().toLowerCase() !== "left";
}

function createDisabledGestureResult() {
  return {
    label: "None",
    confidence: 0,
    source: "disabled",
    disabled: true,
    isV: false,
    isPaper: false
  };
}

export function setModelPredictionProvider(provider) {
  modelPredictionProvider = typeof provider === "function" ? provider : getDefaultModelPrediction;
}

function distance(a, b) {
  const dx = a.x - b.x; // 두 점의 가로 차이입니다.
  const dy = a.y - b.y; // 두 점의 세로 차이입니다.
  return Math.hypot(dx, dy); // 피타고라스 방식으로 실제 거리를 계산합니다.
}

// 손가락 사이의 거리나 위치를 보고 "지금 주먹인가?" 하고 판단을 내리는 기능입니다.
export function detectGesture(landmarks, now, sessionStarted) {
  if (!sessionStarted) { // 아직 시작 전이면 손동작을 굳이 판정하지 않습니다.
    return { isV: false, isPaper: false, isFist: false }; // 아무 동작도 아니라고 돌려줍니다.
  }

  const indexTip = landmarks[8]; // 검지 끝 좌표입니다.
  const indexPip = landmarks[6]; // 검지 중간 마디 좌표입니다.
  const indexMcp = landmarks[5]; // 검지 시작 마디 좌표입니다.

  const middleTip = landmarks[12]; // 중지 끝 좌표입니다.
  const middlePip = landmarks[10]; // 중지 중간 마디 좌표입니다.
  const middleMcp = landmarks[9]; // 중지 시작 마디 좌표입니다.

  const ringTip = landmarks[16]; // 약지 끝 좌표입니다.
  const ringPip = landmarks[14]; // 약지 중간 마디 좌표입니다.
  const ringMcp = landmarks[13]; // 약지 시작 마디 좌표입니다.

  const pinkyTip = landmarks[20]; // 새끼손가락 끝 좌표입니다.
  const pinkyPip = landmarks[18]; // 새끼손가락 중간 마디 좌표입니다.
  const pinkyMcp = landmarks[17]; // 새끼손가락 시작 마디 좌표입니다.

  const wrist = landmarks[0]; // 손목 좌표입니다.
  const palmScale = Math.max(distance(wrist, landmarks[9]), 0.001); // 손 크기에 따라 기준값이 달라지지 않도록 손바닥 크기를 구합니다.
  const vSpread = distance(indexTip, middleTip) / palmScale; // 검지와 중지가 얼마나 벌어졌는지 비율로 계산합니다.

  const indexExtended = indexTip.y < indexPip.y && indexPip.y < indexMcp.y; // 검지가 위로 펴졌는지 확인합니다.
  const middleExtended = middleTip.y < middlePip.y && middlePip.y < middleMcp.y; // 중지가 위로 펴졌는지 확인합니다.
  const ringExtended = ringTip.y < ringPip.y && ringPip.y < ringMcp.y; // 약지가 위로 펴졌는지 확인합니다.
  const pinkyExtended = pinkyTip.y < pinkyPip.y && pinkyPip.y < pinkyMcp.y; // 새끼손가락이 위로 펴졌는지 확인합니다.

  const ringFolded = ringTip.y > ringPip.y; // 약지가 접혔는지 확인합니다.
  const pinkyFolded = pinkyTip.y > pinkyPip.y; // 새끼손가락이 접혔는지 확인합니다.
  const indexFolded = indexTip.y > indexPip.y; // 검지가 접혔는지 확인합니다.
  const middleFolded = middleTip.y > middlePip.y; // 중지가 접혔는지 확인합니다.

  const isVGesture = indexExtended && middleExtended && ringFolded && pinkyFolded && vSpread > 0.38; // 브이 모양인지 판정합니다.
  const isPaperGesture = indexExtended && middleExtended && ringExtended && pinkyExtended; // 손바닥을 편 상태인지 판정합니다.
  const isFistGesture = indexFolded && middleFolded && ringFolded && pinkyFolded; // 주먹 상태인지 판정합니다.

  return {
    isV: isVGesture, // 브이 여부를 담습니다.
    isPaper: isPaperGesture, // 손바닥 여부를 담습니다.
    isFist: isFistGesture // 주먹 여부를 담습니다.
  };
}

const DEFAULT_CONFIDENCE_ENTER = 0.58;
const DEFAULT_CONFIDENCE_HOLD = 0.46;
const DEFAULT_STABLE_FRAMES = 1;
const DEFAULT_CLEAR_FRAMES = 1;

function parseNumberParam(name, fallback, min, max) {
  const value = Number(new URLSearchParams(window.location.search).get(name));
  if (!Number.isFinite(value)) return fallback;
  return Math.min(max, Math.max(min, value));
}

const CONFIDENCE_ENTER = parseNumberParam("gestureEnter", DEFAULT_CONFIDENCE_ENTER, 0.2, 0.95);
const CONFIDENCE_HOLD = parseNumberParam("gestureHold", DEFAULT_CONFIDENCE_HOLD, 0.1, CONFIDENCE_ENTER);
const STABLE_FRAMES = Math.round(parseNumberParam("gestureStableFrames", DEFAULT_STABLE_FRAMES, 1, 8));
const CLEAR_FRAMES = Math.round(parseNumberParam("gestureClearFrames", DEFAULT_CLEAR_FRAMES, 1, 6));

const CLASS_SPECIFIC_ENTER = {
  Pinky: parseNumberParam("gestureEnterPinky", 0.36, 0.2, 0.95),
  Animal: parseNumberParam("gestureEnterAnimal", 0.34, 0.2, 0.95),
  KHeart: parseNumberParam("gestureEnterKHeart", 0.36, 0.2, 0.95)
};

const CLASS_SPECIFIC_HOLD = {
  Pinky: parseNumberParam("gestureHoldPinky", 0.24, 0.1, CLASS_SPECIFIC_ENTER.Pinky),
  Animal: parseNumberParam("gestureHoldAnimal", 0.22, 0.1, CLASS_SPECIFIC_ENTER.Animal),
  KHeart: parseNumberParam("gestureHoldKHeart", 0.24, 0.1, CLASS_SPECIFIC_ENTER.KHeart)
};

const CLASS_SPECIFIC_STABLE_FRAMES = {
  Pinky: Math.round(parseNumberParam("gestureStableFramesPinky", 1, 1, 8)),
  Animal: Math.round(parseNumberParam("gestureStableFramesAnimal", 1, 1, 8)),
  KHeart: Math.round(parseNumberParam("gestureStableFramesKHeart", 1, 1, 8))
};

const stableStateByHand = new Map();

export function resetGestureState(handKey = null) {
  if (handKey == null) {
    stableStateByHand.clear();
    return;
  }
  stableStateByHand.delete(handKey);
}

function getStableState(handKey = "default") {
  if (!stableStateByHand.has(handKey)) {
    stableStateByHand.set(handKey, {
      candidateLabel: "None",
      candidateFrames: 0,
      noneFrames: 0,
      stableLabel: "None",
      confidence: 0,
      source: "rules"
    });
  }
  return stableStateByHand.get(handKey);
}

function resolveInitialGestureMode() {
  const params = new URLSearchParams(window.location.search); // 주소창의 옵션 값을 읽습니다.
  const queryMode = params.get("gestureMode"); // URL에 적힌 gestureMode 값을 가져옵니다.
  const globalMode = typeof window.__JAMJAM_GESTURE_MODE === "string"
    ? window.__JAMJAM_GESTURE_MODE // 전역 설정값이 있으면 그것도 후보로 씁니다.
    : null; // 없으면 비워둡니다.
  const raw = (queryMode || globalMode || "model").trim().toLowerCase(); // 없으면 기본값 model을 사용합니다 (ONNX 모델만 사용).

  if (raw === "rules" || raw === "model" || raw === "hybrid") {
    return raw; // 허용된 값이면 그대로 사용합니다.
  }
  return "model"; // 이상한 값이 들어오면 model 모드로 돌아갑니다.
}

let gestureMode = resolveInitialGestureMode();

export function getGestureMode() {
  return gestureMode; // 현재 사용 중인 손동작 판정 방식을 알려줍니다.
}

export function setGestureMode(nextMode) {
  const raw = String(nextMode || "").trim().toLowerCase();
  if (raw === "rules" || raw === "model" || raw === "hybrid") {
    gestureMode = raw;
    return gestureMode;
  }
  return gestureMode;
}

function normalizeModelLabel(rawLabel) {
  const label = String(rawLabel || "").trim().toLowerCase(); // AI가 보낸 라벨을 비교하기 쉬운 글자 형태로 바꿉니다.
  if (label === "none" || label === "class0") return "None"; // 0번 클래스(중립)는 None으로 처리합니다.
  if (label === "fist" || label === "isfist" || label === "class1") return "Fist"; // 주먹 관련 이름들을 하나로 통일합니다.
  if (
    label === "open palm" ||
    label === "open_palm" ||
    label === "openpalm" ||
    label === "paper" ||
    label === "ispaper" ||
    label === "class2"
  ) return "OpenPalm"; // 손바닥 관련 이름들을 하나로 통일합니다.
  if (label === "v" || label === "isv" || label === "class3") return "V"; // 브이 관련 이름들을 하나로 통일합니다.
  if (label === "pinky" || label === "ispinky" || label === "pinky class" || label === "pinky_class" || label === "class4" || label === "class 4" || label === "4") return "Pinky"; // 새끼손가락 제스처 이름을 맞춥니다.
  if (label === "animal" || label === "isanimal" || label === "class5" || label === "class 5" || label === "5") return "Animal"; // 애니멀 제스처 이름을 맞춥니다.
  if (label === "k-heart" || label === "kheart" || label === "is_k_heart" || label === "class6" || label === "class 6" || label === "6") return "KHeart"; // K-하트 제스처 이름을 맞춥니다.
  return "None"; // 어디에도 해당하지 않으면 인식 실패로 처리합니다.
}

function getStableFramesForLabel(label) {
  return CLASS_SPECIFIC_STABLE_FRAMES[label] ?? STABLE_FRAMES;
}

function mapRulesToResult(rules) {
  if (rules.isFist) return { label: "Fist", confidence: 1, source: "rules" }; // 규칙상 주먹이면 확신 100%로 적습니다.
  if (rules.isV) return { label: "V", confidence: 1, source: "rules" }; // 규칙상 브이면 확신 100%로 적습니다.
  if (rules.isPaper) return { label: "OpenPalm", confidence: 1, source: "rules" }; // 규칙상 손바닥이면 확신 100%로 적습니다.
  return { label: "None", confidence: 0, source: "rules" }; // 아니면 아무 제스처도 아니라고 적습니다.
}

function mapModelToResult(modelPrediction, stableState) {
  if (!modelPrediction) return null; // 아직 AI 답변이 없으면 더 할 일이 없습니다.

  const normalized = normalizeModelLabel(modelPrediction.label); // AI 라벨 이름을 우리 코드 기준 이름으로 바꿉니다.
  const classLabel = Number.isFinite(modelPrediction.classId)
    ? normalizeModelLabel(`class${modelPrediction.classId}`)
    : "None";
  const label = normalized !== "None" ? normalized : classLabel;
  const confidence = Number.isFinite(modelPrediction.confidence) ? modelPrediction.confidence : 0; // 신뢰도가 숫자가 아니면 0으로 처리합니다.
  const enterThreshold = CLASS_SPECIFIC_ENTER[label] ?? CONFIDENCE_ENTER;
  const holdBaseThreshold = CLASS_SPECIFIC_HOLD[label] ?? CONFIDENCE_HOLD;
  const holdThreshold = stableState.stableLabel === label ? holdBaseThreshold : enterThreshold;

  if (label === "None" || confidence < holdThreshold) {
    return { label: "None", confidence, source: "model" }; // 신뢰도가 낮으면 아직 확실하지 않다고 봅니다.
  }

  return { label, confidence, source: "model" }; // AI가 낸 결론을 우리 형식으로 돌려줍니다.
}

// 같은 동작이 여러 번 연달아 인식되어야 "진짜 이 동작을 하고 있구나"라고 확신하는 '안정화' 기능입니다.
function stabilize(rawResult, stableState) {
  if (rawResult.label === "None") {
    stableState.noneFrames += 1; // 아무 동작도 안 보인 프레임 수를 늘립니다.
    stableState.candidateLabel = "None"; // 후보 동작도 비웁니다.
    stableState.candidateFrames = 0; // 후보가 유지된 횟수도 초기화합니다.

    if (stableState.noneFrames >= CLEAR_FRAMES) {
      stableState.stableLabel = "None"; // 충분히 안 보였으면 최종 상태도 없음으로 바꿉니다.
      stableState.confidence = 0; // 확신도도 0으로 내립니다.
      stableState.source = rawResult.source; // 어떤 방식으로 판정했는지 기억합니다.
    }
  } else {
    stableState.noneFrames = 0; // 동작이 보이면 '없음' 카운트는 리셋합니다.

    if (stableState.candidateLabel === rawResult.label) {
      stableState.candidateFrames += 1; // 같은 후보가 계속 나오면 횟수를 늘립니다.
    } else {
      stableState.candidateLabel = rawResult.label; // 새로운 동작이 보이면 후보를 교체합니다.
      stableState.candidateFrames = 1; // 첫 번째 등장으로 기록합니다.
    }

    if (stableState.candidateFrames >= getStableFramesForLabel(rawResult.label)) {
      stableState.stableLabel = rawResult.label; // 여러 번 반복되면 최종 동작으로 채택합니다.
      stableState.confidence = rawResult.confidence; // 그때의 확신도도 저장합니다.
      stableState.source = rawResult.source; // 규칙인지 모델인지도 같이 저장합니다.
    }
  }

  return {
    label: stableState.stableLabel, // 안정화가 끝난 최종 라벨입니다.
    confidence: stableState.confidence, // 최종 신뢰도입니다.
    source: stableState.source, // 어떤 방식의 결과인지 알려줍니다.
    isV: stableState.stableLabel === "V", // 브이인지 빠르게 확인할 수 있도록 함께 넣어둡니다.
    isPaper: stableState.stableLabel === "OpenPalm" // 손바닥인지도 함께 넣어둡니다.
  };
}

// [최종 판단] 규칙(단순 계산)과 모델(AI)의 결과를 종합하여 지금 어떤 손동작인지 최종 결론을 내립니다.
export function resolveGesture(landmarks, now, sessionStarted, handKey = "default") {
  if (!isGestureEnabledForHand(handKey)) {
    return createDisabledGestureResult();
  }

  if (!sessionStarted) {
    return {
      label: "None", // 시작 전에는 아무 동작도 없다고 판단합니다.
      confidence: 0, // 신뢰도도 0입니다.
      source: "rules", // 기본 출처는 규칙 방식으로 둡니다.
      isV: false, // 브이도 아닙니다.
      isPaper: false // 손바닥도 아닙니다.
    };
  }

  const stableState = getStableState(handKey);
  const modelPrediction = gestureMode === "rules"
    ? null
    : modelPredictionProvider(landmarks, now, handKey); // 규칙 전용 모드에서는 모델 통신 자체를 하지 않습니다.
  const modelResult = mapModelToResult(modelPrediction, stableState) || {
    label: "None", // AI 답변이 없으면 일단 인식 안 됨으로 둡니다.
    confidence: 0, // 신뢰도도 0입니다.
    source: "model" // 출처는 모델이라고 표시합니다.
  };

  let rawResult; // 안정화 전에 사용할 임시 결과를 담을 변수입니다.
  if (gestureMode === "model") {
    rawResult = modelResult; // 모델만 쓰는 모드면 AI 답변을 그대로 씁니다.
  } else {
    const rules = detectGesture(landmarks, now, sessionStarted); // 손가락 위치만 보고 규칙 판정을 실행합니다.
    const ruleResult = mapRulesToResult(rules); // 그 결과를 공통 형식으로 바꿉니다.

    if (gestureMode === "rules") {
      rawResult = ruleResult; // 규칙만 쓰는 모드면 이 값을 그대로 씁니다.
    } else {
      rawResult = ruleResult.label === "Fist"
        ? ruleResult // 주먹은 규칙 기반이 더 단순하고 안정적이어서 우선합니다.
        : (modelResult.label !== "None" ? modelResult : ruleResult); // 그 외에는 AI가 확실하면 AI를, 아니면 규칙 결과를 씁니다.
    }
  }

  return stabilize(rawResult, stableState); // 마지막으로 흔들리는 결과를 안정화해서 돌려줍니다.
}

```

## Interaction Runtime Debug Snapshot Source

Path: `/home/user/projects/JamJamBeat-model3/frontend/src/js/interaction_runtime.js`

```js
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
  const gestureHoldStartByKey = new Map();
  const activeGestureHands = new Set();
  const rectCache = new WeakMap();
  let lastStatusText = "";
  const HELD_ONESHOT_INTERVAL_MS = Math.max(gestureCooldownMs, 220);

  function isGestureEnabledForHand(handKey = "default") {
    return String(handKey || "default").trim().toLowerCase() !== "left";
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
    if (now - lastHitAt < instrument.cooldownMs) return false;
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
      gestureHoldStartByKey.delete(handKey);
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
    if (!isSessionStarted()) return;
    if (isAdminEditMode()) return;
    if (interactionMode === "gesture") return;
    if (!hasRecognizedGesture()) return;
    const triggeredElements = new Set();

    for (const instrument of instruments) {
      if (triggeredElements.has(instrument.el)) continue;
      const hit = points.some((point) => isInsideElement(point, instrument.el));
      if (!hit) continue;
      if (now - instrument.lastHitAt < instrument.cooldownMs) continue;
      markInstrumentTriggered(instrument, now, "touch");
      audioApi.setPlaybackContext({
        instrumentId: instrument.id,
        gestureLabel: "Touch",
        gestureSource: "touch",
        triggerTs: now,
        handKey: "touch"
      });
      const playedTag = instrument.onHit();
      activateInstrumentElement(instrument);
      registerHit(now);
      triggeredElements.add(instrument.el);
      setStatusText(`${instrument.name} - ${playedTag || instrument.soundTag}`);
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
    const instrumentId = getGestureInstrumentId?.(label);
    if (!instrumentId) return;
    const playback = getGesturePlayback?.(label, instrumentId) || getInstrumentPlayback(instrumentId);
    if (!playback) return;
    const handState = getHandState(handKey);

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
      if (gesture.label === "Fist") activeGestureHands.add(handKey);
      else activeGestureHands.delete(handKey);
      setGestureObjectVariant?.(activeGestureHands.size > 0);

      if (!audioApi.getAudioState().running) {
        setStatusText("소리가 꺼져 있어요. '소리 켜기' 버튼을 눌러주세요.");
        handState.lastGestureLabel = gesture.label;
        return;
      }

      runGestureReaction(gesture.label, now, handKey);
      showSquirrelEffect();

      handState.lastGestureLabel = gesture.label;
      gestureHoldStartByKey.set(handKey, now); // 새로운 제스처면 홀드 시작 시각을 기록합니다.

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
          runGestureReaction(gesture.label, now, handKey);
          handState.lastGestureTriggerAt = now;
        }
      }
      const displayName = getGestureDisplayName(gesture.label);
      setStatusText(`${handKey}손 ${displayName} 유지 중... 🎵 (${(gesture.confidence * 100).toFixed(0)}%)`);
    }
  }

  // 손 커서의 위치와 보이기/숨기기를 담당합니다.
  function setPointer(point, now, landmarks = null) {
    const baseTransform = "translate(-50%, -86%)";
    handCursor.style.opacity = 1;
    handCursor.style.left = `${point.x}px`;
    handCursor.style.top = `${point.y}px`;

    // 손목과 검지 끝의 각도를 계산하여 세로 지휘봉 이미지를 회전합니다.
    if (landmarks && landmarks.length > 8) {
      const wrist = landmarks[0];
      const indexTip = landmarks[8];
      const dx = indexTip.x - wrist.x;
      const dy = indexTip.y - wrist.y;
      const angle = Math.atan2(dy, dx) * (180 / Math.PI) + 90;
      handCursor.style.transform = `${baseTransform} rotate(${angle}deg)`;
      return;
    }

    handCursor.style.transform = baseTransform;
  }

  // 모든 터치 포인트(손가락 끝)에 대해 비눗방울 충돌을 검사합니다.
  function processBubbleCollisions(points) {
    if (!hasRecognizedGesture()) return;
    if (checkBubbleCollision(points)) {
      // 터질 때 효과음 (성능을 위해 짧고 가볍게)
      Audio.playKids_Triangle(68 + Math.random() * 8);
    }
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
      gestureHoldStartByKey.delete(handKey);
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

```

## Runtime Sequence Config

Path: `/home/user/projects/JamJamBeat-model3/frontend/public/runtime_sequence/config.json`

```json
{
  "bundle_id": "pos_scale_mlp_sequence_delta_20260319_162806",
  "model_id": "mlp_sequence_delta",
  "mode": "sequence",
  "dataset_key": "pos_scale",
  "normalization_family": "pos_scale",
  "input_type": "model_feature_sequence_joint63_delta63",
  "caller_input_type": "raw_mediapipe_landmark_frame_stream",
  "input_dim": 126,
  "seq_len": 8,
  "num_classes": 7,
  "neutral_index": 0,
  "default_tau": 0.85,
  "supported_backends": ["onnx-web"],
  "streaming_supported": true,
  "no_hand_resets_buffer": true,
  "preprocess": {
    "normalization": "pos_scale",
    "delta": true,
    "delta_order": "first",
    "seq_len": 8,
    "allowed_frame_shapes": [[63], [21, 3]],
    "formula": "(pts - pts[0]) / ||pts[9] - pts[0]||",
    "eps": 1e-8
  },
  "checkpoint_fingerprint": "2f9e81b7a142a0cd92707eb903fe4880f55cbcf920676f6433104ddf8c006d09"
}

```

## Runtime Sequence Input Spec

Path: `/home/user/projects/JamJamBeat-model3/frontend/public/runtime_sequence/input_spec.json`

```json
{
  "name": "raw_mediapipe_landmark_frame_stream",
  "shape": [63],
  "dtype": "float32",
  "layout": "streaming_frame",
  "caller_input_shapes": [[63], [21, 3]],
  "model_input_shape": [8, 126],
  "streaming_seq_len": 8,
  "default_tau": 0.85,
  "feature_order_path": "runtime/feature_order.json",
  "description": "Caller provides raw MediaPipe landmark frames one by one. The frontend-only helper keeps an 8-frame rolling buffer, internally applies pos_scale normalization per frame, builds first-order temporal deltas, and produces the final [8,126] ONNX input.",
  "notes": [
    "Official caller contract is a single raw frame only: shape [63] or [21,3].",
    "The helper performs warmup until 8 valid frames are collected.",
    "pushNoHand() clears the buffer and returns no_hand immediately.",
    "Internal normalization is pos_scale with viewer-aligned zero-distance fallback: shifted frame only when ||pts[9]-pts[0]|| is too small.",
    "Internal model features are x0..z20 (63d joint) + d_x0..d_z20 (63d delta).",
    "Delta = current_frame - previous_frame and the first frame delta is all zeros.",
    "pred != neutral and confidence < 0.85 results in tau_neutralized.",
    "feature_order.json describes the internal 126d model feature order, not the caller raw input order."
  ]
}

```
