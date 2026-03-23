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
import handGesturesGuide from "../../assets/objects/Change_the_cats_hand_gesture_to_make_a_pinky_fing-1774251764915.png";
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

  const landingBgmRef = React.useRef(null);

  React.useEffect(() => {
    const audio = new Audio("/assets/sounds/봄을 부르는 피아노 음악 🌷 싱그러운 선율 들어요 [bExNhDN12HI].mp3");
    audio.loop = true;
    audio.volume = 0.22; // 배경음악 사운드 줄임
    landingBgmRef.current = audio;

    const tryPlay = () => {
      if (!landingBgmRef.current) return;
      audio.play().catch(() => {
        window.addEventListener('click', tryPlay, { once: true });
        window.addEventListener('keydown', tryPlay, { once: true });
      });
    };
    tryPlay();

    return () => {
      window.removeEventListener('click', tryPlay);
      window.removeEventListener('keydown', tryPlay);
      if (landingBgmRef.current) {
        landingBgmRef.current.pause();
        landingBgmRef.current.removeAttribute('src');
        landingBgmRef.current = null;
      }
    };
  }, []);

  const stopLandingBgm = () => {
    if (landingBgmRef.current) {
      landingBgmRef.current.pause();
    }
  };

  const handleStartClick = () => {
    stopLandingBgm();
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

              <div className="tutorial-vision-head">
                <p className="test-mode-eyebrow">연습 미리보기</p>
                <p id="tutorialVisionSummary" className="tutorial-vision-summary">왼쪽은 원본 손, 오른쪽은 모델 입력 좌표입니다.</p>
              </div>
              <div className="tutorial-vision-grid">
                <article className="tutorial-vision-card">
                  <h3>원본 손</h3>
                  <div className="tutorial-mirror-zone">
                    {tutorialPracticeEnabled ? (
                      <>
                        <div className="camera-shell tutorial-camera-shell" aria-hidden="true">
                          <video id="webcam" autoPlay playsInline muted />
                        </div>
                        <canvas id="handCanvas" className="tutorial-hand-canvas" />
                      </>
                    ) : (
                      <div className="tutorial-camera-placeholder" aria-hidden="true" />
                    )}
                  </div>
                </article>
                <article className="tutorial-vision-card">
                  <h3>모델 입력</h3>
                  <canvas
                    id="tutorialNormalizedCanvas"
                    className="test-mode-normalized-canvas tutorial-normalized-canvas"
                    width="220"
                    height="220"
                    aria-hidden="true"
                  />
                </article>
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

        <div id="testModeDock" className="test-mode-dock is-hidden">
          <section id="testModeVision" className="test-mode-vision" aria-live="polite">
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
                <p id="testModeNormalizedGestureDisplay" className="test-mode-vision-gesture-display is-idle">대기 중</p>
                <canvas id="testModeNormalizedCanvas" className="test-mode-normalized-canvas" width="220" height="220" aria-hidden="true" />
              </article>
            </div>
          </section>

          <section id="testModePanel" className="test-mode-panel" aria-live="polite">
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

            <div className="test-mode-columns test-mode-columns-right-only">
              <article className="test-mode-card">
                <h3>오디오 진단</h3>
                <dl className="test-mode-grid">
                  <dt>AudioCtx</dt><dd id="testModeAudioState">-</dd>
                  <dt>sound</dt><dd id="testModeSoundEnabled">-</dd>
                  <dt>unlocked</dt><dd id="testModeAudioUnlocked">-</dd>
                  <dt>voices</dt><dd id="testModeAudioVoices">-</dd>
                  <dt>pan mode</dt><dd id="testModeAudioPanMode">-</dd>
                  <dt>last key</dt><dd id="testModeLastSoundKey">-</dd>
                  <dt>play mode</dt><dd id="testModeLastPlayMode">-</dd>
                  <dt>last pan</dt><dd id="testModeLastPan">-</dd>
                  <dt>gain mode</dt><dd id="testModeLastGainMode">-</dd>
                  <dt>last sound</dt><dd id="testModeLastSound">-</dd>
                </dl>
                <div className="test-mode-actions">
                  <button id="testModePlayBeepButton" className="test-mode-action-button" type="button">테스트 비프</button>
                  <button id="testModeCenterPanButton" className="test-mode-action-button" type="button">센터 출력으로 테스트</button>
                  <button id="testModeRestorePanButton" className="test-mode-action-button is-secondary" type="button">기본 출력 복원</button>
                </div>
              </article>

              <article className="test-mode-card test-mode-card-hidden" aria-hidden="true">
                <h3>왼손</h3>
                <p id="testModeLeftGestureDisplay" className="test-mode-gesture-display is-idle">대기 중</p>
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
                <p id="testModeRightGestureDisplay" className="test-mode-gesture-display is-idle">대기 중</p>
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
        </div>
      </main>
    </>
  );
}
