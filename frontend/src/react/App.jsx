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

  const handleStartClick = () => {
    setShowTutorial(true);
  };

  const closeTutorial = () => {
    setShowTutorial(false);
    requestStart();
  };

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
            <h2 className="tutorial-title">숲의 지휘자 가이드</h2>
            <p className="tutorial-description">손동작을 하면 음악이 나옵니다!</p>
            <div className="tutorial-image-container">
              <img 
                src="/assets/hand_gestures_guide.png" 
                alt="6가지 손동작과 악기 매칭 가이드" 
                className="tutorial-image" 
              />
            </div>
            <button 
              type="button" 
              className="tutorial-close-button" 
              onClick={closeTutorial}
            >
              알겠어요! 모험 시작하기
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
          <div className="camera-shell" aria-hidden="true">
            <video id="webcam" autoPlay playsInline muted />
          </div>
          <div className="guide-silhouette" aria-hidden="true" />
          <div id="gestureSquirrelEffect" className="gesture-squirrel" aria-hidden="true">🐿️</div>
          <p id="pulseMessage" className="pulse-message">손을 잼잼! 해서 숲을 깨워봐!</p>
        </section>

        <section className="hud" aria-live="polite">
          <h2>잼잼비트 숲의 입구</h2>
          <p id="status" className="status">카메라 준비 중...</p>
          <button id="soundUnlockButton" className="sound-unlock" type="button" onClick={requestSoundToggle}>{soundButtonLabel}</button>
          <button id="testModeToggleButton" className="test-mode-toggle" type="button">테스트 모드 켜기</button>
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

        <canvas id="handCanvas" />
        <canvas id="effectCanvas" aria-hidden="true" />
        <div id="handCursor" className="hand-cursor" aria-hidden="true">
          <img src={batonImage} alt="" aria-hidden="true" />
        </div>

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
