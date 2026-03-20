import React from "react";
import { DEFAULT_LAYOUT } from "../js/instrument_layout.js";
import { useInstrumentLayout } from "./hooks/useInstrumentLayout.js";
import { useMainControls } from "./hooks/useMainControls.js";
import { useLegacyMainRuntime } from "./hooks/useLegacyMainRuntime.js";
import logoWebp from "../../assets/로고-removebg-preview.webp";
import logoPng from "../../assets/로고-removebg-preview.png";
import hedgehogBaseVideo from "../../assets/objects/고슴도치1.mov?url";
import hedgehogGestureVideo from "../../assets/objects/고슴도치2.mov?url";
import foxImage from "../../assets/objects/여우.webp";
import rabbitImage from "../../assets/objects/토끼.webp";
import squirrelImage from "../../assets/objects/다람쥐.webp";
import catImage from "../../assets/objects/고양이.webp";
import penguinBaseVideo from "../../assets/objects/팽귄1.mov?url";
import penguinImage from "../../assets/objects/팽귄.webp";
import backgroundVideo from "../../assets/objects/움직이는_동화_영상_만들기.mp4?url";

const INSTRUMENTS = [
  {
    id: "drum",
    elementId: "instrumentDrum",
    className: "instrument instrument-drum instrument-hedgehog1",
    label: "고슴도치 드럼",
    content: (
      <>
        <video
          className="instrument-art instrument-video-source"
          src={hedgehogBaseVideo}
          data-variant-base={hedgehogBaseVideo}
          data-variant-gesture={hedgehogGestureVideo}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          aria-hidden="true"
        />
        <canvas className="instrument-art instrument-art-canvas" aria-hidden="true" />
      </>
    )
  },
  {
    id: "xylophone",
    elementId: "instrumentXylophone",
    className: "instrument instrument-xylophone",
    label: "여우 실로폰",
    content: (
      <img className="instrument-art" src={foxImage} alt="여우" aria-hidden="true" />
    )
  },
  {
    id: "tambourine",
    elementId: "instrumentTambourine",
    className: "instrument instrument-tambourine",
    label: "토끼 탬버린",
    content: (
      <img className="instrument-art" src={rabbitImage} alt="토끼" aria-hidden="true" />
    )
  },
  {
    id: "a",
    elementId: "instrumentA",
    className: "instrument instrument-squirrel",
    label: "다람쥐 심벌즈",
    content: (
      <img className="instrument-art" src={squirrelImage} alt="다람쥐" aria-hidden="true" />
    )
  },
  {
    id: "cat",
    elementId: "instrumentCat",
    className: "instrument instrument-cat",
    label: "고양이 하트",
    content: (
      <img className="instrument-art" src={catImage} alt="고양이" aria-hidden="true" />
    )
  },
  {
    id: "penguin",
    elementId: "instrumentPenguin",
    className: "instrument instrument-penguin instrument-penguin1",
    label: "펭귄 실로폰",
    content: (
      <>
        <video
          className="instrument-art instrument-video-source"
          src={penguinBaseVideo}
          data-variant-base={penguinBaseVideo}
          data-variant-gesture={penguinBaseVideo}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          aria-hidden="true"
        />
        <canvas className="instrument-art instrument-art-canvas" aria-hidden="true" />
        <img className="instrument-art instrument-fallback-art" src={penguinImage} alt="펭귄" aria-hidden="true" />
      </>
    )
  }
];

function getInstrumentStyle(id) {
  const position = DEFAULT_LAYOUT[id];
  if (!position) return undefined;

  return {
    left: `${position.x}vw`,
    bottom: `${position.y}vh`,
    right: "auto"
  };
}

function InstrumentButton({ instrument }) {
  return (
    <button
      id={instrument.elementId}
      className={instrument.className}
      type="button"
      aria-label={instrument.label}
      style={getInstrumentStyle(instrument.id)}
    >
      {instrument.content}
    </button>
  );
}

export default function App() {
  useInstrumentLayout();
  useLegacyMainRuntime();
  const { soundButtonLabel, requestStart, requestSoundToggle } = useMainControls();

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

      <main id="scene" className="scene hide-camera" data-fever="off">
        <section id="landingOverlay" className="landing-overlay" aria-live="polite">
          <div className="landing-panel">
            <div className="landing-logo-container">
              <img className="landing-logo" src={logoPng} srcSet={logoWebp} alt="JamJam Beat Logo" loading="eager" />
            </div>
            <button id="landingStartButton" type="button" onClick={requestStart}>시작하기</button>
            <a className="landing-admin-link" href="./index.html?admin=1">관리자 모드 (직접 배치)</a>
            <a className="landing-admin-link" href="./mapping.html">매칭 설정 페이지</a>
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
          <img id="gestureSquirrelEffect" className="gesture-squirrel" src={squirrelImage} alt="" aria-hidden="true" />
          <p id="pulseMessage" className="pulse-message">손을 잼잼! 해서 숲을 깨워봐!</p>
        </section>

        <section className="hud" aria-live="polite">
          <h2>잼잼비트 숲의 입구</h2>
          <p id="status" className="status">카메라 준비 중...</p>
          <button id="soundUnlockButton" className="sound-unlock" type="button" onClick={requestSoundToggle}>{soundButtonLabel}</button>
          <button id="testModeToggleButton" className="test-mode-toggle" type="button">테스트 모드 켜기</button>
          <a className="hud-main-link" href="./index.html">메인화면으로</a>
          <a className="hud-main-link" href="./mapping.html">매칭 설정</a>
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
        <div id="handCursor" className="hand-cursor" aria-hidden="true" />

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
