import React from "react";
import { DEFAULT_LAYOUT } from "../js/instrument_layout.js";
import { useInstrumentLayout } from "./hooks/useInstrumentLayout.js";
import { useMainControls } from "./hooks/useMainControls.js";
import { useLegacyMainRuntime } from "./hooks/useLegacyMainRuntime.js";

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
          src="/assets/objects/고슴도치1.mov"
          data-variant-base="/assets/objects/고슴도치1.mov"
          data-variant-gesture="/assets/objects/고슴도치2.mov"
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
      <picture>
        <source srcSet="/assets/objects/여우.webp" type="image/webp" />
        <img className="instrument-art" src="/assets/objects/여우.png" alt="여우" aria-hidden="true" />
      </picture>
    )
  },
  {
    id: "tambourine",
    elementId: "instrumentTambourine",
    className: "instrument instrument-tambourine",
    label: "토끼 탬버린",
    content: (
      <picture>
        <source srcSet="/assets/objects/토끼.webp" type="image/webp" />
        <img className="instrument-art" src="/assets/objects/토끼.png" alt="토끼" aria-hidden="true" />
      </picture>
    )
  },
  {
    id: "a",
    elementId: "instrumentA",
    className: "instrument instrument-squirrel",
    label: "다람쥐 심벌즈",
    content: (
      <picture>
        <source srcSet="/assets/objects/다람쥐.webp" type="image/webp" />
        <img className="instrument-art" src="/assets/objects/다람쥐.png" alt="다람쥐" aria-hidden="true" />
      </picture>
    )
  },
  {
    id: "cat",
    elementId: "instrumentCat",
    className: "instrument instrument-cat",
    label: "고양이 하트",
    content: (
      <picture>
        <source srcSet="/assets/objects/고양이.webp" type="image/webp" />
        <img className="instrument-art" src="/assets/objects/고양이.png" alt="고양이" aria-hidden="true" />
      </picture>
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
          src="/assets/objects/팽귄1.mov"
          data-variant-base="/assets/objects/팽귄1.mov"
          data-variant-gesture="/assets/objects/팽귄1.mov"
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
        </defs>
      </svg>

      <main id="scene" className="scene hide-camera" data-fever="off">
        <section id="landingOverlay" className="landing-overlay" aria-live="polite">
          <div className="landing-panel">
            <div className="landing-logo-container">
              <picture>
                <source srcSet="/assets/로고-removebg-preview.webp" type="image/webp" />
                <img className="landing-logo" src="/assets/로고-removebg-preview.png" alt="JamJam Beat Logo" loading="eager" />
              </picture>
            </div>
            <button id="landingStartButton" type="button" onClick={requestStart}>시작하기</button>
            <a className="landing-admin-link" href="./index.html?admin=1">관리자 모드 (직접 배치)</a>
            <a className="landing-admin-link" href="./mapping.html">매칭 설정 페이지</a>
            <a className="landing-admin-link" href="./performance.html">📊 성능확인 페이지</a>
          </div>
        </section>

        <section className="layer layer-background" aria-hidden="true">
          <div className="bg-video-wrap" aria-hidden="true">
            <video className="bg-video bg-video-a is-active" muted playsInline preload="auto" autoPlay loop>
              <source src="/assets/objects/움직이는_동화_영상_만들기.mp4" type="video/mp4" />
            </video>
            <video className="bg-video bg-video-b is-preload" muted playsInline preload="auto" autoPlay loop>
              <source src="/assets/objects/움직이는_동화_영상_만들기.mp4" type="video/mp4" />
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
          <img id="gestureSquirrelEffect" className="gesture-squirrel" src="/assets/objects/다람쥐.webp" alt="" aria-hidden="true" />
          <p id="pulseMessage" className="pulse-message">손을 잼잼! 해서 숲을 깨워봐!</p>
        </section>

        <section className="hud" aria-live="polite">
          <h2>잼잼비트 숲의 입구</h2>
          <p id="status" className="status">카메라 준비 중...</p>
          <button id="soundUnlockButton" className="sound-unlock" type="button" onClick={requestSoundToggle}>{soundButtonLabel}</button>
          <a className="hud-main-link" href="./index.html">메인화면으로</a>
          <a className="hud-main-link" href="./mapping.html">매칭 설정</a>
          <a className="hud-main-link" href="./performance.html">📊 성능확인 페이지</a>
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
      </main>
    </>
  );
}
