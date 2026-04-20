import React from "react";
import { DEFAULT_LAYOUT } from "../js/instrument_layout.js";
import { useInstrumentLayout } from "./hooks/useInstrumentLayout.js";
import { useMainControls } from "./hooks/useMainControls.js";
import { useLegacyMainRuntime } from "./hooks/useLegacyMainRuntime.js";
import {
  getInstrumentVolumes,
  resetInstrumentVolumes as resetSavedInstrumentVolumes,
  setInstrumentVolume as saveInstrumentVolume
} from "../js/audio.js";
import {
  DEFAULT_GESTURE_MAPPING,
  DEFAULT_OBJECT_SAMPLE_MAPPING,
  loadGestureMapping,
  loadObjectSampleMapping,
  OBJECT_SAMPLE_OPTIONS,
  resetObjectSampleMapping as resetSavedObjectSampleMapping,
  resetGestureMapping as resetSavedGestureMapping,
  setObjectSampleMapping as saveObjectSampleMapping,
  setGestureMapping as saveGestureMapping
} from "../js/sound_mapping.js";
import {
  getAllModelConfigs,
  getCurrentModelId,
  switchModel
} from "../js/model_manager.js";

const logoWebp = "/assets/로고-removebg-preview.webp";
const logoPng = logoWebp;
const batonImage = "/assets/objects/지휘봉.png";
const backgroundVideo = "/assets/objects/움직이는_동화_영상_만들기.mp4";
const handGesturesGuide = "/assets/objects/Change_the_cats_hand_gesture_to_make_a_pinky_fing-1774251764915.png";
const hedgehogCreditVideo = "/assets/objects/고슴도치1.mov";
const penguinCreditVideo = "/assets/objects/팽귄1.mov";

const BGM_PLAYLIST = [
  {
    id: "spring-piano",
    title: "봄을 부르는 피아노",
    src: "/assets/sounds/봄을 부르는 피아노 음악 🌷 싱그러운 선율 들어요 [bExNhDN12HI].mp3",
    slots: ["evening", "night"]
  }
];
const AUTO_BGM_TRACK_ID = "spring-piano";

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

const GESTURE_OPTIONS = [
  { id: "Fist", label: "주먹" },
  { id: "OpenPalm", label: "손바닥" },
  { id: "V", label: "브이" },
  { id: "Pinky", label: "새끼손가락" },
  { id: "Animal", label: "애니멀" },
  { id: "KHeart", label: "케이하트" }
];

function buildInstrumentVolumeState() {
  const savedVolumes = getInstrumentVolumes();
  return Object.fromEntries(
    INSTRUMENTS.map((instrument) => [instrument.id, savedVolumes[instrument.id] ?? 1])
  );
}

function buildGestureMappingState() {
  const savedMapping = loadGestureMapping();
  return Object.fromEntries(
    GESTURE_OPTIONS.map((gesture) => [gesture.id, savedMapping[gesture.id] ?? DEFAULT_GESTURE_MAPPING[gesture.id]])
  );
}

function buildObjectSampleMappingState() {
  const savedMapping = loadObjectSampleMapping();
  return Object.fromEntries(
    INSTRUMENTS.map((instrument) => [instrument.id, savedMapping[instrument.id] ?? DEFAULT_OBJECT_SAMPLE_MAPPING[instrument.id]])
  );
}

function resolveBgmTrack(selection) {
  if (selection !== "auto") {
    return BGM_PLAYLIST.find((track) => track.id === selection) || BGM_PLAYLIST[0];
  }

  return BGM_PLAYLIST.find((track) => track.id === AUTO_BGM_TRACK_ID) || BGM_PLAYLIST[0];
}

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

function TutorialGesturePreview() {
  return (
    <div className="tutorial-vision-grid tutorial-vision-grid-single">
      <article className="tutorial-vision-card">
        <h3 className="tutorial-vision-title">나의 손동작</h3>
        <div className="tutorial-gesture-preview">
          <canvas
            id="tutorialNormalizedCanvas"
            className="test-mode-normalized-canvas tutorial-normalized-canvas"
            width="220"
            height="220"
            aria-hidden="true"
          />
          <div id="tutorialGestureResult" className="tutorial-gesture-result">
            인식 중...
          </div>
        </div>
      </article>
    </div>
  );
}

function LandingModelSelect({
  selectedModelId,
  modelLoading,
  modelConfigs,
  onChange
}) {
  return (
    <div className="landing-model-select">
      <label className="landing-model-label" htmlFor="landingModelSelect">
        🤖 AI 모델 선택
      </label>
      <select
        id="landingModelSelect"
        className="landing-model-field"
        value={selectedModelId}
        onChange={onChange}
        disabled={modelLoading}
      >
        {modelConfigs.map((config) => (
          <option key={config.id} value={config.id}>
            {config.name} - {config.description}
          </option>
        ))}
      </select>
      {modelLoading && (
        <p className="landing-model-loading">모델 로딩 중...</p>
      )}
    </div>
  );
}

function BgmPlayer({
  activeBgmTrack,
  bgmEnabled,
  bgmSelection,
  bgmVolume,
  onSelectionChange,
  onToggle,
  onVolumeChange
}) {
  return (
    <div className="bgm-horizontal-player">
      <div className="bgm-player-track">
        <span className="bgm-player-icon" aria-hidden="true">🎵</span>
        <div className="bgm-player-text">
          <span className="bgm-player-title">{activeBgmTrack.title}</span>
          <span className="bgm-player-mode">
            {bgmSelection === "auto" ? "자동 선택" : "수동 선택"}
          </span>
        </div>
      </div>

      <div className="bgm-player-divider" aria-hidden="true" />

      <select
        className="bgm-player-select"
        value={bgmSelection}
        onChange={onSelectionChange}
        aria-label="배경음악 선택"
      >
        <option value="auto">자동 선택</option>
        {BGM_PLAYLIST.map((track) => (
          <option key={track.id} value={track.id}>
            {track.title}
          </option>
        ))}
      </select>

      <button
        type="button"
        className="bgm-player-toggle"
        onClick={onToggle}
        title={bgmEnabled ? "배경음악 끄기" : "배경음악 켜기"}
        aria-pressed={bgmEnabled}
      >
        {bgmEnabled ? "⏸" : "▶"}
      </button>

      <label className={`bgm-player-volume ${bgmEnabled ? "" : "is-disabled"}`}>
        <span className="bgm-player-volume-icon" aria-hidden="true">
          {bgmEnabled ? "🔊" : "🔈"}
        </span>
        <input
          className="bgm-player-range"
          type="range"
          min="0"
          max="0.8"
          step="0.01"
          value={bgmVolume}
          onChange={onVolumeChange}
          disabled={!bgmEnabled}
          aria-label="배경음악 볼륨"
        />
      </label>
    </div>
  );
}

function SoundSettingsPanel({
  instrumentVolumes,
  gestureMapping,
  objectSampleMapping,
  onVolumeChange,
  onObjectSampleChange,
  onGestureMappingChange,
  onReset,
  onResetObjectSampleMapping,
  onResetGestureMapping
}) {
  return (
    <div className="sound-settings-panel">
      <div className="sound-settings-head">
        <p className="sound-settings-title">오브젝트 샘플</p>
        <button type="button" className="sound-settings-reset" onClick={onResetObjectSampleMapping}>
          샘플 기본값
        </button>
      </div>
      <div className="sound-settings-list sound-settings-list-mapping">
        {INSTRUMENTS.map((instrument) => (
          <label key={instrument.id} className="sound-settings-item sound-settings-item-select">
            <span className="sound-settings-label">{instrument.label}</span>
            <select
              className="sound-settings-select"
              value={objectSampleMapping[instrument.id] ?? DEFAULT_OBJECT_SAMPLE_MAPPING[instrument.id]}
              onChange={(event) => onObjectSampleChange(instrument.id, event.target.value)}
              aria-label={`${instrument.label} 샘플 선택`}
            >
              {OBJECT_SAMPLE_OPTIONS.map((sample) => (
                <option key={sample.id} value={sample.id}>
                  {sample.label}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
      <div className="sound-settings-head">
        <p className="sound-settings-title">오브젝트 볼륨</p>
        <button type="button" className="sound-settings-reset" onClick={onReset}>
          기본값 복원
        </button>
      </div>
      <div className="sound-settings-list">
        {INSTRUMENTS.map((instrument) => {
          const volume = instrumentVolumes[instrument.id] ?? 1;
          return (
            <label key={instrument.id} className="sound-settings-item">
              <span className="sound-settings-label">{instrument.label}</span>
              <div className="sound-settings-control">
                <input
                  className="sound-settings-range"
                  type="range"
                  min="0"
                  max="1.5"
                  step="0.01"
                  value={volume}
                  onChange={(event) => onVolumeChange(instrument.id, event.target.value)}
                  aria-label={`${instrument.label} 소리 크기`}
                />
                <span className="sound-settings-value">{Math.round(volume * 100)}%</span>
              </div>
            </label>
          );
        })}
      </div>
      <div className="sound-settings-head sound-settings-head-mapping">
        <p className="sound-settings-title">손동작 매핑</p>
        <button type="button" className="sound-settings-reset" onClick={onResetGestureMapping}>
          제스처 기본값
        </button>
      </div>
      <div className="sound-settings-list sound-settings-list-mapping">
        {GESTURE_OPTIONS.map((gesture) => (
          <label key={gesture.id} className="sound-settings-item sound-settings-item-select">
            <span className="sound-settings-label">{gesture.label}</span>
            <select
              className="sound-settings-select"
              value={gestureMapping[gesture.id] ?? DEFAULT_GESTURE_MAPPING[gesture.id]}
              onChange={(event) => onGestureMappingChange(gesture.id, event.target.value)}
              aria-label={`${gesture.label} 오브젝트 매핑`}
            >
              {INSTRUMENTS.map((instrument) => (
                <option key={instrument.id} value={instrument.id}>
                  {instrument.label}
                </option>
              ))}
            </select>
          </label>
        ))}
      </div>
    </div>
  );
}

export default function App() {
  useInstrumentLayout();
  useLegacyMainRuntime();
  const { soundButtonLabel, requestStart, requestSoundToggle } = useMainControls();
  const [showTutorial, setShowTutorial] = React.useState(false);
  const [showCredits, setShowCredits] = React.useState(false);
  const [tutorialPracticeEnabled, setTutorialPracticeEnabled] = React.useState(false);
  const [bgmEnabled, setBgmEnabled] = React.useState(true);
  const [bgmVolume, setBgmVolume] = React.useState(0.22);
  const [bgmSelection, setBgmSelection] = React.useState("auto");
  const [selectedModelId, setSelectedModelId] = React.useState(() => getCurrentModelId());
  const [modelLoading, setModelLoading] = React.useState(false);
  const [landingVisible, setLandingVisible] = React.useState(true);
  const [showSoundSettings, setShowSoundSettings] = React.useState(false);
  const [instrumentVolumes, setInstrumentVolumes] = React.useState(() => buildInstrumentVolumeState());
  const [gestureMapping, setGestureMapping] = React.useState(() => buildGestureMappingState());
  const [objectSampleMapping, setObjectSampleMapping] = React.useState(() => buildObjectSampleMappingState());

  const [isHudCollapsed, setIsHudCollapsed] = React.useState(false);

  const landingBgmRef = React.useRef(null);

  const modelConfigs = getAllModelConfigs();
  const activeBgmTrack = resolveBgmTrack(bgmSelection);
  const sceneUiState = showCredits
    ? "credits"
    : showTutorial
      ? "tutorial"
      : landingVisible
        ? "landing"
        : "playing";
  const shouldPlayBgm = bgmEnabled && sceneUiState !== "playing";

  const requestCameraRefresh = React.useCallback(() => {
    window.dispatchEvent(new CustomEvent("jamjam:refresh-camera-target"));
  }, []);

  React.useEffect(() => {
    const landingOverlay = document.getElementById("landingOverlay");
    if (!landingOverlay) return undefined;

    const syncLandingVisibility = () => {
      setLandingVisible(!landingOverlay.classList.contains("is-hidden"));
    };

    syncLandingVisibility();

    const observer = new MutationObserver(syncLandingVisibility);
    observer.observe(landingOverlay, {
      attributes: true,
      attributeFilter: ["class"]
    });

    return () => observer.disconnect();
  }, []);

  React.useEffect(() => {
    const handleInstrumentVolumesChanged = (event) => {
      const nextVolumes = event.detail?.volumes || {};
      setInstrumentVolumes((current) => ({
        ...current,
        ...Object.fromEntries(
          INSTRUMENTS.map((instrument) => [instrument.id, nextVolumes[instrument.id] ?? 1])
        )
      }));
    };

    window.addEventListener("jamjam:instrument-volumes-changed", handleInstrumentVolumesChanged);
    return () => {
      window.removeEventListener("jamjam:instrument-volumes-changed", handleInstrumentVolumesChanged);
    };
  }, []);

  React.useEffect(() => {
    const handleGestureMappingChanged = (event) => {
      const nextMapping = event.detail?.mapping || {};
      setGestureMapping((current) => ({
        ...current,
        ...Object.fromEntries(
          GESTURE_OPTIONS.map((gesture) => [
            gesture.id,
            nextMapping[gesture.id] ?? DEFAULT_GESTURE_MAPPING[gesture.id]
          ])
        )
      }));
    };

    window.addEventListener("jamjam:gesture-mapping-changed", handleGestureMappingChanged);
    return () => {
      window.removeEventListener("jamjam:gesture-mapping-changed", handleGestureMappingChanged);
    };
  }, []);

  React.useEffect(() => {
    const handleObjectSampleMappingChanged = (event) => {
      const nextMapping = event.detail?.mapping || {};
      setObjectSampleMapping((current) => ({
        ...current,
        ...Object.fromEntries(
          INSTRUMENTS.map((instrument) => [
            instrument.id,
            nextMapping[instrument.id] ?? DEFAULT_OBJECT_SAMPLE_MAPPING[instrument.id]
          ])
        )
      }));
    };

    window.addEventListener("jamjam:object-sample-mapping-changed", handleObjectSampleMappingChanged);
    return () => {
      window.removeEventListener("jamjam:object-sample-mapping-changed", handleObjectSampleMappingChanged);
    };
  }, []);

  React.useEffect(() => {
    const audio = new Audio(activeBgmTrack.src);
    audio.preload = "auto";
    audio.loop = true;
    audio.volume = bgmVolume;
    landingBgmRef.current = audio;

    let disposed = false;

    const tryPlay = () => {
      if (disposed) return;
      if (!landingBgmRef.current || !shouldPlayBgm) return;
      const playPromise = audio.play();
      if (playPromise && typeof playPromise.catch === "function") {
        playPromise.catch(() => {
          // 첫 사용자 입력 또는 미디어 버퍼 준비 이후 다시 재생을 시도합니다.
        });
      }
    };

    const handleFirstInteraction = () => {
      tryPlay();
    };

    const handleCanPlay = () => {
      tryPlay();
    };

    audio.addEventListener("canplay", handleCanPlay);
    audio.addEventListener("canplaythrough", handleCanPlay);
    audio.load();

    window.addEventListener("pointerdown", handleFirstInteraction, { passive: true });
    window.addEventListener("touchend", handleFirstInteraction, { passive: true });
    window.addEventListener("keydown", handleFirstInteraction);

    tryPlay();

    return () => {
      disposed = true;
      audio.removeEventListener("canplay", handleCanPlay);
      audio.removeEventListener("canplaythrough", handleCanPlay);
      window.removeEventListener("pointerdown", handleFirstInteraction);
      window.removeEventListener("touchend", handleFirstInteraction);
      window.removeEventListener("keydown", handleFirstInteraction);
      if (landingBgmRef.current) {
        landingBgmRef.current.pause();
        landingBgmRef.current.removeAttribute('src');
        landingBgmRef.current = null;
      }
    };
  }, [activeBgmTrack.src, bgmVolume, shouldPlayBgm]);

  // 배경음악 켜기/끄기 상태 관리
  React.useEffect(() => {
    if (!landingBgmRef.current) return;
    if (shouldPlayBgm) {
      landingBgmRef.current.play().catch(() => { });
    } else {
      landingBgmRef.current.pause();
    }
  }, [shouldPlayBgm]);

  // 음량 변경
  React.useEffect(() => {
    if (landingBgmRef.current) {
      landingBgmRef.current.volume = bgmVolume;
    }
  }, [bgmVolume]);

  const toggleBgm = () => {
    setBgmEnabled(!bgmEnabled);
  };

  const handleVolumeChange = (event) => {
    const newVolume = parseFloat(event.target.value);
    setBgmVolume(newVolume);
  };

  const handleBgmSelectionChange = (event) => {
    setBgmSelection(event.target.value);
  };

  const handleInstrumentVolumeChange = (instrumentId, value) => {
    const nextVolume = parseFloat(value);
    setInstrumentVolumes((current) => ({
      ...current,
      [instrumentId]: nextVolume
    }));
    saveInstrumentVolume(instrumentId, nextVolume);
  };

  const handleResetInstrumentVolumes = () => {
    resetSavedInstrumentVolumes();
    setInstrumentVolumes(buildInstrumentVolumeState());
  };

  const handleObjectSampleChange = (instrumentId, sampleId) => {
    setObjectSampleMapping((current) => ({
      ...current,
      [instrumentId]: sampleId
    }));
    saveObjectSampleMapping(instrumentId, sampleId);
  };

  const handleResetObjectSampleMapping = () => {
    resetSavedObjectSampleMapping();
    setObjectSampleMapping(buildObjectSampleMappingState());
  };

  const handleGestureMappingChange = (gestureId, instrumentId) => {
    setGestureMapping((current) => ({
      ...current,
      [gestureId]: instrumentId
    }));
    saveGestureMapping(gestureId, instrumentId);
  };

  const handleResetGestureMapping = () => {
    resetSavedGestureMapping();
    setGestureMapping(buildGestureMappingState());
  };

  const handleStartClick = () => {
    setTutorialPracticeEnabled(false);
    setShowTutorial(true);
  };

  const closeTutorial = () => {
    setShowTutorial(false);
    requestStart();
    requestCameraRefresh();
  };

  const startTutorialPractice = () => {
    if (!tutorialPracticeEnabled) {
      setTutorialPracticeEnabled(true);
    }
    requestCameraRefresh();
  };

  const skipTutorialAndStart = () => {
    setShowTutorial(false);
    setTutorialPracticeEnabled(false);
    requestStart();
    requestCameraRefresh();
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
    if (!landingVisible || tutorialPracticeEnabled) {
      requestCameraRefresh();
    }
  }, [landingVisible, tutorialPracticeEnabled, requestCameraRefresh]);

  // 모델 로딩 이벤트 리스너
  React.useEffect(() => {
    const handleModelLoading = () => setModelLoading(true);
    const handleModelLoaded = (event) => {
      setModelLoading(false);
      setSelectedModelId(event.detail.modelId);
    };
    const handleModelError = () => setModelLoading(false);

    window.addEventListener("jamjam:model-loading", handleModelLoading);
    window.addEventListener("jamjam:model-loaded", handleModelLoaded);
    window.addEventListener("jamjam:model-load-error", handleModelError);

    return () => {
      window.removeEventListener("jamjam:model-loading", handleModelLoading);
      window.removeEventListener("jamjam:model-loaded", handleModelLoaded);
      window.removeEventListener("jamjam:model-load-error", handleModelError);
    };
  }, []);

  const handleModelChange = async (event) => {
    const newModelId = event.target.value;
    if (newModelId === selectedModelId) return;

    const success = await switchModel(newModelId);
    if (!success) {
      alert("모델 로딩에 실패했습니다. 콘솔을 확인해주세요.");
    }
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
            <header className="tutorial-header">
              <span className="tutorial-icon" aria-hidden="true">✨</span>
              <h2 className="tutorial-title">숲의 지휘자 연습실</h2>
            </header>

            <p className="tutorial-description">
              나의 손동작을 카메라에 비춰보세요!<br />
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
                <p id="tutorialVisionSummary" className="tutorial-vision-summary">아래 화면을 보며 동작을 연습해 보세요.</p>
              </div>
              <TutorialGesturePreview />
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

      <main
        id="scene"
        className="scene hide-camera"
        data-fever="off"
        data-ui={sceneUiState}
        data-bgm={shouldPlayBgm ? "on" : "off"}
      >
        <section id="landingOverlay" className="landing-overlay" aria-live="polite">
          <div className="landing-panel">
            <div className="landing-logo-container">
              <img className="landing-logo" src={logoPng} srcSet={logoWebp} alt="JamJam Beat Logo" loading="eager" />
            </div>

            <LandingModelSelect
              selectedModelId={selectedModelId}
              modelLoading={modelLoading}
              modelConfigs={modelConfigs}
              onChange={handleModelChange}
            />

            <button id="landingStartButton" type="button" onClick={handleStartClick}>시작하기</button>
            <button className="landing-credits-button" type="button" onClick={openCredits}>만든 사람들</button>
            <button className="landing-exit-button" type="button" onClick={handleExitClick}>종료하기</button>
          </div>
        </section>
        <BgmPlayer
          activeBgmTrack={activeBgmTrack}
          bgmEnabled={bgmEnabled}
          bgmSelection={bgmSelection}
          bgmVolume={bgmVolume}
          onSelectionChange={handleBgmSelectionChange}
          onToggle={toggleBgm}
          onVolumeChange={handleVolumeChange}
        />

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
          <div className={`camera-shell${showTutorial ? " is-muted" : ""}`} aria-hidden="true">
            <video id="webcam" autoPlay playsInline muted />
          </div>
          <canvas id="handCanvas" className={showTutorial ? "is-muted" : undefined} />
          <div className="guide-silhouette" aria-hidden="true" />
          <div id="gestureSquirrelEffect" className="gesture-squirrel" aria-hidden="true">🐿️</div>
          <p id="pulseMessage" className="pulse-message">손을 잼잼! 해서 숲을 깨워봐!</p>
        </section>

        <section className={`hud hud-panel${isHudCollapsed ? " is-collapsed" : ""}`} aria-live="polite">
          <div className="hud-header">
            <h2>잼잼비트 숲의 입구</h2>
            <button
              type="button"
              className="hud-collapse-toggle"
              onClick={() => setIsHudCollapsed(!isHudCollapsed)}
              aria-label={isHudCollapsed ? "HUD 펼치기" : "HUD 접기"}
            >
              {isHudCollapsed ? "▼" : "▲"}
            </button>
          </div>
          {!isHudCollapsed && (
            <>
              <p id="status" className="status">카메라 준비 중...</p>
              <div className="hud-actions">
                <button id="soundUnlockButton" className="sound-unlock" type="button" onClick={requestSoundToggle}>{soundButtonLabel}</button>
                <button id="testModeToggleButton" className="test-mode-toggle" type="button">테스트 모드 켜기</button>
                <button
                  className="hud-settings-toggle"
                  type="button"
                  onClick={() => setShowSoundSettings((current) => !current)}
                  aria-expanded={showSoundSettings}
                >
                  {showSoundSettings ? "환경설정 닫기" : "환경설정"}
                </button>
              </div>
              {showSoundSettings && (
                <SoundSettingsPanel
                  instrumentVolumes={instrumentVolumes}
                  gestureMapping={gestureMapping}
                  objectSampleMapping={objectSampleMapping}
                  onVolumeChange={handleInstrumentVolumeChange}
                  onObjectSampleChange={handleObjectSampleChange}
                  onGestureMappingChange={handleGestureMappingChange}
                  onReset={handleResetInstrumentVolumes}
                  onResetObjectSampleMapping={handleResetObjectSampleMapping}
                  onResetGestureMapping={handleResetGestureMapping}
                />
              )}
              <div className="hud-footer">
                <button className="hud-exit-button" type="button" onClick={handleExitClick}>프로그램 끝내기</button>
                <a className="hud-main-link" href="./index.html">메인화면으로</a>
                <p className="privacy-note">영상은 서버로 전송되지 않고 오직 연주에만 사용됩니다.</p>
              </div>
            </>
          )}
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
