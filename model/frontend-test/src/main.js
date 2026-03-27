// main.js - frontend-test webcam sequence inference monitor
import "./styles.css";
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import {
  getConfiguredHandLandmarkerTaskPath,
  getConfiguredMediaPipeWasmRoot
} from "../../frontend/src/js/env_config.js";
import {
  getModelPrediction as getSequenceModelPrediction,
  getModelInferenceStatus,
  preloadModel,
  pushNoHand
} from "../../frontend/src/js/model_inference_sequence.js";
import {
  resolveGesture,
  resetGestureState,
  setModelPredictionProvider
} from "../../frontend/src/js/gestures.js";

const LANDMARK_STALE_MS = 260;
const HAND_KEYS = ["left", "right"];
const DEFAULT_SETTINGS = Object.freeze({
  inferWidth: 96,
  inferFps: 15,
  modelIntervalMs: 150,
  tau: 0.85,
  gestureVoteN: 5,
  gestureDebounceK: 3,
  gestureClearFrames: 2,
  grabMotionWindow: 4,
  grabMotionThreshold: 0.028,
  numHands: 2,
  minHandDetectionConfidence: 0.25,
  minHandPresenceConfidence: 0.25,
  minTrackingConfidence: 0.25
});
const VIEWER_LIKE_SETTINGS = Object.freeze({
  inferWidth: 0,
  inferFps: 30,
  modelIntervalMs: 60,
  tau: 0.9,
  gestureVoteN: 5,
  gestureDebounceK: 3,
  gestureClearFrames: 2,
  grabMotionWindow: 4,
  grabMotionThreshold: 0.028,
  numHands: 1,
  minHandDetectionConfidence: 0.5,
  minHandPresenceConfidence: 0.5,
  minTrackingConfidence: 0.5
});
const PRESET_SETTINGS = Object.freeze({
  "frontend-default": DEFAULT_SETTINGS,
  "viewer-like": VIEWER_LIKE_SETTINGS
});
const SETTINGS_QUERY_KEYS = Object.freeze({
  inferWidth: "inferWidth",
  inferFps: "inferFps",
  modelIntervalMs: "modelIntervalMs",
  tau: "tau",
  gestureVoteN: "gestureVoteN",
  gestureDebounceK: "gestureDebounceK",
  gestureClearFrames: "gestureClearFrames",
  grabMotionWindow: "grabMotionWindow",
  grabMotionThreshold: "grabMotionThreshold",
  numHands: "numHands",
  minHandDetectionConfidence: "mpDetConf",
  minHandPresenceConfidence: "mpPresenceConf",
  minTrackingConfidence: "mpTrackConf"
});
const RUNTIME_QUERY_KEYS = Object.freeze({
  bundle: "runtimeBundle",
  root: "runtimeRoot"
});
const RUNTIME_OPTIONS = Object.freeze([
  {
    key: "frontend-sequence",
    label: "Frontend Runtime (7-class)",
    runtimeRoot: "/runtime_sequence/"
  },
  {
    key: "grab-run-20260323-005909",
    label: "Grab Fine-tune 20260323_005909 (8-class)",
    runtimeRoot: "/runtime_sequence_grab_20260323_005909/"
  }
]);
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
const DEFAULT_CLASS_NAMES = ["neutral", "fist", "open_palm", "V", "pinky", "animal", "k-heart", "grab"];
const runtimeOptionsMarkup = RUNTIME_OPTIONS.map(
  (option) => `<option value="${option.key}">${option.label}</option>`
).join("");

const app = document.querySelector("#app");

app.innerHTML = `
  <main class="monitor-shell">
    <header class="monitor-header">
      <div class="monitor-title-wrap">
        <h1>JamJamBeat Frontend Sequence Monitor</h1>
        <p>웹캠 → MediaPipe → pos_scale sequence ONNX → raw/final gesture를 실시간으로 확인합니다.</p>
      </div>
      <div class="monitor-actions">
        <button class="monitor-button primary" id="startButton" type="button">Start Monitor</button>
        <button class="monitor-button" id="stopButton" type="button">Stop</button>
      </div>
    </header>

    <section class="monitor-status">
      <article class="panel status-panel">
        <div class="status-pill" id="statusPill">Idle</div>
        <p class="status-text" id="statusText">Start Monitor를 눌러 Windows 웹캠과 sequence 추론 모니터를 시작하세요.</p>
        <p class="status-subtext" id="statusSubtext">기본값: inferFps=15, modelIntervalMs=150, tau=0.85, vote=5, debounce=3, clear=2, motion=4/0.028</p>
      </article>
      <section class="panel latency-strip">
        <article class="metric-card">
          <span class="label">Current</span>
          <span class="value" id="latencyCurrent">-</span>
        </article>
        <article class="metric-card">
          <span class="label">Average</span>
          <span class="value" id="latencyAvg">-</span>
        </article>
        <article class="metric-card">
          <span class="label">P50</span>
          <span class="value" id="latencyP50">-</span>
        </article>
        <article class="metric-card">
          <span class="label">P95</span>
          <span class="value" id="latencyP95">-</span>
        </article>
      </section>
    </section>

    <section class="panel settings-panel">
      <div class="settings-header">
        <div>
          <h2 class="section-title">Inference Settings</h2>
          <p class="settings-copy">MediaPipe와 detection cadence를 브라우저에서 직접 바꾸고, 재시작으로 새 설정을 반영합니다.</p>
        </div>
        <div class="settings-state-wrap">
          <span class="settings-state-badge applied" id="settingsStateBadge">Applied</span>
          <span class="settings-summary" id="settingsSummary">Frontend Runtime (7-class) · inferWidth=96 · inferFps=15 · modelIntervalMs=150 · numHands=2</span>
        </div>
      </div>

      <div class="settings-grid">
        <label class="settings-field">
          <span class="settings-label">Checkpoint</span>
          <select id="runtimeBundleSelect">${runtimeOptionsMarkup}</select>
          <span class="settings-help">runtime bundle 전환 시 페이지를 다시 로드합니다.</span>
        </label>

        <label class="settings-field">
          <span class="settings-label">Preset</span>
          <select id="settingsPreset">
            <option value="frontend-default">Frontend Default</option>
            <option value="viewer-like">Viewer-like</option>
            <option value="custom">Custom</option>
          </select>
        </label>

        <label class="settings-field">
          <span class="settings-label">inferWidth</span>
          <input id="inferWidthInput" type="number" min="0" max="640" step="1" />
          <span class="settings-help">0 = full frame</span>
        </label>

        <label class="settings-field">
          <span class="settings-label">inferFps</span>
          <input id="inferFpsInput" type="number" min="1" max="60" step="1" />
        </label>

        <label class="settings-field">
          <span class="settings-label">modelIntervalMs</span>
          <input id="modelIntervalInput" type="number" min="60" max="400" step="1" />
        </label>

        <label class="settings-field">
          <span class="settings-label">tau</span>
          <input id="tauInput" type="number" min="0.50" max="0.99" step="0.01" />
          <span class="settings-help">0.50 ~ 0.99, 소수 둘째 자리까지</span>
        </label>

        <label class="settings-field">
          <span class="settings-label">Vote N</span>
          <input id="gestureVoteNInput" type="number" min="1" max="9" step="1" />
        </label>

        <label class="settings-field">
          <span class="settings-label">Debounce K</span>
          <input id="gestureDebounceKInput" type="number" min="1" max="9" step="1" />
        </label>

        <label class="settings-field">
          <span class="settings-label">Clear Frames</span>
          <input id="gestureClearFramesInput" type="number" min="1" max="6" step="1" />
        </label>

        <label class="settings-field">
          <span class="settings-label">Grab Motion Window</span>
          <input id="grabMotionWindowInput" type="number" min="1" max="7" step="1" />
        </label>

        <label class="settings-field">
          <span class="settings-label">Grab Motion Threshold</span>
          <input id="grabMotionThresholdInput" type="number" min="0.001" max="0.300" step="0.001" />
        </label>

        <label class="settings-field">
          <span class="settings-label">numHands</span>
          <select id="numHandsSelect">
            <option value="1">1</option>
            <option value="2">2</option>
          </select>
        </label>

        <label class="settings-field">
          <span class="settings-label">minHandDetectionConfidence</span>
          <input id="mpDetConfInput" type="number" min="0" max="1" step="0.05" />
        </label>

        <label class="settings-field">
          <span class="settings-label">minHandPresenceConfidence</span>
          <input id="mpPresenceConfInput" type="number" min="0" max="1" step="0.05" />
        </label>

        <label class="settings-field">
          <span class="settings-label">minTrackingConfidence</span>
          <input id="mpTrackConfInput" type="number" min="0" max="1" step="0.05" />
        </label>
      </div>

      <div class="settings-actions">
        <button class="monitor-button primary" id="applySettingsButton" type="button">Apply &amp; Restart</button>
        <button class="monitor-button" id="resetSettingsButton" type="button">Reset to Frontend Default</button>
      </div>
      <p class="settings-hint" id="settingsHint">현재 활성 설정과 폼 값이 동일합니다.</p>
    </section>

    <section class="monitor-layout">
      <section class="visual-stack">
        <section class="panel visual-panel">
          <div class="visual-grid">
            <article class="visual-card">
              <h2>Webcam + Raw Landmarks</h2>
              <div class="visual-frame">
                <video id="webcam" autoplay playsinline muted></video>
                <canvas id="rawCanvas"></canvas>
              </div>
              <p class="visual-caption" id="rawCaption">손 감지 전</p>
            </article>
            <article class="visual-card">
              <h2>pos_scale Preview</h2>
              <div class="visual-frame normalized-frame">
                <canvas id="normalizedCanvas" width="320" height="240"></canvas>
              </div>
              <p class="visual-caption" id="normalizedCaption">정규화 대상 손: 없음</p>
            </article>
          </div>
        </section>

        <article class="panel hands-panel">
          <h2 class="section-title">Per-hand Monitor</h2>
          <div class="hand-cards">
            <section class="hand-card" id="handCardRight"></section>
            <section class="hand-card" id="handCardLeft"></section>
          </div>
        </article>
      </section>

      <section class="side-stack">
        <article class="panel meta-panel">
          <h2 class="section-title">Loaded Runtime Metadata</h2>
          <div class="kv-grid">
            <div class="kv-item"><span class="k">bundle_id</span><span class="v" id="metaBundleId">-</span></div>
            <div class="kv-item"><span class="k">model_id</span><span class="v" id="metaModelId">-</span></div>
            <div class="kv-item"><span class="k">normalization</span><span class="v" id="metaNormalization">-</span></div>
            <div class="kv-item"><span class="k">seq_len</span><span class="v" id="metaSeqLen">-</span></div>
            <div class="kv-item"><span class="k">tau</span><span class="v" id="metaTau">-</span></div>
            <div class="kv-item"><span class="k">runtimeRoot</span><span class="v" id="metaRuntimeRoot">-</span></div>
            <div class="kv-item"><span class="k">checkpoint_fingerprint</span><span class="v" id="metaFingerprint">-</span></div>
            <div class="kv-item"><span class="k">focal gamma</span><span class="v" id="metaGamma">-</span></div>
          </div>
        </article>

        <article class="panel notes-panel">
          <h2 class="section-title">Notes</h2>
          <ul>
            <li>체크포인트 드롭다운에서 기존 frontend runtime과 grab fine-tune runtime을 전환할 수 있습니다.</li>
            <li>새 grab bundle은 <code>/runtime_sequence_grab_20260323_005909/</code>를 사용하며 class 7 = <code>grab</code>를 포함합니다.</li>
            <li>왼손은 현재 frontend 구현과 동일하게 추론 비활성 상태를 명시합니다.</li>
          </ul>
        </article>
      </section>
    </section>
  </main>
`;

const refs = {
  startButton: document.querySelector("#startButton"),
  stopButton: document.querySelector("#stopButton"),
  statusPill: document.querySelector("#statusPill"),
  statusText: document.querySelector("#statusText"),
  statusSubtext: document.querySelector("#statusSubtext"),
  latencyCurrent: document.querySelector("#latencyCurrent"),
  latencyAvg: document.querySelector("#latencyAvg"),
  latencyP50: document.querySelector("#latencyP50"),
  latencyP95: document.querySelector("#latencyP95"),
  webcam: document.querySelector("#webcam"),
  rawCanvas: document.querySelector("#rawCanvas"),
  normalizedCanvas: document.querySelector("#normalizedCanvas"),
  rawCaption: document.querySelector("#rawCaption"),
  normalizedCaption: document.querySelector("#normalizedCaption"),
  metaBundleId: document.querySelector("#metaBundleId"),
  metaModelId: document.querySelector("#metaModelId"),
  metaNormalization: document.querySelector("#metaNormalization"),
  metaSeqLen: document.querySelector("#metaSeqLen"),
  metaTau: document.querySelector("#metaTau"),
  metaRuntimeRoot: document.querySelector("#metaRuntimeRoot"),
  metaFingerprint: document.querySelector("#metaFingerprint"),
  metaGamma: document.querySelector("#metaGamma"),
  handCardLeft: document.querySelector("#handCardLeft"),
  handCardRight: document.querySelector("#handCardRight"),
  runtimeBundleSelect: document.querySelector("#runtimeBundleSelect"),
  settingsPreset: document.querySelector("#settingsPreset"),
  inferWidthInput: document.querySelector("#inferWidthInput"),
  inferFpsInput: document.querySelector("#inferFpsInput"),
  modelIntervalInput: document.querySelector("#modelIntervalInput"),
  tauInput: document.querySelector("#tauInput"),
  gestureVoteNInput: document.querySelector("#gestureVoteNInput"),
  gestureDebounceKInput: document.querySelector("#gestureDebounceKInput"),
  gestureClearFramesInput: document.querySelector("#gestureClearFramesInput"),
  grabMotionWindowInput: document.querySelector("#grabMotionWindowInput"),
  grabMotionThresholdInput: document.querySelector("#grabMotionThresholdInput"),
  numHandsSelect: document.querySelector("#numHandsSelect"),
  mpDetConfInput: document.querySelector("#mpDetConfInput"),
  mpPresenceConfInput: document.querySelector("#mpPresenceConfInput"),
  mpTrackConfInput: document.querySelector("#mpTrackConfInput"),
  applySettingsButton: document.querySelector("#applySettingsButton"),
  resetSettingsButton: document.querySelector("#resetSettingsButton"),
  settingsStateBadge: document.querySelector("#settingsStateBadge"),
  settingsSummary: document.querySelector("#settingsSummary"),
  settingsHint: document.querySelector("#settingsHint")
};

const rawCtx = refs.rawCanvas.getContext("2d");
const normalizedCtx = refs.normalizedCanvas.getContext("2d");
const inferenceCanvas = document.createElement("canvas");
const inferenceCtx = inferenceCanvas.getContext("2d", { willReadFrequently: true });

function clampFloat(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function normalizeInferWidth(value, fallback = DEFAULT_SETTINGS.inferWidth) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  const rounded = Math.round(numeric);
  if (rounded === 0) return 0;
  return Math.min(640, Math.max(96, rounded));
}

function normalizeInferFps(value, fallback = DEFAULT_SETTINGS.inferFps) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(60, Math.max(1, Math.round(numeric)));
}

function normalizeModelIntervalMs(value, fallback = DEFAULT_SETTINGS.modelIntervalMs) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(400, Math.max(60, Math.round(numeric)));
}

function normalizeTau(value, fallback = DEFAULT_SETTINGS.tau) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Number(clampFloat(numeric, 0.5, 0.99).toFixed(2));
}

function normalizeGestureVoteN(value, fallback = DEFAULT_SETTINGS.gestureVoteN) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(9, Math.max(1, Math.round(numeric)));
}

function normalizeGestureDebounceK(value, fallback = DEFAULT_SETTINGS.gestureDebounceK) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(9, Math.max(1, Math.round(numeric)));
}

function normalizeGestureClearFrames(value, fallback = DEFAULT_SETTINGS.gestureClearFrames) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(6, Math.max(1, Math.round(numeric)));
}

function normalizeGrabMotionWindow(value, fallback = DEFAULT_SETTINGS.grabMotionWindow) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(7, Math.max(1, Math.round(numeric)));
}

function normalizeGrabMotionThreshold(value, fallback = DEFAULT_SETTINGS.grabMotionThreshold) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Number(clampFloat(numeric, 0.001, 0.3).toFixed(3));
}

function normalizeNumHands(value, fallback = DEFAULT_SETTINGS.numHands) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Math.min(2, Math.max(1, Math.round(numeric)));
}

function normalizeThreshold(value, fallback) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return fallback;
  return Number(clampFloat(numeric, 0, 1).toFixed(2));
}

function getRuntimeOptionByKey(key) {
  return RUNTIME_OPTIONS.find((option) => option.key === key) || null;
}

function getRuntimeOptionByRoot(runtimeRoot) {
  const normalized = normalizeRootPath(runtimeRoot, "/runtime_sequence/");
  return RUNTIME_OPTIONS.find((option) => option.runtimeRoot === normalized) || null;
}

function getDefaultRuntimeOption() {
  return RUNTIME_OPTIONS[0];
}

function loadRuntimeSelectionFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const bundleKey = params.get(RUNTIME_QUERY_KEYS.bundle);
  const byKey = getRuntimeOptionByKey(bundleKey);
  if (byKey) return byKey.key;

  const runtimeRoot = params.get(RUNTIME_QUERY_KEYS.root);
  const byRoot = getRuntimeOptionByRoot(runtimeRoot);
  if (byRoot) return byRoot.key;

  return getDefaultRuntimeOption().key;
}

function cloneSettings(settings) {
  return {
    inferWidth: settings.inferWidth,
    inferFps: settings.inferFps,
    modelIntervalMs: settings.modelIntervalMs,
    tau: settings.tau,
    gestureVoteN: settings.gestureVoteN,
    gestureDebounceK: settings.gestureDebounceK,
    gestureClearFrames: settings.gestureClearFrames,
    grabMotionWindow: settings.grabMotionWindow,
    grabMotionThreshold: settings.grabMotionThreshold,
    numHands: settings.numHands,
    minHandDetectionConfidence: settings.minHandDetectionConfidence,
    minHandPresenceConfidence: settings.minHandPresenceConfidence,
    minTrackingConfidence: settings.minTrackingConfidence
  };
}

function normalizeSettings(settings = {}) {
  return {
    inferWidth: normalizeInferWidth(settings.inferWidth, DEFAULT_SETTINGS.inferWidth),
    inferFps: normalizeInferFps(settings.inferFps, DEFAULT_SETTINGS.inferFps),
    modelIntervalMs: normalizeModelIntervalMs(settings.modelIntervalMs, DEFAULT_SETTINGS.modelIntervalMs),
    tau: normalizeTau(settings.tau, DEFAULT_SETTINGS.tau),
    gestureVoteN: normalizeGestureVoteN(settings.gestureVoteN, DEFAULT_SETTINGS.gestureVoteN),
    gestureDebounceK: normalizeGestureDebounceK(settings.gestureDebounceK, DEFAULT_SETTINGS.gestureDebounceK),
    gestureClearFrames: normalizeGestureClearFrames(settings.gestureClearFrames, DEFAULT_SETTINGS.gestureClearFrames),
    grabMotionWindow: normalizeGrabMotionWindow(settings.grabMotionWindow, DEFAULT_SETTINGS.grabMotionWindow),
    grabMotionThreshold: normalizeGrabMotionThreshold(
      settings.grabMotionThreshold,
      DEFAULT_SETTINGS.grabMotionThreshold
    ),
    numHands: normalizeNumHands(settings.numHands, DEFAULT_SETTINGS.numHands),
    minHandDetectionConfidence: normalizeThreshold(
      settings.minHandDetectionConfidence,
      DEFAULT_SETTINGS.minHandDetectionConfidence
    ),
    minHandPresenceConfidence: normalizeThreshold(
      settings.minHandPresenceConfidence,
      DEFAULT_SETTINGS.minHandPresenceConfidence
    ),
    minTrackingConfidence: normalizeThreshold(
      settings.minTrackingConfidence,
      DEFAULT_SETTINGS.minTrackingConfidence
    )
  };
}

function settingsEqual(a, b) {
  return (
    a.inferWidth === b.inferWidth &&
    a.inferFps === b.inferFps &&
    a.modelIntervalMs === b.modelIntervalMs &&
    Math.abs(a.tau - b.tau) < 1e-6 &&
    a.gestureVoteN === b.gestureVoteN &&
    a.gestureDebounceK === b.gestureDebounceK &&
    a.gestureClearFrames === b.gestureClearFrames &&
    a.grabMotionWindow === b.grabMotionWindow &&
    Math.abs(a.grabMotionThreshold - b.grabMotionThreshold) < 1e-6 &&
    a.numHands === b.numHands &&
    Math.abs(a.minHandDetectionConfidence - b.minHandDetectionConfidence) < 1e-6 &&
    Math.abs(a.minHandPresenceConfidence - b.minHandPresenceConfidence) < 1e-6 &&
    Math.abs(a.minTrackingConfidence - b.minTrackingConfidence) < 1e-6
  );
}

function getPresetKeyForSettings(settings) {
  if (settingsEqual(settings, DEFAULT_SETTINGS)) return "frontend-default";
  if (settingsEqual(settings, VIEWER_LIKE_SETTINGS)) return "viewer-like";
  return "custom";
}

function loadSettingsFromUrl() {
  const params = new URLSearchParams(window.location.search);
  return normalizeSettings({
    inferWidth: params.get(SETTINGS_QUERY_KEYS.inferWidth),
    inferFps: params.get(SETTINGS_QUERY_KEYS.inferFps),
    modelIntervalMs: params.get(SETTINGS_QUERY_KEYS.modelIntervalMs),
    tau: params.get(SETTINGS_QUERY_KEYS.tau),
    gestureVoteN: params.get(SETTINGS_QUERY_KEYS.gestureVoteN),
    gestureDebounceK: params.get(SETTINGS_QUERY_KEYS.gestureDebounceK),
    gestureClearFrames: params.get(SETTINGS_QUERY_KEYS.gestureClearFrames),
    grabMotionWindow: params.get(SETTINGS_QUERY_KEYS.grabMotionWindow),
    grabMotionThreshold: params.get(SETTINGS_QUERY_KEYS.grabMotionThreshold),
    numHands: params.get(SETTINGS_QUERY_KEYS.numHands),
    minHandDetectionConfidence: params.get(SETTINGS_QUERY_KEYS.minHandDetectionConfidence),
    minHandPresenceConfidence: params.get(SETTINGS_QUERY_KEYS.minHandPresenceConfidence),
    minTrackingConfidence: params.get(SETTINGS_QUERY_KEYS.minTrackingConfidence)
  });
}

function getActiveSettings() {
  return state.settings.active || cloneSettings(DEFAULT_SETTINGS);
}

function getActiveRuntimeOption() {
  return getRuntimeOptionByKey(state.runtime.activeKey) || getDefaultRuntimeOption();
}

function getFormRuntimeOption() {
  return getRuntimeOptionByKey(refs.runtimeBundleSelect.value) || getDefaultRuntimeOption();
}

function getSettingsSummary(settings) {
  const widthLabel = settings.inferWidth === 0 ? "inferWidth=full" : `inferWidth=${settings.inferWidth}`;
  return [
    widthLabel,
    `inferFps=${settings.inferFps}`,
    `modelIntervalMs=${settings.modelIntervalMs}`,
    `tau=${settings.tau.toFixed(2)}`,
    `vote=${settings.gestureVoteN}`,
    `debounce=${settings.gestureDebounceK}`,
    `clear=${settings.gestureClearFrames}`,
    `motion=${settings.grabMotionWindow}/${settings.grabMotionThreshold.toFixed(3)}`,
    `numHands=${settings.numHands}`,
    `conf=${settings.minHandDetectionConfidence.toFixed(2)}/${settings.minHandPresenceConfidence.toFixed(2)}/${settings.minTrackingConfidence.toFixed(2)}`
  ].join(" · ");
}

function getInferIntervalMs(settings = getActiveSettings()) {
  return Math.round(1000 / settings.inferFps);
}

function buildUrlWithSelections(settings, runtimeKey) {
  const url = new URL(window.location.href);
  const runtimeOption = getRuntimeOptionByKey(runtimeKey) || getDefaultRuntimeOption();
  url.searchParams.set(SETTINGS_QUERY_KEYS.inferWidth, String(settings.inferWidth));
  url.searchParams.set(SETTINGS_QUERY_KEYS.inferFps, String(settings.inferFps));
  url.searchParams.set(SETTINGS_QUERY_KEYS.modelIntervalMs, String(settings.modelIntervalMs));
  url.searchParams.set(SETTINGS_QUERY_KEYS.tau, String(settings.tau));
  url.searchParams.set(SETTINGS_QUERY_KEYS.gestureVoteN, String(settings.gestureVoteN));
  url.searchParams.set(SETTINGS_QUERY_KEYS.gestureDebounceK, String(settings.gestureDebounceK));
  url.searchParams.set(SETTINGS_QUERY_KEYS.gestureClearFrames, String(settings.gestureClearFrames));
  url.searchParams.set(SETTINGS_QUERY_KEYS.grabMotionWindow, String(settings.grabMotionWindow));
  url.searchParams.set(SETTINGS_QUERY_KEYS.grabMotionThreshold, String(settings.grabMotionThreshold));
  url.searchParams.set(SETTINGS_QUERY_KEYS.numHands, String(settings.numHands));
  url.searchParams.set(SETTINGS_QUERY_KEYS.minHandDetectionConfidence, String(settings.minHandDetectionConfidence));
  url.searchParams.set(SETTINGS_QUERY_KEYS.minHandPresenceConfidence, String(settings.minHandPresenceConfidence));
  url.searchParams.set(SETTINGS_QUERY_KEYS.minTrackingConfidence, String(settings.minTrackingConfidence));
  url.searchParams.set(RUNTIME_QUERY_KEYS.bundle, runtimeOption.key);
  url.searchParams.set(RUNTIME_QUERY_KEYS.root, runtimeOption.runtimeRoot);
  return url;
}

function writeUrlWithSelections(settings, runtimeKey) {
  const url = buildUrlWithSelections(settings, runtimeKey);
  window.history.replaceState({}, "", url);
  return url;
}

function setFormSettings(settings) {
  const normalized = normalizeSettings(settings);
  refs.inferWidthInput.value = String(normalized.inferWidth);
  refs.inferFpsInput.value = String(normalized.inferFps);
  refs.modelIntervalInput.value = String(normalized.modelIntervalMs);
  refs.tauInput.value = normalized.tau.toFixed(2);
  refs.gestureVoteNInput.value = String(normalized.gestureVoteN);
  refs.gestureDebounceKInput.value = String(normalized.gestureDebounceK);
  refs.gestureClearFramesInput.value = String(normalized.gestureClearFrames);
  refs.grabMotionWindowInput.value = String(normalized.grabMotionWindow);
  refs.grabMotionThresholdInput.value = normalized.grabMotionThreshold.toFixed(3);
  refs.numHandsSelect.value = String(normalized.numHands);
  refs.mpDetConfInput.value = normalized.minHandDetectionConfidence.toFixed(2);
  refs.mpPresenceConfInput.value = normalized.minHandPresenceConfidence.toFixed(2);
  refs.mpTrackConfInput.value = normalized.minTrackingConfidence.toFixed(2);
  refs.settingsPreset.value = getPresetKeyForSettings(normalized);
  state.settings.form = cloneSettings(normalized);
}

function setFormRuntimeKey(runtimeKey) {
  refs.runtimeBundleSelect.value = (getRuntimeOptionByKey(runtimeKey) || getDefaultRuntimeOption()).key;
  state.runtime.formKey = refs.runtimeBundleSelect.value;
}

function readFormSettings() {
  return normalizeSettings({
    inferWidth: refs.inferWidthInput.value,
    inferFps: refs.inferFpsInput.value,
    modelIntervalMs: refs.modelIntervalInput.value,
    tau: refs.tauInput.value,
    gestureVoteN: refs.gestureVoteNInput.value,
    gestureDebounceK: refs.gestureDebounceKInput.value,
    gestureClearFrames: refs.gestureClearFramesInput.value,
    grabMotionWindow: refs.grabMotionWindowInput.value,
    grabMotionThreshold: refs.grabMotionThresholdInput.value,
    numHands: refs.numHandsSelect.value,
    minHandDetectionConfidence: refs.mpDetConfInput.value,
    minHandPresenceConfidence: refs.mpPresenceConfInput.value,
    minTrackingConfidence: refs.mpTrackConfInput.value
  });
}

function refreshSettingsPanel() {
  const activeSettings = getActiveSettings();
  const formSettings = readFormSettings();
  const activeRuntime = getActiveRuntimeOption();
  const formRuntime = getFormRuntimeOption();
  state.settings.form = cloneSettings(formSettings);
  state.runtime.formKey = formRuntime.key;
  refs.settingsPreset.value = getPresetKeyForSettings(formSettings);
  refs.settingsSummary.textContent = `${activeRuntime.label} · ${getSettingsSummary(activeSettings)}`;

  const hasPendingChanges = !settingsEqual(activeSettings, formSettings) || activeRuntime.key !== formRuntime.key;
  refs.settingsStateBadge.textContent = hasPendingChanges ? "Pending changes" : "Applied";
  refs.settingsStateBadge.classList.toggle("pending", hasPendingChanges);
  refs.settingsStateBadge.classList.toggle("applied", !hasPendingChanges);
  refs.settingsHint.textContent = hasPendingChanges
    ? "입력값 또는 체크포인트가 현재 활성 상태와 다릅니다. Apply & Restart를 누르면 새 설정으로 다시 시작합니다."
    : "현재 활성 설정과 폼 값이 동일합니다.";
}

const state = {
  running: false,
  detectInFlight: false,
  rafId: 0,
  handLandmarker: null,
  stream: null,
  classNames: [...DEFAULT_CLASS_NAMES],
  runtimeConfig: null,
  lastDetectAt: 0,
  latencySamples: [],
  seenLatencyTokens: new Set(),
  settings: {
    active: null,
    form: null
  },
  runtime: {
    activeKey: null,
    formKey: null
  },
  hands: new Map(
    HAND_KEYS.map((handKey) => [
      handKey,
      {
        landmarks: null,
        lastSeenAt: 0,
        rawModel: handKey === "left" ? createDisabledPrediction() : createNoHandPrediction(),
        finalGesture: handKey === "left" ? createDisabledGesture() : createNoHandGesture()
      }
    ])
  )
};

const initialSettings = loadSettingsFromUrl();
const initialRuntimeKey = loadRuntimeSelectionFromUrl();
state.settings.active = cloneSettings(initialSettings);
state.settings.form = cloneSettings(initialSettings);
state.runtime.activeKey = initialRuntimeKey;
state.runtime.formKey = initialRuntimeKey;
setFormSettings(initialSettings);
setFormRuntimeKey(initialRuntimeKey);

function applyTauOverride(prediction) {
  if (!prediction || typeof prediction !== "object") return prediction;
  if (prediction.disabled) return prediction;

  const tau = getActiveSettings().tau;
  const rawPredIndex = Number.isFinite(prediction.raw_pred_index)
    ? prediction.raw_pred_index
    : (Number.isFinite(prediction.classId) ? prediction.classId : 0);
  const confidence = Number.isFinite(prediction.confidence) ? prediction.confidence : 0;
  const shouldNeutralize = rawPredIndex !== 0 && confidence < tau;
  const finalIndex = shouldNeutralize ? 0 : rawPredIndex;

  return {
    ...prediction,
    label: state.classNames[finalIndex] || prediction.label || "None",
    classId: finalIndex,
    tau_applied: tau,
    tau_neutralized: shouldNeutralize,
    raw_pred_index: rawPredIndex,
    raw_pred_label: state.classNames[rawPredIndex] || prediction.raw_pred_label || "None"
  };
}

function getMonitorModelPrediction(landmarks, now = performance.now(), handKey = "default") {
  return applyTauOverride(getSequenceModelPrediction(landmarks, now, handKey));
}

setModelPredictionProvider(getMonitorModelPrediction);

function createNeutralProbs(classCount = DEFAULT_CLASS_NAMES.length) {
  return new Array(classCount).fill(0).map((_, index) => (index === 0 ? 1 : 0));
}

function normalizeRootPath(value, fallback = "/") {
  if (typeof value !== "string") return fallback;
  const trimmed = value.trim();
  if (!trimmed) return fallback;
  if (/^https?:\/\//i.test(trimmed)) {
    return trimmed.replace(/\/+$/, "") + "/";
  }
  return ("/" + trimmed.replace(/^\/+/, "").replace(/\/+$/, "") + "/").replace(/\/+/g, "/");
}

function resolveRuntimeRoot() {
  const params = new URLSearchParams(window.location.search);
  const queryRoot = params.get("runtimeRoot");
  if (queryRoot && queryRoot.trim()) return normalizeRootPath(queryRoot, "/runtime_sequence/");
  return "/runtime_sequence/";
}

function formatMs(value) {
  return Number.isFinite(value) ? `${value.toFixed(1)}ms` : "-";
}

function formatMotionScore(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

function clamp(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function percentile(values, ratio) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(sorted.length * ratio) - 1));
  return sorted[idx];
}

function summarizeLatency(values) {
  if (!values.length) {
    return { avg: null, p50: null, p95: null };
  }
  const avg = values.reduce((sum, value) => sum + value, 0) / values.length;
  return {
    avg,
    p50: percentile(values, 0.5),
    p95: percentile(values, 0.95)
  };
}

function normalizeModelLabel(rawLabel) {
  const label = String(rawLabel || "").trim().toLowerCase();
  if (label === "none" || label === "class0" || label === "neutral") return "None";
  if (label === "fist" || label === "class1") return "Fist";
  if (label === "open palm" || label === "open_palm" || label === "openpalm" || label === "paper" || label === "class2") return "OpenPalm";
  if (label === "v" || label === "class3") return "V";
  if (label === "pinky" || label === "class4" || label === "4") return "Pinky";
  if (label === "animal" || label === "class5" || label === "5") return "Animal";
  if (label === "k-heart" || label === "kheart" || label === "class6" || label === "6") return "KHeart";
  if (label === "grab" || label === "class7" || label === "class 7" || label === "7") return "Grab";
  return rawLabel || "None";
}

function formatDisplayGesture(label, confidence = null, classId = null) {
  const normalized = String(label || "").trim().toLowerCase();
  let displayLabel = "아무것도 아님";
  if (normalized === "fist" || classId === 1) displayLabel = "주먹";
  else if (normalized === "openpalm" || normalized === "open_palm" || normalized === "open palm" || classId === 2) displayLabel = "손바닥";
  else if (normalized === "v" || classId === 3) displayLabel = "브이";
  else if (normalized === "pinky" || classId === 4) displayLabel = "새끼손가락";
  else if (normalized === "animal" || classId === 5) displayLabel = "애니멀";
  else if (normalized === "kheart" || normalized === "k-heart" || classId === 6) displayLabel = "K-하트";
  else if (normalized === "grab" || classId === 7) displayLabel = "그랩";
  else if (normalized && normalized !== "none" && normalized !== "class0") displayLabel = label;

  if (!Number.isFinite(confidence)) return displayLabel;
  return `${displayLabel} ${(confidence * 100).toFixed(0)}%`;
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

function getTestModeHandColor(handKey = "default") {
  return TEST_MODE_HAND_COLORS[handKey] || {
    stroke: "rgba(255, 255, 255, 0.92)",
    fill: "rgba(245, 245, 245, 0.92)"
  };
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

function snapshotLandmarks(landmarks) {
  return Array.isArray(landmarks)
    ? landmarks.map((point) => ({
      x: Number.isFinite(point?.x) ? point.x : 0,
      y: Number.isFinite(point?.y) ? point.y : 0,
      z: Number.isFinite(point?.z) ? point.z : 0
    }))
    : null;
}

function clonePrediction(prediction) {
  if (!prediction || typeof prediction !== "object") return prediction;
  return {
    ...prediction,
    probs: Array.isArray(prediction.probs) ? [...prediction.probs] : prediction.probs
  };
}

function createDisabledPrediction() {
  return {
    label: "None",
    confidence: 0,
    classId: 0,
    source: "disabled",
    disabled: true,
    status: "disabled",
    probs: null,
    framesCollected: 0,
    elapsed_ms: null
  };
}

function createNoHandPrediction(classCount = DEFAULT_CLASS_NAMES.length) {
  return {
    label: "neutral",
    confidence: 0,
    classId: 0,
    source: "onnx-sequence",
    status: "no_hand",
    probs: createNeutralProbs(classCount),
    framesCollected: 0,
    elapsed_ms: null,
    tau_neutralized: false
  };
}

function createDisabledGesture() {
  return {
    label: "None",
    confidence: 0,
    source: "disabled",
    disabled: true,
    motionScore: 0,
    motionGateBlocked: false,
    votedLabel: "None",
    debouncedLabel: "None"
  };
}

function createNoHandGesture() {
  return {
    label: "None",
    confidence: 0,
    source: "no_hand",
    motionScore: 0,
    motionGateBlocked: false,
    votedLabel: "None",
    debouncedLabel: "None"
  };
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

function resizeInferenceCanvas(sourceWidth, sourceHeight, maxWidth = getActiveSettings().inferWidth) {
  const scale = maxWidth === 0 || sourceWidth <= maxWidth ? 1 : (maxWidth / sourceWidth);
  const width = Math.max(1, Math.round(sourceWidth * scale));
  const height = Math.max(1, Math.round(sourceHeight * scale));
  if (inferenceCanvas.width !== width || inferenceCanvas.height !== height) {
    inferenceCanvas.width = width;
    inferenceCanvas.height = height;
  }
}

function drawVideoToInferenceCanvas() {
  const { webcam } = refs;
  if (!webcam.videoWidth || !webcam.videoHeight) return false;
  resizeInferenceCanvas(webcam.videoWidth, webcam.videoHeight, getActiveSettings().inferWidth);
  inferenceCtx.clearRect(0, 0, inferenceCanvas.width, inferenceCanvas.height);
  inferenceCtx.drawImage(webcam, 0, 0, inferenceCanvas.width, inferenceCanvas.height);
  return true;
}

function ensureCanvasMatchesVideo(canvas) {
  const frame = canvas.parentElement?.getBoundingClientRect();
  if (!frame) return;
  const width = Math.max(1, Math.round(frame.width));
  const height = Math.max(1, Math.round(frame.height));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
}

function updateStatus(pill, text, subtext = null) {
  refs.statusPill.textContent = pill;
  refs.statusText.textContent = text;
  if (subtext != null) {
    refs.statusSubtext.textContent = subtext;
  }
}

function updateRuntimeMeta(config) {
  refs.metaBundleId.textContent = config?.bundle_id || "-";
  refs.metaModelId.textContent = config?.model_id || "-";
  refs.metaNormalization.textContent = config?.normalization_family || "-";
  refs.metaSeqLen.textContent = Number.isFinite(config?.seq_len) ? String(config.seq_len) : "-";
  const activeTau = getActiveSettings().tau;
  if (config?.default_tau != null) {
    refs.metaTau.textContent = Math.abs(activeTau - Number(config.default_tau)) < 1e-6
      ? activeTau.toFixed(2)
      : `${activeTau.toFixed(2)} (override; runtime ${Number(config.default_tau).toFixed(2)})`;
  } else {
    refs.metaTau.textContent = activeTau.toFixed(2);
  }
  refs.metaRuntimeRoot.textContent = resolveRuntimeRoot();
  refs.metaFingerprint.textContent = config?.checkpoint_fingerprint || "-";
  refs.metaGamma.textContent = Number.isFinite(config?.focal_gamma)
    ? String(config.focal_gamma)
    : "Not recorded in runtime asset";
}

function getSequenceLength() {
  return Number.isFinite(state.runtimeConfig?.seq_len) ? state.runtimeConfig.seq_len : 8;
}

function recordLatency(rawModel, handKey) {
  if (handKey !== "right") return;
  if (!rawModel || !Number.isFinite(rawModel.elapsed_ms) || !Number.isFinite(rawModel.completed_at_ms)) return;
  const token = `${handKey}:${rawModel.completed_at_ms}`;
  if (state.seenLatencyTokens.has(token)) return;
  state.seenLatencyTokens.add(token);
  state.latencySamples.push(rawModel.elapsed_ms);
  if (state.latencySamples.length > 120) {
    state.latencySamples.splice(0, state.latencySamples.length - 120);
  }
  if (state.seenLatencyTokens.size > 480) {
    state.seenLatencyTokens.clear();
    state.seenLatencyTokens.add(token);
  }
}

function getTopK(probs, classNames, count = 3) {
  if (!Array.isArray(probs)) return [];
  return probs
    .map((value, index) => ({
      index,
      label: classNames[index] || `class${index}`,
      value
    }))
    .sort((a, b) => b.value - a.value)
    .slice(0, count);
}

function pickPreviewHand() {
  const right = state.hands.get("right");
  if (right?.landmarks) return { handKey: "right", landmarks: right.landmarks };
  const left = state.hands.get("left");
  if (left?.landmarks) return { handKey: "left", landmarks: left.landmarks };
  return null;
}

function renderRawOverlay() {
  ensureCanvasMatchesVideo(refs.rawCanvas);
  rawCtx.clearRect(0, 0, refs.rawCanvas.width, refs.rawCanvas.height);

  const activeHands = HAND_KEYS
    .map((handKey) => ({ handKey, snapshot: state.hands.get(handKey) }))
    .filter(({ snapshot }) => Array.isArray(snapshot?.landmarks));

  refs.rawCaption.textContent = activeHands.length
    ? `표시 손: ${activeHands.map(({ handKey }) => handKey).join(", ")}`
    : "표시 손: 없음";

  activeHands.forEach(({ handKey, snapshot }) => {
    const rawPoints = snapshot.landmarks.map((point) => ({
      x: (1 - point.x) * refs.rawCanvas.width,
      y: point.y * refs.rawCanvas.height
    }));
    drawHandGraph(rawCtx, rawPoints, handKey, { pointRadius: 3.4, lineWidth: 2 });
  });
}

function renderNormalizedPreview() {
  normalizedCtx.clearRect(0, 0, refs.normalizedCanvas.width, refs.normalizedCanvas.height);
  normalizedCtx.save();
  normalizedCtx.strokeStyle = "rgba(255, 255, 255, 0.14)";
  normalizedCtx.lineWidth = 1;
  normalizedCtx.setLineDash([4, 5]);
  normalizedCtx.beginPath();
  normalizedCtx.moveTo(refs.normalizedCanvas.width * 0.5, 14);
  normalizedCtx.lineTo(refs.normalizedCanvas.width * 0.5, refs.normalizedCanvas.height - 14);
  normalizedCtx.moveTo(14, refs.normalizedCanvas.height * 0.5);
  normalizedCtx.lineTo(refs.normalizedCanvas.width - 14, refs.normalizedCanvas.height * 0.5);
  normalizedCtx.stroke();
  normalizedCtx.restore();

  const preview = pickPreviewHand();
  if (!preview) {
    refs.normalizedCaption.textContent = "정규화 대상 손: 없음";
    return;
  }

  const sanitized = sanitizePreviewLandmarks(preview.landmarks, preview.handKey);
  const normalized = normalizeSequencePreviewFrame(sanitized);
  if (!normalized) {
    refs.normalizedCaption.textContent = `정규화 대상 손: ${preview.handKey}`;
    return;
  }

  let maxAbs = 0.25;
  for (let i = 0; i < 63; i += 3) {
    maxAbs = Math.max(maxAbs, Math.abs(normalized[i]), Math.abs(normalized[i + 1]));
  }
  const scale = (Math.min(refs.normalizedCanvas.width, refs.normalizedCanvas.height) * 0.34) / maxAbs;
  const normalizedPoints = [];
  for (let i = 0; i < 63; i += 3) {
    normalizedPoints.push({
      x: refs.normalizedCanvas.width * 0.5 + normalized[i] * scale,
      y: refs.normalizedCanvas.height * 0.5 + normalized[i + 1] * scale
    });
  }
  drawHandGraph(normalizedCtx, normalizedPoints, preview.handKey, { pointRadius: 3.8, lineWidth: 2.1 });
  refs.normalizedCaption.textContent = `정규화 대상 손: ${preview.handKey}`;
}

function renderLatency() {
  const right = state.hands.get("right");
  const current = right?.rawModel?.elapsed_ms ?? getModelInferenceStatus(performance.now()).lastDurationMs;
  const summary = summarizeLatency(state.latencySamples);
  refs.latencyCurrent.textContent = formatMs(current);
  refs.latencyAvg.textContent = formatMs(summary.avg);
  refs.latencyP50.textContent = formatMs(summary.p50);
  refs.latencyP95.textContent = formatMs(summary.p95);
}

function renderHandCard(handKey) {
  const target = handKey === "left" ? refs.handCardLeft : refs.handCardRight;
  const snapshot = state.hands.get(handKey);
  const raw = snapshot?.rawModel ?? (
    handKey === "left"
      ? createDisabledPrediction()
      : applyTauOverride(createNoHandPrediction(state.classNames.length))
  );
  const finalGesture = snapshot?.finalGesture ?? (handKey === "left" ? createDisabledGesture() : createNoHandGesture());
  const topK = getTopK(raw?.probs, state.classNames);
  const badgeStatus = String(raw?.status || finalGesture?.status || "idle");
  const finalLabel = formatDisplayGesture(finalGesture?.label, finalGesture?.confidence);
  const rawLabel = formatDisplayGesture(raw?.label, raw?.confidence, raw?.classId ?? null);
  const source = finalGesture?.source || raw?.source || "-";
  const framesCollected = Number.isFinite(raw?.framesCollected) ? `${raw.framesCollected} / ${getSequenceLength()}` : "-";
  const motionScore = Number.isFinite(finalGesture?.motionScore) ? finalGesture.motionScore : raw?.recent_motion_score;
  const votedLabel = formatDisplayGesture(finalGesture?.votedLabel ?? "None");
  const debouncedLabel = formatDisplayGesture(finalGesture?.debouncedLabel ?? "None");

  target.innerHTML = `
    <header>
      <h3>${handKey === "right" ? "오른손" : "왼손"}</h3>
      <span class="hand-badge ${badgeStatus}">${badgeStatus}</span>
    </header>
    <div class="hand-details">
      <div class="detail-row"><span class="label">Raw model</span><span class="value">${rawLabel}</span></div>
      <div class="detail-row"><span class="label">Final gesture</span><span class="value">${finalLabel}</span></div>
      <div class="detail-row"><span class="label">Source</span><span class="value">${source}</span></div>
      <div class="detail-row"><span class="label">Frames collected</span><span class="value">${framesCollected}</span></div>
      <div class="detail-row"><span class="label">Inference ms</span><span class="value">${formatMs(raw?.elapsed_ms)}</span></div>
      <div class="detail-row"><span class="label">Tau neutralized</span><span class="value">${raw?.tau_neutralized ? "Yes" : "No"}</span></div>
      <div class="detail-row"><span class="label">Motion score</span><span class="value">${formatMotionScore(motionScore)}</span></div>
      <div class="detail-row"><span class="label">Motion gate</span><span class="value">${finalGesture?.motionGateBlocked ? "Blocked" : "Pass"}</span></div>
      <div class="detail-row"><span class="label">Voted label</span><span class="value">${votedLabel}</span></div>
      <div class="detail-row"><span class="label">Debounced label</span><span class="value">${debouncedLabel}</span></div>
    </div>
    <div class="topk-list">
      ${topK.length
        ? topK.map((item, index) => `
          <div class="topk-item">
            <span class="topk-rank">#${index + 1}</span>
            <span class="topk-label">${normalizeModelLabel(item.label)}</span>
            <span class="topk-score">${(item.value * 100).toFixed(1)}%</span>
          </div>
        `).join("")
        : `<div class="topk-item"><span class="topk-rank">-</span><span class="topk-label">확률 정보 없음</span><span class="topk-score">-</span></div>`
      }
    </div>
  `;
}

function renderMonitor() {
  renderRawOverlay();
  renderNormalizedPreview();
  renderLatency();
  renderHandCard("left");
  renderHandCard("right");
  refreshSettingsPanel();

  const modelStatus = getModelInferenceStatus(performance.now());
  const activeHands = HAND_KEYS.filter((handKey) => Array.isArray(state.hands.get(handKey)?.landmarks));
  const right = state.hands.get("right");
  const activeSettings = getActiveSettings();
  const activeRuntime = getActiveRuntimeOption();
  const pill = state.running ? (modelStatus.endpointConfigured ? "Running" : "Loading") : "Idle";
  const summary = right?.rawModel?.status
    ? `right=${right.rawModel.status}${Number.isFinite(right.rawModel.framesCollected) ? `, frames=${right.rawModel.framesCollected}` : ""}`
    : "right=idle";
  updateStatus(
    pill,
    state.running
      ? `활성 손: ${activeHands.length ? activeHands.join(", ") : "없음"} / ${summary}`
      : "Start Monitor를 눌러 Windows 웹캠과 sequence 추론 모니터를 시작하세요.",
    modelStatus.endpointConfigured
      ? `${activeRuntime.label} · ${getSettingsSummary(activeSettings)} · mode=${modelStatus.mode}, lastInference=${formatMs(modelStatus.lastDurationMs)}, runtimeRoot=${resolveRuntimeRoot()}`
      : `${activeRuntime.label} · ${getSettingsSummary(activeSettings)} · seq_len=${getSequenceLength()}`
  );
}

function parsePreferredDelegate() {
  const params = new URLSearchParams(window.location.search);
  const raw = (params.get("mpDelegate") || "gpu").trim().toUpperCase();
  return raw === "CPU" ? "CPU" : "GPU";
}

async function initializeMediaPipe() {
  if (state.handLandmarker) return state.handLandmarker;
  const activeSettings = getActiveSettings();
  const vision = await FilesetResolver.forVisionTasks(getConfiguredMediaPipeWasmRoot());
  const modelAssetPath = getConfiguredHandLandmarkerTaskPath();
  const preferredDelegate = parsePreferredDelegate();
  const fallbackDelegate = preferredDelegate === "GPU" ? "CPU" : "GPU";

  try {
    state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: preferredDelegate
      },
      runningMode: "VIDEO",
      numHands: activeSettings.numHands,
      minHandDetectionConfidence: activeSettings.minHandDetectionConfidence,
      minHandPresenceConfidence: activeSettings.minHandPresenceConfidence,
      minTrackingConfidence: activeSettings.minTrackingConfidence
    });
  } catch (delegateError) {
    console.warn("[frontend-test] MediaPipe delegate fallback", delegateError);
    state.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath,
        delegate: fallbackDelegate
      },
      runningMode: "VIDEO",
      numHands: activeSettings.numHands,
      minHandDetectionConfidence: activeSettings.minHandDetectionConfidence,
      minHandPresenceConfidence: activeSettings.minHandPresenceConfidence,
      minTrackingConfidence: activeSettings.minTrackingConfidence
    });
  }

  return state.handLandmarker;
}

async function loadRuntimeMetadata() {
  const runtimeRoot = resolveRuntimeRoot();
  const [configResponse, classResponse] = await Promise.all([
    fetch(`${runtimeRoot}config.json`),
    fetch(`${runtimeRoot}class_names.json`)
  ]);
  if (!configResponse.ok) {
    throw new Error(`runtime config load failed: ${configResponse.status}`);
  }
  if (!classResponse.ok) {
    throw new Error(`class names load failed: ${classResponse.status}`);
  }
  state.runtimeConfig = await configResponse.json();
  state.classNames = await classResponse.json();
  updateRuntimeMeta(state.runtimeConfig);
}

function waitForLoadedMetadata(video) {
  return new Promise((resolve) => {
    if (video.readyState >= 1 && video.videoWidth > 0) {
      resolve();
      return;
    }
    video.onloadedmetadata = () => resolve();
  });
}

async function initializeCamera() {
  const attempts = [
    {
      width: { ideal: 640 },
      height: { ideal: 360 },
      frameRate: { ideal: 30, max: 30 }
    },
    {
      width: { ideal: 640 },
      height: { ideal: 360 },
      frameRate: { ideal: 24, max: 30 }
    },
    true
  ];

  let stream = null;
  let lastError = null;
  for (const constraints of attempts) {
    try {
      stream = await navigator.mediaDevices.getUserMedia({ video: constraints });
      break;
    } catch (error) {
      lastError = error;
      const errorName = String(error?.name || "Error");
      if (errorName === "NotAllowedError" || errorName === "SecurityError" || errorName === "NotFoundError") {
        break;
      }
    }
  }
  if (!stream) {
    throw lastError || new Error("Could not start webcam");
  }

  refs.webcam.srcObject = stream;
  refs.webcam.playsInline = true;
  refs.webcam.muted = true;
  refs.webcam.setAttribute("playsinline", "");
  await waitForLoadedMetadata(refs.webcam);
  await refs.webcam.play().catch(() => {});
  state.stream = stream;
}

function resetHandsToIdle() {
  HAND_KEYS.forEach((handKey) => {
    pushNoHand(handKey);
    resetGestureState(handKey);
    state.hands.set(handKey, {
      landmarks: null,
      lastSeenAt: 0,
      rawModel: handKey === "left" ? createDisabledPrediction() : createNoHandPrediction(state.classNames.length),
      finalGesture: handKey === "left" ? createDisabledGesture() : createNoHandGesture()
    });
  });
}

function updateMissingHands(now, seenKeys) {
  HAND_KEYS.forEach((handKey) => {
    if (seenKeys.has(handKey)) return;
    const snapshot = state.hands.get(handKey);
    if (snapshot?.lastSeenAt && now - snapshot.lastSeenAt <= LANDMARK_STALE_MS) return;
    const rawModel = clonePrediction(pushNoHand(handKey));
    resetGestureState(handKey);
    state.hands.set(handKey, {
      landmarks: null,
      lastSeenAt: 0,
      rawModel,
      finalGesture: handKey === "left" ? createDisabledGesture() : createNoHandGesture()
    });
  });
}

function releaseMediaPipe() {
  if (typeof state.handLandmarker?.close === "function") {
    try {
      state.handLandmarker.close();
    } catch (error) {
      console.warn("[frontend-test] failed to close MediaPipe instance", error);
    }
  }
  state.handLandmarker = null;
}

async function runDetection(now) {
  if (!state.running || state.detectInFlight || !state.handLandmarker) return;
  if (now - state.lastDetectAt < getInferIntervalMs(getActiveSettings())) return;
  if (refs.webcam.readyState < 2 || !refs.webcam.videoWidth) return;
  if (!drawVideoToInferenceCanvas()) return;

  state.detectInFlight = true;
  state.lastDetectAt = now;
  try {
    const detectTimestamp = Math.max(1, Math.round(now));
    const result = typeof state.handLandmarker.detectForVideo === "function"
      ? state.handLandmarker.detectForVideo(inferenceCanvas, detectTimestamp)
      : state.handLandmarker.detect(inferenceCanvas);
    const hands = buildHandsWithKeys(result);
    const seenKeys = new Set();

    hands.forEach(({ handKey, landmarks }) => {
      if (!HAND_KEYS.includes(handKey)) return;
      seenKeys.add(handKey);
      const rawModel = clonePrediction(getMonitorModelPrediction(landmarks, now, handKey));
      const finalGesture = resolveGesture(landmarks, now, true, handKey);
      state.hands.set(handKey, {
        landmarks: snapshotLandmarks(landmarks),
        lastSeenAt: now,
        rawModel,
        finalGesture: finalGesture ? { ...finalGesture } : (handKey === "left" ? createDisabledGesture() : createNoHandGesture())
      });
      recordLatency(rawModel, handKey);
    });

    updateMissingHands(now, seenKeys);
  } catch (error) {
    console.error("[frontend-test] detection failed", error);
    updateStatus("Error", String(error?.message || error || "Detection failed"));
  } finally {
    state.detectInFlight = false;
  }
}

function loop(now) {
  if (!state.running) return;
  runDetection(now);
  renderMonitor();
  state.rafId = requestAnimationFrame(loop);
}

async function startMonitor() {
  if (state.running) return;
  try {
    updateStatus("Booting", "모델과 MediaPipe를 초기화하는 중입니다...");
    await loadRuntimeMetadata();
    await preloadModel();
    await initializeMediaPipe();
    updateStatus("Camera", "Windows 웹캠을 여는 중입니다...");
    await initializeCamera();
    state.detectInFlight = false;
    state.lastDetectAt = 0;
    state.latencySamples = [];
    state.seenLatencyTokens.clear();
    resetHandsToIdle();
    state.running = true;
    renderMonitor();
    state.rafId = requestAnimationFrame(loop);
  } catch (error) {
    console.error("[frontend-test] start failed", error);
    const message = String(error?.message || error || "");
    if (message.includes("NotAllowedError") || message.includes("SecurityError")) {
      updateStatus("Permission", "카메라 권한을 허용해 주세요.");
    } else if (message.includes("NotFoundError")) {
      updateStatus("No Camera", "사용 가능한 카메라를 찾지 못했습니다.");
    } else {
      updateStatus("Error", `시작 실패: ${message || "알 수 없는 오류"}`);
    }
  }
}

function stopMonitor({ releaseHandLandmarker = false, preserveStatus = false } = {}) {
  state.running = false;
  state.detectInFlight = false;
  state.lastDetectAt = 0;
  if (state.rafId) {
    cancelAnimationFrame(state.rafId);
    state.rafId = 0;
  }
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
    state.stream = null;
  }
  refs.webcam.pause();
  refs.webcam.srcObject = null;
  if (releaseHandLandmarker) {
    releaseMediaPipe();
  }
  resetHandsToIdle();
  renderMonitor();
  if (!preserveStatus) {
    updateStatus("Stopped", "모니터를 중지했습니다. 다시 시작하려면 Start Monitor를 누르세요.");
  }
}

async function applySettingsAndMaybeRestart() {
  const nextSettings = readFormSettings();
  const nextRuntime = getFormRuntimeOption();
  const currentRuntime = getActiveRuntimeOption();
  const runtimeChanged = currentRuntime.key !== nextRuntime.key;
  const wasRunning = state.running;
  state.settings.active = cloneSettings(nextSettings);
  state.settings.form = cloneSettings(nextSettings);
  state.runtime.activeKey = nextRuntime.key;
  state.runtime.formKey = nextRuntime.key;
  const nextUrl = writeUrlWithSelections(nextSettings, nextRuntime.key);
  refreshSettingsPanel();

  if (runtimeChanged) {
    if (wasRunning) {
      stopMonitor({ releaseHandLandmarker: true, preserveStatus: true });
    } else {
      releaseMediaPipe();
      renderMonitor();
    }
    updateStatus(
      "Reloading",
      `${nextRuntime.label} 체크포인트로 전환하는 중입니다...`,
      `${nextRuntime.label} · ${getSettingsSummary(nextSettings)} · runtimeRoot=${nextRuntime.runtimeRoot}`
    );
    window.location.assign(nextUrl.toString());
    return;
  }

  if (wasRunning) {
    stopMonitor({ releaseHandLandmarker: true, preserveStatus: true });
    await startMonitor();
    return;
  }

  releaseMediaPipe();
  renderMonitor();
  updateStatus(
    "Ready",
    "설정이 적용되었습니다. Start Monitor를 눌러 새 설정으로 시작하세요.",
    `${nextRuntime.label} · ${getSettingsSummary(nextSettings)} · seq_len=${getSequenceLength()}`
  );
}

refs.startButton.addEventListener("click", () => {
  startMonitor().catch((error) => {
    console.error(error);
  });
});

refs.stopButton.addEventListener("click", stopMonitor);
refs.settingsPreset.addEventListener("change", (event) => {
  const presetKey = event.target.value;
  if (presetKey === "frontend-default") {
    setFormSettings(DEFAULT_SETTINGS);
  } else if (presetKey === "viewer-like") {
    setFormSettings(VIEWER_LIKE_SETTINGS);
  }
  refreshSettingsPanel();
});

refs.runtimeBundleSelect.addEventListener("change", refreshSettingsPanel);

[
  refs.inferWidthInput,
  refs.inferFpsInput,
  refs.modelIntervalInput,
  refs.tauInput,
  refs.gestureVoteNInput,
  refs.gestureDebounceKInput,
  refs.gestureClearFramesInput,
  refs.grabMotionWindowInput,
  refs.grabMotionThresholdInput,
  refs.numHandsSelect,
  refs.mpDetConfInput,
  refs.mpPresenceConfInput,
  refs.mpTrackConfInput
].forEach((element) => {
  element.addEventListener("input", refreshSettingsPanel);
  element.addEventListener("change", refreshSettingsPanel);
});

refs.applySettingsButton.addEventListener("click", () => {
  applySettingsAndMaybeRestart().catch((error) => {
    console.error("[frontend-test] apply settings failed", error);
    updateStatus("Error", `설정 적용 실패: ${String(error?.message || error || "알 수 없는 오류")}`);
  });
});

refs.resetSettingsButton.addEventListener("click", () => {
  setFormSettings(DEFAULT_SETTINGS);
  refreshSettingsPanel();
});

updateRuntimeMeta(null);
renderMonitor();
loadRuntimeMetadata()
  .then(() => {
    renderMonitor();
  })
  .catch((error) => {
    console.warn("[frontend-test] runtime metadata preload failed", error);
  });
