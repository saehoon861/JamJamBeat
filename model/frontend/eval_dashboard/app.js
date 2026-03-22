const state = {
  catalog: null,
  index: null,
  modelCache: new Map(),
  selectedSuiteName: null,
  selectedModelId: null,
  selectedSource: "all",
  filters: {
    user: "all",
    motion: "all",
    side: "all",
    search: "",
  },
  activeEventIndex: 0,
  currentFrameIndex: 0,
  frameSyncReady: false,
  videoLoadToken: 0,
  pendingSeekFrame: null,
  landmarkCache: new Map(),
  landmarkPromiseCache: new Map(),
  landmarkLoadToken: 0,
  fullInferenceCache: new Map(),
  fullInferencePromiseCache: new Map(),
  fullInferenceLoadToken: 0,
  videoFrameCallbackId: null,
  animationFrameId: null,
  isScrubbing: false,
  scrubWasPlaying: false,
};

const els = {
  modelList: document.querySelector("#model-list"),
  suiteNameChip: document.querySelector("#suite-name-chip"),
  suiteSelector: document.querySelector("#suite-selector"),
  heroTitle: document.querySelector("#hero-title"),
  heroCopy: document.querySelector("#hero-copy"),
  bestMacroF1: document.querySelector("#best-macro-f1"),
  recommendedModel: document.querySelector("#recommended-model"),
  availableUsers: document.querySelector("#available-users"),
  availableSources: document.querySelector("#available-sources"),
  leaderboardBody: document.querySelector("#leaderboard-body"),
  selectedModelTitle: document.querySelector("#selected-model-title"),
  selectedModelTags: document.querySelector("#selected-model-tags"),
  selectedMacroF1: document.querySelector("#selected-macro-f1"),
  selectedAccuracy: document.querySelector("#selected-accuracy"),
  selectedFpPerMin: document.querySelector("#selected-fp-per-min"),
  selectedLatency: document.querySelector("#selected-latency"),
  runSnapshot: document.querySelector("#run-snapshot"),
  modelExplainer: document.querySelector("#model-explainer"),
  userFilter: document.querySelector("#user-filter"),
  motionFilter: document.querySelector("#motion-filter"),
  sideFilter: document.querySelector("#side-filter"),
  sourceFilter: document.querySelector("#source-filter"),
  sourceContext: document.querySelector("#source-context"),
  sourceExplorer: document.querySelector("#source-explorer"),
  frameAnalysisShell: document.querySelector("#frame-analysis-shell"),
  frameOverlayTop: document.querySelector("#frame-overlay-top"),
  frameOverlayBottom: document.querySelector("#frame-overlay-bottom"),
  frameProbabilityOverlay: document.querySelector("#frame-probability-overlay"),
  videoPlayer: document.querySelector("#video-player"),
  landmarkOverlay: document.querySelector("#landmark-overlay"),
  frameJumpBack: document.querySelector("#frame-jump-back"),
  framePrev: document.querySelector("#frame-prev"),
  framePrevEvent: document.querySelector("#frame-prev-event"),
  framePlayPause: document.querySelector("#frame-play-pause"),
  frameNextEvent: document.querySelector("#frame-next-event"),
  frameNext: document.querySelector("#frame-next"),
  frameJumpForward: document.querySelector("#frame-jump-forward"),
  frameRestart: document.querySelector("#frame-restart"),
  frameScrubber: document.querySelector("#frame-scrubber"),
  frameEventRail: document.querySelector("#frame-event-rail"),
  frameScrubberMeta: document.querySelector("#frame-scrubber-meta"),
  videoStats: document.querySelector("#video-stats"),
  landmarkStatus: document.querySelector("#landmark-status"),
  landmarkPreviewCanvas: document.querySelector("#landmark-preview-canvas"),
  currentPrediction: document.querySelector("#current-prediction"),
  timelineList: document.querySelector("#timeline-list"),
  timelineCount: document.querySelector("#timeline-count"),
  perClassBars: document.querySelector("#per-class-bars"),
  trainingCurve: document.querySelector("#training-curve"),
  confusionMatrix: document.querySelector("#confusion-matrix"),
  artifactLinks: document.querySelector("#artifact-links"),
  reviewerName: document.querySelector("#reviewer-name"),
  reviewNotes: document.querySelector("#review-notes"),
  checkMetric: document.querySelector("#check-metric"),
  checkVideo: document.querySelector("#check-video"),
  checkUser: document.querySelector("#check-user"),
  checkArtifacts: document.querySelector("#check-artifacts"),
  checkSummary: document.querySelector("#check-summary"),
  reviewStatus: document.querySelector("#review-status"),
  exportReview: document.querySelector("#export-review"),
  importReview: document.querySelector("#import-review"),
  importReviewFile: document.querySelector("#import-review-file"),
  reloadData: document.querySelector("#reload-data"),
  saveReview: document.querySelector("#save-review"),
  clearReview: document.querySelector("#clear-review"),
  modelSearch: document.querySelector("#model-search"),
};

const reviewCheckboxes = [
  els.checkMetric,
  els.checkVideo,
  els.checkUser,
  els.checkArtifacts,
  els.checkSummary,
];

const HAND_CONNECTIONS = [
  [0, 1], [0, 5], [5, 9], [9, 13], [13, 17], [0, 17],
  [1, 2], [2, 3], [3, 4],
  [5, 6], [6, 7], [7, 8],
  [9, 10], [10, 11], [11, 12],
  [13, 14], [14, 15], [15, 16],
  [17, 18], [18, 19], [19, 20],
];
const HUMAN_USER_PATTERN = /^(man\d+|woman\d+)$/i;

function currentSuiteEntry() {
  return state.catalog?.suites?.find((suite) => suite.suite_name === state.selectedSuiteName) || null;
}

function currentModelData() {
  return state.selectedModelId ? state.modelCache.get(state.selectedModelId) || null : null;
}

function suiteOptionLabel(suite) {
  const datasetTag = suite.dataset_tag || suite.suite_name;
  return `${datasetTag} · ${suite.suite_name}`;
}

async function loadCatalog() {
  const response = await fetch(`./data/suite-catalog.json?ts=${Date.now()}`);
  if (!response.ok) {
    throw new Error(`Failed to load dashboard catalog: ${response.status}`);
  }
  state.catalog = await response.json();

  const storedSuiteName = localStorage.getItem("jamjambeat-selected-suite");
  const availableSuites = state.catalog.suites || [];
  const initialSuite =
    availableSuites.find((suite) => suite.suite_name === storedSuiteName) ||
    availableSuites.find((suite) => suite.suite_name === state.catalog.default_suite_name) ||
    availableSuites[0];

  if (!initialSuite) {
    throw new Error("No local evaluation datasets were found for the dashboard.");
  }

  renderSuiteSelector();
  await loadSuiteIndex(initialSuite.suite_name);
}

async function loadSuiteIndex(suiteName) {
  const suiteEntry = state.catalog?.suites?.find((suite) => suite.suite_name === suiteName) || state.catalog?.suites?.[0];
  if (!suiteEntry) {
    throw new Error("Selected evaluation dataset could not be found.");
  }

  state.selectedSuiteName = suiteEntry.suite_name;
  localStorage.setItem("jamjambeat-selected-suite", state.selectedSuiteName);
  cancelPlaybackFrameLoop();
  state.modelCache.clear();
  state.landmarkCache.clear();
  state.landmarkPromiseCache.clear();
  state.fullInferenceCache.clear();
  state.fullInferencePromiseCache.clear();
  state.selectedSource = "all";
  state.activeEventIndex = 0;
  state.currentFrameIndex = 0;
  state.frameSyncReady = false;
  state.pendingSeekFrame = null;
  state.filters.user = "all";
  state.filters.motion = "all";
  state.filters.side = "all";

  const response = await fetch(`./${suiteEntry.index_path}?ts=${Date.now()}`);
  if (!response.ok) {
    throw new Error(`Failed to load suite index: ${response.status}`);
  }
  state.index = await response.json();
  state.selectedModelId = state.index.recommended_model_id || state.index.models?.[0]?.model_id || null;
  renderSuiteSelector();
  renderIndex();
  await selectModel(state.selectedModelId);
}

function renderSuiteSelector() {
  const suites = state.catalog?.suites || [];
  if (!suites.length) {
    els.suiteSelector.innerHTML = `<option>평가셋 없음</option>`;
    els.suiteSelector.disabled = true;
    return;
  }

  els.suiteSelector.disabled = false;
  els.suiteSelector.innerHTML = suites
    .map(
      (suite) => `
        <option value="${escapeHtml(suite.suite_name)}" ${suite.suite_name === state.selectedSuiteName ? "selected" : ""}>
          ${escapeHtml(suiteOptionLabel(suite))}
        </option>
      `,
    )
    .join("");
}

async function getModelData(modelId) {
  if (state.modelCache.has(modelId)) {
    return state.modelCache.get(modelId);
  }
  const summary = state.index.models.find((model) => model.model_id === modelId);
  const response = await fetch(`./${summary.detail_path}?ts=${Date.now()}`);
  if (!response.ok) {
    throw new Error(`Failed to load model details: ${response.status}`);
  }
  const payload = await response.json();
  state.modelCache.set(modelId, payload);
  return payload;
}

function formatNumber(value, digits = 4) {
  if (value === null || value === undefined || value === "") return "-";
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return String(value);
  return numeric.toFixed(digits);
}

function formatPercent(value) {
  if (value === null || value === undefined || value === "") return "-";
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return String(value);
  return `${(numeric * 100).toFixed(1)}%`;
}

function formatMs(value) {
  if (value === null || value === undefined || value === "") return "-";
  const numeric = Number(value);
  if (Number.isNaN(numeric)) return String(value);
  return `${numeric.toFixed(2)} ms`;
}

function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function parseTimestampToSeconds(timestamp) {
  if (!timestamp || typeof timestamp !== "string") return 0;
  const parts = timestamp.split(":").map((part) => Number(part));
  if (parts.length !== 3 || parts.some((part) => Number.isNaN(part))) return 0;
  const [minutes, seconds, millis] = parts;
  return minutes * 60 + seconds + millis / 1000;
}

function formatPlaybackTime(seconds) {
  const safe = Math.max(0, Number(seconds) || 0);
  const minutes = Math.floor(safe / 60);
  const remainder = safe - minutes * 60;
  const wholeSeconds = Math.floor(remainder);
  const millis = Math.round((remainder - wholeSeconds) * 1000);
  return `${String(minutes).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}:${String(millis).padStart(3, "0")}`;
}

function activeVideoMeta(model) {
  return selectedVideo(model) || null;
}

function allVideos(model) {
  return model?.videos || [];
}

function fullInferenceKey(video) {
  if (!video || !state.selectedSuiteName || !state.selectedModelId) return null;
  return [state.selectedSuiteName, state.selectedModelId, video.source_file].join(":");
}

function activeInferenceData(video) {
  const key = fullInferenceKey(video);
  if (!key) return null;
  return state.fullInferenceCache.get(key) || null;
}

function comparisonEvents(video) {
  return activeInferenceData(video)?.events || video?.events || [];
}

function currentVideoFps(model) {
  return Number(activeVideoMeta(model)?.fps || 30);
}

function currentVideoTotalFrames(model) {
  return Number(activeVideoMeta(model)?.total_frames || activeVideoMeta(model)?.frame_count || 0);
}

function frameDuration(model) {
  return 1 / Math.max(currentVideoFps(model), 1);
}

function focusFrameAnalysisShell() {
  requestAnimationFrame(() => {
    els.frameAnalysisShell.focus({ preventScroll: true });
  });
}

function shouldIgnoreFrameShortcut(event) {
  if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.altKey) {
    return true;
  }

  const target = event.target;
  if (!(target instanceof HTMLElement)) {
    return false;
  }

  if (target.isContentEditable) {
    return true;
  }

  const formField = target.closest("input, textarea, select");
  if (!formField) {
    return false;
  }

  return !(formField instanceof HTMLInputElement && formField.type === "range");
}

function matchesShortcut(event, code, key) {
  return event.code === code || event.key.toLowerCase() === key;
}

function handleFrameShortcutKeydown(event) {
  if (shouldIgnoreFrameShortcut(event)) return;
  const model = currentModelData();
  if (!model) return;

  if (event.code === "Space" || event.key === " ") {
    event.preventDefault();
    if (els.videoPlayer.paused) {
      void els.videoPlayer.play().catch(() => {});
    } else {
      els.videoPlayer.pause();
    }
    focusFrameAnalysisShell();
    return;
  }

  if (matchesShortcut(event, "KeyR", "r")) {
    event.preventDefault();
    els.videoPlayer.pause();
    seekToFrame(model, 0);
    focusFrameAnalysisShell();
    return;
  }

  if (matchesShortcut(event, "KeyA", "a") || event.key === "ArrowLeft") {
    event.preventDefault();
    stepFrame(model, event.shiftKey ? -10 : -1);
    focusFrameAnalysisShell();
    return;
  }

  if (matchesShortcut(event, "KeyD", "d") || event.key === "ArrowRight") {
    event.preventDefault();
    stepFrame(model, event.shiftKey ? 10 : 1);
    focusFrameAnalysisShell();
    return;
  }

  if (matchesShortcut(event, "KeyW", "w") || event.key === "ArrowUp") {
    event.preventDefault();
    jumpRelativeToCurrentEvent(model, -1);
    focusFrameAnalysisShell();
    return;
  }

  if (matchesShortcut(event, "KeyS", "s") || event.key === "ArrowDown") {
    event.preventDefault();
    jumpRelativeToCurrentEvent(model, 1);
    focusFrameAnalysisShell();
  }
}

function beginScrubInteraction() {
  if (state.isScrubbing) return;
  state.isScrubbing = true;
  state.scrubWasPlaying = !els.videoPlayer.paused && !els.videoPlayer.ended;
  els.videoPlayer.pause();
}

function finishScrubInteraction() {
  if (!state.isScrubbing) return;
  const shouldResume = state.scrubWasPlaying;
  state.isScrubbing = false;
  state.scrubWasPlaying = false;
  focusFrameAnalysisShell();
  if (shouldResume) {
    void els.videoPlayer.play().catch(() => {});
  }
}

function handleFrameScrubberInput() {
  const model = currentModelData();
  if (!model) return;
  if (!state.isScrubbing) {
    beginScrubInteraction();
  }
  seekToFrame(model, Number(els.frameScrubber.value || 0));
}

function currentPlayerFrame(model) {
  const video = selectedVideo(model);
  if (!video) return 0;
  const totalFrames = Math.max(currentVideoTotalFrames(model), video.frame_count || 1);
  return Math.max(0, Math.min(Math.round((els.videoPlayer.currentTime || 0) * currentVideoFps(model)), totalFrames - 1));
}

function landmarkCsvPath(video) {
  return video?.landmark_path || `../../../data/landmark_data/${video?.source_file || "unknown"}.csv`;
}

function clearCanvas(canvas) {
  const context = canvas.getContext("2d");
  context.clearRect(0, 0, canvas.width, canvas.height);
}

function resizeCanvasToBox(canvas, width, height) {
  const dpr = window.devicePixelRatio || 1;
  const safeWidth = Math.max(1, Math.round(width));
  const safeHeight = Math.max(1, Math.round(height));
  canvas.width = safeWidth * dpr;
  canvas.height = safeHeight * dpr;
  canvas.style.width = `${safeWidth}px`;
  canvas.style.height = `${safeHeight}px`;
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.clearRect(0, 0, safeWidth, safeHeight);
  return { context, width: safeWidth, height: safeHeight };
}

function placeholderLandmarkInspector(message) {
  const bounds = els.landmarkPreviewCanvas.getBoundingClientRect();
  const { context, width, height } = resizeCanvasToBox(
    els.landmarkPreviewCanvas,
    bounds.width || 280,
    bounds.height || 200,
  );
  context.fillStyle = "rgba(99, 102, 241, 0.10)";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(167, 139, 250, 0.45)";
  context.strokeRect(12, 12, width - 24, height - 24);
  context.fillStyle = "rgba(243, 244, 246, 0.82)";
  context.font = "600 14px system-ui";
  context.textAlign = "center";
  context.textBaseline = "middle";
  context.fillText(message, width / 2, height / 2);
}

function emptyLandmarkData(video, reason = "missing") {
  return {
    sourceFile: video?.source_file || "unknown",
    available: false,
    reason,
    frames: new Map(),
    validFrameCount: 0,
    dedicatedFrameCount: 0,
    fallbackFrameCount: 0,
  };
}

function parseLandmarkCsv(text, video) {
  const trimmed = text.trim();
  if (!trimmed) {
    return emptyLandmarkData(video, "empty");
  }
  const lines = trimmed.split(/\r?\n/);
  const header = lines[0].split(",");
  const frameIdxColumn = header.indexOf("frame_idx");
  const sourceFileColumn = header.indexOf("source_file");
  const coordinateColumns = Array.from({ length: 21 }, (_, index) => ({
    x: header.indexOf(`x${index}`),
    y: header.indexOf(`y${index}`),
    z: header.indexOf(`z${index}`),
  }));
  const frames = new Map();

  for (const line of lines.slice(1)) {
    if (!line) continue;
    const values = line.split(",");
    if (sourceFileColumn >= 0) {
      const rowSourceFile = String(values[sourceFileColumn] || "").trim();
      if (rowSourceFile !== String(video?.source_file || "").trim()) {
        continue;
      }
    }
    const frameIdx = Number(values[frameIdxColumn] || 0);
    const x0 = values[coordinateColumns[0].x];
    if (x0 === undefined || x0 === "") continue;
    const points = coordinateColumns.map(({ x, y, z }) => ({
      x: Number(values[x]),
      y: Number(values[y]),
      z: Number(values[z]),
    }));
    frames.set(frameIdx, points);
  }

  return {
    sourceFile: video?.source_file || "unknown",
    available: frames.size > 0,
    reason: frames.size > 0 ? "ok" : "no_valid_rows",
    frames,
    validFrameCount: frames.size,
    dedicatedFrameCount: frames.size,
    fallbackFrameCount: 0,
  };
}

function datasetCsvPath(video) {
  return video?.dataset_csv_path || "";
}

function mergeLandmarkData(primaryData, fallbackData, video) {
  const mergedFrames = new Map(primaryData?.frames || []);
  let fallbackFrameCount = 0;
  for (const [frameIdx, points] of fallbackData?.frames || []) {
    if (mergedFrames.has(frameIdx)) continue;
    mergedFrames.set(frameIdx, points);
    fallbackFrameCount += 1;
  }
  if (!mergedFrames.size) {
    return emptyLandmarkData(video, primaryData?.reason || fallbackData?.reason || "no_valid_rows");
  }
  return {
    sourceFile: video?.source_file || "unknown",
    available: true,
    reason: fallbackFrameCount > 0 ? "merged_with_dataset_fallback" : primaryData?.reason || "ok",
    frames: mergedFrames,
    validFrameCount: mergedFrames.size,
    dedicatedFrameCount: primaryData?.validFrameCount || 0,
    fallbackFrameCount,
  };
}

async function ensureLandmarkData(video) {
  if (!video) return emptyLandmarkData(null, "no_video");
  const cacheKey = video.source_file;
  if (state.landmarkCache.has(cacheKey)) {
    return state.landmarkCache.get(cacheKey);
  }
  if (state.landmarkPromiseCache.has(cacheKey)) {
    return state.landmarkPromiseCache.get(cacheKey);
  }

  if (video.landmark_exists === false) {
    const data = emptyLandmarkData(video, "missing_file");
    state.landmarkCache.set(cacheKey, data);
    return data;
  }

  const request = (async () => {
    try {
      const timestamp = Date.now();
      const [landmarkResponse, datasetResponse] = await Promise.all([
        fetch(`${landmarkCsvPath(video)}?ts=${timestamp}`),
        video.dataset_csv_path ? fetch(`${datasetCsvPath(video)}?ts=${timestamp}`) : Promise.resolve(null),
      ]);
      const primary = landmarkResponse.ok
        ? parseLandmarkCsv(await landmarkResponse.text(), video)
        : emptyLandmarkData(video, `http_${landmarkResponse.status}`);
      const fallback = datasetResponse?.ok
        ? parseLandmarkCsv(await datasetResponse.text(), video)
        : emptyLandmarkData(video, datasetResponse ? `dataset_http_${datasetResponse.status}` : "no_dataset_csv");
      const data = mergeLandmarkData(primary, fallback, video);
      if (!data.available && !landmarkResponse.ok) {
        data.reason = `http_${landmarkResponse.status}`;
      }
      state.landmarkCache.set(cacheKey, data);
      return data;
    } catch (error) {
      console.error(error);
      const failed = emptyLandmarkData(video, "fetch_failed");
      state.landmarkCache.set(cacheKey, failed);
      return failed;
    } finally {
      state.landmarkPromiseCache.delete(cacheKey);
    }
  })();

  state.landmarkPromiseCache.set(cacheKey, request);
  return request;
}

async function ensureFullInferenceData(video) {
  if (!video) return null;
  const key = fullInferenceKey(video);
  if (!key) return null;
  if (state.fullInferenceCache.has(key)) {
    return state.fullInferenceCache.get(key);
  }
  if (state.fullInferencePromiseCache.has(key)) {
    return state.fullInferencePromiseCache.get(key);
  }

  if (!video.landmark_exists) {
    const empty = { suite_name: state.selectedSuiteName, model_id: state.selectedModelId, source_file: video.source_file, frame_count: 0, events: [] };
    state.fullInferenceCache.set(key, empty);
    return empty;
  }

  const params = new URLSearchParams({
    suite: state.selectedSuiteName,
    model: state.selectedModelId,
    source: video.source_file,
  });
  const request = (async () => {
    try {
      const response = await fetch(`/api/full-source-inference?${params.toString()}`);
      if (!response.ok) {
        throw new Error(`Full inference request failed: ${response.status}`);
      }
      const payload = await response.json();
      state.fullInferenceCache.set(key, payload);
      return payload;
    } catch (error) {
      console.error(error);
      const fallback = { suite_name: state.selectedSuiteName, model_id: state.selectedModelId, source_file: video.source_file, frame_count: 0, events: video.events || [] };
      state.fullInferenceCache.set(key, fallback);
      return fallback;
    } finally {
      state.fullInferencePromiseCache.delete(key);
    }
  })();

  state.fullInferencePromiseCache.set(key, request);
  return request;
}

function drawLandmarkConnections(context, points, projector, lineColor, pointColor) {
  context.lineWidth = 2;
  context.lineCap = "round";
  context.lineJoin = "round";
  context.strokeStyle = lineColor;
  HAND_CONNECTIONS.forEach(([start, end]) => {
    const from = projector(points[start]);
    const to = projector(points[end]);
    if (!from || !to) return;
    context.beginPath();
    context.moveTo(from.x, from.y);
    context.lineTo(to.x, to.y);
    context.stroke();
  });

  context.fillStyle = pointColor;
  points.forEach((point) => {
    const projected = projector(point);
    if (!projected) return;
    context.beginPath();
    context.arc(projected.x, projected.y, 3.4, 0, Math.PI * 2);
    context.fill();
  });
}

function renderLandmarkOverlay(points) {
  const bounds = els.videoPlayer.getBoundingClientRect();
  const { context, width, height } = resizeCanvasToBox(
    els.landmarkOverlay,
    bounds.width || els.frameAnalysisShell.clientWidth || 640,
    bounds.height || els.frameAnalysisShell.clientHeight || 360,
  );
  if (!points) return;

  drawLandmarkConnections(
    context,
    points,
    (point) => {
      if (!Number.isFinite(point?.x) || !Number.isFinite(point?.y)) return null;
      return { x: point.x * width, y: point.y * height };
    },
    "rgba(43, 93, 255, 0.95)",
    "rgba(55, 250, 124, 0.98)",
  );
}

function renderLandmarkInspector(points, frameIndex, landmarkData) {
  if (!points) {
    placeholderLandmarkInspector(`frame ${frameIndex}: no hand landmarks`);
    return;
  }

  const bounds = els.landmarkPreviewCanvas.getBoundingClientRect();
  const { context, width, height } = resizeCanvasToBox(
    els.landmarkPreviewCanvas,
    bounds.width || 280,
    bounds.height || 200,
  );
  const validPoints = points.filter((point) => Number.isFinite(point.x) && Number.isFinite(point.y));
  if (!validPoints.length) {
    placeholderLandmarkInspector(`frame ${frameIndex}: no hand landmarks`);
    return;
  }

  const minX = Math.min(...validPoints.map((point) => point.x));
  const maxX = Math.max(...validPoints.map((point) => point.x));
  const minY = Math.min(...validPoints.map((point) => point.y));
  const maxY = Math.max(...validPoints.map((point) => point.y));
  const padding = 22;
  const spanX = Math.max(maxX - minX, 0.08);
  const spanY = Math.max(maxY - minY, 0.08);
  const scale = Math.min((width - padding * 2) / spanX, (height - padding * 2) / spanY);
  const offsetX = (width - spanX * scale) / 2;
  const offsetY = (height - spanY * scale) / 2;

  context.fillStyle = "rgba(34, 41, 74, 0.92)";
  context.fillRect(0, 0, width, height);
  context.strokeStyle = "rgba(167, 139, 250, 0.5)";
  context.lineWidth = 1.5;
  context.strokeRect(10, 10, width - 20, height - 20);

  drawLandmarkConnections(
    context,
    points,
    (point) => {
      if (!Number.isFinite(point?.x) || !Number.isFinite(point?.y)) return null;
      return {
        x: offsetX + (point.x - minX) * scale,
        y: offsetY + (point.y - minY) * scale,
      };
    },
    "rgba(132, 94, 247, 0.98)",
    "rgba(196, 181, 253, 0.98)",
  );

  context.fillStyle = "rgba(243, 244, 246, 0.9)";
  context.font = "600 12px system-ui";
  context.textAlign = "left";
  context.fillText(`frame ${frameIndex}`, 18, 24);
  context.textAlign = "right";
  context.fillText(`${landmarkData.validFrameCount} valid landmark frames`, width - 18, 24);
}

function renderLandmarkFrame(model, frameIndex) {
  const video = selectedVideo(model);
  if (!video) {
    clearCanvas(els.landmarkOverlay);
    placeholderLandmarkInspector("no source selected");
    els.landmarkStatus.textContent = "landmark unavailable";
    return;
  }

  const landmarkData = state.landmarkCache.get(video.source_file);
  if (!landmarkData) {
    clearCanvas(els.landmarkOverlay);
    placeholderLandmarkInspector("loading landmark data...");
    els.landmarkStatus.textContent = "landmark loading";
    return;
  }

  const points = landmarkData.frames.get(frameIndex) || null;
  renderLandmarkOverlay(points);
  renderLandmarkInspector(points, frameIndex, landmarkData);

  if (points) {
    els.landmarkStatus.textContent = `frame ${frameIndex} · 21 landmarks`;
    return;
  }

  if (!landmarkData.available) {
    els.landmarkStatus.textContent = "landmark file unavailable";
    return;
  }

  els.landmarkStatus.textContent = `frame ${frameIndex} · no hand detected`;
}

function currentReviewer() {
  const name = els.reviewerName.value.trim();
  return name || "anonymous";
}

function currentReviewKey() {
  return [
    "jamjambeat-eval-review",
    state.selectedSuiteName || "no-suite",
    currentReviewer(),
    state.selectedModelId || "no-model",
    state.selectedSource || "all",
  ].join(":");
}

function renderIndex() {
  const models = state.index.models || [];
  const best = models[0];
  const sourceCount = (state.index.available_source_files || []).length;
  const suiteEntry = currentSuiteEntry();
  const datasetTag = state.index.suite_meta?.dataset_tag || suiteEntry?.dataset_tag || "-";
  els.suiteNameChip.textContent = datasetTag;
  els.heroTitle.textContent = state.index.suite_name || state.index.latest_suite_name || "Evaluation dataset";
  els.heroCopy.textContent = `로컬 evaluation dataset ${datasetTag} 기준 ${models.length}개 모델을 비교하고, 선택한 모델의 최신 run을 브라우저에서 검사합니다.`;
  els.bestMacroF1.textContent = best ? formatNumber(best.macro_f1) : "-";
  els.recommendedModel.textContent = state.index.recommended_model_id || "-";
  els.availableUsers.textContent = String((state.index.available_users || []).length);
  els.availableSources.textContent = String(sourceCount);
  renderModelList();
  renderLeaderboard();
}

function filteredModels() {
  const needle = state.filters.search.trim().toLowerCase();
  if (!needle) return state.index.models || [];
  return (state.index.models || []).filter((model) => model.model_id.toLowerCase().includes(needle));
}

function renderModelList() {
  const models = filteredModels();
  if (!models.length) {
    els.modelList.innerHTML = `<div class="empty-state">검색 결과가 없습니다.</div>`;
    return;
  }
  els.modelList.innerHTML = models
    .map(
      (model) => `
        <button class="model-button ${model.model_id === state.selectedModelId ? "active" : ""}" data-model-id="${escapeHtml(model.model_id)}">
          <span class="model-name">${escapeHtml(model.model_id)}</span>
          <span class="model-meta">${escapeHtml(model.mode)} · F1 ${formatNumber(model.macro_f1, 3)}</span>
        </button>
      `,
    )
    .join("");

  els.modelList.querySelectorAll(".model-button").forEach((button) => {
    button.addEventListener("click", async () => {
      await selectModel(button.dataset.modelId);
    });
  });
}

function renderLeaderboard() {
  const models = filteredModels();
  if (!models.length) {
    els.leaderboardBody.innerHTML = `<tr><td colspan="7"><div class="empty-state">표시할 모델이 없습니다.</div></td></tr>`;
    return;
  }
  els.leaderboardBody.innerHTML = models
    .map(
      (model) => `
        <tr class="${model.model_id === state.selectedModelId ? "active" : ""}" data-model-id="${escapeHtml(model.model_id)}">
          <td>${escapeHtml(model.model_id)}</td>
          <td>${escapeHtml(model.mode)}</td>
          <td>${formatNumber(model.macro_f1)}</td>
          <td>${formatNumber(model.accuracy)}</td>
          <td>${formatNumber(model.fp_per_min, 3)}</td>
          <td>${formatMs(model.latency_p95_ms)}</td>
          <td>${escapeHtml(String(model.source_count || 0))}</td>
        </tr>
      `,
    )
    .join("");

  els.leaderboardBody.querySelectorAll("tr[data-model-id]").forEach((row) => {
    row.addEventListener("click", async () => {
      await selectModel(row.dataset.modelId);
    });
  });
}

async function selectModel(modelId) {
  if (!modelId) return;
  const preferredFrame = state.selectedSource !== "all" ? state.currentFrameIndex : null;
  state.selectedModelId = modelId;
  state.activeEventIndex = 0;
  renderModelList();
  renderLeaderboard();
  const model = await getModelData(modelId);
  hydrateSelectedModel(model, { preferredFrame });
}

function hydrateSelectedModel(model, options = {}) {
  const { preferredFrame = null } = options;
  const summary = model.comparison_row || {};
  els.selectedModelTitle.textContent = `${model.model_id} · ${model.mode}`;
  els.selectedMacroF1.textContent = formatNumber(summary.macro_f1);
  els.selectedAccuracy.textContent = formatNumber(summary.accuracy);
  els.selectedFpPerMin.textContent = formatNumber(summary.fp_per_min, 3);
  els.selectedLatency.textContent = formatMs(summary.latency_p95_ms);
  els.selectedModelTags.innerHTML = [
    `<span class="tag">mode: ${escapeHtml(model.mode)}</span>`,
    `<span class="tag">epochs: ${escapeHtml(String(summary.epochs_ran || model.run_summary?.epochs_ran || "-"))}</span>`,
    `<span class="tag">test samples: ${escapeHtml(String(summary.test_samples || model.metrics_summary?.total_samples || "-"))}</span>`,
  ].join("");

  renderExplainer(model);
  renderRunSnapshot(model);
  populateFilters(model);
  renderPerClass(model.per_class_report || []);
  renderTrainingCurve(model.train_history || []);
  renderConfusionMatrix(model.confusion_matrix || { labels: [], matrix: [] });
  renderArtifacts(model.artifacts || {});

  const firstAvailableSource = allVideos(model)[0]?.source_file || "all";
  const canPreserveSource =
    state.selectedSource !== "all" && allVideos(model).some((video) => video.source_file === state.selectedSource);
  if (!canPreserveSource) {
    state.selectedSource = firstAvailableSource;
  }
  populateSourceOptions(model);
  renderSourceExplorer(model);
  hydrateVideoArea(model, { preferredFrame: canPreserveSource ? preferredFrame : null });
  loadReview();
}

function renderExplainer(model) {
  const explainer = model.model_explainer || {};
  const metrics = explainer.key_metrics || {};
  els.modelExplainer.innerHTML = `
    <div class="explainer-card">
      <p class="eyebrow">Headline</p>
      <h3>${escapeHtml(explainer.headline || model.model_id)}</h3>
      <p>${escapeHtml(explainer.summary || "")}</p>
    </div>
    <div class="explainer-card">
      <p class="eyebrow">Strengths</p>
      <ul class="explainer-list">
        ${(explainer.strengths || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </div>
    <div class="explainer-card">
      <p class="eyebrow">Tradeoffs</p>
      <ul class="explainer-list">
        ${(explainer.tradeoffs || []).map((item) => `<li>${escapeHtml(item)}</li>`).join("")}
      </ul>
    </div>
    <div class="explainer-card">
      <p class="eyebrow">Key Metrics</p>
      <ul class="explainer-list">
        <li>Accuracy: ${formatNumber(metrics.accuracy)}</li>
        <li>Macro F1: ${formatNumber(metrics.macro_f1)}</li>
        <li>FP/min: ${formatNumber(metrics.fp_per_min, 3)}</li>
        <li>Latency p95: ${formatMs(metrics.latency_p95_ms)}</li>
      </ul>
    </div>
  `;
}

function uniqueValues(items, key) {
  return [...new Set(items.map((item) => item[key]).filter(Boolean))].sort();
}

function renderSelect(select, values, currentValue, allLabel) {
  select.innerHTML = [`<option value="all">${allLabel}</option>`]
    .concat(values.map((value) => `<option value="${escapeHtml(value)}" ${value === currentValue ? "selected" : ""}>${escapeHtml(value)}</option>`))
    .join("");
}

function datasetUserSplits(model) {
  return model?.dataset_user_splits || model?.available_filters?.dataset_user_splits || {};
}

function datasetUserOptions(model) {
  const fromVideos = uniqueValues(model.videos || [], "user");
  const splitMap = datasetUserSplits(model);
  const unionUsers = [...new Set([...Object.keys(splitMap), ...fromVideos])].sort();
  const filteredUsers = unionUsers.filter((user) => HUMAN_USER_PATTERN.test(user));
  const displayUsers = filteredUsers.length ? filteredUsers : unionUsers;
  return displayUsers.map((user) => ({
    value: user,
    label: splitMap[user]?.length ? `${user} (${splitMap[user].join("/")})` : user,
  }));
}

function renderUserFilter(model) {
  const options = datasetUserOptions(model);
  els.userFilter.innerHTML = options
    .map(
      (option) =>
        `<option value="${escapeHtml(option.value)}" ${option.value === state.filters.user ? "selected" : ""}>${escapeHtml(option.label)}</option>`,
    )
    .join("");
}

function selectedDatasetUserHint(model) {
  if (!state.filters.user || state.filters.user === "all") return "";
  const splits = datasetUserSplits(model)[state.filters.user];
  if (!splits?.length) return "";
  return `${state.filters.user} 사용자는 이 suite에서 ${splits.join("/")} split에 포함됩니다.`;
}

function populateFilters(model) {
  const videos = model.videos || [];
  const allowedUsers = datasetUserOptions(model).map((option) => option.value);
  if (allowedUsers.length) {
    if (!allowedUsers.includes(state.filters.user)) {
      state.filters.user = allowedUsers[0];
    }
  } else {
    state.filters.user = "all";
  }
  renderUserFilter(model);
  renderSelect(els.motionFilter, uniqueValues(videos, "motion"), state.filters.motion, "모든 motion");
  renderSelect(els.sideFilter, uniqueValues(videos, "side"), state.filters.side, "모든 side");
}

function getFilteredVideos(model) {
  return allVideos(model).filter((video) => {
    if (state.filters.user !== "all" && video.user !== state.filters.user) return false;
    if (state.filters.motion !== "all" && video.motion !== state.filters.motion) return false;
    if (state.filters.side !== "all" && video.side !== state.filters.side) return false;
    return true;
  });
}

function populateSourceOptions(model) {
  const videos = getFilteredVideos(model);
  if (!videos.length) {
    els.sourceFilter.innerHTML = `<option value="all">매칭되는 source 없음</option>`;
    state.selectedSource = "all";
    return;
  }
  if (!videos.some((video) => video.source_file === state.selectedSource)) {
    state.selectedSource = videos[0].source_file;
  }
  els.sourceFilter.innerHTML = videos
    .map(
      (video) => {
        const parts = [video.source_file, video.user, video.motion, video.side];
        if (!video.prediction_available) {
          parts.push("raw only");
        }
        return `<option value="${escapeHtml(video.source_file)}" ${video.source_file === state.selectedSource ? "selected" : ""}>${escapeHtml(parts.join(" · "))}</option>`;
      },
    )
    .join("");
}

function renderRunSnapshot(model) {
  const hyper = model.run_summary?.hyperparameters || {};
  const split = model.run_summary?.split_sizes || model.run_summary?.dataset_sizes || {};
  const datasetInfo = model.metrics_summary?.dataset_info || model.run_summary?.dataset_info || {};
  const userSplitSummary = Object.entries(datasetUserSplits(model))
    .map(([user, splits]) => `${user}:${splits.join("/")}`)
    .join(", ");
  els.runSnapshot.innerHTML = `
    <div class="snapshot-card">
      <p class="eyebrow">Hyperparameters</p>
      <ul>
        <li>epochs: ${escapeHtml(String(hyper.epochs ?? "-"))}</li>
        <li>batch size: ${escapeHtml(String(hyper.batch_size ?? "-"))}</li>
        <li>lr: ${escapeHtml(String(hyper.lr ?? "-"))}</li>
        <li>optimizer: ${escapeHtml(String(hyper.optimizer ?? "-"))}</li>
      </ul>
    </div>
    <div class="snapshot-card">
      <p class="eyebrow">Dataset Split</p>
      <ul>
        <li>train: ${escapeHtml(String(split.train ?? "-"))}</li>
        <li>val: ${escapeHtml(String(split.val ?? "-"))}</li>
        <li>test: ${escapeHtml(String(split.test ?? "-"))}</li>
        <li>source groups: ${escapeHtml(String((datasetInfo.source_groups || []).join(", ") || "-"))}</li>
        <li>dataset users: ${escapeHtml(userSplitSummary || "-")}</li>
      </ul>
    </div>
    <div class="snapshot-card">
      <p class="eyebrow">Operational Notes</p>
      <ul>
        <li>best val loss: ${formatNumber(model.comparison_row?.best_val_loss)}</li>
        <li>class0 FNR: ${formatNumber(model.comparison_row?.class0_fnr)}</li>
        <li>class0 FPR: ${formatNumber(model.comparison_row?.class0_fpr)}</li>
        <li>inputs: ${escapeHtml(String((model.run_summary?.inputs || []).join(", ") || "-"))}</li>
      </ul>
    </div>
  `;
}

function renderSourceExplorer(model) {
  if (!els.sourceExplorer) return;
  els.sourceExplorer.innerHTML = "";
}

function selectedVideo(model) {
  const filteredVideos = getFilteredVideos(model);
  if (state.selectedSource === "all") {
    return filteredVideos[0] || null;
  }
  return filteredVideos.find((video) => video.source_file === state.selectedSource) || filteredVideos[0] || null;
}

function nearestEventIndex(video, frameIndex) {
  const events = comparisonEvents(video);
  if (!events.length) return 0;
  let nearestIndex = 0;
  let nearestDistance = Number.POSITIVE_INFINITY;
  events.forEach((event, index) => {
    const eventFrame = Number(event.frame_idx ?? 0);
    const distance = Math.abs(eventFrame - frameIndex);
    if (distance < nearestDistance) {
      nearestDistance = distance;
      nearestIndex = index;
    }
  });
  return nearestIndex;
}

function renderFrameOverlay(video, event, frameIndex) {
  const totalFrames = Math.max(Number(video.total_frames || video.frame_count || 0), 1);
  const frameTime = formatPlaybackTime(frameIndex / Math.max(Number(video.fps || 30), 1));
  const hasEvent = Boolean(event);
  const exactPredictionFrame = hasEvent ? Number(event?.frame_idx ?? frameIndex) === frameIndex : false;
  const predLabel = !hasEvent ? "RAW FRAME" : exactPredictionFrame ? "PRED" : "NEAREST PRED";
  const eventFrame = Number(event?.frame_idx ?? frameIndex);
  const eventDelta = frameIndex - eventFrame;
  const eventDeltaText = !hasEvent
    ? "prediction event unavailable for this source"
    : eventDelta === 0
      ? "exact prediction frame"
      : `${eventDelta > 0 ? "+" : ""}${eventDelta}f from event #${eventFrame}`;
  els.frameOverlayTop.innerHTML = `
    <div>
      <div class="frame-overlay-title">${escapeHtml(video.source_file)}</div>
      <div class="frame-overlay-subtitle">
        frame=${escapeHtml(String(frameIndex))} / ${escapeHtml(String(totalFrames - 1))} · ${escapeHtml(frameTime)}
      </div>
    </div>
    <div class="frame-overlay-caption">
      ${escapeHtml(video.user)} · ${escapeHtml(video.motion)} · ${escapeHtml(video.side)}<br />
      ${escapeHtml(eventDeltaText)}
    </div>
  `;

  els.frameOverlayBottom.innerHTML = `
    <div>
      <div class="frame-overlay-pred">${escapeHtml(predLabel)}: ${escapeHtml(event?.predicted || "no model event")}${hasEvent ? ` (${formatNumber(event?.confidence, 2)})` : ""}</div>
      <div class="frame-overlay-gt">GT: ${escapeHtml(event?.ground_truth || video.gesture_name || "-")}</div>
    </div>
    <div class="frame-overlay-caption">
      Space 재생/정지 · A,D 또는 ←,→ 1프레임 · W,S 또는 ↑,↓ 이벤트 점프 · Shift + ←,→ 10프레임
    </div>
  `;

  const sortedProbabilities = (event?.probabilities || []).slice().sort((a, b) => b.value - a.value).slice(0, 6);
  els.frameProbabilityOverlay.innerHTML = sortedProbabilities.length
    ? sortedProbabilities
        .map(
          (entry) => `
            <div class="probability-bar">
              <span>${escapeHtml(entry.label)}</span>
              <div class="probability-track"><div class="probability-fill" style="width: ${entry.value * 100}%"></div></div>
              <span>${Math.round(entry.value * 100)}%</span>
            </div>
          `,
        )
        .join("")
    : `<div class="current-meta">이 source는 현재 landmark 기반 전체-frame 추론 결과를 불러오는 중이거나 사용할 수 없습니다.</div>`;
}

function renderEventRail(video) {
  const events = comparisonEvents(video);
  const totalFrames = Math.max(Number(video?.total_frames || video?.frame_count || 0), 1);
  if (!events.length) {
    els.frameEventRail.innerHTML = "";
    return;
  }

  els.frameEventRail.innerHTML = events
    .map((event, index) => {
      const left = (Number(event.frame_idx || 0) / Math.max(totalFrames - 1, 1)) * 100;
      const classes = ["frame-event-marker"];
      if (event.is_mismatch) classes.push("mismatch");
      if (index === state.activeEventIndex) classes.push("active");
      return `
        <button
          class="${classes.join(" ")}"
          type="button"
          data-index="${index}"
          style="left:${left.toFixed(4)}%"
          title="${escapeHtml(
            `${event.timestamp || "-"} · frame ${event.frame_idx ?? "-"} · GT ${event.ground_truth || "-"} · Pred ${event.predicted || "-"} · ${event.is_mismatch ? "mismatch" : "match"}`,
          )}"
          aria-label="${escapeHtml(`prediction event ${index + 1}`)}"
        ></button>
      `;
    })
    .join("");

  els.frameEventRail.querySelectorAll(".frame-event-marker").forEach((button) => {
    button.addEventListener("click", async () => {
      const model = await getModelData(state.selectedModelId);
      jumpToEvent(model, Number(button.dataset.index || 0));
    });
  });
}

function updateFrameScrubber(model) {
  const video = selectedVideo(model);
  if (!video) return;
  const totalFrames = Math.max(currentVideoTotalFrames(model), video.frame_count || 1);
  els.frameScrubber.max = String(Math.max(totalFrames - 1, 0));
  els.frameScrubber.value = String(Math.min(state.currentFrameIndex, totalFrames - 1));
  els.frameScrubberMeta.innerHTML = `
    <span>frame ${escapeHtml(String(state.currentFrameIndex))} / ${escapeHtml(String(Math.max(totalFrames - 1, 0)))}</span>
    <span>${escapeHtml(formatPlaybackTime(els.videoPlayer.currentTime || 0))} · ${escapeHtml(String(Number(video.fps || 30).toFixed(2)))} fps</span>
  `;
  els.framePlayPause.textContent = els.videoPlayer.paused ? "Play" : "Pause";
}

function syncFramePanels(model, frameIndex, options = {}) {
  const { forceTimeline = false } = options;
  const video = selectedVideo(model);
  if (!video) return;
  const totalFrames = Math.max(currentVideoTotalFrames(model), video.frame_count || 1);
  const previousEventIndex = state.activeEventIndex;
  state.currentFrameIndex = Math.max(0, Math.min(frameIndex, totalFrames - 1));
  state.activeEventIndex = nearestEventIndex(video, state.currentFrameIndex);
  const activeEvent = comparisonEvents(video)[state.activeEventIndex];
  updateFrameScrubber(model);
  renderFrameOverlay(video, activeEvent, state.currentFrameIndex);
  renderLandmarkFrame(model, state.currentFrameIndex);
  syncCurrentPrediction(video, state.activeEventIndex, state.currentFrameIndex);
  if (forceTimeline || state.activeEventIndex !== previousEventIndex) {
    renderTimeline(video);
    renderEventRail(video);
  }
}

function cancelPlaybackFrameLoop() {
  if (state.videoFrameCallbackId !== null && typeof els.videoPlayer.cancelVideoFrameCallback === "function") {
    els.videoPlayer.cancelVideoFrameCallback(state.videoFrameCallbackId);
  }
  if (state.animationFrameId !== null) {
    cancelAnimationFrame(state.animationFrameId);
  }
  state.videoFrameCallbackId = null;
  state.animationFrameId = null;
}

function startPlaybackFrameLoop() {
  cancelPlaybackFrameLoop();
  if (els.videoPlayer.paused || els.videoPlayer.ended) return;

  if (typeof els.videoPlayer.requestVideoFrameCallback === "function") {
    const tick = () => {
      state.videoFrameCallbackId = null;
      if (els.videoPlayer.paused || els.videoPlayer.ended) return;
      const model = currentModelData();
      if (model) {
        syncFrameState(model);
      }
      state.videoFrameCallbackId = els.videoPlayer.requestVideoFrameCallback(tick);
    };
    state.videoFrameCallbackId = els.videoPlayer.requestVideoFrameCallback(tick);
    return;
  }

  const tick = () => {
    state.animationFrameId = null;
    if (els.videoPlayer.paused || els.videoPlayer.ended) return;
    const model = currentModelData();
    if (model) {
      syncFrameState(model);
    }
    state.animationFrameId = requestAnimationFrame(tick);
  };
  state.animationFrameId = requestAnimationFrame(tick);
}

function syncFrameState(model) {
  if (!state.frameSyncReady) return;
  const frameIndex = currentPlayerFrame(model);
  syncFramePanels(model, frameIndex);
}

function seekToFrame(model, frameIndex) {
  const video = selectedVideo(model);
  if (!video) return;
  const totalFrames = Math.max(currentVideoTotalFrames(model), video.frame_count || 1);
  const clamped = Math.max(0, Math.min(frameIndex, totalFrames - 1));
  const targetTime = clamped * frameDuration(model);
  const currentTime = Number(els.videoPlayer.currentTime || 0);
  state.pendingSeekFrame = null;
  if (Math.abs(currentTime - targetTime) <= frameDuration(model) / 4) {
    state.frameSyncReady = true;
    syncFramePanels(model, clamped);
    return;
  }
  state.frameSyncReady = false;
  state.pendingSeekFrame = clamped;
  els.videoPlayer.currentTime = targetTime;
  syncFramePanels(model, clamped);
}

function stepFrame(model, delta) {
  els.videoPlayer.pause();
  seekToFrame(model, state.currentFrameIndex + delta);
}

function jumpToEvent(model, eventIndex) {
  const video = selectedVideo(model);
  const events = comparisonEvents(video);
  if (!events.length) return;
  const clampedIndex = Math.max(0, Math.min(eventIndex, events.length - 1));
  const targetFrame = Number(events[clampedIndex]?.frame_idx || 0);
  els.videoPlayer.pause();
  seekToFrame(model, targetFrame);
}

function jumpRelativeToCurrentEvent(model, delta) {
  const video = selectedVideo(model);
  if (!video) return;
  const baseIndex = nearestEventIndex(video, state.currentFrameIndex);
  jumpToEvent(model, baseIndex + delta);
}

function hydrateVideoArea(model, options = {}) {
  const { preferredFrame = null } = options;
  const video = selectedVideo(model);
  if (!video) {
    cancelPlaybackFrameLoop();
    state.pendingSeekFrame = null;
    state.frameSyncReady = false;
    const hint = selectedDatasetUserHint(model);
    els.sourceContext.innerHTML = `<div class="empty-state">현재 필터에 맞는 source가 없습니다.${hint ? `<br />${escapeHtml(hint)}` : ""}</div>`;
    els.videoPlayer.removeAttribute("src");
    els.videoStats.innerHTML = `<div class="empty-state">비디오 없음</div>`;
    els.frameOverlayTop.innerHTML = "";
    els.frameOverlayBottom.innerHTML = "";
    els.frameProbabilityOverlay.innerHTML = "";
    els.frameEventRail.innerHTML = "";
    els.frameScrubberMeta.innerHTML = "";
    clearCanvas(els.landmarkOverlay);
    placeholderLandmarkInspector("no source selected");
    els.landmarkStatus.textContent = "landmark unavailable";
    els.currentPrediction.innerHTML = `<div class="empty-state">이벤트 없음</div>`;
    els.timelineList.innerHTML = `<div class="empty-state">이벤트 없음</div>`;
    els.timelineCount.textContent = "0 events";
    return;
  }

  state.selectedSource = video.source_file;
  const totalFrames = Math.max(currentVideoTotalFrames(model), Number(video.total_frames || video.frame_count || 1), 1);
  const initialFrame = Number.isFinite(preferredFrame)
    ? Math.max(0, Math.min(Number(preferredFrame), totalFrames - 1))
    : Number(video.events?.[0]?.frame_idx || 0);
  state.currentFrameIndex = initialFrame;
  state.activeEventIndex = nearestEventIndex(video, initialFrame);
  state.videoLoadToken += 1;
  state.pendingSeekFrame = initialFrame;
  state.frameSyncReady = false;
  state.landmarkLoadToken += 1;
  state.fullInferenceLoadToken += 1;
  state.isScrubbing = false;
  state.scrubWasPlaying = false;
  cancelPlaybackFrameLoop();
  const loadToken = state.videoLoadToken;
  const landmarkLoadToken = state.landmarkLoadToken;
  const fullInferenceLoadToken = state.fullInferenceLoadToken;
  const isOutsideFilter = !getFilteredVideos(model).some((entry) => entry.source_file === video.source_file);
  els.sourceContext.innerHTML = `
    <strong>${escapeHtml(video.source_file)}</strong>
    <p class="source-meta">
      user: ${escapeHtml(video.user)} · motion: ${escapeHtml(video.motion)} · side: ${escapeHtml(video.side)} · GT: ${escapeHtml(video.gesture_name)}
    </p>
    <p class="source-meta">
      full comparison dots: GT vs Pred on available inference frames · landmark frames ${escapeHtml(String(video.landmark_frame_count || "-"))} · total frames ${escapeHtml(String(video.total_frames || "-"))}
    </p>
    ${isOutsideFilter ? `<p class="source-meta">현재 frame checker는 필터와 무관하게 suite 전체 source에서 직접 연 상태입니다.</p>` : ""}
  `;

  els.videoPlayer.pause();
  els.videoPlayer.defaultPlaybackRate = 1;
  els.videoPlayer.playbackRate = 1;
  els.videoPlayer.src = video.video_path;
  els.videoPlayer.load();
  focusFrameAnalysisShell();
  clearCanvas(els.landmarkOverlay);
  placeholderLandmarkInspector("loading landmark data...");
  els.landmarkStatus.textContent = "landmark loading";
  let initialSeekRequested = false;
  const applyInitialSeek = () => {
    if (initialSeekRequested || loadToken !== state.videoLoadToken) return;
    initialSeekRequested = true;
    els.videoPlayer.pause();
    els.videoPlayer.currentTime = initialFrame / Math.max(Number(video.fps || 30), 1);
  };
  if (els.videoPlayer.readyState >= 2) {
    applyInitialSeek();
  } else {
    els.videoPlayer.addEventListener("loadeddata", applyInitialSeek, { once: true });
    els.videoPlayer.addEventListener("canplay", applyInitialSeek, { once: true });
  }
  els.videoStats.innerHTML = [
    statBlock("Dataset User", video.user),
    statBlock("Ground Truth", video.gesture_name),
    statBlock("Motion", video.motion),
    statBlock("Test Accuracy", video.accuracy === null || video.accuracy === undefined ? "예측 없음" : formatPercent(video.accuracy)),
    statBlock("FPS", Number(video.fps || 30).toFixed(2)),
    statBlock("Total Frames", String(video.total_frames || "-")),
    statBlock("Duration", formatPlaybackTime(Number(video.duration_sec || 0))),
    statBlock("Resolution", `${video.width || "-"}x${video.height || "-"}`),
    statBlock("Test Prediction Frames", String(video.prediction_frame_count || 0)),
    statBlock("Landmark Frames", String(video.landmark_frame_count || "-")),
    statBlock("Landmark Source", String(video.landmark_source_kind || "-")),
    statBlock("Dataset CSV", video.dataset_csv_path ? "available" : "none"),
    statBlock("Full Comparison", "loading..."),
  ].join("");

  updateFrameScrubber(model);
  syncFramePanels(model, initialFrame, { forceTimeline: true });
  void ensureLandmarkData(video).then((landmarkData) => {
    if (landmarkLoadToken !== state.landmarkLoadToken) return;
    if ((!video.prediction_available || !video.prediction_frame_count) && landmarkData?.available && !landmarkData.frames.has(state.currentFrameIndex)) {
      const firstLandmarkFrame = [...landmarkData.frames.keys()].sort((a, b) => a - b)[0];
      if (Number.isFinite(firstLandmarkFrame)) {
        seekToFrame(model, firstLandmarkFrame);
        return;
      }
    }
    renderLandmarkFrame(model, state.currentFrameIndex);
  });
  void ensureFullInferenceData(video).then((payload) => {
    if (fullInferenceLoadToken !== state.fullInferenceLoadToken) return;
    const blocks = els.videoStats.querySelectorAll(".video-stat-value");
    const labelBlocks = els.videoStats.querySelectorAll(".video-stat-label");
    labelBlocks.forEach((label, index) => {
      if (label.textContent === "Full Comparison" && blocks[index]) {
        blocks[index].textContent = `${payload?.frame_count || 0} frames`;
      }
    });
    syncFramePanels(model, state.currentFrameIndex, { forceTimeline: true });
  });
}

function statBlock(label, value) {
  return `
    <div>
      <span class="video-stat-label">${escapeHtml(label)}</span>
      <span class="video-stat-value">${escapeHtml(String(value))}</span>
    </div>
  `;
}

function renderTimeline(video) {
  const events = comparisonEvents(video);
  els.timelineCount.textContent = `${events.length} events`;
  if (!events.length) {
    els.timelineList.innerHTML = `<div class="empty-state">landmark 기반 전체-frame 추론 결과가 아직 없거나 로딩 중입니다.</div>`;
    return;
  }

  const items = events
    .map((event, index) => {
      const probabilityPreview = (event.probabilities || [])
        .slice()
        .sort((a, b) => b.value - a.value)
        .slice(0, 2)
        .map((entry) => `${entry.label} ${Math.round(entry.value * 100)}%`)
        .join(" · ");
      return `
        <div class="timeline-event ${event.is_mismatch ? "mismatch" : ""} ${index === state.activeEventIndex ? "active" : ""}" data-index="${index}">
          <div class="timeline-event-top">
            <span>${escapeHtml(event.timestamp || "-")} / #${escapeHtml(String(event.frame_idx ?? "-"))}</span>
            <span>${escapeHtml(event.predicted)}</span>
          </div>
          <div class="current-meta">
            GT ${escapeHtml(event.ground_truth)} · conf ${formatPercent(event.confidence)} · ${escapeHtml(probabilityPreview)}
          </div>
        </div>
      `;
    })
    .join("");
  els.timelineList.innerHTML = items;
  els.timelineList.querySelectorAll(".timeline-event").forEach((item) => {
    item.addEventListener("click", async () => {
      const model = await getModelData(state.selectedModelId);
      jumpToEvent(model, Number(item.dataset.index || 0));
    });
  });
}

function syncCurrentPrediction(video, index, frameIndex = null) {
  const event = comparisonEvents(video)[index];
  if (!event) {
    els.currentPrediction.innerHTML = `<div class="empty-state">선택된 이벤트가 없습니다. landmark 기반 전체-frame 추론 결과를 불러오는 중일 수 있습니다.</div>`;
    return;
  }

  const exactPredictionFrame = Number(event.frame_idx ?? frameIndex ?? 0) === Number(frameIndex ?? event.frame_idx ?? 0);
  const eventFrame = Number(event.frame_idx ?? frameIndex ?? 0);
  const playbackFrame = Number(frameIndex ?? eventFrame);
  const deltaFrames = playbackFrame - eventFrame;
  const deltaMs = Math.round((deltaFrames / Math.max(Number(video.fps || 30), 1)) * 1000);
  const relationLabel = exactPredictionFrame
    ? event.is_mismatch
      ? "Mismatch"
      : "Match"
    : `Nearest Event ${deltaFrames > 0 ? "+" : ""}${deltaFrames}f`;
  const sortedProbabilities = (event.probabilities || []).slice().sort((a, b) => b.value - a.value).slice(0, 5);
  els.currentPrediction.innerHTML = `
    <div class="timeline-event-top">
      <span>${escapeHtml(event.timestamp || "-")} / event #${escapeHtml(String(event.frame_idx ?? "-"))}</span>
      <span>${escapeHtml(relationLabel)}</span>
    </div>
    <p class="current-meta">
      playback frame ${escapeHtml(String(playbackFrame))} · event frame ${escapeHtml(String(eventFrame))} · delta ${escapeHtml(`${deltaFrames > 0 ? "+" : ""}${deltaFrames}f / ${deltaMs}ms`)} · GT ${escapeHtml(event.ground_truth)} · Pred ${escapeHtml(event.predicted)} · latency ${formatMs(event.latency_total_ms)}
    </p>
    <div class="probability-row">
      ${sortedProbabilities
        .map(
          (entry) => `
            <div class="probability-bar">
              <span>${escapeHtml(entry.label)}</span>
              <div class="probability-track"><div class="probability-fill" style="width: ${entry.value * 100}%"></div></div>
              <span>${Math.round(entry.value * 100)}%</span>
            </div>
          `,
        )
        .join("")}
    </div>
  `;
}

function renderPerClass(rows) {
  els.perClassBars.innerHTML = rows
    .map(
      (row) => `
        <div class="bar-item">
          <div class="bar-head">
            <span>${escapeHtml(String(row.class))}</span>
            <span>F1 ${formatNumber(row.f1)}</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width: ${Number(row.f1 || 0) * 100}%"></div></div>
          <div class="current-meta">precision ${formatNumber(row.precision)} · recall ${formatNumber(row.recall)} · support ${escapeHtml(String(row.support))}</div>
        </div>
      `,
    )
    .join("");
}

function linePath(values, width, height, minValue, maxValue) {
  if (!values.length) return "";
  const range = Math.max(maxValue - minValue, 1e-6);
  return values
    .map((value, index) => {
      const x = (index / Math.max(values.length - 1, 1)) * width;
      const y = height - ((value - minValue) / range) * height;
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function formatAxisTick(value) {
  const numeric = Number(value || 0);
  if (Math.abs(numeric) >= 1) return numeric.toFixed(1);
  if (Math.abs(numeric) >= 0.1) return numeric.toFixed(2);
  return numeric.toFixed(3);
}

function buildLinearTicks(minValue, maxValue, count) {
  if (count <= 1) return [minValue];
  const range = maxValue - minValue;
  if (Math.abs(range) < 1e-9) {
    return Array.from({ length: count }, (_, index) => minValue + index * 0.001);
  }
  return Array.from({ length: count }, (_, index) => minValue + (range * index) / (count - 1));
}

function buildEpochTickIndices(length, maxTicks = 5) {
  if (length <= 0) return [];
  if (length <= maxTicks) return Array.from({ length }, (_, index) => index);
  const ticks = new Set([0, length - 1]);
  for (let step = 1; step < maxTicks - 1; step += 1) {
    ticks.add(Math.round((step / (maxTicks - 1)) * (length - 1)));
  }
  return [...ticks].sort((a, b) => a - b);
}

function renderTrainingCurve(rows) {
  if (!rows.length) {
    els.trainingCurve.innerHTML = `<div class="empty-state">학습 이력이 없습니다.</div>`;
    return;
  }
  const width = 720;
  const height = 280;
  const margin = { top: 18, right: 18, bottom: 48, left: 62 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = height - margin.top - margin.bottom;
  const epochs = rows.map((row, index) => Number(row.epoch || index + 1));
  const trainLosses = rows.map((row) => Number(row.train_loss || 0));
  const valLosses = rows.map((row) => Number(row.val_loss || 0));
  const allLosses = trainLosses.concat(valLosses).filter((value) => Number.isFinite(value));
  const rawMin = Math.min(...allLosses);
  const rawMax = Math.max(...allLosses);
  const padding = Math.max((rawMax - rawMin) * 0.12, 0.002);
  const minLoss = Math.max(0, rawMin - padding);
  const maxLoss = rawMax + padding;
  const yTicks = buildLinearTicks(minLoss, maxLoss, 5);
  const xTickIndices = buildEpochTickIndices(epochs.length, 5);

  els.trainingCurve.innerHTML = `
    <div class="curve-labels">
      <span><i class="curve-label-dot train"></i>train loss</span>
      <span><i class="curve-label-dot val"></i>val loss</span>
    </div>
    <svg class="curve-svg" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none">
      ${yTicks
        .map((tick) => {
          const y = margin.top + plotHeight - ((tick - minLoss) / Math.max(maxLoss - minLoss, 1e-6)) * plotHeight;
          return `
            <line class="curve-grid-line" x1="${margin.left}" y1="${y.toFixed(2)}" x2="${(margin.left + plotWidth).toFixed(2)}" y2="${y.toFixed(2)}"></line>
            <text class="curve-tick-label" x="${(margin.left - 10).toFixed(2)}" y="${(y + 4).toFixed(2)}" text-anchor="end">${formatAxisTick(tick)}</text>
          `;
        })
        .join("")}
      ${xTickIndices
        .map((tickIndex) => {
          const x = margin.left + (tickIndex / Math.max(epochs.length - 1, 1)) * plotWidth;
          return `
            <line class="curve-grid-line vertical" x1="${x.toFixed(2)}" y1="${margin.top}" x2="${x.toFixed(2)}" y2="${(margin.top + plotHeight).toFixed(2)}"></line>
            <text class="curve-tick-label" x="${x.toFixed(2)}" y="${(margin.top + plotHeight + 22).toFixed(2)}" text-anchor="middle">${escapeHtml(String(epochs[tickIndex]))}</text>
          `;
        })
        .join("")}
      <line class="curve-axis-line" x1="${margin.left}" y1="${margin.top}" x2="${margin.left}" y2="${(margin.top + plotHeight).toFixed(2)}"></line>
      <line class="curve-axis-line" x1="${margin.left}" y1="${(margin.top + plotHeight).toFixed(2)}" x2="${(margin.left + plotWidth).toFixed(2)}" y2="${(margin.top + plotHeight).toFixed(2)}"></line>
      <g transform="translate(${margin.left}, ${margin.top})">
        <path d="${linePath(trainLosses, plotWidth, plotHeight, minLoss, maxLoss)}" fill="none" stroke="#cf6a32" stroke-width="4" stroke-linecap="round"></path>
        <path d="${linePath(valLosses, plotWidth, plotHeight, minLoss, maxLoss)}" fill="none" stroke="#0f766e" stroke-width="4" stroke-linecap="round"></path>
      </g>
      <text class="curve-axis-label" x="${width / 2}" y="${height - 6}" text-anchor="middle">epoch</text>
      <text class="curve-axis-label" x="18" y="${height / 2}" text-anchor="middle" transform="rotate(-90 18 ${height / 2})">loss</text>
    </svg>
  `;
}

function renderConfusionMatrix(confusionMatrix) {
  const labels = confusionMatrix.labels || [];
  const matrix = confusionMatrix.matrix || [];
  if (!labels.length || !matrix.length) {
    els.confusionMatrix.innerHTML = `<div class="empty-state">confusion matrix가 없습니다.</div>`;
    return;
  }

  const maxValue = Math.max(...matrix.flat(), 1);
  const head = labels.map((label) => `<th>${escapeHtml(label)}</th>`).join("");
  const body = matrix
    .map((row, rowIndex) => {
      const cells = row
        .map((value) => {
          const intensity = Number(value) / maxValue;
          const background = `rgba(15, 118, 110, ${Math.max(intensity, 0.08)})`;
          const textColor = intensity > 0.45 ? "#ffffff" : "#122117";
          return `<td style="background:${background};color:${textColor}">${escapeHtml(String(value))}</td>`;
        })
        .join("");
      return `<tr><th>${escapeHtml(labels[rowIndex])}</th>${cells}</tr>`;
    })
    .join("");

  els.confusionMatrix.innerHTML = `
    <table class="confusion-table">
      <thead><tr><th>GT \\ Pred</th>${head}</tr></thead>
      <tbody>${body}</tbody>
    </table>
  `;
}

function renderArtifacts(artifacts) {
  els.artifactLinks.innerHTML = Object.entries(artifacts)
    .map(
      ([label, href]) => `
        <a class="artifact-link" href="${escapeHtml(href)}" target="_blank" rel="noreferrer">
          <span>${escapeHtml(label)}</span>
          <span class="chip">open</span>
        </a>
      `,
    )
    .join("");
}

function loadReview() {
  const raw = localStorage.getItem(currentReviewKey());
  if (!raw) {
    els.reviewNotes.value = "";
    reviewCheckboxes.forEach((checkbox) => {
      checkbox.checked = false;
    });
    els.reviewStatus.textContent = "아직 저장되지 않았습니다.";
    return;
  }
  try {
    const review = JSON.parse(raw);
    els.reviewNotes.value = review.notes || "";
    reviewCheckboxes.forEach((checkbox) => {
      checkbox.checked = Boolean(review.checks?.[checkbox.id]);
    });
    els.reviewStatus.textContent = `마지막 저장: ${review.saved_at || "-"}`;
  } catch (error) {
    console.error(error);
    els.reviewStatus.textContent = "저장된 리뷰를 불러오지 못했습니다.";
  }
}

function saveReview() {
  const payload = {
    reviewer: currentReviewer(),
    model_id: state.selectedModelId,
    source_file: state.selectedSource,
    notes: els.reviewNotes.value,
    checks: Object.fromEntries(reviewCheckboxes.map((checkbox) => [checkbox.id, checkbox.checked])),
    saved_at: new Date().toLocaleString("ko-KR"),
  };
  localStorage.setItem(currentReviewKey(), JSON.stringify(payload));
  localStorage.setItem("jamjambeat-eval-reviewer", currentReviewer());
  els.reviewStatus.textContent = `저장 완료: ${payload.saved_at}`;
}

function clearReview() {
  localStorage.removeItem(currentReviewKey());
  loadReview();
}

function exportReviews() {
  const prefix = "jamjambeat-eval-review:";
  const bundle = {};
  for (let index = 0; index < localStorage.length; index += 1) {
    const key = localStorage.key(index);
    if (key && key.startsWith(prefix)) {
      bundle[key] = JSON.parse(localStorage.getItem(key));
    }
  }
  const blob = new Blob([JSON.stringify(bundle, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = `jamjambeat-reviews-${currentReviewer()}.json`;
  link.click();
  URL.revokeObjectURL(url);
}

async function importReviewsFromFile(file) {
  const text = await file.text();
  const payload = JSON.parse(text);
  Object.entries(payload).forEach(([key, value]) => {
    localStorage.setItem(key, JSON.stringify(value));
  });
  loadReview();
  els.reviewStatus.textContent = "리뷰 JSON을 가져왔습니다.";
}

function bindEvents() {
  els.modelSearch.addEventListener("input", () => {
    state.filters.search = els.modelSearch.value;
    renderModelList();
    renderLeaderboard();
  });

  els.suiteSelector.addEventListener("change", async () => {
    await loadSuiteIndex(els.suiteSelector.value);
  });

  els.userFilter.addEventListener("change", async () => {
    state.filters.user = els.userFilter.value;
    const model = await getModelData(state.selectedModelId);
    populateSourceOptions(model);
    renderSourceExplorer(model);
    hydrateVideoArea(model);
    loadReview();
  });

  els.motionFilter.addEventListener("change", async () => {
    state.filters.motion = els.motionFilter.value;
    const model = await getModelData(state.selectedModelId);
    populateSourceOptions(model);
    renderSourceExplorer(model);
    hydrateVideoArea(model);
    loadReview();
  });

  els.sideFilter.addEventListener("change", async () => {
    state.filters.side = els.sideFilter.value;
    const model = await getModelData(state.selectedModelId);
    populateSourceOptions(model);
    renderSourceExplorer(model);
    hydrateVideoArea(model);
    loadReview();
  });

  els.sourceFilter.addEventListener("change", async () => {
    state.selectedSource = els.sourceFilter.value;
    const model = await getModelData(state.selectedModelId);
    populateSourceOptions(model);
    hydrateVideoArea(model);
    loadReview();
  });

  els.reviewerName.addEventListener("change", () => {
    loadReview();
  });

  els.saveReview.addEventListener("click", saveReview);
  els.clearReview.addEventListener("click", clearReview);
  els.exportReview.addEventListener("click", exportReviews);
  els.importReview.addEventListener("click", () => {
    els.importReviewFile.click();
  });
  els.importReviewFile.addEventListener("change", async () => {
    const file = els.importReviewFile.files?.[0];
    if (!file) return;
    try {
      await importReviewsFromFile(file);
    } catch (error) {
      console.error(error);
      els.reviewStatus.textContent = "리뷰 JSON을 가져오지 못했습니다.";
    } finally {
      els.importReviewFile.value = "";
    }
  });
  els.reloadData.addEventListener("click", async () => {
    state.modelCache.clear();
    state.landmarkCache.clear();
    state.landmarkPromiseCache.clear();
    await loadCatalog();
  });

  els.frameJumpBack.addEventListener("click", async () => {
    const model = await getModelData(state.selectedModelId);
    stepFrame(model, -10);
    focusFrameAnalysisShell();
  });
  els.framePrev.addEventListener("click", async () => {
    const model = await getModelData(state.selectedModelId);
    stepFrame(model, -1);
    focusFrameAnalysisShell();
  });
  els.framePrevEvent.addEventListener("click", async () => {
    const model = await getModelData(state.selectedModelId);
    jumpRelativeToCurrentEvent(model, -1);
    focusFrameAnalysisShell();
  });
  els.framePlayPause.addEventListener("click", () => {
    if (els.videoPlayer.paused) {
      els.videoPlayer.play();
    } else {
      els.videoPlayer.pause();
    }
    focusFrameAnalysisShell();
  });
  els.frameNextEvent.addEventListener("click", async () => {
    const model = await getModelData(state.selectedModelId);
    jumpRelativeToCurrentEvent(model, 1);
    focusFrameAnalysisShell();
  });
  els.frameNext.addEventListener("click", async () => {
    const model = await getModelData(state.selectedModelId);
    stepFrame(model, 1);
    focusFrameAnalysisShell();
  });
  els.frameJumpForward.addEventListener("click", async () => {
    const model = await getModelData(state.selectedModelId);
    stepFrame(model, 10);
    focusFrameAnalysisShell();
  });
  els.frameRestart.addEventListener("click", async () => {
    const model = await getModelData(state.selectedModelId);
    els.videoPlayer.pause();
    seekToFrame(model, 0);
    focusFrameAnalysisShell();
  });
  els.frameScrubber.addEventListener("pointerdown", beginScrubInteraction);
  els.frameScrubber.addEventListener("pointerup", finishScrubInteraction);
  els.frameScrubber.addEventListener("pointercancel", finishScrubInteraction);
  els.frameScrubber.addEventListener("change", finishScrubInteraction);
  els.frameScrubber.addEventListener("blur", finishScrubInteraction);
  els.frameScrubber.addEventListener("input", handleFrameScrubberInput);
  els.frameAnalysisShell.addEventListener("click", () => {
    focusFrameAnalysisShell();
  });

  window.addEventListener("keydown", handleFrameShortcutKeydown, { capture: true });

  els.videoPlayer.addEventListener("timeupdate", async () => {
    if (typeof els.videoPlayer.requestVideoFrameCallback === "function") return;
    const model = await getModelData(state.selectedModelId);
    syncFrameState(model);
  });
  els.videoPlayer.addEventListener("seeked", async () => {
    const model = await getModelData(state.selectedModelId);
    if (state.pendingSeekFrame !== null) {
      state.currentFrameIndex = state.pendingSeekFrame;
      state.pendingSeekFrame = null;
      state.frameSyncReady = true;
      syncFramePanels(model, state.currentFrameIndex, { forceTimeline: true });
    } else {
      syncFrameState(model);
    }
    if (!els.videoPlayer.paused) {
      startPlaybackFrameLoop();
    }
  });
  els.videoPlayer.addEventListener("play", async () => {
    const model = await getModelData(state.selectedModelId);
    updateFrameScrubber(model);
    startPlaybackFrameLoop();
  });
  els.videoPlayer.addEventListener("pause", async () => {
    cancelPlaybackFrameLoop();
    const model = await getModelData(state.selectedModelId);
    updateFrameScrubber(model);
  });
  els.videoPlayer.addEventListener("ended", () => {
    cancelPlaybackFrameLoop();
  });

  window.addEventListener("resize", async () => {
    if (!state.selectedModelId) return;
    const model = await getModelData(state.selectedModelId);
    renderLandmarkFrame(model, state.currentFrameIndex);
  });
}

async function boot() {
  const storedReviewer = localStorage.getItem("jamjambeat-eval-reviewer");
  if (storedReviewer) {
    els.reviewerName.value = storedReviewer;
  }
  bindEvents();
  try {
    await loadCatalog();
  } catch (error) {
    console.error(error);
    document.body.innerHTML = `
      <div class="main-content">
        <div class="panel">
          <h2>대시보드를 로드하지 못했습니다.</h2>
          <p>${escapeHtml(error.message)}</p>
          <p class="hint">먼저 <code>uv run python model/frontend/eval_dashboard/generate_dashboard_data.py</code> 를 실행해 주세요.</p>
        </div>
      </div>
    `;
  }
}

boot();
