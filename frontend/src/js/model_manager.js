// [model_manager.js] 모델 동적 로딩 및 전환 관리

const MODEL_CONFIGS = {
  "frame_basic": {
    id: "frame_basic",
    name: "기본 프레임",
    description: "단일 프레임",
    mode: "frame",
    importPath: "./model_inference_onnx.js"
  },
  "sequence_delta": {
    id: "sequence_delta",
    name: "시퀀스 델타",
    description: "8프레임 버퍼",
    mode: "sequence",
    importPath: "./model_inference_sequence.js"
  },
  "frame_spatial_transformer": {
    id: "frame_spatial_transformer",
    name: "Spatial Transformer",
    description: "최신 모델 (추천)",
    mode: "frame",
    importPath: "./model_inference_frame_spatial_transformer.js"
  }
};

const DEFAULT_MODEL_ID = "frame_spatial_transformer";
const STORAGE_KEY = "jamjam:selected-model";

let currentModelId = null;
let currentModelApi = null;
let isLoading = false;

// 저장된 모델 ID 불러오기
function loadSavedModelId() {
  try {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved && MODEL_CONFIGS[saved]) {
      return saved;
    }
  } catch {
    // LocalStorage 접근 실패 시 기본값 사용
  }
  return DEFAULT_MODEL_ID;
}

// 모델 ID 저장
function saveModelId(modelId) {
  try {
    localStorage.setItem(STORAGE_KEY, modelId);
  } catch {
    // 저장 실패는 무시 (다음 세션에서 기본값 사용)
  }
}

// 현재 활성 모델 API 가져오기
export function getCurrentModelApi() {
  return currentModelApi;
}

// 현재 활성 모델 ID 가져오기
export function getCurrentModelId() {
  return currentModelId || loadSavedModelId();
}

// 모델 설정 정보 가져오기
export function getModelConfig(modelId) {
  return MODEL_CONFIGS[modelId] || null;
}

// 사용 가능한 모든 모델 목록
export function getAllModelConfigs() {
  return Object.values(MODEL_CONFIGS);
}

// 모델 동적 로딩
async function loadModelModule(modelId) {
  const config = MODEL_CONFIGS[modelId];
  if (!config) {
    throw new Error(`Unknown model ID: ${modelId}`);
  }

  console.info(`[ModelManager] Loading model: ${config.name} (${modelId})`);

  try {
    const module = await import(/* @vite-ignore */ config.importPath);

    // 필수 API 검증
    if (typeof module.getModelPrediction !== "function") {
      throw new Error(`Model ${modelId} does not export getModelPrediction`);
    }
    if (typeof module.getModelInferenceStatus !== "function") {
      throw new Error(`Model ${modelId} does not export getModelInferenceStatus`);
    }

    console.info(`[ModelManager] ✅ Model loaded: ${config.name}`);
    return module;
  } catch (error) {
    console.error(`[ModelManager] ❌ Failed to load model ${modelId}:`, error);
    throw error;
  }
}

// 모델 전환
export async function switchModel(modelId) {
  if (isLoading) {
    console.warn("[ModelManager] Model loading in progress, please wait");
    return false;
  }

  if (modelId === currentModelId && currentModelApi) {
    console.info("[ModelManager] Model already active:", modelId);
    return true;
  }

  const config = MODEL_CONFIGS[modelId];
  if (!config) {
    console.error("[ModelManager] Invalid model ID:", modelId);
    return false;
  }

  isLoading = true;
  try {
    // 이벤트 발송: 모델 로딩 시작
    window.dispatchEvent(new CustomEvent("jamjam:model-loading", {
      detail: { modelId, config }
    }));

    const modelModule = await loadModelModule(modelId);

    // 모델 사전 로딩 (있으면)
    if (typeof modelModule.preloadModel === "function") {
      await modelModule.preloadModel();
    }

    currentModelId = modelId;
    currentModelApi = modelModule;
    saveModelId(modelId);

    // 이벤트 발송: 모델 로딩 완료
    window.dispatchEvent(new CustomEvent("jamjam:model-loaded", {
      detail: { modelId, config, api: currentModelApi }
    }));

    console.info(`[ModelManager] ✅ Model switched to: ${config.name}`);
    return true;
  } catch (error) {
    console.error(`[ModelManager] ❌ Failed to switch model:`, error);

    // 이벤트 발송: 모델 로딩 실패
    window.dispatchEvent(new CustomEvent("jamjam:model-load-error", {
      detail: { modelId, config, error }
    }));

    return false;
  } finally {
    isLoading = false;
  }
}

// 초기화: 저장된 모델 또는 기본 모델 로드
export async function initializeDefaultModel() {
  const savedModelId = loadSavedModelId();
  console.info(`[ModelManager] Initializing with model: ${savedModelId}`);
  return await switchModel(savedModelId);
}

// 모델 로딩 상태 확인
export function isModelLoading() {
  return isLoading;
}
