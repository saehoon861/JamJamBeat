// inference.ts - browser-only ONNX runtime for the pos_scale sequence-delta bundle
import * as ort from "onnxruntime-web";

export type PredictorStatus =
  | "warmup"
  | "no_hand"
  | "ready"
  | "tau_neutralized"
  | "invalid_frame";

export type RawFrame = number[] | Float32Array | ArrayLike<number> | ArrayLike<ArrayLike<number>>;

export interface PredictionResult {
  status: PredictorStatus;
  predIndex: number;
  predLabel: string;
  confidence: number;
  probs: number[];
  rawPredIndex: number;
  rawPredLabel: string;
  tauApplied: number;
  tauNeutralized: boolean;
  framesCollected: number;
}

export interface SequenceDeltaPredictorOptions {
  modelUrl?: string | URL;
  configUrl?: string | URL;
  classNamesUrl?: string | URL;
  tau?: number;
  sessionOptions?: ort.InferenceSession.SessionOptions;
}

interface BundleConfig {
  bundle_id: string;
  model_id: string;
  mode: string;
  dataset_key: string;
  normalization_family: string;
  input_type: string;
  caller_input_type: string;
  input_dim: number;
  seq_len: number;
  num_classes: number;
  neutral_index: number;
  default_tau: number | null;
  supported_backends?: string[];
  streaming_supported?: boolean;
  no_hand_resets_buffer?: boolean;
  preprocess?: {
    normalization?: string;
    delta?: boolean;
    delta_order?: string;
    seq_len?: number;
    allowed_frame_shapes?: number[][];
    formula?: string;
    eps?: number;
  };
}

const DEFAULT_TAU = 0.85;
const DEFAULT_EPS = 1e-8;
const DEFAULT_EXECUTION_PROVIDERS: ort.InferenceSession.SessionOptions["executionProviders"] = ["wasm"];

function assetUrl(relativePath: string): string {
  return new URL(relativePath, import.meta.url).toString();
}

function resolveUrl(value: string | URL | undefined, fallbackRelativePath: string): string {
  if (value instanceof URL) {
    return value.toString();
  }
  if (typeof value === "string" && value.trim()) {
    return value;
  }
  return assetUrl(fallbackRelativePath);
}

async function fetchJson<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: ${response.status} ${response.statusText}`);
  }
  return (await response.json()) as T;
}

function neutralProbs(classCount: number, neutralIndex: number): number[] {
  const probs = new Array<number>(classCount).fill(0);
  if (neutralIndex >= 0 && neutralIndex < classCount) {
    probs[neutralIndex] = 1;
  }
  return probs;
}

function softmax(values: ArrayLike<number>): number[] {
  const arr = Array.from(values, (value) => Number(value));
  const maxValue = Math.max(...arr);
  const exps = arr.map((value) => Math.exp(value - maxValue));
  const sumExp = exps.reduce((total, value) => total + value, 0);
  return exps.map((value) => value / sumExp);
}

function validateTau(tau: number): number {
  if (!Number.isFinite(tau) || tau <= 0 || tau > 1) {
    throw new Error(`tau must be in the range (0, 1], got ${tau}`);
  }
  return tau;
}

function isTuple3(value: unknown): value is ArrayLike<number> {
  if (value == null || typeof value !== "object") {
    return false;
  }
  return (value as ArrayLike<unknown>).length === 3;
}

function coerceRawFrame(rawFrame: RawFrame): Float32Array {
  const direct = Array.from(rawFrame as ArrayLike<number>, (value) => Number(value));
  if (direct.length === 63 && direct.every((value) => Number.isFinite(value))) {
    return Float32Array.from(direct);
  }

  const rows = rawFrame as ArrayLike<ArrayLike<number>>;
  if (rows != null && typeof rows === "object" && rows.length === 21) {
    const flattened: number[] = [];
    for (let rowIndex = 0; rowIndex < rows.length; rowIndex += 1) {
      const row = rows[rowIndex];
      if (!isTuple3(row)) {
        throw new Error("Each landmark row must have shape [3]");
      }
      flattened.push(Number(row[0]), Number(row[1]), Number(row[2]));
    }
    if (flattened.every((value) => Number.isFinite(value))) {
      return Float32Array.from(flattened);
    }
  }

  throw new Error("Raw frame must be shape [63] or [21,3] with finite numeric values");
}

function assertFiniteFrame(frame: Float32Array): void {
  for (let index = 0; index < frame.length; index += 1) {
    if (!Number.isFinite(frame[index])) {
      throw new Error(`Frame contains non-finite value at index ${index}`);
    }
  }
}

function normalizePosScaleFrame(frame: Float32Array, eps: number): Float32Array {
  const normalized = new Float32Array(63);
  const originX = frame[0];
  const originY = frame[1];
  const originZ = frame[2];

  const dx = frame[27] - originX;
  const dy = frame[28] - originY;
  const dz = frame[29] - originZ;
  const denom = Math.hypot(dx, dy, dz);
  const scale = denom <= eps ? 1 : 1 / denom;

  for (let landmarkIndex = 0; landmarkIndex < 63; landmarkIndex += 3) {
    normalized[landmarkIndex] = (frame[landmarkIndex] - originX) * scale;
    normalized[landmarkIndex + 1] = (frame[landmarkIndex + 1] - originY) * scale;
    normalized[landmarkIndex + 2] = (frame[landmarkIndex + 2] - originZ) * scale;
  }
  return normalized;
}

function buildFeatureTensor(buffer: Float32Array[], seqLen: number): Float32Array {
  if (buffer.length !== seqLen) {
    throw new Error(`Expected ${seqLen} buffered frames, got ${buffer.length}`);
  }

  const featureTensor = new Float32Array(seqLen * 126);
  for (let t = 0; t < seqLen; t += 1) {
    const baseOffset = t * 126;
    const current = buffer[t];
    const previous = t > 0 ? buffer[t - 1] : null;
    for (let featureIndex = 0; featureIndex < 63; featureIndex += 1) {
      const baseValue = current[featureIndex];
      featureTensor[baseOffset + featureIndex] = baseValue;
      featureTensor[baseOffset + 63 + featureIndex] =
        previous === null ? 0 : baseValue - previous[featureIndex];
    }
  }
  return featureTensor;
}

function validateBundleConfig(config: BundleConfig, classNames: string[]): void {
  if (config.model_id !== "mlp_sequence_delta") {
    throw new Error(`Unexpected model_id: ${config.model_id}`);
  }
  if (config.mode !== "sequence") {
    throw new Error(`Unexpected mode: ${config.mode}`);
  }
  if (config.normalization_family !== "pos_scale") {
    throw new Error(`Unexpected normalization family: ${config.normalization_family}`);
  }
  if (config.seq_len !== 8) {
    throw new Error(`Expected seq_len=8, got ${config.seq_len}`);
  }
  if (config.input_dim !== 126) {
    throw new Error(`Expected input_dim=126, got ${config.input_dim}`);
  }
  if (classNames.length !== config.num_classes) {
    throw new Error(
      `class_names length mismatch: expected ${config.num_classes}, got ${classNames.length}`,
    );
  }
}

export class SequenceDeltaPredictor {
  private readonly options: SequenceDeltaPredictorOptions;
  private readonly jointBuffer: Float32Array[] = [];
  private session: ort.InferenceSession | null = null;
  private config: BundleConfig | null = null;
  private classNames: string[] = [];
  private tau = DEFAULT_TAU;
  private eps = DEFAULT_EPS;

  constructor(options: SequenceDeltaPredictorOptions = {}) {
    this.options = options;
  }

  async load(): Promise<void> {
    const configUrl = resolveUrl(this.options.configUrl, "./config.json");
    const classNamesUrl = resolveUrl(this.options.classNamesUrl, "./class_names.json");
    const modelUrl = resolveUrl(this.options.modelUrl, "./model.onnx");

    const [config, classNames] = await Promise.all([
      fetchJson<BundleConfig>(configUrl),
      fetchJson<string[]>(classNamesUrl),
    ]);

    validateBundleConfig(config, classNames);

    this.config = config;
    this.classNames = classNames;
    this.tau = validateTau(this.options.tau ?? config.default_tau ?? DEFAULT_TAU);
    this.eps = Number(config.preprocess?.eps ?? DEFAULT_EPS);
    this.session = await ort.InferenceSession.create(
      modelUrl,
      this.options.sessionOptions ?? { executionProviders: DEFAULT_EXECUTION_PROVIDERS },
    );
  }

  reset(): void {
    this.jointBuffer.length = 0;
  }

  pushNoHand(): PredictionResult {
    this.ensureLoaded();
    this.reset();
    return this.buildNeutralResult("no_hand");
  }

  async pushFrame(rawFrame: RawFrame): Promise<PredictionResult> {
    this.ensureLoaded();

    try {
      const flatFrame = coerceRawFrame(rawFrame);
      assertFiniteFrame(flatFrame);
      const joint63 = normalizePosScaleFrame(flatFrame, this.eps);

      if (this.jointBuffer.length === this.config!.seq_len) {
        this.jointBuffer.shift();
      }
      this.jointBuffer.push(joint63);

      if (this.jointBuffer.length < this.config!.seq_len) {
        return this.buildNeutralResult("warmup");
      }

      const features = buildFeatureTensor(this.jointBuffer, this.config!.seq_len);
      const tensor = new ort.Tensor("float32", features, [1, this.config!.seq_len, this.config!.input_dim]);
      const inputName = this.session!.inputNames[0];
      const outputName = this.session!.outputNames[0];
      const outputs = await this.session!.run({ [inputName]: tensor });
      const logits = outputs[outputName];

      if (!logits || !("data" in logits)) {
        throw new Error("ONNX output did not contain logits data");
      }

      return this.postprocessLogits(logits.data as ArrayLike<number>);
    } catch (_error) {
      this.reset();
      return this.buildNeutralResult("invalid_frame");
    }
  }

  getFramesCollected(): number {
    return this.jointBuffer.length;
  }

  private ensureLoaded(): void {
    if (this.session === null || this.config === null || this.classNames.length === 0) {
      throw new Error("SequenceDeltaPredictor is not loaded. Call await predictor.load() first.");
    }
  }

  private buildNeutralResult(status: "warmup" | "no_hand" | "invalid_frame"): PredictionResult {
    const neutralIndex = this.config!.neutral_index;
    const neutralLabel = this.classNames[neutralIndex];
    return {
      status,
      predIndex: neutralIndex,
      predLabel: neutralLabel,
      confidence: 0,
      probs: neutralProbs(this.classNames.length, neutralIndex),
      rawPredIndex: neutralIndex,
      rawPredLabel: neutralLabel,
      tauApplied: this.tau,
      tauNeutralized: false,
      framesCollected: this.jointBuffer.length,
    };
  }

  private postprocessLogits(logits: ArrayLike<number>): PredictionResult {
    const probs = softmax(logits);
    const rawPredIndex = probs.reduce(
      (bestIndex, value, index, values) => (value > values[bestIndex] ? index : bestIndex),
      0,
    );
    const confidence = probs[rawPredIndex];
    const neutralIndex = this.config!.neutral_index;
    const tauNeutralized = rawPredIndex !== neutralIndex && confidence < this.tau;
    const predIndex = tauNeutralized ? neutralIndex : rawPredIndex;

    return {
      status: tauNeutralized ? "tau_neutralized" : "ready",
      predIndex,
      predLabel: this.classNames[predIndex],
      confidence,
      probs,
      rawPredIndex,
      rawPredLabel: this.classNames[rawPredIndex],
      tauApplied: this.tau,
      tauNeutralized,
      framesCollected: this.jointBuffer.length,
    };
  }
}

export function createSequenceDeltaPredictor(
  options: SequenceDeltaPredictorOptions = {},
): SequenceDeltaPredictor {
  return new SequenceDeltaPredictor(options);
}
