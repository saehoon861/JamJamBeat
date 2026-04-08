# `runtime_frame_spatial_transformer` 전달 문서

대상 구조는 이 프로젝트의 `frontend`와 비슷한 형태를 가정합니다.

- MediaPipe로 손 landmark를 얻는다
- 브라우저에서 `onnxruntime-web`으로 ONNX 추론을 한다
- 결과를 gesture label과 confidence로 후처리한다

이 모델은 sequence 모델이 아니라 single-frame 모델입니다. 즉 시간축 버퍼 8프레임을 모아서 넣는 방식이 아니라, 매 프레임마다 `21 x 3` landmark 하나를 바로 추론합니다.

## 1. 이 폴더에서 전달해야 하는 핵심 파일

이 `frame_spatial_transformer` 폴더 안에서 실제 source-of-truth로 보면 되는 파일은 아래입니다.

- `config.json`
- `model.onnx`
- `model.onnx.data`
- `dataset.py`
- `학습모델_웹배포_정리.md`
- `123.png`

의미는 다음과 같습니다.

- `config.json`
  - 클래스 순서와 전처리 방식의 기준
- `model.onnx`
  - 실제 브라우저 추론 모델
- `model.onnx.data`
  - ONNX external data
  - `model.onnx`와 같은 runtime root에 반드시 같이 있어야 함
- `dataset.py`
  - 학습 시 landmark를 어떻게 정규화했는지 확인하는 기준 코드
- `학습모델_웹배포_정리.md`
  - 웹 배포 기준 설명 문서
- `123.png`
  - 현재 테스트 때 사용한 운영 파라미터 스냅샷

## 2. 모델 계약 요약

이 모델은 아래 계약으로 이해하면 됩니다.

- 모델 종류: single-frame ONNX
- 입력 landmark: MediaPipe normalized landmarks 21개
- 좌표: `(x, y, z)`
- 입력 shape: `[1, 21, 3]`
- 입력 dtype: `float32`
- 출력 shape: `[1, 7]`
- 출력 의미: 7개 gesture class에 대한 logits
- 클래스 순서:
  1. `neutral`
  2. `fist`
  3. `open_palm`
  4. `V`
  5. `pinky`
  6. `animal`
  7. `k-heart`

중요한 점:

- 이 모델은 sequence length, warmup window, temporal padding이 필요하지 않습니다.
- `effectiveSeqLen`, `framesCollected` 같은 개념은 실제 모델 입력에는 적용되지 않습니다.
- 프레임 하나가 준비되면 바로 추론하면 됩니다.

## 3. 전처리 계약

이 모델의 전처리는 `pos_scale` 입니다.

기준은 `config.json`과 `dataset.py`이며 공식은 아래와 같습니다.

```text
(pts - pts[0]) / max(||pts[9] - pts[0]||, eps)
```

설명:

- `pts[0]` = wrist landmark
- `pts[9]` = middle finger MCP landmark
- 모든 점에서 wrist를 원점으로 빼고
- `pts[9] - pts[0]` 거리로 전체 스케일을 정규화합니다
- `eps`는 0 나눗셈 방지용 작은 값입니다

현재 runtime에서는 아래 값으로 쓰면 됩니다.

```js
const POS_SCALE_EPS = 1e-8;
const ORIGIN_IDX = 0;
const SCALE_IDX = 9;
```

주의:

- MediaPipe `worldLandmarks`를 쓰면 안 됩니다.
- MediaPipe `normalized landmarks`를 그대로 사용해야 합니다.
- landmark 순서는 MediaPipe hand landmark 기본 순서를 반드시 유지해야 합니다.
- 입력은 `21 x 3` 이어야 하며, landmark 하나라도 누락되면 no-hand 또는 invalid frame 처리하는 쪽이 안전합니다.

## 4. 브라우저 runtime bundle 재구성 방법

상대 프로젝트가 `frontend`와 유사한 구조라면, 일반적으로 `public/runtime_frame_spatial_transformer/` 같은 폴더를 만들어 배치하면 됩니다.

예시:

```text
public/
  runtime_frame_spatial_transformer/
    model.onnx
    model.onnx.data
    config.json
    class_names.json
    input_spec.json
```

여기서 `model.onnx`와 `model.onnx.data`는 이 `frame_spatial_transformer` 폴더의 파일을 그대로 복사하면 됩니다.

### 4-1. runtime `config.json` 예시

아래 JSON은 현재 브라우저 runtime에서 실제로 사용한 형태입니다.

```json
{
  "bundle_id": "runtime_frame_spatial_transformer",
  "model_id": "Landmark_Spatial_Transformer",
  "mode": "frame",
  "dataset_key": "pos_scale_frame",
  "normalization_family": "pos_scale",
  "input_type": "model_input_frame_joint21_xyz3_pos_scale",
  "caller_input_type": "raw_mediapipe_landmark_frame",
  "input_shape": [21, 3],
  "input_dim": 63,
  "num_classes": 7,
  "neutral_index": 0,
  "default_device": "cpu",
  "default_tau": null,
  "class_names": [
    "neutral",
    "fist",
    "open_palm",
    "V",
    "pinky",
    "animal",
    "k-heart"
  ],
  "preprocess": {
    "normalization": "pos_scale",
    "allowed_frame_shapes": [
      [63],
      [21, 3]
    ],
    "formula": "(pts - pts[0]) / max(||pts[9]-pts[0]||, eps)",
    "origin_idx": 0,
    "scale_idxs": [0, 9],
    "eps": 1e-8
  }
}
```

### 4-2. `class_names.json` 예시

`config.json.classes` 또는 `config.json.class_names`를 source-of-truth로 써도 되지만, runtime 단에서 별도 파일을 두는 편이 더 안전합니다.

```json
[
  "neutral",
  "fist",
  "open_palm",
  "V",
  "pinky",
  "animal",
  "k-heart"
]
```

### 4-3. `input_spec.json` 예시

```json
{
  "name": "raw_mediapipe_landmark_frame",
  "shape": [63],
  "dtype": "float32",
  "layout": "frame",
  "caller_input_shapes": [
    [63],
    [21, 3]
  ],
  "model_input_shape": [21, 3],
  "description": "Caller provides a single raw MediaPipe landmark frame. The frontend frame runtime applies pos_scale normalization and reshapes the result to [1,21,3] before ONNX inference.",
  "notes": [
    "Input must contain exactly 21 landmarks in MediaPipe normalized landmark order.",
    "The runtime uses normalized landmarks, not world landmarks.",
    "pos_scale preprocessing is (pts - pts[0]) / max(||pts[9]-pts[0]||, eps).",
    "This bundle is a single-frame model with no warmup window or temporal padding."
  ]
}
```

## 5. 브라우저 추론 방식 제안

이 모델은 브라우저에서 `onnxruntime-web` + `wasm` execution provider 조합으로 붙이는 것이 가장 단순합니다.

권장 원칙:

- device는 기본 `CPU/wasm`
- `inputNames[0]`, `outputNames[0]`를 동적으로 읽기
- 매 프레임마다 바로 추론 가능
- 하지만 호출 폭주를 막기 위해 `modelIntervalMs` 게이트는 두는 것이 좋음

### 5-1. ONNX 세션 생성 예시

```js
import * as ort from "onnxruntime-web";

const RUNTIME_ROOT = "/runtime_frame_spatial_transformer/";
const MODEL_PATH = RUNTIME_ROOT + "model.onnx";
const MODEL_EXTERNAL_DATA_PATH = RUNTIME_ROOT + "model.onnx.data";

async function createFrameSession() {
  if (ort?.env?.wasm) {
    ort.env.wasm.wasmPaths = "/";
    if (typeof SharedArrayBuffer === "undefined") {
      ort.env.wasm.numThreads = 1;
    }
  }

  const session = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
    externalData: [
      {
        path: "model.onnx.data",
        data: MODEL_EXTERNAL_DATA_PATH
      }
    ]
  });

  return session;
}
```

핵심:

- 이 모델은 `model.onnx.data`를 반드시 같이 읽어야 합니다.
- `model.onnx`만 두면 로드 실패할 수 있습니다.

### 5-2. MediaPipe landmark를 raw frame feature로 정리하는 예시

입력은 `21`개 landmark 각각의 `x, y, z`를 그대로 순서대로 펼치면 됩니다.

```js
function sanitizeLandmarksTo63(landmarks) {
  if (!Array.isArray(landmarks) || landmarks.length < 21) return null;

  const features = new Float32Array(63);
  for (let i = 0; i < 21; i += 1) {
    const p = landmarks[i];
    const offset = i * 3;
    features[offset] = Number.isFinite(p?.x) ? p.x : 0;
    features[offset + 1] = Number.isFinite(p?.y) ? p.y : 0;
    features[offset + 2] = Number.isFinite(p?.z) ? p.z : 0;
  }
  return features;
}
```

### 5-3. `pos_scale` 정규화 예시

```js
const POS_SCALE_EPS = 1e-8;

function normalizePosScale(features63) {
  if (!features63 || features63.length !== 63) return null;

  const originX = features63[0];
  const originY = features63[1];
  const originZ = features63[2];

  const scaleX = features63[27] - originX;
  const scaleY = features63[28] - originY;
  const scaleZ = features63[29] - originZ;
  const denom = Math.hypot(scaleX, scaleY, scaleZ);
  const scale = denom <= POS_SCALE_EPS ? 1 : 1 / denom;

  const normalized = new Float32Array(63);
  for (let i = 0; i < 63; i += 3) {
    normalized[i] = (features63[i] - originX) * scale;
    normalized[i + 1] = (features63[i + 1] - originY) * scale;
    normalized[i + 2] = (features63[i + 2] - originZ) * scale;
  }
  return normalized;
}
```

### 5-4. softmax / argmax 예시

```js
function softmax(logits) {
  const maxLogit = Math.max(...logits);
  const exps = logits.map((v) => Math.exp(v - maxLogit));
  const sum = exps.reduce((acc, v) => acc + v, 0);
  return exps.map((v) => v / sum);
}

function getTopClass(probs) {
  let bestIndex = 0;
  let bestValue = probs[0] ?? 0;
  for (let i = 1; i < probs.length; i += 1) {
    if (probs[i] > bestValue) {
      bestValue = probs[i];
      bestIndex = i;
    }
  }
  return { index: bestIndex, confidence: bestValue };
}
```

### 5-5. 실제 frame inference 함수 예시

아래 함수 하나면 상대 프로젝트 AI가 기본 추론 경로를 바로 연결할 수 있습니다.

```js
const CLASS_NAMES = [
  "neutral",
  "fist",
  "open_palm",
  "V",
  "pinky",
  "animal",
  "k-heart"
];

function createNoHandPrediction() {
  return {
    label: "neutral",
    classId: 0,
    confidence: 0,
    probs: [1, 0, 0, 0, 0, 0, 0],
    status: "no_hand",
    mode: "frame",
    framesCollected: 0,
    effective_seq_len: 1,
    model_seq_len: 1,
    padded_seq_input: false
  };
}

async function runFrameInference(session, landmarks) {
  if (!landmarks || !Array.isArray(landmarks) || landmarks.length < 21) {
    return createNoHandPrediction();
  }

  const raw63 = sanitizeLandmarksTo63(landmarks);
  const normalized63 = normalizePosScale(raw63);
  if (!normalized63) {
    return createNoHandPrediction();
  }

  const inputName = session.inputNames[0];
  const outputName = session.outputNames[0];

  const tensor = new ort.Tensor("float32", normalized63, [1, 21, 3]);
  const results = await session.run({ [inputName]: tensor });
  const outputTensor = results[outputName] || results[Object.keys(results)[0]];
  const logits = Array.from(outputTensor.data);
  const probs = softmax(logits);
  const top = getTopClass(probs);

  return {
    label: CLASS_NAMES[top.index] || "neutral",
    classId: top.index,
    confidence: top.confidence,
    probs,
    status: "ready",
    mode: "frame",
    framesCollected: 1,
    effective_seq_len: 1,
    model_seq_len: 1,
    padded_seq_input: false
  };
}
```

### 5-6. `modelIntervalMs` 게이트 예시

이 모델은 frame-based라서 warmup은 없지만, 브라우저 메인 스레드에 과도한 부하를 주지 않으려면 최소 호출 간격은 두는 편이 좋습니다.

`123.png` 기준 운영값은 `modelIntervalMs = 60` 입니다.

```js
let lastRequestAt = 0;
let requestInFlight = false;
const MODEL_INTERVAL_MS = 60;

async function maybeRunFrameInference(session, landmarks, now = performance.now()) {
  if (requestInFlight) return null;
  if (now - lastRequestAt < MODEL_INTERVAL_MS) return null;

  requestInFlight = true;
  lastRequestAt = now;
  try {
    return await runFrameInference(session, landmarks);
  } finally {
    requestInFlight = false;
  }
}
```

## 6. `123.png` 기준 운영 설정

이 값들은 모델 내부 파라미터가 아니라, 현재 테스트 환경에서 사용한 운영값입니다. 상대 프로젝트에서도 우선 같은 값으로 맞춘 뒤 조정하는 것을 권장합니다.

### 6-1. capture / detection

- `inferWidth = 0`
  - 의미: full frame 사용
- `inferFps = 30`
- `numHands = 1`
- `minHandDetectionConfidence = 0.50`
- `minHandPresenceConfidence = 0.50`
- `minTrackingConfidence = 0.50`

### 6-2. model trigger

- `seqFrames = 1 (frame)`
- `modelIntervalMs = 60`

참고:

- 스크린샷에 `effectiveSeqLen = 4`가 보이더라도, frame model에서는 disabled 상태였고 실제 모델 입력에는 사용되지 않았습니다.
- frame model에서는 `effectiveSeqLen`을 무시하거나 강제로 `1`로 보는 것이 맞습니다.

### 6-3. 후처리 / 안정화

- `tau = 0.65`
- `vote = 1`
- `debounce = 1`
- `clear = 1`
- `motion window = 2`
- `motion threshold = 0.028`

의미를 분리해서 이해하면 좋습니다.

- `tau`
  - 모델이 학습해서 내장한 값이 아니라 상위 앱의 decision threshold
- `vote`, `debounce`
  - label을 얼마나 안정화할지 정하는 UI/후처리 정책
- `clear`
  - no-hand 또는 reset 조건에서 frame history를 비우는 정책값
- `motion window`, `motion threshold`
  - gesture triggering 전에 recent motion을 얼마나 볼지 정하는 상위 정책

이 모델 자체는 frame classifier이므로, 위 값들은 모델 구조가 아니라 host app policy입니다.

## 7. 왼손 처리 주의

현재 구현 기준으로는 왼손 추론을 비활성화한 정책이 들어가 있었습니다.

즉:

- 오른손만 실사용
- 왼손은 `disabled` 또는 inference skip

상대 프로젝트에서 선택지는 두 가지입니다.

1. 현재 정책 그대로 오른손만 유지
2. 왼손도 허용하되, 별도 mirror 정책을 추가

지금 전달하는 모델 파일만 기준으로 보면, 필수 계약은 `21 x 3 normalized landmarks + pos_scale + 7class output` 입니다. 왼손 처리 정책은 host app 쪽 의사결정입니다.

## 8. 통합 체크리스트

상대 프로젝트 AI가 실제로 붙일 때는 아래만 확인하면 됩니다.

### 필수

- `onnxruntime-web` 설치 여부 확인
- runtime root에 `model.onnx`와 `model.onnx.data`를 같이 배치
- `config.json`, `class_names.json`, `input_spec.json` 생성
- MediaPipe normalized landmarks 21개를 그대로 입력
- `worldLandmarks` 미사용
- `pos_scale` 적용 후 `[1,21,3]` 텐서로 추론
- output logits에 softmax 적용
- argmax로 class 선택

### 권장

- `modelIntervalMs = 60`으로 시작
- `inferFps = 30`, `numHands = 1`로 시작
- no-hand 시 즉시 neutral/no_hand 상태로 리셋
- frame model에서는 warmup, temporal padding, seq buffer 로직 제거

## 9. 빠른 검증 방법

상대 환경에서 아래가 맞으면 거의 정상 통합으로 보면 됩니다.

1. ONNX session 로드 성공
2. `inputNames[0]`, `outputNames[0]`가 정상 조회됨
3. landmark 21개가 들어왔을 때 `[1,21,3]` 텐서 생성 가능
4. output shape가 사실상 7 logits으로 나옴
5. gesture label이 아래 7개 중 하나로 나옴

```text
neutral
fist
open_palm
V
pinky
animal
k-heart
```

6. 손이 안 잡히면 즉시 `no_hand` 또는 `neutral`로 떨어짐
7. sequence warmup 없이 바로 ready prediction이 갱신됨

## 10. 한 줄 결론

이 모델은 "MediaPipe normalized hand landmarks 21개를 `pos_scale`로 정규화해서 `[1,21,3]` ONNX 텐서로 넣고, 7-class logits를 softmax/argmax로 해석하는 single-frame classifier" 로 이해하면 됩니다.

상대 프로젝트 AI는 이 문서와 `frame_spatial_transformer` 폴더만 받아도 runtime bundle 재구성, 브라우저 추론 연결, 후처리 초기값 세팅까지 진행할 수 있어야 합니다.
