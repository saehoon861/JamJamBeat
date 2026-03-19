# Frontend 연결 가이드

## Runtime 파일 구조

```
runtime/
├── model.onnx            # 브라우저 추론용 (onnxruntime-web)
├── model.pt              # PyTorch 체크포인트 (서버 전용)
├── model.py              # MLPEmbedding 아키텍처 정의
├── loader.py             # 번들 로드 + 체크포인트 검증
├── inference.py          # predict() 함수 (tau 후처리 포함)
├── config.json           # 입력 차원, 클래스 수, fingerprint 등
├── class_names.json      # ['neutral','fist','open_palm','V','pinky','animal','k-heart']
├── feature_order.json    # 입력 컬럼 순서 (x0,y0,z0,...,x20,y20,z20)
└── input_spec.json       # 입력 shape, dtype 명세
```

---

## 입력 / 출력 계약

**입력**
```
float32 배열 63개 — MediaPipe 21 keypoint 원본 좌표
순서: x0,y0,z0, x1,y1,z1, ..., x20,y20,z20
정규화 없음 (raw MediaPipe 좌표 그대로)
```

**출력**
```json
{
  "pred_index": 2,
  "pred_label": "open_palm",
  "confidence": 0.913,
  "probs": [0.01, 0.02, 0.91, 0.01, 0.02, 0.02, 0.01],
  "tau_applied": 0.85,
  "tau_neutralized": false
}
```

---

## 브라우저 연결 (ONNX Runtime Web)

```
카메라 → MediaPipe Hands JS → 63d float → model.onnx → 제스처
```

```bash
npm install onnxruntime-web
```

```js
import * as ort from 'onnxruntime-web';

const CLASS_NAMES = ['neutral','fist','open_palm','V','pinky','animal','k-heart'];

function softmax(arr) {
  const max = Math.max(...arr);
  const exp = arr.map(x => Math.exp(x - max));
  const sum = exp.reduce((a, b) => a + b, 0);
  return exp.map(x => x / sum);
}

const session = await ort.InferenceSession.create('/model.onnx');

async function predict(joints63, tau = 0.85) {
  const tensor = new ort.Tensor('float32', Float32Array.from(joints63), [1, 63]);
  const result = await session.run({ joints: tensor });
  const probs = softmax(Array.from(result.logits.data));
  const rawIdx = probs.indexOf(Math.max(...probs));
  const conf = probs[rawIdx];
  const predIdx = conf < tau ? 0 : rawIdx;
  return {
    predIdx,
    predLabel: CLASS_NAMES[predIdx],
    conf,
    probs,
    tauNeutralized: predIdx !== rawIdx,
  };
}
```

---

## 후처리 수치

| 항목 | 값 | 설명 |
|------|-----|------|
| tau | 0.85 | p_max < tau → neutral(0) 강제 |
| neutral_index | 0 | class 0 = neutral |
| vote_n | 7 | 연속 프레임 다수결 (선택적) |
| debounce_k | 3 | 상태 전환 억제 (선택적) |

---

## 권장 스택

| 레이어 | 기술 |
|--------|------|
| 손 검출 | MediaPipe Hands JS (WebAssembly) |
| 모델 추론 | ONNX Runtime Web (`model.onnx`) |
| 프레임워크 | React |
| 백엔드 | 불필요 |
| SQL / DB | 불필요 |
