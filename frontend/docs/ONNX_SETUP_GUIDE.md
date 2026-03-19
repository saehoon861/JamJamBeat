# ONNX Runtime Web 실행 가이드

## ✅ 완료된 작업

1. ✅ `onnxruntime-web` 패키지 설치
2. ✅ 런타임 모델 파일을 `public/runtime/` 폴더로 복사
3. ✅ `model_inference_onnx.js` 작성 (브라우저 로컬 추론)
4. ✅ 기존 `model_inference.js` 백업 및 교체
5. ✅ `gestures.js`의 기본 모드를 `hybrid`로 변경

---

## 🚀 실행 방법

### 방법 1: 기본 실행 (백엔드 없이)

```bash
cd /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend
npm run dev
```

브라우저에서:
```
http://localhost:5173/
```

**이제 백엔드 서버 없이 브라우저에서만 모델 추론이 됩니다!**

---

### 방법 2: 제스처 모드 변경 (URL 파라미터)

#### A. Hybrid 모드 (기본값, 권장)
```
http://localhost:5173/?gestureMode=hybrid
```
- 주먹: 규칙 기반 (빠르고 정확)
- 나머지: AI 모델 (pinky, animal, k-heart 등)

#### B. Model 모드 (모든 제스처를 AI로)
```
http://localhost:5173/?gestureMode=model
```
- 모든 제스처를 ONNX 모델로 판단

#### C. Rules 모드 (AI 없이)
```
http://localhost:5173/?gestureMode=rules
```
- 주먹, 브이, 손바닥만 인식 (규칙 기반)

---

## 🔍 성능 모니터링

```
http://localhost:5173/?profilePerf=1
```

브라우저 콘솔(F12)에서 확인:
```
[Perf][ModelInferenceONNX] {
  requests: 67,
  successes: 67,
  avgMs: 3.2,  ← ONNX는 매우 빠름 (백엔드 12ms vs ONNX 3ms)
  maxMs: 8.5,
  mode: "onnx-local"
}
```

---

## 📂 파일 구조

```
frontend/
├── public/
│   └── runtime/                      ← 새로 추가됨
│       ├── model.onnx                (2.8KB)
│       ├── model.onnx.data           (132KB)
│       ├── class_names.json
│       ├── config.json
│       ├── feature_order.json
│       └── input_spec.json
│
├── src/js/
│   ├── model_inference.js            ← ONNX 버전 (새로 교체됨)
│   ├── model_inference.backup.js     ← 백엔드 버전 (백업)
│   ├── model_inference_onnx.js       ← ONNX 원본 (참고용)
│   └── gestures.js                   ← 기본 모드 'hybrid'로 변경됨
│
└── node_modules/
    └── onnxruntime-web/              ← 새로 설치됨
```

---

## 🎯 주요 변경 사항

### 이전 (백엔드 방식)
```
카메라 → MediaPipe → 프론트엔드 → HTTP → 백엔드(Python) → 응답
지연시간: 12~28ms (평균)
서버 필요: ✅ Python + PyTorch
```

### 현재 (ONNX 방식)
```
카메라 → MediaPipe → 프론트엔드 → ONNX Runtime Web → 결과
지연시간: 2~8ms (예상)
서버 필요: ❌ 없음
```

---

## 🔧 트러블슈팅

### 1. "Failed to load model.onnx" 오류
```bash
# public/runtime/ 폴더에 파일이 있는지 확인
ls -lh /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/public/runtime/

# 파일이 없으면 다시 복사
cp /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/runtime-20260319T051605Z-1-001/runtime/model.onnx* \
   /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/public/runtime/
```

### 2. ONNX 모델이 로드되지 않음
브라우저 콘솔에서:
```javascript
// ONNX 모델 상태 확인
import { getModelInferenceStatus } from './src/js/model_inference.js'
console.log(getModelInferenceStatus())
// { endpointConfigured: true, mode: "onnx-local" }
```

### 3. 제스처가 인식되지 않음
```
http://localhost:5173/?gestureMode=model&profilePerf=1
```
- 콘솔에서 `[ModelInferenceONNX]` 로그 확인
- 모델 로드 성공 메시지 확인

---

## 🔙 백엔드 방식으로 되돌리기

```bash
cd /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend

# 백업 파일을 다시 복원
cp src/js/model_inference.backup.js src/js/model_inference.js

# .env 파일 생성
echo "VITE_MODEL_ENDPOINT=http://127.0.0.1:8008/infer" > .env

# 백엔드 서버 실행 (별도 터미널)
cd ../test_backend
python infer_server.py

# 프론트엔드 실행
cd ../frontend
npm run dev
```

---

## 📊 성능 비교

| 항목 | 백엔드 방식 | ONNX 방식 |
|------|------------|-----------|
| 추론 속도 | 12~28ms | 2~8ms (예상) |
| 네트워크 지연 | O | X |
| 서버 비용 | O | X |
| 오프라인 작동 | X | O |
| 배포 난이도 | 높음 | 낮음 |
| 확장성 | 서버 스케일링 필요 | 클라이언트 분산 |

---

## 🎮 테스트 시나리오

### 1. 기본 테스트
```
http://localhost:5173/
```
- 카메라 앞에서 손동작 (주먹, 브이, 손바닥, pinky 등)
- 화면 하단 상태 텍스트에서 인식 결과 확인

### 2. 성능 테스트
```
http://localhost:5173/?profilePerf=1
```
- F12 콘솔 열기
- 손동작 10초간 반복
- 평균 추론 시간 확인 (3~5ms 예상)

### 3. 모드 비교 테스트
```
# Hybrid 모드
http://localhost:5173/?gestureMode=hybrid

# Model 전용 모드
http://localhost:5173/?gestureMode=model

# Rules 전용 모드
http://localhost:5173/?gestureMode=rules
```

---

## 💡 추가 최적화 (선택 사항)

### 1. 모델 미리 로드 (로딩 시간 단축)

`main.js` 또는 `performance.js`에 추가:
```javascript
import { preloadModel } from './model_inference.js';

// 앱 시작 시 모델 미리 로드
preloadModel().then(() => {
  console.log('✅ ONNX 모델 미리 로드 완료');
});
```

### 2. WASM 경로 최적화

`model_inference.js`의 22번째 줄 수정:
```javascript
// 현재
ort.env.wasm.wasmPaths = "/node_modules/onnxruntime-web/dist/";

// 최적화 (Vite 빌드 시 public 폴더로 복사)
ort.env.wasm.wasmPaths = "/wasm/";
```

---

## 🎉 완료!

이제 백엔드 서버 없이 브라우저에서만 실시간 손동작 인식이 가능합니다!

**테스트 명령:**
```bash
cd /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend
npm run dev
```

브라우저: `http://localhost:5173/?profilePerf=1`
