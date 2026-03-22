# NNX(ONNX) 모델 로드 실패 RCA

작성일: 2026-03-19

## 한 줄 요약
장애의 직접 원인은 runtime-20260319T051605Z-1-001 폴더 병합 자체가 아니라, ONNX 자산 및 wasm 로딩 경로 해석과 배포 환경의 정적 서빙 조건 불일치였다.

## 관찰된 증상
- 모델 초기화 시점에 ONNX 로드 실패 발생.
- 브라우저 로그 중 아래 2개는 원인 로그가 아닌 일반 초기화 로그.
  - browser-control-accessibility-tree.js:1 [Browser Control] Accessibility tree injected
  - client.ts:16 [vite] connecting...

## 코드 근거 기반 원인 체인

### 1) 런타임 자산 경로가 배포 base path에 민감
- 경로 해석 보강 포인트:
  - frontend/src/js/model_inference.js:29 resolveRuntimeRoot
  - frontend/src/js/model_inference.js:59 MODEL_PATH
  - frontend/src/js/model_inference.js:61 MODEL_EXTERNAL_DATA_PATH
- 절대경로 가정이 강하면 하위 경로 배포에서 404 가능성이 커진다.

### 2) wasm 로딩은 CDN 접근성 제약 영향
- 관련 포인트:
  - frontend/src/js/model_inference.js:15 ORT_CDN_WASM_BASE
  - frontend/src/js/model_inference.js:43 resolveOrtWasmRoot
  - frontend/src/js/model_inference.js:183 로컬 wasm 기본 경로 설정
  - frontend/src/js/model_inference.js:274 로컬 우선 세션 생성
  - frontend/src/js/model_inference.js:279 로컬 실패 시 CDN 재시도

### 3) 앱 기본 동작이 model 모드라 ONNX 실패가 즉시 노출
- 관련 포인트:
  - frontend/src/js/gestures.js:115 기본값 model
  - frontend/src/js/gestures.js:120 fallback도 model

### 4) MediaPipe 경로는 ONNX와 별도
- 관련 포인트:
  - frontend/src/js/env_config.js:35 getConfiguredMediaPipeWasmRoot
  - frontend/src/js/env_config.js:44 기본 /mediapipe
  - frontend/src/js/env_config.js:47 getConfiguredHandLandmarkerTaskPath
  - frontend/src/js/env_config.js:56 기본 /hand_landmarker.task
  - frontend/src/js/main.js:525 forVisionTasks(getConfiguredMediaPipeWasmRoot())
  - frontend/src/js/main.js:527 modelAssetPath = getConfiguredHandLandmarkerTaskPath()

## 왜 runtime 스냅샷 폴더 병합이 직접 원인이 아닌가
- 실행 코드에서 runtime-20260319T051605Z-1-001 폴더명을 직접 참조하지 않음.
- 식별자는 문서에서만 확인되며, 실제 실행 경로는 /runtime/*, /mediapipe/* 체계.
- 따라서 원인은 폴더 존재 여부가 아니라 실제 서비스 URL에서 자산을 어떻게 로드하는지의 문제.

## 외부 근거 요약
- onnxruntime-web 실패 패턴 공통점:
  1) 절대경로와 배포 경로 불일치
  2) CDN 접근 실패
  3) 정적 호스팅의 MIME/CORS/응답 문제
- 참고 링크:
  - https://stackoverflow.com/questions/77179151/onnxruntime-web-fails-to-find-ort-wasm-simd-wasm-doesnt-use-my-static-folder
  - https://github.com/microsoft/onnxruntime/issues/22010
  - https://github.com/google-ai-edge/mediapipe/issues/5961

## 적용된 수정
- ONNX 경로 동적 해석 추가: runtimeRoot query/global/env/base-url 지원.
- ORT wasm 로딩을 로컬 우선으로 변경하고 필요 시 CDN fallback 적용.
- 성공 로그에 runtimeRoot, wasmBase를 남겨 운영 디버깅성을 강화.

## 검증 결과
- 빌드 성공: npm run build 통과.
- 산출물 확인:
  - frontend/dist/runtime/model.onnx
  - frontend/dist/runtime/model.onnx.data
  - frontend/dist/runtime/class_names.json
  - frontend/dist/mediapipe/vision_wasm_internal.js
  - frontend/dist/mediapipe/vision_wasm_internal.wasm
  - frontend/dist/ort-wasm-simd-threaded.wasm

## 재발 방지 체크리스트
- 배포가 루트가 아니면 runtimeRoot, ortWasmRoot를 명시.
- 배포 후 Network 탭 필수 확인 항목:
  - .../runtime/model.onnx
  - .../runtime/model.onnx.data
  - .../ort-wasm-*.wasm
  - .../mediapipe/vision_wasm_internal.wasm
- CDN 접근 불안정 환경은 로컬 wasm-only 정책으로 운영.
