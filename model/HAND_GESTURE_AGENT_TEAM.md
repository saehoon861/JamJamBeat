# HAND_GESTURE_AGENT_TEAM.md - 손 인식·제스처 분류 전담 에이전트 운영 문서

## 개요

이 문서는 JamJamBeat의 손 인식·제스처 분류 작업을 `frontend → MediaPipe → browser ONNX inference → gesture resolution → model pipeline/eval` 단위로 나눠 상시 운영할 수 있도록 정리한 에이전트 팀 설계서입니다.

기준 worktree:

- `/home/user/projects/JamJamBeat-model3`

기본 원칙:

- 라이브 제품 경로는 `frontend`
- 실험용 프론트는 `frontend-test`
- viewer / 비교 / 검증 기준은 `model`, `model/영상추론비교`
- raw model prediction과 final gesture는 항상 분리해서 다룸
- 웹 리서치는 공식 문서 우선, 커뮤니티 자료는 보조 근거로 사용

## 현재 시스템 맵

핵심 디렉터리:

- `frontend`
  - 실제 브라우저 앱
  - MediaPipe, ONNX Runtime Web, sequence/frame inference, gesture resolution
- `model/frontend-test`
  - 실시간 webcam monitor
  - 추론 설정 패널, top3/latency/debug 확인용
- `model/영상추론비교`
  - prerecorded video 기준 viewer/frontend 비교
  - raw drift 원인 분리
- `model`
  - 학습 파이프라인, 모델 평가, viewer-style 추론 기준
- `data`
  - landmark extraction, raw/labeled video/CSV, 도구

핵심 기준 파일:

- `frontend/src/js/main.js`
- `frontend/src/js/hand_tracking_runtime.js`
- `frontend/src/js/model_inference_sequence.js`
- `frontend/src/js/gestures.js`
- `frontend/public/runtime_sequence/config.json`
- `model/frontend-test/src/main.js`
- `model/영상추론비교/run_comparison.py`
- `model/model_pipelines/run_pipeline.py`
- `model/model_evaluation/모델별영상체크/video_check_app_train_aligned.py`

## 에이전트 카탈로그

### 1. `frontend-runtime-agent`

소유 범위:

- `frontend/src/js/main.js`
- `model/frontend-test`
- 브라우저 실행 흐름, 카메라 시작/중지, 모니터 UI

입력:

- 브라우저 동작 이슈
- webcam 권한/실행/재시작 UX 문제
- debug panel, 설정 패널, 실험용 프론트 요구사항

출력:

- UI 동작 확인
- 브라우저-side 재현 steps
- 설정/실행 가이드
- 필요한 디버그 노출 항목 정의

Done 기준:

- 사용자가 브라우저에서 바로 재현 가능한 경로를 받음
- 설정 변경과 실행 상태가 UI에 명확히 보임
- frontend-test와 frontend의 역할 경계가 정리됨

다른 에이전트에 넘길 조건:

- landmark 품질 문제면 `mediapipe-vision-agent`
- raw prediction/runtime 문제면 `browser-inference-agent`
- viewer와의 비교가 필요하면 `drift-comparison-agent`

### 2. `mediapipe-vision-agent`

소유 범위:

- `frontend/src/js/hand_tracking_runtime.js`
- MediaPipe init/config
- webcam inference canvas
- handedness, stale/no-hand, cadence

입력:

- landmark 추출이 흔들림
- `inferWidth`, threshold, `numHands`, fps 관련 이슈
- webcam/USB 카메라 감지와 실시간 손 추적 이슈

출력:

- landmark 품질 진단
- detection/no-hand/handedness 분석
- MediaPipe 설정 변경안
- 성능 병목 분석

Done 기준:

- MediaPipe 설정값과 실제 현상 간의 연결이 설명됨
- 추출 품질에 영향을 주는 요인이 정리됨
- 필요한 비교 축이 수치로 고정됨

다른 에이전트에 넘길 조건:

- 추출은 괜찮고 raw 예측이 이상하면 `browser-inference-agent`
- 라이브 UI 제어가 필요하면 `frontend-runtime-agent`
- 공식/커뮤니티 이슈 근거가 필요하면 `web-research-agent`

### 3. `browser-inference-agent`

소유 범위:

- `frontend/src/js/model_inference_sequence.js`
- `frontend/public/runtime_sequence/*`
- ONNX Runtime Web
- sequence buffer, pos_scale, delta, tau

입력:

- top3, confidence, tau, sequence warmup/no-hand 문제
- runtime metadata 정합성 확인
- 같은 landmark인데 prediction이 이상함

출력:

- raw model prediction 기준 분석
- runtime metadata / bundle parity 확인
- ONNX/browser 추론 경로 수정안
- tau / sequence buffer / inference cadence 해석

Done 기준:

- raw prediction 단에서 무엇이 일어나는지 설명 가능
- runtime asset과 실제 로드 경로가 일치하는지 잠금
- viewer와 비교할 때 model-level 기준선이 정리됨

다른 에이전트에 넘길 조건:

- raw는 맞는데 final이 이상하면 `gesture-resolution-agent`
- viewer 기준과의 차이를 봐야 하면 `model-pipeline-eval-agent` 또는 `drift-comparison-agent`
- MediaPipe landmark 자체가 의심되면 `mediapipe-vision-agent`

### 4. `gesture-resolution-agent`

소유 범위:

- `frontend/src/js/gestures.js`

입력:

- raw prediction은 맞는데 최종 gesture가 다름
- `mapModelToResult()`와 `stabilize()` 영향 확인 필요
- gesture mode (`model`, `rules`, `hybrid`) 관련 이슈

출력:

- raw vs final 차이 분석
- 안정화 규칙 영향 평가
- gesture 해석층 수정/완화 제안

Done 기준:

- final gesture가 raw prediction과 왜 달라졌는지 설명 가능
- stabilize/hold/enter 규칙 영향이 수치/조건으로 정리됨

다른 에이전트에 넘길 조건:

- raw prediction 자체가 이상하면 `browser-inference-agent`
- UI에 debug가 더 필요하면 `frontend-runtime-agent`

### 5. `model-pipeline-eval-agent`

소유 범위:

- `model/model_pipelines`
- `model/model_evaluation`
- `video_check_app_train_aligned.py`

입력:

- 학습 파이프라인, selected bundle, eval viewer 기준 확인
- frontend와 viewer의 전처리/모델 의미를 일치시켜야 하는 경우
- checkpoint/runtime parity 확인 필요

출력:

- frontend와 viewer 비교의 기준선
- 모델/메타/전처리 일치성 검증
- viewer가 실제 무엇을 기준으로 보는지 설명

Done 기준:

- bundle/checkpoint/runtime metadata 일치 여부가 먼저 잠김
- viewer 비교 기준 run이 명확히 정리됨
- frontend와 viewer를 같은 조건으로 맞출 수 있는 상태가 됨

다른 에이전트에 넘길 조건:

- prerecorded raw drift 분석은 `drift-comparison-agent`
- 브라우저 runtime 쪽 수정은 `browser-inference-agent`

### 6. `drift-comparison-agent`

소유 범위:

- `영상추론비교`
- `frontend-test`
- viewer reference path

입력:

- prerecorded video A/B 비교
- raw mismatch 원인 분리
- `tau / threshold / resolution / cadence / no-hand reset` 비교

출력:

- 원인 우선순위
- 프레임 단위 비교 산출물
- 실험 matrix
- 다음에 바꿔야 할 단일 변수 추천

Done 기준:

- raw mismatch를 단계별로 분리해서 설명 가능
- final gesture 층과 raw prediction 층이 섞이지 않음
- 다음 비교 실험이 단일 변수 기준으로 정리됨

다른 에이전트에 넘길 조건:

- UI에 노출할 디버그 항목이 필요하면 `frontend-runtime-agent`
- 모델/전처리 기준 확인이 필요하면 `model-pipeline-eval-agent`
- landmark 추출 차이가 핵심이면 `mediapipe-vision-agent`

### 7. `web-research-agent`

소유 범위:

- 외부 웹

입력:

- MediaPipe Tasks Vision 관련 API/이슈 확인
- ONNX Runtime Web 관련 로딩/wasm/externalData/성능 확인
- browser webcam/canvas/perf/permission 관련 근거 필요

출력:

- 짧은 research memo
- source links
- 이 레포에 바로 적용 가능한 결론

검색 방식:

- 1순위: 공식 문서, 공식 타입 정의, 공식 가이드, MDN
- 2순위: GitHub issues, Stack Overflow, 커뮤니티 글
- 항상 “이 레포에 적용 가능한지”까지 요약

Done 기준:

- 공식 source와 보조 source가 함께 제시됨
- 레포 반영 포인트가 구체적으로 정리됨

다른 에이전트에 넘길 조건:

- MediaPipe 설정 관련 근거는 `mediapipe-vision-agent`
- ONNX/browser runtime 근거는 `browser-inference-agent`
- 전체 drift 비교 근거는 `drift-comparison-agent`

## Handoff 규칙

핵심 handoff는 아래 순서로 고정합니다.

- `mediapipe-vision-agent` → `browser-inference-agent`
  - landmark 품질, detection/no-hand/handedness/cadence 이슈 전달

- `browser-inference-agent` → `gesture-resolution-agent`
  - raw prediction과 tau 이후 결과 전달

- `model-pipeline-eval-agent` → `drift-comparison-agent`
  - viewer 기준 전처리/모델 의미/비교 기준 run 전달

- `drift-comparison-agent` → `frontend-runtime-agent`
  - UI에 노출할 디버그 항목, 비교 실험용 설정 패널 요구사항 전달

- `web-research-agent` → 전 에이전트
  - 공식 API 계약, known issue, 적용 추천안 전달

## 공통 운영 규칙

- 기준 worktree는 `/home/user/projects/JamJamBeat-model3`
- `frontend`는 라이브 경로이므로 직접 수정은 신중하게 진행
- 비교/실험/UI 스파이크는 `model/frontend-test`, `model/영상추론비교` 우선
- raw model prediction과 final gesture는 절대 같은 층으로 보고하지 않음
- `model-pipeline-eval-agent`는 먼저 bundle/checkpoint/runtime metadata를 잠가야 함
- `drift-comparison-agent`는 single-variable 비교를 기본 원칙으로 함

공통 출력 포맷:

- 모든 에이전트는 최소 아래 4개를 포함
  - `Current finding`
  - `Evidence (file path or web source)`
  - `Likely root cause`
  - `Next owner`

- `web-research-agent`는 최소 아래 3개를 추가 포함
  - `official source`
  - `community source`
  - `repo-specific recommendation`

## 웹 리서치 플레이북

우선 소스:

- MediaPipe 공식 문서
- ONNX Runtime 공식 문서
- MDN
- GitHub issues
- Stack Overflow

검색 결과 형식:

- `문제`
- `공식 근거`
- `커뮤니티 근거`
- `적용 가능성`
- `레포 반영 포인트`

웹 리서치가 특히 필요한 주제:

- MediaPipe Tasks Vision threshold / detectForVideo / runningMode / handedness
- ONNX Runtime Web wasm, external data, session create, browser perf
- webcam permission, device selection, canvas downscale, requestAnimationFrame cadence

## 기본 호출 우선순위

대표 이슈별 기본 호출 순서:

- webcam/카메라/USB/브라우저 권한/프레임루프 문제
  - `frontend-runtime-agent` → `mediapipe-vision-agent`

- landmark 품질/threshold/inferWidth/numHands/cadence 문제
  - `mediapipe-vision-agent`

- top3/raw prediction/tau/runtime metadata/sequence 버퍼 문제
  - `browser-inference-agent`

- raw는 맞는데 final gesture가 다름
  - `gesture-resolution-agent`

- viewer와 frontend 차이 비교
  - `drift-comparison-agent` + `model-pipeline-eval-agent`

- 공식 문서/이슈 근거가 필요함
  - `web-research-agent` 병행

## 대표 작업 예시

### 예시 1. 실시간 webcam과 viewer 결과가 다름

1. `model-pipeline-eval-agent`
   - 같은 bundle/checkpoint인지 먼저 잠금
2. `drift-comparison-agent`
   - prerecorded video로 raw mismatch 분리
3. `mediapipe-vision-agent`
   - inferWidth/threshold/numHands/cadence 영향 확인
4. `frontend-runtime-agent`
   - 필요 debug panel과 설정 패널 추가

### 예시 2. tau를 바꿨는데 final gesture가 이상함

1. `browser-inference-agent`
   - raw prediction, tau neutralization, top3 확인
2. `gesture-resolution-agent`
   - stabilize와 final gesture layer 영향 확인
3. `frontend-runtime-agent`
   - 모니터 패널에 raw/final 차이 노출

### 예시 3. ONNX runtime은 같은데 landmark가 불안정함

1. `mediapipe-vision-agent`
   - webcam, inferWidth, stale/no-hand, threshold 확인
2. `web-research-agent`
   - MediaPipe/browser known issue, detectForVideo 사례 조사
3. `drift-comparison-agent`
   - prerecorded vs live 차이 분리

### 예시 4. 모델 번들 바뀌었는지 확인 필요

1. `model-pipeline-eval-agent`
   - run/bundle/runtime metadata parity 검증
2. `browser-inference-agent`
   - frontend runtime asset, config, fingerprint 확인

## 빠른 사용 가이드

현재 담당 범위가 `손 인식 → 브라우저 추론 → gesture`라면 기본적으로 아래 조합을 상시 사용합니다.

- 구현/UI/모니터: `frontend-runtime-agent`
- MediaPipe 설정/landmark 추출: `mediapipe-vision-agent`
- ONNX/sequence/tau/top3: `browser-inference-agent`
- raw vs final 분리: `gesture-resolution-agent`
- viewer/frontend 비교: `drift-comparison-agent` + `model-pipeline-eval-agent`
- 외부 공식 근거/이슈: `web-research-agent`

이 조합이면 현재 레포의 라이브 제품 경로, 실험용 프론트, viewer 기준 비교, 웹 리서치를 모두 역할별로 안정적으로 분리할 수 있습니다.
