# MediaPipe 설정 가이드

JamJam Beat 프론트엔드는 MediaPipe 손 인식 엔진 경로를 하드코딩하지 않고 설정으로 관리합니다.

## 기본값

- wasm 루트: `/mediapipe`
- hand landmarker task 경로: `/hand_landmarker.task`

기본적으로 위 경로를 사용하며, 별도 설정이 없으면 현재 프로젝트의 정적 파일 경로를 기준으로 동작합니다.

## 우선순위

경로 설정은 아래 우선순위로 적용됩니다.

1. URL 파라미터
2. `window` 전역 오버라이드
3. Vite `.env`
4. 코드 기본값

## URL 파라미터

브라우저 주소에 아래 값을 붙여서 바로 테스트할 수 있습니다.

```text
?mediapipeRoot=/mediapipe
?handLandmarkerTask=/hand_landmarker.task
```

예시:

```text
http://localhost:3002/?mediapipeRoot=/mediapipe&handLandmarkerTask=/hand_landmarker.task
```

## window 전역 오버라이드

앱이 시작되기 전에 아래 전역 값을 주입하면 경로를 바꿀 수 있습니다.

```js
window.__JAMJAM_MEDIAPIPE_WASM_ROOT = "/mediapipe";
window.__JAMJAM_HAND_LANDMARKER_TASK_PATH = "/hand_landmarker.task";
```

운영 환경에서 서버 템플릿이나 별도 부트스트랩 스크립트로 넣을 때 유용합니다.

## Vite 환경변수

`.env`, `.env.local`, `.env.production` 등에 아래 값을 넣어 사용할 수 있습니다.

```env
VITE_MEDIAPIPE_WASM_ROOT=/mediapipe
VITE_HAND_LANDMARKER_TASK_PATH=/hand_landmarker.task
```

## 실제 사용 위치

아래 화면들이 동일한 설정 함수를 사용합니다.

- 메인 손 인식 화면
- 테마 선택 화면
- 성능 점검 화면

공통 설정 함수는 `frontend/src/js/env_config.js`에 있습니다.

## 관련 정적 파일

현재 프로젝트에서 로컬 서빙하는 MediaPipe 관련 파일은 아래 위치에 있습니다.

- `frontend/public/mediapipe`
- `frontend/public/hand_landmarker.task`

## 문제 해결

### 1. MediaPipe가 로드되지 않을 때

아래를 확인합니다.

- `/mediapipe/vision_wasm_internal.js`
- `/mediapipe/vision_wasm_internal.wasm`
- `/hand_landmarker.task`

브라우저 네트워크 탭에서 404가 없는지 확인합니다.

### 2. 인터넷이 느릴 때

현재는 CDN이 아니라 로컬 정적 파일을 사용하므로, 외부 네트워크보다 내부 정적 파일 경로가 더 중요합니다.

### 3. 배포 환경에서 경로가 달라질 때

코드를 수정하지 말고 아래 중 하나만 바꾸는 것을 권장합니다.

- URL 파라미터
- `window` 전역 주입
- `.env` 값

## 참고 코드

- 설정 함수: `frontend/src/js/env_config.js`
- 메인 초기화: `frontend/src/js/main.js`
- 테마 초기화: `frontend/src/js/theme.js`
- 성능 페이지 초기화: `frontend/src/js/performance.js`
