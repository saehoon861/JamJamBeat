# quickstart.md - frontend-test 실시간 웹캠 추론 모니터 빠른 실행 가이드

`frontend-test`는 `frontend/`의 runtime 자산을 읽어서, 웹캠 → MediaPipe → sequence ONNX 추론 결과를 모니터링하는 별도 실험 앱입니다.

## 1. 작업 위치

아래 경로에서 실행합니다.

```bash
cd /home/user/projects/JamJamBeat-model3/frontend-test
```

중요:
- `frontend-test`는 `feature/model3` worktree 기준입니다.
- 원본 `/home/user/projects/JamJamBeat`가 아니라 `/home/user/projects/JamJamBeat-model3`에서 실행해야 합니다.

## 2. 의존성 설치

처음 한 번만 설치합니다.

```bash
npm install --no-package-lock
```

## 3. 개발 서버 실행

```bash
npm run dev
```

기본 포트:

- `http://127.0.0.1:3003/`
- `http://localhost:3003/`

WSL 환경에서 USB 웹캠을 쓰는 경우 권장:
- 서버는 WSL에서 실행
- 페이지는 Windows 브라우저(Chrome/Edge)에서 열기

## 4. 브라우저에서 테스트 시작

브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:3003/
```

또는

```text
http://localhost:3003/
```

페이지가 열리면:

1. 웹캠 권한을 허용합니다.
2. 필요하면 `Inference Settings` 패널에서 프리셋을 고릅니다.
3. `Viewer-like`로 비교하고 싶으면:
   - `inferWidth = 0`
   - `inferFps = 30`
   - `modelIntervalMs = 60`
   - `tau = 0.90`
   - `numHands = 1`
   - `0.5 / 0.5 / 0.5`
   가 자동으로 채워집니다.
4. 설정을 바꿨다면 `Apply & Restart`를 눌러 반영합니다.
5. `Start Monitor` 버튼을 누릅니다.
6. 손을 카메라에 보여줍니다.

## 5. 화면에서 바로 확인할 것

좌측:
- `Webcam + Raw Landmarks`
- 실제 웹캠 영상과 raw landmark overlay

우측:
- `pos_scale Preview`
- 모델 입력에 쓰이는 정규화 landmark preview

메타 패널:
- `bundle_id`
- `model_id`
- `normalization`
- `seq_len`
- `tau`
- `checkpoint_fingerprint`

설정 패널:
- `Preset`
- `inferWidth`
- `inferFps`
- `modelIntervalMs`
- `tau` (숫자 입력)
- `numHands`
- `minHandDetectionConfidence`
- `minHandPresenceConfidence`
- `minTrackingConfidence`

손별 패널:
- `Raw model`
- `Final gesture`
- `Status`
- `Frames`
- `Top3`
- `Inference`

latency 패널:
- `Current`
- `Average`
- `P50`
- `P95`

## 6. 정상 동작 기준

손이 없을 때:
- `status = no_hand`

손을 막 보여주기 시작했을 때:
- `status = warmup`
- `Frames`가 `1/8`부터 증가

8프레임이 쌓이면:
- `status = ready` 또는 `tau_neutralized`

현재 active runtime 기준 기대값:
- `bundle_id = pos_scale_mlp_sequence_delta_20260319_162806`
- `model_id = mlp_sequence_delta`
- `normalization = pos_scale`
- `seq_len = 8`
- `tau = 0.85`

## 7. 빌드 확인

실행 전/후로 번들 검증만 하고 싶으면:

```bash
npm run build
```

## 8. 종료

개발 서버를 띄운 터미널에서:

```bash
Ctrl+C
```

## 9. 자주 보는 문제

웹캠이 안 뜰 때:
- Windows 브라우저에서 열었는지 확인
- 브라우저 권한 설정에서 카메라 허용 확인
- 다른 앱이 카메라를 점유 중인지 확인

페이지는 뜨는데 추론이 안 될 때:
- `Loaded Runtime Metadata`가 채워지는지 확인
- `Start Monitor`를 눌렀는지 확인
- `status`가 계속 `no_hand`인지, `warmup`까지는 가는지 확인

포트 충돌이 날 때:
- 3003 포트를 이미 다른 프로세스가 쓰는지 확인

## 10. 관련 참고 파일

- [FRONTEND_WEBCAM_SEQUENCE_MONITOR_CURRENT.md](/home/user/projects/JamJamBeat-model3/frontend-test/FRONTEND_WEBCAM_SEQUENCE_MONITOR_CURRENT.md)
- [package.json](/home/user/projects/JamJamBeat-model3/frontend-test/package.json)
- [vite.config.mjs](/home/user/projects/JamJamBeat-model3/frontend-test/vite.config.mjs)
- [main.js](/home/user/projects/JamJamBeat-model3/frontend-test/src/main.js)
