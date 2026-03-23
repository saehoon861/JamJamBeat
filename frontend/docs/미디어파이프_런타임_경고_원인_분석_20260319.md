# MediaPipe 런타임 경고 원인 정리

작성일: 2026-03-19

## 결론
- 문제의 본질은 JavaScript API 계약과 코드 사용 방식의 불일치였다.
- RunningMode 런타임 import 사용과 LIVE_STREAM, detectAsync 가정이 핵심 원인이었다.

## 사용자 증상
- Failed to boot legacy main runtime 메시지와 RunningMode export 오류 발생.
- handLandmarker.detectAsync is not a function 오류 반복 발생.

## 공식 문서 근거
- RunningMode는 IMAGE 또는 VIDEO 두 값만 허용.
- HandLandmarker 메서드는 detect, detectForVideo가 공식 계약.
- JavaScript API에서 detectAsync는 제공되지 않음.
- 참고 링크:
  - https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/tasks/web/vision/core/vision_task_options.d.ts
  - https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/tasks/web/vision/hand_landmarker/hand_landmarker.ts
  - https://unpkg.com/@mediapipe/tasks-vision@0.10.32/vision.d.ts

## 코드에서 잘못되었던 점
- main.js에서 RunningMode를 런타임 import 대상으로 사용했던 점.
- VIDEO 전용 루프에서 detectAsync 호출 가정을 넣었던 점.
- requestAnimationFrame 루프 구조상 에러가 프레임마다 반복 출력된 점.

## 적용한 수정
- main.js import 정리: FilesetResolver, HandLandmarker만 사용.
- runningMode를 VIDEO 문자열로 통일.
- hand_tracking_runtime.js에서 detectForVideo 또는 detect 경로만 사용하도록 정리.
- 준비 완료 로그를 VIDEO mode 기준으로 정리.

## 근거 파일 위치
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:4
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:542
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:565
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:581
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/hand_tracking_runtime.js:333
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/hand_tracking_runtime.js:340
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/hand_tracking_runtime.js:372

## 검증 결과
- npm run build 성공.
- dev server 응답 확인 결과 main.js, hand_tracking_runtime.js에 RunningMode, detectAsync, LIVE_STREAM 문자열 없음.
- dist 폴더를 삭제 후 재빌드하여 이전 해시 번들 잔재 제거 완료.

## 다음 확인 방법
- 브라우저 DevTools에서 Disable cache 활성화 후 Hard Reload.
- 이전 콘솔 로그를 지우고 새로 발생하는 로그만 확인.
- 문제가 재현되면 에러 스택과 Network의 src/js/main.js 응답 상단 import 라인을 함께 확인.
