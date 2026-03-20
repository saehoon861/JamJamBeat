# MediaPipe 경고 오인 RCA (Audio Callback Error)

작성일: 2026-03-19

## 증상
- 콘솔에 MediaPipe detection error가 반복 출력됨
- 실제 스택 최상단은 audio.js 668의 ReferenceError frequencyFromNote is not defined

## 실제 원인
- MediaPipe 추론 자체 실패가 아니라 오디오 경로 예외
- 전파 경로: audio.js 668 -> main.js 122 -> main.js 217 -> main.js 693 -> interaction_runtime.js 293 -> interaction_runtime.js 360 -> hand_tracking_runtime.js 290 -> main.js 723

## 반복 출력 이유
- hand tracking 루프가 requestAnimationFrame으로 계속 실행되며 같은 제스처 트리거가 반복됨
- 오디오 예외가 프레임마다 재발하고 동일 로그가 반복됨

## 적용 수정
- audio.js에 frequencyFromNote(note, fallback) 헬퍼 추가
- main.js 에러 라벨을 HandTracking loop error로 변경해 오인 방지

## 근거 파일
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/audio.js:218
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:723

## 검증
- npm run build 성공
- dev server 응답에서 audio.js helper 정의와 main.js 라벨 변경 확인

## 결론
- 이번 경고의 본질은 MediaPipe 엔진 실패가 아니라 오디오 콜백 오류였다
- 누락 함수 보완과 라벨 수정으로 원인과 로그 의미를 분리했다
