# Camera NotReadableError RCA

작성일: 2026-03-19

## 문제 요약
- 로그: [MediaPipe] initCamera:failed NotReadableError: Could not start video source
- 의미: 권한은 허용되었지만 OS 또는 브라우저 또는 장치 레벨에서 카메라를 열지 못한 상태

## 공식 근거
- MDN getUserMedia exceptions
  - NotReadableError 정의: 장치 접근 권한은 있으나 하드웨어 또는 OS 레벨 오류로 접근 실패
  - https://developer.mozilla.org/en-US/docs/Web/API/MediaDevices/getUserMedia#exceptions
- W3C Media Capture and Streams
  - getUserMedia 정의: https://w3c.github.io/mediacapture-main/#dom-mediadevices-getusermedia
  - 트랙 종료 및 장치 해제: https://w3c.github.io/mediacapture-main/#track-ended-by-the-user-agent
- MDN MediaStreamTrack stop
  - https://developer.mozilla.org/en-US/docs/Web/API/MediaStreamTrack/stop

## 기존 코드에서 부족했던 점
- initCamera에서 단순 2단계 시도 후 실패 시 권한 오류 문구로만 처리
- 기존 스트림 또는 srcObject를 재시도 전에 명시적으로 정리하지 않음
- unload 시점 정리도 cameraStream 위주라 srcObject 기반 점유 해제가 불완전할 수 있음

## 적용한 수정
1) 카메라 정리 유틸 추가
- stopCameraTracks(stream)
- clearCameraSource()
- 위치: /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js

2) NotReadableError 대응 강화
- detailed -> compat -> basic 순차 시도
- 각 시도 실패 시 이름과 메시지 로그 남김
- NotReadableError일 때 clearCameraSource 호출 후 짧은 지연을 두고 재시도

3) 사용자 메시지 세분화
- NotAllowedError 또는 SecurityError: 카메라 권한 안내
- NotFoundError: 카메라 장치 없음 안내
- NotReadableError: 다른 앱 점유 또는 장치 상태 확인 안내

4) 종료 시 정리 개선
- beforeunload에서 clearCameraSource 호출로 통합 정리

## 근거 코드 라인
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:474
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:487
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:502
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:535
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:544
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:588
- /home/roh/workspace/JAMMJAMM/JamJamBeat/frontend/src/js/main.js:738

## 검증
- npm run build 성공
- dev server 응답 main.js에서 initCamera 로그 문구 및 재시도 코드 반영 확인

## 운영 체크리스트
- 다른 앱에서 카메라 점유 중인지 확인
- 브라우저 탭 중복 사용 여부 확인
- 권한 재승인 후 Hard Reload
- 계속 실패하면 장치 관리자 또는 OS 카메라 개인정보 설정 확인
