# Frontend Refactoring Notes

## 작업 목적

`frontend/src/js/main.js` 가 1000줄을 넘기면서 한 파일 안에 너무 많은 책임이 섞여 있었습니다.

- 악기 배치 저장/로드
- 관리자 드래그 편집
- 배경 비디오 루프
- 피버 상태 관리
- 파티클 효과
- 사운드 매핑 로딩
- 실시간 손 추적 루프
- UI 이벤트 바인딩

이번 리팩토링의 목표는 기능을 바꾸지 않고, 책임 단위를 분리해서 유지보수 난도를 낮추는 것이었습니다.

## 변경 결과

### `main.js` 축소

- 변경 전: `1039` lines
- 1차 리팩토링 후: `688` lines
- 2차 리팩토링 후: `496` lines

핵심 루프와 앱 조립 코드는 `main.js` 에 남기고, 독립적인 책임은 별도 모듈로 분리했습니다.

### 새로 분리한 모듈

#### `frontend/src/js/instrument_layout.js`

담당 역할:

- 기본 악기 배치 값
- 배치 저장/불러오기
- 화면에 배치 적용
- 관리자 드래그 편집

분리 이유:

- localStorage 처리와 관리자 모드는 게임 플레이 루프와 직접 관계가 없습니다.
- UI 편집 로직은 단독 모듈로 보는 편이 재사용과 테스트에 유리합니다.

#### `frontend/src/js/scene_runtime.js`

담당 역할:

- 배경 비디오 루프
- 씬 모드 적용
- 피버 상태 관리

분리 이유:

- 화면 전체 상태 전환은 악기 충돌이나 제스처 판정과 별개 책임입니다.
- 배경/피버 관련 로직은 `main.js` 안에서 가장 길고 결합도가 높은 블록 중 하나였습니다.

#### `frontend/src/js/particle_system.js`

담당 역할:

- 파티클 생성
- 파티클 프레임 업데이트
- 재시작형 CSS 애니메이션 헬퍼

분리 이유:

- 시각 효과는 게임 규칙과 분리할수록 수정이 쉽습니다.
- 이후 이펙트 디자인만 바꾸고 싶을 때 진입점이 명확해집니다.

#### `frontend/src/js/sound_mapping.js`

담당 역할:

- 기본 사운드 매핑
- 저장된 매핑 로딩
- 악기별 사운드 프로필 선택

분리 이유:

- 사운드 설정 로딩은 오디오 재생과는 다르지만, 기존엔 `main.js` 에 섞여 있었습니다.
- 설정 계층을 모듈로 빼두면 관리자 페이지와도 규칙을 맞추기 쉬워집니다.

#### `frontend/src/js/interaction_runtime.js`

담당 역할:

- 시작 버튼 hover 처리
- 악기 충돌 처리
- 제스처 반응 처리
- 손 커서 위치 표시

분리 이유:

- 사용자 상호작용은 프레임 추적 루프와 구분되는 별도 책임입니다.
- 터치 모드와 제스처 모드가 한 파일 안에서 뒤엉키던 부분을 분리해 변경 범위를 줄였습니다.

#### `frontend/src/js/hand_tracking_runtime.js`

담당 역할:

- MediaPipe 프레임 루프
- 랜드마크 캐시 관리
- 손 감지 성공/실패 분기
- 상호작용 런타임 호출 orchestration

분리 이유:

- `predict()` 는 이전 구조에서 가장 결합도가 높은 함수였습니다.
- 프레임 루프만 독립시켜야 렌더링, 상호작용, 상태 관리의 경계를 분명하게 만들 수 있습니다.

## 리팩토링 과정

### 1. 책임 분리 기준 정의

아래 기준으로 분리했습니다.

- 브라우저 상태 저장: `instrument_layout.js`, `sound_mapping.js`
- 화면 분위기/상태 전이: `scene_runtime.js`
- 시각 효과: `particle_system.js`
- 실시간 orchestration: `main.js`

### 2. `main.js` 를 "조립 파일"로 정리

`main.js` 안에서는 아래만 남기도록 정리했습니다.

- DOM 참조 수집
- MediaPipe 초기화
- 손 추적 루프
- 충돌/제스처 반응
- 버튼 이벤트 바인딩

즉, `main.js` 는 이제 세부 구현보다 "어떤 모듈을 어떤 순서로 연결하는지"가 더 잘 보이는 구조입니다.

### 3. 상태 객체를 팩토리 형태로 이동

아래 두 상태성 로직은 팩토리 패턴으로 옮겼습니다.

- `createParticleSystem(...)`
- `createFeverController(...)`

이유:

- 내부 상태를 외부 전역 변수 대신 모듈 내부에 숨길 수 있습니다.
- `main.js` 에서 필요한 동작만 메서드로 받아 쓸 수 있습니다.

### 4. `predict()` 와 interaction 로직 분리

2차 리팩토링에서는 다음 두 덩어리를 추가 분리했습니다.

- `createInteractionRuntime(...)`
- `createHandTrackingRuntime(...)`

분리 후 구조는 아래처럼 바뀌었습니다.

- `main.js`: 앱 조립, 초기화, 의존성 연결
- `interaction_runtime.js`: 사용자 반응 규칙
- `hand_tracking_runtime.js`: 프레임 기반 손 추적 루프

이 단계까지 진행하면서 `main.js` 를 500줄 이하로 줄였습니다.

## 검증

실행 환경에서 브라우저 테스트까지는 하지 못했고, 최소 문법 검사는 수행했습니다.

실행한 검사:

```bash
node --check frontend/src/js/main.js
node --check frontend/src/js/instrument_layout.js
node --check frontend/src/js/scene_runtime.js
node --check frontend/src/js/particle_system.js
node --check frontend/src/js/sound_mapping.js
node --check frontend/src/js/interaction_runtime.js
node --check frontend/src/js/hand_tracking_runtime.js
```

결과:

- 모두 문법 오류 없이 통과

## 후속 검증 보강

리팩토링 이후 스모크 테스트 파일도 현재 구조에 맞게 정리했습니다.

- `frontend/test_theme.js`
  - `.theme-mode-card` 선택자로 수정
  - 클릭 후 `mode=calm` 이동 확인 추가
- `frontend/test_index.js`
  - 메인 화면 주요 버튼/악기 요소 존재 확인
  - 시작 버튼 클릭 후 landing overlay 숨김 확인

실제 브라우저 자동화 실행은 시스템 라이브러리 부족으로 이 환경에서 완료하지 못했습니다.

## 이번 턴에서 일부러 안 건드린 것

아래는 아직 남겨둔 영역입니다.

- 카메라/MediaPipe 초기화 공통화
- `theme.js` 와 `main.js` 의 중복 초기화 로직 통합
- 사운드 매핑 규칙을 `admin.js` 와 공용 계층으로 통일
- `performance.js` 의 라벨 정규화 로직과 제스처 표기 유틸 통합

이 부분은 다음 리팩토링 단계에서 다루는 편이 안전합니다. 이번 턴은 `main.js` 핵심 병목을 먼저 해소하는 데 집중했습니다.

## 다음 추천 작업

우선순위는 아래 순서가 적절합니다.

1. `main.js` 에서 제스처/악기 반응 로직을 `interaction_runtime.js` 로 분리
2. `theme.js` 와 `main.js` 의 MediaPipe/카메라 초기화 코드를 공통 유틸로 통합
3. `performance.js` 와 제스처 라벨 정규화 규칙을 공용 유틸로 정리
4. `admin.js` 와 런타임의 사운드 매핑 규칙을 하나의 공용 모듈로 정리
