# 🎨 JamJamBeat Lottie 애니메이션 가이드

## 📋 목차
1. [개요](#개요)
2. [필요한 파일](#필요한-파일)
3. [애니메이션 제작 방법](#애니메이션-제작-방법)
4. [파일 위치 및 설정](#파일-위치-및-설정)
5. [애니메이션 상태별 동작](#애니메이션-상태별-동작)
6. [최적화 가이드](#최적화-가이드)
7. [문제 해결](#문제-해결)

---

## 개요

JamJamBeat에서는 각 악기(드럼, 실로폰, 탬버린, a 오브젝트)마다 **Lottie 애니메이션**을 사용하여 생동감 있는 움직임을 구현합니다.

### 왜 Lottie인가?
- ✅ **벡터 기반**: 해상도 독립적, 확대/축소해도 깨끗함
- ✅ **가벼움**: GIF보다 10배 이상 작은 파일 크기
- ✅ **디자이너 친화적**: After Effects에서 직접 제작 가능
- ✅ **상태 제어**: 속도, 루프, 세그먼트 재생 등 세밀한 제어

---

## 필요한 파일

각 악기마다 **1개의 JSON 파일**이 필요합니다:

```
frontend/public/assets/animations/
├── drum.json          # 드럼 애니메이션
├── xylophone.json     # 실로폰 애니메이션
├── tambourine.json    # 탬버린 애니메이션
└── a.json             # a 오브젝트 애니메이션
```

---

## 애니메이션 제작 방법

### 방법 1: Adobe After Effects (추천)
1. **After Effects**에서 애니메이션 제작
2. **Bodymovin** 플러그인 설치
   - [Bodymovin 다운로드](https://aescripts.com/bodymovin/)
3. 애니메이션 컴포지션 선택
4. `Window > Extensions > Bodymovin` 실행
5. **Export** 클릭하여 JSON 파일 생성

#### After Effects 제작 팁
- **컴포지션 크기**: 500x500px (정사각형 권장)
- **프레임 레이트**: 30fps
- **지속 시간**:
  - Idle: 2-4초 (루프)
  - Trigger: 0.5-1초 (원샷)
- **레이어 이름**: 영문으로 작성 (한글 X)
- **이펙트**: 기본 이펙트만 사용 (커스텀 플러그인 X)

### 방법 2: LottieFiles 에디터 (무료)
1. [LottieFiles.com](https://lottiefiles.com/editor) 접속
2. 온라인 에디터에서 애니메이션 제작
3. **Export as JSON** 클릭

### 방법 3: 기존 Lottie 수정
1. [LottieFiles Library](https://lottiefiles.com/) 에서 무료 애니메이션 다운로드
2. 색상/속도 등 커스터마이징
3. JSON 파일로 내보내기

---

## 파일 위치 및 설정

### 1. JSON 파일 배치
제작한 애니메이션을 아래 경로에 저장:
```
frontend/public/assets/animations/drum.json
```

### 2. 자동 로드 확인
코드에서 자동으로 다음 경로를 참조합니다:
```javascript
// frontend/src/js/instrument_animations.js
const INSTRUMENT_ANIMATION_CONFIG = {
  drum: {
    path: "/assets/animations/drum.json",
    // ...
  }
}
```

### 3. 테스트
브라우저 콘솔에서 로드 상태 확인:
```
[InstrumentAnimation] drum loaded successfully
```

---

## 애니메이션 상태별 동작

JamJamBeat는 **4가지 상태**를 지원합니다:

### 1. **IDLE** (기본 대기 상태)
- 언제: 아무도 터치하지 않을 때
- 특징: 천천히, 부드럽게 루프
- 속도: 0.6-0.9x
- 예시: 숨 쉬듯이 살짝 흔들리기, 반짝이기

### 2. **HOVER** (손이 근처에 있을 때)
- 언제: 손이 악기 근처에 왔을 때 (현재 미구현, 추후 확장 가능)
- 특징: 약간 빠르게, 기대하는 듯한 움직임
- 속도: 1.0-1.3x
- 예시: 손 쪽으로 살짝 기울기, 빛 증가

### 3. **TRIGGER** (연주될 때)
- 언제: 악기가 실제로 연주될 때
- 특징: 강렬하고 빠르게, 원샷 재생 후 IDLE 복귀
- 속도: 1.5-2.0x
- 예시: 튀어오르기, 회전, 폭발적인 반짝임

### 4. **FEVER** (피버 타임)
- 언제: 피버 모드가 활성화될 때
- 특징: 매우 빠르고 화려하게
- 속도: 2.0-2.3x
- 예시: 무지개 색상, 빠른 회전, 강한 빛

---

## 최적화 가이드

### 파일 크기 줄이기
- ✅ 불필요한 레이어 제거
- ✅ 복잡한 패스 단순화
- ✅ 이미지 임베딩 최소화
- ✅ 키프레임 수 최소화
- ❌ 고화질 이미지 삽입 금지

### 목표 파일 크기
- **권장**: 50KB 이하
- **최대**: 200KB

### 성능 테스트
```javascript
// 브라우저 콘솔에서 실행
performance.measure('animation-render');
```

---

## 문제 해결

### Q1. 애니메이션이 로드되지 않아요
**A1**: 콘솔 확인
```
[InstrumentAnimation] drum failed to load, using fallback
```
- JSON 파일 경로 확인: `/assets/animations/drum.json`
- JSON 유효성 검증: [JSONLint](https://jsonlint.com/)
- 네트워크 탭에서 404 에러 확인

### Q2. 애니메이션이 너무 느려요
**A2**: 설정 파일 수정
```javascript
// frontend/src/js/instrument_animations.js
drum: {
  states: {
    idle: { speed: 1.5 }, // 속도 증가
  }
}
```

### Q3. 이모지만 표시돼요
**A3**: Fallback 모드 (JSON 로드 실패)
- After Effects 버전 확인 (최신 Bodymovin 사용)
- 지원되지 않는 이펙트 제거
- 단순한 애니메이션으로 다시 제작

### Q4. 기존 이미지를 완전히 숨기고 싶어요
**A4**: 코드 수정
```javascript
// frontend/src/js/instrument_animations.js:93
existingImage.style.opacity = "0"; // 0.3 → 0
```

---

## 예제 애니메이션 구조

### 간단한 Idle 애니메이션 (JSON 예시)
```json
{
  "v": "5.7.4",
  "fr": 30,
  "ip": 0,
  "op": 90,
  "w": 500,
  "h": 500,
  "nm": "Drum Idle",
  "ddd": 0,
  "assets": [],
  "layers": [
    {
      "ty": 4,
      "nm": "Circle",
      "ks": {
        "o": { "a": 0, "k": 100 },
        "p": { "a": 0, "k": [250, 250] },
        "s": {
          "a": 1,
          "k": [
            { "t": 0, "s": [100, 100] },
            { "t": 45, "s": [110, 110] },
            { "t": 90, "s": [100, 100] }
          ]
        }
      }
    }
  ]
}
```

---

## 리소스

### 무료 Lottie 애니메이션
- [LottieFiles](https://lottiefiles.com/)
- [IconScout](https://iconscout.com/lottie-animations)
- [Lordicon](https://lordicon.com/)

### 제작 도구
- [After Effects](https://www.adobe.com/products/aftereffects.html)
- [LottieFiles Editor](https://lottiefiles.com/editor)
- [Haiku Animator](https://www.haikuforteams.com/)

### 플러그인
- [Bodymovin](https://aescripts.com/bodymovin/)
- [LottieFiles for After Effects](https://lottiefiles.com/plugins/after-effects)

---

## 추가 도움이 필요하신가요?

1. 콘솔 로그 확인: `[InstrumentAnimation]` 태그 검색
2. 브라우저 개발자 도구 > Network 탭에서 JSON 로드 확인
3. 간단한 테스트 애니메이션으로 먼저 시도
4. LottieFiles에서 무료 애니메이션을 다운로드하여 테스트

---

**Happy Animating! 🎵✨**
