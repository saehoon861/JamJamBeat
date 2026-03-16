# 🎵 JamJamBeat 악기 애니메이션 시스템

## 📌 개요

각 악기에 Lottie 애니메이션을 적용하여 **살아 움직이는 느낌**을 구현한 시스템입니다.

## 🎯 주요 기능

### ✨ 자동 상태 전환
- **IDLE**: 기본 대기 상태 (호흡하는 듯한 애니메이션)
- **TRIGGER**: 악기 연주 시 강렬한 반응
- **FEVER**: 피버 타임 시 화려한 애니메이션

### 🎨 특징
- ✅ 각 악기마다 독립적인 Lottie 애니메이션
- ✅ 연주 시 자동 트리거
- ✅ 피버 모드 동기화
- ✅ JSON 로드 실패 시 이모지 폴백
- ✅ 성능 최적화 (SVG 렌더러)

## 📁 파일 구조

```
frontend/
├── src/js/
│   └── instrument_animations.js    # 애니메이션 매니저 모듈
├── public/assets/animations/
│   ├── drum.json                   # 드럼 애니메이션
│   ├── xylophone.json              # 실로폰 애니메이션
│   ├── tambourine.json             # 탬버린 애니메이션
│   └── a.json                      # a 오브젝트 애니메이션
└── docs/
    ├── LOTTIE_ANIMATION_GUIDE.md   # 애니메이션 제작 가이드
    └── INSTRUMENT_ANIMATION_README.md  # 이 문서
```

## 🚀 사용 방법

### 1. 기본 사용 (이미 적용됨)

코드에서 자동으로 초기화 및 실행됩니다:

```javascript
// main.js에서 자동 초기화
const animationManager = createInstrumentAnimationManager();

// 악기별 애니메이션 로드
instruments.forEach((instrument) => {
  animationManager.initAnimation(instrument.id, instrument.el);
});

// 악기 연주 시 자동 트리거
instrument.onHit(note) {
  animationManager.trigger(this.id); // 자동 실행
}
```

### 2. 수동 제어 (고급)

```javascript
// 특정 악기 애니메이션 트리거
animationManager.trigger("drum");

// 상태 변경
animationManager.setState("xylophone", "idle");

// Hover 효과
animationManager.hover("tambourine", true);

// 피버 모드 전환
animationManager.setFeverMode(true);

// 거리 기반 반응 (손 추적 연동)
animationManager.updateProximity("a", 150, 200);

// 일시정지
animationManager.setPaused("drum", true);
```

## 🎨 애니메이션 커스터마이징

### 속도 조절
```javascript
// frontend/src/js/instrument_animations.js
const INSTRUMENT_ANIMATION_CONFIG = {
  drum: {
    states: {
      idle: { speed: 0.8 },    // 느리게
      trigger: { speed: 2.0 }  // 빠르게
    }
  }
}
```

### 폴백 이모지 변경
```javascript
const INSTRUMENT_ANIMATION_CONFIG = {
  drum: {
    fallbackEmoji: "🥁" // 드럼 이모지로 변경
  }
}
```

### 기존 이미지 표시 설정
```javascript
// instrument_animations.js:93
existingImage.style.opacity = "0.3"; // 반투명
existingImage.style.opacity = "0";   // 완전 숨김
existingImage.style.opacity = "1";   // 그대로 유지
```

## 📊 상태 다이어그램

```
IDLE (대기)
   ↓ 손 근처
HOVER (반응)
   ↓ 연주
TRIGGER (폭발)
   ↓ 완료 후
IDLE (복귀)

[피버 모드]
모든 악기 → FEVER 상태
```

## 🔧 디버깅

### 콘솔 로그 확인
```
[InstrumentAnimation] drum loaded successfully
[InstrumentAnimation] xylophone loaded successfully
[InstrumentAnimation] tambourine loaded successfully
[InstrumentAnimation] a loaded successfully
```

### 로드 실패 시
```
[InstrumentAnimation] drum failed to load, using fallback
```
→ 브라우저 개발자 도구 > Network 탭에서 404 확인
→ JSON 파일 경로: `/assets/animations/drum.json`

### 애니메이션 상태 확인
```javascript
// 브라우저 콘솔에서 실행
animationManager.getState("drum");    // 현재 상태 확인
animationManager.isLoaded("drum");    // 로드 여부 확인
```

## 🎬 샘플 애니메이션

제공된 샘플 파일:
- **drum.json**: 호흡하는 듯한 원형 애니메이션 (주황색)
- **xylophone.json**: 파도치는 사각형 애니메이션 (청록색)
- **tambourine.json**: 회전하는 별 애니메이션 (노란색)
- **a.json**: 튀어오르는 하트 애니메이션 (핑크색)

## 🛠 새 애니메이션 추가하기

1. **After Effects 또는 LottieFiles에서 제작**
   - [제작 가이드 참고](./LOTTIE_ANIMATION_GUIDE.md)

2. **JSON 파일 저장**
   ```
   frontend/public/assets/animations/[악기ID].json
   ```

3. **브라우저 새로고침**
   - 자동으로 새 애니메이션 로드됨

## ⚡ 성능 최적화

### 이미 적용된 최적화
- ✅ SVG 렌더러 사용 (Canvas보다 가벼움)
- ✅ 애니메이션 인스턴스 재사용
- ✅ 로드 실패 시 graceful fallback
- ✅ Progressive load 활성화

### 추가 최적화 (필요 시)
```javascript
// 특정 악기 애니메이션 비활성화
animationManager.setPaused("drum", true);

// 완전 정리
animationManager.destroy();
```

## 📦 의존성

```json
{
  "dependencies": {
    "lottie-web": "^5.12.2"
  }
}
```

## 🔗 관련 문서

- [Lottie 애니메이션 제작 가이드](./LOTTIE_ANIMATION_GUIDE.md)
- [LottieFiles 공식 문서](https://lottiefiles.github.io/lottie-docs/)
- [After Effects Bodymovin 플러그인](https://aescripts.com/bodymovin/)

## 🎯 향후 확장 가능성

### Hover 상태 구현
손이 악기 근처에 왔을 때 반응하도록 확장 가능:

```javascript
// hand_tracking_runtime.js에서 추가
const distance = calculateDistance(handPosition, instrumentPosition);
if (distance < 200) {
  animationManager.hover(instrumentId, true);
} else {
  animationManager.hover(instrumentId, false);
}
```

### 세그먼트 재생
특정 구간만 재생하도록 설정:

```javascript
drum: {
  states: {
    trigger: {
      segments: [0, 30], // 0~30 프레임만 재생
      loop: false
    }
  }
}
```

### 커스텀 이벤트
```javascript
anim.addEventListener('complete', () => {
  console.log('Animation completed!');
});
```

## 🐛 알려진 제한사항

1. **After Effects 이펙트 제한**
   - 일부 고급 이펙트는 Lottie에서 미지원
   - 권장: 기본 Transform, Shape 레이어만 사용

2. **파일 크기**
   - 복잡한 애니메이션은 파일이 커질 수 있음
   - 권장: 50KB 이하, 최대 200KB

3. **모바일 성능**
   - 너무 많은 레이어는 성능 저하 가능
   - 권장: 레이어 10개 이하

## 🤝 기여 및 개선

새로운 애니메이션을 추가하거나 개선하려면:

1. `frontend/public/assets/animations/` 에 JSON 추가
2. `INSTRUMENT_ANIMATION_CONFIG`에 설정 추가 (필요 시)
3. 테스트 후 커밋

---

**Happy Coding! 🎨🎵**
