# JamJamBeat Frontend 버그 수정 기록

> 2026-03-17 수정

---

## 1. 오브젝트 검은 박스 표시

**증상**: 메인 페이지에서 캐릭터(고슴도치) 뒤에 검은 반투명 사각형이 보임

**원인**: instrument_animations.js의 Lottie 초기화 시 .instrument-art 셀렉터로 video 요소를 잡아서 opacity 0.3으로 설정. .mov 영상의 검은 배경이 반투명 상태로 노출됨.

**수정 파일**: src/js/instrument_animations.js line 101

```diff
- const existingImage = targetElement.querySelector(.instrument-art);
+ const existingImage = targetElement.querySelector(img.instrument-art);
```

---

## 2. Lottie 컨테이너 중복 생성

**증상**: 같은 DOM 요소에 Lottie 컨테이너가 2개 겹쳐져 렌더링 이상

**원인**: instrumentElements에서 drum과 a가 동일한 DOM 요소 instrumentA를 공유. initAnimation이 같은 요소에 2번 호출되어 컨테이너가 중복 생성됨.

**수정 파일**: src/js/main.js line 681-688

```diff
+ const animatedElements = new Set();
  instruments.forEach((instrument) => {
+   if (instrument.el && !animatedElements.has(instrument.el)) {
      animationManager.initAnimation(instrument.id, instrument.el);
+     animatedElements.add(instrument.el);
    }
  });
```

---

## 3. 오브젝트 터치 시 사라짐

**증상**: 관리자 배치 모드에서 오브젝트를 클릭/드래그하면 사라짐

**원인**: .instrument-hedgehog1 .instrument-art에 CSS SVG 필터 filter: url(#remove-white) 적용됨. Chrome에서 SVG 필터 참조가 있는 요소를 드래그/리플로우하면 필터 참조를 잃어 렌더링이 깨지는 알려진 버그. JS drawInstrumentAFrame이 이미 픽셀 단위로 검은 배경을 제거하고 있어 해당 CSS 필터는 중복이었음.

**수정 파일**: src/styles/main.css line 282

```diff
- filter: url(#remove-white);
+ /* filter: url(#remove-white); -- JS가 이미 검은배경 제거 처리함 */
```

---

## 참고: Lottie 애니메이션 JSON 파일 부재

**현황**: /assets/animations/ 디렉터리에 drum.json, a.json 등 Lottie JSON 파일이 없음. gitkeep만 존재. Lottie가 빈 SVG를 렌더링하여 추가적인 시각 이상 가능성 있음.

**조치 필요**: 실제 Lottie JSON 파일 추가 또는 Lottie 초기화 로직에서 파일 존재 확인 후 스킵 처리

---

## 참고: 테스트 파일 정리

삭제된 파일:
- test_kids_sounds.html
- test_index.js
- test_theme.js

---

## 4. ONNX 모델 로드 실패 — 외부 데이터 파일 마운트 불가

> 2026-03-19 수정

**증상**: 앱 시작 시 모델 추론 초기화에서 아래 에러 발생, 손동작 인식 불가

```
[ModelInferenceONNX] ❌ 모델 로드 실패: Error: Can't create a session.
ERROR_CODE: 1, ERROR_MESSAGE: Deserialize tensor embed.1.bias failed.
Failed to load external data file "model.onnx.data",
error: Module.MountedFiles is not available.
```

**원인**: ONNX 모델이 가중치를 외부 파일로 분리 저장하는 구조(`model.onnx` 2.8KB + `model.onnx.data` 134KB). `InferenceSession.create(url)` 호출 시 WASM 런타임이 모델 내부에 기록된 `model.onnx.data` 참조를 읽고 자동으로 로드하려 하지만, 브라우저 환경에는 `Module.MountedFiles` 가상 파일시스템이 없어 파일을 찾지 못함.

**수정 파일**: `src/js/model_inference.js` line 202

```diff
  onnxSession = await ort.InferenceSession.create(MODEL_PATH, {
    executionProviders: ["wasm"],
-   graphOptimizationLevel: "all"
+   graphOptimizationLevel: "all",
+   externalData: [
+     {
+       path: "model.onnx.data",
+       data: "/runtime/model.onnx.data"
+     }
+   ]
  });
```

**설명**:
- `path`: 모델 protobuf 내부에 기록된 외부 데이터 파일명 (모델이 참조하는 이름)
- `data`: 브라우저가 실제로 fetch할 URL 경로 (`public/runtime/model.onnx.data`가 서빙되는 경로)
- `externalData` 옵션은 onnxruntime-web v1.17+에서 지원 (현재 v1.24.3 사용 중)
