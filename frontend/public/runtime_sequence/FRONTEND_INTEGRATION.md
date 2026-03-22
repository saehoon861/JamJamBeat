# FRONTEND_INTEGRATION.md - pos_scale sequence delta 프론트엔드 ONNX 연동 메모

## 핵심 원칙

이 모델은 `pos_scale` 정규화된 데이터셋으로 학습되었다.  
따라서 추론도 동일한 정규화를 반드시 적용해야 한다.

이 번들의 공식 런타임 진입점은 `runtime/inference.ts` 하나다.  
별도 `normalizer.ts`나 서버 API는 없다.

프론트엔드는 raw MediaPipe landmark frame만 넘기면 되고,  
`inference.ts`가 내부에서 정규화, delta 생성, warmup, no-hand reset, tau 후처리를 모두 처리한다.

기본 `tau`는 `0.85`다.

## Caller 입력 계약

허용 frame 입력:

- `[63]`
- `[21, 3]`

의미:

- 단일 raw MediaPipe hand landmark frame
- 프레임별 좌표 순서는 `x0,y0,z0,...,x20,y20,z20`

## 내부 상태 머신

- valid hand frame이 들어오면 buffer에 추가
- buffer 길이 `< 8`이면 `warmup`
- hand 미검출이면 `pushNoHand()`를 호출하고 buffer를 비운다
- 8프레임이 차면 ONNX 추론
- non-neutral 예측이면서 confidence `< 0.85`면 `tau_neutralized`
- invalid frame이면 `invalid_frame`을 반환하고 buffer를 비운다

## 내부 전처리

```text
pos_scale(frame) = (frame - frame[0]) / ||frame[9] - frame[0]||
```

단, `||frame[9] - frame[0]||`가 너무 작으면 viewer와 동일하게  
`(frame - frame[0])`까지만 적용하고 스케일은 `1.0`으로 둔다.

그 다음 내부적으로:

1. 프레임별 63차원 flatten
2. 최근 8프레임 유지
3. delta 63차원 생성
4. `[joint63, delta63]` 결합
5. 최종 모델 입력 `[8, 126]`

즉 프론트에서 `pos_scale`를 따로 구현하면 중복 처리된다.  
공식 연동 경로에서는 raw frame만 유지하는 것이 맞다.

## TypeScript 예시

```ts
import { createSequenceDeltaPredictor } from "./inference";

const predictor = createSequenceDeltaPredictor();
await predictor.load();

const result = await predictor.pushFrame(rawFrame63Or21x3);
console.log(result.status, result.predLabel, result.confidence);

const noHandResult = predictor.pushNoHand();
console.log(noHandResult.status);
```

## 디버깅 포인트

- `model.onnx`와 `model.onnx.data`는 반드시 같은 정적 경로에 함께 있어야 한다.
- 1~7번째 valid frame은 `warmup`이어야 한다.
- 첫 delta frame은 항상 0이어야 한다.
- `pushNoHand()` 직후 다음 frame은 다시 `warmup`부터 시작해야 한다.
- `feature_order.json`은 내부 126차원 feature 정의 파일이다.
- 공식 경로는 프론트엔드 ONNX helper이며, 백엔드 API는 없다.
