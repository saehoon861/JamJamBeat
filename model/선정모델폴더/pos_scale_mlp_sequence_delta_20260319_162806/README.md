# pos_scale_mlp_sequence_delta_20260319_162806

이 번들은 `pos_scale` 정규화 데이터셋으로 학습된 `mlp_sequence_delta` 시퀀스 모델의 프론트엔드 전달본입니다.

중요한 점:
- 이 모델은 **브라우저 프론트엔드에서 ONNX로 직접 추론하는 용도**입니다.
- 서버 API나 Python 추론 API를 공식 경로로 사용하지 않습니다.
- 공식 진입점은 [runtime/inference.ts](/home/user/projects/JamJamBeat/model/선정모델폴더/pos_scale_mlp_sequence_delta_20260319_162806/runtime/inference.ts) 입니다.
- 이 모델은 **raw MediaPipe landmark를 그대로 바로 넣는 모델이 아닙니다.**
- 학습 시 `pos_scale` 정규화와 `delta(프레임 간 1차 차분)`가 적용된 입력으로 학습되었습니다.
- 따라서 추론 시에도 **동일한 정규화 규칙**을 반드시 적용해야 합니다.

이 번들은 그 전처리를 **번들 내부에서 수행**합니다.

즉 caller는 단일 raw frame만 넘기면 됩니다.
- shape `[63]` raw landmark frame
- shape `[21, 3]` raw landmark frame

번들 내부 처리:
1. 각 프레임에 `pos_scale = (pts - pts[0]) / ||pts[9] - pts[0]||` 적용
2. 최근 8프레임 buffer 유지
3. 프레임별 63차원 flatten
4. delta 63차원 생성
5. `[joint63, delta63]` 결합
6. 최종 `[8, 126]` 입력으로 ONNX 추론

실시간 상태 처리:
- 1~7프레임: `warmup`
- hand 미검출: `no_hand` + buffer reset
- invalid frame: `invalid_frame` + buffer reset
- non-neutral confidence `< 0.85`: `tau_neutralized`

기본 사용 예시:

```ts
import { createSequenceDeltaPredictor } from "./runtime/inference";

const predictor = createSequenceDeltaPredictor();
await predictor.load();

const frameResult = await predictor.pushFrame(rawFrame);
const noHandResult = predictor.pushNoHand();
```

추가 설명은 [runtime/FRONTEND_INTEGRATION.md](/home/user/projects/JamJamBeat/model/선정모델폴더/pos_scale_mlp_sequence_delta_20260319_162806/runtime/FRONTEND_INTEGRATION.md)를 보면 됩니다.
