# pos_scale_mlp_sequence_delta_20260319_162806

이 번들은 `pos_scale` 정규화 데이터셋으로 학습된 `mlp_sequence_delta` 시퀀스 모델 전달본입니다.

중요한 점:
- 이 모델은 **raw MediaPipe landmark를 그대로 넣는 모델이 아닙니다.**
- 학습 시 `pos_scale` 정규화와 `delta(프레임 간 1차 차분)`가 적용된 입력으로 학습되었습니다.
- 따라서 추론 시에도 **동일한 정규화 규칙**을 반드시 적용해야 합니다.

이 번들은 그 전처리를 **번들 내부에서 수행**합니다.

즉 caller는 아래 중 하나만 넘기면 됩니다.
- shape `[8, 63]` raw landmark sequence
- shape `[8, 21, 3]` raw landmark sequence

번들 내부 처리:
1. 각 프레임에 `pos_scale = (pts - pts[0]) / ||pts[9] - pts[0]||` 적용
2. 프레임별 63차원 flatten
3. delta 63차원 생성
4. `[joint63, delta63]` 결합
5. 최종 `[8, 126]` 입력으로 모델 추론

기본 사용 예시:

```python
from pathlib import Path
import numpy as np

from runtime import load_bundle, predict

bundle_dir = Path(__file__).resolve().parent
bundle = load_bundle(bundle_dir, device="cpu", backend="torch")

raw_sequence = np.zeros((8, 21, 3), dtype=np.float32)
result = predict(bundle, raw_sequence, tau=0.90)
print(result)
```

ONNX backend를 쓰려면:

```python
bundle = load_bundle(bundle_dir, device="cpu", backend="onnx")
result = predict(bundle, raw_sequence)
```

추가 설명은 [runtime/FRONTEND_INTEGRATION.md](/home/user/projects/JamJamBeat/model/선정모델폴더/pos_scale_mlp_sequence_delta_20260319_162806/runtime/FRONTEND_INTEGRATION.md)를 보면 됩니다.
