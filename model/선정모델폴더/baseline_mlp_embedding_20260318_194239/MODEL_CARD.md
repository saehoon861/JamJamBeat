# MODEL_CARD - baseline / mlp_embedding / 20260318_194239

## 요약
이 번들은 `baseline` 정규화 계열의 `mlp_embedding` 프레임 모델입니다.  
raw MediaPipe hand landmark 63차원 입력을 받아 7개 제스처 클래스를 분류합니다.

선정 원본 run:
- suite: `20260318_194005__baseline`
- model: `mlp_embedding`
- run: `20260318_194239`

## 왜 이 모델을 선택했는가
`baseline` suite 안에서 전체 균형이 좋고, 정확도와 macro F1이 높으면서 class0 false positive도 과도하지 않은 편이라 메인스트림 전달 후보로 적합했습니다.

선정 기준 수치:
- `accuracy = 0.7747`
- `macro_f1 = 0.7948`
- `class0_fpr = 0.0884`
- `class0_fnr = 0.5929`
- `epochs_ran = 20`

## 입력 / 출력
입력:
- 한 프레임당 `63`개 float
- 순서: `x0,y0,z0,...,x20,y20,z20`
- 입력 타입: `float32`

출력:
- 클래스 7개에 대한 softmax 확률
- 기본 동작은 pure top-1 argmax
- 선택적으로 `tau`를 넣으면 `max_prob < tau` 인 경우 `neutral` 로 강제 가능

클래스 순서:
1. `neutral`
2. `fist`
3. `open_palm`
4. `V`
5. `pinky`
6. `animal`
7. `k-heart`

## 아키텍처
`MLPEmbedding` 구조:
- `63 -> 128` projection + LayerNorm + GELU + Dropout
- `128 -> 128 -> 64 -> 7` classifier head

즉 입력 landmark를 먼저 learnable embedding으로 투영한 뒤 분류하는 경량 MLP입니다.

## 제한사항
- 이 모델은 `baseline/raw landmark` 전용입니다.
- sequence windowing, 이미지 렌더링, bone/angle feature는 포함하지 않습니다.
- `neutral` 검출 성능은 완전히 해결된 상태가 아니며 `class0_fnr = 0.5929` 이라 실서비스에서는 `tau` 또는 상위 레벨 후처리를 함께 검토하는 편이 좋습니다.
- MediaPipe landmark 품질이 흔들리는 구간에서는 공통 입력 병목의 영향을 받을 수 있습니다.

## 함께 확인할 파일
- `MANIFEST.json`
- `runtime/input_spec.json`
- `artifacts/source/selected_metrics.json`
- `artifacts/source/checkpoint_verification.json`
