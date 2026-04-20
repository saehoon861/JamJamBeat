# frontend_추론설정가이드.md - 기존 frontend 설정 변경 전달용 스펙

## 목적

이 문서는 다른 시스템의 AI에게 `기존 frontend`의 MediaPipe/추론 설정값 수정만 전달할 때 쓰는 최소 스펙이다.

## 기준 파일

- `frontend/src/js/main.js`
- `frontend/src/js/hand_tracking_runtime.js`
- `frontend/src/js/model_inference_sequence.js`
- `frontend/public/runtime_sequence/config.json`
- `model/model_evaluation/모델별영상체크/video_check_app.py`
- `model/model_evaluation/모델별영상체크/video_check_app_train_aligned.py`

## 유지할 값

- `runtime_sequence` 사용
- `bundle_id = pos_scale_mlp_sequence_delta_20260319_162806`
- `model_id = mlp_sequence_delta`
- `normalization_family = pos_scale`
- `seq_len = 8`
- `pos_scale + delta63` 유지
- `leftHandMirror = on`
- `splitHands = off`
- 왼손 추론 비활성 구조 유지

## 변경할 값

| 항목 | 기존 값 | 변경 값 |
|---|---:|---:|
| `inferWidth` | `96` | `0` |
| `inferFps` | `15` | `30` |
| `modelIntervalMs` | `150` | `60` |
| `numHands` | `2` | `1` |
| `minHandDetectionConfidence` | `0.25` | `0.50` |
| `minHandPresenceConfidence` | `0.25` | `0.50` |
| `minTrackingConfidence` | `0.25` | `0.50` |
| `tau` | `0.85` | `0.75` |

## 전달용 지시문

기존 frontend 코드에서 아래 값만 수정하라.

- `inferWidth: 96 -> 0`
- `inferFps: 15 -> 30`
- `modelIntervalMs: 150 -> 60`
- `numHands: 2 -> 1`
- `minHandDetectionConfidence: 0.25 -> 0.5`
- `minHandPresenceConfidence: 0.25 -> 0.5`
- `minTrackingConfidence: 0.25 -> 0.5`
- `tau: 0.85 -> 0.75`

그 외 모델 번들, sequence 구조, normalization family, left/right hand 구조는 유지하라.
