# CURRENT_STATE

## 잠근 비교 기준

- viewer/runtime bundle: `/home/user/projects/JamJamBeat-model3/runtime_mlp_sequence_delta 20260319_162614_pos_scale 20260319_162806`
- frontend runtime bundle: `/home/user/projects/JamJamBeat-model3/frontend/pos_scale_mlp_sequence_delta_20260319_162806/runtime`
- 두 번들의 `config.json`, `class_names.json`, `input_spec.json`, `model.pt`, `model.onnx`는 동일 번들로 확인됨
- 공통 fingerprint:
  - `checkpoint_fingerprint = 2f9e81b7a142a0cd92707eb903fe4880f55cbcf920676f6433104ddf8c006d09`

## checkpoint mismatch 상태

- 이전 가설이던 `checkpoint mismatch`는 폐기
- 현재 비교의 핵심은 같은 모델에서 발생하는 runtime drift 분리

## 남은 차이 후보

1. `tau`
   - frontend: `0.85`
   - viewer baseline: `0.90`

2. MediaPipe 설정
   - frontend: `0.25 / 0.25 / 0.25`, `num_hands=2`
   - viewer: `0.5 / 0.5 / 0.5`, `num_hands=1`

3. 입력 cadence / resolution
   - frontend: `inferWidth=96`, `inferFps=15`, `modelIntervalMs=150`
   - viewer: full frame, source video every frame

4. no-hand reset
   - frontend: inference tick 기준 reset + stale hold 영향
   - viewer: no-hand frame 즉시 sequence clear

5. final gesture layer
   - frontend만 `mapModelToResult() + stabilize()` 적용

## 현재 기본 비교 영상

- `/home/user/projects/JamJamBeat/data/raw_data/2_fast_right_man1.mp4`
