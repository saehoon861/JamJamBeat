# 영상추론비교

`frontend`, `frontend-test`, `model` 기존 파일을 건드리지 않고, 현재 active sequence runtime과 viewer-style semantics를 같은 prerecorded video 기준으로 비교하는 워크스페이스입니다.

## 목적

- 같은 모델 번들을 쓰는 상태에서 drift 원인을 분리
- `raw model prediction`
- `frontend final gesture`
- `live capture 이전 prerecorded video 차이`
를 분리해서 확인

## 기본 비교 기준

- bundle: `pos_scale_mlp_sequence_delta_20260319_162806`
- model: `mlp_sequence_delta`
- checkpoint fingerprint: `2f9e81b7a142a0cd92707eb903fe4880f55cbcf920676f6433104ddf8c006d09`
- default video: `/home/user/projects/JamJamBeat/data/raw_data/2_fast_right_man1.mp4`

## 실행

```bash
cd /home/user/projects/JamJamBeat-model3/영상추론비교
uv run python run_comparison.py
```

기본 산출물:

- `artifacts/viewer_probe.json`
- `artifacts/viewer_probe_tau085.json`
- `artifacts/frontend_probe.json`
- `artifacts/comparison_stats.json`
- `artifacts/summary.md`

## probe 설명

- `viewer_probe/run_viewer_probe.py`
  - `video_check_app_train_aligned.py`의 핵심 sequence semantics를 현재 bundle 기준으로 재현
  - full frame
  - MediaPipe thresholds `0.5 / 0.5 / 0.5`
  - `num_hands=1`
  - every frame inference

- `frontend_probe/run_frontend_probe.py`
  - `frontend/src/js/model_inference_sequence.js` + `gestures.js`의 핵심 semantics 재현
  - infer width `96`
  - infer fps `15`
  - model interval `150ms`
  - MediaPipe thresholds `0.25 / 0.25 / 0.25`
  - `num_hands=2`
  - raw model prediction과 final gesture를 분리 기록

## 주의

- frontend probe는 `actual frontend semantics`를 우선합니다.
- 따라서 handedness 때문에 selected hand가 `left`로 잡히면 실제 frontend처럼 disabled 상태가 기록될 수 있습니다.
- 이건 버그가 아니라 drift 원인 후보로 남기기 위한 의도된 동작입니다.
