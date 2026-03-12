# Model Video Check

This folder contains a viewer that loads a trained checkpoint from `model/model_evaluation/pipelines/**/model.pt`, runs MediaPipe hand tracking on a selected video, and overlays the model prediction on the video.

## Entry Point

- `video_check_app.py`

## What It Does

- discovers trained runs automatically from `model/model_evaluation/pipelines`
- discovers videos automatically from `data/raw_data`
- shows dropdowns for run selection and video selection
- loads the selected checkpoint directly
- extracts hand landmarks with `hand_landmarker.task`
- rebuilds the same joint / bone / angle features used by training
- runs inference on CPU only and overlays prediction, confidence, and skeleton on the video

## Run

From the project root:

```bash
python3.11 model/model_evaluation/모델별영상체크/video_check_app.py
```

Optional direct launch without the dropdown UI:

```bash
python3.11 model/model_evaluation/모델별영상체크/video_check_app.py \
  --run-dir model/model_evaluation/pipelines/20260312_120000__man1_right_for_poc__man2_right_for_poc/cnn1d_tcn \
  --video data/raw_data/4_slow_right_man3.mp4
```

`--run-dir`에는 실제 run timestamp 폴더 또는 `latest.json`이 있는 모델 폴더 둘 다 넣을 수 있다.

## Controls

- `Space`: pause / resume
- `A` or `Left`: previous frame
- `D` or `Right`: next frame
- `R`: restart
- `Q` or `Esc`: quit

## Notes

- `hand_landmarker.task` must exist at the project root.
- PyTorch inference in this viewer is fixed to CPU only.
- The viewer first analyzes the whole video sequentially, then opens the playback window. This keeps the MediaPipe VIDEO mode behavior consistent with the existing extractor.
- Sequence models such as `cnn1d_tcn` and `transformer_embedding` need a warm-up window before they can emit normal predictions.
