# landmark-extractor-agent

## 책임

- `raw_data/*.mp4` 에서 MediaPipe 손 랜드마크 추출
- `landmark_data/*.csv` 생성
- 외부 asset 없이 `model/동작테스트/hand_landmarker.task` 사용

## 고정 설정

- `running_mode = VIDEO`
- `num_hands = 1`
- `min_hand_detection_confidence = 0.5`
- `min_hand_presence_confidence = 0.5`
- `min_tracking_confidence = 0.5`

## 입출력

- 입력: `raw_data/7_grab.mp4`
- 출력: `landmark_data/7_grab.csv`
- 스키마: `frame_idx, timestamp, gesture, x0..z20`

## 완료 기준

- `7_grab.csv` 생성
- 프레임 수와 타임스탬프가 원본 영상과 대응
- 외부 경로 import/path resolve 없음
