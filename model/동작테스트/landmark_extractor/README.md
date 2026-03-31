# 동작테스트 랜드마크 추출기

## 워크스페이스 구조

이 추출기는 `model/동작테스트` 내부 경로만 사용합니다.

```text
model/동작테스트/
├── raw_data/
│   └── 7_grab.mp4
├── landmark_data/
├── hand_landmarker.task
└── landmark_extractor/
```

## 실행 방법

### 0. 시스템 의존성 설치 (최초 1회, Ubuntu/WSL)

MediaPipe 실행에 필요한 시스템 라이브러리를 먼저 설치합니다.

```bash
sudo apt-get update && sudo apt-get install -y libgles2 libegl1 libglib2.0-0
```

### 1. 파이썬 환경 설치

```bash
cd /home/user/projects/JamJamBeat/model/동작테스트/landmark_extractor
uv sync
```

### 2. 랜드마크 추출 실행

```bash
cd /home/user/projects/JamJamBeat/model/동작테스트/landmark_extractor
uv run python main.py
```

## 고정 설정

- `running_mode = VIDEO`
- `num_hands = 1`
- `min_hand_detection_confidence = 0.5`
- `min_hand_presence_confidence = 0.5`
- `min_tracking_confidence = 0.5`

## 결과물

- 저장 경로: `model/동작테스트/landmark_data/7_grab.csv`
- CSV 컬럼: `frame_idx`, `timestamp`, `gesture`, `x0..z20`

## 확인 포인트

- 실행 시 pending video 로 `7_grab.mp4` 가 보여야 합니다.
- 저장 후 `model/동작테스트/landmark_data/7_grab.csv` 가 생성되어야 합니다.
