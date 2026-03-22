# 📋 동작테스트 수동 데이터 라벨링 도구

## 워크스페이스 구조

이 툴은 `동작테스트` 내부 경로만 사용합니다.

```text
동작테스트/
├── raw_data/
│   └── 7_grab.mp4
├── labeled_data/
└── labeling_tool/
```

## 준비

라벨링 대상 영상은 아래 위치에 있어야 합니다.

```text
동작테스트/raw_data/7_grab.mp4
```

파일명 규칙은 `{class_id}_{label}.mp4` 입니다.

- 현재 대상: `7_grab.mp4`
- target class: `7_Grab`

## 실행 방법

### 1. 시스템 의존성 설치 (최초 1회, Ubuntu/WSL)

```bash
sudo apt-get update && sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0 libsm6 libice6
```

### 2. 파이썬 환경 설치

```bash
cd /home/user/projects/JamJamBeat-model3/동작테스트/labeling_tool
uv sync
```

### 3. 라벨링 툴 실행

```bash
cd /home/user/projects/JamJamBeat-model3/동작테스트/labeling_tool
uv run python main.py
```

실행하면 `../raw_data/` 안의 아직 라벨링하지 않은 `.mp4` 파일을 자동으로 찾아 순서대로 보여줍니다.

## 단축키

| 키 | 동작 |
|---|---|
| **D** 또는 **→** | 다음 프레임 (+1) |
| **A** 또는 **←** | 이전 프레임 (-1) |
| **E** | +10 프레임 건너뛰기 |
| **Q** | -10 프레임 건너뛰기 |
| **Space** | 현재 프레임 라벨 토글 (0 ↔ Target) |
| **S** | 구간 라벨링 시작점 마킹 |
| **F** | 구간 라벨링 끝점 마킹 |
| **Enter** | 현재 영상 라벨링 저장 후 종료/다음 영상 이동 |
| **ESC** | 저장하지 않고 즉시 종료 |

## 저장 결과

- 저장 경로: `동작테스트/labeled_data/7_grab.csv`
- CSV 컬럼: `frame_idx`, `timestamp`, `gesture`

## 확인 포인트

- 실행 시 pending video 로 `7_grab.mp4` 가 보여야 합니다.
- target class 가 `7_Grab` 으로 표시되어야 합니다.
- 저장 후 `동작테스트/labeled_data/7_grab.csv` 가 생성되어야 합니다.
