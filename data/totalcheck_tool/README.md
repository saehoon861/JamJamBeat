# 🔍 통합 검수 도구 (Total Check Tool)

동영상 + MediaPipe 랜드마크 + 수동 라벨링 결과를 한 화면에서 확인하며 라벨을 수정할 수 있는 도구입니다.

## 실행 방법

### 1. 사전 준비 (최초 1회)
```bash
# 시스템 라이브러리 (Ubuntu/WSL)
sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0 libsm6 libice6

# 파이썬 환경
cd data/totalcheck_tool
uv sync
```

### 2. 실행
```bash
cd data/totalcheck_tool
uv run python main.py
```

검수 가능한 영상 목록이 자동으로 스캔됩니다.
**조건**: `raw_data/`, `labeled_data/`, `landmark_data/` 3개 폴더 모두에 동일한 파일명이 존재해야 합니다.

---

## 조작법

| 키 | 동작 |
|---|---|
| **D** 또는 **→** | 다음 프레임 (+1) |
| **A** 또는 **←** | 이전 프레임 (-1) |
| **E** | +10 프레임 건너뛰기 |
| **Q** | -10 프레임 건너뛰기 |
| **Space** | 현재 프레임 라벨 토글 (0 ↔ Target) |
| **0~6** (숫자키/텐키패드) | 현재 프레임에 해당 클래스 라벨 직접 지정 |
| **S** | 구간 라벨링 시작점 지정 |
| **F** | 구간 라벨링 끝점 지정 (S~F 구간 전체 Target으로 덮어쓰기) |
| **Enter** | 수정 내용 저장 후 다음 영상으로 이동 |
| **n** | 저장하지 않고 다음 영상으로 스킵 |
| **Esc** | 저장하지 않고 즉시 종료 |

### 주의사항
- **S 마커가 활성인 상태에서는 Enter(저장)가 차단됩니다.** F 키로 구간을 닫거나, Esc로 종료하세요.
- **n 키(스킵)는 변경 내용을 저장하지 않습니다.** 원본 CSV가 그대로 보존됩니다.
- **Esc는 프로그램 전체를 즉시 종료합니다.**

---

## 파일 구조
```
totalcheck_tool/
├── main.py                # [Controller] 이벤트 루프 진입점
├── pyproject.toml         # uv 의존성
├── core/
│   ├── annotation_state.py  # [Model] 라벨 상태 (기존 CSV로 초기화)
│   ├── class_pipeline.py    # [Singleton] 라벨 클래스 설정
│   └── video_handler.py     # OpenCV VideoCapture 래퍼
├── ui/
│   ├── osd_renderer.py      # [View] OSD 텍스트 + 타임라인 바
│   └── skeleton_renderer.py # [View] 손 랜드마크 스켈레톤 드로잉
└── file_io/
    ├── data_loader.py       # 3종 파일 로드 및 유효성 검증
    └── data_exporter.py     # labeled_data CSV 덮어쓰기 저장
```

---

## 계획 대비 변경된 사항

### IO 패키지명: `io/` → `file_io/`
| | 내용 |
|---|---|
| **기존 계획** | `io/` 폴더 사용 |
| **문제 원인** | 파이썬 내장 모듈 `io`와 이름 충돌 → `ImportError` 발생 |
| **해결 방안** | `labeling_tool`과 동일하게 `file_io/`로 변경 |

### `config/` 폴더 미생성
| | 내용 |
|---|---|
| **기존 계획** | `totalcheck_tool/config/` 폴더에 설정 파일 추가 |
| **문제 원인** | `style_config.yaml`은 불필요(색상 하드코딩)하고, `label_config.yaml`은 `labeling_tool`에 이미 존재 |
| **해결 방안** | `class_pipeline.py`의 기본 yaml 경로를 `labeling_tool/config/label_config.yaml`로 설정하여 파일 중복 제거 |

### `AnnotationState` 초기화 방식 변경
| | 내용 |
|---|---|
| **기존 계획** | `total_frames`와 `target_class`를 인자로 받아 전부 0으로 초기화 |
| **변경 사항** | `existing_labels` 리스트를 인자로 받아 기존 labeled_data의 gesture 값으로 초기화 |
| **이유** | 검수 목적이므로, 기존 라벨을 복원한 후 수정하는 방식이 올바름 |
