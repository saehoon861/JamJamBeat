# 📋 수동 데이터 라벨링 도구

## 실행 방법

### 0. 영상 준비
data/raw_data 폴더에 .mp4 파일들을 드래그 앤 드롭 또는,

```bash
cd data/raw_data
# 여기에 .mp4 파일들을 넣습니다. 파일명은 1_slow_right_woman1.mp4 와 같은 형식으로 합니다.
```

### 1. 시스템 의존성 설치 (최초 1회, Ubuntu/WSL)
OpenCV GUI 표시에 필요한 시스템 라이브러리를 먼저 설치합니다.
```bash
sudo apt-get update && sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0 libsm6 libice6
```

### 2. 파이썬 환경 설치 (최초 1회)
```bash
cd data/labeling_tool
uv sync
```

### 3. 라벨링 툴 실행
```bash
cd data/labeling_tool
uv run python main.py
```

실행하면 `data/raw_data/` 안의 아직 라벨링하지 않은 `.mp4` 파일을 자동으로 찾아서 순서대로 보여줍니다.

---

## 조작법 (단축키)

| 키 | 동작 |
|---|---|
| **D** 또는 **→** | 다음 프레임 (+1) |
| **A** 또는 **←** | 이전 프레임 (-1) |
| **E** | +10 프레임 건너뛰기 |
| **Q** | -10 프레임 건너뛰기 |
| **Space** | 현재 프레임 라벨 토글 (0 ↔ Target) |
| **S** | 구간 라벨링 시작점 마킹 |
| **F** | 구간 라벨링 끝점 마킹 (S~F 사이 전부 Target으로 덮어쓰기) |
| **Enter** | 현재 영상 라벨링 저장 후 다음 영상으로 이동 |
| **ESC** | 저장하지 않고 즉시 종료 |

### 주의사항
- **S를 누른 상태에서는 Enter(저장)가 차단됩니다.** 반드시 F를 눌러 구간을 닫거나, ESC로 나가야 합니다.
- **ESC를 누르면 현재 작업 내용은 절대 저장되지 않습니다.**
- S를 누르지 않고 F만 누르면 아무 일도 일어나지 않습니다.

---

## 결과물
- 저장 경로: `data/labeled_data/{영상파일명}.csv`
- CSV 컬럼: `frame_idx`, `timestamp`, `gesture`

---

## 파일 구조
```
data/
├── README.md               # 이 문서
├── plan.md                  # 개발 기획서
├── raw_data/                # 원본 .mp4 동영상 (라벨링 대상)
├── labeled_data/            # 라벨링 완료된 .csv 결과물
└── labeling_tool/           # 라벨링 도구 소스코드
    ├── main.py              # [진입점] 이벤트 루프 컨트롤러
    ├── pyproject.toml       # uv 프로젝트 의존성
    ├── config/
    │   └── label_config.yaml  # 클래스 ID↔이름, 색상 설정
    ├── core/
    │   ├── class_pipeline.py  # YAML 설정 로더 (Singleton)
    │   ├── annotation_state.py # 프레임별 라벨 상태 관리 (Model)
    │   └── video_handler.py   # OpenCV VideoCapture 래퍼
    ├── ui/
    │   └── osd_renderer.py    # OSD 텍스트 + 타임라인 바 렌더링 (View)
    └── file_io/
        ├── file_scanner.py    # 미작업 영상 스캔 및 필터링
        └── data_exporter.py   # CSV 내보내기
```

---

## plan.md 대비 구현 변경/특이사항

### 계획을 준수한 부분
- **MVC 패턴**: Model(`annotation_state.py`), View(`osd_renderer.py`), Controller(`main.py`) 분리 구현
- **Singleton 패턴**: `LabelConfigManager`를 싱글톤으로 구현하여 YAML을 한 번만 로드
- **하드코딩 배제**: 클래스 ID↔이름, 색상을 전부 `label_config.yaml`에 분리
- **타임라인 바**: `np.vstack`으로 원본 프레임과 40px 타임라인을 합성하여 `imshow`
- **경계 프레임 처리**: 타임라인에서 범위 밖 프레임은 검은색(Blank)으로 표시
- **경고 메세지 라이프사이클**: `warning_msg` 변수를 매 루프 초기화하여 1회만 표시
- **최초 렌더링**: `while` 루프 진입 전 `render_frame()` 선제 호출
- **S 마커 잠금**: `is_range_active()` 체크로 Enter 저장 차단 + 화면 경고 출력
- **F 단독 입력 무시**: `start_marker`가 `None`이면 조용히 Pass
- **슬라이싱 +1 포함**: `set_range_end`에서 `range(start, end + 1)`으로 F 프레임도 포함
- **stem 매칭**: `.mp4` ↔ `.csv` 파일명 stem 기준 1:1 대조
- **uv 가상환경 격리**: `data/labeling_tool/` 내부에 독립 uv 프로젝트 구성
- **data/ 폴더 격리**: 기존 `src/` 폴더는 미사용, 전부 `data/` 내부에서 작업

### 계획 대비 변경된 부분
| 항목 | 계획 (plan.md) | 구현 |
|---|---|---|
| IO 패키지명 | `io/` | `file_io/` (파이썬 내장 `io` 모듈 이름 충돌 방지) |
| `requirements.txt` | 별도 파일 | `pyproject.toml`에 `uv add`로 의존성 통합 관리 (uv 표준 방식) |
| `.mp4.mp4` 이중 확장자 | 미언급 | `class_pipeline.py`, `file_scanner.py`에서 이중 확장자 자동 제거 로직 추가 (`raw_data/` 내 일부 파일이 해당) |
