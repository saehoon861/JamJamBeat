# 손 랜드마크 추출기

`raw/data`에 있는 동영상들에 대해서
미디어파이프 손 랜드마커로 프레임에 대한 랜드마크를 추출함.


---

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
cd data/landmark_extractor
uv sync
```

### 3. 랜드마크 추출 툴 실행
```bash
cd data/landmark_extractor
uv run python main.py
```
---

## 모드 변경 기능.
모드를 전환하고 싶을 때 **`main.py` 상단의 `RUNNING_MODE` 상수 한 줄만 수정**:

```python
# 이미지 모드 
RUNNING_MODE = VisionRunningMode.IMAGE

# 비디오 모드 
RUNNING_MODE = VisionRunningMode.VIDEO
```