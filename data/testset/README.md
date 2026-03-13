

## 실행방법 

### 윈도우 하위 리눅스 가상환경(WSL) 세팅일 시 
```powershell
# 1. usbipd 설치 (winget으로)
winget install usbipd

# 2. powershell을 관리자 권한으로 새로 열어서 
# 2-1. 웹캠 장치 확인
usbipd list

# 2-2. 웹캠 BUSID 확인 후 연결 
# (예: 2-1에서 확인한 웹캠 장치의 BUSID가 1-1인 경우)
usbipd bind --busid 1-1
usbipd attach --wsl --busid 1-1
```



```bash
# 가상환경 활성화
uv sync

# 데이터 수집
uv run python data/testset/capture.py

# 데이터 검수 (gesture 번호 필수)
uv run python data/testset/review.py --gesture 1

```

