# eval_dashboard - JamJamBeat 모델 평가 브라우저 대시보드

`model/frontend/eval_dashboard` 는 현재 프로젝트의 로컬 모델 평가 산출물을 브라우저에서 확인하기 위한 정적 대시보드입니다.

이 앱은 `로컬 evaluation dataset(suite) 전환 + 모델별 비디오 체크 + 사용자별 로컬 리뷰 저장` 에 초점을 맞춘 MVP입니다.

## 현재 가능한 것

- 로컬 evaluation dataset 드롭다운 전환
- 선택된 suite 기준 모델 비교 리더보드
- 모델 설명 패널
- run summary / hyperparameter / dataset split snapshot
- `user / motion / side / source(video)` 필터
- 모델별 비디오 체커
- 비디오 위 21점 손 랜드마크 오버레이 + 구조 미리보기 박스
- 필터와 무관하게 suite 전체 raw source를 직접 여는 `Workspace Video / Source` 드롭다운
- prediction event가 없는 source도 raw frame + 21개 손 노드/엣지 확인
- source explorer 카드
- per-class report, confusion matrix, training history
- artifact 링크 열기
- 브라우저 localStorage 기반 개인 리뷰 메모/체크리스트 저장
- 리뷰 JSON 내보내기 / 가져오기

## 아직 없는 것

- 브라우저 안에서 `model.pt` checkpoint를 직접 로드해서 새 추론을 돌리는 기능
- 실시간 webcam inference
- backend API 서버
- multi-user 서버 저장소

즉, 현재 버전은 `기존 평가 결과를 브라우저에서 검토하는 플랫폼` 입니다.  
`새 raw input에 대해 모델을 직접 추론하는 실행기` 는 아직 붙지 않았습니다.

## 현재 완성도

대략 `MVP 기준 75~80%` 정도로 보면 됩니다.

- 완료:
  - 로컬 suite catalog 자동 로딩
  - 브라우저 상단 evaluation dataset 전환
  - 모델 비교/상세/영상체크/리뷰 저장 흐름
  - 데스크톱 / 모바일 레이아웃 확인
- 미완료:
  - checkpoint 직접 추론
  - webcam / live inference
  - 서버 저장 / 계정 시스템
  - 추론 파이프라인과 완전한 통합

## 구조

```text
model/frontend/eval_dashboard/
├── README.md
├── .gitignore
├── generate_dashboard_data.py   # 로컬 suite들 -> 브라우저용 JSON 생성
├── index.html                   # 메인 화면 마크업
├── styles.css                   # UI 스타일
├── app.js                       # 대시보드 동작 로직
└── data/
    ├── suite-catalog.json       # 로컬 evaluation dataset 목록
    ├── latest-suite-index.json  # 최신 suite 요약 인덱스(호환용)
    ├── models/
    │   └── *.json               # 최신 suite 모델 상세 데이터(호환용)
    └── suites/
        └── <suite_name>/
            ├── index.json
            └── models/
                └── *.json
```

## 데이터 흐름

1. `model/model_evaluation/pipelines/` 아래의 로컬 evaluation suite를 스캔
2. 각 suite의 `comparison_results.csv`, `run_summary.json`, `metrics_summary.json`, `preds_test.csv`, `train_history.csv`, `per_class_report.csv`, `confusion_matrix.csv` 를 모음
3. `generate_dashboard_data.py` 가 이를 `model/frontend/eval_dashboard/data/suites/<suite_name>/` 아래 JSON으로 변환하고 `suite-catalog.json` 을 생성
4. 추가로 suite의 `input_csv_paths` 를 읽어, `preds_test.csv` 에 없더라도 suite 전체 raw source와 landmark source를 브라우저에서 직접 열 수 있는 source catalog를 생성
5. 브라우저는 상단 드롭다운에서 선택된 suite와 로컬 비디오 경로를 읽어 화면에 렌더링

## 파일별 역할

- `generate_dashboard_data.py`
  - 로컬 evaluation suite를 읽고 브라우저용 JSON 생성
  - suite 드롭다운용 catalog 생성
  - source file 이름을 `user / motion / side / gesture` 메타데이터로 분해
  - 모델 설명 패널용 기본 설명도 함께 생성
- `index.html`
  - 리더보드, 모델 상세, 필터, 비디오 체커, 리뷰 영역 구성
- `styles.css`
  - 대시보드 레이아웃과 반응형 스타일 정의
- `app.js`
  - 모델 선택
  - source/video 필터와 독립적인 frame workspace source 선택
  - prediction timeline
  - 재생 중 `requestVideoFrameCallback` 기반 프레임 동기화
  - localStorage 리뷰 저장/복원
  - 리뷰 JSON import/export

## 브라우저 Video Checker 동작 방식

- `Dataset User / Motion / Side` 는 source 탐색용 필터입니다.
- `Workspace Video / Source` 드롭다운은 현재 필터와 관계없이 suite 전체 raw source를 직접 엽니다.
- 즉 person-wise split suite여도 `preds_test.csv` 에 없는 `man1`, `man2`, `man3` source를 브라우저에서 직접 열어 프레임 단위 확인이 가능합니다.
- `serve_dashboard.py` 로 열면 브라우저가 로컬 checkpoint 추론을 다시 돌려, `landmark가 있는 모든 프레임`에 대해 GT vs prediction 비교를 보여줍니다.
- 여기서 `test prediction frames` 는 기존 `preds_test.csv` 에 저장된 test split 프레임 수이고, `full comparison` 은 landmark가 존재하는 전체 프레임 수입니다.
- 단순 `http.server` 로만 열면 API가 없어서 기존 `preds_test.csv` 비교만 보이고, `serve_dashboard.py` 또는 Cloudflare 공유 서버를 쓰면 full landmark inference 비교가 활성화됩니다.
- landmark는 우선 `data/landmark_data/<source>.csv` 를 사용하고, 없으면 suite의 `input_csv_paths` (예: `model/data_fusion/*.csv`) 를 fallback landmark source로 사용합니다.

## 실행 방법

아래 명령은 모두 `repo 루트(/home/user/projects/JamJamBeat)` 에서 실행하는 기준이며, `Python 실행은 model/.venv` 를 기준으로 통일합니다.

### 1. 최신 평가 데이터를 JSON으로 생성

```bash
model/.venv/bin/python model/frontend/eval_dashboard/generate_dashboard_data.py
```

### 2-A. 가장 단순한 로컬 서버 실행

대시보드를 내 PC 브라우저에서만 확인할 때 가장 간단한 방법입니다.

```bash
model/.venv/bin/python -m http.server 8000
```

브라우저 주소:

```text
http://localhost:8000/model/frontend/eval_dashboard/
```

종료:

- 터미널에서 `Ctrl + C`

### 2-B. 대시보드 전용 서버 실행

repo 전체가 아니라 대시보드와 필요한 파일만 노출하고 싶다면 이 방법을 권장합니다.

```bash
model/.venv/bin/python model/frontend/eval_dashboard/serve_dashboard.py --port 8123
```

브라우저 주소:

```text
http://127.0.0.1:8123/model/frontend/eval_dashboard/
```

이 서버는 아래 경로만 허용합니다.

- `/model/frontend/eval_dashboard/`
- `/data/raw_data/`
- `/data/landmark_data/`
- `/model/data_fusion/`
- `/model/model_evaluation/pipelines/`

종료:

- 터미널에서 `Ctrl + C`

### 3. 실행 추천 순서

로컬에서 확인만 할 때:

```bash
model/.venv/bin/python model/frontend/eval_dashboard/generate_dashboard_data.py
model/.venv/bin/python model/frontend/eval_dashboard/serve_dashboard.py --port 8123
```

그 다음 브라우저에서:

```text
http://127.0.0.1:8123/model/frontend/eval_dashboard/
```

## Cloudflare Tunnel로 공유

`cloudflared` 가 이미 설치되어 있다면, 이 프로젝트에서는 아래 스크립트로 `eval_dashboard` 전용 서버와 Quick Tunnel을 한 번에 띄울 수 있습니다.

### Quick Tunnel 실행

```bash
bash model/frontend/eval_dashboard/share_via_cloudflared.sh
```

기본 포트는 `8123` 이고, 다른 포트를 쓰고 싶으면 뒤에 붙이면 됩니다.

```bash
bash model/frontend/eval_dashboard/share_via_cloudflared.sh 9000
```

이 스크립트는 아래 순서로 동작합니다.

1. `model/frontend/eval_dashboard/serve_dashboard.py` 로 전용 정적 서버 실행
2. 로컬 서버가 뜨는지 확인
3. `cloudflared tunnel --url http://127.0.0.1:<port>` 로 Quick Tunnel 오픈

즉, 공유까지 한 번에 하려면 아래 한 줄이면 됩니다.

```bash
model/.venv/bin/python model/frontend/eval_dashboard/generate_dashboard_data.py
bash model/frontend/eval_dashboard/share_via_cloudflared.sh
```

종료:

- Quick Tunnel을 띄운 터미널에서 `Ctrl + C`
- 이 스크립트는 종료 시 내부적으로 실행한 전용 서버도 함께 정리합니다.

중요:

- 공개 주소는 `cloudflared` 출력에 표시됩니다.
- Cloudflare Quick Tunnel 은 가끔 `api.trycloudflare.com` 쪽 일시 오류(`1101`)로 첫 시도가 실패할 수 있습니다. 이 스크립트는 최대 3번 자동 재시도합니다.
- `failed to unmarshal quick Tunnel`, `Worker threw exception`, `1101` 이 보이면 대개 로컬 서버 문제가 아니라 `TryCloudflare` 쪽 장애입니다.
- 이 서버는 repo 전체가 아니라 아래 경로만 노출합니다.
  - `/model/frontend/eval_dashboard/`
  - `/data/raw_data/`
  - `/data/landmark_data/`
  - `/model/data_fusion/`
  - `/model/model_evaluation/pipelines/`
- 브라우저 localStorage 리뷰 기록은 `도메인(origin)` 기준이라, Quick Tunnel 주소가 바뀌면 기존 공개 URL에서 저장한 기록은 새 URL과 분리되어 보일 수 있습니다.
- 안정적인 같은 주소가 필요하면 Quick Tunnel 대신 `cloudflared tunnel login` 후 named tunnel + 고정 hostname 구성을 추천합니다.

### 고정 주소용 named tunnel

Quick Tunnel 대신 고정 주소를 쓰고 싶다면 Cloudflare 공식 로컬 관리 방식은 아래 순서입니다.

```bash
cloudflared tunnel login
cloudflared tunnel create jamjambeat-eval-dashboard
cloudflared tunnel route dns jamjambeat-eval-dashboard eval-dashboard.<your-domain>
```

그 다음 `~/.cloudflared/config.yml` 또는 별도 설정 파일에 아래처럼 origin 을 연결합니다.

```yaml
tunnel: <Tunnel-UUID>
credentials-file: /home/<user>/.cloudflared/<Tunnel-UUID>.json

ingress:
  - hostname: eval-dashboard.<your-domain>
    service: http://127.0.0.1:8123
  - service: http_status:404
```

그리고 아래 두 프로세스를 함께 실행하면 됩니다.

```bash
model/.venv/bin/python model/frontend/eval_dashboard/serve_dashboard.py --port 8123
cloudflared tunnel run jamjambeat-eval-dashboard
```

## ngrok으로 공유

`ngrok` 이 이미 설치되어 있고 `ngrok config add-authtoken <YOUR_TOKEN>` 까지 끝난 상태라면, 아래 스크립트로 대시보드 전용 서버와 ngrok 터널을 한 번에 띄울 수 있습니다.

### 기본 공유

```bash
bash model/frontend/eval_dashboard/share_via_ngrok.sh
```

다른 포트를 쓰고 싶으면:

```bash
bash model/frontend/eval_dashboard/share_via_ngrok.sh 9000
```

이 스크립트는 아래 순서로 동작합니다.

1. `model/frontend/eval_dashboard/serve_dashboard.py` 로 전용 서버 실행
2. 로컬 서버가 뜨는지 확인
3. `ngrok http 127.0.0.1:<port>` 로 공개 주소 생성

즉, 공유까지 한 번에 하려면 아래 한 줄이면 됩니다.

```bash
model/.venv/bin/python model/frontend/eval_dashboard/generate_dashboard_data.py
bash model/frontend/eval_dashboard/share_via_ngrok.sh
```

종료:

- ngrok 을 띄운 터미널에서 `Ctrl + C`
- 이 스크립트는 종료 시 내부적으로 실행한 전용 서버도 함께 정리합니다.

중요:

- 공개 주소는 `ngrok` 출력에 표시됩니다.
- free/dev 사용에서는 실행할 때마다 주소가 바뀔 수 있습니다.
- 브라우저 localStorage 리뷰 기록은 `도메인(origin)` 기준이라, 공개 URL이 바뀌면 저장된 리뷰가 분리되어 보일 수 있습니다.

### 예약 도메인 사용

이미 예약한 `ngrok` 도메인이 있으면 `NGROK_DOMAIN` 환경변수로 고정 주소를 넣을 수 있습니다.

```bash
NGROK_DOMAIN="eval-dashboard.ngrok.app" bash model/frontend/eval_dashboard/share_via_ngrok.sh
```

## 로컬 PC에서 실제로 도는 범위

### raw video

네. 현재 비디오 체크에 쓰는 raw video는 `내 로컬 PC` 에 있는 파일을 그대로 씁니다.

- 원본 위치: `data/raw_data/*.mp4`
- 브라우저에서는 로컬 정적 서버를 통해 이 파일을 읽습니다.
- 즉 `http.server` 를 네 PC에서 띄우면, 그 PC의 raw video가 그대로 재생됩니다.

### evaluation 결과

네. 현재 대시보드가 읽는 평가 결과도 `내 로컬 PC` 에 있는 파일입니다.

- 원본 위치:
  - `model/model_evaluation/pipelines/.../comparison_results.csv`
  - `run_summary.json`
  - `metrics_summary.json`
  - `preds_test.csv`
  - `train_history.csv`
  - `per_class_report.csv`
  - `confusion_matrix.csv/png`

### checkpoint (`model.pt`)

체크포인트 파일도 네 로컬 PC에 있습니다.  
다만 `현재 대시보드는 이 checkpoint를 브라우저에서 직접 실행하지는 않습니다`.

- 체크포인트 위치 예:
  - `model/model_evaluation/pipelines/<suite>/<model>/<timestamp>/model.pt`
- 현재 버전에서 checkpoint는:
  - 평가 결과를 만든 원본 산출물의 일부로 로컬에 존재
  - 브라우저 UI가 직접 로드/추론하지는 않음

정리하면:

- `raw video` 는 로컬에서 실제 재생됨
- `평가 결과 JSON/CSV` 도 로컬에서 읽음
- `Workspace Video / Source` 로 여는 suite source와 fallback landmark CSV도 로컬에서 읽음
- `checkpoint` 도 로컬에 존재함
- 하지만 `checkpoint 직접 추론 실행` 은 아직 미구현

## 로컬 저장 방식

- 리뷰 메모와 체크리스트는 브라우저 `localStorage` 에 저장됩니다.
- 키는 `suite + reviewer + model_id + source_file` 기준으로 분리됩니다.
- 같은 머신에서 다른 사용자는 Reviewer 이름을 다르게 입력하면 분리 저장됩니다.
- `리뷰 JSON 내보내기` 버튼으로 개인 기록을 별도 파일로 백업할 수 있습니다.
- `리뷰 JSON 가져오기` 버튼으로 기존 백업을 다시 로컬 브라우저에 복원할 수 있습니다.

## 검증 상태

확인 완료:

- `model/.venv/bin/python model/frontend/eval_dashboard/generate_dashboard_data.py`
- `node --check model/frontend/eval_dashboard/app.js`
- `model/.venv/bin/python -m py_compile model/frontend/eval_dashboard/generate_dashboard_data.py`
- Playwright CLI로 데스크톱/모바일 스크린샷 캡처

참고:

- Playwright MCP 자체는 timeout이 있었지만, Playwright CLI 검증은 통과했습니다.

## 다음 단계 추천

1. `model.pt` 를 실제로 추론에 쓰는 로컬 inference backend 추가
2. webcam 또는 업로드 비디오 추론 추가
3. 리뷰 기록을 파일이 아니라 서버/DB로 저장하는 옵션 추가
4. 모델 간 동일 source 동시 비교 뷰 추가

## Google CLI 관련 메모

- 현재 요구인 `사용자가 자기 로컬에 체크 기록을 저장하고 다시 확인` 하는 목적에는 Google CLI보다 브라우저 localStorage/JSON export 방식이 더 직접적입니다.
- 이 대시보드는 CLI 없이도 사용자 로컬에 기록을 남길 수 있게 설계했습니다.
