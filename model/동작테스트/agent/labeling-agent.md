# labeling-agent

## 책임

- `raw_data/*.mp4` 를 수동 라벨링 대상으로 관리
- 파일명에서 target class를 해석
- `labeled_data/*.csv` 저장 규칙 유지

## 현재 고정 대상

- 입력: `raw_data/7_grab.mp4`
- target class: `7_Grab`
- 출력: `labeled_data/7_grab.csv`

## 완료 기준

- pending 목록에 `7_grab.mp4` 가 잡힘
- target class 표시가 `7_Grab`
- 저장 시 `7_grab.csv` 생성

## 실패 조건

- 파일명이 `{class_id}_{label}.mp4` 규칙을 따르지 않음
- `labeled_data` 와 stem이 충돌해 이미 처리된 것으로 오인됨
