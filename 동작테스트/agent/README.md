# 동작테스트 agent 운영 문서

## 목적

`동작테스트` 워크스페이스 안에서만 동작하는 데이터 파이프라인 규칙을 정의합니다.

## 파이프라인 맵

```text
raw_data -> labeled_data -> landmark_data -> web_collection/inbox_videos -> quality_screen -> landmark_queue -> labeling_queue -> manual curation
```

## 고정 규칙

- 외부 경로 참조 금지
- 입력 영상: `raw_data/{class_id}_{label}.mp4`
- 라벨 CSV: `labeled_data/{class_id}_{label}.csv`
- 랜드마크 CSV: `landmark_data/{class_id}_{label}.csv`
- 동일 stem 1:1 매칭
- 현재 대상 샘플: `7_grab`

## 문서 구성

- `labeling-agent.md`
- `landmark-extractor-agent.md`
- `dataset-pipeline-agent.md`
- `web-collection-agent.md`

## 현재 구현 상태

- `raw_data/7_grab.mp4`
- `labeled_data/7_grab.csv`
- `landmark_extractor/` 로컬 MediaPipe 추출기
- `web_collection/` Step 0 강제형 웹 비디오 수집/queue 파이프라인
