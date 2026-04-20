# dataset-pipeline-agent

## 책임

- `raw_data / labeled_data / landmark_data` 3단 구조를 유지
- 동일 stem 기반 매칭 규칙 보장
- 이후 train/test/inference 확장 기준 유지

## 현재 규칙

- 원본 영상: `{class_id}_{label}.mp4`
- 수동 라벨: `{class_id}_{label}.csv`
- 랜드마크: `{class_id}_{label}.csv`
- 현재 샘플: `7_grab`

## 다음 단계 확장 기준

- split은 아직 만들지 않음
- 이후 웹 수집 단계에서 메타 또는 상위 폴더로 `train/test/inference` 구분 추가
- 지금 단계에서는 stem 규칙을 단순하게 유지
- `web_collection/manifests/labeling_queue_manifest.jsonl` 를 raw 승격 전 최종 handoff로 사용
