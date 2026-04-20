# web-collection-agent

## 책임

- `model/동작테스트/web_collection` 파이프라인 운영
- `7_grab.mp4`, `7_grab.csv` 시드 기준으로 low-risk 후보만 수집
- Step 0 강제
- `inbox_videos -> quality_screen -> landmark_queue -> labeling_queue` 자동화
- 총 다운로드 용량 `49GB`, 총 비디오 수 `10개` 상한 유지

## 현재 단계에서 확정된 입력/출력

- 입력 영상: `raw_data/{class_id}_{label}.mp4`
- 라벨 결과: `labeled_data/{class_id}_{label}.csv`
- 랜드마크 결과: `landmark_data/{class_id}_{label}.csv`
- 수집 inbox: `web_collection/inbox_videos/**`
- 품질 결과: `web_collection/manifests/quality_manifest.jsonl`
- 랜드마크 queue: `web_collection/landmark_queue/*.csv`
- 라벨링 queue: `web_collection/manifests/labeling_queue_manifest.jsonl`

## 고정 규칙

- `profile_seed.py`가 선행되지 않으면 `discover`, `download`, `report`는 실패
- YouTube/open web search는 기본 파이프라인에서 제거
- `bulk_direct` source는 자동 다운로드하지 않고 `manual_backlog.csv`로만 보냄
- 기본 candidate는 `repack_sources`, `file_direct`만 허용
- `raw_data/` 승격은 자동으로 하지 않음

## 다음 단계 요구사항

- `discover.py`로 최대 10개 이하 candidate 생성
- `download.py`로 candidate만 다운로드
- `ingest_inbox.py`로 intake manifest 생성
- `quality_screen.py`로 accepted/review/rejected 분류
- `extract_landmarks_for_queue.py`로 accepted 항목만 landmark queue 생성
- `build_labeling_queue.py`로 수동 라벨링 handoff manifest 생성
- `commit_labeling_queue.py`로 raw_data 승격(stage) 및 landmark_data 최종 등록(finalize)
- `report.py`로 전체 상태 요약

## 현재 단계와 경계

- 웹 UI는 만들지 않음
- 자동 라벨링/자동 편집은 하지 않음
- split 메타는 아직 만들지 않음
- 이번 단계 책임은 `영상 확보 + 품질 스크리닝 + queue handoff`까지
