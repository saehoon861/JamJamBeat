# web_collection - Step 0 강제형 grab 관련 비디오 수집/queue 파이프라인

`동작테스트` 안에서만 동작하는 self-contained 후속 수집 파이프라인입니다.

## 목표

- 시드 샘플: `../raw_data/7_grab.mp4`, `../labeled_data/7_grab.csv`
- Step 0: `profile_seed.py`가 필수 선행 단계
- 기본 수집 대상: `file_direct` 기반의 실제 비디오 파일
- 기본 제외 대상: `YouTube/open_web_search`, `bulk_direct`, `이미지 zip/repack source`
- 자동 다운로드 후보 조건: `known size`, `개별 파일 15GB 미만`, `no login`
- 총 다운로드 용량: **15GB 이하**
- 총 수집 비디오 수: **20개 이하**
- 자동화 범위: `inbox_videos -> 품질 체크 -> landmark queue -> labeling queue`
- `raw_data/` 승격과 실제 라벨링은 수동

## 구조

```text
web_collection/
├── manifests/
│   ├── seed_profile.json
│   ├── candidate_manifest.jsonl
│   ├── downloaded_manifest.jsonl
│   ├── manual_backlog.csv
│   ├── intake_manifest.jsonl
│   ├── quality_manifest.jsonl
│   └── labeling_queue_manifest.jsonl
├── inbox_videos/
│   ├── official/
│   ├── open_web/
│   └── repacked/
├── landmark_queue/
├── logs/
├── profile_seed.py
├── discover.py
├── download.py
├── ingest_inbox.py
├── quality_screen.py
├── extract_landmarks_for_queue.py
├── build_labeling_queue.py
├── commit_labeling_queue.py
├── report.py
├── source_catalog.yaml
└── query_templates.yaml
```

## 빠른 시작

```bash
cd /home/user/projects/JamJamBeat-model3/동작테스트/web_collection
uv sync
uv run python profile_seed.py
uv run python discover.py
uv run python download.py --max-gb 49
uv run python ingest_inbox.py
uv run python quality_screen.py
uv run python extract_landmarks_for_queue.py
uv run python build_labeling_queue.py
uv run python report.py
```

드라이런 확인:

```bash
uv run python download.py --max-gb 1 --dry-run
```

IPN에서 `grab`에 비교적 가까운 후보 구간만 별도 클립으로 자르기:

```bash
uv run python extract_ipn_grab_candidates.py
```

출력:
- `grab_candidate_clips/ipn_hand/*.avi`
- `manifests/ipn_grab_candidate_manifest.jsonl`

기존에 받은 IPN 샘플과 다른 번호 영상을 archive마다 랜덤으로 더 받기:

```bash
uv run python sample_more_ipn_random.py --per-archive 1
```

## 실행 순서

1. `profile_seed.py`
   - `7_grab.csv`를 읽어 시드 프로파일 생성
   - `seed_profile.json`에 시드 파일 경로, `mtime_ns`, `sha256`, `fps_estimate`, positive span 정보를 기록

2. `discover.py`
   - Step 0이 현재 파일과 일치하는지 먼저 검증
  - `source_catalog.yaml`에서 `file_direct` 기반 실제 비디오 파일만 자동 후보로 취급
  - Google Drive 기반 video archive(`*.tgz`)는 다운로드 후 실제 video 파일만 추출해 inbox에 넣음
  - 이미지 zip이나 slideshow repack source는 현재 범위 밖으로 보고 자동 수집에서 제외
  - `bulk_direct`, `manual_pages`는 `manual_backlog.csv`로만 기록
  - `15GB 이상`, `크기 미확인`, `이미지 기반 source`, `semantic mismatch`는 `rejected_auto` + `manual_backlog`로 내림

3. `download.py`
   - Step 0 검증 후 candidate만 다운로드
  - 기본 tier는 `file_direct`
  - `hard_cap_videos=20`, `hard_cap_gb=15`를 동시에 강제
  - 개별 파일 `15GB 미만` 규칙을 다시 한 번 검사
   - 실행할 때마다 `logs/runs/<timestamp>.jsonl`에 새 run 로그를 만들고
     `logs/download_log.jsonl`은 최신 run 포인터만 유지

4. `ingest_inbox.py`
   - `inbox_videos/**`를 스캔해 `intake_manifest.jsonl` 생성
   - duration, fps, width/height, readable 여부를 기록

5. `quality_screen.py`
   - intake manifest 기준으로 품질 상태 생성
   - 기준:
     - duration `1~120초`
     - fps `15 이상`
     - resolution `320x240 이상`
     - sample frame hand detect ratio `0.30 이상`
   - 상태:
     - `accepted`
     - `needs_manual_review`
     - `rejected`

6. `extract_landmarks_for_queue.py`
   - `accepted` 비디오만 `landmark_queue/{stem}.csv`로 추출
   - 기존 `../landmark_extractor/main.py`의 `process_video()`를 재사용

7. `build_labeling_queue.py`
   - `accepted + landmark_queue 존재` 조건을 만족한 항목만 queue manifest 생성
   - `suggested_final_raw_filename`은 `7_grab__{source_name}__{stem}.mp4` 형식

8. `commit_labeling_queue.py`
   - 상태 조회:
     - `uv run python commit_labeling_queue.py`
   - raw_data 승격(stage):
     - `uv run python commit_labeling_queue.py --stage-all`
   - labeling_tool 라벨링 완료 뒤 landmark_data 최종 등록(finalize):
     - `uv run python commit_labeling_queue.py --finalize-all-ready`

9. `report.py`
   - candidate / downloaded / intake / quality / queue 상태 요약

## 고정 규칙

- `seed_profile.json`이 없거나 stale이면 `discover.py`, `download.py`, `report.py`는 실패
- 새 candidate manifest에는 `youtube_video`가 없어야 함
- `bulk_direct`는 자동 다운로드하지 않음
- 이미지 기반 `repack_sources`는 현재 파이프라인 범위 밖으로 보고 자동 다운로드하지 않음
- `candidate_manifest.jsonl`의 자동 candidate는 모두 `filesize_bytes < 15GB`여야 함
- `filesize_bytes`를 알 수 없는 소스는 자동 후보에 들어가지 않음
- `raw_data/`, `labeled_data/`, `landmark_data/`는 web_collection 자동 단계에서 수정하지 않음
- 최종 handoff는 `manifests/labeling_queue_manifest.jsonl`
- `commit_labeling_queue.py --stage-all` 은 inbox 비디오를 `raw_data/`로 이동
- `commit_labeling_queue.py --finalize-all-ready` 는 라벨 CSV가 존재하는 항목만 `landmark_queue/`에서 `landmark_data/`로 이동
- 현재 실제 다운로드 파일이 없다면 삭제 단계는 `없음 확인`으로 끝내고, stale log/manifests만 새 run 기준으로 정리

## 결과 확인

- `manifests/seed_profile.json`
- `manifests/candidate_manifest.jsonl`
- `manifests/manual_backlog.csv`
- `manifests/intake_manifest.jsonl`
- `manifests/quality_manifest.jsonl`
- `manifests/labeling_queue_manifest.jsonl`
- `commit_labeling_queue.py` 상태 출력
- `landmark_queue/*.csv`
- `summary.md`

## 현재 15GB 영상 수집 해석

- `source_catalog.yaml` 안에 있는 source라도 `15GB 이상`이면 자동 수집 대상이 아닙니다.
- 현재 자동 후보는 direct mp4/webm과, 공식 video archive를 실제 video 파일로 풀어넣는 항목만 포함합니다.
- HaGRID처럼 이미지 zip을 받아 일부를 repack하는 방식은 지금 범위에서 제외했습니다.
- `download_log.jsonl`에 예전 이벤트가 섞여 있었다면, 새 `download.py` 실행 이후에는 최신 run 포인터로 덮어써집니다.
- 실제 다운로드 파일 유무는 `inbox_videos/**`와 `manifests/downloaded_manifest.jsonl`를 함께 보면 됩니다.
