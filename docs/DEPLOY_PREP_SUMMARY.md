# 배포 준비 작업 요약

## 목적

JamJamBeat를 `Firebase Hosting + Cloud Run` 구조로 배포할 수 있도록,
필수 설정/코드/문서를 준비하고 검증했다.

## 진행한 작업

### 1) Firebase Hosting 설정 추가

- 파일: `frontend/firebase.json`
- 내용:
  - 정적 파일 루트: `dist`
  - API Rewrite:
    - `/infer` -> Cloud Run `jamjam-backend` (`asia-northeast3`)
    - `/health` -> Cloud Run `jamjam-backend` (`asia-northeast3`)
  - SPA 라우팅 fallback: `**` -> `/index.html`
  - 캐시 헤더:
    - 정적 자산(js/css/image/font): 장기 캐시 + immutable
    - html: 재검증 캐시

### 2) Firebase 프로젝트 연결 보조 파일 추가

- 파일: `frontend/.firebaserc.example`
  - `your-gcp-project-id` 템플릿 제공
- 파일: `frontend/.firebaseignore`
  - deploy 불필요 파일 제외 설정

### 3) Cloud Run 호환 코드 반영

- 파일: `test_backend/infer_server.py`
- 변경:
  - 서버 포트 결정을 Cloud Run 방식으로 우선 처리
  - 적용 로직: `PORT` 환경변수 우선, 없으면 `JAMJAM_PORT`, 기본값 `8008`

### 4) 실행 가이드 문서 작성

- 파일: `docs/FIREBASE_CLOUDRUN_DEPLOY.md`
- 포함 내용:
  - 사전 준비(gcloud/firebase 로그인, API 활성화)
  - Docker 이미지 빌드/푸시
  - Cloud Run 배포 명령
  - Firebase Hosting 배포 명령
  - 검증/롤백 절차

## 검증 결과

- `test_backend/infer_server.py` LSP 에러 0건 확인
- `python3 -m py_compile test_backend/infer_server.py` 통과
- `frontend/firebase.json` JSON 파싱 검증 통과
- `frontend` 빌드(`npm run build`) 통과

## 현재 상태

- 배포 준비는 완료됨(설정/코드/문서 반영 및 검증 완료)
- 실제 클라우드 반영은 미실행 상태
  - 남은 최종 단계: `gcloud run deploy`, `firebase deploy --only hosting`
