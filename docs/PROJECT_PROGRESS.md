# JamJamBeat 프로젝트 진행 사항

> 최종 업데이트: 2026-03-12

## 프로젝트 개요

**JamJamBeat**는 손 제스처 인식을 통해 음악을 연주할 수 있는 인터랙티브 웹 애플리케이션입니다.

- **기술 스택**: MediaPipe Hands, PyTorch, Vanilla JavaScript
- **주요 기능**: 실시간 손 추적, 제스처 인식, 악기 음향 매핑, 시각 효과
- **아키텍처**: 프론트엔드(웹), 백엔드(추론 서버), ML 파이프라인(데이터 처리 및 학습)

---

## 1. 프로젝트 구조

### 디렉토리 구성

```
JamJamBeat/
├── frontend/               # 웹 프론트엔드 (HTML, CSS, JavaScript)
│   ├── index.html         # 메인 게임 페이지
│   ├── theme.html         # 테마/효과 테스트 페이지
│   ├── admin.html         # 악기 배치 관리자 페이지
│   ├── src/
│   │   ├── js/           # JavaScript 모듈
│   │   └── styles/       # CSS 스타일
│   └── public/assets/    # 사운드, 이미지 등 정적 자산
│
├── data/                  # 학습 데이터 준비 도구
│   ├── landmark_extractor/  # 영상에서 손 랜드마크 추출
│   ├── labeling_tool/      # 제스처 수동 라벨링 도구 (구버전)
│   ├── totalcheck_tool/    # 통합 검수 및 라벨링 도구 (신버전)
│   ├── landmark_data/      # 추출된 랜드마크 CSV
│   ├── labeled_data/       # 라벨링 완료 CSV
│   ├── total_data/         # 병합된 최종 학습 데이터
│   ├── integration.py      # CSV 병합 스크립트
│   └── check_labels.py     # 라벨 분포 확인 스크립트
│
├── src/                   # ML 파이프라인 (학습, 추론, 전처리)
│   ├── dataset/          # 데이터셋 로더
│   ├── models/           # 모델 정의 (MLP, Transformer)
│   ├── training/         # 학습 및 평가 스크립트
│   ├── preprocessing/    # 전처리 유틸
│   ├── pipelines/        # 통합 파이프라인
│   └── utils/            # 공통 유틸 (config, metrics)
│
├── test_backend/          # 추론 서버 (FastAPI)
│   ├── infer_server.py   # 추론 API 엔드포인트
│   └── mlp_classifier_copy.py
│
├── poc/                   # Proof of Concept 실험 코드
├── checkpoints/           # 학습된 모델 체크포인트
├── outputs/               # 학습 결과 로그 및 플롯
├── wandb/                 # Weights & Biases 실험 추적 데이터
└── docs/                  # 프로젝트 문서
    ├── GESTURE_CONTRACT.md    # 프론트엔드-추론 인터페이스 계약
    ├── COLLAB_WORKFLOW.md     # 협업 워크플로우 가이드
    └── PROJECT_PROGRESS.md    # 이 문서
```

---

## 2. 주요 기능별 진행 상황

### 2.1 프론트엔드 (Web UI)

#### 핵심 페이지

- **index.html**: 메인 게임 플레이 페이지
  - 실시간 손 추적 및 제스처 인식
  - 악기와의 충돌 감지 및 사운드 재생
  - 피버 모드 및 파티클 효과

- **theme.html**: 테마/효과 테스트 전용 페이지
  - 배경 비디오 루프 테스트
  - 씬 전환 효과 검증

- **admin.html**: 악기 배치 관리자 페이지
  - 드래그 앤 드롭으로 악기 위치 조정
  - localStorage 기반 배치 저장/불러오기
  - 사운드 매핑 관리

#### JavaScript 모듈 구조 (리팩토링 완료)

`frontend/src/js/` 디렉토리:

| 파일 | 책임 | 상태 |
|------|------|------|
| `main.js` | 앱 조립, 초기화, 의존성 연결 (496줄) | ✅ 리팩토링 완료 |
| `hand_tracking_runtime.js` | MediaPipe 프레임 루프, 랜드마크 캐시 | ✅ 신규 분리 |
| `interaction_runtime.js` | 악기 충돌, 제스처 반응, 손 커서 표시 | ✅ 신규 분리 |
| `instrument_layout.js` | 악기 배치 저장/로드, 관리자 드래그 편집 | ✅ 신규 분리 |
| `scene_runtime.js` | 배경 비디오, 씬 모드, 피버 상태 관리 | ✅ 신규 분리 |
| `particle_system.js` | 파티클 생성 및 업데이트 | ✅ 신규 분리 |
| `sound_mapping.js` | 사운드 매핑 로드 및 관리 | ✅ 신규 분리 |
| `gestures.js` | 제스처 라벨 정규화, 안정화 로직 | ✅ 기존 유지 |
| `audio.js` | 사운드 재생 엔진 | ✅ 기존 유지 |
| `renderer.js` | 캔버스 렌더링 | ✅ 기존 유지 |
| `model_inference.js` | 추론 서버 API 통신 | ✅ 기존 유지 |
| `admin.js` | 관리자 페이지 로직 | ✅ 기존 유지 |
| `theme.js` | 테마 페이지 로직 | ✅ 기존 유지 |

**리팩토링 성과**:
- `main.js`: 1039줄 → 496줄 (52% 감소)
- 책임 분리를 통한 유지보수성 향상
- 모듈화를 통한 재사용성 증대

자세한 리팩토링 내역은 [`frontend/REFACTORING_NOTES.md`](../frontend/REFACTORING_NOTES.md) 참고.

#### 현재 제스처-악기 매핑

| 제스처 | 악기/효과 |
|--------|-----------|
| `Fist` | 드럼 |
| `OpenPalm` | 실로폰 |
| `V` | 탬버린 |
| `Pinky` | 부엉이 |
| `Animal` | 부엉이 |
| `KHeart` | 고사리 |

---

### 2.2 데이터 파이프라인

#### 데이터 수집 및 라벨링 워크플로우

1. **영상 준비** (`data/raw_data/`)
   - 손 제스처 영상 촬영 및 저장

2. **랜드마크 추출** ([`data/landmark_extractor/`](../data/landmark_extractor/README.md))
   - MediaPipe Hands를 사용해 21개 손 랜드마크 추출
   - 프레임별 좌표를 CSV로 저장 (`data/landmark_data/`)

3. **수동 라벨링** ([`data/totalcheck_tool/`](../data/totalcheck_tool/README.md))
   - 영상과 랜드마크를 동시에 보며 제스처 라벨링
   - 실시간 재생, 구간 반복, 프레임 단위 탐색 지원
   - 라벨링 결과를 CSV로 저장 (`data/labeled_data/`)

4. **데이터 병합** (`data/integration.py`)
   - 랜드마크 CSV + 라벨 CSV 병합
   - 최종 학습 데이터 생성 (`data/total_data/`)

5. **라벨 분포 확인** (`data/check_labels.py`)
   - 전체 및 파일별 클래스 분포 확인
   - 데이터 불균형 검증

#### 제스처 클래스

현재 정의된 제스처 라벨 (Canonical Labels):

- `None` (무제스처)
- `Fist` (주먹)
- `OpenPalm` (펼친 손바닥)
- `V` (브이 사인)
- `Pinky` (새끼손가락)
- `Animal` (동물 모양)
- `KHeart` (한국 하트)

제스처 라벨 정규화 및 별칭 규칙은 [`docs/GESTURE_CONTRACT.md`](./GESTURE_CONTRACT.md) 참고.

---

### 2.3 ML 모델 학습 및 추론

#### 모델 아키텍처

`src/models/` 디렉토리:

- **Baseline**: MLP (Multi-Layer Perceptron)
  - 간단한 전결합 신경망
  - 빠른 추론 속도

- **Advanced**: Temporal Transformer
  - 시계열 제스처 패턴 학습
  - Sliding window 기반 시퀀스 처리
  - Landmark embedding 활용

#### 학습 파이프라인

`src/training/` 디렉토리:

- `train.py`: 모델 학습 메인 스크립트
- `evaluation.py`: 평가 지표 계산 (정확도, precision, recall, F1)
- `predict.py`: 단일 프레임 추론 테스트

학습 실험은 **Weights & Biases (wandb)** 로 추적 중.

#### 추론 서버

`test_backend/infer_server.py`:

- **프레임워크**: FastAPI
- **엔드포인트**: `/predict`
- **입력**: 21개 랜드마크 좌표 (`x`, `y`, `z`)
- **출력**:
  ```json
  {
    "label": "Fist",
    "confidence": 0.91,
    "probs": [0.01, 0.91, 0.03, ...],
    "model_version": "v1"
  }
  ```

프론트엔드-추론 인터페이스 계약은 [`docs/GESTURE_CONTRACT.md`](./GESTURE_CONTRACT.md) 참고.

---

### 2.4 협업 체계

#### 역할 분리

- **Design/Effects Owner**: 프론트엔드 UI, 시각/음향 효과
- **Inference Owner**: ML 모델, 추론 서버, 데이터 파이프라인

#### 공유 계약 파일

- [`docs/GESTURE_CONTRACT.md`](./GESTURE_CONTRACT.md): 프론트엔드-추론 인터페이스 명세
- [`docs/COLLAB_WORKFLOW.md`](./COLLAB_WORKFLOW.md): PR 리뷰 규칙, 변경 관리 프로세스

#### 변경 관리 원칙

- **라벨 변경**: 가능한 alias 추가로 처리, rename 최소화
- **신뢰도 의미 변경**: Contract 변경으로 간주, 명시적 문서화 필요
- **스키마 변경**: Request/Response 필드 변경 시 문서 업데이트 및 양측 검증

---

## 3. 주요 커밋 히스토리

최근 20개 커밋:

```
055c8b7 성능테스트 페이지 & 소리매칭 관리자페이지 추가
6964591 feat: JamJamBeat 프론트엔드 Lottie 애니메이션 기본 구조 및 디자인 준비
0b9e545 Merge pull request #9 from saehoon861/feature/readme_note
f2d4f9a 실험 및 코드 정리
d48d16a Merge pull request #8 from saehoon861/fix/data-merge
554628e fix: 병합된 CSV 입력 시 source_file 컬럼 값 보존
510029f fix: 데이터 병합시 pandas 3.0 호환성 - source_file 컬럼 중복 insert 오류 수정
4268771 Merge pull request #7 from saehoon861/feature/dataloder
394c3b8 데이터로더 코드 추가 및 모델 학습
559f0bd Merge pull request #6 from saehoon861/feat/label-status-check
d0588f7 feat: 라벨링 클래스 분포 현황 확인 스크립트 추가
6705af4 Merge pull request #5 from saehoon861/fix/labeling-tool
e63ed81 fix: labeling tool 창깨짐 오류 수정
ff7bdc8 Merge pull request #4 from saehoon861/feat/data_landmarks
a686a96 docs: data 기능 내 문서 정리 및 README 작성
019f037 Merge pull request #3 from saehoon861/feature/poc_model_upgrade
fddfad9 poc model upgrade
664e5ef chore: 작업 데이터셋 삭제 및 gitignore 추가.
67314cb feat: 데이터 통합 검수 기능 확대 구현 완료
3065463 docs: 데이터 통합 검수 추가 기능 구현 계획서 작성
```

---

## 4. 현재 브랜치 상태

**현재 브랜치**: `feature/front_test`
**메인 브랜치**: `main`

### 수정된 파일

- `frontend/src/js/main.js` (리팩토링 진행 중)

### 추적되지 않은 파일 (새로 추가됨)

**데이터 파일**:
- `data/landmark_data/`: 새로운 랜드마크 데이터 (`0-6_fast/slow_right_man2.csv`)
- `data/labeled_data/`: 새로운 라벨링 데이터 (`0-2_*.csv`)

**프론트엔드 모듈** (리팩토링 결과):
- `frontend/src/js/hand_tracking_runtime.js`
- `frontend/src/js/instrument_layout.js`
- `frontend/src/js/interaction_runtime.js`
- `frontend/src/js/particle_system.js`
- `frontend/src/js/scene_runtime.js`
- `frontend/src/js/sound_mapping.js`
- `frontend/REFACTORING_NOTES.md`

**문서**:
- `docs/` (새 디렉토리)

---

## 5. 기술적 도전 과제 및 해결

### 5.1 프론트엔드 코드 복잡도

**문제**: `main.js`가 1000줄 초과, 여러 책임이 혼재

**해결**:
- 책임 단위로 6개 모듈 분리
- 팩토리 패턴 도입 (`createParticleSystem`, `createFeverController` 등)
- `main.js`를 앱 조립 파일로 단순화 (496줄)

### 5.2 데이터 병합 시 Pandas 호환성

**문제**: Pandas 3.0에서 `source_file` 컬럼 중복 insert 오류

**해결**: 컬럼 존재 여부 체크 후 조건부 insert (커밋 `510029f`)

### 5.3 라벨링 도구 UI 버그

**문제**: 라벨링 도구 창 깨짐 현상

**해결**: UI 레이아웃 수정 (커밋 `e63ed81`)

### 5.4 프론트엔드-추론 인터페이스 안정성

**문제**: 라벨 변경 시 프론트엔드 충돌 위험

**해결**:
- `GESTURE_CONTRACT.md` 작성으로 명시적 계약 관리
- `gestures.js`에서 alias 기반 라벨 정규화
- 신뢰도 임계값 및 안정화 로직 명문화

---

## 6. 남은 작업 및 향후 계획

### 6.1 프론트엔드

**우선순위 높음**:
- [ ] `theme.js`와 `main.js`의 MediaPipe/카메라 초기화 코드 통합
- [ ] 실제 브라우저 환경에서 리팩토링 코드 검증

**우선순위 중간**:
- [ ] `admin.js`와 런타임의 사운드 매핑 규칙 통합

**우선순위 낮음**:
- [ ] Lottie 애니메이션 통합 (기본 구조는 준비됨)
- [ ] 피버 모드 효과 고도화

### 6.2 데이터 파이프라인

- [ ] 추가 제스처 데이터 수집 및 라벨링
- [ ] 클래스 불균형 해소 (데이터 증강 또는 가중치 조정)
- [ ] 라벨링 도구 UX 개선

### 6.3 모델 학습

- [ ] Transformer 모델 성능 최적화
- [ ] 하이퍼파라미터 튜닝 (wandb sweep)
- [ ] 모델 경량화 (추론 속도 개선)
- [ ] 모델 버전 관리 및 A/B 테스트

### 6.4 추론 서버

- [ ] 배포 환경 구축 (Docker, 클라우드)
- [ ] 모니터링 및 로깅 강화
- [ ] 신뢰도 기반 fallback 로직 개선

### 6.5 문서화

- [x] 프로젝트 진행 사항 문서 작성
- [ ] API 문서 자동 생성 (FastAPI Swagger)
- [ ] 사용자 매뉴얼 작성

---

## 7. 참고 문서

- **프론트엔드 리팩토링**: [`frontend/REFACTORING_NOTES.md`](../frontend/REFACTORING_NOTES.md)
- **데이터 파이프라인**: [`data/README.md`](../data/README.md)
- **제스처 계약**: [`docs/GESTURE_CONTRACT.md`](./GESTURE_CONTRACT.md)
- **협업 워크플로우**: [`docs/COLLAB_WORKFLOW.md`](./COLLAB_WORKFLOW.md)

---

## 8. 팀 및 연락처

프로젝트에 대한 질문이나 피드백은 GitHub Issues를 통해 남겨주세요.

**Repository**: [JAMMJAMM/JamJamBeat](https://github.com/saehoon861/JamJamBeat)

---

**Last Updated**: 2026-03-12
**Document Version**: 1.0
