# evaluation_guide.md - JamJamBeat 평가 결과 사용 가이드

# JamJamBeat 평가 결과 가이드

관련 문서:
- `model/model_pipelines/EVALUATION.md` : 실행 흐름과 CLI 중심 설명
- `model/model_evaluation/모델검증관련파일/evaluation_runtime.py` : 실제 평가 산출물 생성 코드

이 문서는 구현 예제가 아니라, **평가 결과가 어디에 저장되고 어떻게 읽는지**를 빠르게 확인하기 위한 안내서다.

## 개요

현재 평가 결과는 모델 run 단위로 저장된다.

- 공식 랭킹 기준: `*_test.csv`
- 공식 test 해석: `static_images_63d`
- sequence 모델 공식 test 해석: `independent_repeat`
- `preds_inference.csv`는 별도 hold-out 예측 결과이며, sequence 모델은 기존 sliding 해석을 유지한다
- `fp_per_min`은 공식 static-image test에서는 비활성 또는 `N/A`다

## 평가가 생성되는 시점

### 학습 파이프라인

`run_pipeline.py` 또는 `run_all.py`를 실행하면 학습 완료 후 자동으로 평가가 생성된다.

흐름:

```text
학습 완료
  -> preds_test.csv 생성
  -> evaluate_predictions() 호출
  -> evaluation/ 폴더에 CSV / PNG / JSON 저장
```

### 이미지 평가

`image_check_app_train_aligned.py`를 실행하면 이미지 폴더 추론 후 별도 evaluation 결과가 생성된다.

흐름:

```text
이미지 추론 완료
  -> preds_images.csv 생성
  -> evaluate_predictions() 호출
  -> image_inference/.../evaluation/ 폴더에 결과 저장
```

## 결과 저장 경로

### 학습 run 결과

기본 경로:

```text
model/model_evaluation/pipelines/{suite_name}/{model_id}/{run_id}/
```

예시:

```text
model/model_evaluation/pipelines/20260313_172323__ds_1_pos_scale/mlp_baseline/20260313_172340/
```

이 폴더에는 보통 아래 파일이 있다.

- `model.pt`
- `train_history.csv`
- `preds_test.csv`
- `preds_inference.csv`
- `run_summary.json`
- `evaluation/`

### 이미지 평가 결과

기본 경로:

```text
{run_dir}/image_inference/{dataset_slug}/
```

예시:

```text
model/model_evaluation/pipelines/.../{run_id}/image_inference/images/
```

이 폴더에는 보통 아래 파일이 있다.

- `preds_images.csv`
- `inference_summary.json`
- `evaluation/`

## `evaluation/` 폴더 산출물

### 항상 생성되는 기본 파일

- `confusion_matrix.csv`
  - 클래스 기준 confusion matrix 원본 수치
- `confusion_matrix.png`
  - confusion matrix 시각화
- `per_class_report.csv`
  - 클래스별 `precision / recall / f1 / support`
- `per_class_summary.csv`
  - 클래스별 `total_samples / correct_samples / incorrect_samples / accuracy`와 대표 오분류 요약
- `per_class_misclassifications.csv`
  - 실제 클래스와 예측 클래스 조합별 오분류 count
- `latency_cdf.png`
  - latency 분포 CDF
- `metrics_summary.json`
  - 핵심 지표 요약
- `dataset_info.json`
  - 입력 CSV, split 정책, 평가 메타데이터

### `source_file`이 있을 때 추가 생성되는 파일

테스트케이스는 현재 `source_file` 기준으로 본다.

- `per_test_case_report.csv`
  - 케이스별 `total_samples / correct_samples / incorrect_samples / accuracy`
  - 대표 오분류 클래스 포함
- `per_test_case_misclassifications.csv`
  - `source_file`, `true_class`, `pred_class`, `count`
- `test_case_accuracy.png`
  - 케이스별 정확도 그래프
- `test_case_confusion.png`
  - 케이스별 오분류 히트맵

### 이미지 평가에서만 추가 생성될 수 있는 파일

이미지 포즈케이스는 파일명 기반으로 파싱한다.

지원 토큰:

- `BaseP`
- `RollP`
- `PitchP`
- `YawP`
- `NoneNetural`
- `NoneOther`

추가 파일:

- `per_image_pose_case_report.csv`
  - 포즈케이스별 `total_samples / correct_samples / incorrect_samples / error_rate`
- `per_image_pose_case_misclassifications.csv`
  - `pose_case`, `true_class`, `pred_class`, `count`
- `image_pose_case_accuracy.png`
  - 포즈케이스별 정확도 또는 오답률 비교
- `image_pose_case_confusion.png`
  - 포즈케이스별 오분류 히트맵

## 공식 test와 inference / image eval 차이

### 공식 test

- 입력: `학습데이터셋/*_test.csv`
- 역할: 공식 비교와 랭킹
- 성격: 독립 정지사진 기반 63d landmark test
- sequence 모델: `independent_repeat`

### inference

- 입력: `학습데이터셋/*_inference.csv`
- 역할: 별도 hold-out 예측 확인
- 성격: 연속 시퀀스 해석 유지
- sequence 모델: sliding

### image eval

- 입력: `추론용데이터셋` 이미지 폴더
- 역할: 이미지 기반 추론 확인
- 추가 축: `source_file`, `user_id`, `pose_case`, `no_hand`

## 자주 보는 해석 포인트

### 1. 클래스별 성능

먼저 볼 파일:

- `per_class_report.csv`
- `per_class_summary.csv`

주로 보는 값:

- `support`: 클래스 샘플 수
- `precision / recall / f1`
- `correct_samples / incorrect_samples`
- `top_misclassified_pred_class`

### 2. 테스트케이스별 성능

먼저 볼 파일:

- `per_test_case_report.csv`
- `test_case_accuracy.png`
- `per_test_case_misclassifications.csv`

이 파일로 아래 질문에 답할 수 있다.

- 특정 케이스에서 총 몇 개 중 몇 개를 맞췄는지
- 특정 케이스에서 주로 어떤 클래스로 틀렸는지
- 케이스 간 성능 차이가 큰지

### 3. 이미지 포즈케이스별 성능

먼저 볼 파일:

- `per_image_pose_case_report.csv`
- `image_pose_case_accuracy.png`
- `per_image_pose_case_misclassifications.csv`

이 파일로 아래 질문에 답할 수 있다.

- `RollP`, `PitchP`, `YawP` 중 어느 포즈에서 오답률이 높은지
- 특정 포즈케이스가 어떤 클래스로 많이 오분류되는지

### 4. `no_hand`

`status` 컬럼이 있는 평가에서만 집계된다.

- 이미지 평가에서는 `no_hand_count`가 기록될 수 있다
- 일반 pipeline test처럼 `status`가 없으면 no-hand 통계는 생성되지 않는다

### 5. 메타데이터

`metrics_summary.json`과 `dataset_info.json`에서 아래를 확인할 수 있다.

- 입력 CSV 경로
- split 정보
- 공식 test 정책
- 생성된 평가 산출물 목록
- no-hand 통계 사용 여부

## 빠른 확인 순서

1. `run_summary.json`
   - 어떤 모델 / 어떤 dataset key인지 확인
2. `evaluation/metrics_summary.json`
   - accuracy, macro_f1, class0 관련 지표 확인
3. `evaluation/per_class_summary.csv`
   - 클래스별 정답/오답과 대표 오분류 확인
4. `evaluation/per_test_case_report.csv`
   - 케이스별 성능 편차 확인
5. 이미지 평가라면 `evaluation/per_image_pose_case_report.csv`
   - `BaseP / RollP / PitchP / YawP / None*` 기준 비교

## 참고

- 평가 파일 집계본:
  - `model/model_evaluation/comparison_results_all_labeled.csv`
  - `model/model_evaluation/comparison_results_all_labeled.md`
  - `model/model_evaluation/comparison_results_all_labeled.xlsx`
- 뷰어/확인 도구:
  - `model/model_evaluation/모델별영상체크/`
- 대시보드 관련:
  - `model/frontend/eval_dashboard/`
