# GestureMLP Training

이 스크립트는 CSV 형식의 손 랜드마크 데이터를 이용해 GestureMLP 모델을 학습하고,  
validation loss 기준으로 best model을 저장하며, test set 성능 평가와 optional grid search를 수행합니다.

## Features

- CSV 기반 손 랜드마크 데이터 로딩
- GestureMLP 모델 학습
- Train / Validation / Test 분할
- Early Stopping 지원
- Best model 저장
- Learning rate scheduler 지원
- Weights & Biases(W&B) 로깅
- Optional grid search 지원

## Input Data Format

입력 CSV는 다음과 같은 컬럼 구조를 가정합니다.

- feature columns: `x0, y0, z0, ..., x20, y20, z20`
- label column: 기본값 `gesture`

총 입력 feature 수는 63개(21개 랜드마크 × 3축)입니다.

## Code Structure

### `EarlyStopping`
validation loss가 일정 epoch 동안 개선되지 않으면 학습을 조기 종료합니다.

### `set_seed(seed)`
실험 재현성을 위해 랜덤 시드를 고정합니다.

### `calculate_accuracy(outputs, labels)`
모델 출력과 정답 라벨을 비교하여 accuracy를 계산합니다.

### `train_one_experiment(config, save_dir)`
단일 설정(config)에 대해 모델 학습, validation 평가, best model 저장, test 평가를 수행합니다.

### `manual_grid_search(param_grid, save_dir)`
여러 하이퍼파라미터 조합을 실험하여 가장 좋은 validation loss를 갖는 설정을 찾습니다.

### `build_parser()`
CLI 인자를 정의합니다.

### `main()`
전체 실행 진입점으로, 일반 학습 또는 grid search를 실행합니다.

## Training Pipeline

1. CLI 인자를 통해 학습 설정을 입력받습니다.
2. 입력 feature column (`x0 ~ z20`)을 구성합니다.
3. CSV 데이터를 train / validation / test로 분할합니다.
4. GestureMLP 모델을 생성합니다.
5. Optimizer, scheduler, early stopping을 설정합니다.
6. Epoch 단위로 train / validation을 반복합니다.
7. validation loss가 개선되면 best model을 저장합니다.
8. early stopping 조건을 만족하면 학습을 종료합니다.
9. 가장 성능이 좋았던 모델(best model)로 test set을 평가합니다.
10. 결과를 W&B 및 출력값으로 기록합니다.

## Model Checkpoint Rule

이 스크립트는 **validation accuracy가 아니라 validation loss 기준으로** best model을 저장합니다.

- validation loss가 개선되면 model checkpoint를 갱신합니다.
- 최종 test 평가는 마지막 epoch 모델이 아니라, 가장 낮은 validation loss를 기록한 best model로 수행합니다.

## Metric Calculation

각 배치의 loss와 accuracy는 배치 크기를 반영하여 누적됩니다.

- batch 평균 loss × batch size
- batch accuracy × batch size

이후 전체 샘플 수로 나누어 epoch 평균 loss / accuracy를 계산합니다.

## Main Arguments

- `--csv-path`: 입력 CSV 파일 경로
- `--label-col`: 라벨 컬럼 이름
- `--input-size`: 입력 차원 수
- `--num-classes`: 클래스 수
- `--hidden-dims`: 은닉층 크기 (예: `"128,64"`)
- `--dropout`: dropout 비율
- `--use-batchnorm`: batch normalization 사용 여부
- `--optimizer-name`: optimizer 종류 (`adam`, `sgd`)
- `--learning-rate`: 학습률
- `--weight-decay`: weight decay 값
- `--batch-size`: 배치 크기
- `--num-epochs`: epoch 수
- `--scheduler-name`: scheduler 종류 (`none`, `StepLR`, `ReduceLROnPlateau`)
- `--early-stopping-patience`: early stopping patience
- `--early-stopping-min-delta`: 개선으로 인정할 최소 변화량
- `--balance-train`: train 데이터 balancing 여부
- `--majority-class`: 다수 클래스 번호
- `--majority-ratio`: 다수 클래스 비율 조정값

## Usage

### Single Training
```bash
make train-mlp DROPOUT=0.005 LR=0.0005 WEIGHT_DECAY=0.0001 BATCH_NORM=True SCHEDULER_NAME=ReduceLROnPlateau

---

# GestureMLP Evaluation

이 스크립트는 CSV 형식의 손 랜드마크 데이터를 이용해  
**학습된 GestureMLP 모델을 불러온 뒤**,  
**test set 성능을 평가하고 confusion matrix 및 classification report를 출력/저장**합니다.

## Features

- CSV 기반 손 랜드마크 데이터 로딩
- 저장된 GestureMLP 모델 체크포인트 로드
- Train / Validation / Test 분할 구조 재사용
- Test set 기준 loss / accuracy 평가
- Confusion Matrix 출력
- Classification Report 출력
- Confusion Matrix CSV 저장
- Test prediction 결과 CSV 저장

## Input Data Format

입력 CSV는 다음과 같은 컬럼 구조를 가정합니다.

- feature columns: `x0, y0, z0, ..., x20, y20, z20`
- label column: 기본값 `gesture`

총 입력 feature 수는 63개(21개 랜드마크 × 3축)입니다.

## Code Structure

### `set_seed(seed)`
실험 재현성을 위해 랜덤 시드를 고정합니다.

### `parse_hidden_dims(hidden_dims_str)`
문자열 형태의 hidden dimension 설정값(예: `"128,64"`)을 정수 리스트로 변환합니다.

### `str2bool(v)`
CLI에서 입력된 문자열 값을 boolean 값으로 변환합니다.

### `evaluate_model(config)`
설정(config)에 따라 데이터로더를 생성하고,  
저장된 GestureMLP 모델을 불러와 test set 성능을 평가합니다.  
평가 결과로 test loss, test accuracy, confusion matrix, classification report를 출력하며,  
옵션에 따라 confusion matrix와 prediction 결과를 파일로 저장합니다.

### `build_parser()`
CLI 인자를 정의합니다.

### `main()`
전체 실행 진입점으로, CLI 인자를 파싱한 뒤 config를 구성하여 평가를 수행합니다.

## Evaluation Pipeline

1. CLI 인자를 통해 평가 설정을 입력받습니다.
2. 입력 feature column (`x0 ~ z20`)을 구성합니다.
3. CSV 데이터를 train / validation / test로 분할합니다.
4. GestureMLP 모델 구조를 생성합니다.
5. 저장된 체크포인트(`.pth`)를 불러옵니다.
6. 모델을 evaluation mode로 전환합니다.
7. Test set 전체에 대해 예측을 수행합니다.
8. Test loss와 accuracy를 계산합니다.
9. 전체 예측 결과를 기반으로 confusion matrix와 classification report를 생성합니다.
10. 결과를 출력하고, 필요 시 confusion matrix와 prediction 결과를 CSV로 저장합니다.

## Model Checkpoint Rule

이 스크립트는 **새로 학습을 수행하지 않고**,  
이미 저장된 모델 체크포인트를 불러와 평가만 수행합니다.

- `--model-path`로 지정한 모델 가중치를 로드합니다.
- 평가에 사용되는 모델 구조는 CLI 인자로 입력한 설정과 동일해야 합니다.
- 최종 평가는 마지막 epoch 모델이 아니라, 사용자가 지정한 checkpoint 파일 기준으로 수행됩니다.

## Metric Calculation

각 배치의 loss와 accuracy는 배치 크기를 반영하여 누적됩니다.

- batch 평균 loss × batch size
- batch 정답 개수 누적

이후 전체 test 샘플 수로 나누어 최종 평균 test loss / accuracy를 계산합니다.

또한 전체 test set의 실제 라벨과 예측 라벨을 수집하여 다음 지표를 계산합니다.

- confusion matrix
- classification report (`precision`, `recall`, `f1-score`, `support`)

## Main Arguments

- `--csv-path`: 입력 CSV 파일 경로
- `--model-path`: 불러올 모델 체크포인트 경로
- `--label-col`: 라벨 컬럼 이름
- `--input-size`: 입력 차원 수
- `--num-classes`: 클래스 수
- `--hidden-dims`: 은닉층 크기 (예: `"128,64"`)
- `--dropout`: dropout 비율
- `--use-batchnorm`: batch normalization 사용 여부
- `--batch-size`: 배치 크기
- `--validation-split-ratio`: validation 데이터 비율
- `--test-split-ratio`: test 데이터 비율
- `--test-split-seed`: test 분할 시드
- `--train-val-split-seed`: train/validation 분할 시드
- `--train-seed`: 랜덤 시드
- `--balance-train`: train 데이터 balancing 여부
- `--majority-class`: 다수 클래스 번호
- `--majority-ratio`: 다수 클래스 비율 조정값
- `--save-confusion-matrix-path`: confusion matrix 저장 경로
- `--save-predictions-path`: test prediction 저장 경로

## Outputs

이 스크립트는 다음 결과를 생성할 수 있습니다.

- terminal 출력
  - Test loss
  - Test accuracy
  - Confusion Matrix
  - Classification Report

- optional file outputs
  - confusion matrix CSV
  - test prediction CSV (`y_true`, `y_pred`)

## Usage

### Single Evaluation
```bash
make eval-mlp DROPOUT=0.005 BATCH_NORM=True

---

