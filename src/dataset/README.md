# dataset/

모델 학습에 사용할 데이터셋을 구성하는 코드입니다.

랜드마크 데이터를 불러오고  
모델 입력 형태로 변환합니다.

---

# Gesture Dataset Utilities

이 스크립트는 CSV 형식의 손 랜드마크 데이터를 읽어와 정제하고,  
PyTorch 학습에 바로 사용할 수 있도록 **DataFrame / Tensor / Dataset / DataLoader** 형태로 변환합니다.  
또한 train / validation / test 분할과 **다수 클래스 언더샘플링 기반 balancing** 기능도 제공합니다. :contentReference[oaicite:0]{index=0}

## Features

- CSV 기반 손 랜드마크 데이터 로딩
- 필수 컬럼 검증
- 결측치 및 잘못된 label 행 제거
- DataFrame → Tensor 변환
- PyTorch `Dataset` 클래스 제공
- Train / Validation / Test 분할
- Stratified split 지원
- Train set 전용 클래스 불균형 보정(undersampling)
- `DataLoader` 생성 지원
- 단독 실행 시 데이터 디버깅 출력 지원

## Input Data Format

입력 CSV는 다음과 같은 컬럼 구조를 가정합니다.

- feature columns: `x0, y0, z0, ..., x20, y20, z20`
- label column: 기본값 `gesture`

총 입력 feature 수는 63개(21개 랜드마크 × 3축)입니다.  
추가적인 시간, 초, 프레임 관련 컬럼이 있어도 사용하지 않으며, 필요한 feature와 label만 남깁니다. :contentReference[oaicite:1]{index=1}

## Code Structure

### `LABEL_COL`
기본 label 컬럼명을 정의합니다. 기본값은 `gesture`입니다. :contentReference[oaicite:2]{index=2}

### `_validate_feature_columns(feature_columns)`
입력으로 전달된 feature column 개수가 정확히 63개인지 검증합니다.  
개수가 다르면 `ValueError`를 발생시킵니다. :contentReference[oaicite:3]{index=3}

### `_check_required_columns(df, feature_columns, label_col)`
CSV/DataFrame에 필요한 feature 컬럼들과 label 컬럼이 모두 존재하는지 확인합니다.  
누락된 컬럼이 있으면 `ValueError`를 발생시킵니다. :contentReference[oaicite:4]{index=4}

### `_drop_invalid_rows(df, feature_columns, label_col, num_classes)`
유효하지 않은 행을 제거합니다.

정제 규칙은 다음과 같습니다.

- 63개 feature 중 하나라도 결측치면 제거
- label이 없으면 제거
- label이 `0 ~ num_classes-1` 범위를 벗어나면 제거
- 숫자로 변환할 수 없는 값은 `NaN`으로 처리 후 제거

정제 후 label은 `int` 타입으로 변환되고, index는 다시 정리됩니다. :contentReference[oaicite:5]{index=5}

### `load_gesture_dataframe(csv_path, feature_columns, label_col, num_classes)`
CSV 파일을 읽은 뒤, 필요한 컬럼만 남기고 유효하지 않은 행을 제거한 **정제된 DataFrame**을 반환합니다.  
추가 컬럼은 자동으로 제외됩니다. :contentReference[oaicite:6]{index=6}

### `load_gesture_tensors(csv_path, feature_columns, label_col, num_classes)`
정제된 CSV 데이터를 PyTorch Tensor로 변환합니다.

- `X`: `float32`, shape `[N, 63]`
- `y`: `long`, shape `[N]`

학습 입력과 라벨을 바로 모델에 넣을 수 있는 형태로 반환합니다. :contentReference[oaicite:7]{index=7}

### `undersample_majority_class(X, y, majority_class, majority_ratio, random_state)`
train set에만 적용하기 위한 언더샘플링 함수입니다.

기본적으로 다수 클래스(`majority_class=0`)의 개수를  
**다른 클래스 중 가장 많은 클래스 개수 × `majority_ratio`** 이하로 제한합니다.

예를 들어:

- 다른 클래스 중 최다 개수 = 200
- `majority_ratio = 1.5`

이면 다수 클래스는 최대 300개까지만 유지됩니다.  
초과하는 경우 랜덤하게 일부 샘플만 남기고 줄입니다. :contentReference[oaicite:8]{index=8}

### `GestureCSVDataset`
CSV를 기반으로 직접 사용할 수 있는 PyTorch `Dataset` 클래스입니다.

- `__len__()` : 전체 샘플 수 반환
- `__getitem__(idx)` : 해당 인덱스의 `(X, y)` 반환

필요하면 `TensorDataset` 대신 이 클래스를 직접 사용할 수 있습니다. :contentReference[oaicite:9]{index=9}

### `create_dataloaders(...)`
이 스크립트의 핵심 함수로, 다음 두 가지 방식 중 하나로 데이터를 받을 수 있습니다.

1. `csv_path + feature_columns`를 전달해서 CSV에서 직접 로드
2. 이미 준비된 `X`, `y` Tensor를 직접 전달

그 후 아래 작업을 수행합니다.

- 전체 데이터 로드
- Tensor 타입 정리 (`X=float32`, `y=long`)
- train_val / test 1차 분할
- train / val 2차 분할
- train set에만 balancing 적용
- `TensorDataset` 생성
- `DataLoader` 생성 및 반환

반환값은 다음과 같습니다.

- `train_loader`
- `val_loader`
- `test_loader`
- `train_dataset`
- `val_dataset`
- `test_dataset` :contentReference[oaicite:10]{index=10}

## Data Cleaning Rule

이 스크립트는 다음 기준으로 데이터를 정제합니다.

- feature 컬럼은 반드시 63개여야 합니다.
- CSV에는 feature 컬럼들과 label 컬럼이 모두 존재해야 합니다.
- feature 또는 label에 결측치가 있는 행은 제거합니다.
- 숫자로 변환할 수 없는 문자열 값은 `NaN` 처리 후 제거합니다.
- label은 반드시 `0 ~ num_classes-1` 범위 안에 있어야 합니다.
- 최종적으로 label은 정수형으로 변환됩니다. :contentReference[oaicite:11]{index=11}

## Data Split Pipeline

`create_dataloaders()`의 분할 과정은 다음과 같습니다.

1. 전체 데이터를 `train_val` / `test`로 먼저 분할합니다.
2. `train_val` 데이터를 다시 `train` / `validation`으로 분할합니다.
3. 모든 분할은 `stratify`를 사용하여 label 분포를 최대한 유지합니다.
4. balancing은 오직 `train set`에만 적용합니다.
5. 이후 각 split을 `TensorDataset`과 `DataLoader`로 변환합니다. :contentReference[oaicite:12]{index=12}

## Split Rule

이 스크립트는 분할 시 seed를 분리해서 관리합니다.

- `test_split_seed`: train_val / test 분할에만 사용
- `train_val_split_seed`: train / validation 분할과 train loader shuffle에 사용

즉, test set은 `test_split_seed`에만 의존하고,  
train/validation 구성은 `train_val_split_seed`에 의존합니다.  
이렇게 하면 test set을 고정한 채 train/validation만 재구성하는 실험이 가능합니다. :contentReference[oaicite:13]{index=13}

## Balancing Rule

클래스 불균형 보정은 **train set에만 적용**됩니다.

- `balance_train=True`일 때만 수행
- `majority_class`로 지정한 클래스만 감소 대상
- 다른 클래스는 그대로 유지
- 최종 인덱스를 다시 한 번 섞어서 반환

validation set과 test set은 원본 분포를 유지합니다.  
이는 평가 데이터 분포를 인위적으로 바꾸지 않기 위함입니다. :contentReference[oaicite:14]{index=14}

## Output Objects

이 스크립트는 상황에 따라 다음 객체들을 생성합니다.

- 정제된 `pandas.DataFrame`
- 입력 Tensor `X`
- 정답 Tensor `y`
- `GestureCSVDataset`
- `TensorDataset`
- `train_loader`, `val_loader`, `test_loader`

즉, 전처리부터 학습용 입력 준비까지 한 번에 담당하는 데이터 유틸리티 모듈입니다. :contentReference[oaicite:15]{index=15}

## Main Arguments

### `load_gesture_dataframe()`
- `csv_path`: 입력 CSV 경로
- `feature_columns`: 사용할 63개 feature 컬럼 리스트
- `label_col`: label 컬럼 이름
- `num_classes`: 클래스 수

### `load_gesture_tensors()`
- `csv_path`: 입력 CSV 경로
- `feature_columns`: 사용할 63개 feature 컬럼 리스트
- `label_col`: label 컬럼 이름
- `num_classes`: 클래스 수

### `undersample_majority_class()`
- `X`: 입력 feature Tensor
- `y`: label Tensor
- `majority_class`: 다수 클래스 번호
- `majority_ratio`: 다수 클래스 비율 조정값
- `random_state`: 랜덤 시드

### `create_dataloaders()`
- `csv_path`: 입력 CSV 경로
- `X`: 입력 feature Tensor
- `y`: label Tensor
- `feature_columns`: 사용할 63개 feature 컬럼 리스트
- `label_col`: label 컬럼 이름
- `num_classes`: 클래스 수
- `batch_size`: 배치 크기
- `validation_split_ratio`: validation 비율
- `test_split_ratio`: test 비율
- `test_split_seed`: test 분할 시드
- `train_val_split_seed`: train/validation 분할 시드
- `balance_train`: train balancing 여부
- `majority_class`: 다수 클래스 번호
- `majority_ratio`: 다수 클래스 비율 조정값

---
