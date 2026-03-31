import os
from typing import Optional, Sequence

import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


LABEL_COL = "gesture"


def _validate_feature_columns(feature_columns: Sequence[str]) -> None:
    if len(feature_columns) != 63:
        raise ValueError(
            f"feature_columns는 반드시 63개여야 합니다. 현재 개수: {len(feature_columns)}"
        )


def _check_required_columns(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_col: str = LABEL_COL,
) -> None:
    required_cols = list(feature_columns) + [label_col]
    missing_cols = [col for col in required_cols if col not in df.columns]

    if missing_cols:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing_cols}")


def _drop_invalid_rows(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_col: str = LABEL_COL,
    num_classes: int = 7,
) -> pd.DataFrame:
    """
    규칙:
    - 63개 feature 중 하나라도 결측치면 제거
    - label이 없거나 0~num_classes-1 범위를 벗어나면 제거
    """
    required_cols = list(feature_columns) + [label_col]

    _check_required_columns(df, feature_columns, label_col)

    df = df.copy()

    # 숫자형으로 강제 변환 (문자/빈칸 등도 NaN 처리)
    for col in required_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # feature 63개 + label 결측 제거
    df = df.dropna(subset=required_cols)

    # label 범위 필터
    df = df[df[label_col].between(0, num_classes - 1)]

    # label int 변환
    df[label_col] = df[label_col].astype(int)

    return df.reset_index(drop=True)


def load_gesture_dataframe(
    csv_path: str,
    feature_columns: Sequence[str],
    label_col: str = LABEL_COL,
    num_classes: int = 7,
) -> pd.DataFrame:
    """
    CSV를 읽고, 필요한 feature와 label만 남긴 뒤 유효하지 않은 행 제거.
    나머지 시간/초/프레임 컬럼은 자동으로 버려짐.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV 파일을 찾을 수 없습니다: {csv_path}")

    _validate_feature_columns(feature_columns)

    df = pd.read_csv(csv_path)
    df = _drop_invalid_rows(
        df=df,
        feature_columns=feature_columns,
        label_col=label_col,
        num_classes=num_classes,
    )

    # 필요한 컬럼만 남김
    df = df[list(feature_columns) + [label_col]].copy()
    return df


def load_gesture_tensors(
    csv_path: str,
    feature_columns: Sequence[str],
    label_col: str = LABEL_COL,
    num_classes: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    CSV -> 정제 -> X, y 텐서 변환
    X: float32, shape [N, 63]
    y: long, shape [N]
    """
    df = load_gesture_dataframe(
        csv_path=csv_path,
        feature_columns=feature_columns,
        label_col=label_col,
        num_classes=num_classes,
    )

    X = torch.tensor(df[list(feature_columns)].values, dtype=torch.float32)
    y = torch.tensor(df[label_col].values, dtype=torch.long)

    return X, y


def undersample_majority_class(
    X: torch.Tensor,
    y: torch.Tensor,
    majority_class: int = 0,
    majority_ratio: float = 1.5,
    random_state: int = 42,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    train set에만 적용할 언더샘플링.
    majority_class(기본 0)의 개수를
    '다른 클래스 중 최다 개수 * majority_ratio' 이하로 제한.

    예:
    다른 클래스 최대 개수가 200이고 majority_ratio=1.5면
    class 0은 최대 300개까지만 유지.
    """
    if majority_ratio <= 0:
        raise ValueError("majority_ratio는 0보다 커야 합니다.")

    y_cpu = y.cpu()
    class_counts = torch.bincount(y_cpu)

    if majority_class >= len(class_counts):
        return X, y

    majority_count = int(class_counts[majority_class].item())

    non_majority_counts = [
        int(class_counts[i].item())
        for i in range(len(class_counts))
        if i != majority_class and class_counts[i].item() > 0
    ]

    if len(non_majority_counts) == 0:
        return X, y

    max_non_majority = max(non_majority_counts)
    target_majority_count = int(max_non_majority * majority_ratio)

    # 이미 충분히 적으면 그대로 반환
    if majority_count <= target_majority_count:
        return X, y

    g = torch.Generator()
    g.manual_seed(random_state)

    majority_indices = torch.where(y == majority_class)[0]
    other_indices = torch.where(y != majority_class)[0]

    perm = torch.randperm(len(majority_indices), generator=g)
    kept_majority_indices = majority_indices[perm[:target_majority_count]]

    final_indices = torch.cat([kept_majority_indices, other_indices], dim=0)

    # 섞기
    final_perm = torch.randperm(len(final_indices), generator=g)
    final_indices = final_indices[final_perm]

    return X[final_indices], y[final_indices]


class GestureCSVDataset(Dataset):
    """
    필요하면 직접 Dataset으로도 쓸 수 있게 만든 버전.
    """
    def __init__(
        self,
        csv_path: str,
        feature_columns: Sequence[str],
        label_col: str = LABEL_COL,
        num_classes: int = 7,
    ):
        self.X, self.y = load_gesture_tensors(
            csv_path=csv_path,
            feature_columns=feature_columns,
            label_col=label_col,
            num_classes=num_classes,
        )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.X[idx], self.y[idx]


def create_dataloaders(
    csv_path: Optional[str] = None,
    X: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    feature_columns: Optional[Sequence[str]] = None,
    label_col: str = LABEL_COL,
    num_classes: int = 7,
    batch_size: int = 32,
    validation_split_ratio: float = 0.2,
    test_split_ratio: float = 0.1,
    test_split_seed: int = 42,
    train_val_split_seed: int = 42,
    balance_train: bool = True,
    majority_class: int = 0,
    majority_ratio: float = 1.5,
):
    """
    사용 방법 1:
        create_dataloaders(
            csv_path="data/train.csv",
            feature_columns=[...63개...],
            ...
        )

    사용 방법 2:
        create_dataloaders(X=X, y=y, ...)
    """
    if csv_path is not None:
        if feature_columns is None:
            raise ValueError("csv_path를 사용할 경우 feature_columns를 반드시 전달해야 합니다.")
        X, y = load_gesture_tensors(
            csv_path=csv_path,
            feature_columns=feature_columns,
            label_col=label_col,
            num_classes=num_classes,
        )
    else:
        if X is None or y is None:
            raise ValueError("csv_path 또는 (X, y) 둘 중 하나는 반드시 제공해야 합니다.")

    if not isinstance(X, torch.Tensor):
        X = torch.tensor(X, dtype=torch.float32)
    else:
        X = X.to(torch.float32)

    if not isinstance(y, torch.Tensor):
        y = torch.tensor(y, dtype=torch.long)
    else:
        y = y.to(torch.long)

    # 1차 분할: train_val / test
    # => test set은 test_split_seed에만 의존
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_split_ratio,
        random_state=test_split_seed,
        stratify=y,
    )

    # 2차 분할: train / val
    # => train/val만 train_val_split_seed에 의존
    val_ratio_adjusted = validation_split_ratio / (1.0 - test_split_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_adjusted,
        random_state=train_val_split_seed,
        stratify=y_train_val,
    )

    # 클래스 불균형 보정은 train에만 적용
    if balance_train:
        X_train, y_train = undersample_majority_class(
            X_train,
            y_train,
            majority_class=majority_class,
            majority_ratio=majority_ratio,
            random_state=train_val_split_seed,
        )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # train loader 셔플도 train_val_split_seed 기준으로 고정
    train_generator = torch.Generator()
    train_generator.manual_seed(train_val_split_seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )


if __name__ == "__main__":
    # 디버깅용: gesture_dataset.py 단독 실행해서 데이터 확인
    csv_path = "/home/kimsaehoon/workspace/JamJamBeat/data/total_data_0309.csv"

    feature_columns = []
    for i in range(0, 21):
        feature_columns.extend([f"x{i}", f"y{i}", f"z{i}"])

    print("=" * 80)
    print("[원본 CSV 확인]")
    print("=" * 80)

    if not os.path.exists(csv_path):
        print(f"CSV 파일이 없습니다: {csv_path}")
    else:
        df_raw = pd.read_csv(csv_path)

        print("\n[원본 head]")
        print(df_raw.head())

        print("\n[원본 shape]")
        print(df_raw.shape)

        required_cols = feature_columns + [LABEL_COL]
        missing_cols = [col for col in required_cols if col not in df_raw.columns]

        if missing_cols:
            print("\n[필수 컬럼 누락]")
            print(missing_cols)
        else:
            print("\n[원본 결측치 개수 합계]")
            print(df_raw[required_cols].isnull().sum().sum())

            print("\n[원본 각 컬럼 결측치 개수 상위 10개]")
            print(df_raw[required_cols].isnull().sum().sort_values(ascending=False).head(10))

            df_clean = load_gesture_dataframe(
                csv_path=csv_path,
                feature_columns=feature_columns,
                label_col=LABEL_COL,
                num_classes=7,
            )

            print("\n" + "=" * 80)
            print("[정제 후 데이터 확인]")
            print("=" * 80)

            print("\n[정제 후 head]")
            print(df_clean.head())

            print("\n[정제 후 shape]")
            print(df_clean.shape)

            print("\n[정제 후 결측치 개수 합계]")
            print(df_clean.isnull().sum().sum())

            print("\n[정제 후 label 분포]")
            print(df_clean[LABEL_COL].value_counts().sort_index())

            X, y = load_gesture_tensors(
                csv_path=csv_path,
                feature_columns=feature_columns,
                label_col=LABEL_COL,
                num_classes=7,
            )

            print("\n[텐서 shape]")
            print("X.shape:", X.shape)
            print("y.shape:", y.shape)

            print("\n[전체 데이터 label 분포]")
            print(pd.Series(y.numpy()).value_counts().sort_index())

            (
                train_loader,
                val_loader,
                test_loader,
                train_dataset,
                val_dataset,
                test_dataset,
            ) = create_dataloaders(
                csv_path=csv_path,
                feature_columns=feature_columns,
                label_col=LABEL_COL,
                num_classes=7,
                batch_size=32,
                validation_split_ratio=0.2,
                test_split_ratio=0.1,
                test_split_seed=42,
                train_val_split_seed=42,
                balance_train=True,
                majority_class=0,
                majority_ratio=1.5,
            )

            y_train = train_dataset.tensors[1]
            y_val = val_dataset.tensors[1]
            y_test = test_dataset.tensors[1]

            print("\n" + "=" * 80)
            print("[split 후 데이터 확인]")
            print("=" * 80)

            print("\n[train/val/test 크기]")
            print("train:", len(train_dataset))
            print("val  :", len(val_dataset))
            print("test :", len(test_dataset))

            print("\n[train label 분포]")
            print(pd.Series(y_train.numpy()).value_counts().sort_index())

            print("\n[val label 분포]")
            print(pd.Series(y_val.numpy()).value_counts().sort_index())

            print("\n[test label 분포]")
            print(pd.Series(y_test.numpy()).value_counts().sort_index())