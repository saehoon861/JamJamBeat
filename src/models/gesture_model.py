import os
from typing import Optional, Sequence

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


LABEL_COL = "gesture"


def _validate_feature_columns(feature_columns: Sequence[str]) -> None:
    if len(feature_columns) != 63:
        raise ValueError(
            f"feature_columns는 반드시 63개여야 합니다. 현재 개수: {len(feature_columns)}"
        )


def _drop_invalid_rows(
    df: pd.DataFrame,
    feature_columns: Sequence[str],
    label_col: str = LABEL_COL,
    num_classes: int = 7,
) -> pd.DataFrame:
    required_cols = list(feature_columns) + [label_col]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing_cols}")

    # feature 중 하나라도 결측치면 제거
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

    df = df[list(feature_columns) + [label_col]].copy()
    return df


def load_gesture_tensors(
    csv_path: str,
    feature_columns: Sequence[str],
    label_col: str = LABEL_COL,
    num_classes: int = 7,
) -> tuple[torch.Tensor, torch.Tensor]:
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
    max_allowed_majority = int(max_non_majority * majority_ratio)

    if majority_count <= max_allowed_majority:
        return X, y

    rng = np.random.default_rng(random_state)

    majority_indices = torch.where(y_cpu == majority_class)[0].numpy()
    other_indices = torch.where(y_cpu != majority_class)[0].numpy()

    sampled_majority_indices = rng.choice(
        majority_indices,
        size=max_allowed_majority,
        replace=False,
    )

    kept_indices = np.concatenate([sampled_majority_indices, other_indices])
    rng.shuffle(kept_indices)

    kept_indices = torch.tensor(kept_indices, dtype=torch.long)

    return X[kept_indices], y[kept_indices]


def create_dataloaders(
    csv_path: Optional[str] = None,
    feature_columns: Optional[Sequence[str]] = None,
    label_col: str = LABEL_COL,
    num_classes: int = 7,
    X: Optional[torch.Tensor] = None,
    y: Optional[torch.Tensor] = None,
    batch_size: int = 32,
    validation_split_ratio: float = 0.2,
    test_split_ratio: float = 0.1,
    random_state: int = 42,
    balance_train: bool = True,
    majority_class: int = 0,
    majority_ratio: float = 1.5,
):
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

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X,
        y,
        test_size=test_split_ratio,
        random_state=random_state,
        stratify=y,
    )

    val_ratio_adjusted = validation_split_ratio / (1.0 - test_split_ratio)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=val_ratio_adjusted,
        random_state=random_state,
        stratify=y_train_val,
    )

    if balance_train:
        X_train, y_train = undersample_majority_class(
            X_train,
            y_train,
            majority_class=majority_class,
            majority_ratio=majority_ratio,
            random_state=random_state,
        )

    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)
    test_dataset = TensorDataset(X_test, y_test)

    # 재현 가능한 셔플을 위해 generator 고정
    train_generator = torch.Generator()
    train_generator.manual_seed(random_state)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset,
        val_dataset,
        test_dataset,
    )


if __name__ == "__main__":
    csv_path = "/home/kimsaehoon/workspace/JamJamBeat/src/dataset/man1_right_for_poc.csv"

    # 실제 CSV가 x0~z20 구조이므로 0~20으로 생성
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
                random_state=42,
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