# dataset.py - grab 제스처 학습용 Dataset 스텁
# TODO: build() 함수 본문을 구현하세요.
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

# MediaPipe 21 관절 raw 좌표 컬럼 (x0,y0,z0 ... x20,y20,z20)
RAW_JOINT_COLS: list[str] = [f"{ax}{i}" for i in range(21) for ax in ("x", "y", "z")]  # 63d


def add_delta_features(x_seq: np.ndarray) -> np.ndarray:
    """(N, T, 63) 배열에 1차 차분을 이어붙여 (N, T, 126)을 반환한다.

    첫 번째 프레임의 delta는 0으로 채운다.
    """
    delta = np.zeros_like(x_seq, dtype=np.float32)
    delta[:, 1:, :] = x_seq[:, 1:, :] - x_seq[:, :-1, :]
    return np.concatenate([x_seq.astype(np.float32), delta], axis=2)


class SequenceDataset(Dataset):
    """(N, T, D) 시퀀스 배열과 레이블을 묶는 Dataset.

    DataLoader에서 (x, y, idx) 튜플을 반환한다.
    train.py의 weighted sampler가 .y를 직접 참조한다.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.from_numpy(x.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], idx


def build(
    train_csv: str,
    val_csv: str | None,
    test_csv: str | None,
    seq_len: int = 8,
    seq_stride: int = 2,
) -> tuple[SequenceDataset, SequenceDataset | None, SequenceDataset | None]:
    """TODO: 이 함수를 구현하세요.

    입력 CSV 스키마 (landmark_extractor 출력):
        frame_idx, timestamp, gesture, x0, y0, z0, ..., x20, y20, z20

    구현 흐름:
        1. 각 CSV를 pandas로 읽어 RAW_JOINT_COLS(63d) 추출
        2. sliding window로 (N, T=seq_len, 63) 시퀀스 배열 생성
        3. add_delta_features()로 (N, T, 126)으로 확장
        4. 레이블은 각 윈도우의 마지막 프레임 gesture 값 사용
        5. SequenceDataset(x, y) 반환
           val_csv / test_csv 가 None 이면 해당 반환값도 None으로 처리

    반환값: (train_ds, val_ds | None, test_ds | None)
    """
    raise NotImplementedError(
        "dataset.py의 build()를 구현해야 합니다.\n"
        "landmark_queue/ 또는 landmark_data/ CSV를 읽어 SequenceDataset을 반환하세요."
    )
