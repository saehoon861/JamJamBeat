# preprocess.py - raw MediaPipe landmark sequence를 pos_scale + delta feature로 변환
from __future__ import annotations

from typing import Any

import numpy as np


EPS = 1e-8


def _as_frame_points(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32)
    if arr.shape == (21, 3):
        return arr.astype(np.float32, copy=False)
    if arr.ndim == 1 and arr.shape[0] == 63:
        return arr.reshape(21, 3).astype(np.float32, copy=False)
    raise ValueError(
        f"Each frame must be shape (21, 3) or (63,), got {arr.shape}"
    )


def _as_raw_sequence(raw_sequence: Any, seq_len: int) -> np.ndarray:
    arr = np.asarray(raw_sequence, dtype=np.float32)
    if arr.shape == (seq_len, 21, 3):
        return arr.astype(np.float32, copy=False)
    if arr.shape == (seq_len, 63):
        return arr.reshape(seq_len, 21, 3).astype(np.float32, copy=False)
    raise ValueError(
        f"Expected raw sequence shape ({seq_len}, 21, 3) or ({seq_len}, 63), got {arr.shape}"
    )


def apply_pos_scale(frame_points: np.ndarray) -> np.ndarray:
    pts = _as_frame_points(frame_points).astype(np.float32, copy=True)
    origin = pts[0].copy()
    denom = float(np.linalg.norm(pts[9] - origin))
    if denom <= EPS:
        return (pts - origin).astype(np.float32)
    return ((pts - origin) / denom).astype(np.float32)


def normalize_sequence_pos_scale(raw_sequence: Any, seq_len: int = 8) -> np.ndarray:
    seq = _as_raw_sequence(raw_sequence, seq_len=seq_len)
    normalized = np.stack([apply_pos_scale(frame) for frame in seq], axis=0)
    return normalized.astype(np.float32)


def build_delta_features(joint_sequence: np.ndarray) -> np.ndarray:
    seq = np.asarray(joint_sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 63:
        raise ValueError(f"Expected joint sequence shape (T, 63), got {seq.shape}")
    delta = np.zeros_like(seq, dtype=np.float32)
    delta[1:] = seq[1:] - seq[:-1]
    return delta.astype(np.float32)


def prepare_features(raw_sequence: Any, seq_len: int = 8) -> np.ndarray:
    normalized = normalize_sequence_pos_scale(raw_sequence, seq_len=seq_len)
    joint63 = normalized.reshape(seq_len, 63).astype(np.float32)
    delta63 = build_delta_features(joint63)
    feature126 = np.concatenate([joint63, delta63], axis=1).astype(np.float32)
    return feature126
