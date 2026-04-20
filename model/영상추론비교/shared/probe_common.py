# probe_common.py - viewer/frontend 비교 probe 공통 계산과 JSON 유틸
from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


ROOT_DIR = Path("/home/user/projects/JamJamBeat-model3").resolve()
DEFAULT_VIDEO_PATH = Path("/home/user/projects/JamJamBeat/data/raw_data/2_fast_right_man1.mp4").resolve()
DEFAULT_BUNDLE_DIR = (
    ROOT_DIR / "runtime_mlp_sequence_delta 20260319_162614_pos_scale 20260319_162806"
).resolve()
DEFAULT_TASK_MODEL_PATH = (ROOT_DIR / "hand_landmarker.task").resolve()

EPS = 1e-8


@dataclass(slots=True)
class TopKItem:
    rank: int
    class_index: int
    class_label: str
    probability: float


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, payload: Any) -> None:
    ensure_parent(path)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_markdown(path: Path, text: str) -> None:
    ensure_parent(path)
    path.write_text(text, encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def frame_timestamp_ms(frame_idx: int, fps: float) -> int:
    return int((frame_idx / max(fps, 1e-6)) * 1000)


def topk_from_probs(class_names: list[str], probs: list[float] | np.ndarray, k: int = 3) -> list[dict[str, Any]]:
    arr = np.asarray(probs, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D probs, got {arr.shape}")
    order = np.argsort(-arr)[:k]
    items = [
        TopKItem(
            rank=rank + 1,
            class_index=int(idx),
            class_label=str(class_names[int(idx)]),
            probability=float(arr[int(idx)]),
        )
        for rank, idx in enumerate(order)
    ]
    return [asdict(item) for item in items]


def summarize_array(name: str, array: np.ndarray | None) -> dict[str, Any] | None:
    if array is None:
        return None
    arr = np.asarray(array, dtype=np.float32)
    flat = arr.reshape(-1)
    head = flat[: min(8, flat.size)].astype(np.float64).tolist()
    return {
        "name": name,
        "shape": list(arr.shape),
        "min": float(np.min(flat)) if flat.size else 0.0,
        "max": float(np.max(flat)) if flat.size else 0.0,
        "mean": float(np.mean(flat)) if flat.size else 0.0,
        "std": float(np.std(flat)) if flat.size else 0.0,
        "head": head,
    }


def pos_scale_frame(raw_landmarks: np.ndarray) -> np.ndarray:
    pts = np.asarray(raw_landmarks, dtype=np.float32).reshape(21, 3)
    origin = pts[0].copy()
    denom = float(np.linalg.norm(pts[9] - origin))
    if denom <= EPS:
        return (pts - origin).astype(np.float32)
    return ((pts - origin) / denom).astype(np.float32)


def build_delta_feature_sequence(joint_sequence: np.ndarray) -> np.ndarray:
    seq = np.asarray(joint_sequence, dtype=np.float32)
    if seq.ndim != 2 or seq.shape[1] != 63:
        raise ValueError(f"Expected joint sequence shape (T, 63), got {seq.shape}")
    delta = np.zeros_like(seq, dtype=np.float32)
    delta[1:] = seq[1:] - seq[:-1]
    return np.concatenate([seq, delta], axis=1).astype(np.float32)


def neutral_probs(num_classes: int, neutral_index: int) -> list[float]:
    probs = [0.0] * num_classes
    probs[neutral_index] = 1.0
    return probs


def normalize_handedness_labels(result: Any) -> list[str | None]:
    raw = getattr(result, "handedness", None) or []
    labels: list[str | None] = []
    for entry in raw:
        first = entry[0] if isinstance(entry, list) and entry else entry
        label = str(
            getattr(first, "display_name", None)
            or getattr(first, "category_name", None)
            or ""
        ).strip().lower()
        labels.append(label if label in {"left", "right"} else None)
    return labels


def build_frontend_hand_keys(result: Any) -> list[str]:
    labels = normalize_handedness_labels(result)
    hand_landmarks = getattr(result, "hand_landmarks", None) or []
    used_keys: set[str] = set()
    keys: list[str] = []
    for index, _ in enumerate(hand_landmarks):
        preferred = labels[index] if index < len(labels) else None
        hand_key = preferred
        if not hand_key or hand_key in used_keys:
            for candidate in ("left", "right"):
                if candidate not in used_keys:
                    hand_key = candidate
                    break
        if not hand_key:
            hand_key = f"hand-{index}"
        used_keys.add(hand_key)
        keys.append(hand_key)
    return keys


def select_frontend_hand_index(
    result: Any,
    preferred_hand: str = "right",
) -> tuple[int | None, dict[str, Any]]:
    hand_landmarks = getattr(result, "hand_landmarks", None) or []
    labels = normalize_handedness_labels(result)
    keys = build_frontend_hand_keys(result)

    if not hand_landmarks:
        return None, {
            "preferred_hand": preferred_hand,
            "selected_index": None,
            "selected_hand_key": None,
            "selected_handedness": None,
            "fallback_selected": False,
            "detected_handedness": labels,
            "detected_hand_keys": keys,
        }

    selected_index = None
    for index, key in enumerate(keys):
        if key == preferred_hand:
            selected_index = index
            break

    fallback_selected = False
    if selected_index is None:
        selected_index = 0
        fallback_selected = True

    selected_handedness = labels[selected_index] if selected_index < len(labels) else None
    selected_hand_key = keys[selected_index] if selected_index < len(keys) else None
    return selected_index, {
        "preferred_hand": preferred_hand,
        "selected_index": int(selected_index),
        "selected_hand_key": selected_hand_key,
        "selected_handedness": selected_handedness,
        "fallback_selected": fallback_selected,
        "detected_handedness": labels,
        "detected_hand_keys": keys,
    }
