# gesture_resolution.py - frontend model gesture 후처리(stabilize) 재현
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


DEFAULT_CONFIDENCE_ENTER = 0.58
DEFAULT_CONFIDENCE_HOLD = 0.46
DEFAULT_STABLE_FRAMES = 1
DEFAULT_CLEAR_FRAMES = 1

CLASS_SPECIFIC_ENTER = {
    "Pinky": 0.36,
    "Animal": 0.34,
    "KHeart": 0.36,
}

CLASS_SPECIFIC_HOLD = {
    "Pinky": 0.24,
    "Animal": 0.22,
    "KHeart": 0.24,
}

CLASS_SPECIFIC_STABLE_FRAMES = {
    "Pinky": 1,
    "Animal": 1,
    "KHeart": 1,
}


def normalize_model_label(raw_label: str | None) -> str:
    label = str(raw_label or "").strip().lower()
    if label in {"none", "class0"}:
        return "None"
    if label in {"fist", "isfist", "class1"}:
        return "Fist"
    if label in {"open palm", "open_palm", "openpalm", "paper", "ispaper", "class2"}:
        return "OpenPalm"
    if label in {"v", "isv", "class3"}:
        return "V"
    if label in {"pinky", "ispinky", "pinky class", "pinky_class", "class4", "class 4", "4"}:
        return "Pinky"
    if label in {"animal", "isanimal", "class5", "class 5", "5"}:
        return "Animal"
    if label in {"k-heart", "kheart", "is_k_heart", "class6", "class 6", "6"}:
        return "KHeart"
    return "None"


@dataclass(slots=True)
class StableState:
    candidate_label: str = "None"
    candidate_frames: int = 0
    none_frames: int = 0
    stable_label: str = "None"
    confidence: float = 0.0
    source: str = "model"


class ModelGestureResolver:
    def __init__(self) -> None:
        self.state = StableState()

    def reset(self) -> None:
        self.state = StableState()

    def _map_model_to_result(self, model_prediction: dict[str, Any] | None) -> dict[str, Any]:
        if not model_prediction:
            return {"label": "None", "confidence": 0.0, "source": "model"}

        normalized = normalize_model_label(model_prediction.get("label"))
        class_id = model_prediction.get("classId")
        class_label = normalize_model_label(f"class{class_id}") if isinstance(class_id, int) else "None"
        label = normalized if normalized != "None" else class_label
        confidence = float(model_prediction.get("confidence") or 0.0)
        enter_threshold = CLASS_SPECIFIC_ENTER.get(label, DEFAULT_CONFIDENCE_ENTER)
        hold_base_threshold = CLASS_SPECIFIC_HOLD.get(label, DEFAULT_CONFIDENCE_HOLD)
        hold_threshold = hold_base_threshold if self.state.stable_label == label else enter_threshold

        if label == "None" or confidence < hold_threshold:
            return {"label": "None", "confidence": confidence, "source": "model"}
        return {"label": label, "confidence": confidence, "source": "model"}

    def _stabilize(self, raw_result: dict[str, Any]) -> dict[str, Any]:
        state = self.state
        if raw_result["label"] == "None":
            state.none_frames += 1
            state.candidate_label = "None"
            state.candidate_frames = 0
            if state.none_frames >= DEFAULT_CLEAR_FRAMES:
                state.stable_label = "None"
                state.confidence = 0.0
                state.source = raw_result["source"]
        else:
            state.none_frames = 0
            if state.candidate_label == raw_result["label"]:
                state.candidate_frames += 1
            else:
                state.candidate_label = raw_result["label"]
                state.candidate_frames = 1

            required_frames = CLASS_SPECIFIC_STABLE_FRAMES.get(raw_result["label"], DEFAULT_STABLE_FRAMES)
            if state.candidate_frames >= required_frames:
                state.stable_label = raw_result["label"]
                state.confidence = float(raw_result["confidence"])
                state.source = raw_result["source"]

        return {
            "label": state.stable_label,
            "confidence": float(state.confidence),
            "source": state.source,
            "isV": state.stable_label == "V",
            "isPaper": state.stable_label == "OpenPalm",
        }

    def push(self, model_prediction: dict[str, Any] | None) -> dict[str, Any]:
        return self._stabilize(self._map_model_to_result(model_prediction))
