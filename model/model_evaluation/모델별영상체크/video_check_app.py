# video_check_app.py - Trained-model video viewer with dropdown UI for run/video selection.
from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import tkinter as tk
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any

os.environ.setdefault("QT_QPA_PLATFORM_PLUGIN_PATH", "")
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import cv2
import mediapipe as mp
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[3]
RUNS_ROOT = PROJECT_ROOT / "model" / "model_evaluation" / "pipelines"
RAW_VIDEO_ROOT = PROJECT_ROOT / "data" / "raw_data"
TASK_MODEL_PATH = PROJECT_ROOT / "hand_landmarker.task"
MODEL_PIPELINES_ROOT = PROJECT_ROOT / "model" / "model_pipelines"

if str(MODEL_PIPELINES_ROOT) not in sys.path:
    # 저장된 model_id 문자열로 각 분류기 구현을 동적 import 하기 위한 경로다.
    sys.path.insert(0, str(MODEL_PIPELINES_ROOT))

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

KEY_LEFT = 65361
KEY_RIGHT = 65363
DEFAULT_SEQ_LEN = 16
DEFAULT_IMAGE_SIZE = 96
FORCED_DEVICE = torch.device("cpu")

HAND_CONNECTIONS = [
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
    (1, 2), (2, 3), (3, 4),
    (5, 6), (6, 7), (7, 8),
    (9, 10), (10, 11), (11, 12),
    (13, 14), (14, 15), (15, 16),
    (17, 18), (18, 19), (19, 20),
]

FINGER_CHAINS = {
    "thumb": [0, 1, 2, 3, 4],
    "index": [0, 5, 6, 7, 8],
    "middle": [0, 9, 10, 11, 12],
    "ring": [0, 13, 14, 15, 16],
    "pinky": [0, 17, 18, 19, 20],
}

FINGER_PAIRS = [
    ("thumb", "index"),
    ("index", "middle"),
    ("middle", "ring"),
    ("ring", "pinky"),
]

SUPPORTED_VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm")


@dataclass(slots=True)
class RunInfo:
    """UI와 CLI가 공통으로 쓰는 '학습 실행 1회분' 메타데이터."""

    model_id: str
    run_dir: Path
    checkpoint_path: Path
    summary_path: Path | None
    mode: str
    macro_f1: float | None
    display_name: str


@dataclass(slots=True)
class FeaturePack:
    """런타임 추론에서 재사용할 feature 묶음.

    하나의 raw landmark 세트에서 frame / two-stream / sequence / image 입력을 모두 만든다.
    """

    raw_landmarks: np.ndarray
    normalized_landmarks: np.ndarray
    joint: np.ndarray
    joint_xy: np.ndarray
    joint_z: np.ndarray
    bone: np.ndarray
    angle: np.ndarray
    full: np.ndarray
    bone_angle: np.ndarray


@dataclass(slots=True)
class FrameResult:
    """분석된 비디오의 단일 프레임 추론 결과."""

    frame_idx: int
    timestamp_text: str
    status: str
    pred_idx: int
    pred_name: str
    confidence: float
    probs: list[float]
    raw_landmarks: np.ndarray | None


@dataclass(slots=True)
class AnalyzedVideo:
    """전체 비디오를 한 번 분석한 후 playback에 넘길 캐시 구조."""

    video_path: Path
    fps: float
    total_frames: int
    frame_results: list[FrameResult]


@dataclass(slots=True)
class RuntimeModel:
    """checkpoint를 실제 추론 가능한 model 객체로 복원한 상태."""

    run_info: RunInfo
    model: torch.nn.Module
    device: torch.device
    model_id: str
    mode: str
    class_names: list[str]
    neutral_idx: int
    seq_len: int
    image_size: int
    input_dim: int | None
    aux_input_dim: int | None


def format_timestamp(frame_idx: int, fps: float) -> str:
    """frame index를 MM:SS:ms 문자열로 변환해 오버레이에 표시한다."""
    total_ms = int((frame_idx / max(fps, 1e-6)) * 1000)
    minutes = total_ms // 60000
    seconds = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{minutes:02d}:{seconds:02d}:{ms:03d}"


def discover_runs() -> list[RunInfo]:
    """결과 폴더에서 checkpoint run들을 스캔해 dropdown 후보를 만든다."""
    runs: list[RunInfo] = []

    for checkpoint_path in sorted(RUNS_ROOT.rglob("model.pt"), reverse=True):
        run_dir = checkpoint_path.parent
        summary_path = run_dir / "run_summary.json"
        mode = "unknown"
        macro_f1: float | None = None
        suite_name: str | None = None
        model_id = run_dir.parent.name

        if summary_path.exists():
            try:
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
                model_id = str(summary.get("model_id") or model_id)
                mode = str(summary.get("mode", "unknown"))
                macro_f1 = summary.get("metrics", {}).get("macro_avg", {}).get("f1")
            except Exception:
                pass

        try:
            rel = run_dir.relative_to(RUNS_ROOT)
            if len(rel.parts) >= 3:
                suite_name = rel.parts[0]
        except ValueError:
            suite_name = None

        stamp = run_dir.name
        f1_text = f"{macro_f1:.4f}" if isinstance(macro_f1, (int, float)) else "-"
        suite_text = f"{suite_name} | " if suite_name else ""
        display_name = f"{model_id} | {suite_text}{stamp} | mode={mode} | macro_f1={f1_text}"

        runs.append(
            RunInfo(
                model_id=model_id,
                run_dir=run_dir,
                checkpoint_path=checkpoint_path,
                summary_path=summary_path if summary_path.exists() else None,
                mode=mode,
                macro_f1=float(macro_f1) if isinstance(macro_f1, (int, float)) else None,
                display_name=display_name,
            )
        )

    return runs


def discover_videos() -> list[Path]:
    """raw_data 아래의 재생 가능한 비디오 후보를 모은다."""
    videos: list[Path] = []
    for ext in SUPPORTED_VIDEO_EXTS:
        videos.extend(sorted(RAW_VIDEO_ROOT.glob(f"*{ext}")))
    return sorted({p.resolve() for p in videos})


def safe_torch_load(path: Path, device: torch.device) -> dict[str, Any]:
    """PyTorch 버전 차이를 흡수하며 checkpoint dict를 로드한다."""
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)

    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unexpected checkpoint format: {path}")
    return checkpoint


def detect_neutral_idx(class_names: list[str]) -> int:
    """neutral/background 클래스 index를 휴리스틱으로 찾는다."""
    for idx, name in enumerate(class_names):
        if str(name).strip().lower() in {"neutral", "none", "background"}:
            return idx
    return 0


def _infer_num_classes(state_dict: dict[str, Any]) -> int:
    """head weight shape로부터 class 수를 역추론한다."""
    for key in ("head.2.weight", "head.6.weight", "head.weight", "net.6.weight", "net.4.weight"):
        tensor = state_dict.get(key)
        if tensor is not None and hasattr(tensor, "shape") and len(tensor.shape) == 2:
            return int(tensor.shape[0])
    raise ValueError("Could not infer num_classes from checkpoint state_dict.")


def instantiate_model(
    model_id: str,
    state_dict: dict[str, Any],
    num_classes: int,
    seq_len_hint: int,
) -> tuple[torch.nn.Module, int, int | None, int | None]:
    """checkpoint state_dict shape를 읽어 정확한 model 클래스를 복원한다."""
    mod = importlib.import_module(f"{model_id}.model")

    if model_id in {"mlp_baseline", "mlp_baseline_full"}:
        input_dim = int(state_dict["net.0.weight"].shape[1])
        model = mod.MLPBaseline(input_dim=input_dim, num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, input_dim, None

    if model_id == "mlp_baseline_seq8":
        flat_dim = int(state_dict["net.0.weight"].shape[1])
        seq_len = 8
        input_dim = flat_dim // seq_len
        model = mod.MLPBaselineSeq(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
        return model, seq_len, input_dim, None

    if model_id == "mlp_sequence_joint":
        input_dim = int(state_dict["net.1.weight"].shape[1])
        seq_len = seq_len_hint or max(input_dim // 63, 1)
        feature_dim = input_dim // max(seq_len, 1)
        model = mod.SequenceJointMLP(seq_len=seq_len, input_dim=feature_dim, num_classes=num_classes)
        return model, seq_len, feature_dim, None

    if model_id == "mlp_temporal_pooling":
        input_dim = int(state_dict["frame_embed.1.weight"].shape[1])
        model = mod.TemporalPoolingMLP(input_dim=input_dim, num_classes=num_classes)
        return model, seq_len_hint or DEFAULT_SEQ_LEN, input_dim, None

    if model_id == "mlp_sequence_delta":
        flat_dim = int(state_dict["net.1.weight"].shape[1])
        seq_len = seq_len_hint or max(flat_dim // 126, 1)
        input_dim = flat_dim // seq_len
        model = mod.SequenceDeltaMLP(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
        return model, seq_len, input_dim, None

    if model_id == "mlp_embedding":
        input_dim = int(state_dict["embed.0.weight"].shape[1])
        model = mod.MLPEmbedding(input_dim=input_dim, num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, input_dim, None

    if model_id == "two_stream_mlp":
        joint_dim = int(state_dict["joint_stream.net.0.weight"].shape[1])
        bone_dim = int(state_dict["bone_stream.net.0.weight"].shape[1])
        model = mod.TwoStreamMLP(joint_dim=joint_dim, bone_dim=bone_dim, num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, joint_dim, bone_dim

    if model_id == "cnn1d_tcn":
        input_dim = int(state_dict["net.0.weight"].shape[1])
        model = mod.TCN1DClassifier(input_dim=input_dim, num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, input_dim, None

    if model_id == "transformer_embedding":
        input_dim = int(state_dict["frame_embed.weight"].shape[1])
        seq_len = int(state_dict["pos_embed"].shape[1])
        model = mod.TemporalTransformer(seq_len=seq_len, input_dim=input_dim, num_classes=num_classes)
        return model, seq_len, input_dim, None

    if model_id == "mobilenetv3_small":
        model = mod.MobileNetLike(num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, None, None

    if model_id == "shufflenetv2_x0_5":
        model = mod.ShuffleNetLike(num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, None, None

    if model_id == "efficientnet_b0":
        model = mod.EfficientNetLike(num_classes=num_classes)
        return model, DEFAULT_SEQ_LEN, None, None

    raise ValueError(f"Unsupported model_id: {model_id}")


def load_runtime_model(run_info: RunInfo) -> RuntimeModel:
    """run 폴더를 읽어 UI/CLI 공용 RuntimeModel로 변환한다."""
    device = FORCED_DEVICE
    checkpoint = safe_torch_load(run_info.checkpoint_path, device)
    state_dict = checkpoint["model_state_dict"]
    class_names = list(checkpoint.get("class_names") or [])
    if not class_names:
        class_names = [str(i) for i in range(_infer_num_classes(state_dict))]

    num_classes = len(class_names)
    model_id = str(checkpoint.get("model_id") or run_info.model_id)
    mode = str(checkpoint.get("mode") or run_info.mode)
    seq_len_hint = int(checkpoint.get("seq_len") or DEFAULT_SEQ_LEN)
    image_size = int(checkpoint.get("image_size") or DEFAULT_IMAGE_SIZE)
    model, seq_len, input_dim, aux_input_dim = instantiate_model(
        model_id,
        state_dict,
        num_classes,
        seq_len_hint=seq_len_hint,
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return RuntimeModel(
        run_info=run_info,
        model=model,
        device=device,
        model_id=model_id,
        mode=mode,
        class_names=class_names,
        neutral_idx=detect_neutral_idx(class_names),
        seq_len=seq_len,
        image_size=image_size,
        input_dim=input_dim,
        aux_input_dim=aux_input_dim,
    )


def create_landmarker() -> HandLandmarker:
    """비디오 추론에 사용할 MediaPipe Hand Landmarker 인스턴스를 만든다."""
    if not TASK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing task model: {TASK_MODEL_PATH}")

    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(TASK_MODEL_PATH)),
        running_mode=VisionRunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    return HandLandmarker.create_from_options(options)


def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
    """학습 전처리와 같은 규칙으로 raw landmark를 정규화한다."""
    pts = landmarks.astype(np.float32).copy()

    # 중심 이동: 손 위치 차이를 줄인다.
    knuckles = pts[[5, 9, 17]]
    center = knuckles.mean(axis=0)
    pts -= center

    # 스케일 정규화: 손 크기 / 카메라 거리 차이를 줄인다.
    knuckle_indices = [1, 5, 9, 13, 17]
    knuckle_pts = pts[knuckle_indices]
    max_dist = np.linalg.norm(knuckle_pts, axis=1).max()
    if max_dist > 1e-8:
        pts /= max_dist

    # xy 평면 회전 정렬: 손 회전 편차를 줄여 모델 입력을 학습 시점과 맞춘다.
    vec1 = pts[0] - pts[9]
    vec2 = pts[17] - pts[5]
    alignment = vec1 + vec2
    alignment_norm = np.linalg.norm(alignment[:2])
    if alignment_norm > 1e-8:
        cos_theta = alignment[1] / alignment_norm
        sin_theta = alignment[0] / alignment_norm
        rot_matrix = np.array(
            [
                [cos_theta, sin_theta, 0.0],
                [-sin_theta, cos_theta, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        pts = pts @ rot_matrix.T

    return pts.astype(np.float32)


def compute_bone_features(landmarks: np.ndarray) -> np.ndarray:
    """정규화 landmark에서 bone vector + length 84차원을 만든다."""
    bone_features: list[np.ndarray] = []
    for u, v in HAND_CONNECTIONS:
        diff = landmarks[v] - landmarks[u]
        length = np.linalg.norm(diff)
        bone_features.append(np.append(diff, length))
    return np.array(bone_features, dtype=np.float32)


def compute_angle_features(landmarks: np.ndarray) -> np.ndarray:
    """손가락 굽힘 / 벌어짐 각도를 계산해 9차원 angle feature를 만든다."""
    angles: list[float] = []

    for chain in FINGER_CHAINS.values():
        pts = landmarks[chain]
        segments = np.diff(pts, axis=0)
        if len(segments) < 2:
            angles.append(0.0)
            continue

        first_seg = segments[0]
        max_angle = 0.0
        for seg in segments[1:]:
            denom = np.linalg.norm(first_seg) * np.linalg.norm(seg) + 1e-8
            cos_val = float(np.dot(first_seg, seg) / denom)
            cos_val = float(np.clip(cos_val, -1.0, 1.0))
            max_angle = max(max_angle, float(np.arccos(cos_val)))
        angles.append(max_angle)

    for name_a, name_b in FINGER_PAIRS:
        chain_a = FINGER_CHAINS[name_a]
        chain_b = FINGER_CHAINS[name_b]
        dir_a = landmarks[chain_a[-1]] - landmarks[chain_a[1]]
        dir_b = landmarks[chain_b[-1]] - landmarks[chain_b[1]]
        denom = np.linalg.norm(dir_a) * np.linalg.norm(dir_b) + 1e-8
        cos_val = float(np.dot(dir_a, dir_b) / denom)
        cos_val = float(np.clip(cos_val, -1.0, 1.0))
        angles.append(float(np.arccos(cos_val)))

    return np.array(angles, dtype=np.float32)


def extract_feature_pack(raw_landmarks: np.ndarray) -> FeaturePack:
    """raw landmark 1프레임에서 모든 런타임 입력 표현을 동시에 생성한다."""
    normalized = normalize_landmarks(raw_landmarks)
    joint = normalized.reshape(-1).astype(np.float32)
    joint_xy = normalized[:, :2].reshape(-1).astype(np.float32)
    joint_z = normalized[:, 2].reshape(-1).astype(np.float32)
    bone = compute_bone_features(normalized).reshape(-1).astype(np.float32)
    angle = compute_angle_features(normalized).astype(np.float32)
    bone_angle = np.concatenate([bone, angle]).astype(np.float32)
    full = np.concatenate([joint, bone, angle]).astype(np.float32)

    return FeaturePack(
        raw_landmarks=raw_landmarks.astype(np.float32),
        normalized_landmarks=normalized,
        joint=joint,
        joint_xy=joint_xy,
        joint_z=joint_z,
        bone=bone,
        angle=angle,
        full=full,
        bone_angle=bone_angle,
    )


def render_skeleton_image(normalized_landmarks: np.ndarray, size: int) -> np.ndarray:
    """image 모델용 1채널 skeleton image를 메모리 상에서 렌더링한다."""
    canvas = np.zeros((size, size), dtype=np.float32)
    pts = normalized_landmarks.reshape(21, 3)
    clipped = np.clip(pts[:, :2], -1.2, 1.2)
    px = ((clipped[:, 0] + 1.2) / 2.4) * (size - 1)
    py = (size - 1) - (((clipped[:, 1] + 1.2) / 2.4) * (size - 1))
    pix_pts = np.stack([px, py], axis=1).astype(np.int32)

    for u, v in HAND_CONNECTIONS:
        p1 = tuple(int(x) for x in pix_pts[u])
        p2 = tuple(int(x) for x in pix_pts[v])
        cv2.line(canvas, p1, p2, color=160 / 255.0, thickness=2, lineType=cv2.LINE_AA)

    for p in pix_pts:
        cv2.circle(canvas, tuple(int(x) for x in p), 2, color=1.0, thickness=-1, lineType=cv2.LINE_AA)

    return canvas[None, :, :].astype(np.float32)


def draw_raw_landmarks(frame: np.ndarray, raw_landmarks: np.ndarray | None) -> np.ndarray:
    """원본 프레임 위에 감지 landmark를 시각화해 재생 화면에 덧그린다."""
    if raw_landmarks is None:
        return frame

    h, w = frame.shape[:2]
    pts: list[tuple[int, int] | None] = []
    for x, y, _ in raw_landmarks:
        if np.isnan(x) or np.isnan(y):
            pts.append(None)
            continue
        pts.append((int(float(x) * w), int(float(y) * h)))

    for u, v in HAND_CONNECTIONS:
        p1 = pts[u]
        p2 = pts[v]
        if p1 is not None and p2 is not None:
            cv2.line(frame, p1, p2, (255, 0, 0), 2, cv2.LINE_AA)

    for p in pts:
        if p is not None:
            cv2.circle(frame, p, 3, (0, 255, 0), -1, cv2.LINE_AA)

    return frame


def neutral_probs(class_count: int, neutral_idx: int) -> list[float]:
    """warmup / no-hand 구간에서 사용할 one-hot neutral 확률 벡터."""
    probs = [0.0] * class_count
    probs[neutral_idx] = 1.0
    return probs


def select_feature_vector(features: FeaturePack, expected_dim: int) -> np.ndarray:
    """checkpoint의 실제 입력 차원에 맞는 feature 표현을 선택한다."""
    candidates = {
        int(features.joint.shape[0]): features.joint,
        int(features.joint_xy.shape[0]): features.joint_xy,
        int(features.joint_z.shape[0]): features.joint_z,
        int(features.bone.shape[0]): features.bone,
        int(features.angle.shape[0]): features.angle,
        int(features.bone_angle.shape[0]): features.bone_angle,
        int(features.full.shape[0]): features.full,
    }
    vector = candidates.get(int(expected_dim))
    if vector is None:
        supported = ", ".join(str(dim) for dim in sorted(candidates))
        raise ValueError(
            f"Unsupported runtime feature dim: {expected_dim}. "
            f"Supported dims from feature pack: {supported}"
        )
    return vector


def add_runtime_delta_features(seq: np.ndarray) -> np.ndarray:
    """
    seq: (T, D)
    returns: (T, 2D) = [base_feature, delta]
    """
    # 학습 때와 동일하게 runtime에서도 delta 채널을 즉석에서 합성한다.
    delta = np.zeros_like(seq, dtype=np.float32)
    delta[1:, :] = seq[1:, :] - seq[:-1, :]
    return np.concatenate([seq.astype(np.float32), delta], axis=1)


def topk_text(class_names: list[str], probs: list[float], k: int = 3) -> str:
    """오버레이용 top-k 확률 텍스트를 만든다."""
    pairs = sorted(enumerate(probs), key=lambda item: item[1], reverse=True)[:k]
    return " | ".join(f"{class_names[idx]} {score:.2f}" for idx, score in pairs)


@torch.inference_mode()
def predict_from_features(
    runtime: RuntimeModel,
    features: FeaturePack,
    seq_buffer: deque[np.ndarray],
) -> tuple[str, int, float, list[float]]:
    """현재 모델 mode에 맞춰 feature를 분기해 단일 프레임 예측을 수행한다."""
    class_count = len(runtime.class_names)

    if runtime.mode == "frame":
        if runtime.input_dim is None:
            raise ValueError(f"Missing runtime input_dim for frame model: {runtime.model_id}")
        vector = select_feature_vector(features, runtime.input_dim)
        tensor = torch.from_numpy(vector).unsqueeze(0).to(runtime.device)
        logits = runtime.model(tensor)

    elif runtime.mode == "two_stream":
        if runtime.input_dim is None or runtime.aux_input_dim is None:
            raise ValueError(f"Missing runtime stream dims for two_stream model: {runtime.model_id}")
        joint_vec = select_feature_vector(features, runtime.input_dim)
        aux_vec = select_feature_vector(features, runtime.aux_input_dim)
        joint = torch.from_numpy(joint_vec).unsqueeze(0).to(runtime.device)
        aux = torch.from_numpy(aux_vec).unsqueeze(0).to(runtime.device)
        logits = runtime.model(joint, aux)

    elif runtime.mode == "sequence":
        if runtime.input_dim is None:
            raise ValueError(f"Missing runtime input_dim for sequence model: {runtime.model_id}")

        # sequence 계열은 프레임마다 buffer를 채우고 seq_len이 찰 때부터 예측 가능하다.
        if runtime.model_id == "mlp_sequence_delta":
            if runtime.input_dim % 2 != 0:
                raise ValueError(f"Unexpected delta input_dim for {runtime.model_id}: {runtime.input_dim}")
            base_vec = select_feature_vector(features, runtime.input_dim // 2)
        else:
            base_vec = select_feature_vector(features, runtime.input_dim)

        seq_buffer.append(base_vec)
        if len(seq_buffer) < runtime.seq_len:
            return "warmup", runtime.neutral_idx, 0.0, neutral_probs(class_count, runtime.neutral_idx)

        seq = np.stack(list(seq_buffer), axis=0).astype(np.float32)
        if runtime.model_id == "mlp_sequence_delta":
            seq = add_runtime_delta_features(seq)
            if seq.shape[1] != runtime.input_dim:
                raise ValueError(
                    f"Delta feature mismatch for {runtime.model_id}: "
                    f"got {seq.shape[1]}, expected {runtime.input_dim}"
                )
        tensor = torch.from_numpy(seq).unsqueeze(0).to(runtime.device)
        logits = runtime.model(tensor)

    elif runtime.mode == "image":
        # image 계열은 normalized landmark를 즉석 skeleton map으로 그린다.
        image = render_skeleton_image(features.normalized_landmarks, runtime.image_size)
        tensor = torch.from_numpy(image).unsqueeze(0).to(runtime.device)
        logits = runtime.model(tensor)

    else:
        raise ValueError(f"Unsupported mode: {runtime.mode}")

    probs = torch.softmax(logits, dim=1)[0].detach().cpu().numpy().astype(float).tolist()
    pred_idx = int(np.argmax(probs))
    return "ready", pred_idx, float(probs[pred_idx]), probs


def analyze_video(runtime: RuntimeModel, video_path: Path) -> AnalyzedVideo:
    """비디오 전 프레임을 분석해 playback용 FrameResult 목록을 만든다."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    results: list[FrameResult] = []
    seq_buffer: deque[np.ndarray] = deque(maxlen=runtime.seq_len)

    print(f"[viewer] analyzing {video_path.name} with {runtime.model_id}")
    print(f"[viewer] total_frames={total_frames}, fps={fps:.2f}, device={runtime.device}")

    with create_landmarker() as landmarker:
        for frame_idx in range(total_frames):
            ok, frame = cap.read()
            if not ok:
                break

            timestamp_ms = int((frame_idx / max(fps, 1e-6)) * 1000)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            mp_result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if not mp_result.hand_landmarks:
                # 손이 사라지면 sequence 상태를 초기화해 다음 gesture 추론이 섞이지 않게 한다.
                seq_buffer.clear()
                pred_idx = runtime.neutral_idx
                probs = neutral_probs(len(runtime.class_names), runtime.neutral_idx)
                results.append(
                    FrameResult(
                        frame_idx=frame_idx,
                        timestamp_text=format_timestamp(frame_idx, fps),
                        status="no_hand",
                        pred_idx=pred_idx,
                        pred_name=runtime.class_names[pred_idx],
                        confidence=0.0,
                        probs=probs,
                        raw_landmarks=None,
                    )
                )
            else:
                raw_landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in mp_result.hand_landmarks[0]],
                    dtype=np.float32,
                )
                features = extract_feature_pack(raw_landmarks)
                status, pred_idx, confidence, probs = predict_from_features(runtime, features, seq_buffer)
                results.append(
                    FrameResult(
                        frame_idx=frame_idx,
                        timestamp_text=format_timestamp(frame_idx, fps),
                        status=status,
                        pred_idx=pred_idx,
                        pred_name=runtime.class_names[pred_idx],
                        confidence=confidence,
                        probs=probs,
                        raw_landmarks=raw_landmarks,
                    )
                )

            if (frame_idx + 1) % 30 == 0 or frame_idx + 1 == total_frames:
                pct = ((frame_idx + 1) / max(total_frames, 1)) * 100.0
                print(f"[viewer] progress {frame_idx + 1}/{total_frames} ({pct:.1f}%)")

    cap.release()
    return AnalyzedVideo(video_path=video_path, fps=fps, total_frames=len(results), frame_results=results)


def overlay_frame(frame: np.ndarray, runtime: RuntimeModel, analyzed: AnalyzedVideo, current_idx: int) -> np.ndarray:
    """현재 프레임의 예측 결과를 사람이 읽기 쉬운 오버레이로 렌더링한다."""
    record = analyzed.frame_results[current_idx]
    display = draw_raw_landmarks(frame.copy(), record.raw_landmarks)
    h, w = display.shape[:2]

    title = f"Run: {runtime.run_info.model_id}/{runtime.run_info.run_dir.name}"
    video_text = f"Video: {analyzed.video_path.name}"
    frame_text = f"Frame: {record.frame_idx}/{max(analyzed.total_frames - 1, 0)}  Time: {record.timestamp_text}"
    mode_text = f"Mode: {runtime.mode}  Device: {runtime.device}"
    pred_text = f"Pred: {record.pred_name}  Conf: {record.confidence:.3f}  Status: {record.status}"
    top3 = f"Top3: {topk_text(runtime.class_names, record.probs)}"
    guide = "Space: pause/resume | A/Left: prev | D/Right: next | R: restart | Q/Esc: quit"

    y = 30
    for text, color in (
        (title, (255, 255, 255)),
        (video_text, (220, 220, 220)),
        (frame_text, (220, 220, 220)),
        (mode_text, (180, 220, 255)),
        (pred_text, (0, 255, 255) if record.status == "ready" else (0, 165, 255)),
        (top3, (180, 255, 180)),
    ):
        cv2.putText(display, text, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
        y += 30

    cv2.putText(display, guide, (12, h - 18), cv2.FONT_HERSHEY_PLAIN, 1.2, (200, 200, 200), 1, cv2.LINE_AA)
    return display


def playback(runtime: RuntimeModel, analyzed: AnalyzedVideo) -> None:
    """분석 결과를 키보드 컨트롤이 가능한 OpenCV 창으로 재생한다."""
    cap = cv2.VideoCapture(str(analyzed.video_path))
    if not cap.isOpened():
        raise IOError(f"Could not reopen video: {analyzed.video_path}")

    window_name = f"JamJamBeat Viewer - {runtime.run_info.model_id}"
    delay_ms = max(int(1000 / max(analyzed.fps, 1e-6)), 1)
    paused = False
    current_idx = 0
    window_created = False

    while 0 <= current_idx < analyzed.total_frames:
        # 창이 생성된 이후에만 닫힘 여부를 확인한다 (생성 전 호출 시 OpenCV 에러 발생).
        if window_created and cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

        cap.set(cv2.CAP_PROP_POS_FRAMES, current_idx)
        ok, frame = cap.read()
        if not ok:
            break

        display = overlay_frame(frame, runtime, analyzed, current_idx)
        cv2.imshow(window_name, display)
        window_created = True

        wait_ms = 0 if paused else delay_ms
        key = cv2.waitKeyEx(wait_ms)

        if key in (-1, 255):
            if not paused:
                current_idx += 1
            continue

        key_low = key & 0xFF
        # 키 입력은 재생/정지/단일 프레임 탐색만 처리한다.
        if key_low in (ord("q"), 27):
            break
        if key_low == ord(" "):
            paused = not paused
            continue
        if key_low == ord("r"):
            current_idx = 0
            paused = True
            continue
        if key_low == ord("a") or key == KEY_LEFT:
            current_idx = max(0, current_idx - 1)
            paused = True
            continue
        if key_low == ord("d") or key == KEY_RIGHT:
            current_idx = min(analyzed.total_frames - 1, current_idx + 1)
            paused = True
            continue

    cap.release()
    cv2.destroyWindow(window_name)
    cv2.waitKey(1)  # WSL2/Linux에서 destroy 이벤트를 즉시 flush한다.


class VideoCheckApp:
    """학습 run 선택 -> 비디오 선택 -> 분석/재생을 묶는 간단한 Tk UI."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.runs: list[RunInfo] = []
        self.videos: list[Path] = []
        self.run_lookup: dict[str, RunInfo] = {}
        self.video_lookup: dict[str, Path] = {}
        self.runtime_cache: dict[str, RuntimeModel] = {}
        self.analysis_cache: dict[tuple[str, str], AnalyzedVideo] = {}

        self.root.title("JamJamBeat Video Check")
        self.root.geometry("980x240")

        self.status_var = tk.StringVar(value="Ready")
        self.run_var = tk.StringVar()
        self.video_var = tk.StringVar()
        self.info_var = tk.StringVar(value="Select a trained run and a video.")

        self._build_ui()
        self.refresh_options()

    def _build_ui(self) -> None:
        # 위젯 배치는 단순하므로 구조만 드러내고 세부 UI 해설은 생략한다.
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(frame, text="Trained Run").grid(row=0, column=0, sticky="w")
        self.run_combo = ttk.Combobox(frame, textvariable=self.run_var, state="readonly", width=100)
        self.run_combo.grid(row=1, column=0, sticky="ew", pady=(4, 12))
        self.run_combo.bind("<<ComboboxSelected>>", lambda _: self.update_info())

        ttk.Label(frame, text="Video").grid(row=2, column=0, sticky="w")
        self.video_combo = ttk.Combobox(frame, textvariable=self.video_var, state="readonly", width=100)
        self.video_combo.grid(row=3, column=0, sticky="ew", pady=(4, 12))
        self.video_combo.bind("<<ComboboxSelected>>", lambda _: self.update_info())

        button_row = ttk.Frame(frame)
        button_row.grid(row=4, column=0, sticky="w", pady=(0, 12))
        ttk.Button(button_row, text="Refresh", command=self.refresh_options).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Analyze And Play", command=self.on_play).pack(side=tk.LEFT, padx=(0, 8))
        ttk.Button(button_row, text="Quit", command=self.root.destroy).pack(side=tk.LEFT)

        ttk.Label(frame, textvariable=self.info_var, justify=tk.LEFT).grid(row=5, column=0, sticky="w")
        ttk.Label(frame, textvariable=self.status_var, foreground="#005f99").grid(row=6, column=0, sticky="w", pady=(12, 0))

        frame.columnconfigure(0, weight=1)

    def refresh_options(self) -> None:
        """디스크 상태를 다시 읽어 run / video dropdown 후보를 갱신한다."""
        self.runs = discover_runs()
        self.videos = discover_videos()
        self.run_lookup = {run.display_name: run for run in self.runs}
        self.video_lookup = {video.name: video for video in self.videos}

        self.run_combo["values"] = list(self.run_lookup.keys())
        self.video_combo["values"] = list(self.video_lookup.keys())

        if self.runs and not self.run_var.get():
            self.run_var.set(self.runs[0].display_name)
        if self.videos and not self.video_var.get():
            self.video_var.set(self.videos[0].name)

        self.update_info()
        self.status_var.set(f"Runs: {len(self.runs)} | Videos: {len(self.videos)}")

    def update_info(self) -> None:
        """현재 선택된 run / video의 핵심 정보만 상태 영역에 표시한다."""
        run = self.run_lookup.get(self.run_var.get())
        video = self.video_lookup.get(self.video_var.get())

        if not run and not video:
            self.info_var.set("No trained run or video found.")
            return

        lines = []
        if run:
            lines.append(f"Run dir: {run.run_dir}")
            lines.append(f"Mode: {run.mode}")
        if video:
            lines.append(f"Video: {video}")
        lines.append("Inference device: CPU (fixed)")
        if not TASK_MODEL_PATH.exists():
            lines.append(f"Missing task model: {TASK_MODEL_PATH}")

        self.info_var.set("\n".join(lines))

    def get_runtime(self, run: RunInfo) -> RuntimeModel:
        """같은 run을 반복 재생할 때 checkpoint 재로딩 비용을 피하기 위한 캐시."""
        cache_key = str(run.run_dir)
        runtime = self.runtime_cache.get(cache_key)
        if runtime is not None:
            return runtime

        runtime = load_runtime_model(run)
        self.runtime_cache[cache_key] = runtime
        return runtime

    def on_play(self) -> None:
        """선택된 run/video 조합을 필요 시 분석하고 곧바로 playback 한다."""
        run = self.run_lookup.get(self.run_var.get())
        video = self.video_lookup.get(self.video_var.get())

        if run is None:
            messagebox.showerror("Missing Run", "Select a trained run first.")
            return
        if video is None:
            messagebox.showerror("Missing Video", "Select a video first.")
            return

        try:
            self.status_var.set("Loading model...")
            self.root.update_idletasks()
            runtime = self.get_runtime(run)

            cache_key = (str(run.run_dir), str(video))
            analyzed = self.analysis_cache.get(cache_key)
            if analyzed is None:
                # 동일 조합 재생 시 MediaPipe 추론 전체를 다시 돌리지 않도록 결과를 캐시한다.
                self.status_var.set("Analyzing video with MediaPipe + model inference...")
                self.root.update_idletasks()
                analyzed = analyze_video(runtime, video)
                self.analysis_cache[cache_key] = analyzed

            self.status_var.set("Playback started")
            self.root.update_idletasks()
            self.root.withdraw()
            try:
                playback(runtime, analyzed)
            finally:
                self.root.deiconify()
                self.root.update()
                self.root.lift()
                self.root.focus_force()
                self.status_var.set("Playback finished")
        except Exception as exc:
            self.status_var.set("Error")
            messagebox.showerror("Viewer Error", str(exc))


def run_cli(run_dir: Path, video_path: Path) -> None:
    """GUI 없이 특정 run/video 조합을 바로 재생하는 CLI 진입점."""
    run_dir = resolve_run_dir_arg(run_dir)
    video_path = video_path.resolve()
    checkpoint_path = run_dir / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    summary_path = run_dir / "run_summary.json"
    mode = "unknown"
    macro_f1: float | None = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mode = str(summary.get("mode", "unknown"))
        macro_f1 = summary.get("metrics", {}).get("macro_avg", {}).get("f1")

    run_info = RunInfo(
        model_id=run_dir.parent.name,
        run_dir=run_dir,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path if summary_path.exists() else None,
        mode=mode,
        macro_f1=float(macro_f1) if isinstance(macro_f1, (int, float)) else None,
        display_name=run_dir.as_posix(),
    )

    runtime = load_runtime_model(run_info)
    analyzed = analyze_video(runtime, video_path)
    playback(runtime, analyzed)


def resolve_run_dir_arg(path: Path) -> Path:
    """CLI의 --run-dir 인자로 run 폴더 또는 model 폴더(latest.json)를 모두 허용한다."""
    candidate = path.resolve()

    if candidate.is_file() and candidate.name == "latest.json":
        latest = json.loads(candidate.read_text(encoding="utf-8"))
        latest_run = Path(str(latest["latest_run"]))
        return latest_run.resolve()

    if (candidate / "model.pt").exists():
        return candidate

    latest_json = candidate / "latest.json"
    if latest_json.exists():
        latest = json.loads(latest_json.read_text(encoding="utf-8"))
        latest_run = Path(str(latest["latest_run"]))
        return latest_run.resolve()

    raise FileNotFoundError(
        "Run directory not found. Expected either "
        "`.../{suite_name}/{model_id}/{timestamp}/model.pt` or "
        "`.../{suite_name}/{model_id}/latest.json`."
    )


def parse_args() -> argparse.Namespace:
    """GUI 모드와 CLI 모드를 나누는 최소 인자만 받는다."""
    parser = argparse.ArgumentParser(description="JamJamBeat trained-model video viewer")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory containing model.pt, or a model directory containing latest.json "
            "(e.g. model/model_evaluation/pipelines/{suite_name}/{model_id})"
        ),
    )
    parser.add_argument("--video", type=Path, default=None, help="Video path to analyze")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Deprecated. The viewer now runs inference on CPU only.",
    )
    return parser.parse_args()


def main() -> None:
    """run-dir/video가 있으면 CLI, 없으면 Tk GUI로 진입한다."""
    args = parse_args()
    if args.run_dir and args.video:
        run_cli(args.run_dir, args.video)
        return

    root = tk.Tk()
    app = VideoCheckApp(root)
    if not app.runs:
        messagebox.showwarning("No Runs", f"No checkpoints found under {RUNS_ROOT}")
    if not app.videos:
        messagebox.showwarning("No Videos", f"No videos found under {RAW_VIDEO_ROOT}")
    root.mainloop()


if __name__ == "__main__":
    main()
