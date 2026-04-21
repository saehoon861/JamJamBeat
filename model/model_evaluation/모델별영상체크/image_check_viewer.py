#!/usr/bin/env python3
"""
image_check_viewer.py - 학습 파이프라인과 동일한 방식으로 단일 이미지를 추론하고,
입력과 출력을 함께 시각화하여 모델을 검증하는 도구.

사용법:
    uv run python model/model_evaluation/모델별영상체크/image_check_viewer.py \
        --run-dir "model/model_evaluation/pipelines/Landmark_Spatial_Transformer/20240725_150000" \
        --image "data/raw_data/test_image.jpg"
"""

import argparse
import sys
from collections import deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# --- 경로 설정 ---
try:
    # 패키지 내부에서 실행될 때
    from . import video_check_app as base
    from . import video_check_app_train_aligned as aligned
except ImportError:
    # 스크립트로 직접 실행될 때
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    import video_check_app as base
    import video_check_app_train_aligned as aligned

# --- 메인 로직 ---

def build_run_info(run_dir_str: str) -> base.RunInfo:
    """CLI 경로를 RunInfo 객체로 변환"""
    run_dir = base.resolve_run_dir_arg(Path(run_dir_str))
    run_info_list = [r for r in base.discover_runs() if r.run_dir == run_dir]
    if not run_info_list:
        raise FileNotFoundError(f"Run info not found for directory: {run_dir}")
    return run_info_list[0]

def create_image_landmarker() -> mp.tasks.vision.HandLandmarker:
    """단일 이미지 추론용 MediaPipe Hand Landmarker 생성"""
    if not base.TASK_MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing task model: {base.TASK_MODEL_PATH}")

    options = base.HandLandmarkerOptions(
        base_options=base.BaseOptions(model_asset_path=str(base.TASK_MODEL_PATH)),
        running_mode=base.VisionRunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
    )
    return base.HandLandmarker.create_from_options(options)

def predict_for_single_image(
    runtime: base.RuntimeModel,
    features: aligned.TrainingAlignedFeaturePack
) -> tuple[str, int, float, list[float]]:
    """
    단일 이미지에 대해 프레임/시퀀스 모델을 모두 예측할 수 있도록 처리.
    시퀀스 모델의 경우, 현재 프레임의 특징을 반복하여 버퍼를 채워 예측을 수행.
    """
    seq_buffer = deque(maxlen=runtime.seq_len)

    if runtime.mode == "sequence" and runtime.input_dim is not None:
        # 시퀀스 모델은 현재 특징 벡터로 버퍼를 채워서 예측
        if runtime.model_id == "mlp_sequence_delta":
            if runtime.input_dim % 2 != 0:
                raise ValueError(f"Unexpected delta input_dim for {runtime.model_id}: {runtime.input_dim}")
            base_vec = aligned.select_feature_vector_training_aligned(features, runtime.input_dim // 2)
        else:
            base_vec = aligned.select_feature_vector_training_aligned(features, runtime.input_dim)

        for _ in range(runtime.seq_len):
            seq_buffer.append(base_vec)

    status, pred_idx, confidence, probs = aligned.predict_from_features_training_aligned(
        runtime, features, seq_buffer
    )
    return status, pred_idx, confidence, probs

def draw_visualization(
    image_bgr: np.ndarray,
    runtime: base.RuntimeModel,
    pred_name: str,
    confidence: float,
    probs: list[float],
    raw_landmarks: np.ndarray | None,
    features: aligned.TrainingAlignedFeaturePack | None,
) -> np.ndarray:
    """추론 결과와 모델 입력 등을 시각화하여 하나의 이미지로 합침"""
    display_img = image_bgr.copy()
    if raw_landmarks is not None:
        display_img = base.draw_raw_landmarks(display_img, raw_landmarks)

    h, w = display_img.shape[:2]
    panel_w = 400
    panel = np.zeros((h, panel_w, 3), dtype=np.uint8)

    y = 30
    cv2.putText(panel, f"Model: {runtime.model_id}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y += 25
    cv2.putText(panel, f"Run: {runtime.run_info.run_dir.name}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    y += 25
    cv2.putText(panel, f"Variant: {runtime.dataset_variant} ({runtime.dataset_variant_source})", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    y += 40

    pred_color = (0, 255, 0) if confidence > 0.5 else (0, 200, 255)
    cv2.putText(panel, f"Prediction: {pred_name}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, pred_color, 2, cv2.LINE_AA)
    y += 30
    cv2.putText(panel, f"Confidence: {confidence:.3f}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2, cv2.LINE_AA)
    y += 40

    cv2.putText(panel, "Probabilities:", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    y += 25
    for i, p in enumerate(probs):
        class_name = runtime.class_names[i]
        bar_w = int(p * (panel_w - 120))
        bar_color = (0, 200, 100) if i == np.argmax(probs) else (80, 80, 80)
        cv2.rectangle(panel, (100, y - 12), (100 + bar_w, y + 4), bar_color, -1)
        cv2.putText(panel, f"{class_name[:9]}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{p:.2f}", (105 + bar_w, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        y += 22
    y += 20

    if features is not None:
        cv2.putText(panel, "Model Input (Normalized):", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        y += 10
        skeleton_size = 200
        skeleton_img = aligned.render_train_skeleton_image(features.train_landmarks, skeleton_size)
        skeleton_img_bgr = cv2.cvtColor(skeleton_img.squeeze(), cv2.COLOR_GRAY2BGR)
        if y + skeleton_size > h: skeleton_size = h - y - 10
        if skeleton_size > 0:
            skeleton_img_bgr = cv2.resize(skeleton_img_bgr, (skeleton_size, skeleton_size))
            panel[y:y+skeleton_size, 10:10+skeleton_size] = (skeleton_img_bgr * 255).astype(np.uint8)

    return cv2.hconcat([display_img, panel])

def main():
    parser = argparse.ArgumentParser(description="학습된 모델로 단일 이미지를 추론하고 시각화합니다.")
    parser.add_argument("--run-dir", type=str, required=True, help="모델 체크포인트가 포함된 run 디렉토리 경로")
    parser.add_argument("--image", type=str, required=True, help="추론할 이미지 파일 경로")
    args = parser.parse_args()

    print(f"Loading model from: {args.run_dir}")
    run_info = build_run_info(args.run_dir)
    runtime = aligned.load_runtime_model_training_aligned(run_info)
    print(f"Model '{runtime.model_id}' loaded. Mode: {runtime.mode}, Variant: {runtime.dataset_variant}")

    image_path = Path(args.image).resolve()
    if not image_path.exists(): raise FileNotFoundError(f"Image not found: {image_path}")
    image_bgr = cv2.imread(str(image_path))
    print(f"Image loaded: {image_path.name}")

    with create_image_landmarker() as landmarker:
        rgb_frame = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = landmarker.detect(mp_image)

    if not detection_result.hand_landmarks:
        print("No hand detected in the image.")
        pred_name, confidence, probs, raw_landmarks, features = runtime.class_names[runtime.neutral_idx], 0.0, base.neutral_probs(len(runtime.class_names), runtime.neutral_idx), None, None
    else:
        print("Hand detected, running inference...")
        raw_landmarks = np.array([[lm.x, lm.y, lm.z] for lm in detection_result.hand_landmarks[0]], dtype=np.float32)
        features = aligned.extract_feature_pack_training_aligned(raw_landmarks, dataset_variant=runtime.dataset_variant)
        _, pred_idx, confidence, probs = predict_for_single_image(runtime, features)
        pred_name = runtime.class_names[pred_idx]
        print(f"Prediction: {pred_name} (Confidence: {confidence:.3f})")

    output_image = draw_visualization(image_bgr, runtime, pred_name, confidence, probs, raw_landmarks, features)
    cv2.imshow("JamJamBeat - Image Check Viewer", output_image)
    print("\nPress any key to exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()