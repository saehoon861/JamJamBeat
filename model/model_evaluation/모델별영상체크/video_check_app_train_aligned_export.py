#!/usr/bin/env python3
"""video_check_app_train_aligned_export.py - Export the training-aligned viewer overlay as an annotated video file."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path
import sys

import cv2
import numpy as np

try:
    from . import video_check_app_train_aligned as aligned
except ImportError:
    THIS_DIR = Path(__file__).resolve().parent
    if str(THIS_DIR) not in sys.path:
        sys.path.insert(0, str(THIS_DIR))
    import video_check_app_train_aligned as aligned


base = aligned.base


def build_run_info(run_dir: Path) -> base.RunInfo:
    """Resolve a run directory or latest.json pointer into the RunInfo structure used by the viewer."""
    resolved_run = base.resolve_run_dir_arg(run_dir)
    checkpoint_path = resolved_run / "model.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    summary_path = resolved_run / "run_summary.json"
    mode = "unknown"
    macro_f1: float | None = None
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        mode = str(summary.get("mode", "unknown"))
        macro_f1 = summary.get("metrics", {}).get("macro_avg", {}).get("f1")

    return base.RunInfo(
        model_id=resolved_run.parent.name,
        run_dir=resolved_run,
        checkpoint_path=checkpoint_path,
        summary_path=summary_path if summary_path.exists() else None,
        mode=mode,
        macro_f1=float(macro_f1) if isinstance(macro_f1, (int, float)) else None,
        display_name=resolved_run.as_posix(),
    )


def default_output_path(run_info: base.RunInfo, video_path: Path) -> Path:
    """Build a predictable export filename next to the source video when --output is omitted."""
    return video_path.with_name(f"{video_path.stem}__{run_info.model_id}__annotated.mp4")


def analyze_video_for_export(
    runtime: base.RuntimeModel,
    video_path: Path,
    max_frames: int | None = None,
) -> base.AnalyzedVideo:
    """Analyze the source video with an optional frame cap for quick exports/debug runs."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if max_frames is not None:
        total_frames = min(total_frames, max_frames)

    results: list[base.FrameResult] = []
    seq_buffer = deque(maxlen=runtime.seq_len)

    print(f"[viewer] analyzing {video_path.name} with {runtime.model_id}")
    print(f"[viewer] total_frames={total_frames}, fps={fps:.2f}, device={runtime.device}")

    try:
        with base.create_landmarker() as landmarker:
            for frame_idx in range(total_frames):
                ok, frame = cap.read()
                if not ok:
                    break

                timestamp_ms = int((frame_idx / max(fps, 1e-6)) * 1000)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = base.mp.Image(image_format=base.mp.ImageFormat.SRGB, data=rgb)
                mp_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if not mp_result.hand_landmarks:
                    seq_buffer.clear()
                    pred_idx = runtime.neutral_idx
                    probs = base.neutral_probs(len(runtime.class_names), runtime.neutral_idx)
                    results.append(
                        base.FrameResult(
                            frame_idx=frame_idx,
                            timestamp_text=base.format_timestamp(frame_idx, fps),
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
                    features = base.extract_feature_pack(raw_landmarks)
                    status, pred_idx, confidence, probs = base.predict_from_features(runtime, features, seq_buffer)
                    results.append(
                        base.FrameResult(
                            frame_idx=frame_idx,
                            timestamp_text=base.format_timestamp(frame_idx, fps),
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
    finally:
        cap.release()

    return base.AnalyzedVideo(video_path=video_path, fps=fps, total_frames=len(results), frame_results=results)


def export_annotated_video(
    runtime: base.RuntimeModel,
    analyzed: base.AnalyzedVideo,
    output_path: Path,
    codec: str = "mp4v",
    max_frames: int | None = None,
) -> Path:
    """Render the same overlay used by playback() and save it as a video file."""
    cap = cv2.VideoCapture(str(analyzed.video_path))
    if not cap.isOpened():
        raise IOError(f"Could not reopen video: {analyzed.video_path}")

    try:
        ok, first_frame = cap.read()
        if not ok:
            raise IOError(f"Could not read first frame from video: {analyzed.video_path}")

        first_display = base.overlay_frame(first_frame, runtime, analyzed, 0)
        height, width = first_display.shape[:2]
        fps = float(analyzed.fps or 30.0)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*codec)
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise IOError(f"Could not open video writer for: {output_path}")

        try:
            total_frames = analyzed.total_frames
            if max_frames is not None:
                total_frames = min(total_frames, max_frames)

            writer.write(first_display)
            print(f"[export] wrote 1/{total_frames} frames")

            for current_idx in range(1, total_frames):
                ok, frame = cap.read()
                if not ok:
                    print(f"[export] stopped early at frame {current_idx}: source video ended")
                    break

                display = base.overlay_frame(frame, runtime, analyzed, current_idx)
                writer.write(display)

                if (current_idx + 1) % 30 == 0 or current_idx + 1 == total_frames:
                    print(f"[export] wrote {current_idx + 1}/{total_frames} frames")
        finally:
            writer.release()
    finally:
        cap.release()

    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export JamJamBeat training-aligned viewer output as an annotated video file."
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help=(
            "Run directory containing model.pt, or a model directory containing latest.json "
            "(e.g. model/model_evaluation/pipelines/mlp_baseline/20260318_152800, "
            "model/model_evaluation/pipelines/mlp_baseline, or "
            "model/model_evaluation/pipelines/{suite_name}/{model_id})"
        ),
    )
    parser.add_argument("--video", type=Path, default=None, help="Video path to analyze")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output annotated video path. Default: next to source video as *_annotated.mp4",
    )
    parser.add_argument("--codec", type=str, default="mp4v", help="OpenCV fourcc codec, default: mp4v")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for export/debug")
    parser.add_argument(
        "--play-after",
        action="store_true",
        help="After export, open the normal viewer playback window as well.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.run_dir and args.video:
        run_info = build_run_info(args.run_dir)
        video_path = args.video.resolve()
        runtime = base.load_runtime_model(run_info)
        analyzed = analyze_video_for_export(runtime, video_path, max_frames=args.max_frames)
        output_path = args.output.resolve() if args.output else default_output_path(run_info, video_path)
        exported = export_annotated_video(
            runtime=runtime,
            analyzed=analyzed,
            output_path=output_path,
            codec=args.codec,
            max_frames=args.max_frames,
        )
        print(f"[export] saved annotated video to {exported}")
        if args.play_after:
            base.playback(runtime, analyzed)
        return

    # If no CLI export arguments are provided, fall back to the existing viewer behavior.
    aligned.main()


if __name__ == "__main__":
    main()
