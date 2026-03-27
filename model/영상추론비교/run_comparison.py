# run_comparison.py - manifest 기반 viewer/frontend probe 실행과 요약 생성
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from shared.gesture_resolution import normalize_model_label


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _run(cmd: list[str]) -> None:
    subprocess.run(cmd, cwd=ROOT_DIR, check=True)


def _collect_stats(viewer: dict[str, Any], frontend: dict[str, Any]) -> dict[str, Any]:
    viewer_by_frame = {int(record["frame_idx"]): record for record in viewer["records"]}
    frontend_infer_records = [record for record in frontend["records"] if record.get("mp_inference_ran")]

    compared = 0
    raw_label_mismatch = 0
    status_mismatch = 0
    tau_mismatch = 0
    no_hand_reset_mismatch = 0
    frontend_disabled_frames = 0
    final_vs_raw_diff = 0
    frontend_reason_counts: dict[str, int] = {}
    frontend_selected_hand_key_counts: dict[str, int] = {}
    first_viewer_ready = None
    first_frontend_ready = None

    for record in viewer["records"]:
        if record.get("status") == "ready" and first_viewer_ready is None:
            first_viewer_ready = int(record["frame_idx"])
            break
    for record in frontend_infer_records:
        raw_model = record.get("raw_model") or {}
        if raw_model.get("status") == "ready" and first_frontend_ready is None:
            first_frontend_ready = int(record["frame_idx"])
            break

    for record in frontend_infer_records:
        reason = str(record.get("reason") or "unknown")
        frontend_reason_counts[reason] = frontend_reason_counts.get(reason, 0) + 1
        selected_key = str((record.get("selected_info") or {}).get("selected_hand_key"))
        frontend_selected_hand_key_counts[selected_key] = frontend_selected_hand_key_counts.get(selected_key, 0) + 1

        frame_idx = int(record["frame_idx"])
        viewer_record = viewer_by_frame.get(frame_idx)
        if viewer_record is None:
            continue
        compared += 1

        raw_model = record.get("raw_model") or {}
        final_gesture = record.get("final_gesture") or {}
        if record.get("actual_frontend_disabled"):
            frontend_disabled_frames += 1

        if raw_model.get("raw_pred_label") != viewer_record.get("raw_pred_label"):
            raw_label_mismatch += 1
        if raw_model.get("status") != viewer_record.get("status"):
            status_mismatch += 1
        if bool(raw_model.get("tau_neutralized")) != bool(viewer_record.get("tau_neutralized")):
            tau_mismatch += 1

        viewer_no_hand = viewer_record.get("status") == "no_hand"
        frontend_no_hand = (raw_model.get("status") == "no_hand") or (record.get("reason") == "no_hand")
        if viewer_no_hand != frontend_no_hand:
            no_hand_reset_mismatch += 1

        mapped_raw = normalize_model_label(raw_model.get("label"))
        if mapped_raw != "None" and final_gesture.get("label") != mapped_raw:
            final_vs_raw_diff += 1

    return {
        "compared_frames": compared,
        "raw_label_mismatch_frames": raw_label_mismatch,
        "status_mismatch_frames": status_mismatch,
        "tau_mismatch_frames": tau_mismatch,
        "no_hand_reset_mismatch_frames": no_hand_reset_mismatch,
        "frontend_disabled_frames": frontend_disabled_frames,
        "frontend_final_vs_raw_diff_frames": final_vs_raw_diff,
        "frontend_reason_counts": frontend_reason_counts,
        "frontend_selected_hand_key_counts": frontend_selected_hand_key_counts,
        "first_viewer_ready_frame": first_viewer_ready,
        "first_frontend_ready_frame": first_frontend_ready,
    }


def _build_summary(manifest: dict[str, Any], viewer: dict[str, Any], frontend: dict[str, Any], stats: dict[str, Any]) -> str:
    return f"""# Frontend vs Viewer Comparison Summary

## Run
- video: `{manifest['video_path']}`
- bundle_id: `{manifest['bundle_id']}`
- model_id: `{manifest['model_id']}`
- hand: `{manifest['hand']}`
- tau_frontend: `{manifest['tau_frontend']}`
- tau_viewer_baseline: `{manifest['tau_viewer_baseline']}`
- tau_viewer_matched: `{manifest['tau_viewer_matched']}`

## Current Result
- compared_frames: `{stats['compared_frames']}`
- raw_label_mismatch_frames: `{stats['raw_label_mismatch_frames']}`
- status_mismatch_frames: `{stats['status_mismatch_frames']}`
- tau_mismatch_frames: `{stats['tau_mismatch_frames']}`
- no_hand_reset_mismatch_frames: `{stats['no_hand_reset_mismatch_frames']}`
- frontend_disabled_frames: `{stats['frontend_disabled_frames']}`
- frontend_final_vs_raw_diff_frames: `{stats['frontend_final_vs_raw_diff_frames']}`
- frontend_reason_counts: `{stats['frontend_reason_counts']}`
- frontend_selected_hand_key_counts: `{stats['frontend_selected_hand_key_counts']}`
- first_viewer_ready_frame: `{stats['first_viewer_ready_frame']}`
- first_frontend_ready_frame: `{stats['first_frontend_ready_frame']}`

## Notes
- checkpoint mismatch는 제외됨: viewer/frontend 기준 bundle fingerprint가 모두 `{manifest['checkpoint_fingerprint']}`
- viewer probe는 `full frame + num_hands=1 + 0.5 thresholds + tau={manifest['tau_viewer_baseline']}` 기준
- frontend probe는 `inferWidth=96 + inferFps=15 + modelIntervalMs=150 + num_hands=2 + 0.25 thresholds + tau={manifest['tau_frontend']}` 기준
- frontend final gesture는 `gestures.js`의 `mapModelToResult() + stabilize()`를 별도 적용한 결과
"""


def main() -> None:
    manifest_path = ROOT_DIR / "comparison_manifest.json"
    manifest = _read_json(manifest_path)

    artifacts_dir = ROOT_DIR / "artifacts"
    viewer_output = artifacts_dir / "viewer_probe.json"
    viewer_matched_output = artifacts_dir / "viewer_probe_tau085.json"
    frontend_output = artifacts_dir / "frontend_probe.json"

    python_exe = sys.executable
    _run(
        [
            python_exe,
            str(ROOT_DIR / "viewer_probe" / "run_viewer_probe.py"),
            "--video",
            manifest["video_path"],
            "--bundle-dir",
            manifest["viewer_bundle_dir"],
            "--output",
            str(viewer_output),
            "--tau",
            str(manifest["tau_viewer_baseline"]),
        ]
    )
    _run(
        [
            python_exe,
            str(ROOT_DIR / "viewer_probe" / "run_viewer_probe.py"),
            "--video",
            manifest["video_path"],
            "--bundle-dir",
            manifest["viewer_bundle_dir"],
            "--output",
            str(viewer_matched_output),
            "--tau",
            str(manifest["tau_viewer_matched"]),
        ]
    )
    _run(
        [
            python_exe,
            str(ROOT_DIR / "frontend_probe" / "run_frontend_probe.py"),
            "--video",
            manifest["video_path"],
            "--bundle-dir",
            manifest["frontend_bundle_dir"],
            "--output",
            str(frontend_output),
            "--tau",
            str(manifest["tau_frontend"]),
            "--infer-fps",
            str(manifest["infer_fps"]),
            "--model-interval-ms",
            str(manifest["model_interval_ms"]),
            "--infer-width",
            str(manifest["infer_width"]),
            "--hand",
            manifest["hand"],
        ]
    )

    viewer = _read_json(viewer_output)
    viewer_tau_matched = _read_json(viewer_matched_output)
    frontend = _read_json(frontend_output)

    baseline_stats = _collect_stats(viewer, frontend)
    matched_stats = _collect_stats(viewer_tau_matched, frontend)
    stats = {
        "viewer_tau_090": baseline_stats,
        "viewer_tau_085": matched_stats,
    }

    _write_json(artifacts_dir / "comparison_stats.json", stats)
    _write_text(
        artifacts_dir / "summary.md",
        _build_summary(manifest, viewer, frontend, baseline_stats)
        + "\n## Tau Matched Reference\n"
        + "\n".join(f"- {key}: `{value}`" for key, value in matched_stats.items())
        + "\n",
    )


if __name__ == "__main__":
    main()
