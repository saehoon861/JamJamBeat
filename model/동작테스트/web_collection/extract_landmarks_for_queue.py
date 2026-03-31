# extract_landmarks_for_queue.py - accepted 비디오만 landmark_queue로 추출
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

from common import LANDMARK_QUEUE_DIR, ROOT, SEED_PROFILE, ensure_seed_profile_current, load_jsonl


DEFAULT_INPUT = ROOT / "manifests" / "quality_manifest.jsonl"


def load_local_extractor():
    extractor_path = ROOT.parent / "landmark_extractor" / "main.py"
    spec = importlib.util.spec_from_file_location("dongjak_local_extractor", extractor_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"랜드마크 추출기를 로드할 수 없습니다: {extractor_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract landmarks for accepted queue items only.")
    parser.add_argument("--seed-profile", type=Path, default=SEED_PROFILE)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=LANDMARK_QUEUE_DIR)
    args = parser.parse_args()

    ensure_seed_profile_current(args.seed_profile)
    rows = load_jsonl(args.input)
    accepted = [row for row in rows if row.get("quality_status") == "accepted"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not accepted:
        print(json.dumps({"output_dir": str(args.output_dir), "processed": 0}, ensure_ascii=False, indent=2))
        return

    extractor = load_local_extractor()
    processed = 0
    for row in accepted:
        video_path = row["path"]
        stem = Path(video_path).stem
        output_csv = args.output_dir / f"{stem}.csv"
        if output_csv.exists():
            processed += 1
            continue
        extractor.process_video(video_path, str(args.output_dir), str(extractor.MODEL_PATH))
        processed += 1

    print(json.dumps({"output_dir": str(args.output_dir), "processed": processed}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
