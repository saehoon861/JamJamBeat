# example_usage.py - 선정 모델 번들을 로드하고 샘플 입력으로 1회 추론하는 예제
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from runtime import load_bundle


def main() -> None:
    bundle = load_bundle(ROOT, device="cpu")
    sample_input = json.loads((ROOT / "examples/sample_input.json").read_text(encoding="utf-8"))
    expected = json.loads((ROOT / "examples/expected_output.json").read_text(encoding="utf-8"))

    result = bundle.predict(sample_input["raw_joint63"], tau=None)

    print("bundle_id:", bundle.config["bundle_id"])
    print("sample:", sample_input["source_file"], sample_input["frame_idx"], sample_input["timestamp"])
    print("pred:", result["pred_label"], result["confidence"])
    print("expected_pred:", expected["pred_label"], expected["confidence"])
    print("match_pred:", result["pred_label"] == expected["pred_label"])
    print("match_argmax:", result["raw_pred_index"] == expected["raw_pred_index"])


if __name__ == "__main__":
    main()
