#!/usr/bin/env python3
# run_no_pretrained.py - explicit split dataset 세트를 비-pretrained 모델 위주로 순차 실행한다.
from __future__ import annotations

import run_all


EXCLUDED = {"mobilenetv3_small", "shufflenetv2_x0_5", "efficientnet_b0"}
DEFAULT_MODELS = [model for model in run_all.ALL_MODELS if model not in EXCLUDED]


def main() -> None:
    parser = run_all.build_parser()
    parser.description = "Run non-pretrained JamJamBeat models for explicit split datasets."
    parser.add_argument(
        "--include-pretrained",
        action="store_true",
        help="기본 제외되는 pretrained CNN 3종도 함께 실행",
    )
    args = parser.parse_args()

    if not args.models:
        args.models = run_all.ALL_MODELS if args.include_pretrained else DEFAULT_MODELS

    run_all.run_all(args)


if __name__ == "__main__":
    main()
