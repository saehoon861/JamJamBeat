# run_no_pretrained.py - mobilenetv3_small / shufflenetv2_x0_5 / efficientnet_b0 제외 배치 실행
"""
이미지 렌더링 기반 CNN 3개 모델을 제외하고 나머지 11개 모델만 순차 실행한다.
run_all.py와 동일한 CLI 인터페이스를 공유한다.

Usage:
    python run_no_pretrained.py
    python run_no_pretrained.py --csv-path path/to/data.csv
    python run_no_pretrained.py --epochs 30 --models mlp_original mlp_baseline
"""
import sys
from pathlib import Path

# run_all.py의 모든 로직을 그대로 재사용하고 ALL_MODELS만 교체한다.
sys.path.insert(0, str(Path(__file__).parent))
import run_all

EXCLUDED = {"mobilenetv3_small", "shufflenetv2_x0_5", "efficientnet_b0"}

ALL_MODELS = [m for m in run_all.ALL_MODELS if m not in EXCLUDED]


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="JamJamBeat — 이미지 CNN 3종 제외 모델 파이프라인 순차 실행"
    )
    parser.add_argument(
        "--models", nargs="*", default=None,
        help=f"실행할 모델 지정 (미지정 시 전체). 선택: {ALL_MODELS}",
    )
    parser.add_argument(
        "--csv-path", action="append", dest="csv_path", default=[],
        help="입력 CSV 경로 (반복 사용 가능). 미지정 시 DEFAULT_INPUTS 사용.",
    )
    parser.add_argument("--output-root", default="model/model_evaluation/pipelines")
    parser.add_argument(
        "--upload-wandb", action="store_true",
        help="실험 완료 후 wandb에 결과 업로드",
    )

    # 학습 하이퍼파라미터
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--batch-size",  type=int,   default=32)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--patience",    type=int,   default=6)
    parser.add_argument("--focal-gamma", type=float, default=2.0)

    # 시퀀스 / 이미지
    parser.add_argument("--seq-len",     type=int, default=8)
    parser.add_argument("--seq-stride",  type=int, default=2)
    parser.add_argument("--image-size",  type=int, default=96)

    # 후처리 파라미터
    parser.add_argument("--tau",         type=float, default=0.90)
    parser.add_argument("--vote-n",      type=int,   default=7)
    parser.add_argument("--debounce-k",  type=int,   default=5)
    parser.add_argument("--fallback-fps",type=float, default=30.0)

    # train/val/test 비율 (단일 CSV 입력 시 row-level split에 적용)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio",   type=float, default=0.1)
    parser.add_argument("--test-ratio",  type=float, default=0.1)

    # 공통
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--device",      default="cpu", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int,   default=0)

    args = parser.parse_args()

    # --models 미지정 시 이미지 3종 제외 목록으로 덮어쓴다.
    if not args.models:
        args.models = ALL_MODELS

    run_all.run_all(args)


if __name__ == "__main__":
    main()
