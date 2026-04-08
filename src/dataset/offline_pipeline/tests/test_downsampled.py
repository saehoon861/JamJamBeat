import sys
import argparse
import pandas as pd
from pathlib import Path

# --- path 설정 ---
current_dir = Path(__file__).resolve().parent
offline_dir = current_dir.parent
if str(offline_dir) not in sys.path:
    sys.path.insert(0, str(offline_dir))

# config import
from config import PROCESSED_DIR


def test_downsampled(csv_path: Path):
    print(f"\n[Test] Analyzing {csv_path}...")
    df = pd.read_csv(csv_path)

    # 0_neutral / 0_hardneg check
    bGroup = df[(df['gesture'] == 0) & (df['source_file'].str.startswith('0_'))]
    print(f"Survived Protected (0_neutral, 0_hardneg): {len(bGroup)} frames")

    # gesture distribution
    dist = df.groupby('gesture')['source_file'].count()

    print("\n--- Gesture Distribution ---")
    print(dist.to_string())

    aGroup_counts = df[df['gesture'] != 0].groupby('gesture')['source_file'].count()

    if not aGroup_counts.empty:
        mean_val = aGroup_counts.mean()
        class_0_total = dist.get(0, 0)

        print(f"\nAverage 1~6 count = {mean_val:.1f}")
        print(f"Current Class 0 count = {class_0_total}")

        if mean_val > 0:
            print(f"Ratio Class 0 to Average(1~6) = {class_0_total / mean_val:.2f}:1")
    else:
        print("No gestures 1~6 found in this file.")


def collect_csv_files(path: Path):
    if path.is_file():
        return [path]
    elif path.is_dir():
        return sorted(path.glob("*.csv"))
    else:
        raise ValueError(f"Invalid path: {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Specific csv file or directory to analyze"
    )

    args = parser.parse_args()

    # default: config의 PROCESSED_DIR 사용
    target_path = Path(args.csv_path) if args.csv_path else PROCESSED_DIR

    csv_files = collect_csv_files(target_path)

    print(f"\nFound {len(csv_files)} CSV files")

    for csv_file in csv_files:
        test_downsampled(csv_file)