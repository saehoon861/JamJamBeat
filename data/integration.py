import pandas as pd
from pathlib import Path

TOTAL_DATA_DIR = Path(__file__).parent / "total_data"
OUTPUT_FILE = Path(__file__).parent / "total_data" / "kimsg_total.csv"

# 이름 바꾸지 않으면 실행 차단
if OUTPUT_FILE.name == "sample.csv":
    raise ValueError("OUTPUT_FILE 이름을 지정해주세요. ")

def main():
    csv_files = sorted(TOTAL_DATA_DIR.glob("*.csv"))
    
    # .gitkeep 등 비CSV 파일 방어
    csv_files = [f for f in csv_files if f.suffix == ".csv"]

    if not csv_files:
        print("[ERROR] total_data/ 폴더에 CSV 파일이 없습니다.")
        return

    print(f"[INFO] 총 {len(csv_files)}개 파일 병합 시작...")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        df.insert(0, "source_file", f.stem)  # 어느 파일에서 왔는지 추적용 컬럼
        dfs.append(df)
        print(f"  ✅ {f.name} ({len(df)}행)")

    merged = pd.concat(dfs, ignore_index=True)
    merged.to_csv(OUTPUT_FILE, index=False)

    print(f"\n[DONE] 병합 완료 → {OUTPUT_FILE.name}")
    print(f"       총 {len(merged)}행 / {len(merged.columns)}컬럼")

if __name__ == "__main__":
    main()