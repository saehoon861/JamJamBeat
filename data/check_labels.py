import pandas as pd
import glob
import os

# total_data 폴더의 모든 CSV 읽기
files = glob.glob("total_data/*.csv")

all_data = []
for f in files:
    df = pd.read_csv(f)
    df["source"] = os.path.basename(f)
    all_data.append(df)

df_all = pd.concat(all_data, ignore_index=True)

# 전체 클래스별 분포
print("=" * 40)
print("📊 전체 라벨링 현황")
print("=" * 40)
counts = df_all["gesture"].value_counts().sort_index()
total = len(df_all)

for label, cnt in counts.items():
    bar = "█" * int(cnt / total * 40)
    print(f"  Class {label}: {cnt:5d}개  ({cnt/total*100:.1f}%)  {bar}")

print(f"\n  총 프레임 수: {total}개")
print(f"  균등 기준:   {total//len(counts)}개/클래스")

# 파일별 분포
print("\n" + "=" * 40)
print("📁 파일별 클래스 분포")
print("=" * 40)
pivot = df_all.groupby(["source", "gesture"]).size().unstack(fill_value=0)
print(pivot.to_string())