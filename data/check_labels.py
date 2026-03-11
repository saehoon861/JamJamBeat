import pandas as pd
import glob
import os

# total_data 폴더의 모든 CSV 읽기
files = glob.glob("total_data/*.csv")

all_data = []
for f in files:
    df = pd.read_csv(f)
    # null 값 행 제거
    df = df.dropna(subset=['x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'x5', 'y5', 'z5', 'x6', 'y6', 'z6', 'x7', 'y7', 'z7', 'x8', 'y8', 'z8', 'x9', 'y9', 'z9', 'x10', 'y10', 'z10', 'x11', 'y11', 'z11', 'x12', 'y12', 'z12', 'x13', 'y13', 'z13', 'x14', 'y14', 'z14', 'x15', 'y15', 'z15', 'x16', 'y16', 'z16', 'x17', 'y17', 'z17', 'x18', 'y18', 'z18', 'x19', 'y19', 'z19', 'x20', 'y20', 'z20'])
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