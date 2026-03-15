'''
total_data 폴더 안의 모든 csv 파일을 읽어서
좌표 컬럼에서 하나라도 null인 행을 제거하고
total_data 폴더 안에 각 기존 파일의 이름 뒤에 _notnull 을 붙여서 저장

'''

import pandas as pd
from pathlib import Path
import glob
import os

# total_data 폴더의 모든 CSV 읽기
files = glob.glob("total_data/*.csv")



all_data = []
for f in files:
    df = pd.read_csv(f)
    # 좌표 컬럼에서 하나라도 null인 행 제거
    df = df.dropna(subset=['x0', 'y0', 'z0', 'x1', 'y1', 'z1', 'x2', 'y2', 'z2', 'x3', 'y3', 'z3', 'x4', 'y4', 'z4', 'x5', 'y5', 'z5', 'x6', 'y6', 'z6', 'x7', 'y7', 'z7', 'x8', 'y8', 'z8', 'x9', 'y9', 'z9', 'x10', 'y10', 'z10', 'x11', 'y11', 'z11', 'x12', 'y12', 'z12', 'x13', 'y13', 'z13', 'x14', 'y14', 'z14', 'x15', 'y15', 'z15', 'x16', 'y16', 'z16', 'x17', 'y17', 'z17', 'x18', 'y18', 'z18', 'x19', 'y19', 'z19', 'x20', 'y20', 'z20'])
    # total_data 폴더 안에 _notnull 파일들을 모아놓을 폴더 생성
    # 만일 이미 있을 시 skip
    if not os.path.exists("total_data/notnull"):
        os.makedirs("total_data/notnull")
    # 기존 파일의 이름 뒤에 _notnull을 추가해서 total_data/notnull에 .csv 파일 형태로 저장
    f = Path(f)
    df.to_csv(f.with_name(f.stem + "_notnull.csv"), index=False) 
    
   

    
        
    

