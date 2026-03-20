"""
run_augment.py — 데이터 증강 파이프라인 CLI 진입점
===================================================
processed_scenarios/ 디렉토리의 시나리오 CSV를 선택하여
결합 증강 파이프라인(Mirror 50% → BLP 100% → Gaussian Noise 100%)을 적용한 뒤
augmented_scenarios/ 디렉토리에 저장한다.

사용법:
    uv run python src/dataset/offline_pipeline/runners/run_augment.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np

# config & modules 임포트를 위한 경로 설정
current_dir = Path(__file__).resolve().parent
offline_dir = current_dir.parent
if str(offline_dir) not in sys.path:
    sys.path.insert(0, str(offline_dir))

import config
from modules.augmentor import apply_mirroring, apply_blp, apply_gaussian_noise


def main():
    # ─── 1. 재현성 확보를 위한 글로벌 시드 고정 ───
    np.random.seed(config.AUG_RANDOM_SEED)

    # ─── 2. 출력 디렉토리 안전 생성 ───
    os.makedirs(config.DIR_AUGMENTED, exist_ok=True)

    print("\n=== 데이터 증강 파이프라인 ===")
    print("q 입력 시 종료\n")

    # ─── 3. 메인 루프 (q 입력 전까지 반복) ───
    while True:
        # 루프마다 목록을 새로 읽어 신규 파일도 반영
        csv_files = sorted([
            f for f in os.listdir(config.PROCESSED_DIR)
            if f.endswith('.csv')
        ])
        if not csv_files:
            print(f"[오류] {config.PROCESSED_DIR} 디렉토리에 CSV 파일이 없습니다.")
            break

        # 메뉴 출력
        print("처리 가능한 시나리오 목록:")
        for i, name in enumerate(csv_files, 1):
            print(f"  {i}. {name}")
        print(f"  0. 전체 일괄 처리")

        choice = input("\n번호 선택 (q=종료): ").strip()

        # 종료
        if choice.lower() == 'q':
            print("종료합니다.")
            break

        # 선택 파싱
        if choice == '0':
            targets = csv_files
        elif choice.isdigit() and 1 <= int(choice) <= len(csv_files):
            targets = [csv_files[int(choice) - 1]]
        else:
            print("[오류] 잘못된 입력입니다.\n")
            continue

        # ─── 4. 선택된 시나리오별 파이프라인 구동 ───
        for filename in targets:
            filepath = config.PROCESSED_DIR / filename
            scenario_name = Path(filename).stem

            print(f"\n--- [{scenario_name}] 증강 시작 ---")

            # 원본 CSV 로딩
            df_origin = pd.read_csv(filepath)
            print(f"  원본 로드 완료: {len(df_origin)} rows")

            # 원본 데이터에 증강 추적 메타 컬럼 초기화
            df_origin['aug_mirror'] = False
            df_origin['aug_blp'] = False
            df_origin['aug_noise_sigma'] = 0.0

            # 증강용 딥카피 생성 (원본 보호)
            df_aug = df_origin.copy()

            # ─── 파이프라인 순서 엄수: Mirror → BLP → Noise ───

            # [1단계] Mirroring (50% 확률 적용)
            mirror_prob = config.AUG_PARAMS['prob']['mirror']
            mirror_mask = apply_mirroring(df_aug, mirror_prob)
            mirror_count = mirror_mask.sum()
            print(f"  [Mirror] {mirror_count}/{len(df_aug)} 샘플 반전 적용 ({mirror_count/len(df_aug)*100:.1f}%)")

            # [2단계] BLP (100% 적용)
            apply_blp(df_aug, config.AUG_PARAMS['blp_scales'])
            print(f"  [BLP] 전체 {len(df_aug)} 샘플 뼈 길이 축소 적용 완료")

            # [3단계] Gaussian Noise (100% 적용, 파일명 기반 σ 분기)
            if 'scale' in filename:
                sigma_range = config.AUG_PARAMS['noise_sigma_range']['scale']
            else:
                sigma_range = config.AUG_PARAMS['noise_sigma_range']['non_scale']

            sigmas = apply_gaussian_noise(df_aug, sigma_range)
            print(f"  [Noise] σ 범위 {sigma_range} → 평균 σ={sigmas.mean():.6f}")

            # 증강 추적 메타 컬럼 기입
            df_aug['aug_mirror'] = mirror_mask
            df_aug['aug_blp'] = True
            df_aug['aug_noise_sigma'] = np.round(sigmas, 6)

            # 원본 + 증강 합병 (2배수 결합)
            df_final = pd.concat([df_origin, df_aug], ignore_index=True)
            print(f"  합병 완료: {len(df_origin)} (원본) + {len(df_aug)} (증강) = {len(df_final)} rows")

            # 최종 CSV 저장
            out_path = config.DIR_AUGMENTED / f"{scenario_name}_aug.csv"
            df_final.to_csv(out_path, index=False)
            print(f"  ✅ 저장 완료: {out_path.name}")

        print()  # 다음 메뉴 출력 전 줄바꿈


if __name__ == "__main__":
    main()
