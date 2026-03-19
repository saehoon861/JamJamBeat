import pandas as pd
import numpy as np

# 서브그룹 샘플링 비율 상수
RATIO_HARDNEG = 0.4
RATIO_TRANS   = 0.3
RATIO_NEUTRAL = 0.2
RATIO_SAFE    = 0.1

LANDMARK_COLS = [f"{axis}{i}" for i in range(21) for axis in ("x", "y", "z")]


def _remove_null_landmarks(df: pd.DataFrame) -> pd.DataFrame:
    """랜드마크 63개 컬럼이 전부 null인 프레임 제거"""
    null_mask = df[LANDMARK_COLS].isnull().all(axis=1)
    return df[~null_mask].copy()


def _classify_group_c(
    df_original: pd.DataFrame,
    group_c: pd.DataFrame,
    margin_drop: int,
    margin_collect: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    원본 df에서 전환점을 frame_idx 기준으로 탐지하고,
    group_c의 각 프레임을 C_transition / drop / C_safe 로 분류.

    - drop 구간    : 전환점 frame_idx ± margin_drop 이내 → 제거
    - collect 구간 : 전환점 frame_idx ± (margin_drop+1 ~ margin_collect) 이내,
                     drop 구간과 미겹침, gesture==0 → C_transition
    - 나머지       : C_safe
    """
    collect_indices = set()
    drop_indices    = set()

    for source_file, file_group in df_original.groupby('source_file'):
        # frame_idx 기준으로 정렬
        file_group = file_group.sort_values('frame_idx')
        frame_idxs = file_group['frame_idx'].values
        gestures   = file_group['gesture'].values
        orig_index = file_group.index.values  # DataFrame index (행 번호)

        # frame_idx → DataFrame index 매핑
        fidx_to_dfidx = dict(zip(frame_idxs, orig_index))

        # 전환점: gesture가 바뀌는 frame_idx 탐지
        transition_frame_idxs = []
        for i in range(len(gestures) - 1):
            if gestures[i] != gestures[i + 1]:
                # 전환점은 두 프레임 사이 → 앞 프레임의 frame_idx 기록
                transition_frame_idxs.append(frame_idxs[i])

        # 1) drop 구간: 전환점 ± margin_drop frame_idx 범위에 속하는 프레임
        file_drop_dfidx = set()
        for t_fidx in transition_frame_idxs:
            for fidx, dfidx in fidx_to_dfidx.items():
                if abs(fidx - t_fidx) <= margin_drop:
                    file_drop_dfidx.add(dfidx)
        drop_indices.update(file_drop_dfidx)

        # 2) collect 구간: margin_drop < |frame_idx - t_fidx| <= margin_collect
        #    drop 구간과 미겹침, gesture==0인 프레임만
        for t_fidx in transition_frame_idxs:
            for fidx, dfidx in fidx_to_dfidx.items():
                dist = abs(fidx - t_fidx)
                if margin_drop < dist <= margin_collect:
                    if dfidx in file_drop_dfidx:
                        continue  # 혹시 다른 전환점 drop과 겹치면 skip
                    # gesture==0인지 확인 (group_c에 있는 프레임인지)
                    if dfidx in group_c.index:
                        collect_indices.add(dfidx)

    c_transition = group_c[group_c.index.isin(collect_indices)]
    c_safe = group_c[
        ~group_c.index.isin(drop_indices) &
        ~group_c.index.isin(collect_indices)
    ]
    return c_transition, c_safe


def _sample_by_source_class(df: pd.DataFrame, n: int, random_state: int = 42) -> pd.DataFrame:
    """
    source_file 앞 숫자(제스처 클래스)별로 비례 배분해서 n개 샘플링.
    각 클래스 최소 1개 보장 (데이터 있을 경우).
    """
    if len(df) == 0 or n == 0:
        return df.iloc[0:0]
    if n >= len(df):
        return df

    df = df.copy()
    df['_src_class'] = df['source_file'].str.split('_').str[0]

    class_counts = df['_src_class'].value_counts()
    total        = len(df)
    sampled_parts = []
    remainder     = n
    classes       = class_counts.index.tolist()

    for i, cls in enumerate(classes):
        cls_df = df[df['_src_class'] == cls]
        if i == len(classes) - 1:
            quota = remainder
        else:
            quota = max(1, round(n * len(cls_df) / total))
            quota = min(quota, remainder - (len(classes) - i - 1))

        quota = min(quota, len(cls_df))
        sampled_parts.append(cls_df.sample(n=quota, random_state=random_state))
        remainder -= quota
        if remainder <= 0:
            break

    result = pd.concat(sampled_parts).drop(columns=['_src_class'])
    return result


def _apply_fallback_sampling(
    group_hardneg : pd.DataFrame,
    group_trans   : pd.DataFrame,
    group_neutral : pd.DataFrame,
    group_safe    : pd.DataFrame,
    target_total  : int,
    random_state  : int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    우선순위 기반 fallback 샘플링.
    부족 시 다음 순위로 이월, 초과 시 target * 비율만큼만 샘플링.
    우선순위: hardneg → transition → neutral → safe
    """
    targets = {
        'hardneg': int(target_total * RATIO_HARDNEG),
        'trans'  : int(target_total * RATIO_TRANS),
        'neutral': int(target_total * RATIO_NEUTRAL),
        'safe'   : int(target_total * RATIO_SAFE),
    }
    groups = {
        'hardneg': group_hardneg,
        'trans'  : group_trans,
        'neutral': group_neutral,
        'safe'   : group_safe,
    }
    priority   = ['hardneg', 'trans', 'neutral', 'safe']
    sampled    = {}
    carry_over = 0

    for key in priority:
        g = groups[key]
        t = targets[key] + carry_over

        if len(g) == 0:
            sampled[key] = g.iloc[0:0]
            carry_over   = t
            continue

        if len(g) >= t:
            sampled[key] = _sample_by_source_class(g, t, random_state)
            carry_over   = 0
        else:
            sampled[key] = g.copy()
            carry_over   = t - len(g)

    return sampled['hardneg'], sampled['trans'], sampled['neutral'], sampled['safe']


def apply_downsampling(
    df            : pd.DataFrame,
    target_ratio  : str,
    margin_drop   : int = 2,
    margin_collect: int = 10,
) -> pd.DataFrame:
    """
    target_ratio에 따라 Class 0 프레임을 4개 서브그룹으로 나눠 다운샘플링.

    서브그룹 우선순위 및 비율:
      1. B_hardneg    (0_hardneg_ 파일,   40%) : 유사동작 오인식 방지 핵심
      2. C_transition (제스처 전환 주변,  30%) : 중간동작 커버
      3. B_neutral    (0_neutral_ 파일,   20%) : 일반 배경
      4. C_safe       (그 외 gesture==0,  10%) : 실제 환경 맥락

    Args:
        df            : 원본 DataFrame
        target_ratio  : 'origin' | '4:1' | '1:1'
        margin_drop   : 전환점 주변 제거 프레임 수, frame_idx 기준 (default 2)
        margin_collect: 전환점 주변 수집 최대 프레임 수, frame_idx 기준 (default 10)
    """
    if target_ratio == "origin":
        return df.copy()

    # ── 전처리: 랜드마크 전부 null 프레임 제거 ──────────────────────────
    # 원본 df는 전환점 탐지용으로 보존, null 제거는 별도 처리
    df_for_transition = df.copy()   # 전환점 탐지용 원본 (null 제거 전)
    df = _remove_null_landmarks(df) # 실제 학습 데이터용

    # ── 그룹 분리 ────────────────────────────────────────────────────────
    group_a = df[df['gesture'] != 0]

    group_b_hardneg = df[
        (df['gesture'] == 0) & df['source_file'].str.startswith('0_hardneg_')
    ]
    group_b_neutral = df[
        (df['gesture'] == 0) & df['source_file'].str.startswith('0_neutral_')
    ]
    group_c = df[
        (df['gesture'] == 0) & ~df['source_file'].str.startswith('0_')
    ]

    # ── Group C → C_transition / C_safe 분리 ────────────────────────────
    # 전환점 탐지는 null 제거 전 원본 df 기준, 실제 수집은 group_c(null 제거 후) 기준
    c_transition, c_safe = _classify_group_c(
        df_original=df_for_transition,
        group_c=group_c,
        margin_drop=margin_drop,
        margin_collect=margin_collect,
    )

    # ── target 계산 ──────────────────────────────────────────────────────
    class_counts = group_a['gesture'].value_counts()
    mean_count   = class_counts.mean() if not class_counts.empty else 0

    if target_ratio == "4:1":
        target_total_0 = int(mean_count * 4)
    elif target_ratio == "1:1":
        target_total_0 = int(mean_count * 1)
    else:
        raise ValueError(f"지원하지 않는 target_ratio: {target_ratio}")

    # ── fallback 샘플링 ──────────────────────────────────────────────────
    hardneg_final, trans_final, neutral_final, safe_final = _apply_fallback_sampling(
        group_hardneg=group_b_hardneg,
        group_trans=c_transition,
        group_neutral=group_b_neutral,
        group_safe=c_safe,
        target_total=target_total_0,
    )

    # ── 결합 및 정렬 ─────────────────────────────────────────────────────
    final_df = pd.concat([group_a, hardneg_final, trans_final, neutral_final, safe_final])
    final_df = final_df.sort_index()

    return final_df