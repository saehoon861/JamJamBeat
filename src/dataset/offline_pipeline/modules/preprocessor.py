import pandas as pd
import numpy as np
from typing import Optional

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


def _collect_padding_for_gesture_segments(
    df_original   : pd.DataFrame,
    group_c       : pd.DataFrame,
    pad_size      : int,
    target_trans  : int,
) -> pd.DataFrame:
    """
    시퀀스 모드 전용.
    gesture 1~6 구간의 시작/끝 frame_idx 기준으로
    앞뒤 pad_size개의 gesture==0 프레임을 수집.

    총량이 target_trans 초과하면 pad_size를 1씩 줄여서 재시도.
    gesture!=0 프레임은 skip (이미 group_a로 분리됨).
    """
    group_c_index = set(group_c.index)

    def _collect_with_pad(pad: int) -> set:
        collect_indices = set()

        for _, file_group in df_original.groupby('source_file'):
            file_group = file_group.sort_values('frame_idx')
            frame_idxs = file_group['frame_idx'].values
            gestures   = file_group['gesture'].values
            orig_index = file_group.index.values

            fidx_to_dfidx   = dict(zip(frame_idxs, orig_index))
            fidx_to_gesture = dict(zip(frame_idxs, gestures))

            # gesture 1~6 구간의 시작/끝 frame_idx 탐지
            in_gesture = False
            seg_start  = None

            for i, (fidx, gest) in enumerate(zip(frame_idxs, gestures)):
                if not in_gesture and gest != 0:
                    # 구간 시작: 앞쪽 pad 수집
                    in_gesture = True
                    seg_start  = fidx
                    for prev_fidx in frame_idxs:
                        if seg_start - pad <= prev_fidx < seg_start:
                            dfidx = fidx_to_dfidx[prev_fidx]
                            # gesture==0이고 group_c에 속한 프레임만
                            if fidx_to_gesture[prev_fidx] == 0 and dfidx in group_c_index:
                                collect_indices.add(dfidx)

                elif in_gesture and gest == 0:
                    # 구간 끝: 뒤쪽 pad 수집
                    in_gesture = False
                    seg_end    = frame_idxs[i - 1]  # 마지막 gesture!=0 frame_idx
                    for next_fidx in frame_idxs:
                        if seg_end < next_fidx <= seg_end + pad:
                            dfidx = fidx_to_dfidx[next_fidx]
                            if fidx_to_gesture[next_fidx] == 0 and dfidx in group_c_index:
                                collect_indices.add(dfidx)

            # 파일이 gesture!=0으로 끝나는 경우 뒤쪽 pad 처리
            if in_gesture:
                seg_end = frame_idxs[-1]
                for next_fidx in frame_idxs:
                    if seg_end < next_fidx <= seg_end + pad:
                        dfidx = fidx_to_dfidx[next_fidx]
                        if fidx_to_gesture[next_fidx] == 0 and dfidx in group_c_index:
                            collect_indices.add(dfidx)

        return collect_indices

    # pad_size부터 1씩 줄이며 target_trans 이하가 될 때까지 재시도
    for pad in range(pad_size, 0, -1):
        collect_indices = _collect_with_pad(pad)
        if len(collect_indices) <= target_trans:
            break

    return group_c[group_c.index.isin(collect_indices)]


def _classify_group_c_frame_mode(
    df_original   : pd.DataFrame,
    group_c       : pd.DataFrame,
    margin_drop   : int,
    margin_collect: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    프레임 단위 모드 전용.
    전환점 ± margin_drop 프레임 제거,
    전환점 ± (margin_drop+1 ~ margin_collect) 이내 gesture==0 → C_transition,
    나머지 → C_safe.
    """
    collect_indices = set()
    drop_indices    = set()

    for _, file_group in df_original.groupby('source_file'):
        file_group = file_group.sort_values('frame_idx')
        frame_idxs = file_group['frame_idx'].values
        gestures   = file_group['gesture'].values
        orig_index = file_group.index.values

        fidx_to_dfidx   = dict(zip(frame_idxs, orig_index))
        fidx_to_gesture = dict(zip(frame_idxs, gestures))

        # 전환점 탐지
        transition_frame_idxs = [
            frame_idxs[i]
            for i in range(len(gestures) - 1)
            if gestures[i] != gestures[i + 1]
        ]

        # drop 구간
        file_drop_dfidx = set()
        for t_fidx in transition_frame_idxs:
            for fidx, dfidx in fidx_to_dfidx.items():
                if abs(fidx - t_fidx) <= margin_drop:
                    file_drop_dfidx.add(dfidx)
        drop_indices.update(file_drop_dfidx)

        # collect 구간
        for t_fidx in transition_frame_idxs:
            for fidx, dfidx in fidx_to_dfidx.items():
                dist = abs(fidx - t_fidx)
                if margin_drop < dist <= margin_collect:
                    if dfidx in file_drop_dfidx:
                        continue
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

    class_counts  = df['_src_class'].value_counts()
    total         = len(df)
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

    return pd.concat(sampled_parts).drop(columns=['_src_class'])


def _apply_fallback_sampling(
    group_hardneg: pd.DataFrame,
    group_trans  : pd.DataFrame,
    group_neutral: pd.DataFrame,
    group_safe   : pd.DataFrame,
    target_total : int,
    random_state : int = 42,
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
    margin_drop   : Optional[int] = None,
    margin_collect: int = 8,
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
        margin_drop   : None  → 시퀀스 모드, gesture 구간 앞뒤 패딩 수집
                        int   → 프레임 단위 모드, 전환점 ± margin_drop 제거
        margin_collect: 시퀀스 모드: pad_size 시작값 (= 윈도우 크기, default 8)
                        프레임 모드: collect 최대 프레임 수
    """
    if target_ratio == "origin":
        return df.copy()

    # ── 전처리: 랜드마크 전부 null 프레임 제거 ──────────────────────────
    df_for_transition = df.copy()
    df = _remove_null_landmarks(df)

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

    # ── target 계산 ──────────────────────────────────────────────────────
    class_counts = group_a['gesture'].value_counts()
    mean_count   = class_counts.mean() if not class_counts.empty else 0

    if target_ratio == "4:1":
        target_total_0 = int(mean_count * 4)
    elif target_ratio == "1:1":
        target_total_0 = int(mean_count * 1)
    else:
        raise ValueError(f"지원하지 않는 target_ratio: {target_ratio}")

    target_trans = int(target_total_0 * RATIO_TRANS)

    # ── Group C → C_transition / C_safe 분리 ────────────────────────────
    if margin_drop is None:
        # 시퀀스 모드: gesture 구간 앞뒤 패딩 수집, 총량 초과 시 pad_size 줄임
        c_transition = _collect_padding_for_gesture_segments(
            df_original=df_for_transition,
            group_c=group_c,
            pad_size=margin_collect,
            target_trans=target_trans,
        )
        c_safe = group_c[~group_c.index.isin(c_transition.index)]
    else:
        # 프레임 단위 모드: 전환점 기준 drop/collect
        c_transition, c_safe = _classify_group_c_frame_mode(
            df_original=df_for_transition,
            group_c=group_c,
            margin_drop=margin_drop,
            margin_collect=margin_collect,
        )

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