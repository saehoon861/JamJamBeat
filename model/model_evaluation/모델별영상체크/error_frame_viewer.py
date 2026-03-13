# error_frame_viewer.py - 예측 오류 프레임만 추출해 영상으로 저장하는 분석 스크립트
"""
학습된 모델 추론 결과와 ground truth 라벨을 프레임 단위로 비교,
틀린 프레임(+ 선택적으로 앞뒤 context)만 모아 오버레이 영상을 저장한다.

Usage (model/ 디렉토리 기준):
    # 단일 CSV
    uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
        --run-dir model_evaluation/pipelines/mlp_baseline/20260313_120557 \
        --csv data_fusion/man1_right_for_poc.csv

    # 4개 CSV 한번에
    uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
        --run-dir model_evaluation/pipelines/mlp_baseline/20260313_120557 \
        --csv data_fusion/man1_right_for_poc.csv \
        --csv data_fusion/man2_right_for_poc.csv \
        --csv data_fusion/man3_right_for_poc.csv \
        --csv data_fusion/woman1_right_for_poc.csv \
        --context-frames 5

    # latest.json 포인터 사용
    uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
        --run-dir model_evaluation/pipelines/mlp_baseline \
        --csv data_fusion/man1_right_for_poc.csv

    # 특정 source_file만 분석
    uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
        --run-dir model_evaluation/pipelines/mlp_baseline/20260313_120557 \
        --csv data_fusion/man1_right_for_poc.csv \
        --source-filter 3_fast_right_man1 3_slow_right_man1
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODEL_PIPELINES_ROOT = PROJECT_ROOT / "model" / "model_pipelines"
VIDEO_CHECK_DIR = Path(__file__).resolve().parent
RAW_VIDEO_ROOT = PROJECT_ROOT / "data" / "raw_data"

for _p in (str(MODEL_PIPELINES_ROOT), str(VIDEO_CHECK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import video_check_app as vca  # 모델 로딩 / 추론 / 랜드마크 함수 재사용

# ── 상수 ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["neutral", "fist", "open_palm", "V", "pinky", "animal", "k-heart"]

# BGR 색상
_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "neutral":    (180, 180, 180),
    "fist":       (0,   100, 255),
    "open_palm":  (0,   200, 100),
    "V":          (255, 180,   0),
    "pinky":      (255,   0, 200),
    "animal":     ( 50, 200, 255),
    "k-heart":    (  0,   0, 255),
}
_DEFAULT_COLOR: tuple[int, int, int] = (200, 200, 200)
_COLOR_ERROR   = (0,   0, 255)   # 빨강 — 실제 오류 프레임 테두리
_COLOR_CONTEXT = (0, 200, 255)   # 노랑 — context 프레임 테두리


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────
class ErrorFrame(NamedTuple):
    source_file: str
    frame_idx: int
    gt_idx: int
    pred_idx: int
    confidence: float
    probs: list[float]
    raw_landmarks: np.ndarray | None
    is_context: bool  # True = 주변 context 프레임, False = 실제 오류 프레임


# ── 유틸리티 ──────────────────────────────────────────────────────────────────
def _resolve_path(path_str: str) -> Path:
    """상대경로를 CWD → PROJECT_ROOT 순으로 탐색해 절대경로로 변환한다."""
    p = Path(path_str)
    if not p.is_absolute():
        cwd_cand = Path.cwd() / p
        p = cwd_cand if cwd_cand.exists() else PROJECT_ROOT / p
    return p.resolve()


def resolve_run_dir(path_str: str) -> Path:
    """latest.json 포인터 또는 실제 run 디렉토리를 model.pt 위치로 변환한다."""
    p = _resolve_path(path_str)

    # model 폴더(latest.json 있음) → 최신 run 경로로 이동
    latest = p / "latest.json"
    if latest.exists():
        data = json.loads(latest.read_text(encoding="utf-8"))
        p = Path(data["latest_run"]).resolve()

    # suite 구조 하위에서 탐색
    if not (p / "model.pt").exists():
        candidates = sorted(p.rglob("model.pt"), reverse=True)
        if not candidates:
            raise FileNotFoundError(f"model.pt not found under: {p}")
        p = candidates[0].parent

    return p


def load_gt(csv_paths: list[Path]) -> dict[str, dict[int, int]]:
    """CSV → {source_file: {frame_idx: gesture}} 매핑을 만든다."""
    import pandas as pd

    gt: dict[str, dict[int, int]] = {}
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        for sf, grp in df.groupby("source_file"):
            mapping = dict(zip(grp["frame_idx"].astype(int), grp["gesture"].astype(int)))
            gt.setdefault(str(sf), {}).update(mapping)
    return gt


def find_video(source_file: str, video_root: Path) -> Path | None:
    """source_file 이름과 매칭되는 영상 파일을 반환한다."""
    for ext in vca.SUPPORTED_VIDEO_EXTS:
        candidate = video_root / f"{source_file}{ext}"
        if candidate.exists():
            return candidate
    # 확장자 약간 다른 경우 fuzzy 탐색
    for p in video_root.iterdir():
        if p.stem == source_file and p.suffix.lower() in vca.SUPPORTED_VIDEO_EXTS:
            return p
    return None


# ── 오버레이 렌더링 ────────────────────────────────────────────────────────────
def draw_overlay(
    frame: np.ndarray,
    ef: ErrorFrame,
    fps: float,
    class_names: list[str],
) -> np.ndarray:
    """오류/context 프레임 위에 GT·예측·랜드마크 오버레이를 그린다."""
    out = frame.copy()
    h, w = out.shape[:2]

    # ① 랜드마크
    if ef.raw_landmarks is not None:
        out = vca.draw_raw_landmarks(out, ef.raw_landmarks)

    # ② 테두리 (오류=빨강, context=노랑)
    border_color = _COLOR_CONTEXT if ef.is_context else _COLOR_ERROR
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), border_color, 6)

    # ③ 상단 정보 바 (반투명)
    bar_h = 38
    overlay_bar = out[:bar_h].copy()
    cv2.rectangle(overlay_bar, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay_bar, 0.65, out[:bar_h], 0.35, 0, out[:bar_h])
    ts = vca.format_timestamp(ef.frame_idx, fps)
    info = f"{ef.source_file}  |  frame={ef.frame_idx}  {ts}"
    cv2.putText(out, info, (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    # ④ 하단 GT / PRED 텍스트
    gt_name   = class_names[ef.gt_idx]   if ef.gt_idx   < len(class_names) else str(ef.gt_idx)
    pred_name = class_names[ef.pred_idx] if ef.pred_idx < len(class_names) else str(ef.pred_idx)
    gt_color   = _LABEL_COLORS.get(gt_name,   _DEFAULT_COLOR)
    pred_color = (0, 220, 0) if ef.gt_idx == ef.pred_idx else (0, 0, 255)

    label_y = h - 14
    cv2.putText(out, f"GT: {gt_name}",
                (8, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, gt_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"PRED: {pred_name}  ({ef.confidence:.2f})",
                (w // 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pred_color, 2, cv2.LINE_AA)

    # ⑤ 우측 상단 확률 바
    bar_top = bar_h + 6
    bar_max_w = 110
    for i, prob in enumerate(ef.probs):
        cname  = class_names[i] if i < len(class_names) else str(i)
        color  = _LABEL_COLORS.get(cname, _DEFAULT_COLOR)
        bar_x  = w - bar_max_w - 8
        bar_y  = bar_top + i * 19
        bar_len = max(int(prob * bar_max_w), 0)
        cv2.rectangle(out, (bar_x, bar_y), (bar_x + bar_len, bar_y + 14), color, -1)
        label_text = f"{cname[:7]} {prob:.2f}"
        cv2.putText(out, label_text, (bar_x - 100, bar_y + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 220), 1, cv2.LINE_AA)

    return out


def make_title_card(
    source_file: str,
    n_errors: int,
    total_gt: int,
    frame_size: tuple[int, int],  # (w, h)
) -> np.ndarray:
    """source_file 전환 시 삽입하는 타이틀 카드를 만든다."""
    w, h = frame_size
    card = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(card, source_file, (40, h // 2 - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (200, 200, 200), 2, cv2.LINE_AA)
    rate = n_errors / max(total_gt, 1)
    summary = f"오류 {n_errors} / {total_gt} frames  ({rate:.1%})"
    cv2.putText(card, summary, (40, h // 2 + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 200, 255), 1, cv2.LINE_AA)
    return card


# ── 분석 로직 ─────────────────────────────────────────────────────────────────
def analyze_and_compare(
    runtime: vca.RuntimeModel,
    video_path: Path,
    gt_map: dict[int, int],
    context_frames: int,
) -> tuple[list[ErrorFrame], int, int]:
    """영상 추론 → GT 비교 → 오류 프레임 목록 반환."""
    analyzed = vca.analyze_video(runtime, video_path)
    result_map = {r.frame_idx: r for r in analyzed.frame_results}

    # 오류 frame_idx 수집
    error_indices: set[int] = set()
    for fidx, gt_gesture in gt_map.items():
        fr = result_map.get(fidx)
        if fr is None:
            continue
        pred = runtime.neutral_idx if fr.status in ("no_hand", "warmup") else fr.pred_idx
        if pred != gt_gesture:
            error_indices.add(fidx)

    # context 포함 대상 계산
    collect_indices: set[int] = set()
    for eidx in error_indices:
        for offset in range(-context_frames, context_frames + 1):
            cidx = eidx + offset
            if cidx in gt_map:
                collect_indices.add(cidx)

    error_frames: list[ErrorFrame] = []
    sf = video_path.stem
    for fidx in sorted(collect_indices):
        fr = result_map.get(fidx)
        if fr is None:
            continue
        pred_idx = runtime.neutral_idx if fr.status in ("no_hand", "warmup") else fr.pred_idx
        error_frames.append(ErrorFrame(
            source_file=sf,
            frame_idx=fidx,
            gt_idx=gt_map[fidx],
            pred_idx=pred_idx,
            confidence=fr.confidence,
            probs=fr.probs,
            raw_landmarks=fr.raw_landmarks,
            is_context=(fidx not in error_indices),
        ))

    n_comparable = sum(1 for fidx in gt_map if fidx in result_map)
    return error_frames, len(error_indices), n_comparable


def collect_frames_from_video(
    error_frames: list[ErrorFrame],
    video_path: Path,
) -> list[tuple[ErrorFrame, np.ndarray]]:
    """2차 패스: 오류 프레임 인덱스에 해당하는 실제 BGR 프레임을 추출한다."""
    target_map = {ef.frame_idx: ef for ef in error_frames}
    if not target_map:
        return []

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[warn] 영상 열기 실패: {video_path}")
        return []

    results: list[tuple[ErrorFrame, np.ndarray]] = []
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx in target_map:
            results.append((target_map[fidx], frame))
        fidx += 1

    cap.release()
    results.sort(key=lambda x: x[0].frame_idx)
    return results


# ── 출력 저장 ─────────────────────────────────────────────────────────────────
def write_error_video(
    segments: list[tuple[str, list[tuple[ErrorFrame, np.ndarray]], int, int]],
    output_path: Path,
    fps: float,
    class_names: list[str],
) -> None:
    """
    segments: [(source_file, [(ErrorFrame, bgr_frame), ...], n_errors, n_total), ...]
    """
    all_frames: list[tuple[ErrorFrame, np.ndarray]] = []
    for seg in segments:
        all_frames.extend(seg[1])

    if not all_frames:
        print("[render] 오류 프레임 없음 — 영상 저장 생략")
        return

    sample_h, sample_w = all_frames[0][1].shape[:2]
    frame_size = (sample_w, sample_h)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)

    for sf, rendered_list, n_errors, n_total in segments:
        if not rendered_list:
            continue
        # 타이틀 카드 (0.5초)
        title = make_title_card(sf, n_errors, n_total, frame_size)
        for _ in range(max(1, int(fps * 0.5))):
            writer.write(title)

        for ef, bgr in rendered_list:
            overlaid = draw_overlay(bgr, ef, fps, class_names)
            writer.write(overlaid)

    writer.release()
    print(f"[render] 저장: {output_path}  ({len(all_frames)} frames)")


def write_summary_csv(
    all_error_frames: list[ErrorFrame],
    output_dir: Path,
    class_names: list[str],
) -> None:
    summary_path = output_dir / "error_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "source_file", "frame_idx", "gt", "pred", "confidence", "is_context"
        ])
        writer.writeheader()
        for ef in all_error_frames:
            writer.writerow({
                "source_file": ef.source_file,
                "frame_idx":   ef.frame_idx,
                "gt":          class_names[ef.gt_idx]   if ef.gt_idx   < len(class_names) else ef.gt_idx,
                "pred":        class_names[ef.pred_idx] if ef.pred_idx < len(class_names) else ef.pred_idx,
                "confidence":  round(ef.confidence, 4),
                "is_context":  ef.is_context,
            })
    print(f"[summary] {summary_path}")


# ── main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="예측 오류 프레임만 추출해 영상으로 저장하는 분석 도구",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", required=True,
                        help="model.pt가 있는 run 폴더 또는 latest.json이 있는 model 폴더")
    parser.add_argument("--csv", action="append", dest="csv_paths", default=[], required=True,
                        help="ground truth CSV (source_file, frame_idx, gesture 필수). 반복 가능")
    parser.add_argument("--output-dir", default=None,
                        help="출력 폴더 (기본: run_dir/error_analysis/)")
    parser.add_argument("--video-root", default=None,
                        help="영상 폴더 (기본: data/raw_data/)")
    parser.add_argument("--context-frames", type=int, default=0,
                        help="오류 프레임 앞뒤로 포함할 context 수 (기본 0)")
    parser.add_argument("--source-filter", nargs="*", default=None,
                        help="분석할 source_file 이름 목록 (미지정 시 전체)")
    parser.add_argument("--fps", type=float, default=10.0,
                        help="출력 영상 FPS (기본 10 — 오류 프레임만 모아 보기 편한 속도)")
    args = parser.parse_args()

    # ── 경로 해석 ──
    run_dir    = resolve_run_dir(args.run_dir)
    video_root = Path(args.video_root).resolve() if args.video_root else RAW_VIDEO_ROOT
    output_dir = Path(args.output_dir).resolve() if args.output_dir else run_dir / "error_analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = [_resolve_path(p) for p in args.csv_paths]

    print(f"[load] run_dir : {run_dir}")
    print(f"[load] video   : {video_root}")
    print(f"[load] output  : {output_dir}")

    # ── 모델 로드 ──
    summary_path = run_dir / "run_summary.json"
    model_id = run_dir.parent.name
    mode     = "unknown"
    macro_f1 = None
    if summary_path.exists():
        summary  = json.loads(summary_path.read_text(encoding="utf-8"))
        model_id = str(summary.get("model_id") or model_id)
        mode     = str(summary.get("mode", "unknown"))
        macro_f1 = summary.get("metrics", {}).get("macro_avg", {}).get("f1")

    run_info = vca.RunInfo(
        model_id=model_id,
        run_dir=run_dir,
        checkpoint_path=run_dir / "model.pt",
        summary_path=summary_path if summary_path.exists() else None,
        mode=mode,
        macro_f1=float(macro_f1) if macro_f1 is not None else None,
        display_name=run_dir.name,
    )
    print(f"[load] model={model_id}  mode={mode}  macro_f1={macro_f1}")
    runtime     = vca.load_runtime_model(run_info)
    class_names = runtime.class_names

    # ── GT 로드 ──
    gt_all = load_gt(csv_paths)
    source_files = sorted(gt_all.keys())
    if args.source_filter:
        source_files = [s for s in source_files if s in set(args.source_filter)]
    print(f"[gt] {len(source_files)}개 source_file 분석 대상")

    # ── source_file별 처리 ──
    segments: list[tuple[str, list[tuple[ErrorFrame, np.ndarray]], int, int]] = []
    all_error_frames: list[ErrorFrame] = []
    total_gt_all = 0
    total_error_all = 0

    for sf in source_files:
        gt_map    = gt_all[sf]
        video_path = find_video(sf, video_root)
        if video_path is None:
            print(f"[skip] 영상 없음: {sf}")
            continue

        print(f"\n[analyze] {sf}  ({len(gt_map)} GT frames) ...")
        try:
            error_frames, n_errors, n_comparable = analyze_and_compare(
                runtime, video_path, gt_map, args.context_frames
            )
        except Exception as e:
            print(f"[error] {sf}: {e}")
            continue

        error_rate = n_errors / max(n_comparable, 1)
        print(f"  오류 {n_errors} / {n_comparable}  ({error_rate:.1%})")

        all_error_frames.extend(error_frames)
        total_gt_all    += n_comparable
        total_error_all += n_errors

        if error_frames:
            rendered = collect_frames_from_video(error_frames, video_path)
        else:
            rendered = []

        segments.append((sf, rendered, n_errors, n_comparable))

    # ── 영상 저장 ──
    output_path = output_dir / f"error_frames_{model_id}.mp4"
    write_error_video(segments, output_path, args.fps, class_names)

    # ── 요약 CSV ──
    if all_error_frames:
        write_summary_csv(all_error_frames, output_dir, class_names)

    # ── 최종 요약 출력 ──
    actual_errors = [ef for ef in all_error_frames if not ef.is_context]
    print(f"\n{'='*50}")
    print(f"전체 오류  : {total_error_all} / {total_gt_all}  ({total_error_all/max(total_gt_all,1):.1%})")
    print(f"출력 영상  : {output_path}")
    print(f"요약 CSV   : {output_dir / 'error_summary.csv'}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
