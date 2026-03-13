# error_frame_viewer.py - 예측 오류 프레임 분석 뷰어 (UI + CLI)
"""
UI 모드  : 인자 없이 실행하면 Tk 드롭다운 UI가 열린다.
CLI 모드 : --run-dir / --csv 를 모두 지정하면 바로 분석·저장한다.

UI 사용법 (model/ 디렉토리 기준):
    uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py"

CLI 사용법:
    uv run python "model_evaluation/모델별영상체크/error_frame_viewer.py" \
        --run-dir model_evaluation/pipelines/mlp_baseline/20260313_120557 \
        --csv data_fusion/man1_right_for_poc.csv \
        --context-frames 5

재생 컨트롤 (OpenCV 창):
    Space   : 재생 / 일시정지
    A / ←   : 이전 오류 프레임
    D / →   : 다음 오류 프레임
    R       : 처음으로
    Q / Esc : 종료
"""
from __future__ import annotations

import argparse
import csv as csv_module
import json
import sys
import threading
from pathlib import Path
from typing import NamedTuple

import cv2
import numpy as np

# ── 경로 설정 ─────────────────────────────────────────────────────────────────
PROJECT_ROOT        = Path(__file__).resolve().parents[3]
MODEL_PIPELINES_ROOT = PROJECT_ROOT / "model" / "model_pipelines"
VIDEO_CHECK_DIR     = Path(__file__).resolve().parent
RAW_VIDEO_ROOT      = PROJECT_ROOT / "data" / "raw_data"
DATA_FUSION_ROOT    = PROJECT_ROOT / "model" / "data_fusion"

for _p in (str(MODEL_PIPELINES_ROOT), str(VIDEO_CHECK_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import video_check_app as vca  # 모델 로딩 / 추론 / 랜드마크 함수 재사용

# ── 상수 ──────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["neutral", "fist", "open_palm", "V", "pinky", "animal", "k-heart"]

_LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "neutral":   (180, 180, 180),
    "fist":      (0,   100, 255),
    "open_palm": (0,   200, 100),
    "V":         (255, 180,   0),
    "pinky":     (255,   0, 200),
    "animal":    ( 50, 200, 255),
    "k-heart":   (  0,   0, 255),
}
_DEFAULT_COLOR = (200, 200, 200)
_COLOR_ERROR   = (0,   0, 255)   # 빨강 — 실제 오류 프레임 테두리
_COLOR_CONTEXT = (0, 200, 255)   # 노랑 — context 프레임 테두리

KEY_LEFT  = 65361
KEY_RIGHT = 65363

SOURCE_FILE_ALL = "[ All ]"


# ── 데이터 클래스 ──────────────────────────────────────────────────────────────
class ErrorFrame(NamedTuple):
    source_file: str
    frame_idx:   int
    gt_idx:      int
    pred_idx:    int
    confidence:  float
    probs:       list[float]
    raw_landmarks: np.ndarray | None
    is_context:  bool


# ── 경로 유틸 ─────────────────────────────────────────────────────────────────
def _resolve(path_str: str) -> Path:
    p = Path(path_str)
    if not p.is_absolute():
        cwd_cand = Path.cwd() / p
        p = cwd_cand if cwd_cand.exists() else PROJECT_ROOT / p
    return p.resolve()


def resolve_run_dir(path_str: str) -> Path:
    p = _resolve(path_str)
    latest = p / "latest.json"
    if latest.exists():
        data = json.loads(latest.read_text(encoding="utf-8"))
        p = Path(data["latest_run"]).resolve()
    if not (p / "model.pt").exists():
        candidates = sorted(p.rglob("model.pt"), reverse=True)
        if not candidates:
            raise FileNotFoundError(f"model.pt not found under: {p}")
        p = candidates[0].parent
    return p


# ── 탐색 ──────────────────────────────────────────────────────────────────────
_GT_CSV_NAMES = [
    "man1_right_for_poc.csv",
    "man2_right_for_poc.csv",
    "man3_right_for_poc.csv",
    "woman1_right_for_poc.csv",
]


def discover_gt_csvs() -> list[Path]:
    """ground truth CSV 4개만 반환한다. 존재하는 파일만 포함."""
    result: list[Path] = []
    for name in _GT_CSV_NAMES:
        p = DATA_FUSION_ROOT / name
        if p.exists():
            result.append(p)
    return result


def load_source_files(csv_path: Path) -> list[str]:
    """CSV에서 고유 source_file 목록을 반환한다."""
    import pandas as pd
    df = pd.read_csv(csv_path, usecols=["source_file"])
    return sorted(df["source_file"].unique())


def load_gt(csv_path: Path, source_filter: set[str] | None = None) -> dict[str, dict[int, int]]:
    """CSV → {source_file: {frame_idx: gesture}} 매핑."""
    import pandas as pd
    df = pd.read_csv(csv_path)
    if source_filter:
        df = df[df["source_file"].isin(source_filter)]
    gt: dict[str, dict[int, int]] = {}
    for sf, grp in df.groupby("source_file"):
        gt[str(sf)] = dict(zip(grp["frame_idx"].astype(int), grp["gesture"].astype(int)))
    return gt


def find_video(source_file: str) -> Path | None:
    for ext in vca.SUPPORTED_VIDEO_EXTS:
        p = RAW_VIDEO_ROOT / f"{source_file}{ext}"
        if p.exists():
            return p
    for p in RAW_VIDEO_ROOT.iterdir():
        if p.stem == source_file and p.suffix.lower() in vca.SUPPORTED_VIDEO_EXTS:
            return p
    return None


def build_run_info(run_dir: Path) -> vca.RunInfo:
    summary_path = run_dir / "run_summary.json"
    model_id = run_dir.parent.name
    mode     = "unknown"
    macro_f1 = None
    if summary_path.exists():
        s = json.loads(summary_path.read_text(encoding="utf-8"))
        model_id = str(s.get("model_id") or model_id)
        mode     = str(s.get("mode", "unknown"))
        macro_f1 = s.get("metrics", {}).get("macro_avg", {}).get("f1")
    return vca.RunInfo(
        model_id=model_id,
        run_dir=run_dir,
        checkpoint_path=run_dir / "model.pt",
        summary_path=summary_path if summary_path.exists() else None,
        mode=mode,
        macro_f1=float(macro_f1) if macro_f1 is not None else None,
        display_name=run_dir.name,
    )


# ── 오버레이 렌더링 ────────────────────────────────────────────────────────────
def draw_overlay(frame: np.ndarray, ef: ErrorFrame, fps: float, class_names: list[str]) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    if ef.raw_landmarks is not None:
        out = vca.draw_raw_landmarks(out, ef.raw_landmarks)

    border_color = _COLOR_CONTEXT if ef.is_context else _COLOR_ERROR
    cv2.rectangle(out, (0, 0), (w - 1, h - 1), border_color, 6)

    # 상단 정보 바
    bar_h = 38
    bar_overlay = out[:bar_h].copy()
    cv2.rectangle(bar_overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(bar_overlay, 0.65, out[:bar_h], 0.35, 0, out[:bar_h])
    ts = vca.format_timestamp(ef.frame_idx, fps)
    cv2.putText(out, f"{ef.source_file}  |  frame={ef.frame_idx}  {ts}",
                (8, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    # 하단 GT / PRED
    gt_name   = class_names[ef.gt_idx]   if ef.gt_idx   < len(class_names) else str(ef.gt_idx)
    pred_name = class_names[ef.pred_idx] if ef.pred_idx < len(class_names) else str(ef.pred_idx)
    gt_color   = _LABEL_COLORS.get(gt_name, _DEFAULT_COLOR)
    pred_color = (0, 220, 0) if ef.gt_idx == ef.pred_idx else (0, 0, 255)
    label_y = h - 14
    cv2.putText(out, f"GT: {gt_name}",
                (8, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, gt_color, 2, cv2.LINE_AA)
    cv2.putText(out, f"PRED: {pred_name}  ({ef.confidence:.2f})",
                (w // 2, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, pred_color, 2, cv2.LINE_AA)

    # 우측 상단 확률 바
    bar_top, bar_max_w = bar_h + 6, 110
    for i, prob in enumerate(ef.probs):
        cname  = class_names[i] if i < len(class_names) else str(i)
        color  = _LABEL_COLORS.get(cname, _DEFAULT_COLOR)
        bx     = w - bar_max_w - 8
        by     = bar_top + i * 19
        cv2.rectangle(out, (bx, by), (bx + max(int(prob * bar_max_w), 0), by + 14), color, -1)
        cv2.putText(out, f"{cname[:7]} {prob:.2f}", (bx - 100, by + 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.37, (220, 220, 220), 1, cv2.LINE_AA)

    return out


# ── 분석 로직 ─────────────────────────────────────────────────────────────────
def analyze_source(
    runtime: vca.RuntimeModel,
    video_path: Path,
    gt_map: dict[int, int],
    context_frames: int,
    progress_cb=None,
) -> tuple[list[ErrorFrame], int, int]:
    """1개 source_file에 대해 추론 → GT 비교 → 오류 프레임 목록 반환."""
    analyzed = vca.analyze_video(runtime, video_path)
    result_map = {r.frame_idx: r for r in analyzed.frame_results}

    error_indices: set[int] = set()
    for fidx, gt in gt_map.items():
        fr = result_map.get(fidx)
        if fr is None:
            continue
        pred = runtime.neutral_idx if fr.status in ("no_hand", "warmup") else fr.pred_idx
        if pred != gt:
            error_indices.add(fidx)

    collect: set[int] = set()
    for eidx in error_indices:
        for off in range(-context_frames, context_frames + 1):
            if eidx + off in gt_map:
                collect.add(eidx + off)

    sf = video_path.stem
    error_frames: list[ErrorFrame] = []
    for fidx in sorted(collect):
        fr = result_map.get(fidx)
        if fr is None:
            continue
        pred_idx = runtime.neutral_idx if fr.status in ("no_hand", "warmup") else fr.pred_idx
        error_frames.append(ErrorFrame(
            source_file=sf, frame_idx=fidx, gt_idx=gt_map[fidx],
            pred_idx=pred_idx, confidence=fr.confidence, probs=fr.probs,
            raw_landmarks=fr.raw_landmarks, is_context=(fidx not in error_indices),
        ))

    n_comparable = sum(1 for f in gt_map if f in result_map)
    return error_frames, len(error_indices), n_comparable


def fetch_bgr_frames(
    error_frames: list[ErrorFrame],
    video_path: Path,
) -> list[tuple[ErrorFrame, np.ndarray]]:
    """2차 패스: 오류 프레임 인덱스에 해당하는 BGR 프레임을 추출한다."""
    target = {ef.frame_idx: ef for ef in error_frames}
    if not target:
        return []
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return []
    results: list[tuple[ErrorFrame, np.ndarray]] = []
    fidx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if fidx in target:
            results.append((target[fidx], frame))
        fidx += 1
    cap.release()
    results.sort(key=lambda x: x[0].frame_idx)
    return results


# ── OpenCV 오류 프레임 재생 ────────────────────────────────────────────────────
def playback_errors(
    data: list[tuple[ErrorFrame, np.ndarray]],
    fps: float,
    class_names: list[str],
    title: str = "JamJamBeat Error Viewer",
) -> None:
    """오류 프레임 목록을 키보드로 탐색 가능한 OpenCV 창으로 재생한다."""
    if not data:
        return

    total   = len(data)
    current = 0
    paused  = True
    delay   = max(int(1000 / max(fps, 1e-6)), 1)
    window_created = False

    while True:
        if window_created and cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break

        ef, bgr = data[current]
        frame = draw_overlay(bgr, ef, fps, class_names)
        h, w = frame.shape[:2]

        tag   = "ERROR" if not ef.is_context else "context"
        guide = (f"[{current + 1}/{total}] {tag}  |  "
                 "Space: 재생/정지  A/←: 이전  D/→: 다음  R: 처음  Q: 종료")
        cv2.putText(frame, guide, (8, h - 8),
                    cv2.FONT_HERSHEY_PLAIN, 1.1, (200, 200, 200), 1, cv2.LINE_AA)

        cv2.imshow(title, frame)
        window_created = True

        key = cv2.waitKeyEx(0 if paused else delay)
        if key in (-1, 255):
            if not paused:
                current = (current + 1) % total
            continue

        key_low = key & 0xFF
        if key_low in (ord("q"), 27):
            break
        elif key_low == ord(" "):
            paused = not paused
        elif key_low == ord("r"):
            current = 0
            paused = True
        elif key_low == ord("a") or key == KEY_LEFT:
            current = max(0, current - 1)
            paused = True
        elif key_low == ord("d") or key == KEY_RIGHT:
            current = min(total - 1, current + 1)
            paused = True

    cv2.destroyWindow(title)
    cv2.waitKey(1)


# ── MP4 내보내기 ───────────────────────────────────────────────────────────────
def export_mp4(
    data: list[tuple[ErrorFrame, np.ndarray]],
    output_path: Path,
    fps: float,
    class_names: list[str],
) -> None:
    if not data:
        return
    h, w = data[0][1].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for ef, bgr in data:
        writer.write(draw_overlay(bgr, ef, fps, class_names))
    writer.release()


def export_summary_csv(error_frames: list[ErrorFrame], output_path: Path, class_names: list[str]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv_module.DictWriter(f, fieldnames=[
            "source_file", "frame_idx", "gt", "pred", "confidence", "is_context"
        ])
        writer.writeheader()
        for ef in error_frames:
            writer.writerow({
                "source_file": ef.source_file,
                "frame_idx":   ef.frame_idx,
                "gt":   class_names[ef.gt_idx]   if ef.gt_idx   < len(class_names) else ef.gt_idx,
                "pred": class_names[ef.pred_idx] if ef.pred_idx < len(class_names) else ef.pred_idx,
                "confidence":  round(ef.confidence, 4),
                "is_context":  ef.is_context,
            })


# ── Tk UI ─────────────────────────────────────────────────────────────────────
class ErrorFrameApp:
    """Run / CSV / source_file 드롭다운으로 오류 프레임을 분석·재생하는 Tk UI."""

    def __init__(self, root):
        import tkinter as tk
        from tkinter import ttk, messagebox

        self._tk = tk
        self._ttk = ttk
        self._mb = messagebox

        self.root = root
        self.root.title("JamJamBeat Error Frame Viewer")
        self.root.geometry("1020x340")

        # 상태 변수
        self.run_var        = tk.StringVar()
        self.csv_var        = tk.StringVar()
        self.src_var        = tk.StringVar()
        self.context_var    = tk.StringVar(value="0")
        self.status_var     = tk.StringVar(value="Ready")
        self.info_var       = tk.StringVar(value="Run과 CSV를 선택 후 [Analyze & View]를 누르세요.")

        # 캐시
        self._run_lookup:    dict[str, vca.RunInfo] = {}
        self._csv_lookup:    dict[str, Path]        = {}
        self._runtime_cache: dict[str, vca.RuntimeModel] = {}
        # (run_dir, csv_path, source_file, context) → (data, fps, error_frames)
        self._analysis_cache: dict[tuple, tuple] = {}
        self._last_data:     list[tuple[ErrorFrame, np.ndarray]] = []
        self._last_fps:      float = 10.0
        self._last_error_frames: list[ErrorFrame] = []
        self._last_class_names:  list[str] = []
        self._last_run_dir:      Path | None = None

        self._build_ui()
        self.refresh_options()

    # ── UI 구성 ────────────────────────────────────────────────────────────────
    def _build_ui(self) -> None:
        tk  = self._tk
        ttk = self._ttk

        # WSL2/Linux에서 CJK 폰트가 없으면 한글이 깨지므로 사용 가능한 폰트로 fallback
        try:
            import tkinter.font as tkfont
            default_font = tkfont.nametofont("TkDefaultFont")
            for candidate in ("Noto Sans CJK KR", "NanumGothic", "Malgun Gothic",
                              "WenQuanYi Micro Hei", "DejaVu Sans"):
                try:
                    default_font.configure(family=candidate, size=10)
                    self.root.option_add("*Font", default_font)
                    break
                except Exception:
                    pass
        except Exception:
            pass

        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill=tk.BOTH, expand=True)

        # Row 0-1: Trained Run
        ttk.Label(frame, text="Trained Run").grid(row=0, column=0, columnspan=3, sticky="w")
        self.run_combo = ttk.Combobox(frame, textvariable=self.run_var, state="readonly", width=110)
        self.run_combo.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(2, 10))
        self.run_combo.bind("<<ComboboxSelected>>", lambda _: self._on_run_select())

        # Row 2-3: CSV + source_file
        ttk.Label(frame, text="Ground Truth CSV").grid(row=2, column=0, sticky="w", padx=(0, 8))
        ttk.Label(frame, text="Source File").grid(row=2, column=1, sticky="w", padx=(0, 8))
        ttk.Label(frame, text="Context Frames").grid(row=2, column=2, sticky="w")

        self.csv_combo = ttk.Combobox(frame, textvariable=self.csv_var, state="readonly", width=55)
        self.csv_combo.grid(row=3, column=0, sticky="ew", padx=(0, 8), pady=(2, 10))
        self.csv_combo.bind("<<ComboboxSelected>>", lambda _: self._on_csv_select())

        self.src_combo = ttk.Combobox(frame, textvariable=self.src_var, state="readonly", width=40)
        self.src_combo.grid(row=3, column=1, sticky="ew", padx=(0, 8), pady=(2, 10))

        self.ctx_entry = ttk.Spinbox(frame, textvariable=self.context_var,
                                     from_=0, to=30, width=8)
        self.ctx_entry.grid(row=3, column=2, sticky="w", pady=(2, 10))

        # Row 4: 버튼
        btn_frame = ttk.Frame(frame)
        btn_frame.grid(row=4, column=0, columnspan=3, sticky="w", pady=(0, 10))
        ttk.Button(btn_frame, text="Refresh",         command=self.refresh_options).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="Analyze & View",  command=self.on_analyze).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="Export MP4",      command=self.on_export).pack(side=tk.LEFT, padx=(0, 6))
        ttk.Button(btn_frame, text="Quit",            command=self.root.destroy).pack(side=tk.LEFT)

        # Row 5-6: 정보 / 상태
        ttk.Label(frame, textvariable=self.info_var, justify=tk.LEFT).grid(
            row=5, column=0, columnspan=3, sticky="w")
        ttk.Label(frame, textvariable=self.status_var, foreground="#005f99").grid(
            row=6, column=0, columnspan=3, sticky="w", pady=(10, 0))

        frame.columnconfigure(0, weight=3)
        frame.columnconfigure(1, weight=2)
        frame.columnconfigure(2, weight=0)

    # ── 드롭다운 갱신 ──────────────────────────────────────────────────────────
    def refresh_options(self) -> None:
        runs = vca.discover_runs()
        csvs = discover_gt_csvs()

        self._run_lookup = {r.display_name: r for r in runs}
        self._csv_lookup = {p.name: p for p in csvs}

        self.run_combo["values"] = list(self._run_lookup.keys())
        self.csv_combo["values"] = list(self._csv_lookup.keys())

        if runs and not self.run_var.get():
            self.run_var.set(runs[0].display_name)
        if csvs and not self.csv_var.get():
            self.csv_var.set(csvs[0].name)
            self._populate_source_files(csvs[0])

        self.status_var.set(f"Runs: {len(runs)} | CSVs: {len(csvs)}")
        self._update_info()

    def _on_run_select(self) -> None:
        self._update_info()

    def _on_csv_select(self) -> None:
        csv_path = self._csv_lookup.get(self.csv_var.get())
        if csv_path:
            self._populate_source_files(csv_path)
        self._update_info()

    def _populate_source_files(self, csv_path: Path) -> None:
        try:
            sfs = load_source_files(csv_path)
            # 영상 존재 여부 표시: 없으면 "[no video]" 접미어
            labeled = []
            for sf in sfs:
                v = find_video(sf)
                labeled.append(sf if v else f"{sf}  [no video]")
            values = [SOURCE_FILE_ALL] + labeled
            self.src_combo["values"] = values
            self.src_var.set(SOURCE_FILE_ALL)
        except Exception as e:
            self.src_combo["values"] = [SOURCE_FILE_ALL]
            self.src_var.set(SOURCE_FILE_ALL)
            self.status_var.set(f"CSV read error: {e}")

    def _update_info(self) -> None:
        run  = self._run_lookup.get(self.run_var.get())
        csv_path = self._csv_lookup.get(self.csv_var.get())
        lines = []
        if run:
            lines.append(f"Run  : {run.run_dir}")
            lines.append(f"Model: {run.model_id}  mode={run.mode}  macro_f1={run.macro_f1 or '-'}")
        if csv_path:
            lines.append(f"CSV  : {csv_path}")
        if not vca.TASK_MODEL_PATH.exists():
            lines.append(f"[경고] hand_landmarker.task 없음: {vca.TASK_MODEL_PATH}")
        self.info_var.set("\n".join(lines) if lines else "Run과 CSV를 선택하세요.")

    # ── 런타임 모델 캐시 ───────────────────────────────────────────────────────
    def _get_runtime(self, run_info: vca.RunInfo) -> vca.RuntimeModel:
        key = str(run_info.run_dir)
        if key not in self._runtime_cache:
            self.status_var.set("모델 로딩 중...")
            self.root.update_idletasks()
            self._runtime_cache[key] = vca.load_runtime_model(run_info)
        return self._runtime_cache[key]

    # ── 분석 실행 ──────────────────────────────────────────────────────────────
    def _run_analysis(self) -> bool:
        """선택된 조건으로 분석을 실행하고 캐시에 저장한다. 성공 시 True."""
        run_info = self._run_lookup.get(self.run_var.get())
        csv_path = self._csv_lookup.get(self.csv_var.get())
        if not run_info:
            self._mb.showerror("Error", "Select a Trained Run first.")
            return False
        if not csv_path:
            self._mb.showerror("Error", "Select a Ground Truth CSV first.")
            return False

        sf_sel_raw = self.src_var.get()
        # "[no video]" 접미어 제거 후 실제 source_file 이름 추출
        sf_sel = sf_sel_raw.replace("  [no video]", "").strip()
        source_filter = None if sf_sel_raw == SOURCE_FILE_ALL else {sf_sel}
        try:
            ctx = int(self.context_var.get())
        except ValueError:
            ctx = 0

        cache_key = (str(run_info.run_dir), str(csv_path), sf_sel_raw, ctx)
        if cache_key in self._analysis_cache:
            self._last_data, self._last_fps, self._last_error_frames, self._last_class_names = \
                self._analysis_cache[cache_key]
            self._last_run_dir = run_info.run_dir
            self.status_var.set(f"캐시 사용 — 오류 {sum(1 for e in self._last_error_frames if not e.is_context)}프레임")
            return True

        try:
            runtime = self._get_runtime(run_info)
        except Exception as e:
            self._mb.showerror("Model Load Error", str(e))
            return False

        gt_all = load_gt(csv_path, source_filter)
        if not gt_all:
            self._mb.showwarning("No Data", "No GT data found for the selected source_file.")
            return False

        all_error_frames: list[ErrorFrame] = []
        all_rendered:     list[tuple[ErrorFrame, np.ndarray]] = []
        fps_found = 10.0
        total_errors = 0
        total_gt = 0

        missing_videos: list[str] = []
        for i, (sf, gt_map) in enumerate(gt_all.items()):
            self.status_var.set(f"[{i+1}/{len(gt_all)}] Analyzing: {sf} ...")
            self.root.update_idletasks()

            video_path = find_video(sf)
            if video_path is None:
                missing_videos.append(sf)
                continue

            try:
                error_frames, n_err, n_comp = analyze_source(runtime, video_path, gt_map, ctx)
            except Exception as e:
                self.status_var.set(f"Error: {sf}: {e}")
                continue

            cap = cv2.VideoCapture(str(video_path))
            fps_found = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
            cap.release()

            all_error_frames.extend(error_frames)
            total_errors += n_err
            total_gt     += n_comp

            if error_frames:
                rendered = fetch_bgr_frames(error_frames, video_path)
                all_rendered.extend(rendered)

        self._last_data         = all_rendered
        self._last_fps          = fps_found
        self._last_error_frames = all_error_frames
        self._last_class_names  = runtime.class_names
        self._last_run_dir      = run_info.run_dir
        self._analysis_cache[cache_key] = (all_rendered, fps_found, all_error_frames, runtime.class_names)

        # 영상 없는 source_file 경고
        if missing_videos:
            names = ", ".join(missing_videos)
            self._mb.showwarning(
                "Video Not Found",
                f"No video file found for:\n{names}\n\n"
                f"Expected location: {RAW_VIDEO_ROOT}",
            )
            if not all_rendered:
                self.status_var.set(f"Skipped all — no video found: {names}")
                return True  # _last_data=[] → on_analyze에서 처리

        actual = sum(1 for e in all_error_frames if not e.is_context)
        self.status_var.set(
            f"Done — {actual} / {total_gt} error frames  "
            f"({actual/max(total_gt,1):.1%})  |  "
            f"Total frames (incl. context): {len(all_rendered)}"
        )
        return True

    # ── 버튼 핸들러 ────────────────────────────────────────────────────────────
    def on_analyze(self) -> None:
        if not self._run_analysis():
            return
        if not self._last_data:
            self._mb.showinfo("No Error Frames", "No error frames found.\nThe model predicted all frames correctly (or videos were not found).")
            return

        self.root.withdraw()
        try:
            playback_errors(
                self._last_data,
                self._last_fps,
                self._last_class_names,
                title="JamJamBeat Error Viewer",
            )
        finally:
            self.root.deiconify()
            self.root.update()
            self.root.lift()
            self.root.focus_force()
            self.status_var.set("재생 종료")

    def on_export(self) -> None:
        if not self._last_data:
            if not self._run_analysis():
                return
        if not self._last_data:
            self._mb.showinfo("No Data", "No error frames to export. Run analysis first.")
            return

        run_info = self._run_lookup.get(self.run_var.get())
        if run_info is None:
            return

        out_dir   = run_info.run_dir / "error_analysis"
        out_dir.mkdir(parents=True, exist_ok=True)
        mp4_path  = out_dir / f"error_frames_{run_info.model_id}.mp4"
        csv_path2 = out_dir / "error_summary.csv"

        self.status_var.set("MP4 저장 중...")
        self.root.update_idletasks()
        export_mp4(self._last_data, mp4_path, self._last_fps, self._last_class_names)
        export_summary_csv(self._last_error_frames, csv_path2, self._last_class_names)
        self.status_var.set(f"Saved: {mp4_path.name}")
        self._mb.showinfo("Saved", f"Video: {mp4_path}\nSummary: {csv_path2}")


# ── CLI 모드 ──────────────────────────────────────────────────────────────────
def run_cli(args: argparse.Namespace) -> None:
    run_dir  = resolve_run_dir(args.run_dir)
    csv_paths = [_resolve(p) for p in args.csv_paths]
    run_info  = build_run_info(run_dir)

    print(f"[load] {run_info.model_id}  mode={run_info.mode}")
    runtime = vca.load_runtime_model(run_info)

    gt_all: dict[str, dict[int, int]] = {}
    for cp in csv_paths:
        for sf, gmap in load_gt(cp).items():
            gt_all.setdefault(sf, {}).update(gmap)
    if args.source_filter:
        gt_all = {k: v for k, v in gt_all.items() if k in set(args.source_filter)}

    all_error_frames: list[ErrorFrame] = []
    all_rendered:     list[tuple[ErrorFrame, np.ndarray]] = []
    fps_found = 10.0

    for sf, gt_map in gt_all.items():
        video_path = find_video(sf)
        if video_path is None:
            print(f"[skip] 영상 없음: {sf}")
            continue
        print(f"[analyze] {sf}  ({len(gt_map)} GT frames)")
        try:
            error_frames, n_err, n_comp = analyze_source(runtime, video_path, gt_map, args.context_frames)
        except Exception as e:
            print(f"[error] {sf}: {e}")
            continue
        print(f"  오류 {n_err}/{n_comp}  ({n_err/max(n_comp,1):.1%})")

        cap = cv2.VideoCapture(str(video_path))
        fps_found = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        cap.release()

        all_error_frames.extend(error_frames)
        if error_frames:
            all_rendered.extend(fetch_bgr_frames(error_frames, video_path))

    if not all_rendered:
        print("[결과] 오류 프레임 없음")
        return

    out_dir  = run_dir / "error_analysis"
    mp4_path = out_dir / f"error_frames_{run_info.model_id}.mp4"
    csv_path2 = out_dir / "error_summary.csv"
    export_mp4(all_rendered, mp4_path, fps_found, runtime.class_names)
    export_summary_csv(all_error_frames, csv_path2, runtime.class_names)

    actual = sum(1 for e in all_error_frames if not e.is_context)
    total  = sum(len(v) for v in gt_all.values())
    print(f"\n오류 {actual}/{total}  ({actual/max(total,1):.1%})")
    print(f"영상: {mp4_path}\n요약: {csv_path2}")


# ── 진입점 ────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="예측 오류 프레임 분석 뷰어 (UI 모드 / CLI 모드)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--run-dir", default=None,
                        help="model.pt 위치 또는 latest.json 있는 폴더 (CLI 모드)")
    parser.add_argument("--csv", action="append", dest="csv_paths", default=[],
                        help="ground truth CSV (CLI 모드). 반복 사용 가능")
    parser.add_argument("--context-frames", type=int, default=0)
    parser.add_argument("--source-filter", nargs="*", default=None)
    args = parser.parse_args()

    # CLI 모드: --run-dir + --csv 모두 지정된 경우
    if args.run_dir and args.csv_paths:
        run_cli(args)
        return

    # UI 모드
    import tkinter as tk
    from tkinter import messagebox

    root = tk.Tk()
    app  = ErrorFrameApp(root)

    if not app._run_lookup:
        messagebox.showwarning("No Runs", f"No trained runs found under:\n{vca.RUNS_ROOT}")
    if not app._csv_lookup:
        messagebox.showwarning("No CSVs", f"No GT CSVs found under:\n{DATA_FUSION_ROOT}")

    root.mainloop()


if __name__ == "__main__":
    main()
