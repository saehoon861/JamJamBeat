"""
review.py - 제스처 테스트셋 데이터 검수 도구
============================================
관련 계획서: data/docs/plan_testset.md (섹션 2, 5.2)

실행 방법:
    uv run python review.py --gesture 1

기능 개요:
    - 특정 gesture 클래스의 CSV 및 이미지를 frame_idx 기준으로 동기화하여 표시
    - 좌측: 원본 이미지 / 우측: MediaPipe 3D 랜드마크 시각화
    - Prev / Keep / Drop 버튼으로 검수 진행
    - 마지막 프레임에서 Keep/Drop 클릭 시 후처리 자동 실행
      (CSV에서 drop행 삭제, 이미지를 drop_images/로 이동, 리포트 출력)
"""

import argparse
import glob
import os
import shutil
import sys
import time

import pandas as pd
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, callback_context, dcc, html, no_update

# ─────────────────────────────────────────────
# 상수 및 경로 설정
# ─────────────────────────────────────────────

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
TESTDATA_DIR = os.path.join(SCRIPT_DIR, "testdata")
IMAGES_DIR   = os.path.join(TESTDATA_DIR, "images")
DROP_DIR     = os.path.join(TESTDATA_DIR, "drop_images")

# MediaPipe Hand 랜드마크 연결 순서 (HAND_EDGES)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # 엄지
    (0, 5), (5, 6), (6, 7), (7, 8),           # 검지
    (0, 9), (9, 10), (10, 11), (11, 12),      # 중지
    (0, 13), (13, 14), (14, 15), (15, 16),    # 약지
    (0, 17), (17, 18), (18, 19), (19, 20),    # 소지
    (5, 9), (9, 13), (13, 17),                 # 손바닥 가로 연결
]

LANDMARK_COUNT = 21


# ─────────────────────────────────────────────
# 데이터 로드 / 유효성 검증 유틸
# ─────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """
    역할: --gesture 인자를 파싱한다.
          인자가 없으면 argparse가 에러를 출력하고 즉시 종료한다.
    반환: 파싱된 Namespace (gesture: int)
    """
    parser = argparse.ArgumentParser(
        description="제스처 테스트셋 검수 도구",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--gesture",
        type=int,
        required=True,   # 필수 인자: 생략 시 에러 종료
        choices=list(range(7)),
        metavar="GESTURE",
        help="검수할 제스처 클래스 번호 (0~6)\n예시: python review.py --gesture 1",
    )
    return parser.parse_args()


def load_csv(gesture: int) -> pd.DataFrame:
    """
    역할: 해당 gesture의 landmarks_{gesture}.csv를 로드한다.
          파일이 없으면 에러 메시지를 출력하고 즉시 종료한다.
    파라미터:
        gesture - 검수할 제스처 클래스 번호 (0~6)
    반환: 로드된 DataFrame
    """
    csv_path = os.path.join(TESTDATA_DIR, f"landmarks_{gesture}.csv")
    if not os.path.exists(csv_path):
        print(f"[오류] landmarks_{gesture}.csv 파일을 찾을 수 없습니다: {csv_path}")
        print("       capture.py로 데이터를 수집한 후 실행해 주세요.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    return df.reset_index(drop=True)


def build_image_map(gesture: int, df: pd.DataFrame) -> dict[int, str | None]:
    """
    역할: frame_idx → 이미지 절대경로 매핑 딕셔너리를 구성한다.
          {gesture}_{frame_idx}_*.jpg 패턴으로 이미지를 탐색한다.
          이미지가 없는 경우 None을 저장한다.
    파라미터:
        gesture - 제스처 클래스 번호
        df      - 로드된 랜드마크 DataFrame
    반환: {frame_idx: 이미지 절대경로 | None}
    """
    img_dir = os.path.join(IMAGES_DIR, str(gesture))
    image_map: dict[int, str | None] = {}
    for fidx in df["frame_idx"].tolist():
        # glob 패턴으로 condition/user_name 무관하게 탐색
        pattern = os.path.join(img_dir, f"{gesture}_{fidx}_*.jpg")
        matches = glob.glob(pattern)
        image_map[int(fidx)] = matches[0] if matches else None
    return image_map


def ensure_drop_dir() -> None:
    """
    역할: drop_images/ 폴더가 없으면 자동 생성한다.
          review.py 시작 시점에 반드시 호출한다.
    """
    os.makedirs(DROP_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Plotly 3D 랜드마크 그래프 생성
# ─────────────────────────────────────────────

def make_landmark_figure(row: pd.Series) -> go.Figure:
    """
    역할: 단일 DataFrame 행에서 21개 랜드마크 좌표를 읽어 3D Plotly Figure를 생성한다.
          - 축 비율: aspectmode='cube'  (손 모양에 무관하게 균일 비율 유지)
          - 고정 범위: 각 축 [-1, 1]     (항상 정사각형 무대 내 표시)
          - y축 반전: MediaPipe y좌표는 하단이 1이므로 반전하여 손이 위를 향하게 함
          - 관절 연결선: HAND_EDGES 기준 선분 시각화
    파라미터:
        row - landmarks_{gesture}.csv의 단일 행 (pd.Series)
    반환: go.Figure 객체
    """
    wx, wy, wz = row["x0"], row["y0"], row["z0"]
    xs = [row[f"x{i}"] - wx for i in range(LANDMARK_COUNT)]
    ys = [row[f"y{i}"] - wy for i in range(LANDMARK_COUNT)]
    zs = [row[f"z{i}"] - wz for i in range(LANDMARK_COUNT)]

    # z만 x,y 범위에 맞게 스케일 조정
    xy_range = max(max(xs)-min(xs), max(ys)-min(ys))
    z_range = max(zs)-min(zs) if max(zs) != min(zs) else 1
    zs = [z * (xy_range / z_range) for z in zs]

    # # 손 크기에 맞춰 좌표 정규화 (중앙 정렬 및 비율 유지 스케일링)
    # cx = sum(xs) / len(xs)
    # cy = sum(ys) / len(ys)
    # cz = sum(zs) / len(zs)
    
    # xs = [x - cx for x in xs]
    # ys = [y - cy for y in ys]
    # zs = [z - cz for z in zs]
    
    # max_range = max(max(xs)-min(xs), max(ys)-min(ys), max(zs)-min(zs))
    # if max_range == 0:
    #     max_range = 1
        
    # scale = max_range / 1.5
    # xs = [x / scale for x in xs]
    # ys = [y / scale for y in ys]
    # zs = [z / scale for z in zs]

    traces = []

    # 관절 연결선 추가 (HAND_EDGES 정의 순서)
    for (a, b) in HAND_EDGES:
        traces.append(go.Scatter3d(
            x=[xs[a], xs[b]],
            y=[ys[a], ys[b]],
            z=[zs[a], zs[b]],
            mode="lines",
            line=dict(color="#00CC66", width=4),
            showlegend=False,
            hoverinfo="skip",
        ))

    # 랜드마크 점 추가
    traces.append(go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode="markers",
        marker=dict(size=5, color="#FFFFFF", line=dict(color="#00CC66", width=1)),
        showlegend=False,
        hovertemplate="idx:%{pointNumber}<br>x:%{x:.3f}<br>y:%{y:.3f}<br>z:%{z:.3f}<extra></extra>",
    ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="#1a1a2e",
        scene=dict(
            aspectmode="cube",
            xaxis=dict(
                range=[-0.25,0.25],
                tickmode="linear",
                tick0=-0.25,
                dtick=0.25,
                title="X",
                color="#aaa",
                showbackground=False
            ),
            yaxis=dict(
                range=[-1,1],
                tickmode="linear",
                tick0=-1,
                dtick=0.25,
                autorange="reversed",
                title="Y",
                color="#aaa",
                showbackground=False
            ),
            zaxis=dict(
                range=[-1,1],
                tickmode="linear",
                tick0=-1,
                dtick=0.5,
                title="Z",
                color="#aaa",
                showbackground=False
            ),
            bgcolor="#1a1a2e",
        ),
    )
    return fig


# ─────────────────────────────────────────────
# 후처리
# ─────────────────────────────────────────────

def run_postprocess(
    gesture: int,
    df: pd.DataFrame,
    drop_list: list[int],
    image_map: dict[int, str | None],
) -> None:
    """
    역할: 검수 완료 후 처리를 순서대로 실행한다.
          ① drop_list 기반으로 CSV 행 삭제 후 덮어쓰기
          ② drop된 이미지를 drop_images/ 폴더로 이동 (원본 파일명 유지)
          ③ 최종 요약 리포트 출력
    파라미터:
        gesture   - 검수한 제스처 클래스 번호
        df        - 원본 DataFrame
        drop_list - Drop된 frame_idx 리스트
        image_map - {frame_idx: 이미지 절대경로 | None}
    """
    csv_path = os.path.join(TESTDATA_DIR, f"landmarks_{gesture}.csv")

    # ① CSV 업데이트: drop_list에 해당하는 행 영구 삭제
    if drop_list:
        df_filtered = df[~df["frame_idx"].isin(drop_list)].reset_index(drop=True)
        df_filtered.to_csv(csv_path, mode='w', header=True, index=False)
        print(f"[후처리] CSV 업데이트: {len(drop_list)}행 삭제 → {csv_path}")
    else:
        print("[후처리] Drop 항목 없음. CSV 변경 없음.")

    # ② 이미지 이동: drop_images/ 폴더로 shutil.move
    moved: list[str] = []
    for fidx in drop_list:
        src = image_map.get(fidx)
        if src and os.path.exists(src):
            dst = os.path.join(DROP_DIR, os.path.basename(src))
            shutil.move(src, dst)
            moved.append(os.path.basename(src))
            print(f"[후처리] 이미지 이동: {os.path.basename(src)} → drop_images/")
        else:
            print(f"[후처리] frame_idx={fidx}에 해당하는 이미지 없음 (이동 생략)")

    # ③ 최종 요약 리포트 출력
    print("\n" + "=" * 50)
    print(f"  검수 완료 리포트 (gesture={gesture})")
    print("=" * 50)
    print(f"  총 Drop 수: {len(drop_list)}개")
    if drop_list:
        print(f"  Drop된 frame_idx: {sorted(drop_list)}")
    else:
        print("  Drop된 프레임 없음.")
    print("=" * 50 + "\n")


# ─────────────────────────────────────────────
# Dash 앱
# ─────────────────────────────────────────────

def build_app(
    gesture: int,
    df: pd.DataFrame,
    image_map: dict[int, str | None],
) -> Dash:
    """
    역할: 검수용 Dash 웹 앱을 구성하고 반환한다.
    파라미터:
        gesture   - 검수할 제스처 클래스 번호
        df        - 랜드마크 DataFrame
        image_map - {frame_idx: 이미지 절대경로 | None}
    반환: 초기화된 Dash 앱 인스턴스
    """

    total = len(df)
    if total == 0:
        print(f"[오류] landmarks_{gesture}.csv에 데이터가 없습니다.")
        sys.exit(1)

    frame_ids = df["frame_idx"].tolist()

    # 이미지를 Dash에서 서빙하기 위한 정적 경로 설정
    app = Dash(
        __name__,
        assets_folder=TESTDATA_DIR,      # testdata/ 를 정적 파일 루트로 사용
        suppress_callback_exceptions=True,
    )

    # ── 레이아웃 정의 ──
    app.layout = html.Div(
        style={
            "background": "#0f0f1a",
            "color": "#e0e0e0",
            "fontFamily": "monospace",
            "minHeight": "100vh",
            "padding": "20px",
        },
        children=[
            # 상태 저장용 Store (클라이언트 사이드 메모리)
            dcc.Store(id="cur-idx",   data=0),            # 현재 프레임 인덱스
            dcc.Store(id="drop-list", data=[]),           # 드롭된 frame_idx 목록
            dcc.Store(id="done",      data=False),        # 후처리 완료 플래그

            # 상단 정보 바
            html.Div(
                style={"marginBottom": "12px", "borderBottom": "1px solid #333", "paddingBottom": "8px"},
                children=[
                    html.H3(
                        f"검수 도구  |  Gesture {gesture}  |  총 {total}프레임",
                        style={"margin": 0, "color": "#00CC66"},
                    ),
                    html.Div(id="status-bar", style={"fontSize": "13px", "color": "#aaa", "marginTop": "4px"}),
                ],
            ),

            # 2분할 메인 뷰
            html.Div(
                style={"display": "flex", "gap": "16px", "alignItems": "flex-start"},
                children=[
                    # 좌측: 실제 이미지
                    html.Div(
                        style={"flex": "1", "background": "#1a1a2e", "borderRadius": "8px", "padding": "8px"},
                        children=[
                            html.P("원본 이미지", style={"margin": "0 0 6px 0", "color": "#888", "fontSize": "12px"}),
                            html.Img(
                                id="frame-image",
                                style={"width": "100%", "borderRadius": "4px", "display": "block"},
                            ),
                        ],
                    ),
                    # 우측: 3D 랜드마크
                    html.Div(
                        style={"flex": "1", "background": "#1a1a2e", "borderRadius": "8px", "padding": "8px"},
                        children=[
                            html.P("3D 랜드마크", style={"margin": "0 0 6px 0", "color": "#888", "fontSize": "12px"}),
                            dcc.Graph(
                                id="landmark-graph",
                                style={"height": "480px"},
                                config={"displayModeBar": False},
                            ),
                        ],
                    ),
                ],
            ),

            # 하단 중앙 버튼 영역
            html.Div(
                style={"display": "flex", "justifyContent": "center", "gap": "24px", "marginTop": "24px"},
                children=[
                    html.Button(
                        "◀ Prev",
                        id="btn-prev",
                        n_clicks=0,
                        style=_btn_style("#444", "#eee"),
                    ),
                    html.Button(
                        "✓ Keep",
                        id="btn-keep",
                        n_clicks=0,
                        style=_btn_style("#006633", "#00ff88"),
                    ),
                    html.Button(
                        "✕ Drop",
                        id="btn-drop",
                        n_clicks=0,
                        style=_btn_style("#660000", "#ff4444"),
                    ),
                ],
            ),

            # 후처리 완료 메시지 영역
            html.Div(id="done-message", style={"marginTop": "20px", "textAlign": "center"}),
        ],
    )

    # ── 콜백: Prev / Keep / Drop 버튼 ──
    @app.callback(
        Output("cur-idx",        "data"),
        Output("drop-list",      "data"),
        Output("done",           "data"),
        Output("frame-image",    "src"),
        Output("landmark-graph", "figure"),
        Output("status-bar",     "children"),
        Output("done-message",   "children"),
        Input("btn-prev",  "n_clicks"),
        Input("btn-keep",  "n_clicks"),
        Input("btn-drop",  "n_clicks"),
        State("cur-idx",   "data"),
        State("drop-list", "data"),
        State("done",      "data"),
    )
    def handle_buttons(
        n_prev, n_keep, n_drop,
        cur_idx: int,
        drop_list: list[int],
        done: bool,
    ):
        """
        트리거: Prev / Keep / Drop 버튼 클릭
        입력:
            cur_idx   - 현재 프레임 인덱스 (0-based)
            drop_list - 지금까지 Drop된 frame_idx 목록
            done      - 후처리 완료 여부
        출력:
            cur-idx        - 갱신된 인덱스
            drop-list      - 갱신된 drop 목록
            done           - 후처리 완료 플래그
            frame-image    - 이미지 src URL
            landmark-graph - Plotly Figure
            status-bar     - 상태 텍스트
            done-message   - 완료 메시지 HTML
        """
        if cur_idx is None: cur_idx = 0
        if drop_list is None: drop_list = []
        if done is None: done = False

        if done:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update

        # 어떤 버튼이 눌렸는지 판별
        triggered = callback_context.triggered_id

        new_idx   = cur_idx
        new_drops = list(drop_list)
        new_done  = False
        done_msg  = ""

        if triggered == "btn-prev":
            # Prev: 인덱스 0이면 동작하지 않음 (음수 인덱스 방지)
            if cur_idx > 0:
                new_idx = cur_idx - 1
                # 재결정 지원: 이전 프레임이 drop 상태였으면 drop_list에서 제거
                prev_fidx = int(frame_ids[new_idx])
                if prev_fidx in new_drops:
                    new_drops.remove(prev_fidx)

        elif triggered in ("btn-keep", "btn-drop"):
            if triggered == "btn-drop":
                # Drop: 현재 frame_idx를 drop_list에 추가 (중복 방지)
                cur_fidx = int(frame_ids[cur_idx])
                if cur_fidx not in new_drops:
                    new_drops.append(cur_fidx)

            # 마지막 프레임인지 확인 → 자동 후처리 트리거
            if cur_idx >= total - 1:
                # 마지막 프레임에서 Keep/Drop 클릭 → 후처리 자동 실행
                run_postprocess(gesture, df, new_drops, image_map)
                new_done = True
                done_msg = html.Div([
                    html.H3("✅ 검수 완료!", style={"color": "#00CC66"}),
                    html.P(f"총 Drop: {len(new_drops)}개 | {sorted(new_drops)}", style={"color": "#aaa"}),
                    html.P("터미널에서 상세 리포트를 확인하세요.", style={"color": "#666"}),
                ])
            else:
                new_idx = cur_idx + 1

        # 현재 프레임 정보 렌더링
        row      = df.iloc[new_idx]
        cur_fidx = int(frame_ids[new_idx])
        img_path = image_map.get(cur_fidx)

        # 이미지 src: 캐시 버스팅을 위해 ?t={ms timestamp} 쿼리스트링 추가
        if img_path and os.path.exists(img_path):
            # Dash는 assets_folder 기준 상대경로를 /assets/ URL로 서빙
            rel_path = os.path.relpath(img_path, TESTDATA_DIR)
            img_src  = f"/assets/{rel_path.replace(os.sep, '/')}?t={int(time.time() * 1000)}"
        else:
            # 이미지 없음 → 플레이스홀더
            img_src = "https://via.placeholder.com/640x360/1a1a2e/666666?text=이미지+없음"

        # 3D 랜드마크 Figure 생성
        fig = make_landmark_figure(row)

        # 상태 텍스트
        is_dropped = cur_fidx in new_drops
        drop_mark  = "🔴 DROP" if is_dropped else "🟢 KEEP"
        status = (
            f"프레임 {new_idx + 1} / {total}  |  frame_idx={cur_fidx}  |  {drop_mark}  "
            f"|  누적 Drop: {len(new_drops)}개"
        )

        return new_idx, new_drops, new_done, img_src, fig, status, done_msg

    return app


def _btn_style(bg: str, fg: str) -> dict:
    """버튼 공통 스타일 헬퍼"""
    return {
        "background":   bg,
        "color":        fg,
        "border":       "none",
        "borderRadius": "8px",
        "padding":      "14px 36px",
        "fontSize":     "16px",
        "fontWeight":   "bold",
        "cursor":       "pointer",
        "letterSpacing": "1px",
    }


# ─────────────────────────────────────────────
# 메인
# ─────────────────────────────────────────────

def main() -> None:
    args    = parse_args()
    gesture = args.gesture

    print(f"\n[검수 시작] Gesture {gesture}")

    # drop_images/ 폴더 사전 생성
    ensure_drop_dir()

    # 데이터 로드
    df        = load_csv(gesture)
    image_map = build_image_map(gesture, df)

    total = len(df)
    print(f"[로드 완료] {total}개 프레임 | 이미지 매핑: "
          f"{sum(1 for v in image_map.values() if v)}개 일치")

    # Dash 앱 구축 및 실행
    app = build_app(gesture, df, image_map)
    print("\n[안내] 브라우저에서 http://127.0.0.1:8050 을 열어 검수를 진행하세요.")
    print("[안내] 마지막 프레임에서 Keep/Drop 클릭 시 후처리가 자동 실행됩니다.\n")
    app.run(debug=True, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()
