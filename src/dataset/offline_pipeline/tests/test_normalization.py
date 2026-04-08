import sys
import argparse
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from pathlib import Path

current_dir = Path(__file__).resolve().parent
offline_dir = current_dir.parent
if str(offline_dir) not in sys.path:
    sys.path.insert(0, str(offline_dir))

import config


# mediapipe hand connections
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def add_skeleton(fig, landmarks, row, col, title):

    x = landmarks[:, 0]
    y = landmarks[:, 1]
    z = landmarks[:, 2]

    # joints
    fig.add_trace(
        go.Scatter3d(
            x=x,
            y=y,
            z=z,
            mode="markers+text",
            marker=dict(size=4),
            text=[str(i) for i in range(21)],
            showlegend=False
        ),
        row=row,
        col=col
    )

    # bones
    for s, e in CONNECTIONS:
        fig.add_trace(
            go.Scatter3d(
                x=[x[s], x[e]],
                y=[y[s], y[e]],
                z=[z[s], z[e]],
                mode="lines",
                line=dict(width=3),
                showlegend=False
            ),
            row=row,
            col=col
        )


def test_normalization(csv_path: str):

    print(f"\n[Test] Running gesture-wise visualization on {csv_path}")

    df = pd.read_csv(csv_path)
    

    if len(df) == 0:
        print("Empty dataframe")
        return

    # gesture 0 제외
    df = df[df["gesture"] != 0]

    # gesture별 1개 샘플
    sampled_df = df.groupby("gesture").sample(n=1, random_state=42)

    # 1~6만 (최대 6개)
    sampled_df = sampled_df.sort_values("gesture").head(6)

    fig = make_subplots(
        rows=2,
        cols=3,
        specs=[[{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}],
               [{'type': 'scene'}, {'type': 'scene'}, {'type': 'scene'}]],
        subplot_titles=[
            f"gesture {g}" for g in sampled_df["gesture"]
        ]
    )

    for i, (_, row) in enumerate(sampled_df.iterrows()):

        coords = np.array([
            [row[f'x{j}'], row[f'y{j}'], row[f'z{j}']]
            for j in range(21)
        ])

        r = i // 3 + 1
        c = i % 3 + 1

        add_skeleton(
            fig,
            coords,
            r,
            c,
            f"gesture {row['gesture']}"
        )

    fig.update_layout(
        title="Gesture-wise Skeleton Samples",
        height=900
    )

    fig.show()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "csv_path",
        help="Path to processed csv file"
    )

    args = parser.parse_args()

    test_normalization(args.csv_path)