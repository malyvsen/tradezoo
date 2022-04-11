import pandas as pd
import plotly.graph_objects as go
from typing import List

from tradezoo.training import TrainResult


def training_plot(train_results: List[TrainResult], smoothing=16):
    td_errors = pd.Series([train_result.td_error for train_result in train_results])
    return go.Figure(
        layout=dict(
            xaxis_title="Training step",
            yaxis_title="TD error",
        ),
        data=[
            go.Scatter(
                name="Raw",
                opacity=0.25,
                y=td_errors,
            ),
            go.Scatter(
                name="Smoothed",
                y=td_errors.rolling(window=smoothing).mean(),
            ),
            go.Scatter(
                name="Absolute",
                y=td_errors.abs(),
            ),
            go.Scatter(
                name="Absolute smoothed",
                y=td_errors.abs().rolling(window=smoothing).mean(),
            ),
        ],
    )
