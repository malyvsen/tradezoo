import pandas as pd
import plotly.graph_objects as go
from typing import List

from tradezoo.training import TrainResult


def training_plot(train_results: List[TrainResult], smoothing=16):
    losses = pd.Series([train_result.loss for train_result in train_results])
    return go.Figure(
        layout=dict(
            xaxis_title="Training step",
            yaxis_title="Loss",
        ),
        data=[
            go.Scatter(
                name="Raw",
                opacity=0.25,
                y=losses,
            ),
            go.Scatter(
                name="Smoothed",
                y=losses.rolling(window=smoothing).mean(),
            ),
        ],
    )
