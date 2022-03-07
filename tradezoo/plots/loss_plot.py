import pandas as pd
import plotly
import plotly.graph_objects as go
from typing import List

from tradezoo.trainer import TrainResult


def loss_plot(train_results: List[TrainResult]):
    return go.Figure(
        layout=dict(
            xaxis_title="Training step",
            yaxis_title="Loss",
            yaxis_type="log",
        ),
        data=[
            go.Scatter(
                name="Actor",
                marker=dict(color=plotly.colors.qualitative.Plotly[0]),
                opacity=0.25,
                y=[train_result.actor_loss for train_result in train_results],
            ),
            go.Scatter(
                name="Actor, smoothed",
                marker=dict(color=plotly.colors.qualitative.Plotly[0]),
                y=pd.Series([train_result.actor_loss for train_result in train_results])
                .rolling(window=16)
                .mean(),
            ),
            go.Scatter(
                name="Critic",
                marker=dict(color=plotly.colors.qualitative.Plotly[1]),
                opacity=0.25,
                y=[train_result.critic_loss for train_result in train_results],
            ),
            go.Scatter(
                name="Critic, smoothed",
                marker=dict(color=plotly.colors.qualitative.Plotly[1]),
                y=pd.Series(
                    [train_result.critic_loss for train_result in train_results]
                )
                .rolling(window=16)
                .mean(),
            ),
        ],
    )
