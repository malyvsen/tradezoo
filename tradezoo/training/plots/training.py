import pandas as pd
import plotly.graph_objects as go
from typing import List

from ..learning_agent import LearningAgent
from ..train_result import TrainResult


def training_plot(agent: LearningAgent, train_results: List[TrainResult], smoothing=16):
    losses = pd.Series([train_result.loss for train_result in train_results])
    smoothed_losses = losses.rolling(window=smoothing).mean()
    target_update_steps = list(
        range(0, len(train_results), agent.steps_per_target_update)
    )
    return go.Figure(
        layout=dict(
            xaxis=dict(title="Training step", range=[0, len(train_results) - 1]),
            yaxis=dict(title="Loss", type="log"),
        ),
        data=[
            go.Scatter(
                name="Raw",
                opacity=0.25,
                y=losses,
            ),
            go.Scatter(
                name="Smoothed",
                y=smoothed_losses,
            ),
            go.Scatter(
                name="Target update",
                mode="markers",
                x=target_update_steps,
                y=smoothed_losses[target_update_steps],
            ),
        ],
    )
