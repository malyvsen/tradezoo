import plotly.graph_objects as go
from typing import List

from tradezoo.game import TurnResult


def uncertainty_plot(turn_results: List[TurnResult]):
    return go.Figure(
        layout=dict(
            xaxis_title="Turn number",
            yaxis_title="Underlying standard deviation",
            yaxis_type="log",
        ),
        data=[
            go.Scatter(
                name="Mid-price",
                y=[
                    turn_result.decision_batch.log_mid_price.stddev.exp().item()
                    for turn_result in turn_results
                ],
            ),
            go.Scatter(
                name="Spread",
                y=[
                    turn_result.decision_batch.log_spread.stddev.exp().item()
                    for turn_result in turn_results
                ],
            ),
        ],
    )
