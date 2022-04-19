import plotly.graph_objects as go
from typing import List

from tradezoo.game import TurnResult


def price_plot(turn_results: List[TurnResult]):
    return go.Figure(
        layout=dict(
            xaxis_title="Turn number",
            yaxis_title="Trade price",
        ),
        data=[
            go.Scatter(
                x=[
                    turn_result.time_step
                    for turn_result in turn_results
                    for trade in turn_result.trades
                ],
                y=[
                    trade.price
                    for turn_result in turn_results
                    for trade in turn_result.trades
                ],
            ),
        ],
    )
