import plotly.graph_objects as go
from typing import List

from tradezoo.game import TurnResult


def trades_plot(turn_results: List[TurnResult]):
    return go.Figure(
        layout=dict(
            xaxis=dict(title="Turn number", range=[0, len(turn_results)]),
            yaxis=dict(title="Price", type="log"),
        ),
        data=[
            go.Scatter(
                name="Trade",
                mode="markers",
                x=[
                    idx
                    for idx, turn_result in enumerate(turn_results)
                    for trade in turn_result.trades
                ],
                y=[
                    trade.price
                    for turn_result in turn_results
                    for trade in turn_result.trades
                ],
            ),
            go.Scatter(
                name="Best ask",
                y=[turn_result.state.best_ask for turn_result in turn_results],
            ),
            go.Scatter(
                name="Best bid",
                y=[turn_result.state.best_bid for turn_result in turn_results],
            ),
        ],
    )
