import plotly
import plotly.graph_objects as go
from typing import List

from tradezoo.game import TurnResult


def trades_plot(turn_results: List[TurnResult], num_samples=1, opacity=0.1):
    return go.Figure(
        layout=dict(
            xaxis_title="Turn number",
            yaxis_title="Price",
            yaxis_type="log",
        ),
        data=[
            go.Scatter(
                name="Trade",
                mode="markers",
                marker=dict(color=plotly.colors.qualitative.Plotly[0]),
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
                marker=dict(color=plotly.colors.qualitative.Plotly[1]),
                y=[turn_result.observation.best_ask for turn_result in turn_results],
            ),
            go.Scatter(
                name="Best bid",
                marker=dict(color=plotly.colors.qualitative.Plotly[1]),
                y=[turn_result.observation.best_bid for turn_result in turn_results],
            ),
            go.Scatter(
                name="Trader ask distribution",
                mode="markers",
                marker=dict(color=plotly.colors.qualitative.Plotly[2], opacity=opacity),
                x=[
                    turn_result.turn_number
                    for turn_result in turn_results
                    for sample_idx in range(num_samples)
                ],
                y=[
                    turn_result.decision_batch.sample()[0].ask
                    for turn_result in turn_results
                    for sample_idx in range(num_samples)
                ],
            ),
            go.Scatter(
                name="Trader bid distribution",
                mode="markers",
                marker=dict(color=plotly.colors.qualitative.Plotly[3], opacity=opacity),
                x=[
                    turn_result.turn_number
                    for turn_result in turn_results
                    for sample_idx in range(num_samples)
                ],
                y=[
                    turn_result.decision_batch.sample()[0].bid
                    for turn_result in turn_results
                    for sample_idx in range(num_samples)
                ],
            ),
        ],
    )
