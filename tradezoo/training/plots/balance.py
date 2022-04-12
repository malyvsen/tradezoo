import plotly.graph_objects as go
from typing import List

from tradezoo.game import TurnResult


def balance_plot(turn_results: List[TurnResult]):
    return go.Figure(
        layout=dict(
            xaxis_title="Turn number",
            yaxis_title="Balance",
        ),
        data=[
            go.Scatter(
                name="Cash",
                y=[turn_result.state.cash_balance for turn_result in turn_results],
            ),
            go.Scatter(
                name="Asset",
                y=[turn_result.state.asset_balance for turn_result in turn_results],
            ),
            go.Scatter(
                name="Total",
                y=[
                    turn_result.state.cash_balance + turn_result.state.asset_balance
                    for turn_result in turn_results
                ],
            ),
            go.Scatter(
                name="Net worth",
                y=[turn_result.state.net_worth for turn_result in turn_results],
            ),
        ],
    )
