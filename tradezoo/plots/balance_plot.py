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
                y=[
                    turn_result.observation.cash_balance for turn_result in turn_results
                ],
            ),
            go.Scatter(
                name="Asset",
                y=[
                    turn_result.observation.asset_balance
                    for turn_result in turn_results
                ],
            ),
            go.Scatter(
                name="Reward",
                y=[turn_result.reward for turn_result in turn_results],
            ),
            go.Scatter(
                name="Expected future reward",
                y=[
                    turn_result.trader.agent.evaluate(
                        turn_result.observation.batch
                    ).item()
                    * (1 - turn_result.trader.agent.discount_factor)
                    for turn_result in turn_results
                ],
            ),
        ],
    )
