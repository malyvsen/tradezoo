import plotly.graph_objects as go
from typing import List

from tradezoo.game import TurnResult


def reward_plot(turn_results: List[TurnResult]):
    return go.Figure(
        layout=dict(
            xaxis_title="Turn number",
            yaxis_title="Reward",
        ),
        data=[
            go.Scatter(
                name="Instantaneous",
                y=[turn_result.reward for turn_result in turn_results],
            ),
            go.Scatter(
                name="Estimated discounted reward",
                y=[
                    turn_result.trader.agent.evaluate(
                        turn_result.observation.batch
                    ).item()
                    for turn_result in turn_results
                ],
            ),
        ],
    )
