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
                name="Instantaneous reward",
                y=[turn_result.reward for turn_result in turn_results[:-1]],
            ),
            go.Scatter(
                name="Expected future reward",
                y=[
                    turn_result.trader.agent.evaluate(
                        turn_result.observation.batch
                    ).item()
                    * (1 - turn_result.trader.agent.discount_factor)
                    for turn_result in turn_results[1:]
                ],
            ),
            go.Scatter(
                name="TD error",
                y=[
                    old.reward
                    + new.trader.agent.discount_factor
                    * new.trader.agent.evaluate(new.observation.batch).item()
                    - old.trader.agent.evaluate(old.observation.batch).item()
                    for old, new in zip(turn_results[:-1], turn_results[1:])
                ],
            ),
        ],
    )
