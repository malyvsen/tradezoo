import plotly.graph_objects as go
from typing import List

from tradezoo.game import TurnResult


def td_error_plot(turn_results: List[TurnResult]):
    return go.Figure(
        layout=dict(
            xaxis_title="Turn number",
            yaxis_title="TD error",
        ),
        data=[
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
