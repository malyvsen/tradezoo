import plotly.graph_objects as go
from typing import List

from tradezoo.game import TurnResult
from tradezoo.training import Experience, LearningAgent


def reward_plot(agent: LearningAgent, turn_results: List[TurnResult]):
    return go.Figure(
        layout=dict(
            xaxis_title="Turn number",
            yaxis_title="Reward",
        ),
        data=[
            go.Scatter(
                y=[
                    agent.reward(
                        Experience(
                            old_turn_result=old_result, new_turn_result=new_result
                        )
                    )  # TODO: rewards might actually differ over course of training
                    for old_result, new_result in zip(
                        turn_results[:-1], turn_results[1:]
                    )
                ],
            ),
        ],
    )
