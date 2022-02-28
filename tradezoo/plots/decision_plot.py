import numpy as np
import plotly.graph_objects as go

from tradezoo.agent import Agent, Observation


def decision_plot(
    agent: Agent,
    best_asks=np.linspace(0, 4, 16),
    best_bids=np.linspace(0, 4, 16),
    cash_balance=2,
    asset_balance=0.5,
    num_samples=16,
    opacity=0.1,
):
    observations = [
        Observation(
            cash_balance=cash_balance,
            asset_balance=asset_balance,
            best_ask=best_ask,
            best_bid=best_bid,
        )
        for best_bid in best_bids
        for best_ask in best_asks
        for sample_idx in range(num_samples)
    ]
    actions = [
        agent.decide(observation.batch).sample()[0] for observation in observations
    ]
    return go.Figure(
        layout=dict(
            scene=dict(
                xaxis_title="Best ask",
                yaxis_title="Best bid",
                zaxis_title="Price",
                zaxis_type="log",
            )
        ),
        data=[
            go.Scatter3d(
                name="Agent ask",
                mode="markers",
                marker=dict(opacity=opacity),
                x=[observation.best_ask for observation in observations],
                y=[observation.best_bid for observation in observations],
                z=[action.ask for action in actions],
            ),
            go.Scatter3d(
                name="Agent bid",
                mode="markers",
                marker=dict(opacity=opacity),
                x=[observation.best_ask for observation in observations],
                y=[observation.best_bid for observation in observations],
                z=[action.bid for action in actions],
            ),
        ],
    )
