import numpy as np
import plotly
import plotly.graph_objects as go

from tradezoo.agent import Observation
from tradezoo.game import Trader


def utility_plot(
    trader: Trader,
    cash_balances=np.linspace(0, 256, 16),
    asset_balances=np.linspace(0, 256, 16),
    best_ask=2,
    best_bid=0.5,
):
    observations = [
        [
            Observation(
                cash_balance=cash_balance,
                asset_balance=asset_balance,
                best_ask=best_ask,
                best_bid=best_bid,
            )
            for cash_balance in cash_balances
        ]
        for asset_balance in asset_balances
    ]
    return go.Figure(
        layout=dict(
            scene=dict(
                xaxis_title="Cash balance",
                yaxis_title="Asset balance",
                zaxis_title="Utility",
            )
        ),
        data=[
            go.Surface(
                name="Critic's expectation of future",
                showscale=False,
                showlegend=True,
                colorscale=plotly.colors.sequential.thermal,
                x=cash_balances,
                y=asset_balances,
                z=[
                    [
                        trader.agent.evaluate(observation.batch).item()
                        * (1 - trader.agent.discount_factor)
                        for observation in obs
                    ]
                    for obs in observations
                ],
            ),
            go.Surface(
                name="True instantaneous utility",
                showscale=False,
                showlegend=True,
                colorscale=plotly.colors.sequential.haline,
                x=cash_balances,
                y=asset_balances,
                z=[
                    [trader.utility(observation) for observation in obs]
                    for obs in observations
                ],
            ),
        ],
    )
