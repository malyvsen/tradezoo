import plotly.graph_objects as go
from typing import List

from tradezoo.game import TurnResult


def price_plot(turn_results: List[TurnResult]):
    flat_trades = [
        (turn_result, trade)
        for turn_result in turn_results
        for trade in turn_result.trades
    ]

    def prices(client_side: str):
        if client_side == None:
            return [
                (turn_result.time_step, trade.price)
                for turn_result, trade in flat_trades
                if turn_result.trader.client.account not in [trade.buyer, trade.seller]
            ]
        return [
            (turn_result.time_step, trade.price)
            for turn_result, trade in flat_trades
            if turn_result.trader.client.account is getattr(trade, client_side)
        ]

    def scatter_prices(name: str, client_side: str):
        return go.Scatter(
            name=name,
            mode="markers",
            x=[time_step for time_step, price in prices(client_side=client_side)],
            y=[price for time_step, price in prices(client_side=client_side)],
        )

    return go.Figure(
        layout=dict(
            xaxis_title="Turn number",
            yaxis_title="Trade price",
        ),
        data=[
            scatter_prices(name="Purchase from client", client_side="seller"),
            scatter_prices(name="Sale to client", client_side="buyer"),
            scatter_prices(name="Excange between traders", client_side=None),
        ],
    )
