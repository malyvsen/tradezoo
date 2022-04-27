import plotly.graph_objects as go
from typing import List

from tradezoo.game import Trader, TurnResult
from tradezoo.market import Trade


def trades_plot(trader: Trader, turn_results: List[TurnResult]):
    def select_trades(side: str):
        return [
            (turn_result, trade)
            for turn_result in turn_results
            for trade in turn_result.trades
            if getattr(trade, side) == trader.account
        ]

    def bubble_size(turn_result: TurnResult, trade: Trade):
        relative_mass = trade.volume * trade.price / turn_result.state.net_worth
        return relative_mass**0.5 * 20

    def scatter_trades(name: str, side: str):
        trades = select_trades(side)
        return go.Scatter(
            name=name,
            mode="markers",
            x=[turn_result.time_step for turn_result, trade in trades],
            y=[trade.price for turn_result, trade in trades],
            marker=dict(
                size=[
                    bubble_size(turn_result=turn_result, trade=trade)
                    for turn_result, trade in trades
                ],
                opacity=[
                    0.5 if turn_result.decision.random else 1
                    for turn_result, trade in trades
                ],
            ),
        )

    return go.Figure(
        layout=dict(
            xaxis=dict(title="Turn number", range=[0, len(turn_results) - 1]),
            yaxis=dict(title="Price", type="log"),
            legend=dict(title=dict(text="Opacity indicates randomness")),
        ),
        data=[
            scatter_trades(name="Purchase", side="buyer"),
            scatter_trades(name="Sale", side="seller"),
            go.Scatter(
                name="Best ask",
                y=[turn_result.state.best_ask for turn_result in turn_results],
            ),
            go.Scatter(
                name="Best bid",
                y=[turn_result.state.best_bid for turn_result in turn_results],
            ),
        ],
    )
