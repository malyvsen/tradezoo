import plotly.graph_objects as go
from typing import List

from tradezoo.game import TurnResult
from tradezoo.market import BuyOrder, SellOrder


def orders_plot(turn_results: List[TurnResult]):
    def scatter_orders(name: str, order_type: type):
        selected_results = select_results(order_type)
        return go.Scatter(
            name=name,
            mode="markers",
            x=[turn_result.time_step for turn_result in selected_results],
            y=[turn_result.order.price for turn_result in selected_results],
            marker=dict(
                size=[bubble_size(turn_result) for turn_result in selected_results],
                opacity=[
                    0.5 if turn_result.decision.random else 1
                    for turn_result in selected_results
                ],
            ),
        )

    def select_results(order_type: type):
        return [
            turn_result
            for turn_result in turn_results
            if isinstance(turn_result.order, order_type)
        ]

    def bubble_size(turn_result: TurnResult):
        relative_mass = (
            turn_result.order.volume
            * turn_result.order.price
            / turn_result.state.net_worth
        )
        return relative_mass**0.5 * 20

    return go.Figure(
        layout=dict(
            xaxis=dict(title="Turn number", range=[0, len(turn_results) - 1]),
            yaxis=dict(title="Price", type="log"),
            legend=dict(title=dict(text="Opacity indicates randomness")),
        ),
        data=[
            scatter_orders(name="Buy order", order_type=BuyOrder),
            scatter_orders(name="Sell order", order_type=SellOrder),
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
