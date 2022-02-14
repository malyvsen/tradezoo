from dataclasses import dataclass
from functools import cached_property

from .account import Account
from .order import Order, BuyOrder, SellOrder


@dataclass(frozen=True)
class Trade:
    buyer: Account
    seller: Account
    price: float
    volume: float

    @classmethod
    def from_orders(cls, present_order: Order, incoming_order: Order) -> "Trade":
        assert present_order.matches(incoming_order)
        assert present_order.executable
        assert incoming_order.executable
        buy_order, sell_order = (
            (present_order, incoming_order)
            if isinstance(present_order, BuyOrder)
            else (incoming_order, present_order)
        )
        return Trade(
            buyer=buy_order.submitted_by,
            seller=sell_order.submitted_by,
            price=incoming_order.price,
            volume=min(
                buy_order.volume,
                buy_order.submitted_by.cash_balance / incoming_order.price,
                sell_order.volume,
                sell_order.submitted_by.stock_balance,
            ),
        )

    @cached_property
    def cash_amount(self):
        return self.price * self.volume
