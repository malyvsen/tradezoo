from dataclasses import dataclass
from typing import List

from .account import Account
from .account_filter import AccountFilter, BlacklistFilter


@dataclass
class Order:
    submitted_by: Account
    visibility: AccountFilter
    price: float
    volume: float

    @classmethod
    def public(cls, submitted_by: Account, price: float, volume: float):
        return cls(
            submitted_by=submitted_by,
            visibility=BlacklistFilter([submitted_by]),
            price=price,
            volume=volume,
        )

    @property
    def priority(self) -> float:
        """An order with a higher priority gets executed sooner"""
        raise NotImplementedError()

    def matches(self, other: "Order") -> bool:
        raise NotImplementedError()

    @classmethod
    def match(cls, buy_order: "BuyOrder", sell_order: "SellOrder") -> bool:
        assert isinstance(buy_order, BuyOrder)
        assert isinstance(sell_order, SellOrder)
        if not buy_order.visibility.matches(sell_order.submitted_by):
            return False
        if not sell_order.visibility.matches(buy_order.submitted_by):
            return False
        return buy_order.price >= sell_order.price


@dataclass
class BuyOrder(Order):
    @property
    def priority(self) -> float:
        return self.price

    def matches(self, order) -> bool:
        if not isinstance(order, SellOrder):
            return False
        return Order.match(self, order)


@dataclass
class SellOrder(Order):
    @property
    def priority(self) -> float:
        return -self.price

    def matches(self, order) -> bool:
        if not isinstance(order, BuyOrder):
            return False
        return Order.match(order, self)
