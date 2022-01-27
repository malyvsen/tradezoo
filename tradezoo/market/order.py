from dataclasses import dataclass

from .account import Account


@dataclass
class Order:
    submitted_by: Account
    price: float
    volume: float

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
        if buy_order.submitted_by is sell_order.submitted_by:
            return False
        if buy_order.submitted_by.cash_balance <= 0:
            return False
        if sell_order.submitted_by.stock_balance <= 0:
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
