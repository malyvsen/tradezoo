from dataclasses import dataclass

from .account import Account


@dataclass
class Order:
    submitted_by: Account
    price: float

    @property
    def valid(self) -> bool:
        raise NotImplementedError()

    def execute_(self):
        raise NotImplementedError()

    @classmethod
    def match(cls, buy_order: "BuyOrder", sell_order: "SellOrder") -> bool:
        assert isinstance(buy_order, BuyOrder)
        assert isinstance(sell_order, SellOrder)
        if not buy_order.valid or not sell_order.valid:
            return False
        if buy_order.submitted_by is sell_order.submitted_by:
            return False
        return buy_order.price >= sell_order.price


@dataclass
class BuyOrder:
    @property
    def valid(self) -> bool:
        return self.account.cash_balance >= self.price

    def execute_(self, price: float):
        assert price <= self.price
        self.submitted_by.cash_balance -= price
        self.submitted_by.stock_balance += 1

    def matches(self, order) -> bool:
        if not isinstance(order, SellOrder):
            return False
        return Order.match(self, order)


@dataclass
class SellOrder:
    @property
    def valid(self) -> bool:
        return self.account.stock_balance >= 1

    def execute_(self, price: float):
        assert price >= self.price
        self.submitted_by.cash_balance += price
        self.submitted_by.stock_balance -= 1

    def matches(self, order) -> bool:
        if not isinstance(order, BuyOrder):
            return False
        return Order.match(order, self)
