from dataclasses import dataclass
from typing import List

from .account import Account
from .order import Order, BuyOrder, SellOrder
from .trade import Trade


@dataclass
class Market:
    accounts: List[Account]
    orders: List[Order]

    @classmethod
    def from_accounts(cls, accounts: List[Account]) -> "Market":
        return cls(accounts=accounts, orders=[])

    def orders_by(self, account: Account):
        return [order for order in self.orders if order.submitted_by is account]

    def matching_orders(self, order: Order):
        return [potential for potential in self.orders if order.matches(potential)]

    def best_ask(self, visible_to: Account):
        return min(
            order.price
            for order in self.orders_visible_to(visible_to)
            if isinstance(order, SellOrder)
        )

    def best_bid(self, visible_to: Account):
        return max(
            order.price
            for order in self.orders_visible_to(visible_to)
            if isinstance(order, BuyOrder)
        )

    def orders_visible_to(self, account: Account):
        return [order for order in self.orders if order.visibility.matches(account)]

    def submit_(self, order: Order) -> List[Trade]:
        """Instantly execute the order if possible, otherwise add it to the order book"""
        trades = []
        for match in sorted(
            self.matching_orders(order), key=lambda match: match.priority, reverse=True
        ):
            trade = Trade.from_orders(present_order=match, incoming_order=order)
            if trade.volume <= 0:
                continue
            trades.append(trade)
            trade.buyer.cash_balance -= trade.cash_amount
            trade.buyer.asset_balance += trade.volume
            trade.seller.cash_balance += trade.cash_amount
            trade.seller.asset_balance -= trade.volume
            order.volume -= trade.volume
            match.volume -= trade.volume
            if match.volume <= 0:
                self.orders.remove(match)
            if order.volume <= 0:
                break
        if order.volume > 0:
            self.orders.append(order)
        return trades

    def cancel_(self, order: Order):
        """Cancel the order"""
        self.orders.remove(order)
