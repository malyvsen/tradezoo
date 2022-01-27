from dataclasses import dataclass
from typing import List

from .account import Account
from .order import Order
from .trade import Trade


@dataclass
class Market:
    accounts: List[Account]
    orders: List[Order]

    def orders_by(self, account: Account):
        return [order for order in self.orders if order.submitted_by is account]

    def submit_(self, order: Order):
        """Instantly execute the order if possible, otherwise add it to the order book"""
        matches = [potential for potential in self.orders if order.matches(potential)]
        for match in sorted(matches, key=lambda match: match.priority, reverse=True):
            trade = Trade.from_orders(present_order=match, incoming_order=order)
            trade.buyer.cash_balance -= trade.cash_amount
            trade.buyer.stock_balance += trade.volume
            trade.seller.cash_balance += trade.cash_amount
            trade.seller.stock_balance -= trade.volume
            order.volume -= trade.volume
            match.volume -= trade.volume
            if match.volume <= 0:
                self.orders.remove(match)
            if order.volume <= 0:
                break
        if order.volume > 0:
            self.orders.append(order)

    def cancel_(self, order: Order):
        """Cancel the order"""
        self.orders.remove(order)
