from dataclasses import dataclass
from typing import List

from .account import Account
from .order import Order


@dataclass
class Market:
    accounts: List[Account]
    orders: List[Order]

    def submit_(self, order: Order):
        """Instantly execute the order if possible, otherwise add it to the order book"""
        matches = [potential for potential in self.orders if order.matches(potential)]
        if matches == []:
            self.orders.append(order)
            return
        best_match = min(matches, key=lambda match: abs(match.price - order.price))
        execution_price = order.price

        order.execute_(execution_price)
        best_match.execute_(execution_price)
        self.orders.remove(best_match)
