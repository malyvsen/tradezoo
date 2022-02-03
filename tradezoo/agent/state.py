from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch
from typing import List

from tradezoo.market import Market, Account


@dataclass(frozen=True)
class State:
    cash_balance: float
    stock_balance: float
    best_ask: float
    best_bid: float
    stock_value: float

    @classmethod
    def from_situation(
        cls, market: Market, account: Account, stock_value: float
    ) -> "State":
        return cls(
            cash_balance=account.cash_balance,
            stock_balance=account.stock_balance,
            best_ask=min(order.price for order in market.sell_orders),
            best_bid=max(order.price for order in market.buy_orders),
            stock_value=stock_value,
        )

    @property
    def batch(self):
        return StateBatch(states=[self])

    @property
    def array(self) -> np.ndarray:
        return np.array(
            [
                self.cash_balance,
                self.stock_balance,
                self.best_ask,
                self.best_bid,
                self.stock_value,
            ]
        )


@dataclass(frozen=True)
class StateBatch:
    states: List[State]

    @cached_property
    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(
            np.stack(
                [state.array for state in self.states],
                axis=0,
            )
        )
