from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch
from typing import List

from tradezoo.market import Market, Account


@dataclass(frozen=True)
class Observation:
    cash_balance: float
    stock_balance: float
    best_ask: float
    best_bid: float
    stock_value: float

    @classmethod
    def from_situation(
        cls, market: Market, account: Account, true_stock_value: float, noise: float
    ) -> "Observation":
        return cls(
            cash_balance=account.cash_balance,
            stock_balance=account.stock_balance,
            best_ask=min(order.price for order in market.sell_orders),
            best_bid=max(order.price for order in market.buy_orders),
            stock_value=true_stock_value + noise,
        )

    @property
    def batch(self):
        return ObservationBatch(observations=[self])

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
class ObservationBatch:
    observations: List[Observation]

    @cached_property
    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(
            np.stack(
                [observation.array for observation in self.observations],
                axis=0,
            )
        )
