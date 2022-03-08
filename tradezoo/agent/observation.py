from dataclasses import dataclass
from functools import cached_property
import math
import numpy as np
import torch
from typing import List

from tradezoo.market import Market, Account, BuyOrder, SellOrder


@dataclass(frozen=True)
class Observation:
    cash_balance: float
    asset_balance: float
    best_ask: float
    best_bid: float

    epsilon = 1

    @classmethod
    def from_situation(cls, market: Market, account: Account) -> "Observation":
        # TODO: prevent spoofing with invalid orders
        return cls(
            cash_balance=account.cash_balance,
            asset_balance=account.asset_balance,
            best_ask=min(
                order.price
                for order in market.orders
                if isinstance(order, SellOrder)
                if order.visibility.matches(account)
            ),
            best_bid=max(
                order.price
                for order in market.orders
                if isinstance(order, BuyOrder)
                if order.visibility.matches(account)
            ),
        )

    @property
    def batch(self):
        return ObservationBatch(observations=[self])

    @property
    def array(self) -> np.ndarray:
        return np.array(
            [
                math.log(self.epsilon + self.cash_balance),
                math.log(self.epsilon + self.asset_balance),
                math.log(self.epsilon + self.best_ask),
                math.log(self.epsilon + self.best_bid),
            ],
            dtype=np.float32,
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
