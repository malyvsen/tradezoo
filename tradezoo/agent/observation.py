from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch
from typing import List

from tradezoo.market import Market, Account


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
            best_ask=market.best_ask(visible_to=account),
            best_bid=market.best_bid(visible_to=account),
        )

    @property
    def batch(self):
        return ObservationBatch(observations=[self])

    @property
    def array(self) -> np.ndarray:
        return np.array(
            [
                np.log(self.epsilon + self.cash_balance),
                np.log(self.epsilon + self.asset_balance),
                np.log(self.epsilon + self.best_ask),
                np.log(self.epsilon + self.best_bid),
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
