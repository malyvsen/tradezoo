from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch


@dataclass(frozen=True)
class StateBatch:
    cash_balances: np.ndarray
    stock_balances: np.ndarray
    best_asks: np.ndarray
    best_bids: np.ndarray
    stock_values: np.ndarray

    @cached_property
    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(
            np.stack(
                [
                    self.cash_balances,
                    self.stock_balances,
                    self.best_asks,
                    self.best_bids,
                    self.stock_values,
                ],
                axis=1,
            )
        )
