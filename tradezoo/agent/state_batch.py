from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch

from tradezoo.market import Market, Account


@dataclass(frozen=True)
class StateBatch:
    cash_balances: np.ndarray
    stock_balances: np.ndarray
    best_asks: np.ndarray
    best_bids: np.ndarray
    stock_values: np.ndarray

    @classmethod
    def from_situation(
        cls, market: Market, account: Account, stock_value: float
    ) -> "StateBatch":
        return cls(
            cash_balances=np.array([account.cash_balance]),
            stock_balances=np.array([account.stock_balance]),
            best_asks=np.array([min(order.price for order in market.sell_orders)]),
            best_bids=np.array([max(order.price for order in market.buy_orders)]),
            stock_values=np.array([stock_value]),
        )

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
