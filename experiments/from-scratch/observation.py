from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class Observation:
    cash_balance: float
    asset_balance: float
    best_ask: float
    best_bid: float

    @property
    def tensor(self):
        return torch.tensor(
            [[self.cash_balance, self.asset_balance, self.best_ask, self.best_bid]],
            dtype=torch.float32,
        )
