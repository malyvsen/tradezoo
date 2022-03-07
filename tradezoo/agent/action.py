import numpy as np
from dataclasses import dataclass
from functools import cached_property
import torch
from typing import List


@dataclass(frozen=True)
class Action:
    log_mid_price: float
    log_spread: float

    @cached_property
    def mid_price(self):
        return np.exp(self.log_mid_price)

    @cached_property
    def spread(self):
        return np.exp(self.log_spread)

    @cached_property
    def ask(self) -> float:
        return self.mid_price * (1 + self.spread)

    @cached_property
    def bid(self) -> float:
        return self.mid_price / (1 + self.spread)

    @cached_property
    def array(self) -> np.ndarray:
        return np.array([self.log_mid_price, self.log_spread], dtype=np.float32)


@dataclass(frozen=True)
class ActionBatch:
    actions: List[Action]

    @cached_property
    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(
            np.stack(
                [action.array for action in self.actions],
                axis=0,
            )
        )
