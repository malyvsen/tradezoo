from dataclasses import dataclass
from functools import cached_property
import numpy as np
import torch
from typing import List


@dataclass(frozen=True)
class Decision:
    target_asset_allocation: float
    relative_price: float
    random: bool

    @cached_property
    def array(self):
        return np.array(
            [self.target_asset_allocation, self.relative_price], dtype=np.float32
        )


@dataclass(frozen=True)
class DecisionBatch:
    decisions: List[Decision]

    @property
    def tensor(self) -> torch.Tensor:
        return torch.from_numpy(
            np.stack([decision.array for decision in self.decisions], axis=0)
        )
