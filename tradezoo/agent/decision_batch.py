from dataclasses import dataclass
import torch
from typing import List

from .log_normal_batch import LogNormalBatch
from .action import Action


@dataclass(frozen=True)
class DecisionBatch:
    """A batch of decisions made by the actor regarding the distribution of actions"""

    mid_price: LogNormalBatch
    spread: LogNormalBatch

    def sample(self) -> List[Action]:
        mid_prices = self.mid_price.sample()
        spreads = self.spread.sample()
        return [
            Action(mid_price=mid_price.item(), spread=spread.item())
            for mid_price, spread in zip(mid_prices, spreads)
        ]

    def log_probabilities(self, actions: List[Action]) -> torch.Tensor:
        return self.mid_price.log_probabilities(
            torch.tensor([action.mid_price for action in actions])
        ) + self.spread.log_probabilities(
            torch.tensor([action.spread for action in actions])
        )
