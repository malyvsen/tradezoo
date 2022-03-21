from dataclasses import dataclass
import torch
from typing import List

from .action import Action


@dataclass(frozen=True)
class DecisionBatch:
    """A batch of decisions made by the actor regarding the distribution of actions"""

    asset_allocation: torch.distributions.Distribution

    def sample(self) -> List[Action]:
        asset_allocations = self.asset_allocation.sample()
        return [Action(asset_allocation) for asset_allocation in asset_allocations]

    def log_probabilities(self, actions: List[Action]) -> torch.Tensor:
        return self.asset_allocation.log_prob(
            torch.tensor([action.asset_allocation for action in actions])
        )
