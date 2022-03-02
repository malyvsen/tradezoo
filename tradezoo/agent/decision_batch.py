from dataclasses import dataclass
import torch
from typing import List

from .action import Action


@dataclass(frozen=True)
class DecisionBatch:
    """A batch of decisions made by the actor regarding the distribution of actions"""

    log_mid_price: torch.distributions.Distribution
    log_spread: torch.distributions.Distribution

    def sample(self) -> List[Action]:
        log_mid_prices = self.log_mid_price.sample()
        log_spreads = self.log_spread.sample()
        return [
            Action(log_mid_price=log_mid_price.item(), log_spread=log_spread.item())
            for log_mid_price, log_spread in zip(log_mid_prices, log_spreads)
        ]

    def log_probabilities(self, actions: List[Action]) -> torch.Tensor:
        return self.log_mid_price.log_prob(
            torch.tensor([action.log_mid_price for action in actions])
        ) + self.log_spread.log_prob(
            torch.tensor([action.log_spread for action in actions])
        )
