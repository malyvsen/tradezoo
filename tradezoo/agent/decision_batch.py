from dataclasses import dataclass
import torch
from typing import List

from .normal_batch import NormalBatch
from .action import Action


@dataclass(frozen=True)
class DecisionBatch:
    """A batch of decisions made by the actor regarding the distribution of actions"""

    ask: NormalBatch
    bid: NormalBatch

    def sample(self) -> List[Action]:
        ask_samples = self.ask.sample()
        bid_samples = self.bid.sample()
        return [
            Action(ask=ask_sample.item(), bid=bid_sample.item())
            for ask_sample, bid_sample in zip(ask_samples, bid_samples)
        ]

    def log_probabilities(self, actions: List[Action]) -> torch.Tensor:
        return self.ask.log_probabilities(
            torch.tensor([action.ask for action in actions])
        ) + self.bid.log_probabilities(torch.tensor([action.bid for action in actions]))
