from dataclasses import dataclass
import torch

from .actor import Actor
from .critic import Critic
from .decision_batch import DecisionBatch
from .state_batch import StateBatch


@dataclass(frozen=True)
class Agent:
    actor: Actor
    critic: Critic

    def decide(self, state_batch: StateBatch) -> DecisionBatch:
        decision_parameters = self.actor(state_batch.tensor)
        return DecisionBatch(
            ask_means=decision_parameters[:, 0],
            ask_scales=decision_parameters[:, 1],
            bid_means=decision_parameters[:, 2],
            bid_scales=decision_parameters[:, 3],
        )

    def evaluate(self, state_batch: StateBatch) -> torch.Tensor:
        return self.critic(state_batch.tensor)
