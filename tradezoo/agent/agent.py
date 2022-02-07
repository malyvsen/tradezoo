from dataclasses import dataclass
import torch

from .actor import Actor
from .critic import Critic
from .decision_batch import DecisionBatch
from .log_normal_batch import LogNormalBatch
from .observation import ObservationBatch


@dataclass(frozen=True)
class Agent:
    """The brain of a trader - makes decisions, judges the observed state"""

    actor: Actor
    critic: Critic

    def decide(self, observation_batch: ObservationBatch) -> DecisionBatch:
        decision_parameters = self.actor(observation_batch.tensor)
        return DecisionBatch(
            ask=LogNormalBatch(
                underlying_means=decision_parameters[:, 0],
                underlying_stds=decision_parameters[:, 1],
            ),
            bid=LogNormalBatch(
                underlying_means=decision_parameters[:, 2],
                underlying_stds=decision_parameters[:, 3],
            ),
        )

    def evaluate(self, observation_batch: ObservationBatch) -> torch.Tensor:
        return self.critic(observation_batch.tensor)
