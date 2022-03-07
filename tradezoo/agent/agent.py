from dataclasses import dataclass
import torch

from .actor import Actor
from .critic import Critic
from .decision_batch import DecisionBatch
from .observation import ObservationBatch


@dataclass(frozen=True)
class Agent:
    actor: Actor
    actor_optimizer: torch.optim.Optimizer
    critic: Critic
    critic_optimizer: torch.optim.Optimizer
    discount_factor: float
    uncertainty: float

    def decide(self, observation_batch: ObservationBatch) -> DecisionBatch:
        decision_parameters = self.actor(observation_batch.tensor)
        return DecisionBatch(
            log_mid_price=torch.distributions.Normal(
                loc=decision_parameters[:, 0],
                scale=self.uncertainty + decision_parameters[:, 1].abs(),
            ),
            log_spread=torch.distributions.Normal(
                loc=decision_parameters[:, 2],
                scale=self.uncertainty + decision_parameters[:, 3].abs(),
            ),
        )

    def evaluate(self, observation_batch: ObservationBatch) -> torch.Tensor:
        return self.critic(observation_batch.tensor)

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
