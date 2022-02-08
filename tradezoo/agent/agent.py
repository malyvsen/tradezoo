from dataclasses import dataclass
import torch
from typing import List

from .action import Action
from .actor import Actor
from .critic import Critic
from .decision_batch import DecisionBatch
from .log_normal_batch import LogNormalBatch
from .observation import ObservationBatch


@dataclass(frozen=True)
class Agent:
    """The brain of a trader - makes decisions, judges the observed state"""

    actor: Actor
    actor_optimizer: torch.optim.Optimizer
    critic: Critic
    critic_optimizer: torch.optim.Optimizer
    discount_factor: float

    @classmethod
    def from_modules(
        cls,
        actor: Actor,
        critic: Critic,
        actor_optimizer_curried,
        critic_optimizer_curried,
        discount_factor: float,
    ) -> "Agent":
        return cls(
            actor=actor,
            actor_optimizer=actor_optimizer_curried(actor.parameters()),
            critic=critic,
            critic_optimizer=critic_optimizer_curried(critic.parameters()),
            discount_factor=discount_factor,
        )

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

    def train_step_(
        self,
        old_observations: ObservationBatch,
        actions: List[Action],
        rewards: torch.Tensor,
        new_observations: ObservationBatch,
    ):
        td_error = (
            rewards
            + self.discount_factor * self.evaluate(new_observations).detach()
            - self.evaluate(old_observations)
        )

        critic_loss = td_error.square()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = -td_error.detach() * self.decide(
            old_observations
        ).log_probabilities(actions)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
