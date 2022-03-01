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
    actor: Actor
    actor_optimizer: torch.optim.Optimizer
    max_actor_grad_norm: float
    critic: Critic
    critic_optimizer: torch.optim.Optimizer
    max_critic_grad_norm: float
    discount_factor: float
    uncertainty: float

    def decide(self, observation_batch: ObservationBatch) -> DecisionBatch:
        decision_parameters = self.actor(observation_batch.tensor)
        softplus = torch.nn.Softplus()
        return DecisionBatch(
            mid_price=LogNormalBatch(
                underlying_means=decision_parameters[:, 0],
                underlying_stds=self.uncertainty + softplus(decision_parameters[:, 1]),
            ),
            spread=LogNormalBatch(
                underlying_means=decision_parameters[:, 2],
                underlying_stds=self.uncertainty + softplus(decision_parameters[:, 3]),
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

        critic_loss = td_error.square().mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self.critic.parameters(), self.max_critic_grad_norm
        )
        self.critic_optimizer.step()

        actor_loss = (
            -td_error.detach()
            * self.decide(old_observations).log_probabilities(actions)
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad.clip_grad_norm_(
            self.actor.parameters(), self.max_actor_grad_norm
        )
        self.actor_optimizer.step()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
