from dataclasses import dataclass
import torch
from typing import List

from .action import Action, ActionBatch
from .actor import Actor
from .critic import Critic
from .observation import ObservationBatch


@dataclass(frozen=True)
class Agent:
    actor: Actor
    actor_optimizer: torch.optim.Optimizer
    critic: Critic
    critic_optimizer: torch.optim.Optimizer
    discount_factor: float
    uncertainty: float

    def act(self, observation_batch: ObservationBatch) -> ActionBatch:
        action_parameters = self.actor(observation_batch.tensor).detach().numpy()
        return ActionBatch(
            actions=[
                Action(
                    log_mid_price=element_parameters[0],
                    log_spread=element_parameters[1],
                )
                for element_parameters in action_parameters
            ]
        )

    def evaluate(
        self, observation_batch: ObservationBatch, actions: ActionBatch
    ) -> torch.Tensor:
        return self.critic(torch.cat([observation_batch.tensor, actions.tensor], dim=1))

    def train_step_(
        self,
        old_observations: ObservationBatch,
        actions: ActionBatch,
        rewards: torch.Tensor,
        new_observations: ObservationBatch,
    ):
        td_error = (
            rewards
            + self.discount_factor
            * self.evaluate(new_observations, self.act(new_observations)).detach()
            - self.evaluate(old_observations, actions)
        )

        critic_loss = td_error.square().mean()
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        actor_loss = (
            -td_error.detach()
            * self.decide(old_observations).log_probabilities(actions)
        ).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
