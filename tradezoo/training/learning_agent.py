from dataclasses import dataclass
import math
import numpy as np
import torch
from typing import Callable, List

from tradezoo.agent import (
    Agent,
    Critic,
    DecisionBatch,
    DeterministicAgent,
    ObservationSeriesBatch,
)
from tradezoo.game import TurnResult
from .experience import Experience
from .replay_buffer import ReplayBuffer
from .train_result import TrainResult


@dataclass(frozen=False)
class LearningAgent(Agent):
    exploration_schedule: Callable[[int], float]
    utility_function: Callable[[float], float]
    discount_factor: float

    replay_buffer: ReplayBuffer
    batch_size: int
    train_steps_per_turn: int

    optimizer: torch.optim.Optimizer
    target: Critic
    steps_per_target_update: int
    steps_completed: int

    @property
    def random_decision_probability(self):
        return self.exploration_schedule(self.steps_completed)

    @property
    def deterministic(self) -> DeterministicAgent:
        return DeterministicAgent(
            critic=self.critic,
            horizon=self.horizon,
            allocation_space=self.allocation_space,
            relative_price_space=self.relative_price_space,
        )

    @classmethod
    def good_hyperparameters(cls):
        critic = Critic()
        return cls(
            critic=critic,
            horizon=2,
            allocation_space=np.linspace(0, 1, num=8),
            relative_price_space=2 ** np.linspace(-0.2, 0.2, num=9),
            exploration_schedule=lambda step: 256 / (step + 256),
            utility_function=math.log,
            discount_factor=0.9,
            replay_buffer=ReplayBuffer.empty(capacity=256),
            batch_size=16,
            train_steps_per_turn=64,
            optimizer=torch.optim.Adam(critic.parameters(), lr=2e-4),
            target=Critic(),
            steps_per_target_update=2048,
            steps_completed=0,
        )

    def post_turn_(self, turn_result: TurnResult):
        self.replay_buffer.register_turn_(turn_result)
        if not self.replay_buffer.full:
            return []
        return [self.train_step_() for _ in range(self.train_steps_per_turn)]

    def train_step_(self):
        experiences = self.replay_buffer.sample(self.batch_size)
        td_error = self.td_error(experiences)
        loss = td_error.square().mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.steps_completed % self.steps_per_target_update == 0:
            self.update_target_()
        self.steps_completed += 1
        return TrainResult(loss=loss.item())

    def td_error(self, experiences: List[Experience]):
        old_observations = ObservationSeriesBatch(
            [experience.old_turn_result.observations for experience in experiences]
        )
        decisions = DecisionBatch(
            [experience.old_turn_result.decision for experience in experiences]
        )
        rewards = torch.tensor(
            [self.reward(experience) for experience in experiences], dtype=torch.float32
        )
        new_observations = ObservationSeriesBatch(
            [experience.new_turn_result.observations for experience in experiences]
        )
        return (
            rewards
            + self.discount_factor * self.target_evaluations(new_observations)
            - self.critic.evaluate(old_observations, decisions)
        )

    def target_evaluations(self, observations: ObservationSeriesBatch):
        with torch.no_grad():
            flattened_evaluations: torch.Tensor = self.target(
                observations=observations.tensor.repeat_interleave(
                    len(self.decision_space), dim=0
                ),
                decisions=DecisionBatch(self.decision_space).tensor.tile(
                    [len(observations.series), 1]
                ),
            )
        evaluations = flattened_evaluations.reshape(
            len(observations.series), len(self.decision_space)
        )
        return evaluations.max(dim=1).values

    def reward(self, experience: Experience):
        old_state = experience.old_turn_result.state
        old_utility = self.utility_function(
            old_state.cash_balance + old_state.asset_balance * old_state.mid_price
        )
        new_state = experience.new_turn_result.state
        new_utility = self.utility_function(
            new_state.cash_balance + new_state.asset_balance * new_state.mid_price
        )
        return new_utility - old_utility

    def update_target_(self):
        self.target.load_state_dict(self.critic.state_dict())
