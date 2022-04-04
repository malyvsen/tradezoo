from dataclasses import dataclass
from functools import cached_property
import numpy as np
import random
import torch
from typing import List

from .critic import Critic
from .decision import Decision, DecisionBatch
from .observation import ObservationSeries, ObservationSeriesBatch


@dataclass(frozen=True)
class Agent:
    critic: Critic
    optimizer: torch.optim.Optimizer
    target: Critic
    discount_factor: float
    random_decision_probability: float
    decision_resolution: int

    def decide(self, observations: ObservationSeries) -> Decision:
        if random.random() < self.random_decision_probability:
            return random.choice(self.possible_decisions)
        return self.best_decision(observations)

    def best_decision(self, observations: ObservationSeries) -> Decision:
        evaluations = self.critic.evaluate(
            observations=ObservationSeriesBatch(
                [observations] * len(self.possible_decisions)
            ),
            decisions=DecisionBatch(self.possible_decisions),
        )
        best_index = torch.argmax(evaluations, dim=0).item()
        return self.possible_decisions[best_index]

    @cached_property
    def possible_decisions(self):
        return [
            Decision(
                target_asset_allocation=target_asset_allocation, desperation=desperation
            )
            for target_asset_allocation in np.linspace(
                0, 1, num=self.decision_resolution
            )
            for desperation in np.linspace(0, 1, num=self.decision_resolution)
        ]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other
