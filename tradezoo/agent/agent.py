from dataclasses import dataclass, replace
from functools import cached_property
import numpy as np
import random
import torch

from .critic import Critic
from .decision import Decision, DecisionBatch
from .observation import ObservationSeries, ObservationSeriesBatch


@dataclass(frozen=False)
class Agent:
    """The part of a trader responsible for making decisions."""

    critic: Critic
    horizon: int
    allocation_space: np.array
    relative_price_space: np.array

    @cached_property
    def decision_space(self):
        return [
            Decision(
                target_asset_allocation=target_asset_allocation,
                relative_price=relative_price,
                random=False,
            )
            for target_asset_allocation in self.allocation_space
            for relative_price in self.relative_price_space
        ]

    @property
    def random_decision_probability(self):
        raise NotImplementedError()

    def decide(self, observations: ObservationSeries) -> Decision:
        if random.random() < self.random_decision_probability:
            return replace(random.choice(self.decision_space), random=True)
        return self.best_decision(observations)

    def best_decision(self, observations: ObservationSeries) -> Decision:
        evaluations = self.critic.evaluate(
            observations=ObservationSeriesBatch(
                [observations] * len(self.decision_space)
            ),
            decisions=DecisionBatch(self.decision_space),
        )
        best_index = torch.argmax(evaluations, dim=0).item()
        return self.decision_space[best_index]

    def __hash__(self):
        return id(self)

    def __eq__(self, other):
        return self is other


@dataclass(frozen=False)
class DeterministicAgent(Agent):
    @property
    def random_decision_probability(self):
        return 0
